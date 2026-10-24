#include "multimodular_blas.hxx"

#include "sdp_solve/SDP_Solver/run/bigint_syrk/fmpz/Fmpz_Comb.hxx"
#include "sdp_solve/SDP_Solver/run/bigint_syrk/fmpz/fmpz_mul_blas_util.hxx"
#include "sdpb_util/assert.hxx"

#include <flint/ulong_extras.h>

#include <cblas.h>

#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <tuple>
#include <vector>

void Multimodular_Blas_Timings::add(const Multimodular_Blas_Timings &other)
{
  base_seconds += other.base_seconds;
  to_int_seconds += other.to_int_seconds;
  blas_seconds += other.blas_seconds;
  from_int_seconds += other.from_int_seconds;
  num_gemms += other.num_gemms;
  max_num_primes = std::max(max_num_primes, other.max_num_primes);
}

// Algorithm of multimodular_gemm_sub(C, A, op(B)):
//
// 1. Fixed point: row i of A and column j of op(B) are multiplied by
//    2^(bits - e), e = binary exponent of their largest entry, and truncated
//    to integers a_il, b_lj with |a|, |b| < 2^bits.
// 2. Residues modulo primes p_1..p_n (p < 2^21, p^2 k < 2^53, product M of
//    the primes > 8 max|c_ij|), computed by BLAS: an integer is split into
//    24-bit chunks c_t, and |x| mod p = sum_t c_t (2^(24t) mod p) mod p.
//    The residues of A are multiplied by u_p = (M/p)^-1 mod p (see 4).
// 3. For each prime, cblas_dgemm of the residue matrices (exact in double).
// 4. CRT, also by BLAS. With y_p = c u_p mod p in [0, p),
//      c = sum_p y_p (M/p) - q M,  q = round(sum_p y_p / p),
//    where q is computed in double (|c|/M < 1/8 leaves room for its error).
//    M/p is split into 24-bit chunks m_pt; S_t = sum_p y_p m_pt is exact in
//    double, and one dgemm gives all S_t. Only the chunks t >= t0 are
//    computed: the dropped low part is < 2^(24 t0 + 53) < 2^(bits - 64),
//    far below the truncation error of step 1 (~k 2^bits).
// 5. C_ij -= c_ij 2^(e_i + e_j - 2 bits).

namespace
{
  using Clock = std::chrono::steady_clock;
  double seconds_since(const Clock::time_point &start)
  {
    return std::chrono::duration<double>(Clock::now() - start).count();
  }

  constexpr int chunk_bits = 24;
  constexpr mp_limb_t chunk_mask = (mp_limb_t(1) << chunk_bits) - 1;

  // v - round(v/p) p, in [-(p+1)/2, (p+1)/2], for an integer-valued double
  // |v| < 2^52. Branch-free (rounding by the 1.5*2^52 trick) so that the
  // loops below vectorize.
  inline double round_mod(double v, double p, double inv_p)
  {
    constexpr double magic = 6755399441055744.0; // 1.5 * 2^52
    const double q = (v * inv_p + magic) - magic;
    return v - q * p;
  }
  // v mod p in [0, p)
  inline double mod_p(double v, double p, double inv_p)
  {
    const double r = round_mod(v, p, inv_p);
    return r + ((r < 0) ? p : 0.0);
  }

  // Chunks [chunk_bits t, chunk_bits (t+1)) of the integer
  // y = trunc(|x| * 2^shift) < 2^(chunk_bits num_chunks), times the sign of x,
  // for t = 0..num_chunks-1 (written with stride). buffer: scratch space of
  // at least chunk_bits * num_chunks / 64 + 2 limbs.
  void write_chunks(const __mpf_struct *x, long shift, int num_chunks,
                    double *chunks, size_t stride, std::vector<mp_limb_t> &buffer)
  {
    const long size = std::abs(x->_mp_size);
    const mp_limb_t *d = x->_mp_d;
    const double sign = x->_mp_size < 0 ? -1.0 : 1.0;
    const long num_limbs = long(chunk_bits) * num_chunks / GMP_NUMB_BITS + 2;
    std::fill(buffer.begin(), buffer.begin() + num_limbs, 0);
    // bit position in y of bit 0 of d[0]
    const long offset = long(GMP_NUMB_BITS) * (x->_mp_exp - size) + shift;
    if(offset >= 0)
      {
        const long ls = offset / GMP_NUMB_BITS;
        const unsigned bs = offset % GMP_NUMB_BITS;
        ASSERT(ls + size + 1 <= num_limbs, "fixed-point overflow");
        if(bs == 0)
          std::copy(d, d + size, buffer.data() + ls);
        else
          buffer[ls + size] = mpn_lshift(buffer.data() + ls, d, size, bs);
      }
    else
      {
        const long ld = (-offset) / GMP_NUMB_BITS;
        const unsigned bs = (-offset) % GMP_NUMB_BITS;
        if(ld < size)
          {
            ASSERT(size - ld <= num_limbs, "fixed-point overflow");
            if(bs == 0)
              std::copy(d + ld, d + size, buffer.data());
            else
              mpn_rshift(buffer.data(), d + ld, size - ld, bs);
          }
      }
    // little-endian limbs: chunk t is the 24 bits at byte 3t
    static_assert(chunk_bits == 24, "byte extraction assumes 24-bit chunks");
    const unsigned char *bytes
      = reinterpret_cast<const unsigned char *>(buffer.data());
    for(int t = 0; t < num_chunks; ++t)
      {
        uint32_t v;
        std::memcpy(&v, bytes + 3 * t, sizeof(v));
        chunks[t * stride] = sign * double(v & 0xFFFFFFu);
      }
  }

  // Everything that depends only on (bits, k)
  struct Residue_Context
  {
    Fmpz_Comb comb; // primes; FLINT CRT for config.flint_crt
    size_t num_primes;
    std::vector<double> primes, inverse_primes;
    std::vector<double> u; // (M/p)^-1 mod p
    // residues: |x| < 2^bits in num_chunks chunks; W = 2^(24t) mod p,
    // num_chunks x num_primes, column-major
    int num_chunks;
    std::vector<double> W;
    // CRT: chunks t0..t0+num_high_chunks-1 of M/p,
    // num_primes x num_high_chunks, column-major
    int t0, num_high_chunks;
    std::vector<double> M_chunks;
    // floor(q M / 2^(24 t0)), q = 0..num_primes, num_limbs limbs each
    int num_limbs;
    std::vector<mp_limb_t> qM;

    // Integers |a| < 2^bits_A, |b| < 2^bits_B, inner dimension <= k.
    // Product bound: |c| < k 2^(bits_A + bits_B); 3 more bits make
    // |c|/M < 1/8 for the rounding of q.
    Residue_Context(long bits_A, long bits_B, slong k)
        : comb(bits_A + bits_B + FLINT_BIT_COUNT(k) + 3, k),
          num_primes(comb.num_primes),
          num_chunks((std::max(bits_A, bits_B) + chunk_bits - 1) / chunk_bits)
    {
      const long bits = std::min(bits_A, bits_B);
      primes.resize(num_primes);
      inverse_primes.resize(num_primes);
      u.resize(num_primes);
      W.resize(size_t(num_chunks) * num_primes);

      fmpz_t M, M_p;
      fmpz_init_set_ui(M, 1);
      fmpz_init(M_p);
      for(size_t i = 0; i < num_primes; ++i)
        fmpz_mul_ui(M, M, comb.primes[i]);

      t0 = std::max<long>(0, (bits - 117) / chunk_bits);
      const long M_bits = fmpz_bits(M);
      num_high_chunks = (M_bits + chunk_bits - 1) / chunk_bits - t0;
      M_chunks.resize(num_primes * num_high_chunks);
      // S_high < num_primes * 2^53 * 2^(24 num_high_chunks)
      num_limbs = (chunk_bits * num_high_chunks + 64 + 8) / GMP_NUMB_BITS + 2;

      mpz_t z;
      mpz_init(z);
      for(size_t i = 0; i < num_primes; ++i)
        {
          const mp_limb_t p = comb.primes[i];
          ASSERT(std::ldexp(double(p) * num_chunks, chunk_bits)
                     < std::ldexp(1.0, 53)
                   && std::ldexp(double(p) * num_primes, chunk_bits)
                        < std::ldexp(1.0, 53),
                 "chunked residues/CRT would not be exact: p=", p,
                 " num_chunks=", num_chunks, " num_primes=", num_primes);
          primes[i] = p;
          inverse_primes[i] = 1.0 / double(p);
          mp_limb_t power = 1;
          const mp_limb_t base = (mp_limb_t(1) << chunk_bits) % p;
          for(int t = 0; t < num_chunks; ++t)
            {
              W[t + size_t(num_chunks) * i] = double(power);
              power = (power * base) % p;
            }
          fmpz_divexact_ui(M_p, M, p);
          u[i] = double(n_invmod(fmpz_fdiv_ui(M_p, p), p));
          fmpz_get_mpz(z, M_p);
          for(int t = 0; t < num_high_chunks; ++t)
            {
              // bits [24 (t0+t), 24 (t0+t+1)) of M/p
              mp_limb_t chunk = 0;
              for(int b = 0; b < chunk_bits; ++b)
                chunk |= mp_limb_t(mpz_tstbit(z, chunk_bits * (t0 + t) + b))
                         << b;
              M_chunks[i + num_primes * t] = double(chunk);
            }
        }
      qM.assign((num_primes + 1) * num_limbs, 0);
      for(size_t q = 0; q <= num_primes; ++q)
        {
          fmpz_get_mpz(z, M);
          mpz_mul_ui(z, z, q);
          mpz_fdiv_q_2exp(z, z, chunk_bits * t0);
          ASSERT(mpz_size(z) <= size_t(num_limbs));
          for(size_t l = 0; l < mpz_size(z); ++l)
            qM[q * num_limbs + l] = mpz_getlimbn(z, l);
        }
      mpz_clear(z);
      fmpz_clear(M_p);
      fmpz_clear(M);
    }
  };

  Residue_Context &get_context(long bits_A, long bits_B, slong k)
  {
    static std::map<std::tuple<long, long, slong>,
                    std::unique_ptr<Residue_Context>>
      cache;
    auto &context = cache[{bits_A, bits_B, k}];
    if(!context)
      context = std::make_unique<Residue_Context>(bits_A, bits_B, k);
    return *context;
  }

  // Binary exponent e of x, such that 2^(e-1) <= |x| < 2^e.
  // Returns false for x == 0.
  bool binary_exponent(const El::BigFloat &x, long &e)
  {
    if(mpf_sgn(x.gmp_float.get_mpf_t()) == 0)
      return false;
    mpf_get_d_2exp(&e, x.gmp_float.get_mpf_t());
    return true;
  }

  // Residues of n fixed-point integers trunc(x_e * 2^shift_e):
  // out[e + n * i] = residue modulo prime i in [-(p_i+1)/2, (p_i+1)/2],
  // multiplied by u_i if multiply_by_u.
  // element(e, shift) returns x_e (nullptr for zero) and sets shift.
  template <class Element>
  void compute_residues(const Residue_Context &context, size_t n,
                        const Element &element, bool multiply_by_u,
                        double *out, size_t prime_stride = 0)
  {
    if(prime_stride == 0)
      prime_stride = n;
    const size_t num_primes = context.num_primes;
    const int T = context.num_chunks;
    constexpr size_t batch = 2048;
    std::vector<double> chunks(batch * T);
    std::vector<mp_limb_t> buffer(chunk_bits * T / GMP_NUMB_BITS + 2);
    for(size_t e0 = 0; e0 < n; e0 += batch)
      {
        const size_t nb = std::min(batch, n - e0);
        for(size_t e = 0; e < nb; ++e)
          {
            long shift;
            const __mpf_struct *x = element(e0 + e, shift);
            if(x == nullptr || x->_mp_size == 0)
              {
                for(int t = 0; t < T; ++t)
                  chunks[e + t * nb] = 0;
              }
            else
              write_chunks(x, shift, T, chunks.data() + e, nb, buffer);
          }
        // |sum| < T 2^24 p < 2^53
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nb,
                    num_primes, T, 1.0, chunks.data(), nb, context.W.data(),
                    T, 0.0, out + e0, prime_stride);
        for(size_t i = 0; i < num_primes; ++i)
          {
            const double p = context.primes[i];
            const double inv_p = context.inverse_primes[i];
            double *__restrict r = out + e0 + prime_stride * i;
            if(multiply_by_u)
              {
                const double u = context.u[i];
                for(size_t e = 0; e < nb; ++e)
                  r[e] = round_mod(round_mod(r[e], p, inv_p) * u, p, inv_p);
              }
            else
              {
                for(size_t e = 0; e < nb; ++e)
                  r[e] = round_mod(r[e], p, inv_p);
              }
          }
      }
  }

  // c = sum_t S_t 2^(24 t) - q_M, two's complement in acc[0..num_limbs).
  // Returns the sign and sets n = number of significant limbs of |c|.
  int assemble(const Residue_Context &context, const double *S, size_t stride,
               const mp_limb_t *q_M, mp_limb_t *acc, int &n)
  {
    const int L = context.num_limbs;
    for(int l = 0; l < L; ++l)
      acc[l] = 0;
    for(int t = 0; t < context.num_high_chunks; ++t)
      {
        const mp_limb_t v = mp_limb_t(S[t * stride]);
        const int o = chunk_bits * t;
        int li = o / GMP_NUMB_BITS;
        const int sh = o % GMP_NUMB_BITS;
        mp_limb_t lo = v << sh;
        mp_limb_t hi = sh == 0 ? 0 : v >> (GMP_NUMB_BITS - sh);
        mp_limb_t old = acc[li];
        acc[li] += lo;
        mp_limb_t carry = hi + (acc[li] < old);
        while(carry != 0)
          {
            ++li;
            old = acc[li];
            acc[li] += carry;
            carry = acc[li] < old;
          }
      }
    mpn_sub_n(acc, acc, q_M, L);
    int sign = 1;
    if(acc[L - 1] >> (GMP_NUMB_BITS - 1))
      {
        sign = -1;
        mpn_neg(acc, acc, L);
      }
    n = L;
    while(n > 0 && acc[n - 1] == 0)
      --n;
    return sign;
  }
}

namespace
{
  constexpr long no_exponent = std::numeric_limits<long>::min();

  // Scratch space and the CRT step shared by the GEMM and the Trsm.
  struct Crt_Workspace
  {
    std::vector<double> quotients, S_high;
    std::vector<mp_limb_t> crt_input, acc;
    fmpz_t z;
    mpf_t value;
    explicit Crt_Workspace(const Residue_Context &context)
        : crt_input(context.num_primes), acc(context.num_limbs)
    {
      fmpz_init(z);
      mpf_init2(value, El::gmp::Precision());
    }
    ~Crt_Workspace()
    {
      mpf_clear(value);
      fmpz_clear(z);
    }
  };

  // C_residues: num_primes blocks (stride m*width) of the m x width residue
  // products (unreduced dgemm output; times u_p unless flint_crt).
  // C(i, j) -= c_ij 2^(row_exp[i] + col_exp[j] - shift_base)
  // for row_start[j] <= i < m.
  void crt_subtract(const Residue_Context &context, bool flint_crt,
                    double *C_residues, int m, int width,
                    El::Matrix<El::BigFloat> &C,
                    const std::vector<long> &row_exp,
                    const std::vector<long> &col_exp, long shift_base,
                    const std::vector<int> &row_start, Crt_Workspace &work)
  {
    const size_t num_primes = context.num_primes;
    const size_t C_stride = size_t(m) * width;
    Fmpz_Comb &comb = const_cast<Fmpz_Comb &>(context.comb);
    if(!flint_crt)
      {
        // y_p = (u_p c mod p) in [0, p); q = round(sum_p y_p / p)
        work.quotients.assign(C_stride, 0.0);
        work.S_high.resize(C_stride * context.num_high_chunks);
        for(size_t p = 0; p < num_primes; ++p)
          {
            const double prime = context.primes[p];
            const double inv_p = context.inverse_primes[p];
            double *__restrict y = C_residues + p * C_stride;
            double *__restrict q = work.quotients.data();
            for(size_t e = 0; e < C_stride; ++e)
              {
                const double r = mod_p(y[e], prime, inv_p);
                y[e] = r;
                q[e] += r * inv_p;
              }
          }
        // S_t = sum_p y_p m_pt
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, C_stride,
                    context.num_high_chunks, num_primes, 1.0, C_residues,
                    C_stride, context.M_chunks.data(), num_primes, 0.0,
                    work.S_high.data(), C_stride);
      }
    for(int j = 0; j < width; ++j)
      {
        if(col_exp[j] == no_exponent)
          continue;
        for(int i = std::max(row_start[j], 0); i < m; ++i)
          {
            if(row_exp[i] == no_exponent)
              continue;
            const size_t e = i + size_t(j) * m;
            long shift = row_exp[i] + col_exp[j] - shift_base;
            if(!flint_crt)
              {
                const size_t q = std::lround(work.quotients[e]);
                int size;
                const int sign = assemble(
                  context, work.S_high.data() + e, C_stride,
                  context.qM.data() + q * context.num_limbs, work.acc.data(),
                  size);
                if(size == 0)
                  continue;
                __mpf_struct view;
                view._mp_prec = size;
                view._mp_size = sign * size;
                view._mp_exp = size;
                view._mp_d = work.acc.data();
                shift += chunk_bits * context.t0;
                if(shift >= 0)
                  mpf_mul_2exp(work.value, &view, shift);
                else
                  mpf_div_2exp(work.value, &view, -shift);
              }
            else
              {
                for(size_t p = 0; p < num_primes; ++p)
                  work.crt_input[p] = double_to_uint32_t_residue(
                    C_residues[e + p * C_stride], comb.mods[p]);
                fmpz_multi_CRT_ui(work.z, work.crt_input.data(), comb.comb,
                                  comb.comb_temp, 1);
                if(fmpz_is_zero(work.z))
                  continue;
                fmpz_get_mpf(work.value, work.z);
                if(shift >= 0)
                  mpf_mul_2exp(work.value, work.value, shift);
                else
                  mpf_div_2exp(work.value, work.value, -shift);
              }
            auto &c = C.Ref(i, j).gmp_float;
            mpf_sub(c.get_mpf_t(), c.get_mpf_t(), work.value);
          }
      }
  }

  // Exponents of the largest entry of each row of A (rows = true) or of
  // each column (rows = false); no_exponent for zero rows/columns.
  template <class Get>
  std::vector<long> max_exponents(int count, int length, const Get &get)
  {
    std::vector<long> result(count, no_exponent);
    for(int a = 0; a < count; ++a)
      for(int l = 0; l < length; ++l)
        {
          long e;
          if(binary_exponent(get(a, l), e))
            result[a] = std::max(result[a], e);
        }
    return result;
  }
}

void multimodular_gemm_sub(El::Matrix<El::BigFloat> &C,
                           const El::Matrix<El::BigFloat> &A,
                           const El::Orientation orientation_B,
                           const El::Matrix<El::BigFloat> &B,
                           const bool lower_only,
                           const Multimodular_Blas_Config &config,
                           Multimodular_Blas_Timings &timings)
{
  std::vector<int> row_start(C.Width(), 0);
  if(lower_only)
    for(int j = 0; j < C.Width(); ++j)
      row_start[j] = j;
  multimodular_gemm_sub(C, A, orientation_B, B, row_start, config, timings);
}

void multimodular_gemm_sub(El::Matrix<El::BigFloat> &C,
                           const El::Matrix<El::BigFloat> &A,
                           const El::Orientation orientation_B,
                           const El::Matrix<El::BigFloat> &B,
                           const std::vector<int> &row_start,
                           const Multimodular_Blas_Config &config,
                           Multimodular_Blas_Timings &timings)
{
  const bool transpose_B = orientation_B != El::NORMAL;
  const int m = A.Height();
  const int k = A.Width();
  const int n = transpose_B ? B.Height() : B.Width();
  ASSERT_EQUAL(C.Height(), m);
  ASSERT_EQUAL(C.Width(), n);
  ASSERT_EQUAL(transpose_B ? B.Width() : B.Height(), k);
  ASSERT_EQUAL(row_start.size(), size_t(n));
  if(m == 0 || n == 0 || k == 0)
    return;

  auto op_B = [&](int l, int j) -> const El::BigFloat & {
    return transpose_B ? B.CRef(j, l) : B.CRef(l, j);
  };

  const long bits = El::gmp::Precision() + config.guard_bits;
  auto start = Clock::now();

  const std::vector<long> row_exp = max_exponents(
    m, k, [&](int i, int l) -> const El::BigFloat & { return A.CRef(i, l); });
  const std::vector<long> col_exp
    = max_exponents(n, k, [&](int j, int l) -> const El::BigFloat & {
        return op_B(l, j);
      });

  const Residue_Context &context = get_context(bits, bits, k);
  const size_t num_primes = context.num_primes;
  timings.max_num_primes = std::max(timings.max_num_primes, num_primes);
  ++timings.num_gemms;
  const bool flint_crt = config.flint_crt;

  // Residues of A (times u_p for the BLAS CRT):
  // for prime p, an m x k column-major matrix at p*m*k
  const size_t A_stride = size_t(m) * k;
  std::vector<double> A_residues(num_primes * A_stride);
  compute_residues(
    context, A_stride,
    [&](size_t e, long &shift) -> const __mpf_struct * {
      const int i = e % m, l = e / m;
      if(row_exp[i] == no_exponent)
        return nullptr;
      shift = bits - row_exp[i];
      return A.CRef(i, l).gmp_float.get_mpf_t();
    },
    !flint_crt, A_residues.data());
  timings.to_int_seconds += seconds_since(start);

  const int chunk = std::max(1, std::min(config.column_chunk, n));
  std::vector<double> B_residues(num_primes * size_t(k) * chunk);
  std::vector<double> C_residues(num_primes * size_t(m) * chunk);
  Crt_Workspace work(context);

  for(int j0 = 0; j0 < n; j0 += chunk)
    {
      const int width = std::min(chunk, n - j0);
      const size_t B_stride = size_t(k) * width;
      const size_t C_stride = size_t(m) * width;

      start = Clock::now();
      compute_residues(
        context, B_stride,
        [&](size_t e, long &shift) -> const __mpf_struct * {
          const int l = e % k, j = j0 + e / k;
          if(col_exp[j] == no_exponent)
            return nullptr;
          shift = bits - col_exp[j];
          return op_B(l, j).gmp_float.get_mpf_t();
        },
        false, B_residues.data());
      timings.to_int_seconds += seconds_since(start);

      start = Clock::now();
      for(size_t p = 0; p < num_primes; ++p)
        {
          cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, width, k,
                      1.0, A_residues.data() + p * A_stride, m,
                      B_residues.data() + p * B_stride, k, 0.0,
                      C_residues.data() + p * C_stride, m);
        }
      timings.blas_seconds += seconds_since(start);

      start = Clock::now();
      auto C_chunk = C(El::ALL, El::IR(j0, j0 + width));
      const std::vector<long> chunk_col_exp(col_exp.begin() + j0,
                                            col_exp.begin() + j0 + width);
      const std::vector<int> chunk_row_start(row_start.begin() + j0,
                                             row_start.begin() + j0 + width);
      crt_subtract(context, flint_crt, C_residues.data(), m, width, C_chunk,
                   row_exp, chunk_col_exp, 2 * bits, chunk_row_start, work);
      timings.from_int_seconds += seconds_since(start);
    }
}

// Left-looking blocked Trsm with each conversion done once.
// For each column chunk, for each panel of h <= base_size rows:
//   B_p -= L_{p,<p} X_{<p}   (multimodular GEMM, inner dimension r0)
//   X_p = L_pp^{-1} B_p      (substitution)
//   residues of X_p are stored for the following panels.
// X is stored with one fixed-point scale per column, 2^(bits_X - e_j),
// where e_j = (largest exponent seen in column j) + growth_bits; if a
// later panel exceeds 2^e_j, e_j is raised and the stored residues of that
// column are recomputed from the BigFloat values.
void multimodular_trsm_left_looking(const El::Matrix<El::BigFloat> &L,
                                    El::Matrix<El::BigFloat> &B,
                                    const Multimodular_Blas_Config &config,
                                    Multimodular_Blas_Timings &timings)
{
  const int P = L.Height();
  const int N = B.Width();
  ASSERT_EQUAL(L.Width(), P);
  ASSERT_EQUAL(B.Height(), P);
  const int b = std::max(1, config.base_size);
  if(P == 0 || N == 0)
    return;
  if(P <= b)
    {
      auto start = Clock::now();
      substitution_trsm(L, B);
      timings.base_seconds += seconds_since(start);
      return;
    }
  const long bits_L = El::gmp::Precision() + config.guard_bits;
  const long bits_X = bits_L + config.growth_bits;
  const Residue_Context &context = get_context(bits_L, bits_X, P);
  const size_t num_primes = context.num_primes;
  timings.max_num_primes = std::max(timings.max_num_primes, num_primes);
  const bool flint_crt = config.flint_crt;

  const int chunk = std::max<size_t>(
    1, std::min<size_t>(N, config.left_looking_bytes
                             / (sizeof(double) * num_primes * P)));
  const size_t X_stride = size_t(P) * chunk; // per prime
  std::vector<double> X_residues(num_primes * X_stride);
  std::vector<double> A_residues, C_residues, panel_residues;
  Crt_Workspace work(context);

  for(int j0 = 0; j0 < N; j0 += chunk)
    {
      const int width = std::min(chunk, N - j0);
      std::vector<long> col_e(width, no_exponent);
      for(int r0 = 0; r0 < P; r0 += b)
        {
          const int h = std::min(b, P - r0);
          auto R = B(El::IR(r0, r0 + h), El::IR(j0, j0 + width));
          if(r0 > 0)
            {
              auto start = Clock::now();
              const auto A = L(El::IR(r0, r0 + h), El::IR(0, r0));
              const std::vector<long> row_exp = max_exponents(
                h, r0, [&](int i, int l) -> const El::BigFloat & {
                  return A.CRef(i, l);
                });
              const size_t A_stride = size_t(h) * r0;
              A_residues.resize(num_primes * A_stride);
              compute_residues(
                context, A_stride,
                [&](size_t e, long &shift) -> const __mpf_struct * {
                  const int i = e % h, l = e / h;
                  if(row_exp[i] == no_exponent)
                    return nullptr;
                  shift = bits_L - row_exp[i];
                  return A.CRef(i, l).gmp_float.get_mpf_t();
                },
                !flint_crt, A_residues.data());
              timings.to_int_seconds += seconds_since(start);

              start = Clock::now();
              const size_t C_stride = size_t(h) * width;
              C_residues.resize(num_primes * C_stride);
              for(size_t p = 0; p < num_primes; ++p)
                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, h,
                            width, r0, 1.0, A_residues.data() + p * A_stride,
                            h, X_residues.data() + p * X_stride, P, 0.0,
                            C_residues.data() + p * C_stride, h);
              timings.blas_seconds += seconds_since(start);
              ++timings.num_gemms;

              start = Clock::now();
              crt_subtract(context, flint_crt, C_residues.data(), h, width, R,
                           row_exp, col_e, bits_L + bits_X,
                           std::vector<int>(width, 0), work);
              timings.from_int_seconds += seconds_since(start);
            }

          auto start = Clock::now();
          substitution_trsm(L(El::IR(r0, r0 + h), El::IR(r0, r0 + h)), R);
          timings.base_seconds += seconds_since(start);

          start = Clock::now();
          // Column scales: first nonzero panel sets them; growth beyond the
          // headroom raises them and recomputes the stored residues.
          const std::vector<long> panel_exp = max_exponents(
            width, h,
            [&](int j, int i) -> const El::BigFloat & { return R.CRef(i, j); });
          for(int j = 0; j < width; ++j)
            {
              if(panel_exp[j] == no_exponent || panel_exp[j] <= col_e[j])
                continue;
              const bool had_nonzero = col_e[j] != no_exponent;
              col_e[j] = panel_exp[j] + config.growth_bits;
              if(had_nonzero && r0 > 0)
                {
                  ++timings.num_rescales;
                  compute_residues(
                    context, r0,
                    [&](size_t e, long &shift) -> const __mpf_struct * {
                      shift = bits_X - col_e[j];
                      return B.CRef(e, j0 + j).gmp_float.get_mpf_t();
                    },
                    false, X_residues.data() + size_t(P) * j, X_stride);
                }
            }
          // Residues of the new panel rows, then into the X store
          const size_t panel_stride = size_t(h) * width;
          panel_residues.resize(num_primes * panel_stride);
          compute_residues(
            context, panel_stride,
            [&](size_t e, long &shift) -> const __mpf_struct * {
              const int i = e % h, j = e / h;
              if(col_e[j] == no_exponent)
                return nullptr;
              shift = bits_X - col_e[j];
              return R.CRef(i, j).gmp_float.get_mpf_t();
            },
            false, panel_residues.data());
          for(size_t p = 0; p < num_primes; ++p)
            for(int j = 0; j < width; ++j)
              std::copy_n(panel_residues.data() + p * panel_stride
                            + size_t(j) * h,
                          h,
                          X_residues.data() + p * X_stride + size_t(j) * P
                            + r0);
          timings.to_int_seconds += seconds_since(start);
        }
    }
}
