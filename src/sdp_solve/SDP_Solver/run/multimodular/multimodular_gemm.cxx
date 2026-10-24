#include "multimodular_blas.hxx"

#include "sdp_solve/SDP_Solver/run/bigint_syrk/fmpz/Fmpz_Comb.hxx"
#include "sdp_solve/SDP_Solver/run/bigint_syrk/fmpz/fmpz_mul_blas_util.hxx"
#include "sdpb_util/assert.hxx"

#include <cblas.h>

#include <chrono>
#include <map>
#include <memory>
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

namespace
{
  using Clock = std::chrono::steady_clock;
  double seconds_since(const Clock::time_point &start)
  {
    return std::chrono::duration<double>(Clock::now() - start).count();
  }

  // Residues of fixed-point integers by BLAS.
  // An integer |x| < 2^bits is split into T chunks of chunk_bits bits,
  // |x| = sum_t c_t 2^(chunk_bits t), and then
  //   |x| mod p_i = (sum_t c_t W_ti) mod p_i,  W_ti = 2^(chunk_bits t) mod p_i.
  // The sum is exact in double: T * 2^chunk_bits * p_i < 2^53.
  // For n integers this is one (n x T) * (T x num_primes) dgemm.
  constexpr int chunk_bits = 24;

  struct Residue_Context
  {
    Fmpz_Comb comb; // primes and CRT
    int num_chunks;
    std::vector<double> primes, inverse_primes;
    std::vector<double> W; // num_chunks x num_primes, column-major

    Residue_Context(flint_bitcnt_t bits, slong k)
        : comb(bits, bits, 1, k),
          num_chunks((bits + chunk_bits - 1) / chunk_bits)
    {
      const size_t num_primes = comb.num_primes;
      primes.resize(num_primes);
      inverse_primes.resize(num_primes);
      W.resize(size_t(num_chunks) * num_primes);
      for(size_t i = 0; i < num_primes; ++i)
        {
          const mp_limb_t p = comb.primes.at(i);
          ASSERT(std::ldexp(double(p) * num_chunks, chunk_bits)
                   < std::ldexp(1.0, 53),
                 "chunked residues would not be exact: p=", p,
                 " num_chunks=", num_chunks);
          primes[i] = p;
          inverse_primes[i] = 1.0 / double(p);
          mp_limb_t power = 1; // 2^(chunk_bits t) mod p
          const mp_limb_t base = (mp_limb_t(1) << chunk_bits) % p;
          for(int t = 0; t < num_chunks; ++t)
            {
              W[t + size_t(num_chunks) * i] = double(power);
              power = (power * base) % p;
            }
        }
    }
  };

  // Primes and CRT data depend only on (bits, k); build them once.
  Residue_Context &get_context(flint_bitcnt_t bits, slong k)
  {
    static std::map<std::pair<flint_bitcnt_t, slong>,
                    std::unique_ptr<Residue_Context>>
      cache;
    auto &context = cache[{bits, k}];
    if(!context)
      context = std::make_unique<Residue_Context>(bits, k);
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

  // Chunks of trunc(|x| * 2^shift), x an mpf:
  // chunks[t * stride] = bits [chunk_bits t, chunk_bits (t+1)) of it.
  // Returns the sign of x.
  int write_chunks(const __mpf_struct *x, long shift, int num_chunks,
                   double *chunks, size_t stride)
  {
    const int size = std::abs(x->_mp_size);
    const mp_limb_t *d = x->_mp_d;
    // bit position (in the integer) of bit 0 of limb 0
    const long offset = long(GMP_NUMB_BITS) * (x->_mp_exp - size) + shift;
    constexpr mp_limb_t mask = (mp_limb_t(1) << chunk_bits) - 1;
    for(int t = 0; t < num_chunks; ++t)
      {
        // bit position q in the mantissa limbs of bit chunk_bits*t
        const long q = long(chunk_bits) * t - offset;
        mp_limb_t value = 0;
        if(q + chunk_bits > 0 && q < long(GMP_NUMB_BITS) * size)
          {
            if(q < 0)
              value = d[0] << (-q);
            else
              {
                const long li = q / GMP_NUMB_BITS;
                const int bo = q % GMP_NUMB_BITS;
                value = d[li] >> bo;
                if(bo > GMP_NUMB_BITS - chunk_bits && li + 1 < size)
                  value |= d[li + 1] << (GMP_NUMB_BITS - bo);
              }
          }
        chunks[t * stride] = double(value & mask);
      }
    return x->_mp_size < 0 ? -1 : 1;
  }

  // Residues of n fixed-point integers trunc(x_e * 2^shift_e):
  // out[e + n * i] = residue modulo prime i in (-p_i/2, p_i/2).
  // element(e, shift) returns x_e (nullptr for zero) and sets shift.
  template <class Element>
  void compute_residues(const Residue_Context &context, size_t n,
                        const Element &element, double *out)
  {
    const size_t num_primes = context.primes.size();
    const int T = context.num_chunks;
    constexpr size_t batch = 4096;
    std::vector<double> chunks(batch * T);
    std::vector<int8_t> signs(batch);
    for(size_t e0 = 0; e0 < n; e0 += batch)
      {
        const size_t nb = std::min(batch, n - e0);
        for(size_t e = 0; e < nb; ++e)
          {
            long shift;
            const __mpf_struct *x = element(e0 + e, shift);
            if(x == nullptr || x->_mp_size == 0)
              {
                signs[e] = 0;
                for(int t = 0; t < T; ++t)
                  chunks[e + t * nb] = 0;
              }
            else
              signs[e] = write_chunks(x, shift, T, chunks.data() + e, nb);
          }
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nb,
                    num_primes, T, 1.0, chunks.data(), nb, context.W.data(),
                    T, 0.0, out + e0, n);
        for(size_t i = 0; i < num_primes; ++i)
          {
            const double p = context.primes[i];
            const double inv_p = context.inverse_primes[i];
            const double half_p = 0.5 * p;
            double *r = out + e0 + n * i;
            for(size_t e = 0; e < nb; ++e)
              {
                // y mod p in [0, p), exact: y < 2^53
                const double y = r[e];
                double v = y - p * std::floor(y * inv_p);
                if(v < 0)
                  v += p;
                else if(v >= p)
                  v -= p;
                if(v > half_p)
                  v -= p;
                r[e] = signs[e] < 0 ? -v : v;
              }
          }
      }
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
  const bool transpose_B = orientation_B != El::NORMAL;
  const int m = A.Height();
  const int k = A.Width();
  const int n = transpose_B ? B.Height() : B.Width();
  ASSERT_EQUAL(C.Height(), m);
  ASSERT_EQUAL(C.Width(), n);
  ASSERT_EQUAL(transpose_B ? B.Width() : B.Height(), k);
  if(m == 0 || n == 0 || k == 0)
    return;

  auto op_B = [&](int l, int j) -> const El::BigFloat & {
    return transpose_B ? B.CRef(j, l) : B.CRef(l, j);
  };

  const long bits = El::gmp::Precision() + config.guard_bits;
  auto start = Clock::now();

  // Row exponents of A and column exponents of op(B).
  // Zero rows/columns get the flag false and are skipped.
  constexpr long no_exponent = std::numeric_limits<long>::min();
  std::vector<long> row_exp(m, no_exponent), col_exp(n, no_exponent);
  for(int l = 0; l < k; ++l)
    for(int i = 0; i < m; ++i)
      {
        long e;
        if(binary_exponent(A.CRef(i, l), e))
          row_exp[i] = std::max(row_exp[i], e);
      }
  for(int j = 0; j < n; ++j)
    for(int l = 0; l < k; ++l)
      {
        long e;
        if(binary_exponent(op_B(l, j), e))
          col_exp[j] = std::max(col_exp[j], e);
      }

  Residue_Context &context = get_context(bits, k);
  Fmpz_Comb &comb = context.comb;
  const size_t num_primes = comb.num_primes;
  timings.max_num_primes = std::max(timings.max_num_primes, num_primes);
  ++timings.num_gemms;

  fmpz_t z;
  fmpz_init(z);

  // Residues of A: for prime p, an m x k column-major matrix at p*m*k
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
    A_residues.data());
  timings.to_int_seconds += seconds_since(start);

  const int chunk = std::max(1, std::min(config.column_chunk, n));
  std::vector<double> B_residues(num_primes * size_t(k) * chunk);
  std::vector<double> C_residues(num_primes * size_t(m) * chunk);
  std::vector<mp_limb_t> crt_input(num_primes);
  mpf_t value;
  mpf_init2(value, El::gmp::Precision());

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
        B_residues.data());
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
      for(int jj = 0; jj < width; ++jj)
        {
          const int j = j0 + jj;
          if(col_exp[j] == no_exponent)
            continue;
          for(int i = lower_only ? std::max(j, 0) : 0; i < m; ++i)
            {
              if(row_exp[i] == no_exponent)
                continue;
              const double *in = C_residues.data() + i + size_t(jj) * m;
              for(size_t p = 0; p < num_primes; ++p)
                {
                  const double d = in[p * C_stride];
                  crt_input[p] = double_to_uint32_t_residue(d, comb.mods[p]);
                }
              fmpz_multi_CRT_ui(z, crt_input.data(), comb.comb,
                                comb.comb_temp, 1);
              if(fmpz_is_zero(z))
                continue;
              fmpz_get_mpf(value, z);
              const long shift = row_exp[i] + col_exp[j] - 2 * bits;
              if(shift >= 0)
                mpf_mul_2exp(value, value, shift);
              else
                mpf_div_2exp(value, value, -shift);
              auto &c = C.Ref(i, j).gmp_float;
              mpf_sub(c.get_mpf_t(), c.get_mpf_t(), value);
            }
        }
      timings.from_int_seconds += seconds_since(start);
    }
  mpf_clear(value);
  fmpz_clear(z);
}
