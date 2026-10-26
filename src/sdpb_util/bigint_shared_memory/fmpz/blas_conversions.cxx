#include "blas_conversions.hxx"

#include "residues.hxx"
#include "sdpb_util/assert.hxx"

#include <flint/ulong_extras.h>

#include <cblas.h>

#include <cmath>
#include <cstring>

// lab (bfss #77). See blas_conversions.hxx.

namespace Lab_Blas_Conversions
{
  namespace
  {
    // Bits [24 t, 24 (t+1)) of trunc(|x|), t = 0..num_chunks-1, times the
    // sign of x, to chunks[t * stride]. Returns false if trunc(|x|) does not
    // fit in num_chunks chunks (chunks then undefined).
    bool write_chunks(const __mpf_struct *x, const int num_chunks,
                      double *chunks, const size_t stride,
                      std::vector<mp_limb_t> &buffer)
    {
      const long size = std::abs(x->_mp_size);
      const long num_limbs = buffer.size();
      std::fill(buffer.begin(), buffer.end(), 0);
      if(size != 0 && x->_mp_exp > 0)
        {
          const mp_limb_t *d = x->_mp_d;
          // integer part: limbs d[size - exp .. size) (or all, padded
          // with exp - size zero limbs below)
          const long exp = x->_mp_exp;
          if(exp > num_limbs)
            return false;
          if(exp >= size)
            std::copy(d, d + size, buffer.data() + (exp - size));
          else
            std::copy(d + (size - exp), d + size, buffer.data());
          // check that the integer has at most 24 num_chunks bits
          const long top_bits
            = long(chunk_bits) * num_chunks - GMP_NUMB_BITS * (exp - 1);
          if(top_bits < GMP_NUMB_BITS)
            {
              if(top_bits <= 0 || (buffer[exp - 1] >> top_bits) != 0)
                return false;
            }
        }
      const double sign = x->_mp_size < 0 ? -1.0 : 1.0;
      const unsigned char *bytes
        = reinterpret_cast<const unsigned char *>(buffer.data());
      for(int t = 0; t < num_chunks; ++t)
        {
          uint32_t v;
          std::memcpy(&v, bytes + 3 * t, sizeof(v));
          chunks[t * stride] = sign * double(v & 0xFFFFFFu);
        }
      return true;
    }

    // v - round(v/p) p in [-(p+1)/2, (p+1)/2], for an integer-valued
    // double |v| < 2^52 (as bootstrap-speed's round_mod). Rounding by the
    // 1.5*2^52 trick, no floor() call (SDPB builds without -march), and
    // branch-free, so that the loops vectorize.
    inline double round_mod(const double v, const double p, const double inv_p)
    {
      constexpr double magic = 6755399441055744.0; // 1.5 * 2^52
      const double q = (v * inv_p + magic) - magic;
      return v - q * p;
    }
    // v mod p in [0, p)
    inline double mod_p(const double v, const double p, const double inv_p)
    {
      const double r = round_mod(v, p, inv_p);
      return r + ((r < 0) ? p : 0.0);
    }
    // v mod p in FLINT's representative [-(p-1)/2, (p-1)/2]
    inline double symmetric_mod_p(const double v, const double p,
                                  const double inv_p, const double half)
    {
      double r = round_mod(v, p, inv_p);
      r -= (r > half) ? p : 0.0;
      r += (r < -half) ? p : 0.0;
      return r;
    }

    // Length of the run of consecutive addresses ptrs[e], ptrs[e] + 1, ...
    template <class T>
    size_t run_length(T *const *ptrs, const size_t e, const size_t n)
    {
      size_t len = 1;
      while(e + len < n && ptrs[e + len] == ptrs[e] + len)
        ++len;
      return len;
    }
  }

  Context::Context(const Fmpz_Comb &comb, const long input_bits)
      : comb(comb),
        num_primes(comb.num_primes),
        num_chunks((input_bits + chunk_bits - 1) / chunk_bits)
  {
    primes.resize(num_primes);
    inverse_primes.resize(num_primes);
    half_primes.resize(num_primes);
    u.resize(num_primes);
    W.resize(size_t(num_chunks) * num_primes);

    fmpz_t M, M_p;
    fmpz_init_set_ui(M, 1);
    fmpz_init(M_p);
    for(size_t i = 0; i < num_primes; ++i)
      fmpz_mul_ui(M, M, comb.primes[i]);
    ASSERT(fmpz_equal(M, comb.primes_product.value));
    const long M_bits = fmpz_bits(M);
    num_crt_chunks = (M_bits + chunk_bits - 1) / chunk_bits;
    M_chunks.resize(num_primes * num_crt_chunks);
    // sum_t S_t 2^(24 t) < num_primes 2^45 2^(24 num_crt_chunks)
    num_limbs = (chunk_bits * num_crt_chunks + 64) / GMP_NUMB_BITS + 2;

    mpz_t z;
    mpz_init(z);
    for(size_t i = 0; i < num_primes; ++i)
      {
        const mp_limb_t p = comb.primes[i];
        ASSERT(p % 2 == 1 && p < (mp_limb_t(1) << 21)
                 && std::ldexp(double(p) * num_chunks, chunk_bits)
                      < std::ldexp(1.0, 52)
                 && std::ldexp(double(p) * num_primes, chunk_bits)
                      < std::ldexp(1.0, 52),
               "chunked residues/CRT would not be exact: p=", p,
               " num_chunks=", num_chunks, " num_primes=", num_primes);
        primes[i] = p;
        inverse_primes[i] = 1.0 / double(p);
        half_primes[i] = double((p - 1) / 2);
        mp_limb_t power = 1;
        const mp_limb_t base = (mp_limb_t(1) << chunk_bits) % p;
        for(int t = 0; t < num_chunks; ++t)
          {
            W[t + size_t(num_chunks) * i] = double(power);
            power = (power * base) % p;
          }
        fmpz_divexact_ui(M_p, M, p);
        u[i] = n_invmod(fmpz_fdiv_ui(M_p, p), p);
        fmpz_get_mpz(z, M_p);
        for(int t = 0; t < num_crt_chunks; ++t)
          {
            mp_limb_t chunk = 0;
            for(int b = 0; b < chunk_bits; ++b)
              chunk |= mp_limb_t(mpz_tstbit(z, chunk_bits * t + b)) << b;
            M_chunks[i + num_primes * t] = double(chunk);
          }
      }
    qM.assign((num_primes + 1) * num_limbs, 0);
    for(size_t q = 0; q <= num_primes; ++q)
      {
        fmpz_get_mpz(z, M);
        mpz_mul_ui(z, z, q);
        ASSERT(mpz_size(z) <= size_t(num_limbs));
        for(size_t l = 0; l < mpz_size(z); ++l)
          qM[q * num_limbs + l] = mpz_getlimbn(z, l);
      }
    mpz_clear(z);
    fmpz_clear(M_p);
    fmpz_clear(M);
  }

  Workspace::Workspace(const Context &context)
      : chunks(batch_size * context.num_chunks),
        residues(batch_size * context.num_primes),
        S(batch_size * context.num_crt_chunks),
        quotient(batch_size),
        buffer((long(chunk_bits) * context.num_chunks + 3 * 8)
                 / GMP_NUMB_BITS
               + 2),
        acc(context.num_limbs),
        crt_input(context.num_primes)
  {}

  void compute_residues(const Context &context, const size_t n,
                        const El::BigFloat *const *values, double *const *out,
                        const size_t prime_stride, Workspace &work)
  {
    ASSERT(n <= batch_size);
    if(n == 0)
      return;
    const size_t num_primes = context.num_primes;
    const int T = context.num_chunks;
    // Elements that don't fit (should not happen) are done by FLINT
    std::vector<size_t> fallback;
    for(size_t e = 0; e < n; ++e)
      {
        if(!write_chunks(values[e]->gmp_float.get_mpf_t(), T,
                         work.chunks.data() + e, n, work.buffer))
          {
            fallback.push_back(e);
            for(int t = 0; t < T; ++t)
              work.chunks[e + t * n] = 0;
          }
      }
    // |sum| < T 2^24 p < 2^52
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, num_primes, T,
                1.0, work.chunks.data(), n, context.W.data(), T, 0.0,
                work.residues.data(), n);
    for(size_t i = 0; i < num_primes; ++i)
      {
        const double p = context.primes[i];
        const double inv_p = context.inverse_primes[i];
        const double half = context.half_primes[i];
        double *__restrict r = work.residues.data() + n * i;
        for(size_t e = 0; e < n; ++e)
          {
            // FLINT's representative, [-(p-1)/2, (p-1)/2]; +0.0, not -0.0
            r[e] = symmetric_mod_p(r[e], p, inv_p, half) + 0.0;
          }
      }
    // write runs of consecutive elements with memcpy
    for(size_t e = 0; e < n;)
      {
        const size_t len = run_length(out, e, n);
        for(size_t i = 0; i < num_primes; ++i)
          std::memcpy(out[e] + prime_stride * i,
                      work.residues.data() + n * i + e, len * sizeof(double));
        e += len;
      }
    for(const size_t e : fallback)
      {
        work.bigint.from_BigFloat(*values[e]);
        bigint_to_residues(work.bigint, context.comb, out[e], prime_stride);
      }
  }

  void restore(const Context &context, const size_t n,
               const double *const *in, const size_t prime_stride,
               El::BigFloat *const *output, Workspace &work)
  {
    ASSERT(n <= batch_size);
    if(n == 0)
      return;
    const size_t num_primes = context.num_primes;
    const int H = context.num_crt_chunks;
    const int L = context.num_limbs;
    // y_p = c u_p mod p in [0, p), as doubles, n x num_primes column-major
    double *Y = work.residues.data();
    for(size_t e = 0; e < n;)
      {
        const size_t len = run_length(in, e, n);
        for(size_t i = 0; i < num_primes; ++i)
          std::memcpy(Y + n * i + e, in[e] + prime_stride * i,
                      len * sizeof(double));
        e += len;
      }
    double *__restrict quotient = work.quotient.data();
    std::fill(quotient, quotient + n, 0.0);
    bool in_range = true;
    for(size_t i = 0; i < num_primes; ++i)
      {
        const double p = context.primes[i];
        const double inv_p = context.inverse_primes[i];
        const double u = double(context.u[i]);
        double *__restrict y = Y + n * i;
        for(size_t e = 0; e < n; ++e)
          {
            const double d = y[e];
            in_range &= std::abs(d) < 0x1p52;
            const double v = mod_p(mod_p(d, p, inv_p) * u, p, inv_p);
            y[e] = v;
            quotient[e] += v * inv_p;
          }
      }
    ASSERT(in_range, "BLAS CRT: residue outside (-2^52, 2^52)");
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, H, num_primes,
                1.0, Y, n, context.M_chunks.data(), num_primes, 0.0,
                work.S.data(), n);
    mp_limb_t *acc = work.acc.data();
    for(size_t e = 0; e < n; ++e)
      {
        const double quotient = work.quotient[e];
        const double q_rounded = std::nearbyint(quotient);
        if(std::abs(quotient - q_rounded) > 0.5 - 1e-6)
          {
            // q ambiguous: exact FLINT CRT, as residues_to_bigint()
            for(size_t i = 0; i < num_primes; ++i)
              {
                const auto &mod = context.comb.mods.at(i);
                work.crt_input[i]
                  = double_to_uint32_t_residue(in[e][prime_stride * i], mod);
              }
            Fmpz_Comb &comb = const_cast<Fmpz_Comb &>(context.comb);
            fmpz_multi_CRT_ui(work.bigint.value, work.crt_input.data(),
                              comb.comb, comb.comb_temp, 1);
            work.bigint.to_BigFloat(*output[e]);
            continue;
          }
        const size_t q = size_t(q_rounded);
        ASSERT(q <= num_primes, DEBUG_STRING(q));
        // acc = sum_t S_t 2^(24 t), then - q M, two's complement
        std::fill(acc, acc + L, 0);
        for(int t = 0; t < H; ++t)
          {
            const mp_limb_t v = mp_limb_t(work.S[e + n * t]);
            const int o = chunk_bits * t;
            int li = o / GMP_NUMB_BITS;
            const int sh = o % GMP_NUMB_BITS;
            const mp_limb_t lo = v << sh;
            const mp_limb_t hi = sh == 0 ? 0 : v >> (GMP_NUMB_BITS - sh);
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
        mpn_sub_n(acc, acc, context.qM.data() + q * L, L);
        int sign = 1;
        if(acc[L - 1] >> (GMP_NUMB_BITS - 1))
          {
            sign = -1;
            mpn_neg(acc, acc, L);
          }
        int size = L;
        while(size > 0 && acc[size - 1] == 0)
          --size;
        // as fmpz_get_mpf: mpf_set_z (mpf_set_si for |c| < 2^62 gives the
        // same value and limbs)
        __mpz_struct z;
        z._mp_alloc = L;
        z._mp_size = sign * size;
        z._mp_d = acc;
        mpf_set_z(output[e]->gmp_float.get_mpf_t(), &z);
      }
  }
}
