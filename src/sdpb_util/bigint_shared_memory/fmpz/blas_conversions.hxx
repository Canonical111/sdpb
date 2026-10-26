#pragma once

// lab (bfss #77): residues and CRT by BLAS for the residue windows of
// sdpa-solver's trmm (Initialize_P_Context), on the primes of an existing
// Fmpz_Comb. Adapted from bootstrap-speed's blas_residues (sdpb
// lab/176-mm-pr 8a22a21e, bootstrap-speed #145/#176), with two changes so
// that the results are bitwise equal to the FLINT routines in residues.cxx:
//
// Residues. x = trunc(v) (as fmpz_set_mpf) is split into 24-bit chunks c_t;
//   x mod p = sum_t c_t (2^(24t) mod p) mod p, exact in double for
//   T 2^24 p < 2^53. One dgemm per batch. The residue is returned in FLINT's
//   representative, [-(p-1)/2, (p-1)/2] (uint32_t_residue_to_double), as a
//   double. So the window holds the same doubles as with
//   bigint_to_residues().
//
// CRT. With M = prod p, |c| < M/2, y_p = c u_p mod p in [0, p),
//   u_p = (M/p)^-1 mod p:  c = sum_p y_p (M/p) - q M, q = round(sum_p y_p/p).
//   M/p is split into 24-bit chunks m_pt; S_t = sum_p y_p m_pt < 2^53 is
//   exact, and one dgemm gives all S_t of a batch. Unlike #176 all chunks are
//   kept (t0 = 0), so c is exact; it is converted with mpf_set_z, as
//   fmpz_get_mpf does. The result is therefore bitwise equal to
//   residues_to_bigint() + to_BigFloat(). If sum_p y_p/p is within 1e-6 of a
//   half-integer (q ambiguous; impossible for |c| < M/4), FLINT's CRT is used.

#include "Fmpz_Comb.hxx"

#include <El.hpp>

#include <cstdint>
#include <vector>

namespace Lab_Blas_Conversions
{
  constexpr int chunk_bits = 24;

  struct Context
  {
    const Fmpz_Comb &comb;
    size_t num_primes;
    std::vector<double> primes, inverse_primes, half_primes;
    std::vector<mp_limb_t> u; // (M/p)^-1 mod p
    // residues of integers |x| < 2^(24 num_chunks):
    // W = 2^(24t) mod p, num_chunks x num_primes, column-major
    int num_chunks;
    std::vector<double> W;
    // CRT: chunks of M/p, num_primes x num_crt_chunks, column-major
    int num_crt_chunks;
    std::vector<double> M_chunks;
    // q M for q = 0..num_primes, num_limbs limbs each
    int num_limbs;
    std::vector<mp_limb_t> qM;

    // input_bits: |trunc(v)| < 2^input_bits for the residue inputs
    Context(const Fmpz_Comb &comb, long input_bits);
  };

  // Scratch space, one per thread of work
  struct Workspace
  {
    std::vector<double> chunks, residues, S, quotient;
    std::vector<mp_limb_t> buffer, acc;
    std::vector<mp_limb_t> crt_input;
    Fmpz_BigInt bigint;
    explicit Workspace(const Context &context);
  };

  constexpr size_t batch_size = 1024;

  // Residues of n <= batch_size values: residue of trunc(*values[e]) modulo
  // prime i is written to out[e][i * prime_stride].
  void compute_residues(const Context &context, size_t n,
                        const El::BigFloat *const *values, double *const *out,
                        size_t prime_stride, Workspace &work);

  // CRT of n <= batch_size integers whose residues (integer-valued doubles,
  // |d| <= 2^53, any representative) are in[e][i * prime_stride];
  // output[e] := the integer, as fmpz_get_mpf with output[e]'s precision.
  void restore(const Context &context, size_t n, const double *const *in,
               size_t prime_stride, El::BigFloat *const *output,
               Workspace &work);
}
