#include "sdp_solve/SDP_Solver.hxx"

#include <gmp.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

// min(gamma \alpha(M, dM), 1), where \alpha(M, dM) denotes the
// largest positive real number such that M + \alpha dM is positive
// semidefinite.
//
// \alpha(M, dM) is computed with a Cholesky decomposition M = L L^T.
// The eigenvalues of M + \alpha dM are equal to the eigenvalues of 1
// + \alpha L^{-1} dM L^{-T}.  The correct \alpha is then -1/lambda,
// where lambda is the smallest eigenvalue of L^{-1} dM L^{-T}.
//
// Inputs:
// - MCholesky = L, the Cholesky decomposition of M (M itself is not needed)
// - dM, a Block_Diagonal_Matrix with the same structure as M
// Workspace:
// - MInvDM (NB: overwritten when computing minEigenvalue)
// - eigenvalues, a Vector of eigenvalues for each block of M
// Output:
// - min(\gamma \alpha(M, dM), 1) (returned)
//
// lab/33-steplength prototype (bootstrap-speed #33), selected at run time by
// the environment variable SDPB_STEP_LENGTH_MODE:
//   unset or "reference": the original algorithm (full BigFloat
//     HermitianEig of every block), with sub-timers "congruence" and
//     "min_eigenvalue".
//   "double": guarded double-precision candidate.  For each locally owned
//     block, A = L^{-1} dM L^{-T} is still computed in BigFloat.  Its lower
//     triangle is cast to double after an exact power-of-two rescaling and
//     the smallest eigenvalue is computed with LAPACK dsyevr (RANGE='I').
//     A block falls back to the BigFloat eigensolve if the cast is
//     nonfinite, LAPACK fails, the block is distributed over several ranks,
//     or the a-priori error estimate 8 n u ||A||_F exceeds
//     SDPB_STEP_LENGTH_TOL (default 1e-8) times max(|lambda|, gamma).
//     The resulting alpha is then checked in full precision: BigFloat
//     Cholesky of I + alpha A for every block whose eigenvalue came from
//     double.  If any of those fails, the step length is recomputed with
//     the reference algorithm.
// Diagnostics: SDPB_STEP_LENGTH_COMPARE=1 also runs the reference
// eigensolve and records the difference (for checks, not timing);
// SDPB_STEP_LENGTH_LOG=<path> appends one line per call;
// SDPB_STEP_LENGTH_INJECT=nonfinite|lapack|gate|cholesky forces each
// fallback path.  SDPB_STEP_LENGTH_CONGRUENCE=twosided replaces the two Trsm
// of the congruence by El::TwoSidedTrsm (lower triangle only), in either
// mode.  Counters are printed to stderr at exit.

// A := L^{-1} A L^{-T}
void lower_triangular_inverse_congruence(const Block_Diagonal_Matrix &L,
                                         Block_Diagonal_Matrix &A);

El::BigFloat min_eigenvalue(Block_Diagonal_Matrix &A);

extern "C" void
dsyevr_(const char *jobz, const char *range, const char *uplo, const int *n,
        double *a, const int *lda, const double *vl, const double *vu,
        const int *il, const int *iu, const double *abstol, int *m, double *w,
        double *z, const int *ldz, int *isuppz, double *work,
        const int *lwork, int *iwork, const int *liwork, int *info,
        size_t jobz_len, size_t range_len, size_t uplo_len);

namespace
{
  struct Step_Length_Stats
  {
    long calls = 0, blocks_double = 0, blocks_fallback_gate = 0,
         blocks_fallback_nonfinite = 0, blocks_fallback_lapack = 0,
         blocks_fallback_dist = 0, guard_failures = 0;
    double max_rel_err_estimate = 0, max_compare_rel_diff = 0;
    ~Step_Length_Stats()
    {
      if(calls > 0)
        std::fprintf(
          stderr,
          "steplength-prototype: calls=%ld blocks_double=%ld "
          "fallback_gate=%ld fallback_nonfinite=%ld fallback_lapack=%ld "
          "fallback_dist=%ld guard_failures=%ld max_rel_err_estimate=%.3e "
          "max_compare_rel_diff=%.3e\n",
          calls, blocks_double, blocks_fallback_gate,
          blocks_fallback_nonfinite, blocks_fallback_lapack,
          blocks_fallback_dist, guard_failures, max_rel_err_estimate,
          max_compare_rel_diff);
    }
  };
  Step_Length_Stats stats;

  std::string env_string(const char *name)
  {
    const char *v = std::getenv(name);
    return v ? std::string(v) : std::string();
  }

  // Subtract in the unsigned domain before narrowing. The ordered difference
  // is representable in unsigned long even across LONG_MIN/LONG_MAX.
  int clamped_exponent_difference(long a, long b)
  {
    if(a >= b)
      return int(std::min(static_cast<unsigned long>(a)
                            - static_cast<unsigned long>(b), 2000UL));
    return -int(std::min(static_cast<unsigned long>(b)
                          - static_cast<unsigned long>(a), 2000UL));
  }

  unsigned long exponent_magnitude(long e)
  {
    return e >= 0 ? static_cast<unsigned long>(e)
                  : 0UL - static_cast<unsigned long>(e);
  }

  enum class Block_Result
  {
    ok,
    nonfinite,
    lapack,
    gate
  };

  // Smallest eigenvalue of the lower triangle of a local BigFloat block:
  // lambda = lambda_scaled * 2^scale_exp.
  Block_Result
  double_min_eigenvalue(const El::Matrix<El::BigFloat> &block,
                        const double gamma, const double tol,
                        const std::string &inject, double &lambda_scaled,
                        long &scale_exp, double &rel_err_estimate)
  {
    const int n = block.Height();
    // mpf_get_d_2exp gives d in [0.5,1) and an exact exponent, so no entry
    // is converted without constructing a double at its original exponent.
    // Exponent differences must also be clamped before signed arithmetic.
    std::vector<double> a(size_t(n) * n, 0.0);
    std::vector<long> exps(size_t(n) * n, 0);
    long max_exp = std::numeric_limits<long>::min();
    for(int j = 0; j < n; ++j)
      for(int i = j; i < n; ++i)
        {
          long e;
          const double d
            = mpf_get_d_2exp(&e, block.CRef(i, j).gmp_float.get_mpf_t());
          a[i + size_t(j) * n] = d;
          exps[i + size_t(j) * n] = e;
          if(d != 0 && e > max_exp)
            max_exp = e;
        }
    if(max_exp == std::numeric_limits<long>::min())
      max_exp = 0; // zero block
    scale_exp = max_exp;
    double frob2 = 0;
    for(int j = 0; j < n; ++j)
      for(int i = j; i < n; ++i)
        {
          const size_t k = i + size_t(j) * n;
          // Entries more than 2^-2000 below the largest flush to zero.
          a[k] = std::ldexp(a[k], clamped_exponent_difference(exps[k], max_exp));
          frob2 += (i == j ? 1.0 : 2.0) * a[k] * a[k];
        }
    if(inject == "nonfinite" && n > 0)
      a[0] = std::numeric_limits<double>::quiet_NaN();
    for(int j = 0; j < n; ++j)
      for(int i = j; i < n; ++i)
        if(!std::isfinite(a[i + size_t(j) * n]))
          return Block_Result::nonfinite;

    const char jobz = 'N', range = 'I', uplo = 'L';
    const int il = 1, iu = 1, ldz = 1, nn = std::max(n, 1);
    const double vl = 0, vu = 0, abstol = 0;
    int m = 0, info = 0, isuppz[2];
    std::vector<double> w(nn);
    double z = 0;
    const int lwork = 26 * nn, liwork = 10 * nn;
    std::vector<double> work(lwork);
    std::vector<int> iwork(liwork);
    dsyevr_(&jobz, &range, &uplo, &n, a.data(), &nn, &vl, &vu, &il, &iu,
            &abstol, &m, w.data(), &z, &ldz, isuppz, work.data(), &lwork,
            iwork.data(), &liwork, &info, 1, 1, 1);
    if(inject == "lapack")
      info = 1;
    if(info != 0 || m != 1 || !std::isfinite(w[0]))
      return Block_Result::lapack;
    lambda_scaled = w[0];

    // A-priori error estimate: cast error u ||A||_F plus a backward-stable
    // eigensolver error of a modest multiple of n u ||A||_2 <= n u ||A||_F.
    // Not a rigorous bound (the constant is heuristic); interiority of the
    // accepted step is checked separately by the guard Cholesky.
    const double u = std::numeric_limits<double>::epsilon() / 2;
    const double err = 8.0 * nn * u * std::sqrt(frob2);
    const double gamma_scaled
      = std::ldexp(gamma, clamped_exponent_difference(0, max_exp));
    const double denom = std::max(std::abs(lambda_scaled), gamma_scaled);
    rel_err_estimate = err / denom;
    if(inject == "gate" || !(rel_err_estimate <= tol))
      return Block_Result::gate;
    return Block_Result::ok;
  }

  El::BigFloat alpha_from_lambda(const El::BigFloat &lambda,
                                 const El::BigFloat &gamma)
  {
    if(lambda > -gamma)
      return 1;
    return -gamma / lambda;
  }
}

El::BigFloat step_length(const Block_Diagonal_Matrix &MCholesky,
                         const Block_Diagonal_Matrix &dM,
                         const El::BigFloat &gamma,
                         const std::string &timer_name,
                         Timers &timers)
{
  Scoped_Timer step_length_timer(timers, timer_name);
  static const std::string mode = env_string("SDPB_STEP_LENGTH_MODE");
  static const std::string inject = env_string("SDPB_STEP_LENGTH_INJECT");
  static const bool compare = env_string("SDPB_STEP_LENGTH_COMPARE") == "1";
  static const std::string log_path = env_string("SDPB_STEP_LENGTH_LOG");
  static const double tol
    = env_string("SDPB_STEP_LENGTH_TOL").empty()
        ? 1e-8
        : std::atof(env_string("SDPB_STEP_LENGTH_TOL").c_str());
  static const std::string congruence
    = env_string("SDPB_STEP_LENGTH_CONGRUENCE");
  ++stats.calls;

  // MInvDM = L^{-1} dM L^{-T}, where M = L L^T
  Block_Diagonal_Matrix MInvDM(dM);
  {
    Scoped_Timer congruence_timer(timers, "congruence");
    if(congruence.empty() || congruence == "trsm")
      lower_triangular_inverse_congruence(MCholesky, MInvDM);
    else if(congruence == "twosided")
      {
        // Symmetric two-sided solve (LAPACK sygst-like), about half the
        // flops of two Trsm.  Only the LOWER triangle of the result is
        // computed; every consumer below (HermitianEig LOWER, dsyevr 'L',
        // Cholesky LOWER) reads only that triangle.
        for(size_t b = 0; b < MInvDM.blocks.size(); ++b)
          {
            if(MInvDM.blocks[b].Grid().Size() == 1)
              El::TwoSidedTrsm(El::UpperOrLowerNS::LOWER,
                               El::UnitOrNonUnit::NON_UNIT,
                               MInvDM.blocks[b].Matrix(),
                               MCholesky.blocks[b].LockedMatrix());
            else
              El::TwoSidedTrsm(El::UpperOrLowerNS::LOWER,
                               El::UnitOrNonUnit::NON_UNIT, MInvDM.blocks[b],
                               MCholesky.blocks[b]);
          }
      }
    else
      RUNTIME_ERROR("Unknown SDPB_STEP_LENGTH_CONGRUENCE: ", congruence);
  }
  if(mode.empty() || mode == "reference")
    {
      Scoped_Timer eig_timer(timers, "min_eigenvalue");
      const El::BigFloat lambda(min_eigenvalue(MInvDM));
      return alpha_from_lambda(lambda, gamma);
    }
  if(mode != "double")
    RUNTIME_ERROR("Unknown SDPB_STEP_LENGTH_MODE: ", mode);

  const double gamma_d = double(gamma);
  const size_t num_blocks = MInvDM.blocks.size();
  std::vector<char> from_double(num_blocks, 0);
  El::BigFloat local_min(El::limits::Max<El::BigFloat>());
  double worst_rel = 0;
  {
    Scoped_Timer eig_timer(timers, "double_min_eigenvalue");
    for(size_t b = 0; b < num_blocks; ++b)
      {
        auto &block = MInvDM.blocks[b];
        if(block.Grid().Size() != 1)
          {
            ++stats.blocks_fallback_dist;
            continue;
          }
        double lambda_scaled = 0, rel = 0;
        long scale_exp = 0;
        const Block_Result result
          = double_min_eigenvalue(block.LockedMatrix(), gamma_d, tol, inject,
                                  lambda_scaled, scale_exp, rel);
        if(result == Block_Result::ok)
          {
            ++stats.blocks_double;
            worst_rel = std::max(worst_rel, rel);
            El::BigFloat lambda(lambda_scaled);
            if(scale_exp >= 0)
              mpf_mul_2exp(lambda.gmp_float.get_mpf_t(),
                           lambda.gmp_float.get_mpf_t(), scale_exp);
            else
              mpf_div_2exp(lambda.gmp_float.get_mpf_t(),
                           lambda.gmp_float.get_mpf_t(), exponent_magnitude(scale_exp));
            local_min = El::Min(local_min, lambda);
            from_double[b] = 1;
          }
        else if(result == Block_Result::nonfinite)
          ++stats.blocks_fallback_nonfinite;
        else if(result == Block_Result::lapack)
          ++stats.blocks_fallback_lapack;
        else
          ++stats.blocks_fallback_gate;
      }
    stats.max_rel_err_estimate
      = std::max(stats.max_rel_err_estimate, worst_rel);
  }
  {
    // Fallback blocks: the BigFloat eigensolve of min_eigenvalue(), block by
    // block (its COMM_WORLD AllReduce is done once below).
    Scoped_Timer eig_timer(timers, "fallback_min_eigenvalue");
    for(size_t b = 0; b < num_blocks; ++b)
      if(!from_double[b])
        {
          auto &blk = MInvDM.blocks[b];
          El::DistMatrix<El::BigFloat, El::VR, El::STAR> eigenvalues(
            blk.Grid());
          El::HermitianEigCtrl<El::BigFloat> ctrl;
          ctrl.tridiagEigCtrl.dcCtrl.cutoff = blk.Height() / 2 + 1;
          ctrl.tridiagEigCtrl.dcCtrl.secularCtrl.maxIterations = 16384;
          El::HermitianEig(El::UpperOrLowerNS::LOWER, blk, eigenvalues, ctrl);
          local_min = El::Min(local_min, El::Min(eigenvalues));
        }
  }
  const El::BigFloat lambda
    = El::mpi::AllReduce(local_min, El::mpi::MIN, El::mpi::COMM_WORLD);
  const El::BigFloat alpha = alpha_from_lambda(lambda, gamma);

  // Numerical working-precision guard for I + alpha A on double-derived
  // blocks. Floating Cholesky success is not an exact PD certificate.
  int local_fail = 0;
  {
    Scoped_Timer guard_timer(timers, "guard_cholesky");
    El::BigFloat guard_alpha(alpha);
    if(inject == "cholesky")
      guard_alpha *= El::BigFloat(1e30);
    for(size_t b = 0; b < num_blocks && !local_fail; ++b)
      if(from_double[b])
        {
          El::Matrix<El::BigFloat> G(MInvDM.blocks[b].LockedMatrix());
          G *= guard_alpha;
          for(int i = 0; i < G.Height(); ++i)
            G(i, i) += El::BigFloat(1);
          try
            {
              El::Cholesky(El::UpperOrLowerNS::LOWER, G);
            }
          catch(std::exception &)
            {
              local_fail = 1;
            }
        }
  }
  const int any_fail
    = El::mpi::AllReduce(local_fail, El::mpi::MAX, El::mpi::COMM_WORLD);

  El::BigFloat result(alpha);
  El::BigFloat reference_lambda(0);
  bool have_reference = false;
  if(any_fail || compare)
    {
      Scoped_Timer fallback_timer(timers, any_fail ? "guard_fallback"
                                                    : "compare_reference");
      Block_Diagonal_Matrix fresh(dM);
      lower_triangular_inverse_congruence(MCholesky, fresh);
      reference_lambda = min_eigenvalue(fresh);
      have_reference = true;
      if(any_fail)
        {
          ++stats.guard_failures;
          result = alpha_from_lambda(reference_lambda, gamma);
        }
    }
  double rel_diff = 0;
  if(have_reference && reference_lambda != El::BigFloat(0))
    {
      rel_diff = double(El::Abs(lambda - reference_lambda)
                        / El::Abs(reference_lambda));
      stats.max_compare_rel_diff
        = std::max(stats.max_compare_rel_diff, rel_diff);
    }
  if(!log_path.empty() && El::mpi::Rank() == 0)
    {
      if(FILE *f = std::fopen(log_path.c_str(), "a"))
        {
          std::fprintf(f,
                       "%s call=%ld lambda=%.17g alpha=%.17g guard_fail=%d "
                       "rel_err_estimate=%.3e",
                       timer_name.c_str(), stats.calls, double(lambda),
                       double(result), any_fail, worst_rel);
          if(have_reference)
            std::fprintf(f, " reference_lambda=%.17g rel_diff=%.3e",
                         double(reference_lambda), rel_diff);
          std::fprintf(f, "\n");
          std::fclose(f);
        }
    }
  return result;
}
