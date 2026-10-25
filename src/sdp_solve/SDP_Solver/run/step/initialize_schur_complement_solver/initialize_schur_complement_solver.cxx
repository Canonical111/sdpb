#include "sdp_solve/SDP.hxx"
#include "sdp_solve/Block_Diagonal_Matrix.hxx"
#include "sdpb_util/Timers/Timers.hxx"
#include "sdp_solve/SDP_Solver/run/bigint_syrk/BigInt_Shared_Memory_Syrk_Context.hxx"
#include "sdp_solve/memory_estimates.hxx"

// Compute the quantities needed to solve the Schur complement
// equation
//
// {{S, -B}, {B^T, 0}} . {dx, dy} = {r, s}
//
// (where S = SchurComplement, B = FreeVarMatrix), using the method
// described in the manual:
//
// - Compute S using BilinearPairingsXInv and BilinearPairingsY.
//
// - Compute the Cholesky decomposition S' = L' L'^T.
//
// - Form B' = (B U) and compute
//
//   - SchurOffDiagonal = L'^{-1} B
//   - L'^{-1} U
//   - Q = (L'^{-1} B')^T (L'^{-1} B') - {{0, 0}, {0, 1}}
//
// - Compute the LU decomposition of Q.
//
// This data is sufficient to efficiently solve the above equation for
// a given r,s.
//
// Inputs:
// - BilinearPairingsXInv, BilinearPairingsY (these are members of
//   SDPSolver, but we include them as arguments to emphasize that
//   they must be computed first). They are freed once S is formed.
// Outputs (members of SDPSolver which are modified by this method and
// used later):
// - SchurComplementCholesky (S is computed in place and factorized in place,
//   so S is never stored separately)
// - SchurOffDiagonal
//

void compute_schur_complement(
  const Block_Info &block_info,
  const std::array<
    std::vector<std::vector<std::vector<El::DistMatrix<El::BigFloat>>>>, 2>
    &A_X_inv,
  const std::array<
    std::vector<std::vector<std::vector<El::DistMatrix<El::BigFloat>>>>, 2>
    &A_Y,
  Block_Diagonal_Matrix &schur_complement, Timers &timers);

void compute_Q(const Environment &env, const SDP &sdp,
               const Block_Info &block_info,
               Block_Matrix &schur_off_diagonal,
               Block_Diagonal_Matrix &schur_complement_cholesky,
               BigInt_Shared_Memory_Syrk_Context &bigint_syrk_context,
               El::DistMatrix<El::BigFloat> &Q, Timers &timers,
               El::Matrix<int32_t> &block_timings_ms, Verbosity verbosity);

void initialize_schur_complement_solver(
  const Environment &env, const Block_Info &block_info, const SDP &sdp,
  std::array<
    std::vector<std::vector<std::vector<El::DistMatrix<El::BigFloat>>>>, 2>
    &A_X_inv,
  std::array<
    std::vector<std::vector<std::vector<El::DistMatrix<El::BigFloat>>>>, 2>
    &A_Y,
  const El::Grid &group_grid, Block_Diagonal_Matrix &schur_complement_cholesky,
  Block_Matrix &schur_off_diagonal,
  BigInt_Shared_Memory_Syrk_Context &bigint_syrk_context,
  El::DistMatrix<El::BigFloat> &Q, Timers &timers,
  El::Matrix<int32_t> &block_timings_ms, const Verbosity verbosity)
{
  Scoped_Timer initialize_timer(timers, "initializeSchurComplementSolver");
  // The Schur complement matrix S: a Block_Diagonal_Matrix with one
  // block for each 0 <= j < J.  SchurComplement.blocks[j] has dimension
  // (d_j+1)*m_j*(m_j+1)/2
  //
  // S is written directly into schur_complement_cholesky, which the caller
  // allocates with the same block sizes and grid (S is fully overwritten),
  // and compute_Q() factorizes it in place. This saves one copy of S
  // (sum_j dim(S_j)^2 BigFloats) at the memory peak; results are unchanged.
  compute_schur_complement(block_info, A_X_inv, A_Y, schur_complement_cholesky,
                           timers);

  // A_X_inv and A_Y are not needed after S is formed (compute_bilinear_pairings()
  // rebuilds them from scratch every iteration). Free them before L^{-1} B is
  // allocated, so that they do not add to the memory peak.
  for(auto &parity : A_X_inv)
    parity.clear();
  for(auto &parity : A_Y)
    parity.clear();

  compute_Q(env, sdp, block_info, schur_off_diagonal,
            schur_complement_cholesky, bigint_syrk_context, Q, timers,
            block_timings_ms, verbosity);

  Scoped_Timer Cholesky_timer(timers, "Cholesky_Q");
  try
    {
      Cholesky(El::UpperOrLowerNS::UPPER, Q);
    }
  catch(std::exception &e)
    {
      RUNTIME_ERROR("Error when computing Cholesky(Q): ", e.what());
    }
}
