#include "sdp_solve/SDP_Solver/run/constraint_matrix_weighted_sum.hxx"
#include "sdpb_util/Timers/Timers.hxx"

// Compute the search direction (dx, dX, dy, dY) for the predictor and
// corrector phases.
//
// Inputs:
// - beta, the centering parameter
// - mu = Tr(X Y) / X.cols
// - correctorPhase: boolean indicating whether we're in the corrector
//   phase or predictor phase.
// Workspace (members of SDPSolver which are modified in-place but not
// used elsewhere):
// - Z, R
// Outputs (members of SDPSolver which are modified in-place):
// - dx, dX, dy, dY
//

// C := alpha*A*B + beta*C
void scale_multiply_add(const El::BigFloat &alpha,
                        const Block_Diagonal_Matrix &A,
                        const Block_Diagonal_Matrix &B,
                        const El::BigFloat &beta, Block_Diagonal_Matrix &C);

// C := A B
inline void multiply(const Block_Diagonal_Matrix &A,
                     const Block_Diagonal_Matrix &B, Block_Diagonal_Matrix &C)
{
  scale_multiply_add(El::BigFloat(1), A, B, El::BigFloat(0), C);
}

// X := ACholesky^{-T} ACholesky^{-1} X = A^{-1} X
void cholesky_solve(const Block_Diagonal_Matrix &ACholesky,
                    Block_Diagonal_Matrix &X);

void compute_schur_RHS(const Block_Info &block_info, const SDP &sdp,
                       const Block_Vector &dual_residues,
                       const Block_Diagonal_Matrix &Z, Block_Vector &dx);

void solve_schur_complement_equation_timed(
  const Block_Diagonal_Matrix &schur_complement_cholesky,
  const Block_Matrix &schur_off_diagonal,
  const El::DistMatrix<El::BigFloat> &Q, Block_Vector &dx, Block_Vector &dy,
  Timers &timers);

void compute_search_direction(
  const Block_Info &block_info, const SDP &sdp, const SDP_Solver &solver,
  const Block_Diagonal_Matrix &minus_XY,
  const Block_Diagonal_Matrix &schur_complement_cholesky,
  const Block_Matrix &schur_off_diagonal,
  const Block_Diagonal_Matrix &X_cholesky, const El::BigFloat &beta,
  const El::BigFloat &mu, const Block_Vector &primal_residue_p,
  const bool &is_corrector_phase, const El::DistMatrix<El::BigFloat> &Q,
  Block_Vector &dx, Block_Diagonal_Matrix &dX, Block_Vector &dy,
  Block_Diagonal_Matrix &dY, Timers &timers)
{
  // R = beta mu I - X Y (predictor phase)
  // R = beta mu I - X Y - dX dY (corrector phase)
  Block_Diagonal_Matrix R(minus_XY);
  Block_Diagonal_Matrix Z(solver.X);
  {
    Scoped_Timer timer(timers, "R_Z");
    if(is_corrector_phase)
      {
        scale_multiply_add(El::BigFloat(-1), dX, dY, El::BigFloat(1), R);
      }
    R.add_diagonal(beta * mu);

    multiply(solver.primal_residues, solver.Y, Z);
    Z -= R;
    cholesky_solve(X_cholesky, Z);
    Z.symmetrize();
  }

  {
    Scoped_Timer timer(timers, "schur_RHS");
    compute_schur_RHS(block_info, sdp, solver.dual_residues, Z, dx);
    dy = primal_residue_p;
  }

  {
    Scoped_Timer timer(timers, "solve_schur");
    solve_schur_complement_equation_timed(schur_complement_cholesky,
                                          schur_off_diagonal, Q, dx, dy,
                                          timers);
  }

  {
    Scoped_Timer timer(timers, "dX");
    constraint_matrix_weighted_sum(block_info, sdp, dx, dX);
    dX += solver.primal_residues;
  }

  {
    Scoped_Timer timer(timers, "dY");
    multiply(dX, solver.Y, dY);
    dY -= R;
    cholesky_solve(X_cholesky, dY);
    dY.symmetrize();
    dY *= El::BigFloat(-1);
  }
}
