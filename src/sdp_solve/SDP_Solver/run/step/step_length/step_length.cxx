#include "sdp_solve/SDP_Solver.hxx"
#include "selected_step.hxx"

// min(gamma \alpha(M, dM), 1), where \alpha(M, dM) denotes the
// largest positive real number such that M + \alpha dM is positive
// semidefinite.
//
// \alpha(M, dM) is computed with a Cholesky decomposition M = L L^T.
// M + \alpha dM is congruent to I + \alpha L^{-1} dM L^{-T}.
// For exact arithmetic the boundary \alpha is then -1/lambda,
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

// A := L^{-1} A L^{-T}
void lower_triangular_inverse_congruence(const Block_Diagonal_Matrix &L,
                                         Block_Diagonal_Matrix &A);

El::BigFloat min_eigenvalue(Block_Diagonal_Matrix &A);

El::BigFloat step_length(const Block_Diagonal_Matrix &MCholesky,
                         const Block_Diagonal_Matrix &dM,
                         const El::BigFloat &gamma,
                         const std::string &timer_name,
                         Timers &timers)
{
  Scoped_Timer step_length_timer(timers, timer_name);
  // MInvDM = L^{-1} dM L^{-T}, where M = L L^T
  Block_Diagonal_Matrix MInvDM(dM);
  const double begin=selected::seconds();
  lower_triangular_inverse_congruence(MCholesky, MInvDM);
  const double congruence=selected::seconds()-begin;
  const char* backend=std::getenv("SDPB_SELECTED_STEP");
  if(backend && std::string(backend)=="1") {
    auto outcome=selected::attempt(MInvDM,gamma);
    double fallback=0;
    El::BigFloat alpha=outcome.alpha;
    if(outcome.status!=selected::certified) {
      const double start=selected::seconds();
      const auto lambda=min_eigenvalue(MInvDM);
      alpha=lambda > -gamma ? El::BigFloat(1) : El::BigFloat(-gamma/lambda);
      fallback=selected::seconds()-start;
    }
    double costs[4]={congruence,outcome.reduction,outcome.kernel,fallback};
    for(auto& cost:costs) cost=El::mpi::AllReduce(cost,El::mpi::MAX,El::mpi::COMM_WORLD);
    if(El::mpi::Rank()==0)
      std::cerr << "SELECTED_STEP " << outcome.status << " " << costs[0] << " "
                << costs[1] << " " << costs[2] << " " << costs[3] << std::endl;
    return alpha;
  }
  const El::BigFloat lambda(min_eigenvalue(MInvDM));
  if(lambda > -gamma)
    {
      return 1;
    }
  else
    {
      return -gamma / lambda;
    }
}
