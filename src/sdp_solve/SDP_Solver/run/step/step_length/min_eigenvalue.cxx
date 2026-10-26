#include "sdp_solve/Block_Diagonal_Matrix.hxx"

// Minimum eigenvalue of A.  A is assumed to be symmetric.

// Annoyingly, El::HermitianEig modifies 'block'.  It is OK, because
// it is only called from step_length(), which passes in a temporary.
// Still ugly.
El::BigFloat min_eigenvalue(Block_Diagonal_Matrix &A)
{
  El::BigFloat local_min(El::limits::Max<El::BigFloat>());

  for(auto &block : A.blocks)
    {
      El::DistMatrix<El::BigFloat, El::VR, El::STAR> eigenvalues(block.Grid());
      /// Eigenvalues of the tridiagonal matrix by the implicit QR
      /// algorithm. The default divide-and-conquer algorithm needed
      /// workarounds (a cutoff allowing one level of recursion, many
      /// secular-equation iterations) and still divided by zero in its
      /// merge step for some blocks: GMP then raises SIGFPE and SDPB
      /// aborts. This happened in a few iterations of degree-140 and
      /// degree-200 polynomial blocks on an interval (bootstrap-speed
      /// #185). Only eigenvalues are needed here, which QR computes in
      /// O(n^2) operations after the tridiagonal reduction.
      El::HermitianEigCtrl<El::BigFloat> hermitian_eig_ctrl;
      hermitian_eig_ctrl.tridiagEigCtrl.alg = El::HERM_TRIDIAG_EIG_QR;
      El::HermitianEig(El::UpperOrLowerNS::LOWER, block, eigenvalues,
                       hermitian_eig_ctrl);
      local_min = El::Min(local_min, El::Min(eigenvalues));
    }
  return El::mpi::AllReduce(local_min, El::mpi::MIN, El::mpi::COMM_WORLD);
}
