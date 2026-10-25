#include "sdpa_solve/SDP_Solver.hxx"

namespace Sdpb::Sdpa
{
  // dual_residues[p] = c[p] - F[p,a,b] Y[a,b]
  //
  // #184 sparse path: each rank accumulates Tr(F_p Y) over the Y elements it owns, for all p;
  // ownership is disjoint across the world, so a single AllReduce of the length-m vector
  // replaces m per-constraint reductions.
  void
  compute_dual_residues_and_error(const SDP &sdp,
                                  const Block_Diagonal_Matrix &Y,
                                  Primal_Dist_Vector &dual_residues,
                                  El::BigFloat &dual_error, Timers &timers)
  {
    Scoped_Timer dual_residues_timer(timers, "computeDualResidues");
    const size_t m = sdp.primal_dimension();
    std::vector<El::BigFloat> traces(m, El::BigFloat(0));
    for(size_t local_block = 0; local_block < sdp.sparse_F.entries.size();
        ++local_block)
      {
        const auto &Y_block = Y.blocks.at(local_block);
        const auto &per_block = sdp.sparse_F.entries.at(local_block);
        for(size_t i = 0; i < m; ++i)
          {
            sparse_dotu_local_update(per_block.at(i), Y_block, traces.at(i));
          }
      }
    El::mpi::AllReduce(traces.data(), m, El::mpi::SUM, El::mpi::COMM_WORLD);

    dual_residues = sdp.primal_objective_c;
    for(size_t i = 0; i < m; ++i)
      {
        const auto d = dual_residues.Get(i, 0);
        dual_residues.Set(i, 0, d - traces.at(i));
      }
    dual_error = El::MaxAbs(dual_residues);
  }
}
