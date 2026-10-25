#pragma once

#include "SDP.hxx"
#include "SDP_Solver.hxx"
#include "sdp_solve/memory_estimates.hxx"

namespace Sdpb::Sdpa
{
  inline size_t get_S_size_local(const SDP &sdp)
  {
    // #S = NxN, distributed over all nodes.
    return std::ceil(1.0 * sdp.primal_dimension() * sdp.primal_dimension()
                     / El::mpi::Size());
    // TODO: in fact, different ranks can own slightly different number of elements.
    // So this is not a precise estimate.
    // Sum of get_S_size_local() over all ranks will give upper bound to the real #(S).

    // NB: reduce-scatter needs also ~(2 * S_size / split_factor^2 / num_nodes) per node for MPI buffers.
    // We account for it inside --maxSharedMemory limit, see BigInt_Shared_Memory_Syrk_Context() constructor.
  }
  inline size_t get_S_allocated_bytes(const SDP &sdp)
  {
    return sizeof(El::DistMatrix<El::BigFloat>)
           + bigfloat_bytes() * get_S_size_local(sdp);
  }
  inline size_t get_SDP_size_local(const SDP &sdp)
  {
    // #184: F_1..F_m are sparse triplets; only F_0 is dense.
    const size_t limb_bytes = (El::gmp::Precision() + 63) / 64 * 8;
    size_t SDP_size
      = get_matrix_size_local(sdp.sdp_block_F_0)
        + sdp.sparse_F.num_entries()
            * ((sizeof(Sparse_F_Entry) + limb_bytes) / bigfloat_bytes() + 1)
        + sdp.primal_objective_c.AllocatedMemory();
    return SDP_size;
  }
  inline size_t get_heap_allocated_bytes(const SDP &sdp)
  {
    return sdp.sparse_F.heap_bytes()
           + get_heap_allocated_bytes(sdp.sdp_block_F_0)
           + ::get_heap_allocated_bytes(sdp.primal_objective_c);
  }
  inline size_t get_heap_allocated_bytes(const SDP_Solver &solver)
  {
    // TODO get rid of ugly global namespace prefix
    return ::get_heap_allocated_bytes(solver.x)
           + get_heap_allocated_bytes(solver.X)
           + get_heap_allocated_bytes(solver.Y)
           + ::get_heap_allocated_bytes(solver.primal_objective)
           + ::get_heap_allocated_bytes(solver.dual_objective)
           + ::get_heap_allocated_bytes(solver.duality_gap)
           + ::get_heap_allocated_bytes(solver.primal_residues)
           + ::get_heap_allocated_bytes(solver.primal_error)
           + ::get_heap_allocated_bytes(solver.dual_residues)
           + ::get_heap_allocated_bytes(solver.dual_error)
           + ::get_heap_allocated_bytes(solver.R_error);
  }
}
