#pragma once

#include "sdpa_solve/SDP.hxx"

namespace Sdpb::Sdpa
{
  // result = \sum_p a[p] F_p
  //
  // #184 sparse path. The dense version copied each dense F block and scaled the copy --
  // a full dense temporary per (constraint, block). Sparse scatter-add touches each stored
  // nonzero once; per element, constraints still accumulate in ascending p, matching the
  // dense loop's addition order.
  inline void
  constraint_matrix_weighted_sum(const SDP &sdp, const Primal_Dist_Vector &a,
                                 Block_Diagonal_Matrix &result)
  {
    ASSERT_EQUAL(static_cast<size_t>(a.Height()), sdp.primal_dimension());
    ASSERT_EQUAL(a.Width(), 1);
    result.set_zero();
    const size_t m = sdp.primal_dimension();
    for(size_t local_block = 0; local_block < sdp.sparse_F.entries.size();
        ++local_block)
      {
        auto &result_block = result.blocks.at(local_block);
        const auto &per_block = sdp.sparse_F.entries.at(local_block);
        for(size_t i = 0; i < m; ++i)
          {
            sparse_axpy_local(a.Get(i, 0), per_block.at(i), sdp.sparse_F.values,
                              result_block);
          }
      }
  }
}
