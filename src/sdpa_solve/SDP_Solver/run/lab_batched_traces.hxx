#pragma once

#include "sdpa_solve/SDP.hxx"
#include "sdpb_util/assert.hxx"

#include <cstdlib>
#include <string>
#include <vector>

// lab (bfss #77): opt-in LAB_BATCHED_TRACES=1.
//
// The dual residues and the Schur right-hand side need t_p = Tr(F_p M) for
// all p (M = Y or Z). dotu(F_p, M) does one AllReduce over the block's grid
// per block (inside El::Dotu) and one over COMM_WORLD per p. At level 12
// (m = 8219, 1,280 ranks) each of the three loops per iteration takes ~16 s,
// almost all of it latency (55280437).
//
// lab_traces() computes the same sums with one AllReduce: each rank adds the
// products of its local entries of all its blocks for every p, and one
// AllReduce over COMM_WORLD of m values combines the ranks. Each entry is
// owned by exactly one rank, so each product is counted once. The terms are
// the same, but they are added in another order, so t_p agrees with dotu()
// to rounding at the working precision, not bitwise. Zero entries of F_p
// (most of them) are skipped; adding their zero products would not change
// any partial sum.
namespace Sdpb::Sdpa
{
  inline bool lab_batched_traces()
  {
    static const bool value = [] {
      const char *env = std::getenv("LAB_BATCHED_TRACES");
      return env != nullptr && std::string(env) == "1";
    }();
    return value;
  }

  inline std::vector<El::BigFloat>
  lab_traces(const SDP &sdp, const Block_Diagonal_Matrix &M)
  {
    const size_t m = sdp.sdp_blocks_F.size();
    std::vector<El::BigFloat> traces(m, El::BigFloat(0));
    El::BigFloat product;
    for(size_t p = 0; p < m; ++p)
      {
        const auto &F = sdp.sdp_blocks_F.at(p);
        ASSERT_EQUAL(F.blocks.size(), M.blocks.size());
        auto &t = traces.at(p);
        for(size_t b = 0; b < F.blocks.size(); ++b)
          {
            const auto &F_block = F.blocks.at(b);
            const auto &M_block = M.blocks.at(b);
            // as El::Dotu: the same distribution
            ASSERT(F_block.DistData() == M_block.DistData(),
                   "lab_traces: F_p and M blocks are distributed differently",
                   DEBUG_STRING(p), DEBUG_STRING(b));
            const auto &A = F_block.LockedMatrix();
            const auto &B = M_block.LockedMatrix();
            ASSERT_EQUAL(A.Height(), B.Height());
            ASSERT_EQUAL(A.Width(), B.Width());
            for(int j = 0; j < A.Width(); ++j)
              for(int i = 0; i < A.Height(); ++i)
                {
                  const auto &a = A.CRef(i, j);
                  if(mpf_sgn(a.gmp_float.get_mpf_t()) == 0)
                    continue;
                  product = a;
                  product *= B.CRef(i, j);
                  t += product;
                }
          }
      }
    El::mpi::AllReduce(traces.data(), traces.size(), El::mpi::COMM_WORLD);
    return traces;
  }
}
