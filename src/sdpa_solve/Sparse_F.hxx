#pragma once
// #184 (bootstrap-speed): sparse storage for the SDPA constraint matrices F_1..F_m.
//
// The .dat-s inputs this solver exists for are sparse: on the 12_min benchmark the F_i carry
// ~535k nonzeros against (m+1)*sum(dim^2) = 6.5M dense slots (8.2% density), and dense storage
// of every F_i was ~700 MB of the measured ~8 GB peak -- before counting the parser's transient
// dense copy of the same data. Triplet storage keeps the same results (consumers visit the same
// nonzero elements; zeros never contributed to any sum) at a small fraction of the memory, and
// turns every F-consumer loop into O(nnz) instead of O(dense).
//
// Layout contract:
//  * entries[local_block][i] holds the nonzeros of F_{i+1} restricted to that block, with BOTH
//    symmetric halves stored explicitly and the list sorted by (col, row) -- Elemental's
//    column-major element order. Storing both halves costs <2x on a tiny object and buys two
//    things: consumers are straight linear loops with no mirror logic, and BigFloat accumulation
//    visits nonzeros in the same order as the dense loops it replaces, so single-grid runs
//    reproduce the dense path bit for bit (MPFR addition of an exact zero is the identity).
//  * Every rank of a block's grid holds the block's full entry list (the lists are small);
//    consumers touch only locally-owned elements of the distributed operands, so each
//    (block, element) pair is counted exactly once across the world. Reductions are the
//    caller's job: disjoint local partial sums, one AllReduce.

#include "sdpb_util/assert.hxx"

#include <El.hpp>

#include <vector>

namespace Sdpb::Sdpa
{
  struct Sparse_F_Entry
  {
    El::Int row;
    El::Int col;
    El::BigFloat value;
  };

  struct Sparse_F_Matrices
  {
    size_t primal_dimension = 0;
    // [local_block][constraint 0..m-1] -> entries of F_{constraint+1}
    std::vector<std::vector<std::vector<Sparse_F_Entry>>> entries;

    size_t num_entries() const
    {
      size_t n = 0;
      for(const auto &per_block : entries)
        for(const auto &per_i : per_block)
          n += per_i.size();
      return n;
    }

    // Heap footprint estimate: the entry structs plus each BigFloat's limb allocation.
    size_t heap_bytes() const
    {
      const size_t limb_bytes = (El::gmp::Precision() + 63) / 64 * 8;
      return num_entries() * (sizeof(Sparse_F_Entry) + limb_bytes);
    }
  };

  // local_sum += Tr(F_i B) restricted to the elements of B owned by this rank, for one block.
  // B must be stored full (both halves), as X and Y are here.
  inline void
  sparse_dotu_local_update(const std::vector<Sparse_F_Entry> &entries,
                           const El::DistMatrix<El::BigFloat> &B,
                           El::BigFloat &local_sum)
  {
    for(const auto &e : entries)
      {
        if(B.IsLocal(e.row, e.col))
          local_sum
            += e.value * B.GetLocal(B.LocalRow(e.row), B.LocalCol(e.col));
      }
  }

  // result += a * F_i for one block, locally-owned elements only.
  //
  // NB: the product is written F * a, not a * F. GMP mpf results depend on the
  // destination operand's limb state (mpf_mul truncates by the FIRST operand's
  // storage, and El::BigFloat's operator* copies its left argument as the
  // destination), so only F.value * a reproduces the dense path's
  // El::Scale(a, F_copy) bit for bit. El::Dotu's A(i,j)*B(i,j) pattern is
  // matched the same way in sparse_dotu_local_update above.
  inline void sparse_axpy_local(const El::BigFloat &a,
                                const std::vector<Sparse_F_Entry> &entries,
                                El::DistMatrix<El::BigFloat> &result)
  {
    const El::BigFloat one(1);
    for(const auto &e : entries)
      {
        if(result.IsLocal(e.row, e.col))
          // one * (...) reproduces El::Axpy's 1*x elementwise stage, whose GMP
          // destination state rounds the contribution exactly as the dense
          // path's += did.
          result.UpdateLocal(result.LocalRow(e.row), result.LocalCol(e.col),
                             one * (e.value * a));
      }
  }

  // dest := F_i for one block: zero, then scatter, locally-owned elements only.
  inline void sparse_densify_local(const std::vector<Sparse_F_Entry> &entries,
                                   El::DistMatrix<El::BigFloat> &dest)
  {
    El::Zero(dest);
    for(const auto &e : entries)
      {
        if(dest.IsLocal(e.row, e.col))
          dest.SetLocal(dest.LocalRow(e.row), dest.LocalCol(e.col), e.value);
      }
  }
}
