#include "SDP.hxx"

#include "parse_sdpa/parse_sdpa.hxx"
#include "sdpb_util/assert.hxx"
#include "sdpb_util/copy_matrix.hxx"

#include <algorithm>
#include <filesystem>
#include <tuple>

namespace fs = std::filesystem;

namespace Sdpb::Sdpa
{
  SDP::SDP(const fs::path &sdp_path, const Block_Info &block_info,
           const El::Grid &grid, Timers &timers)
      : sdp_block_F_0(block_info.block_dimensions, block_info.block_indices,
                      grid)
  {
    Scoped_Timer timer(timers, "SDP_ctor");

    const auto m = block_info.primal_dimension;

    const auto &comm = block_info.mpi_comm.value;

    // For each MPI group, rank=0 parses objective_c and the blocks for F_0..F_m that belong to this group.
    // Other ranks do not parse anything.
    // Non-const deliberately (#184): parsed triplet lists are cleared block by block as they
    // are consumed, so the parse copy and the solver copy never coexist in full.
    auto parse_result = [&] {
      if(comm.Rank() == 0)
        {
          Scoped_Timer parse_timer(timers, "parse");
          std::set block_indices_to_parse(block_info.block_indices.begin(),
                                          block_info.block_indices.end());
          const auto should_parse_block = [&](const size_t index) {
            return block_indices_to_parse.find(index)
                   != block_indices_to_parse.end();
          };
          return read_sdpa(sdp_path, should_parse_block);
        }
      return SDPA_File_Parse_Result();
    }();

    {
      Scoped_Timer sync_timer(timers, "synchronize");

      // primal_objective_c
      {
        // All elements are stored at rank=0
        auto c_circ_circ
          = El::DistMatrix<El::BigFloat, El::CIRC, El::CIRC>(m, 1);
        if(El::mpi::Rank() == 0)
          {
            for(size_t i = 0; i < m; ++i)
              c_circ_circ.Set(i, 0, parse_result.c_objective.at(i));
          }
        // Copy all elements from rank=0 to all ranks.
        primal_objective_c = c_circ_circ;
      }

      // Prepare F_0 (dense, one matrix) and the sparse F_1..F_m triplet lists (#184).
      {
        sdp_block_F_0 = Block_Diagonal_Matrix(block_info.block_dimensions,
                                              block_info.block_indices, grid);
        sparse_F.primal_dimension = m;
        sparse_F.entries.resize(block_info.num_blocks_local());
        for(auto &per_block : sparse_F.entries)
          per_block.resize(m);
      }

      // The value table: group rank 0 parsed it; every rank of the group keeps a copy
      // (it is small -- 106k values on 12_min -- and entries index it directly).
      {
        long long num_values
          = comm.Rank() == 0
              ? static_cast<long long>(parse_result.F_values.size())
              : 0;
        El::mpi::Broadcast(num_values, 0, comm);
        if(comm.Rank() == 0)
          sparse_F.values = std::move(parse_result.F_values);
        else
          sparse_F.values.resize(num_values);
        if(num_values > 0 && comm.Size() > 1)
          El::mpi::Broadcast(sparse_F.values.data(), num_values, 0, comm);
      }

      // Synchronize the triplets: group rank 0 parsed them; broadcast per local block.
      // Every rank of the block's grid keeps the full (small) list; dense operands stay
      // distributed and consumers only touch locally-owned elements.
      for(size_t local_block_index = 0;
          local_block_index < block_info.num_blocks_local();
          ++local_block_index)
        {
          const auto block_index
            = block_info.block_indices.at(local_block_index);

          // Sort rank-0's entries into (constraint, col, row) order. Column-major per
          // constraint is the order the dense loops visited elements in, which keeps
          // single-grid BigFloat accumulations bit-identical to the dense implementation.
          // A duplicate (constraint,row,col) in the input overwrote under the dense reader;
          // keep the last occurrence to preserve that behaviour.
          std::vector<Parsed_F_Entry> sorted_entries;
          if(comm.Rank() == 0)
            {
              auto &parsed
                = parse_result.parsed_block_entries.at(block_index);
              std::stable_sort(parsed.begin(), parsed.end(),
                               [](const auto &a, const auto &b) {
                                 return std::tie(a.constraint, a.col, a.row)
                                        < std::tie(b.constraint, b.col, b.row);
                               });
              for(const auto &e : parsed)
                {
                  if(!sorted_entries.empty()
                     && sorted_entries.back().constraint == e.constraint
                     && sorted_entries.back().row == e.row
                     && sorted_entries.back().col == e.col)
                    sorted_entries.back().value_index = e.value_index;
                  else
                    sorted_entries.push_back(e);
                }
              parsed.clear();
              parsed.shrink_to_fit();
            }

          // Broadcast count and the index arrays over the group communicator.
          long long count = static_cast<long long>(sorted_entries.size());
          El::mpi::Broadcast(count, 0, comm);
          std::vector<El::Int> constraints(count), rows(count), cols(count),
            slots(count);
          if(comm.Rank() == 0)
            for(long long k = 0; k < count; ++k)
              {
                const auto &e = sorted_entries.at(k);
                constraints.at(k) = static_cast<El::Int>(e.constraint);
                rows.at(k) = e.row;
                cols.at(k) = e.col;
                slots.at(k) = static_cast<El::Int>(e.value_index);
              }
          sorted_entries.clear();
          sorted_entries.shrink_to_fit();
          if(count > 0 && comm.Size() > 1)
            {
              El::mpi::Broadcast(constraints.data(), count, 0, comm);
              El::mpi::Broadcast(rows.data(), count, 0, comm);
              El::mpi::Broadcast(cols.data(), count, 0, comm);
              El::mpi::Broadcast(slots.data(), count, 0, comm);
            }

          // F_0 is densified; F_1..m stay triplets.
          auto &F_0_block = sdp_block_F_0.blocks.at(local_block_index);
          El::Zero(F_0_block);
          auto &per_block = sparse_F.entries.at(local_block_index);
          for(long long k = 0; k < count; ++k)
            {
              const auto constraint = static_cast<size_t>(constraints.at(k));
              if(constraint == 0)
                {
                  if(F_0_block.IsLocal(rows.at(k), cols.at(k)))
                    F_0_block.SetLocal(F_0_block.LocalRow(rows.at(k)),
                                       F_0_block.LocalCol(cols.at(k)),
                                       sparse_F.values.at(slots.at(k)));
                }
              else
                {
                  per_block.at(constraint - 1)
                    .push_back({static_cast<int32_t>(rows.at(k)),
                                static_cast<int32_t>(cols.at(k)),
                                static_cast<uint32_t>(slots.at(k))});
                }
            }
          for(auto &per_i : per_block)
            per_i.shrink_to_fit();
        }
    }

    Scoped_Timer validate_timer(timers, "validate");
    validate(block_info);
  }

  size_t SDP::primal_dimension() const
  {
    return sparse_F.primal_dimension;
  }

  void SDP::validate(const Block_Info &block_info) const noexcept(false)
  {
    const auto error_prefix = "Invalid SDP: ";

    // Check primal_objective_c
    {
      ASSERT_EQUAL(static_cast<size_t>(primal_objective_c.Height()),
                   sparse_F.primal_dimension, error_prefix,
                   "c_1..c_m and F_1..Fm should have the same length");
      ASSERT_EQUAL(primal_objective_c.Width(), 1, error_prefix,
                   "c should be a vector");
    }

    for(size_t local_block_index = 0;
        local_block_index < block_info.num_blocks_local(); ++local_block_index)
      {
        const auto block_index
          = block_info.block_indices.at(local_block_index);

        const auto error_prefix_index
          = El::BuildString(error_prefix, "block_index = ", block_index, ": ");

        const auto &comm = block_info.mpi_comm.value;
        const auto dim = block_info.block_dimensions.at(block_index);

        {
          const auto &block = sdp_block_F_0.blocks.at(local_block_index);
          ASSERT(El::mpi::Congruent(block.DistComm(), comm),
                 error_prefix_index, "wrong MPI communicator for F_0");
          ASSERT_EQUAL(block.Height(), dim, error_prefix_index);
          ASSERT_EQUAL(block.Width(), dim, error_prefix_index);
        }

        ASSERT_EQUAL(sparse_F.entries.at(local_block_index).size(),
                     sparse_F.primal_dimension, error_prefix_index);
        for(const auto &per_i : sparse_F.entries.at(local_block_index))
          for(const auto &e : per_i)
            {
              ASSERT(e.row >= 0 && e.col >= 0 && e.row < El::Int(dim)
                       && e.col < El::Int(dim),
                     error_prefix_index, "F entry out of range: (", e.row,
                     ",", e.col, "), dim=", dim);
              ASSERT(e.value_index < sparse_F.values.size(),
                     error_prefix_index, "F value index out of range");
            }
      }
  }
}
