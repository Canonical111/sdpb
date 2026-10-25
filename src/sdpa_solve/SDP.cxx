#include "SDP.hxx"

#include "parse_sdpa/parse_sdpa.hxx"
#include "sdpb_util/assert.hxx"
#include "sdpb_util/copy_matrix.hxx"
#include "sdpb_util/Rss_Log.hxx"

#include <climits>
#include <cstring>
#include <filesystem>
#include <map>

#if defined(__GLIBC__)
#include <malloc.h>
#endif

namespace fs = std::filesystem;

namespace Sdpb::Sdpa
{
  namespace
  {
    // lab (bfss #61), LAB_SDPA_INDEXED_READER=2: fill this group's F_0..F_m
    // blocks from the file without a root. The group's runs, concatenated in
    // file order, are cut into R * G slices (G = group size); in round r,
    // rank q parses slice r * G + q and sends each entry (and its mirror) to
    // the rank owning it. Receivers apply rounds in order and, within a
    // round, senders in rank order, i.e. entries in file order, as the
    // deployed reader does.
    void fill_F_distributed(const fs::path &sdp_path,
                            const SDPA_Sparse_Index &sparse_index,
                            const Block_Info &block_info,
                            Block_Diagonal_Matrix &F_0,
                            std::vector<Block_Diagonal_Matrix> &F,
                            const El::mpi::Comm &comm, Timers &timers)
    {
      const int group_size = comm.Size();
      const int group_rank = comm.Rank();
      const size_t m = F.size();

      std::map<size_t, size_t> local_index;
      for(size_t k = 0; k < block_info.block_indices.size(); ++k)
        local_index.emplace(block_info.block_indices.at(k), k);

      // Our owner computation below assumes DistComm rank == comm rank.
      for(size_t k = 0; k < F_0.blocks.size(); ++k)
        {
          const auto &block = F_0.blocks.at(k);
          ASSERT(El::mpi::Congruent(block.DistComm(), comm));
          for(int iLoc = 0; iLoc < block.LocalHeight(); ++iLoc)
            for(int jLoc = 0; jLoc < block.LocalWidth(); ++jLoc)
              ASSERT_EQUAL(block.Owner(block.GlobalRow(iLoc),
                                       block.GlobalCol(jLoc)),
                           group_rank, "unexpected DistMatrix owner");
        }

      std::vector<SDPA_Block_Run> runs;
      size_t total = 0;
      for(const auto &run : sparse_index.runs)
        if(local_index.count(run.block_index) != 0)
          {
            runs.push_back(run);
            total += run.end - run.begin;
          }

      constexpr size_t slice_bytes = 32 * 1024 * 1024;
      const size_t num_rounds = std::max<size_t>(
        1, (total + group_size * slice_bytes - 1) / (group_size * slice_bytes));
      const size_t num_slices = num_rounds * group_size;

      const size_t value_size = El::BigFloat().SerializedSize();
      constexpr size_t header_size = 4 * sizeof(uint32_t);
      const size_t record_size = header_size + value_size;
      ASSERT(m < UINT32_MAX);

      std::vector<std::vector<El::byte>> send(group_size);
      std::vector<El::byte> send_buffer, recv_buffer;
      std::vector<int> send_counts(group_size), recv_counts(group_size),
        send_displacements(group_size), recv_displacements(group_size);

      const auto add_record = [&](const int to, const size_t matrix_index,
                                  const size_t local_block, const size_t i,
                                  const size_t j, const El::BigFloat &value) {
        auto &buffer = send.at(to);
        const size_t offset = buffer.size();
        buffer.resize(offset + record_size);
        const uint32_t header[4]
          = {static_cast<uint32_t>(matrix_index),
             static_cast<uint32_t>(local_block), static_cast<uint32_t>(i),
             static_cast<uint32_t>(j)};
        std::memcpy(buffer.data() + offset, header, header_size);
        value.Serialize(buffer.data() + offset + header_size);
      };
      const auto accept = [&](const size_t block_index,
                              const size_t matrix_index, const size_t i,
                              const size_t j, const El::BigFloat &value) {
        ASSERT(matrix_index <= m);
        const size_t local_block = local_index.at(block_index);
        const auto &block = F_0.blocks.at(local_block);
        add_record(block.Owner(i, j), matrix_index, local_block, i, j, value);
        if(i != j)
          add_record(block.Owner(j, i), matrix_index, local_block, j, i,
                     value);
      };

      El::BigFloat value;
      for(size_t round = 0; round < num_rounds; ++round)
        {
          {
            Scoped_Timer parse_timer(timers, "parse");
            // Our slice [a, b) of the concatenated runs
            const size_t slice = round * group_size + group_rank;
            const size_t a = total / num_slices * slice
                             + std::min(slice, total % num_slices);
            const size_t b = total / num_slices * (slice + 1)
                             + std::min(slice + 1, total % num_slices);
            size_t offset = 0;
            for(const auto &run : runs)
              {
                const size_t length = run.end - run.begin;
                const size_t lo = std::max(a, offset);
                const size_t hi = std::min(b, offset + length);
                if(lo < hi)
                  read_sdpa_lines(sdp_path, sparse_index, run.begin, run.end,
                                  run.begin + (lo - offset),
                                  run.begin + (hi - offset), accept);
                offset += length;
              }
          }
          {
            Scoped_Timer exchange_timer(timers, "exchange");
            size_t send_total = 0;
            for(int r = 0; r < group_size; ++r)
              {
                ASSERT(send.at(r).size() < static_cast<size_t>(INT_MAX));
                send_counts.at(r) = send.at(r).size();
                send_displacements.at(r) = send_total;
                send_total += send.at(r).size();
                ASSERT(send_total < static_cast<size_t>(INT_MAX));
              }
            send_buffer.resize(send_total);
            for(int r = 0; r < group_size; ++r)
              {
                if(!send.at(r).empty())
                  std::memcpy(send_buffer.data() + send_displacements.at(r),
                              send.at(r).data(), send.at(r).size());
                send.at(r).clear();
                send.at(r).shrink_to_fit();
              }
            MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(),
                         1, MPI_INT, comm.comm);
            size_t recv_total = 0;
            for(int r = 0; r < group_size; ++r)
              {
                recv_displacements.at(r) = recv_total;
                recv_total += recv_counts.at(r);
                ASSERT(recv_total < static_cast<size_t>(INT_MAX));
              }
            recv_buffer.resize(recv_total);
            MPI_Alltoallv(send_buffer.data(), send_counts.data(),
                          send_displacements.data(), MPI_BYTE,
                          recv_buffer.data(), recv_counts.data(),
                          recv_displacements.data(), MPI_BYTE, comm.comm);
          }
          {
            Scoped_Timer apply_timer(timers, "apply");
            // Senders in rank order: slices in file order.
            ASSERT_EQUAL(recv_buffer.size() % record_size, 0);
            for(size_t offset = 0; offset < recv_buffer.size();
                offset += record_size)
              {
                uint32_t header[4];
                std::memcpy(header, recv_buffer.data() + offset, header_size);
                value.Deserialize(recv_buffer.data() + offset + header_size);
                const size_t matrix_index = header[0];
                ASSERT(matrix_index <= m);
                auto &matrix = matrix_index == 0 ? F_0 : F.at(matrix_index - 1);
                auto &block = matrix.blocks.at(header[1]);
                const El::Int i = header[2], j = header[3];
                ASSERT(block.IsLocal(i, j));
                block.SetLocal(block.LocalRow(i), block.LocalCol(j), value);
              }
          }
        }
    }
  }

  SDP::SDP(const fs::path &sdp_path, const Block_Info &block_info,
           const El::Grid &grid, Timers &timers)
      : sdp_block_F_0(block_info.block_dimensions, block_info.block_indices,
                      grid)
  {
    Scoped_Timer timer(timers, "SDP_ctor");

    const auto m = block_info.primal_dimension;

    const auto &comm = block_info.mpi_comm.value;

    // lab (bfss #61): with LAB_SDPA_INDEXED_READER=1 (plain .dat-s only),
    // all ranks first index the file's block runs, so that each group root
    // reads only its own blocks' lines instead of the whole file.
    const int reader_mode = sdpa_indexed_reader_mode(sdp_path);
    const bool indexed = reader_mode != 0;
    SDPA_Sparse_Index sparse_index;
    if(indexed)
      {
        Scoped_Timer index_timer(timers, "index");
        sparse_index = build_sdpa_sparse_index(sdp_path, El::mpi::COMM_WORLD);
        if(El::mpi::Rank() == 0)
          El::Output("Indexed SDPA reader (mode ", reader_mode, "): ",
                     sparse_index.runs.size(),
                     " block runs, entries at bytes ",
                     sparse_index.data_begin, "..", sparse_index.file_size);
      }

    // For each MPI group, rank=0 parses objective_c and the blocks for F_0..F_m that belong to this group.
    // Other ranks do not parse anything.
    auto parse_result = [&] {
      if(reader_mode == 2)
        {
          // Only the objective, on rank 0; F is filled in synchronize.
          SDPA_File_Parse_Result result;
          if(El::mpi::Rank() == 0)
            result.c_objective = read_sdpa_objective(sdp_path);
          return result;
        }
      if(comm.Rank() == 0)
        {
          Scoped_Timer parse_timer(timers, "parse");
          std::set block_indices_to_parse(block_info.block_indices.begin(),
                                          block_info.block_indices.end());
          const auto should_parse_block = [&](const size_t index) {
            return block_indices_to_parse.find(index)
                   != block_indices_to_parse.end();
          };
          if(indexed)
            return read_sdpa_indexed(sdp_path, sparse_index,
                                     should_parse_block);
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

      // Prepare F_0 and F_i structure
      {
        sdp_block_F_0 = Block_Diagonal_Matrix(block_info.block_dimensions,
                                              block_info.block_indices, grid);
        sdp_blocks_F.reserve(m);
        for(size_t i = 0; i < m; ++i)
          {
            sdp_blocks_F.emplace_back(block_info.block_dimensions,
                                      block_info.block_indices, grid);
          }
      }
      rss_log("mark", "SDP_ctor.F_allocated");

      if(reader_mode == 2)
        fill_F_distributed(sdp_path, sparse_index, block_info, sdp_block_F_0,
                           sdp_blocks_F, comm, timers);
      else
        {
          // Synchronize F_0 and F_i blocks
          for(size_t local_block_index = 0;
              local_block_index < block_info.num_blocks_local();
              ++local_block_index)
            {
              const auto block_index
                = block_info.block_indices.at(local_block_index);
              // Run over F_0, F_1..F_m
              // NB: sdp_blocks_F[i] corresponds to parsed_block[i+1]
              // since F_0 is not in sdp_blocks_F
              for(size_t i = 0; i <= m; ++i)
                {
                  const auto &source
                    = comm.Rank() == 0
                        ? parse_result.parsed_blocks.at(block_index).at(i)
                        : El::Matrix<El::BigFloat>();
                  auto &F = i == 0 ? sdp_block_F_0 : sdp_blocks_F.at(i - 1);
                  auto &dest = F.blocks.at(local_block_index);
                  copy_matrix_from_root(source, dest, comm);
                }
            }
        }
    }

    // lab (bfss #44): release the root's parsed copy now and return its pages
    // to the OS; otherwise the freed blocks sit below the distributed F in the
    // heap and stay resident for the whole solve.
    parse_result = SDPA_File_Parse_Result();
#if defined(__GLIBC__)
    ::malloc_trim(0);
#endif
    rss_log("mark", "SDP_ctor.after_parse_release_trim");

    Scoped_Timer validate_timer(timers, "validate");
    validate(block_info);
    validate_timer.stop();
    rss_log("mark", "SDP_ctor.before_parse_result_release");
  }

  size_t SDP::primal_dimension() const
  {
    return sdp_blocks_F.size();
  }

  void SDP::validate(const Block_Info &block_info) const noexcept(false)
  {
    const auto error_prefix = "Invalid SDP: ";

    // Check primal_objective_c
    {
      ASSERT_EQUAL(primal_objective_c.Height(), this->sdp_blocks_F.size(),
                   error_prefix,
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

        for(const auto &F : sdp_blocks_F)
          {
            const auto &block = F.blocks.at(local_block_index);
            ASSERT(El::mpi::Congruent(block.DistComm(), comm),
                   error_prefix_index,
                   "wrong MPI communicator for sdp_blocks_F");
            ASSERT_EQUAL(block.Height(), dim, error_prefix_index);
            ASSERT_EQUAL(block.Width(), dim, error_prefix_index);
          }
      }
  }
}
