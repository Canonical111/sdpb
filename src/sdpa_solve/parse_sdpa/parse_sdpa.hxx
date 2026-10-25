#pragma once

#include "SDPA_Block_Structure.hxx"
#include "SDPA_File_Parse_Result.hxx"

#include <filesystem>
#include <functional>

namespace Sdpb::Sdpa
{
SDPA_Block_Structure
read_block_structure(const std::filesystem::path &input_file);

SDPA_File_Parse_Result
read_sdpa(const std::filesystem::path &input_file,
          const std::function<bool(size_t matrix_index)> &should_parse_block);

// lab (bfss #61): indexed reader for plain .dat-s files.
// A run is a maximal byte range [begin, end) of consecutive entry lines
// that all belong to the same solver block.
struct SDPA_Block_Run
{
  size_t block_index;
  size_t begin;
  size_t end;
};
struct SDPA_Sparse_Index
{
  // Offset of the first entry line (after the objective line)
  size_t data_begin = 0;
  size_t file_size = 0;
  // All runs of the file, in file order
  std::vector<SDPA_Block_Run> runs;
  // Header (m, SDPA block sizes; negative for diagonal blocks)
  size_t m_dim = 0;
  std::vector<int> input_block_sizes;
};

// LAB_SDPA_INDEXED_READER=1 or 2 for a plain .dat-s file; 0 otherwise.
// 1: group roots read their own runs (read_sdpa_indexed).
// 2: all ranks of a group parse slices of its runs and send each entry to
//    the rank that owns it (read_sdpa_lines, in SDP.cxx).
int sdpa_indexed_reader_mode(const std::filesystem::path &input_file);

// Collective over comm: each rank scans 1/comm.Size() of the file;
// the runs are gathered on all ranks.
SDPA_Sparse_Index
build_sdpa_sparse_index(const std::filesystem::path &input_file,
                        const El::mpi::Comm &comm);

// Same result as read_sdpa(), but only the runs of the blocks to parse are read.
SDPA_File_Parse_Result read_sdpa_indexed(
  const std::filesystem::path &input_file, const SDPA_Sparse_Index &index,
  const std::function<bool(size_t matrix_index)> &should_parse_block);

// Objective c_1..c_m (reads only the header).
std::vector<El::BigFloat>
read_sdpa_objective(const std::filesystem::path &input_file);

// Parse the entry lines that start in [begin, end), a range inside the run
// [run_begin, run_end), in file order, with the same parser as read_sdpa().
// accept(block_index, matrix_index, i, j, value): solver block, 0..m, and the
// position inside the block (not mirrored).
void read_sdpa_lines(
  const std::filesystem::path &input_file, const SDPA_Sparse_Index &index,
  size_t run_begin, size_t run_end, size_t begin, size_t end,
  const std::function<void(size_t block_index, size_t matrix_index, size_t i,
                           size_t j, const El::BigFloat &value)> &accept);
}
