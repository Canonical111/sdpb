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
};

// True if LAB_SDPA_INDEXED_READER=1 and input_file is a plain .dat-s file.
bool use_sdpa_indexed_reader(const std::filesystem::path &input_file);

// Collective over comm: each rank scans 1/comm.Size() of the file;
// the runs are gathered on all ranks.
SDPA_Sparse_Index
build_sdpa_sparse_index(const std::filesystem::path &input_file,
                        const El::mpi::Comm &comm);

// Same result as read_sdpa(), but only the runs of the blocks to parse are read.
SDPA_File_Parse_Result read_sdpa_indexed(
  const std::filesystem::path &input_file, const SDPA_Sparse_Index &index,
  const std::function<bool(size_t matrix_index)> &should_parse_block);
}