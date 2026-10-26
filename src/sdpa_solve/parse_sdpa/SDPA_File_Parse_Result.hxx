#pragma once

// #include "SDPA/Polynomial_Vector_Matrix.hxx"

#include <El.hpp>

#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <vector>

namespace Sdpb::Sdpa
{
  // One parsed nonzero of F_{constraint} (constraint = 0 means F_0), solver block indices,
  // both symmetric halves emitted by the reader. The value lives in
  // SDPA_File_Parse_Result::F_values; bit-identical values share one slot.
  struct Parsed_F_Entry
  {
    // 0..m, where 0 is F_0
    uint32_t constraint;
    int32_t row;
    int32_t col;
    uint32_t value_index;
  };

  struct SDPA_File_Parse_Result
  {
    // Vector c_1..c_M
    std::vector<El::BigFloat> c_objective;
    // If file is read by several processes,
    // each process saves only some blocks, according to should_parse_block().
    // parsed_block_entries: block_index -> nonzeros of [F_0, F_1,...F_M] in that block.
    //
    // #184: the parser used to materialize every F_i densely here -- (m+1)*dim^2 zero-filled
    // BigFloats per block, i.e. the whole dense set a second time during setup. The .dat-s
    // format is itself a triplet list, so triplets are what we keep.
    std::map<size_t, std::vector<Parsed_F_Entry>> parsed_block_entries;
    // Distinct values of all parsed F entries. Bootstrap SDPs reuse values heavily
    // (12_min: 535k entries, 106k distinct), and a 512-bit BigFloat costs ~136 B.
    std::vector<El::BigFloat> F_values;

    SDPA_File_Parse_Result() = default;

    static SDPA_File_Parse_Result
    read(const std::filesystem::path &input_path, bool should_parse_objective,
         const std::function<bool(size_t block_index)> &should_parse_block);

    // Allow moving and prevent accidental copying

    SDPA_File_Parse_Result(const SDPA_File_Parse_Result &other) = delete;
    SDPA_File_Parse_Result(SDPA_File_Parse_Result &&other) noexcept = default;
    SDPA_File_Parse_Result &operator=(const SDPA_File_Parse_Result &other)
      = delete;
    SDPA_File_Parse_Result &operator=(SDPA_File_Parse_Result &&other) noexcept
      = default;
  };
}
