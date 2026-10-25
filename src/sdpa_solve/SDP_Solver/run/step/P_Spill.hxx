#pragma once
// #184 (bootstrap-speed): packed, per-rank store of the normalized-and-shifted P matrix.
//
// P = (vec(G_1) ... vec(G_m)) is sum(dim^2) x m and existed as a full BigFloat Block_Matrix
// (~GB at 512 bits on the 12_min benchmark) only so that the syrk driver could re-read row
// bands of it while computing S = P^T P. After column normalization every entry is an integer
// with |v| <= 2^bits, so a fixed-size limb record holds it exactly. initialize_P writes each
// chunk of columns here and frees the BigFloat chunk; the syrk driver reads back bounded bands.
// P never exists as a whole BigFloat matrix.
//
// Layout contract: records hold this rank's local elements of a virtual
// DistMatrix<BigFloat>(sum_dim2 x m) per block with alignment (0, 0) on `grid`.
// A column's run is this rank's rows, ascending, contiguous. Chunks are written with matching
// alignment (0, col_begin % grid.Width()), and bands are read back through fresh matrices
// aligned like views of the virtual P, so no element ever moves between ranks.
//
// Backend: RAM (std::vector) by default, or an unlinked file in $SDPB_P_SPILL_DIR when that
// is set (node-local scratch turns P's footprint into page cache instead of anonymous memory).

#include "sdpb_util/assert.hxx"

#include <El.hpp>

#include <filesystem>
#include <optional>
#include <vector>

namespace Sdpb::Sdpa
{
  struct P_Spill
  {
    int bits = 0;
    size_t limbs_per_record = 0;
    // Heights (dim^2) of this rank's local blocks, in local block order.
    std::vector<El::Int> block_heights;
    El::Int width = 0;
    const El::Grid *grid = nullptr;

    // column_offsets[local_block][global_col] = byte offset of that column's record run,
    // or -1 before it is written. local_heights[b] = records per column run.
    std::vector<std::vector<long long>> column_offsets;
    std::vector<El::Int> local_heights;

    // Backend
    bool use_file = false;
    int fd = -1;
    std::vector<unsigned char> ram;
    long long end_offset = 0;

    P_Spill() = default;
    P_Spill(const P_Spill &) = delete;
    P_Spill &operator=(const P_Spill &) = delete;
    P_Spill(P_Spill &&other) noexcept;
    ~P_Spill();

    size_t record_bytes() const { return 8 * (limbs_per_record + 1); }

    void initialize(const std::vector<El::Int> &block_heights, El::Int width,
                    const El::Grid &grid, int bits,
                    const std::optional<std::filesystem::path> &spill_dir);

    // Append columns [col_begin, col_begin + chunk.Width()) of local block b.
    // chunk_block must be full height and aligned (0, col_begin % grid.Width()).
    void append_chunk_block(size_t local_block, El::Int col_begin,
                            const El::DistMatrix<El::BigFloat> &chunk_block);

    // dest := virtual P block b restricted to (rows x cols), as a fresh matrix
    // aligned (rows.beg % grid.Height(), cols.beg % grid.Width()) so its local
    // elements are exactly the spilled ones. Zero-height ranges are fine.
    void read_band(size_t local_block, El::Range<El::Int> rows,
                   El::Range<El::Int> cols,
                   El::DistMatrix<El::BigFloat> &dest) const;

    size_t bytes_stored() const { return end_offset; }
  };
}
