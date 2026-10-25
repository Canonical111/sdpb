#include "P_Spill.hxx"

#include "sdpb_util/bigint_shared_memory/fmpz/Fmpz_BigInt.hxx"

#include <fcntl.h>
#include <unistd.h>

#include <cstring>

namespace Sdpb::Sdpa
{
  namespace
  {
    // record = [int64 sign][limbs_per_record x uint64 magnitude limbs, little-endian]
    void pack_record(const El::BigFloat &value, Fmpz_BigInt &scratch,
                     size_t limbs_per_record, unsigned char *out)
    {
      scratch.from_BigFloat(value);
      const auto sign = static_cast<long long>(fmpz_sgn(scratch.value));
      std::memcpy(out, &sign, 8);
      auto *limbs = reinterpret_cast<mp_limb_t *>(out + 8);
      const size_t size = fmpz_size(scratch.value);
      ASSERT(size <= limbs_per_record,
             "P_Spill: normalized P entry does not fit the fixed record",
             DEBUG_STRING(size), DEBUG_STRING(limbs_per_record));
      fmpz_get_ui_array(limbs, limbs_per_record, scratch.value);
    }

    void unpack_record(const unsigned char *in, size_t limbs_per_record,
                       Fmpz_BigInt &scratch, El::BigFloat &value)
    {
      long long sign;
      std::memcpy(&sign, in, 8);
      const auto *limbs = reinterpret_cast<const mp_limb_t *>(in + 8);
      fmpz_set_ui_array(scratch.value, limbs, limbs_per_record);
      if(sign < 0)
        fmpz_neg(scratch.value, scratch.value);
      scratch.to_BigFloat(value);
    }
  }

  P_Spill::P_Spill(P_Spill &&other) noexcept
      : bits(other.bits), limbs_per_record(other.limbs_per_record),
        block_heights(std::move(other.block_heights)), width(other.width),
        grid(other.grid), column_offsets(std::move(other.column_offsets)),
        local_heights(std::move(other.local_heights)),
        use_file(other.use_file), fd(other.fd), ram(std::move(other.ram)),
        end_offset(other.end_offset)
  {
    other.fd = -1;
  }

  P_Spill::~P_Spill()
  {
    if(fd >= 0)
      close(fd);
  }

  void P_Spill::initialize(const std::vector<El::Int> &heights,
                           const El::Int _width, const El::Grid &_grid,
                           const int _bits,
                           const std::optional<std::filesystem::path> &dir)
  {
    bits = _bits;
    // Shifted entries are integers with |v| <= 2^bits (the bound is attained
    // by a column with a single nonzero), hence bits + 1 magnitude bits.
    limbs_per_record = (static_cast<size_t>(bits) + 1 + 63) / 64;
    block_heights = heights;
    width = _width;
    grid = &_grid;
    column_offsets.assign(heights.size(),
                          std::vector<long long>(width, -1));
    local_heights.clear();
    local_heights.reserve(heights.size());
    // Virtual P blocks are aligned (0, 0): this rank's rows of a block are
    // those congruent to grid.Row() modulo grid.Height().
    for(const auto h : heights)
      local_heights.push_back(
        El::Length(h, grid->Row(), 0, grid->Height()));

    if(dir.has_value())
      {
        const auto path
          = dir.value()
            / El::BuildString("sdpb_P_spill_rank_",
                              El::mpi::Rank(El::mpi::COMM_WORLD), ".bin");
        fd = open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0600);
        ASSERT(fd >= 0, "P_Spill: cannot create spill file ", path.string());
        // Unlink immediately: the file stays alive while open and cleans
        // itself up however the process exits.
        unlink(path.c_str());
        use_file = true;
      }
  }

  void
  P_Spill::append_chunk_block(const size_t local_block, const El::Int col_begin,
                              const El::DistMatrix<El::BigFloat> &chunk_block)
  {
    ASSERT_EQUAL(chunk_block.Height(), block_heights.at(local_block));
    ASSERT_EQUAL(chunk_block.ColAlign(), 0);
    ASSERT_EQUAL(chunk_block.RowAlign(),
                 col_begin % static_cast<El::Int>(grid->Width()));
    ASSERT_EQUAL(chunk_block.LocalHeight(), local_heights.at(local_block));

    const auto rec = record_bytes();
    const auto run_bytes
      = static_cast<size_t>(chunk_block.LocalHeight()) * rec;
    std::vector<unsigned char> buffer(run_bytes);
    Fmpz_BigInt scratch;
    for(El::Int jLoc = 0; jLoc < chunk_block.LocalWidth(); ++jLoc)
      {
        const El::Int global_col = col_begin + chunk_block.GlobalCol(jLoc);
        ASSERT(column_offsets.at(local_block).at(global_col) == -1,
               "P_Spill: column written twice", DEBUG_STRING(global_col));
        for(El::Int iLoc = 0; iLoc < chunk_block.LocalHeight(); ++iLoc)
          {
            pack_record(chunk_block.GetLocalCRef(iLoc, jLoc), scratch,
                        limbs_per_record, buffer.data() + iLoc * rec);
          }
        column_offsets.at(local_block).at(global_col) = end_offset;
        if(use_file)
          {
            size_t written = 0;
            while(written < run_bytes)
              {
                const auto n
                  = pwrite(fd, buffer.data() + written, run_bytes - written,
                           end_offset + written);
                ASSERT(n > 0, "P_Spill: write failed");
                written += n;
              }
          }
        else
          {
            ram.insert(ram.end(), buffer.begin(), buffer.end());
          }
        end_offset += run_bytes;
      }
  }

  void P_Spill::read_band(const size_t local_block,
                          const El::Range<El::Int> rows,
                          const El::Range<El::Int> cols,
                          El::DistMatrix<El::BigFloat> &dest) const
  {
    const auto grid_height = static_cast<El::Int>(grid->Height());
    const auto grid_width = static_cast<El::Int>(grid->Width());
    dest.SetGrid(*grid);
    dest.Empty();
    dest.Align(rows.beg % grid_height, cols.beg % grid_width);
    dest.Resize(rows.end - rows.beg, cols.end - cols.beg);
    if(dest.LocalHeight() == 0 || dest.LocalWidth() == 0)
      return;

    const auto rec = record_bytes();
    const auto run_bytes = static_cast<size_t>(dest.LocalHeight()) * rec;
    std::vector<unsigned char> buffer(run_bytes);
    // First owned row of the virtual block at or after rows.beg has local
    // index Length(rows.beg, colShift=grid.Row(), stride) in the spilled run.
    const auto first_local_row
      = El::Length(rows.beg, grid->Row(), 0, grid_height);
    Fmpz_BigInt scratch;
    El::BigFloat value;
    for(El::Int jLoc = 0; jLoc < dest.LocalWidth(); ++jLoc)
      {
        const El::Int global_col = cols.beg + dest.GlobalCol(jLoc);
        const auto offset = column_offsets.at(local_block).at(global_col);
        ASSERT(offset >= 0, "P_Spill: reading unwritten column",
               DEBUG_STRING(global_col), DEBUG_STRING(local_block));
        const auto start = offset + static_cast<long long>(first_local_row) * rec;
        if(use_file)
          {
            size_t done = 0;
            while(done < run_bytes)
              {
                const auto n = pread(fd, buffer.data() + done,
                                     run_bytes - done, start + done);
                ASSERT(n > 0, "P_Spill: read failed");
                done += n;
              }
          }
        else
          {
            std::memcpy(buffer.data(), ram.data() + start, run_bytes);
          }
        for(El::Int iLoc = 0; iLoc < dest.LocalHeight(); ++iLoc)
          {
            unpack_record(buffer.data() + iLoc * rec, limbs_per_record,
                          scratch, value);
            dest.SetLocal(iLoc, jLoc, value);
          }
      }
  }
}
