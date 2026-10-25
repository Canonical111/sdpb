#include "parse_sdpa.hxx"
#include "SDPA_File_Parse_Result.hxx"
#include "SDPA_Block_Structure.hxx"
#include "sdpb_util/Rss_Log.hxx"
#include "catch2/catch_amalgamated.hpp"
#include "sdpb_util/assert.hxx"

#include <El.hpp>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <limits>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/device/file.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/device/array.hpp>
#include <boost/iostreams/stream.hpp>

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>

namespace fs = std::filesystem;

namespace Sdpb::Sdpa
{
  namespace
  {
    enum SDPA_File_Type
    {
      // .dat-s
      sparse,
      // .dat
      dense
    };

    constexpr auto max_size = std::numeric_limits<std::streamsize>::max();

    void skip_line(std::istream &is)
    {
      is.ignore(max_size, '\n');
    }

    bool should_skip(const int token, const bool skip_newline)
    {
      if(!skip_newline && token == '\n')
        return false;
      return token == ',' || token == '(' || token == '{' || token == '}'
             || std::isspace(token);
    }

    template <class IStream> void skip_ws(IStream &is, const bool skip_newline)
    {
      while(should_skip(is.peek(), skip_newline))
        {
          is.ignore(1);
        }
    }

    template <class T, class IStream>
    T read_number(IStream &is, const bool skip_newline)
    {
      T value;
      skip_ws(is, skip_newline);
      is >> value;
      ASSERT(is.good());
      return value;
    }

    // Read vector of a given size from single line.
    // Skip comments at the end of line.
    template <class T>
    std::vector<T> read_vector(std::istream &is, const size_t size)
    {
      std::vector<T> result;
      result.reserve(size);

      for(size_t i = 0; i < size; ++i)
        {
          result.push_back(read_number<T>(is, false));
        }
      skip_line(is);
      ASSERT(is.good());
      return result;
    }

    // https://github.com/vsdp/SDPLIB

    // 1. Comments. The file can begin with arbitrarily many lines of comments. Each line of comments must begin with " or *.
    template <class IStream> void read_comments(IStream &is)
    {
      int next = is.peek();
      ASSERT(is.good());
      while(next == '"' || next == '*')
        {
          skip_line(is);
          next = is.peek();
          ASSERT(is.good());
        }
    }

    // 2. The first line after the comments contains m, the number of constraint matrices.
    // Additional text on this line after m is ignored.
    template <class IStream> size_t read_m(IStream &is)
    {
      size_t m = 0;
      is >> m;
      skip_line(is);
      ASSERT(is.good());
      return m;
    }

    // 3. The second line after the comments contains nblocks, the number of blocks in the block diagonal structure of the matrices.
    // Additional text on this line after nblocks is ignored.
    template <class IStream> size_t read_nblocks(IStream &is)
    {
      size_t nblocks;
      is >> nblocks;
      ASSERT(is.good());
      is.ignore(max_size, '\n');
      return nblocks;
    }

    // 4. The third line after the comments contains a vector of numbers that give the sizes of the individual blocks.
    // The special characters ',', '(', ')', '{', and '}' can be used as punctuation and are ignored.
    // Negative numbers may be used to indicate that a block is actually a diagonal submatrix.
    // Thus, a block size of -5 indicates a 5 by 5 block in which only the diagonal elements are nonzero.
    // Further we'll split a diagonal block into 1x1 blocks.
    template <class IStream>
    std::vector<int> read_block_sizes(IStream &is, const size_t &nblocks)
    {
      return read_vector<int>(is, nblocks);
    }

    // 5. The fourth line after the comments contains the objective function vector c.
    template <class IStream>
    std::vector<El::BigFloat> read_c(IStream &is, const size_t m)
    {
      return read_vector<El::BigFloat>(is, m);
    }

    struct SDPA_Matrix_Entry
    {
      // 0..m
      size_t matrix_index{};
      // 1..n_blocks
      size_t block_index{};
      // indices inside block
      size_t i{};
      size_t j{};
      El::BigFloat value;
    };

    // 6. The remaining lines of the file contain entries in the constraint matrices, with one entry per line.
    // The format for each line is
    // <matno> <blkno> <i> <j> <entry>
    // Here <matno> is the number of the matrix to which this entry belongs,
    // <blkno> specifies the block within this matrix,
    // <i> and <j> specify a location within the block,
    // and <entry> gives the value of the entry in the matrix.
    // Note that since all matrices are assumed to be symmetric, only entries in the upper triangle of a matrix are given.
    template <class IStream>
    void read_sparse_matrix_entry(IStream &is, SDPA_Matrix_Entry &entry)
    {
      is >> entry.matrix_index;
      is >> entry.block_index;
      is >> entry.i;
      is >> entry.j;
      is >> entry.value;
      std::ws(is);
      ASSERT(is.eof() || is.good());
    }

    // Matrix entry with indices used by solver,
    // i.e. after splitting diagonal SDP blocks into 1x1 blocks.
    struct Result_Matrix_Entry
    {
      // Number of SDP block
      size_t sdp_block_index;
      // Each matrix element is (degree-0) polynomial vector;
      // This is the index inside this vector.
      size_t vector_index;
      size_t i;
      size_t j;
      El::BigFloat value;

      Result_Matrix_Entry(const SDPA_Matrix_Entry &entry,
                        const SDPA_Block_Structure &block_structure)
      {
        block_structure.get_element_position(entry.block_index, entry.i,
                                             entry.j, sdp_block_index, i, j);

        this->vector_index = entry.matrix_index;
        this->value = entry.value;
      }
    };

    template <class IStream>
    void read_sparse_matrices(
      IStream &is, const std::function<void(SDPA_Matrix_Entry &&)> &accept,
      const std::function<bool(const SDPA_Matrix_Entry &)> &wanted)
    {
      // Read all matrix entries in the file.
      // lab (bfss #44): the indices are read first; the value is converted to
      // BigFloat only if this rank parses the entry's block. Other lines are
      // skipped as text, so each value is converted by exactly one group root.
      while(!is.eof())
        {
          SDPA_Matrix_Entry sdpa_entry;
          is >> sdpa_entry.matrix_index;
          is >> sdpa_entry.block_index;
          is >> sdpa_entry.i;
          is >> sdpa_entry.j;
          ASSERT(is.good(), "Error reading SDPA entry indices");
          if(wanted(sdpa_entry))
            {
              is >> sdpa_entry.value;
              std::ws(is);
              ASSERT(is.eof() || is.good());
              accept(std::move(sdpa_entry));
            }
          else
            {
              is.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
              std::ws(is);
            }
        }
    }

    template <class IStream>
    void read_dense_matrices(
      IStream &is, const SDPA_Block_Structure &block_structure,
      const std::function<void(SDPA_Matrix_Entry &&)> &accept)

    {
      // 0..m
      for(size_t matrix_index = 0; matrix_index <= block_structure.m_dim;
          ++matrix_index)
        {
          // 1..nblocks
          for(size_t block_index = 1;
              block_index <= block_structure.input_block_sizes.size();
              ++block_index)
            {
              const int sdpa_block_size
                = block_structure.input_block_sizes.at(block_index - 1);
              ASSERT(sdpa_block_size != 0);
              try
                {
                  if(sdpa_block_size > 0)
                    {
                      // Parse dense matrix
                      const size_t size = sdpa_block_size;
                      for(size_t i = 1; i <= size; ++i)
                        for(size_t j = 1; j <= size; ++j)
                          {
                            SDPA_Matrix_Entry sdpa_entry;
                            sdpa_entry.block_index = block_index;
                            sdpa_entry.matrix_index = matrix_index;
                            sdpa_entry.i = i;
                            sdpa_entry.j = j;
                            sdpa_entry.value
                              = read_number<El::BigFloat>(is, true);
                            accept(std::move(sdpa_entry));
                          }
                    }
                  else
                    {
                      // Parse diagonal matrix entries
                      const size_t size = -sdpa_block_size;
                      for(size_t i = 1; i <= size; ++i)
                        {
                          SDPA_Matrix_Entry sdpa_entry;
                          sdpa_entry.block_index = block_index;
                          sdpa_entry.matrix_index = matrix_index;
                          sdpa_entry.i = i;
                          sdpa_entry.j = i;
                          sdpa_entry.value
                            = read_number<El::BigFloat>(is, true);
                          accept(std::move(sdpa_entry));
                        }
                    }
                }
              catch(const std::exception &e)
                {
                  RUNTIME_ERROR(
                    "Error when parsing dense SDPA matrix: ",
                    DEBUG_STRING(matrix_index), DEBUG_STRING(block_index),
                    DEBUG_STRING(sdpa_block_size), ":\n", e.what());
                }
            }
        }
    }
  }

  // Read .dat-s input used in SDPA
  // See format description e.g. in https://github.com/vsdp/SDPLIB
  // Read only block structure, do not read objective and SDP matrices.
  template <class IStream>
  SDPA_Block_Structure read_block_structure(IStream &is)
  {
    read_comments(is);

    // m (from SDPA formulation) is equal to N (in SDPB formulation 2.2 or 3.1)
    const size_t m = read_m(is);

    // Number of SDP blocks
    const size_t num_blocks = read_nblocks(is);

    const auto block_sizes = read_block_sizes(is, num_blocks);
    ASSERT_EQUAL(num_blocks, block_sizes.size());
    return SDPA_Block_Structure(m, num_blocks, block_sizes);
  }

  // Read .dat-s input used in SDPA
  // See format description e.g. in https://github.com/vsdp/SDPLIB
  // lab (bfss #61): if read_entries is set, the entries are read by it
  // instead of from the rest of is (is then holds only the header).
  using Read_Entries = std::function<void(
    const std::function<void(SDPA_Matrix_Entry &&)> &accept,
    const std::function<bool(const SDPA_Matrix_Entry &)> &wanted)>;

  template <class IStream>
  SDPA_File_Parse_Result
  read_sdpa(IStream &is, const SDPA_File_Type file_type,
            const std::function<bool(size_t block_index)> &should_parse_block,
            const Read_Entries &read_entries = nullptr)
  {
    SDPA_File_Parse_Result result;
    const SDPA_Block_Structure block_structure = read_block_structure(is);

    result.c_objective = read_c(is, block_structure.m_dim);
    ASSERT_EQUAL(block_structure.m_dim, result.c_objective.size());

    // sdp_blocks_F[i][j] is i-th block of matrix F_j
    std::vector<std::optional<std::vector<El::Matrix<El::BigFloat>>>>
      sdp_blocks_F(block_structure.num_solver_blocks);
    for(size_t index = 0; index < sdp_blocks_F.size(); ++index)
      {
        if(!should_parse_block(index))
          continue;
        const size_t height = block_structure.solver_block_sizes.at(index);
        const size_t width = height;
        El::Matrix<El::BigFloat> zero(height, width);
        El::Zero(zero);
        sdp_blocks_F.at(index).emplace(block_structure.m_dim + 1, zero);
      }
    rss_log("mark", "parse.zeros_allocated");

    const auto accept_sdpa_entry = [&](SDPA_Matrix_Entry &&sdpa_entry) {
      const Result_Matrix_Entry entry(sdpa_entry, block_structure);
      const auto i = entry.i;
      const auto j = entry.j;

      if(!should_parse_block(entry.sdp_block_index))
        return;

      auto &matrices = sdp_blocks_F.at(entry.sdp_block_index).value();

      ASSERT_EQUAL(matrices.size(), block_structure.m_dim + 1, DEBUG_STRING(i),
                   DEBUG_STRING(j), DEBUG_STRING(sdpa_entry.matrix_index),
                   DEBUG_STRING(sdpa_entry.block_index));
      auto &matrix = matrices.at(entry.vector_index);
      matrix(i, j) = entry.value;
      if(i != j && file_type == SDPA_File_Type::sparse)
        matrix(j, i) = entry.value;
    };

    const std::function<bool(const SDPA_Matrix_Entry &)> wanted
      = [&](const SDPA_Matrix_Entry &sdpa_entry) {
          size_t sdp_block_index, i, j;
          block_structure.get_element_position(sdpa_entry.block_index,
                                               sdpa_entry.i, sdpa_entry.j,
                                               sdp_block_index, i, j);
          return should_parse_block(sdp_block_index);
        };
    if(read_entries)
      {
        // The index scan has checked the whole file's entry lines.
        ASSERT(file_type == SDPA_File_Type::sparse);
        read_entries(accept_sdpa_entry, wanted);
      }
    else
      {
        switch(file_type)
          {
          case sparse:
            read_sparse_matrices(is, accept_sdpa_entry, wanted);
            break;
          case dense:
            read_dense_matrices(is, block_structure, accept_sdpa_entry);
            break;
          default: LOGIC_ERROR("Unknown SDPA_File_Type=", file_type);
          }

        skip_ws(is, true);
        if(!is.eof())
          {
            std::string line;
            const auto pos = is.tellg();
            std::getline(is, line);
            RUNTIME_ERROR("Unexpected data at the end of file, position: ",
                          pos, ", next line: ", line);
          }
      }

    rss_log("mark", "parse.entries_read");
    for(size_t index = 0; index < sdp_blocks_F.size(); ++index)
      {
        if(!should_parse_block(index))
          continue;
        // lab (bfss #44): move instead of copy; the copy doubled the root's peak.
        result.parsed_blocks.emplace(index,
                                     std::move(sdp_blocks_F.at(index).value()));
      }
    rss_log("mark", "parse.result_copied");
    return result;
  }

  template <class TResult>
  TResult read_sdpa_file(
    const std::filesystem::path &input_file,
    const std::function<TResult(boost::iostreams::filtering_istream &,
                                const SDPA_File_Type &)> &read_sdpa_stream)
  {
    auto extension = input_file.extension().string();
    bool is_gzip = false;

    const auto path = input_file.string();

    if(extension == ".gz")
      {
        is_gzip = true;
        extension = fs::path(input_file).replace_extension("").extension();
      }

    const SDPA_File_Type file_type = [&] {
      if(extension == ".dat-s")
        return SDPA_File_Type::sparse;
      if(extension == ".dat")
        return SDPA_File_Type::dense;
      RUNTIME_ERROR("Unknown SDPA file type, expected .dat, .dat-s, .dat.gz, "
                    "or .dat-s.gz: ",
                    input_file);
    }();

    boost::iostreams::filtering_istream is;
    if(is_gzip)
      {
        is.push(boost::iostreams::gzip_decompressor());
        is.push(boost::iostreams::file_source(
          input_file, std::ios_base::in | std::ios_base::binary));
      }
    else
      {
        is.push(boost::iostreams::file_source(input_file));
      }

    ASSERT(is.good(), "Could not open ", input_file);
    return read_sdpa_stream(is, file_type);
  }

  SDPA_Block_Structure
  read_block_structure(const std::filesystem::path &input_file)
  {
    const auto f = std::function(
      [](boost::iostreams::filtering_istream &is, const SDPA_File_Type &) {
        return read_block_structure(is);
      });
    return read_sdpa_file(input_file, f);
  }

  SDPA_File_Parse_Result
  read_sdpa(const std::filesystem::path &input_file,
            const std::function<bool(size_t matrix_index)> &should_parse_block)
  {
    const auto f = std::function(
      [&should_parse_block](boost::iostreams::filtering_istream &is,
                            const SDPA_File_Type &file_type) {
        return read_sdpa(is, file_type, should_parse_block);
      });
    return read_sdpa_file(input_file, f);
  }

  // lab (bfss #61): indexed reader for plain .dat-s files.
  //
  // The deployed reader makes every MPI group root scan the whole file
  // (254 GB at level 12, 55 roots: 662-5556 s per root on 10 nodes). Here
  // all ranks first scan 1/N of the file each and record the byte ranges
  // ("runs") of each solver block; then each group root reads only the runs
  // of its own blocks, in file order, with the same entry parser as before.
  namespace
  {
    // Lines longer than this are an error.
    constexpr size_t index_chunk_bytes = 64 * 1024 * 1024;

    class Posix_File
    {
    public:
      explicit Posix_File(const fs::path &path) : path(path)
      {
        fd = ::open(path.c_str(), O_RDONLY);
        ASSERT(fd >= 0, "Could not open ", path, ": ", std::strerror(errno));
      }
      ~Posix_File()
      {
        if(fd >= 0)
          ::close(fd);
      }
      Posix_File(const Posix_File &) = delete;
      Posix_File &operator=(const Posix_File &) = delete;

      // Read up to count bytes at offset; returns the number of bytes read
      // (less than count only at end of file).
      size_t read(char *buffer, const size_t count, const size_t offset) const
      {
        size_t done = 0;
        while(done < count)
          {
            const ssize_t n
              = ::pread(fd, buffer + done, count - done, offset + done);
            if(n < 0 && errno == EINTR)
              continue;
            ASSERT(n >= 0, "pread failed on ", path, " at offset ",
                   offset + done, ": ", std::strerror(errno));
            if(n == 0)
              break;
            done += n;
          }
        return done;
      }

    private:
      fs::path path;
      int fd = -1;
    };

    bool is_blank_char(const char c)
    {
      return c == ' ' || c == '\t' || c == '\r' || c == '\v' || c == '\f';
    }

    // Parse an unsigned decimal integer after optional blanks.
    size_t parse_index_field(const char *&p, const char *end,
                             const size_t line_offset)
    {
      while(p < end && is_blank_char(*p))
        ++p;
      ASSERT(p < end && *p >= '0' && *p <= '9',
             "Error reading SDPA entry indices at offset ", line_offset);
      size_t value = 0;
      while(p < end && *p >= '0' && *p <= '9')
        {
          value = value * 10 + (*p - '0');
          ++p;
        }
      ASSERT(p == end || is_blank_char(*p),
             "Error reading SDPA entry indices at offset ", line_offset);
      return value;
    }

    void add_line_to_runs(std::vector<SDPA_Block_Run> &runs,
                          const SDPA_Block_Structure &block_structure,
                          const char *begin, const char *end,
                          const size_t line_begin, const size_t line_end)
    {
      const char *p = begin;
      while(p < end && is_blank_char(*p))
        ++p;
      if(p == end)
        {
          // Blank line: keep it inside the current run, if adjacent.
          if(!runs.empty() && runs.back().end == line_begin)
            runs.back().end = line_end;
          return;
        }
      const size_t matrix_index = parse_index_field(p, end, line_begin);
      const size_t input_block_index = parse_index_field(p, end, line_begin);
      const size_t input_i = parse_index_field(p, end, line_begin);
      const size_t input_j = parse_index_field(p, end, line_begin);
      while(p < end && is_blank_char(*p))
        ++p;
      ASSERT(p < end, "Missing SDPA entry value at offset ", line_begin);
      ASSERT(matrix_index <= block_structure.m_dim,
             "SDPA entry matrix index out of range at offset ", line_begin,
             DEBUG_STRING(matrix_index));
      ASSERT(input_block_index >= 1
               && input_block_index
                    <= block_structure.input_block_sizes.size(),
             "SDPA entry block index out of range at offset ", line_begin,
             DEBUG_STRING(input_block_index));
      size_t block_index, i, j;
      block_structure.get_element_position(input_block_index, input_i,
                                           input_j, block_index, i, j);
      const size_t dim = block_structure.solver_block_sizes.at(block_index);
      ASSERT(i < dim && j < dim,
             "SDPA entry position out of range at offset ", line_begin,
             DEBUG_STRING(input_block_index), DEBUG_STRING(input_i),
             DEBUG_STRING(input_j));

      if(!runs.empty() && runs.back().block_index == block_index
         && runs.back().end == line_begin)
        runs.back().end = line_end;
      else
        runs.push_back({block_index, line_begin, line_end});
    }

    // Runs of the entry lines starting in [lo, hi).
    std::vector<SDPA_Block_Run>
    scan_range(const Posix_File &file,
               const SDPA_Block_Structure &block_structure,
               const size_t data_begin, const size_t file_size,
               const size_t lo, const size_t hi)
    {
      std::vector<SDPA_Block_Run> runs;
      std::vector<char> buffer(index_chunk_bytes);
      size_t pos = lo;
      if(lo > data_begin && lo < hi)
        {
          // Skip the line that started before lo.
          char previous;
          ASSERT_EQUAL(file.read(&previous, 1, lo - 1), 1);
          if(previous != '\n')
            {
              while(pos < hi)
                {
                  const size_t n = file.read(
                    buffer.data(), std::min(buffer.size(), file_size - pos),
                    pos);
                  ASSERT(n > 0);
                  const auto *nl = static_cast<const char *>(
                    std::memchr(buffer.data(), '\n', n));
                  if(nl != nullptr)
                    {
                      pos += nl - buffer.data() + 1;
                      break;
                    }
                  pos += n;
                }
            }
        }
      while(pos < hi)
        {
          const size_t n = file.read(
            buffer.data(), std::min(buffer.size(), file_size - pos), pos);
          ASSERT(n > 0);
          size_t k = 0;
          while(k < n && pos + k < hi)
            {
              const char *line = buffer.data() + k;
              const auto *nl
                = static_cast<const char *>(std::memchr(line, '\n', n - k));
              size_t line_end;
              const char *content_end;
              if(nl != nullptr)
                {
                  line_end = nl - buffer.data() + 1;
                  content_end = nl;
                }
              else
                {
                  // Incomplete line: read again from its start,
                  // unless this is the last line of the file.
                  if(pos + n < file_size)
                    {
                      ASSERT(k > 0, "SDPA line longer than ",
                             index_chunk_bytes, " bytes at offset ", pos);
                      break;
                    }
                  line_end = n;
                  content_end = buffer.data() + n;
                }
              add_line_to_runs(runs, block_structure, line, content_end,
                               pos + k, pos + line_end);
              k = line_end;
            }
          pos += k;
        }
      return runs;
    }

    // Read the header (comments, m, nblocks, block sizes, c) with the same
    // code as read_sdpa(); returns the offset of the first entry line.
    size_t read_header(const fs::path &input_file,
                       std::optional<SDPA_Block_Structure> &block_structure,
                       std::vector<El::BigFloat> &c_objective)
    {
      std::ifstream is(input_file, std::ios_base::in | std::ios_base::binary);
      ASSERT(is.good(), "Could not open ", input_file);
      block_structure.emplace(read_block_structure(is));
      c_objective = read_c(is, block_structure->m_dim);
      ASSERT_EQUAL(block_structure->m_dim, c_objective.size());
      const auto pos = is.tellg();
      ASSERT(pos >= 0, "tellg failed on ", input_file);
      return pos;
    }
  }

  int sdpa_indexed_reader_mode(const std::filesystem::path &input_file)
  {
    const char *env = std::getenv("LAB_SDPA_INDEXED_READER");
    if(env == nullptr)
      return 0;
    const std::string value(env);
    if(value != "1" && value != "2")
      return 0;
    if(input_file.extension() != ".dat-s" || !fs::is_regular_file(input_file))
      return 0;
    return std::stoi(value);
  }

  SDPA_Sparse_Index
  build_sdpa_sparse_index(const std::filesystem::path &input_file,
                          const El::mpi::Comm &comm)
  {
    const int rank = comm.Rank();
    const int num_ranks = comm.Size();

    // Header: rank 0 reads it and broadcasts
    // [data_begin, file_size, m, nblocks, block sizes...].
    std::vector<int64_t> header;
    if(rank == 0)
      {
        std::optional<SDPA_Block_Structure> block_structure;
        std::vector<El::BigFloat> c_objective;
        const size_t data_begin
          = read_header(input_file, block_structure, c_objective);
        header.push_back(data_begin);
        header.push_back(fs::file_size(input_file));
        header.push_back(block_structure->m_dim);
        header.push_back(block_structure->input_block_sizes.size());
        for(const auto size : block_structure->input_block_sizes)
          header.push_back(size);
      }
    int64_t header_size = header.size();
    MPI_Bcast(&header_size, 1, MPI_INT64_T, 0, comm.comm);
    header.resize(header_size);
    MPI_Bcast(header.data(), header_size, MPI_INT64_T, 0, comm.comm);

    SDPA_Sparse_Index index;
    index.data_begin = header.at(0);
    index.file_size = header.at(1);
    const size_t m = header.at(2);
    const size_t nblocks = header.at(3);
    ASSERT_EQUAL(header.size(), 4 + nblocks);
    const std::vector<int> block_sizes(header.begin() + 4, header.end());
    const SDPA_Block_Structure block_structure(m, nblocks, block_sizes);
    index.m_dim = m;
    index.input_block_sizes = block_sizes;
    ASSERT(index.data_begin <= index.file_size);

    // Each rank scans the lines starting in its share of the entry bytes.
    const size_t span = index.file_size - index.data_begin;
    const size_t lo = index.data_begin + span / num_ranks * rank
                      + std::min<size_t>(rank, span % num_ranks);
    const size_t hi = index.data_begin + span / num_ranks * (rank + 1)
                      + std::min<size_t>(rank + 1, span % num_ranks);
    const Posix_File file(input_file);
    const auto local_runs = scan_range(file, block_structure, index.data_begin,
                                       index.file_size, lo, hi);

    // Gather all runs on all ranks, in rank (= file) order.
    std::vector<uint64_t> send;
    send.reserve(3 * local_runs.size());
    for(const auto &run : local_runs)
      {
        send.push_back(run.block_index);
        send.push_back(run.begin);
        send.push_back(run.end);
      }
    ASSERT(send.size() <= static_cast<size_t>(INT_MAX));
    const int send_count = send.size();
    std::vector<int> counts(num_ranks);
    MPI_Allgather(&send_count, 1, MPI_INT, counts.data(), 1, MPI_INT,
                  comm.comm);
    std::vector<int> displacements(num_ranks);
    size_t total = 0;
    for(int r = 0; r < num_ranks; ++r)
      {
        displacements.at(r) = total;
        total += counts.at(r);
        ASSERT(total <= static_cast<size_t>(INT_MAX));
      }
    std::vector<uint64_t> all(total);
    MPI_Allgatherv(send.data(), send_count, MPI_UINT64_T, all.data(),
                   counts.data(), displacements.data(), MPI_UINT64_T,
                   comm.comm);

    for(size_t k = 0; k < total; k += 3)
      {
        const SDPA_Block_Run run{all.at(k), all.at(k + 1), all.at(k + 2)};
        if(!index.runs.empty() && index.runs.back().block_index == run.block_index
           && index.runs.back().end == run.begin)
          index.runs.back().end = run.end;
        else
          {
            ASSERT(index.runs.empty() || index.runs.back().end <= run.begin);
            index.runs.push_back(run);
          }
      }
    return index;
  }

  SDPA_File_Parse_Result read_sdpa_indexed(
    const std::filesystem::path &input_file, const SDPA_Sparse_Index &index,
    const std::function<bool(size_t matrix_index)> &should_parse_block)
  {
    std::ifstream header_stream(input_file,
                                std::ios_base::in | std::ios_base::binary);
    ASSERT(header_stream.good(), "Could not open ", input_file);

    const Posix_File file(input_file);
    const Read_Entries read_entries
      = [&](const std::function<void(SDPA_Matrix_Entry &&)> &accept,
            const std::function<bool(const SDPA_Matrix_Entry &)> &wanted) {
          ASSERT_EQUAL(static_cast<size_t>(header_stream.tellg()),
                       index.data_begin, "SDPA header does not match index");
          std::vector<char> buffer(index_chunk_bytes);
          for(const auto &run : index.runs)
            {
              if(!should_parse_block(run.block_index))
                continue;
              size_t pos = run.begin;
              while(pos < run.end)
                {
                  const size_t want
                    = std::min(buffer.size(), run.end - pos);
                  ASSERT_EQUAL(file.read(buffer.data(), want, pos), want,
                               "Short read from ", input_file);
                  size_t n = want;
                  if(pos + want < run.end)
                    {
                      // Cut after the last complete line.
                      const char *last = nullptr;
                      for(size_t k = want; k > 0; --k)
                        if(buffer[k - 1] == '\n')
                          {
                            last = buffer.data() + k;
                            break;
                          }
                      ASSERT(last != nullptr, "SDPA line longer than ",
                             index_chunk_bytes, " bytes at offset ", pos);
                      n = last - buffer.data();
                    }
                  const bool blank = std::all_of(
                    buffer.data(), buffer.data() + n, [](const char c) {
                      return std::isspace(static_cast<unsigned char>(c));
                    });
                  if(!blank)
                    {
                      boost::iostreams::stream<boost::iostreams::array_source>
                        chunk(buffer.data(), n);
                      read_sparse_matrices(chunk, accept, wanted);
                    }
                  pos += n;
                }
            }
        };
    return read_sdpa(header_stream, SDPA_File_Type::sparse, should_parse_block,
                     read_entries);
  }

  std::vector<El::BigFloat>
  read_sdpa_objective(const std::filesystem::path &input_file)
  {
    std::optional<SDPA_Block_Structure> block_structure;
    std::vector<El::BigFloat> c_objective;
    read_header(input_file, block_structure, c_objective);
    return c_objective;
  }

  void read_sdpa_lines(
    const std::filesystem::path &input_file, const SDPA_Sparse_Index &index,
    const size_t run_begin, const size_t run_end, const size_t begin,
    const size_t end,
    const std::function<void(size_t block_index, size_t matrix_index,
                             size_t i, size_t j, const El::BigFloat &value)>
      &accept)
  {
    ASSERT(run_begin <= begin && begin <= end && end <= run_end,
           DEBUG_STRING(run_begin), DEBUG_STRING(begin), DEBUG_STRING(end),
           DEBUG_STRING(run_end));
    if(begin == end)
      return;
    const SDPA_Block_Structure block_structure(
      index.m_dim, index.input_block_sizes.size(), index.input_block_sizes);
    const Posix_File file(input_file);
    std::vector<char> small(4096);

    // Position of the first '\n' at or after pos (or run_end), plus one.
    const auto after_newline = [&](size_t pos) {
      while(pos < run_end)
        {
          const size_t n
            = file.read(small.data(), std::min(small.size(), run_end - pos),
                        pos);
          ASSERT(n > 0);
          const auto *nl
            = static_cast<const char *>(std::memchr(small.data(), '\n', n));
          if(nl != nullptr)
            return pos + (nl - small.data()) + 1;
          pos += n;
        }
      return run_end;
    };
    // Lines starting in [begin, end): from the first line start >= begin
    // to the end of the line that contains end - 1.
    size_t start = begin;
    if(begin > run_begin)
      {
        char previous;
        ASSERT_EQUAL(file.read(&previous, 1, begin - 1), 1);
        if(previous != '\n')
          start = after_newline(begin);
      }
    if(start >= end)
      return;
    const size_t stop = after_newline(end - 1);

    const std::function<void(SDPA_Matrix_Entry &&)> accept_entry
      = [&](SDPA_Matrix_Entry &&sdpa_entry) {
          const Result_Matrix_Entry entry(sdpa_entry, block_structure);
          accept(entry.sdp_block_index, entry.vector_index, entry.i, entry.j,
                 entry.value);
        };
    const std::function<bool(const SDPA_Matrix_Entry &)> wanted
      = [](const SDPA_Matrix_Entry &) { return true; };

    // Sized to the range: this is called once per run and slice.
    std::vector<char> buffer(std::min(index_chunk_bytes, stop - start));
    size_t pos = start;
    while(pos < stop)
      {
        const size_t want = std::min(buffer.size(), stop - pos);
        ASSERT_EQUAL(file.read(buffer.data(), want, pos), want,
                     "Short read from ", input_file);
        size_t n = want;
        if(pos + want < stop)
          {
            const char *last = nullptr;
            for(size_t k = want; k > 0; --k)
              if(buffer[k - 1] == '\n')
                {
                  last = buffer.data() + k;
                  break;
                }
            ASSERT(last != nullptr, "SDPA line longer than ",
                   index_chunk_bytes, " bytes at offset ", pos);
            n = last - buffer.data();
          }
        const bool blank
          = std::all_of(buffer.data(), buffer.data() + n, [](const char c) {
              return std::isspace(static_cast<unsigned char>(c));
            });
        if(!blank)
          {
            boost::iostreams::stream<boost::iostreams::array_source> chunk(
              buffer.data(), n);
            read_sparse_matrices(chunk, accept_entry, wanted);
          }
        pos += n;
      }
  }
}
