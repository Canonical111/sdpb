#include "SDPA_File_Parse_Result.hxx"
#include "SDPA_Block_Structure.hxx"
#include "catch2/catch_amalgamated.hpp"
#include "sdpb_util/assert.hxx"

#include <El.hpp>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <unordered_map>
#include <string>
#include <cstring>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/device/file.hpp>
#include <boost/iostreams/filter/gzip.hpp>

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
      IStream &is, const std::function<void(SDPA_Matrix_Entry &&)> &accept)
    {
      // Read all matrix entries in the file
      while(!is.eof())
        {
          SDPA_Matrix_Entry sdpa_entry;
          read_sparse_matrix_entry(is, sdpa_entry);
          std::ws(is);
          accept(std::move(sdpa_entry));
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
  template <class IStream>
  SDPA_File_Parse_Result
  read_sdpa(IStream &is, const SDPA_File_Type file_type,
            const std::function<bool(size_t block_index)> &should_parse_block)
  {
    SDPA_File_Parse_Result result;
    const SDPA_Block_Structure block_structure = read_block_structure(is);

    result.c_objective = read_c(is, block_structure.m_dim);
    ASSERT_EQUAL(block_structure.m_dim, result.c_objective.size());

    // #184: keep what the file is -- a triplet list -- instead of materializing every F_i
    // densely ((m+1) zero-filled dim^2 matrices per block). Both symmetric halves are emitted,
    // matching what the dense reader wrote into its matrices.
    for(size_t index = 0; index < block_structure.num_solver_blocks; ++index)
      {
        if(should_parse_block(index))
          result.parsed_block_entries.try_emplace(index);
      }

    // Interning key: the exact mpf state (size, exponent, limbs). Two values with the
    // same key are the same bits, so sharing one BigFloat changes no arithmetic.
    std::unordered_map<std::string, uint32_t> value_slots;
    const auto intern = [&](const El::BigFloat &value) -> uint32_t {
      const auto *f = value.gmp_float.get_mpf_t();
      const int size = f->_mp_size;
      const long exp = f->_mp_exp;
      const size_t limb_bytes = std::abs(size) * sizeof(mp_limb_t);
      std::string key(sizeof(size) + sizeof(exp) + limb_bytes, '\0');
      std::memcpy(key.data(), &size, sizeof(size));
      std::memcpy(key.data() + sizeof(size), &exp, sizeof(exp));
      std::memcpy(key.data() + sizeof(size) + sizeof(exp), f->_mp_d,
                  limb_bytes);
      const auto [it, inserted] = value_slots.try_emplace(
        std::move(key), static_cast<uint32_t>(result.F_values.size()));
      if(inserted)
        result.F_values.push_back(value);
      return it->second;
    };

    const auto accept_sdpa_entry = [&](SDPA_Matrix_Entry &&sdpa_entry) {
      const Result_Matrix_Entry entry(sdpa_entry, block_structure);
      const auto i = static_cast<int32_t>(entry.i);
      const auto j = static_cast<int32_t>(entry.j);

      if(!should_parse_block(entry.sdp_block_index))
        return;

      const auto constraint = static_cast<uint32_t>(entry.vector_index);
      const auto slot = intern(entry.value);
      auto &entries = result.parsed_block_entries.at(entry.sdp_block_index);
      entries.push_back({constraint, i, j, slot});
      if(i != j && file_type == SDPA_File_Type::sparse)
        entries.push_back({constraint, j, i, slot});
    };

    switch(file_type)
      {
      case sparse: read_sparse_matrices(is, accept_sdpa_entry); break;
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
        RUNTIME_ERROR("Unexpected data at the end of file, position: ", pos,
                      ", next line: ", line);
      }

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
}
