#include "Matrix_Normalizer.hxx"

#include "sdpb_util/assert.hxx"

// Helper functions providing common interface for El::Matrix<T> and
// El::DistMatrix<T>
namespace
{
  template <class T> int local_height(const El::Matrix<T> &matrix)
  {
    return matrix.Height();
  }
  template <class T> int local_height(const El::DistMatrix<T> &matrix)
  {
    return matrix.LocalHeight();
  }

  template <class T> int local_width(const El::Matrix<T> &matrix)
  {
    return matrix.Width();
  }
  template <class T> int local_width(const El::DistMatrix<T> &matrix)
  {
    return matrix.LocalWidth();
  }

  template <class T> int global_row(const El::Matrix<T> &matrix, int iLoc)
  {
    return iLoc;
  }
  template <class T> int global_row(const El::DistMatrix<T> &matrix, int iLoc)
  {
    return matrix.GlobalRow(iLoc);
  }

  template <class T> int global_col(const El::Matrix<T> &matrix, int jLoc)
  {
    return jLoc;
  }
  template <class T> int global_col(const El::DistMatrix<T> &matrix, int jLoc)
  {
    return matrix.GlobalCol(jLoc);
  }

  template <class T>
  const T &get_local_cref(const El::Matrix<T> &matrix, int iLoc, int jLoc)
  {
    return matrix.CRef(iLoc, jLoc);
  }
  template <class T>
  const T &get_local_cref(const El::DistMatrix<T> &matrix, int iLoc, int jLoc)
  {
    return matrix.GetLocalCRef(iLoc, jLoc);
  }

  template <class T>
  void set_local(El::Matrix<T> &matrix, int iLoc, int jLoc, const T &value)
  {
    matrix.Set(iLoc, jLoc, value);
  }
  template <class T>
  void set_local(El::DistMatrix<T> &matrix, int iLoc, int jLoc, const T &value)
  {
    matrix.SetLocal(iLoc, jLoc, value);
  }
}

// Helper functions for calculating matrix column norms
// TMatrix is either Matrix<El::BigFloat> or DistMatrix<El::BigFloat>
namespace
{
  // For each column:
  // Sum of squares of elements stored in this rank
  // Column of P held by global column j of a band (identity without a map;
  // -1 for a padding column)
  El::Int mapped_col(const std::vector<El::Int> *column_map, const int j)
  { return column_map == nullptr ? j : column_map->at(j); }

  template <class TMatrix>
  void add_local_column_norms_squared(
    std::vector<El::BigFloat> &local_norms_squared, const TMatrix &matrix,
    const std::vector<El::Int> *column_map = nullptr)
  {
    if(column_map == nullptr)
      {
        if(local_norms_squared.size() != matrix.Width())
          El::Output("local_norms_squared.size() != matrix.Width(): ",
                     local_norms_squared.size(), " ", matrix.Width());
        ASSERT_EQUAL(local_norms_squared.size(), matrix.Width());
      }
    else
      ASSERT_EQUAL(column_map->size(), matrix.Width());
    // A zero element adds exactly zero, so skipping B's zero columns
    // leaves every sum bitwise unchanged.
    for(int iLoc = 0; iLoc < local_height(matrix); ++iLoc)
      for(int jLoc = 0; jLoc < local_width(matrix); ++jLoc)
        {
          const auto &value = get_local_cref(matrix, iLoc, jLoc);
          const El::Int j = mapped_col(column_map, global_col(matrix, jLoc));
          if(j < 0)
            continue;
          local_norms_squared.at(j) += value * value;
        }
  }

  template <class TMatrix>
  std::vector<El::BigFloat>
  calculate_column_norms(const TMatrix &matrix, El::mpi::Comm comm)
  {
    // For each column,
    // we calculate norm squared for each block
    // and accumulate them for all blocks from all ranks

    std::vector<El::BigFloat> local_norms_squared(matrix.Width(), 0);
    std::vector<El::BigFloat> column_norms(matrix.Width(), 0);

    add_local_column_norms_squared(local_norms_squared, matrix);

    El::mpi::AllReduce(local_norms_squared.data(), local_norms_squared.size(),
                       comm);

    for(size_t j = 0; j < matrix.Width(); ++j)
      {
        column_norms.at(j) = Sqrt(local_norms_squared.at(j));
      }
    return column_norms;
  }

  template <class TMatrix>
  std::vector<El::BigFloat>
  calculate_column_norms(const std::vector<TMatrix> &input_blocks,
                         size_t block_width, El::mpi::Comm comm,
                         const std::vector<std::vector<El::Int>> *column_maps)
  {
    // For each column,
    // we calculate norm squared for each block
    // and accumulate them for all blocks from all ranks

    std::vector<El::BigFloat> local_norms_squared(block_width, 0);
    std::vector<El::BigFloat> column_norms(block_width, 0);

    if(column_maps != nullptr)
      ASSERT_EQUAL(column_maps->size(), input_blocks.size());
    for(size_t b = 0; b < input_blocks.size(); ++b)
      {
        add_local_column_norms_squared(
          local_norms_squared, input_blocks[b],
          column_maps == nullptr ? nullptr : &column_maps->at(b));
      }

    El::mpi::AllReduce(local_norms_squared.data(), local_norms_squared.size(),
                       comm);

    for(size_t j = 0; j < block_width; ++j)
      {
        column_norms.at(j) = Sqrt(local_norms_squared.at(j));
      }
    return column_norms;
  }
}

// Matrix_Normalizer implementation

// constructors

template <class TMatrix>
Matrix_Normalizer::Matrix_Normalizer(const TMatrix &P_matrix, int precision,
                                     El::mpi::Comm comm)
    : precision(precision),
      column_norms(calculate_column_norms(P_matrix, comm))
{}
template Matrix_Normalizer::Matrix_Normalizer(const El::Matrix<El::BigFloat> &,
                                              int, El::mpi::Comm);
template Matrix_Normalizer::Matrix_Normalizer(
  const El::DistMatrix<El::BigFloat> &, int, El::mpi::Comm);

template <class TMatrix>
Matrix_Normalizer::Matrix_Normalizer(
  const std::vector<TMatrix> &P_matrix_blocks, int P_matrix_width,
  int precision, El::mpi::Comm comm,
  const std::vector<std::vector<El::Int>> *column_maps)
    : precision(precision),
      column_norms(calculate_column_norms(P_matrix_blocks, P_matrix_width,
                                          comm, column_maps))
{}
template Matrix_Normalizer::Matrix_Normalizer(
  const std::vector<El::Matrix<El::BigFloat>> &, int, int, El::mpi::Comm,
  const std::vector<std::vector<El::Int>> *);
template Matrix_Normalizer::Matrix_Normalizer(
  const std::vector<El::DistMatrix<El::BigFloat>> &, int, int,
  El::mpi::Comm comm, const std::vector<std::vector<El::Int>> *);

// normalize_and_shift_P

template <class TMatrix>
void Matrix_Normalizer::normalize_and_shift_P(
  TMatrix &P_block, const std::vector<El::Int> *column_map)
{
  if(column_map == nullptr)
    ASSERT_EQUAL(P_block.Width(), column_norms.size());
  else
    ASSERT_EQUAL(P_block.Width(), column_map->size());
  El::BigFloat normalized_value;
  for(int jLoc = 0; jLoc < local_width(P_block); ++jLoc)
    {
      const El::Int j = mapped_col(column_map, global_col(P_block, jLoc));
      // padding column: zero
      if(j < 0)
        continue;
      const auto &norm = column_norms.at(j);
      if(norm == El::BigFloat(0))
        continue;
      for(int iLoc = 0; iLoc < local_height(P_block); ++iLoc)
        {
          normalized_value = get_local_cref(P_block, iLoc, jLoc) / norm;
          set_local(P_block, iLoc, jLoc, normalized_value << precision);
        }
    }
}
template void
Matrix_Normalizer::normalize_and_shift_P(El::Matrix<El::BigFloat> &,
                                         const std::vector<El::Int> *);
template void
Matrix_Normalizer::normalize_and_shift_P(El::DistMatrix<El::BigFloat> &,
                                         const std::vector<El::Int> *);

template <class TMatrix_Blocks>
void Matrix_Normalizer::normalize_and_shift_P_blocks(
  TMatrix_Blocks &P_matrix_blocks,
  const std::vector<std::vector<El::Int>> *column_maps)
{
  if(column_maps != nullptr)
    ASSERT_EQUAL(column_maps->size(), P_matrix_blocks.size());
  for(size_t b = 0; b < P_matrix_blocks.size(); ++b)
    normalize_and_shift_P(P_matrix_blocks[b], column_maps == nullptr
                                                ? nullptr
                                                : &column_maps->at(b));
}
template void Matrix_Normalizer::normalize_and_shift_P_blocks(
  std::vector<El::Matrix<El::BigFloat>> &P_matrix_blocks,
  const std::vector<std::vector<El::Int>> *);
template void Matrix_Normalizer::normalize_and_shift_P_blocks(
  std::vector<El::DistMatrix<El::BigFloat>> &P_matrix_blocks,
  const std::vector<std::vector<El::Int>> *);

// restore P

template <class TMatrix>
void Matrix_Normalizer::restore_P(TMatrix &P_block,
                                  const std::vector<El::Int> *column_map)
{
  if(column_map == nullptr)
    ASSERT_EQUAL(P_block.Width(), column_norms.size());
  else
    ASSERT_EQUAL(P_block.Width(), column_map->size());
  El::BigFloat value;
  for(int jLoc = 0; jLoc < local_width(P_block); ++jLoc)
    {
      const El::Int j = mapped_col(column_map, global_col(P_block, jLoc));
      if(j < 0)
        continue;
      const auto &norm = column_norms.at(j);
      if(norm == El::BigFloat(0))
        continue;
      for(int iLoc = 0; iLoc < local_height(P_block); ++iLoc)
        {
          value = get_local_cref(P_block, iLoc, jLoc) >> precision;
          set_local(P_block, iLoc, jLoc, value * norm);
        }
    }
}
template void
Matrix_Normalizer::restore_P(El::DistMatrix<El::BigFloat> &P_block,
                             const std::vector<El::Int> *);
template void Matrix_Normalizer::restore_P(El::Matrix<El::BigFloat> &P_block,
                                           const std::vector<El::Int> *);

template <class TMatrix_Blocks>
void Matrix_Normalizer::restore_P_blocks(
  TMatrix_Blocks &P_matrix_blocks,
  const std::vector<std::vector<El::Int>> *column_maps)
{
  if(column_maps != nullptr)
    ASSERT_EQUAL(column_maps->size(), P_matrix_blocks.size());
  for(size_t b = 0; b < P_matrix_blocks.size(); ++b)
    restore_P(P_matrix_blocks[b],
              column_maps == nullptr ? nullptr : &column_maps->at(b));
}
template void Matrix_Normalizer::restore_P_blocks(
  std::vector<El::Matrix<El::BigFloat>> &P_matrix_blocks,
  const std::vector<std::vector<El::Int>> *);
template void Matrix_Normalizer::restore_P_blocks(
  std::vector<El::DistMatrix<El::BigFloat>> &P_matrix_blocks,
  const std::vector<std::vector<El::Int>> *);

// restore_Q

template <class TMatrix>
void Matrix_Normalizer::restore_Q(El::UpperOrLower uplo, TMatrix &Q_matrix)
{
  ASSERT_EQUAL(Q_matrix.Height(), column_norms.size());
  ASSERT_EQUAL(Q_matrix.Width(), column_norms.size());
  El::BigFloat restored_value;
  for(int iLoc = 0; iLoc < local_height(Q_matrix); ++iLoc)
    for(int jLoc = 0; jLoc < local_width(Q_matrix); ++jLoc)
      {
        int i = global_row(Q_matrix, iLoc);
        int j = global_col(Q_matrix, jLoc);
        if(uplo == El::UPPER && i > j)
          continue;
        if(uplo == El::LOWER && i < j)
          continue;

        const auto &normshifted_value = get_local_cref(Q_matrix, iLoc, jLoc);
        restored_value = (normshifted_value >> 2 * precision)
                         * column_norms.at(i) * column_norms.at(j);
        set_local(Q_matrix, iLoc, jLoc, restored_value);
      }
}
template void Matrix_Normalizer::restore_Q(El::UpperOrLower uplo,
                                           El::Matrix<El::BigFloat> &);
template void Matrix_Normalizer::restore_Q(El::UpperOrLower uplo,
                                           El::DistMatrix<El::BigFloat> &);
