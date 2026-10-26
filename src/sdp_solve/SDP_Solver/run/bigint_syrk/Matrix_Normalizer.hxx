#pragma once

#include <El.hpp>
#include <boost/noncopyable.hpp>

// Helps to normalize columns of P and multiply them by 2^precision,
// in order to use bigint arithmetics (fmpz_t, see FLINT library)
// for calculating Q := P^T * P
// Usage:
// - Create Matrix_Normalizer
// - normalize_and_shift_P()
// - calculate Q := P^T * P
// - restore_Q()
class Matrix_Normalizer : boost::noncopyable
{
public:
  const int precision;
  const std::vector<El::BigFloat> column_norms;

  // TMatrix can be El::Matrix<El::BigFloat> or El::DistMatrix<El::BigFloat>>
  template <class TMatrix>
  Matrix_Normalizer(const TMatrix &P_matrix, int precision,
                    El::mpi::Comm comm);

  // P_matrix_blocks: horizontal bands of P_matrix stored on this rank
  // column_maps (optional, sdpb --compressLinvB): band b holds the columns
  // (*column_maps)[b] of P (-1: a zero padding column); the other columns
  // of the band are zero.
  template <class TMatrix>
  Matrix_Normalizer(
    const std::vector<TMatrix> &P_matrix_blocks, int P_matrix_width,
    int precision, El::mpi::Comm comm,
    const std::vector<std::vector<El::Int>> *column_maps = nullptr);

  // normalize columns of P matrix (or its horizontal band)
  // column_map: as above, for this band
  template <class TMatrix>
  void normalize_and_shift_P(TMatrix &P_block,
                             const std::vector<El::Int> *column_map
                             = nullptr);
  template <class TMatrix_Blocks>
  void normalize_and_shift_P_blocks(
    TMatrix_Blocks &P_matrix_blocks,
    const std::vector<std::vector<El::Int>> *column_maps = nullptr);

  template <class TMatrix>
  void
  restore_P(TMatrix &P_block, const std::vector<El::Int> *column_map = nullptr);
  template <class TMatrix_Blocks>
  void restore_P_blocks(
    TMatrix_Blocks &P_matrix_blocks,
    const std::vector<std::vector<El::Int>> *column_maps = nullptr);

  template <class TMatrix>
  void restore_Q(El::UpperOrLower uplo, TMatrix &Q_matrix);
};
