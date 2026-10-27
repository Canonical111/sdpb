#pragma once

// Compact storage of one block of the free variable matrix B (lab,
// dsd-lab-playground/bootstrap-speed#164, sdpb --sparseB).
//
// B is 9-44% dense per block on the Ising+T SDPs (structural zeros: a
// polynomial matrix element that vanishes for a given column), and every
// zero stored in an El::DistMatrix<El::BigFloat> costs a full BigFloat
// (184 B at 1024 bits). This keeps only the nonzero elements of the
// rank's local part of the [MC,MR] DistMatrix, by local column:
//
//   local column jLoc has its nonzeros at local rows
//   rows[col_start[jLoc] .. col_start[jLoc+1]), with values values[...]
//
// Every operation that needs B densely gets one block back as a
// DistMatrix with the same grid, alignment and values (to_dense), so
// results are bitwise identical to dense storage.

#include <El.hpp>

#include <cstdint>
#include <vector>

struct Sparse_Block_Matrix
{
  El::Int height = 0, width = 0;
  El::Int local_height = 0, local_width = 0;
  int col_align = 0, row_align = 0;
  std::vector<size_t> col_start;
  std::vector<int32_t> rows;
  std::vector<El::BigFloat> values;

  Sparse_Block_Matrix() = default;
  // Keeps the nonzero elements of dense's local part.
  explicit Sparse_Block_Matrix(const El::DistMatrix<El::BigFloat> &dense);

  // dense := this block, on dense's grid (which must be the grid of the
  // matrix this was made from).
  void to_dense(El::DistMatrix<El::BigFloat> &dense) const;
  // dense := columns [j0, j1) of this block, as a view of them would be distributed;
  // j0 must be a multiple of the grid width, so that the local columns are a range of
  // this block's.
  void to_dense_columns(El::DistMatrix<El::BigFloat> &dense, El::Int j0,
                        El::Int j1) const;
  // local := local columns [first_column, first_column + local.Width()), all local rows
  void to_dense_local(El::Matrix<El::BigFloat> &local,
                      El::Int first_column) const;

  size_t num_nonzeros_local() const { return values.size(); }
  // Elements that the dense local matrix would hold
  size_t dense_size_local() const { return local_height * local_width; }
};
