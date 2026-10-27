#include "Sparse_Block_Matrix.hxx"

#include "sdpb_util/assert.hxx"

#include <limits>

Sparse_Block_Matrix::Sparse_Block_Matrix(
  const El::DistMatrix<El::BigFloat> &dense)
    : height(dense.Height()),
      width(dense.Width()),
      local_height(dense.LocalHeight()),
      local_width(dense.LocalWidth()),
      col_align(dense.ColAlign()),
      row_align(dense.RowAlign())
{
  ASSERT(local_height <= std::numeric_limits<int32_t>::max(),
         DEBUG_STRING(local_height));
  const El::BigFloat zero(0);
  const auto &local = dense.LockedMatrix();

  size_t num_nonzeros = 0;
  for(El::Int jLoc = 0; jLoc < local_width; ++jLoc)
    for(El::Int iLoc = 0; iLoc < local_height; ++iLoc)
      if(local.CRef(iLoc, jLoc) != zero)
        ++num_nonzeros;

  col_start.reserve(local_width + 1);
  rows.reserve(num_nonzeros);
  values.reserve(num_nonzeros);
  col_start.push_back(0);
  for(El::Int jLoc = 0; jLoc < local_width; ++jLoc)
    {
      for(El::Int iLoc = 0; iLoc < local_height; ++iLoc)
        {
          const auto &value = local.CRef(iLoc, jLoc);
          if(value != zero)
            {
              rows.push_back(iLoc);
              values.push_back(value);
            }
        }
      col_start.push_back(rows.size());
    }
}

void Sparse_Block_Matrix::to_dense(El::DistMatrix<El::BigFloat> &dense) const
{
  // As El's Copy into a new DistMatrix: unconstrained alignment, root 0
  dense.Empty();
  dense.AlignCols(col_align, false);
  dense.AlignRows(row_align, false);
  dense.Resize(height, width);
  ASSERT_EQUAL(dense.LocalHeight(), local_height);
  ASSERT_EQUAL(dense.LocalWidth(), local_width);
  El::Zero(dense);
  auto &local = dense.Matrix();
  for(El::Int jLoc = 0; jLoc < local_width; ++jLoc)
    for(size_t k = col_start[jLoc]; k < col_start[jLoc + 1]; ++k)
      local.Set(rows[k], jLoc, values[k]);
}

void Sparse_Block_Matrix::to_dense_columns(El::DistMatrix<El::BigFloat> &dense,
                                           const El::Int j0,
                                           const El::Int j1) const
{
  const El::Int grid_width = dense.Grid().Width();
  ASSERT(j0 % grid_width == 0, DEBUG_STRING(j0), DEBUG_STRING(grid_width));
  ASSERT(0 <= j0 && j0 <= j1 && j1 <= width, DEBUG_STRING(j0),
         DEBUG_STRING(j1), DEBUG_STRING(width));
  // The view of columns [j0, j1) keeps the row alignment (j0 % grid_width == 0).
  dense.Empty();
  dense.AlignCols(col_align, false);
  dense.AlignRows(row_align, false);
  dense.Resize(height, j1 - j0);
  ASSERT_EQUAL(dense.LocalHeight(), local_height);
  to_dense_local(dense.Matrix(), j0 / grid_width);
}

void Sparse_Block_Matrix::to_dense_local(El::Matrix<El::BigFloat> &local,
                                         const El::Int first_column) const
{
  ASSERT_EQUAL(local.Height(), local_height);
  ASSERT(first_column + local.Width() <= local_width,
         DEBUG_STRING(first_column), DEBUG_STRING(local.Width()),
         DEBUG_STRING(local_width));
  El::Zero(local);
  for(El::Int jj = 0; jj < local.Width(); ++jj)
    {
      const El::Int jLoc = first_column + jj;
      for(size_t k = col_start[jLoc]; k < col_start[jLoc + 1]; ++k)
        local.Set(rows[k], jj, values[k]);
    }
}
