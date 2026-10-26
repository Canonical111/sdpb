#pragma once

// sdpb --compressLinvB (lab, dsd-lab-playground/bootstrap-speed#170):
// L_j^{-1} B_j keeps only the columns where B_j has a nonzero element. A zero
// column of B_j stays exactly zero in L_j^{-1} B_j (forward substitution of a
// zero vector), so dropping it changes no result.
//
// Layout: in an [MC,MR] DistMatrix, column g is held by grid column
// (g + RowAlign) % W. The t-th kept column g with g % W = r goes to
// compressed column t * W + r, which is held by the same grid column, and
// the kept columns of a grid column stay in increasing order. Positions
// that no column takes are zero padding (map entry -1). Hence:
// - the compressed block is filled by local copies, without communication;
// - every local loop over columns (Gemv partial sums, column norms,
//   residues) does the dense matrix's arithmetic without its zero terms.

#include <El.hpp>

#include "sdpb_util/assert.hxx"

#include <vector>

// nonzero[g] != 0 for the columns to keep; W = width of the block's grid
inline std::vector<El::Int>
owner_preserving_column_map(const std::vector<int> &nonzero, const El::Int W)
{
  ASSERT(W > 0, DEBUG_STRING(W));
  std::vector<El::Int> count(W, 0);
  std::vector<El::Int> column_map;
  for(El::Int g = 0; g < El::Int(nonzero.size()); ++g)
    {
      if(!nonzero[g])
        continue;
      const El::Int r = g % W;
      const El::Int k = count[r]++ * W + r;
      if(k >= El::Int(column_map.size()))
        column_map.resize(k + 1, -1);
      column_map[k] = g;
    }
  // El::Trsm solves a single column with Trsv; keep at least two columns
  // when the matrix has two, so that the algorithm is the dense one.
  if(column_map.size() == 1 && nonzero.size() > 1)
    column_map.push_back(-1);
  return column_map;
}

// The columns column_map of dense, on dense's grid and alignments (zero for
// padding columns). Local copies only.
inline El::DistMatrix<El::BigFloat>
compress_columns(const El::DistMatrix<El::BigFloat> &dense,
                 const std::vector<El::Int> &column_map)
{
  El::DistMatrix<El::BigFloat> result(dense.Grid());
  result.AlignCols(dense.ColAlign(), false);
  result.AlignRows(dense.RowAlign(), false);
  result.Resize(dense.Height(), column_map.size());
  El::Zero(result);
  ASSERT_EQUAL(result.LocalHeight(), dense.LocalHeight());
  for(El::Int kLoc = 0; kLoc < result.LocalWidth(); ++kLoc)
    {
      const El::Int g = column_map.at(result.GlobalCol(kLoc));
      if(g < 0)
        continue;
      ASSERT(dense.IsLocalCol(g), DEBUG_STRING(g));
      const El::Int jLoc = dense.LocalCol(g);
      for(El::Int iLoc = 0; iLoc < result.LocalHeight(); ++iLoc)
        result.Matrix().Set(iLoc, kLoc, dense.LockedMatrix().CRef(iLoc, jLoc));
    }
  return result;
}

// The El::Trsm algorithm that El's default would choose for the dense
// L^{-1} B of this width (LEFT, LOWER, NORMAL)
inline El::TrsmAlgorithm
trsm_algorithm_for_dense_width(const El::Int dense_width, const int grid_size)
{
  return dense_width > 5 * grid_size ? El::TRSM_LARGE : El::TRSM_MEDIUM;
}
