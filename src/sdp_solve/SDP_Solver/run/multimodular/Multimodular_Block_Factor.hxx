#pragma once

#include "multimodular_blas.hxx"
#include "sdpb_util/Timers/Timers.hxx"

#include <El.hpp>

#include <memory>
#include <vector>

// Multimodular Cholesky factor of one Schur complement block S (an n x n
// DistMatrix on its grid), and the solve P = L^{-1} B with it, as
// El::Cholesky(LOWER, ...) followed by El::Trsm(LEFT, LOWER, NORMAL, ...).
// See Readme.md.
//
// The lower triangle of L is held by every rank of the grid: if the grid is
// on one node, as one copy per node in an MPI-3 shared-memory window
// (Shared_Window_Array), otherwise as a copy per rank.
class Multimodular_Block_Factor
{
public:
  // Cholesky levels of at least this many rows are split over the grid
  static constexpr int distributed_min_size = 256;

  // allow_shared = false: a copy per rank even on one node (for tests)
  Multimodular_Block_Factor(const El::DistMatrix<El::BigFloat> &schur,
                            const Multimodular_Blas_Config &config,
                            bool allow_shared = true);
  ~Multimodular_Block_Factor();
  Multimodular_Block_Factor(const Multimodular_Block_Factor &) = delete;
  Multimodular_Block_Factor &
  operator=(const Multimodular_Block_Factor &) = delete;

  // cholesky := L. As El::Cholesky, the strictly upper triangle of
  // cholesky keeps the values of schur.
  void cholesky(const El::DistMatrix<El::BigFloat> &schur,
                El::DistMatrix<El::BigFloat> &cholesky, Timers &timers);
  // B := L^{-1} B (B on the same grid as schur)
  void solve(El::DistMatrix<El::BigFloat> &B, Timers &timers);
  // The lower triangle of L, whole, as this rank sees it (after cholesky()).
  const El::Matrix<El::BigFloat> &lower() const;

  // #184, streaming Q: each rank solves its columns of L^{-1} B panel by panel against
  // rows [r0, r1) x columns [0, r1) of L (rows.matrix, after cholesky()).
  struct Lower_Rows
  {
    El::Matrix<El::BigFloat> matrix;
    std::vector<__mpf_struct> structs; // matrix's storage after release_lower()
  };
  void lower_rows(int r0, int r1, Lower_Rows &rows);
  // #184: a shared factor's lower() is n x n mpf structs on every rank, pointing into
  // the window: more memory per block than the window's one copy of the limbs. Free
  // them; lower() is then unusable, lower_rows() and copy_lower() build their own.
  void release_lower();
  // The lower triangle of cholesky := L (resized to n x n; the strictly upper
  // triangle is left as is).
  void copy_lower(El::DistMatrix<El::BigFloat> &cholesky);

  bool shared() const;

private:
  struct Storage;
  std::unique_ptr<Storage> storage;
  const Multimodular_Blas_Config config;
};
