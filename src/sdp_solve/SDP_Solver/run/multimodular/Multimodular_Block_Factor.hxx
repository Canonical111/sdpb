#pragma once

#include "multimodular_blas.hxx"
#include "sdpb_util/Timers/Timers.hxx"

#include <El.hpp>

#include <memory>

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
  // dense_width >= 0: B holds some columns of a matrix of this width (the
  // others are zero; sdpb --compressLinvB). The algorithm is chosen as for
  // the full matrix, so each column is computed as in its dense solve.
  void solve(El::DistMatrix<El::BigFloat> &B, Timers &timers,
             El::Int dense_width = -1);

  bool shared() const;

private:
  struct Storage;
  std::unique_ptr<Storage> storage;
  const Multimodular_Blas_Config config;
};
