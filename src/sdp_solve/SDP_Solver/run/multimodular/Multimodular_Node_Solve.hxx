#pragma once

#include "multimodular_blas.hxx"
#include "sdpb_util/Timers/Timers.hxx"

#include <El.hpp>

#include <vector>

// How the solves P_b = L_b^{-1} B_b of --multimodularTrsm are distributed.
// Set by the environment variable SDPB_MULTIMODULAR_NODE_SCHEDULE
// (experimental, see Readme.md):
//   (unset) grid: each block on the ranks of its own grid (default)
//   static        node-wide jobs (block, column range), assigned to the
//                 node's ranks by LPT (LPT_scheduling, as bigint_syrk's
//                 BLAS jobs) after all Cholesky factors are done
//   dynamic       the same jobs, taken by whichever rank of the node is
//                 free, as soon as the block's factor is done
enum class Multimodular_Node_Schedule
{
  grid,
  static_lpt,
  dynamic
};

Multimodular_Node_Schedule multimodular_node_schedule_from_env();

// For each local block k: cholesky[k] := L_k (Multimodular_Block_Factor on
// the block's grid), then B[k] := L_k^{-1} B[k]. The solves of all blocks
// whose grid is on one node are split into jobs (block, column range) that
// any rank of the node may run, with L and B in node-wide shared windows.
// Collective over node_comm. block_timings_ms(global index) gets the
// Cholesky time and the time of the block's solve jobs, on whichever rank
// they ran.
void multimodular_node_cholesky_solve(
  const El::mpi::Comm &node_comm, const std::vector<size_t> &block_indices,
  const std::vector<El::DistMatrix<El::BigFloat>> &schur,
  std::vector<El::DistMatrix<El::BigFloat>> &cholesky,
  std::vector<El::DistMatrix<El::BigFloat>> &B,
  Multimodular_Node_Schedule schedule, const Multimodular_Blas_Config &config,
  Timers &timers, El::Matrix<int32_t> &block_timings_ms);
