#include "sdp_solve/Block_Diagonal_Matrix.hxx"
#include "sdp_solve/SDP.hxx"
#include "sdp_solve/SDP_Solver/run/bigint_syrk/BigInt_Shared_Memory_Syrk_Context.hxx"
#include "sdp_solve/SDP_Solver/run/bigint_syrk/Matrix_Normalizer.hxx"
#include "sdp_solve/memory_estimates.hxx"
#include "sdpb_util/Timers/Timers.hxx"

#include <cstdlib>
#include <fstream>

// Debug/benchmark hook: if SDPB_DUMP_BLOCK=<global block index> and
// SDPB_DUMP_DIR=<dir> are set, write that block's Schur complement and
// free_var_matrix at iteration SDPB_DUMP_ITER (default 1) to
// <dir>/{schur,B}_<block>_iter<k>.bin: int64 height, int64 width,
// int64 precision, then the El::BigFloat::Serialize() bytes, column-major.
namespace
{
  void dump_matrix(const El::DistMatrix<El::BigFloat> &matrix,
                   const std::string &path)
  {
    El::DistMatrix<El::BigFloat, El::STAR, El::STAR> local(matrix);
    if(matrix.Grid().Rank() != 0)
      return;
    std::ofstream out(path, std::ios::binary);
    int64_t header[3] = {local.Height(), local.Width(),
                         static_cast<int64_t>(El::gmp::Precision())};
    out.write(reinterpret_cast<const char *>(header), sizeof(header));
    std::vector<El::byte> buffer;
    for(int j = 0; j < local.Width(); ++j)
      for(int i = 0; i < local.Height(); ++i)
        {
          const auto &value = local.GetLocalCRef(i, j);
          buffer.resize(value.SerializedSize());
          value.Serialize(buffer.data());
          out.write(reinterpret_cast<const char *>(buffer.data()),
                    buffer.size());
        }
  }

  void maybe_dump_block(const El::DistMatrix<El::BigFloat> &schur,
                        const El::DistMatrix<El::BigFloat> &B,
                        size_t global_block_index, int iteration)
  {
    const char *block_env = std::getenv("SDPB_DUMP_BLOCK");
    const char *dir_env = std::getenv("SDPB_DUMP_DIR");
    if(block_env == nullptr || dir_env == nullptr)
      return;
    const char *iter_env = std::getenv("SDPB_DUMP_ITER");
    int dump_iteration = iter_env == nullptr ? 1 : std::atoi(iter_env);
    if(std::stoul(block_env) != global_block_index
       || iteration != dump_iteration)
      return;
    std::string suffix = std::to_string(global_block_index) + "_iter"
                         + std::to_string(iteration) + ".bin";
    dump_matrix(schur, std::string(dir_env) + "/schur_" + suffix);
    dump_matrix(B, std::string(dir_env) + "/B_" + suffix);
  }
}

// schur_off_diagonal = L^{-1} B
void initialize_schur_off_diagonal(
  const Environment &env, const SDP &sdp, const Block_Info &block_info,
  const Block_Diagonal_Matrix &schur_complement,
  Block_Matrix &schur_off_diagonal,
  Block_Diagonal_Matrix &schur_complement_cholesky, Timers &timers,
  El::Matrix<int32_t> &block_timings_ms, const Verbosity verbosity)
{
  static int iteration = 0;
  ++iteration;
  schur_off_diagonal.blocks.clear();
  schur_off_diagonal.blocks.reserve(schur_complement_cholesky.blocks.size());

  size_t num_blocks_local = schur_complement_cholesky.blocks.size();
  for(size_t block = 0; block < num_blocks_local; ++block)
    {
      const auto global_block_index = block_info.block_indices.at(block);
      auto block_index_string = std::to_string(global_block_index);
      {
        maybe_dump_block(schur_complement.blocks[block],
                         sdp.free_var_matrix.blocks[block], global_block_index,
                         iteration);
        Scoped_Timer cholesky_timer(timers, "cholesky_" + block_index_string);
        schur_complement_cholesky.blocks[block]
          = schur_complement.blocks[block];

        try
          {
            Cholesky(El::UpperOrLowerNS::LOWER,
                     schur_complement_cholesky.blocks[block]);
          }
        catch(std::exception &e)
          {
            RUNTIME_ERROR(
              "Error when computing Cholesky decomposition of block_",
              global_block_index, ": ", e.what());
          }
        block_timings_ms(global_block_index, 0)
          += cholesky_timer.elapsed_milliseconds();
      }

      // schur_off_diagonal = L^{-1} B
      Scoped_Timer solve_timer(timers, "solve_" + block_index_string);

      schur_off_diagonal.blocks.push_back(sdp.free_var_matrix.blocks[block]);
      El::Trsm(El::LeftOrRightNS::LEFT, El::UpperOrLowerNS::LOWER,
               El::OrientationNS::NORMAL, El::UnitOrNonUnitNS::NON_UNIT,
               El::BigFloat(1), schur_complement_cholesky.blocks[block],
               schur_off_diagonal.blocks[block]);
      block_timings_ms(global_block_index, 0)
        += solve_timer.elapsed_milliseconds();
    }
  if(verbosity >= Verbosity::trace)
    {
      print_allocation_message_per_node(
        env, "schur_off_diagonal", get_allocated_bytes(schur_off_diagonal));
      // schur_complement_cholesky was already allocated in SDP_Solver::step()
    }
}

// Check that Q_ii = 2^2N, where N = normalizer.precision.
// This follows from the fact that columns of P are normalized and multiplied by 2^N
void check_normalized_Q_diagonal(const El::DistMatrix<El::BigFloat> &Q,
                                 const Matrix_Normalizer &normalizer,
                                 Timers &timers)
{
  Scoped_Timer timer(timers, "check_diagonal");
  for(int iLoc = 0; iLoc < Q.LocalHeight(); ++iLoc)
    for(int jLoc = 0; jLoc < Q.LocalWidth(); ++jLoc)
      {
        int i = Q.GlobalRow(iLoc);
        int j = Q.GlobalCol(jLoc);
        if(i == j)
          {
            auto value = Q.GetLocal(iLoc, jLoc); // should be 2^2N
            auto should_be_one = value >> 2 * normalizer.precision;

            auto diff = El::Abs(should_be_one - El::BigFloat(1));
            // diff should be equal to zero up to some precision.
            // We cannot control rounding errors exactly,
            // so we (conservatively) require that at least N/2 bits are correct.
            auto eps = El::BigFloat(1) >> normalizer.precision / 2;
            ASSERT(diff < eps,
                   "Normalized Q should have ones on diagonal. For i = ", i,
                   ": Q_ii = ", should_be_one, ", |Q_ii - 1| = ", diff,
                   ", eps = ", eps);
          }
      }
}

// Q = P^T P = (L^{-1} B)^T (L^{-1} B) = schur_off_diagonal^T schur_off_diagonal
void syrk_Q(const Environment &env, Block_Matrix &schur_off_diagonal,
            BigInt_Shared_Memory_Syrk_Context &bigint_syrk_context,
            El::DistMatrix<El::BigFloat> &Q, Timers &timers,
            El::Matrix<int32_t> &block_timings_ms, const Verbosity verbosity)
{
  Scoped_Timer syrk_timer(timers, "syrk");
  std::vector<El::DistMatrix<El::BigFloat>> &P_blocks
    = schur_off_diagonal.blocks;

  // Normalize P columns and multiply by 2^N
  int block_width = Q.Width();
  Scoped_Timer normalizer_ctor_timer(timers, "Matrix_Normalizer_ctor");
  Matrix_Normalizer normalizer(P_blocks, block_width, El::gmp::Precision(),
                               El::mpi::COMM_WORLD);
  if(verbosity >= Verbosity::trace)
    {
      print_allocation_message_per_node(
        env, "Matrix_Normalizer",
        get_allocated_bytes(normalizer.column_norms));
    }
  normalizer_ctor_timer.stop();
  {
    Scoped_Timer normalizer_timer(timers, "normalize_P");
    normalizer.normalize_and_shift_P_blocks(P_blocks);
  }

  // Calculate Q = P^T P
  auto uplo = El::UPPER;
  bigint_syrk_context.bigint_syrk_blas(uplo, P_blocks, Q, timers,
                                       block_timings_ms);

  // Check that Q_ii = 2^2N
  check_normalized_Q_diagonal(Q, normalizer, timers);

  // Remove normalization
  Scoped_Timer restore_timer(timers, "restore_P_Q");
  normalizer.restore_P_blocks(P_blocks);
  normalizer.restore_Q(uplo, Q);
}

void compute_Q(const Environment &env, const SDP &sdp,
               const Block_Info &block_info,
               const Block_Diagonal_Matrix &schur_complement,
               Block_Matrix &schur_off_diagonal,
               Block_Diagonal_Matrix &schur_complement_cholesky,
               BigInt_Shared_Memory_Syrk_Context &bigint_syrk_context,
               El::DistMatrix<El::BigFloat> &Q, Timers &timers,
               El::Matrix<int32_t> &block_timings_ms,
               const Verbosity verbosity)
{
  Scoped_Timer timer(timers, "Q");

  initialize_schur_off_diagonal(env, sdp, block_info, schur_complement,
                                schur_off_diagonal, schur_complement_cholesky,
                                timers, block_timings_ms, verbosity);
  syrk_Q(env, schur_off_diagonal, bigint_syrk_context, Q, timers,
         block_timings_ms, verbosity);
}