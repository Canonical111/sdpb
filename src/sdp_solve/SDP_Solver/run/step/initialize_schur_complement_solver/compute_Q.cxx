#include "sdp_solve/Block_Diagonal_Matrix.hxx"
#include "sdp_solve/SDP.hxx"
#include "sdp_solve/SDP_Solver/run/bigint_syrk/BigInt_Shared_Memory_Syrk_Context.hxx"
#include "sdp_solve/SDP_Solver/run/bigint_syrk/Matrix_Normalizer.hxx"
#include "sdp_solve/SDP_Solver/run/multimodular/multimodular_blas.hxx"
#include "sdp_solve/memory_estimates.hxx"
#include "sdpb_util/Timers/Timers.hxx"

#include <boost/noncopyable.hpp>

#include <cstdlib>
#include <memory>
#include <fstream>
#include <iostream>

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

  // Multimodular per-block Cholesky and Trsm (lab #145), enabled by
  // SDPB_MULTIMODULAR_TRSM=1. Optional SDPB_MULTIMODULAR_BASE,
  // SDPB_MULTIMODULAR_GUARD, SDPB_MULTIMODULAR_CHUNK,
  // SDPB_MULTIMODULAR_FLINT_CRT, SDPB_MULTIMODULAR_LEFT_LOOKING,
  // SDPB_MULTIMODULAR_GROWTH, SDPB_MULTIMODULAR_LL_MB,
  // SDPB_MULTIMODULAR_LL_RATIO set the config.
  // Blocks smaller than SDPB_MULTIMODULAR_MIN_SIZE (default 2 * base)
  // use the El routines.
  int env_int(const char *name, int default_value)
  {
    const char *value = std::getenv(name);
    return value == nullptr ? default_value : std::atoi(value);
  }
  bool use_multimodular_trsm()
  {
    static const bool result = env_int("SDPB_MULTIMODULAR_TRSM", 0) != 0;
    return result;
  }
  const Multimodular_Blas_Config &multimodular_config()
  {
    static const Multimodular_Blas_Config config = [] {
      Multimodular_Blas_Config c;
      c.base_size = env_int("SDPB_MULTIMODULAR_BASE", c.base_size);
      c.guard_bits = env_int("SDPB_MULTIMODULAR_GUARD", c.guard_bits);
      c.column_chunk = env_int("SDPB_MULTIMODULAR_CHUNK", c.column_chunk);
      c.flint_crt = env_int("SDPB_MULTIMODULAR_FLINT_CRT", 0) != 0;
      c.left_looking = env_int("SDPB_MULTIMODULAR_LEFT_LOOKING", 1) != 0;
      if(const char *ratio = std::getenv("SDPB_MULTIMODULAR_LL_RATIO"))
        c.left_looking_max_ratio = std::atof(ratio);
      c.growth_bits = env_int("SDPB_MULTIMODULAR_GROWTH", c.growth_bits);
      c.left_looking_bytes
        = size_t(env_int("SDPB_MULTIMODULAR_LL_MB",
                         c.left_looking_bytes >> 20))
          << 20;
      return c;
    }();
    return config;
  }
  int multimodular_min_size()
  {
    static const int result = env_int("SDPB_MULTIMODULAR_MIN_SIZE",
                                      2 * multimodular_config().base_size);
    return result;
  }

  int multimodular_distributed_min_size()
  {
    static const int result = env_int("SDPB_MULTIMODULAR_DIST_MIN", 256);
    return result;
  }

  // Cholesky (lower, in place) of the n x n matrix A, a view of a matrix
  // that every rank of the grid holds in full, with the same values.
  // In each recursion level, A = [A11 *; A21 A22]:
  //   A11 := chol(A11)                       (recursive)
  //   A21 := L21, L21^T = L11^{-1} A21^T     (columns of A21^T split)
  //   A22 -= L21 L21^T, lower triangle       (columns of A22 split)
  //   A22 := chol(A22)                       (recursive)
  // The split parts are computed by each rank for its columns
  // ([STAR,VR]: column j on rank j mod g) and then allgathered.
  // Below SDPB_MULTIMODULAR_DIST_MIN (or on one rank) every rank computes
  // the whole factorization.
  void distributed_multimodular_cholesky(El::Matrix<El::BigFloat> &A,
                                         const El::Grid &grid)
  {
    const int n = A.Height();
    const auto &config = multimodular_config();
    Multimodular_Blas_Timings timings;
    if(grid.Size() == 1 || n < multimodular_distributed_min_size())
      {
        multimodular_cholesky_lower(A, config, timings);
        return;
      }
    const int n1 = n / 2, n2 = n - n1;
    const auto top = El::IR(0, n1), bottom = El::IR(n1, n);
    auto A11 = A(top, top);
    auto A21 = A(bottom, top);
    auto A22 = A(bottom, bottom);

    distributed_multimodular_cholesky(A11, grid);

    // L21^T = L11^{-1} A21^T
    {
      El::DistMatrix<El::BigFloat, El::STAR, El::VR> L21_T(n1, n2, grid);
      auto &local = L21_T.Matrix();
      for(int jLoc = 0; jLoc < L21_T.LocalWidth(); ++jLoc)
        {
          const int j = L21_T.GlobalCol(jLoc);
          for(int i = 0; i < n1; ++i)
            local.Set(i, jLoc, A21.CRef(j, i));
        }
      multimodular_trsm_left_lower(A11, local, config, timings);
      El::DistMatrix<El::BigFloat, El::STAR, El::STAR> L21_T_all(L21_T);
      for(int j = 0; j < n2; ++j)
        for(int i = 0; i < n1; ++i)
          A21.Set(j, i, L21_T_all.GetLocalCRef(i, j));
    }

    // A22 -= L21 L21^T, lower triangle
    {
      El::DistMatrix<El::BigFloat, El::STAR, El::VR> A22_columns(n2, n2,
                                                                  grid);
      auto &local = A22_columns.Matrix();
      const int width = A22_columns.LocalWidth();
      El::Matrix<El::BigFloat> L21_rows(width, n1);
      std::vector<int> row_start(width);
      for(int jLoc = 0; jLoc < width; ++jLoc)
        {
          const int j = A22_columns.GlobalCol(jLoc);
          row_start[jLoc] = j;
          for(int i = j; i < n2; ++i)
            local.Set(i, jLoc, A22.CRef(i, j));
          for(int l = 0; l < n1; ++l)
            L21_rows.Set(jLoc, l, A21.CRef(j, l));
        }
      multimodular_gemm_sub(local, A21, El::TRANSPOSE, L21_rows, row_start,
                            config, timings);
      El::DistMatrix<El::BigFloat, El::STAR, El::STAR> A22_all(A22_columns);
      for(int j = 0; j < n2; ++j)
        for(int i = j; i < n2; ++i)
          A22.Set(i, j, A22_all.GetLocalCRef(i, j));
    }

    distributed_multimodular_cholesky(A22, grid);
  }

  // n x n El::Matrix<BigFloat> whose strictly upper elements have minimal
  // precision: they are never used, and this halves the memory of the
  // replicated factor (El::Matrix::Resize would allocate them in full).
  class Lower_BigFloat_Matrix : boost::noncopyable
  {
    El::BigFloat *data = nullptr;
    size_t size = 0;

  public:
    El::Matrix<El::BigFloat> matrix;
    explicit Lower_BigFloat_Matrix(int n) : size(size_t(n) * n)
    {
      data = static_cast<El::BigFloat *>(
        ::operator new(sizeof(El::BigFloat) * size));
      for(int j = 0; j < n; ++j)
        for(int i = 0; i < n; ++i)
          new(data + i + size_t(j) * n)
            El::BigFloat(0, i >= j ? El::gmp::Precision() : 1);
      matrix.Attach(n, n, data, n);
    }
    ~Lower_BigFloat_Matrix()
    {
      matrix.Empty();
      for(size_t k = 0; k < size; ++k)
        data[k].~BigFloat();
      ::operator delete(data);
    }
  };

  // L := the lower triangle of the distributed matrix `schur`, on every rank
  // of its grid. Gathered in column panels, rows j0..n of columns j0..j1 at
  // a time, so the transient buffers stay small and only the lower triangle
  // moves.
  void gather_lower(const El::DistMatrix<El::BigFloat> &schur,
                    El::Matrix<El::BigFloat> &L)
  {
    const int n = schur.Height();
    ASSERT_EQUAL(L.Height(), n);
    const int panel = 64;
    for(int j0 = 0; j0 < n; j0 += panel)
      {
        const int j1 = std::min(n, j0 + panel);
        El::DistMatrix<El::BigFloat, El::STAR, El::STAR> part(
          schur(El::IR(j0, n), El::IR(j0, j1)));
        for(int j = j0; j < j1; ++j)
          for(int i = j; i < n; ++i)
            mpf_set(L.Ref(i, j).gmp_float.get_mpf_t(),
                    part.GetLocalCRef(i - j0, j - j0).gmp_float.get_mpf_t());
      }
  }

  // Every rank of the block's grid gets the lower triangle of the Schur
  // block, computes its Cholesky factor L (distributed_multimodular_cholesky),
  // and writes its own entries of the distributed factor.
  // As El::Cholesky, the strictly upper triangle keeps the input values.
  void multimodular_cholesky(const El::DistMatrix<El::BigFloat> &schur,
                             El::DistMatrix<El::BigFloat> &cholesky,
                             El::Matrix<El::BigFloat> &L, Timers &timers)
  {
    {
      Scoped_Timer timer(timers, "allgather");
      gather_lower(schur, L);
    }
    {
      Scoped_Timer timer(timers, "multimodular");
      distributed_multimodular_cholesky(L, schur.Grid());
    }
    Scoped_Timer timer(timers, "set_local");
    cholesky = schur;
    for(int jLoc = 0; jLoc < cholesky.LocalWidth(); ++jLoc)
      for(int iLoc = 0; iLoc < cholesky.LocalHeight(); ++iLoc)
        {
          const int i = cholesky.GlobalRow(iLoc), j = cholesky.GlobalCol(jLoc);
          if(i >= j)
            cholesky.SetLocal(iLoc, jLoc, L.CRef(i, j));
        }
  }

  // B := L^{-1} B. The columns of B are split among the ranks of the grid
  // ([STAR,VR] distribution); each rank solves for its own columns with
  // its full copy of L.
  void multimodular_solve(const El::Matrix<El::BigFloat> &L,
                          El::DistMatrix<El::BigFloat> &B, Timers &timers)
  {
    Scoped_Timer to_columns_timer(timers, "to_columns");
    El::DistMatrix<El::BigFloat, El::STAR, El::VR> B_columns(B);
    to_columns_timer.stop();
    {
      Scoped_Timer timer(timers, "multimodular");
      Multimodular_Blas_Timings timings;
      multimodular_trsm_left_lower(L, B_columns.Matrix(),
                                   multimodular_config(), timings);
    }
    Scoped_Timer timer(timers, "from_columns");
    B = B_columns;
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
      const bool use_multimodular
        = use_multimodular_trsm()
          && schur_complement.blocks[block].Height() >= multimodular_min_size();
      std::unique_ptr<Lower_BigFloat_Matrix> L_storage;
      if(use_multimodular)
        L_storage = std::make_unique<Lower_BigFloat_Matrix>(
          schur_complement.blocks[block].Height());
      El::Matrix<El::BigFloat> dummy;
      El::Matrix<El::BigFloat> &L_full
        = use_multimodular ? L_storage->matrix : dummy;
      if(use_multimodular && iteration == 1
         && std::getenv("SDPB_MULTIMODULAR_DEBUG") != nullptr)
        std::cerr << El::BuildString(
          "rank ", El::mpi::Rank(), ": block ", global_block_index, " size ",
          schur_complement.blocks[block].Height(), " on ",
          schur_complement.blocks[block].Grid().Size(),
          " ranks: multimodular\n");
      {
        maybe_dump_block(schur_complement.blocks[block],
                         sdp.free_var_matrix.blocks[block], global_block_index,
                         iteration);
        Scoped_Timer cholesky_timer(timers, "cholesky_" + block_index_string);
        try
          {
            if(use_multimodular)
              {
                multimodular_cholesky(schur_complement.blocks[block],
                                      schur_complement_cholesky.blocks[block],
                                      L_full, timers);
              }
            else
              {
                schur_complement_cholesky.blocks[block]
                  = schur_complement.blocks[block];
                Cholesky(El::UpperOrLowerNS::LOWER,
                         schur_complement_cholesky.blocks[block]);
              }
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
      if(use_multimodular)
        {
          multimodular_solve(L_full, schur_off_diagonal.blocks[block],
                             timers);
        }
      else
        {
          El::Trsm(El::LeftOrRightNS::LEFT, El::UpperOrLowerNS::LOWER,
                   El::OrientationNS::NORMAL, El::UnitOrNonUnitNS::NON_UNIT,
                   El::BigFloat(1), schur_complement_cholesky.blocks[block],
                   schur_off_diagonal.blocks[block]);
        }
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