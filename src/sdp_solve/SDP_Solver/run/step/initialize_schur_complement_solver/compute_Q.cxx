#include "sdp_solve/Block_Diagonal_Matrix.hxx"
#include "sdp_solve/SDP.hxx"
#include "sdp_solve/SDP_Solver/run/bigint_syrk/BigInt_Shared_Memory_Syrk_Context.hxx"
#include "sdp_solve/SDP_Solver/run/bigint_syrk/Matrix_Normalizer.hxx"
#include "sdp_solve/SDP_Solver/run/multimodular/multimodular_blas.hxx"
#include "sdp_solve/memory_estimates.hxx"
#include "sdpb_util/Timers/Timers.hxx"

#include <boost/noncopyable.hpp>

#include <cstdlib>
#include <cstring>
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
  // SDPB_MULTIMODULAR_LL_RATIO, SDPB_MULTIMODULAR_GEMM_A_MB set the config.
  // Blocks smaller than SDPB_MULTIMODULAR_MIN_SIZE (default 1: none) use
  // the El routines.
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
      c.gemm_A_bytes = size_t(env_int("SDPB_MULTIMODULAR_GEMM_A_MB", 0)) << 20;
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
    // Default: every block. Below the base size the path is just the
    // BigFloat substitution/Cholesky with preallocated temporaries, which is
    // still ~2x faster than El (measured: 65 x 1834 Trsm 0.49-0.57 vs 1.10 s).
    static const int result = env_int("SDPB_MULTIMODULAR_MIN_SIZE", 1);
    return result;
  }

  int multimodular_distributed_min_size()
  {
    static const int result = env_int("SDPB_MULTIMODULAR_DIST_MIN", 256);
    return result;
  }

  // Allgather a [STAR,VR] matrix M in 64-column panels (small transient
  // buffers) and call f(i, j, M(i,j)) on every rank; if lower, only i >= j.
  template <class F>
  void allgather_columns(
    const El::DistMatrix<El::BigFloat, El::STAR, El::VR> &M, bool lower,
    const F &f)
  {
    const int height = M.Height(), width = M.Width();
    const int panel = 64;
    for(int j0 = 0; j0 < width; j0 += panel)
      {
        const int j1 = std::min(width, j0 + panel);
        const int r0 = lower ? std::min(j0, height) : 0;
        El::DistMatrix<El::BigFloat, El::STAR, El::STAR> part(
          M(El::IR(r0, height), El::IR(j0, j1)));
        for(int j = j0; j < j1; ++j)
          for(int i = lower ? j : 0; i < height; ++i)
            f(i, j, part.GetLocalCRef(i - r0, j - j0));
      }
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
      allgather_columns(L21_T, false,
                        [&](int i, int j, const El::BigFloat &value) {
                          A21.Set(j, i, value);
                        });
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
      allgather_columns(A22_columns, true,
                        [&](int i, int j, const El::BigFloat &value) {
                          A22.Set(i, j, value);
                        });
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

  // Lower triangle of an n x n BigFloat matrix in an MPI-3 shared-memory
  // window of the block's grid (all its ranks on one node): one copy per
  // block instead of one per rank. The limbs and the (size, exponent) headers
  // are shared; each rank has its own El::Matrix of mpf structs whose limb
  // pointers point into the window (the window may map at different
  // addresses in different processes). A rank writes elements through its
  // structs (the limbs go to the window), then publish() copies their
  // headers to the window; after sync(), pull() refreshes the other ranks'
  // structs. Strictly upper elements are never used: they point to a dummy
  // zero.
  class Shared_Lower_Matrix : boost::noncopyable
  {
    struct Header
    {
      int size;
      int pad;
      long exp;
    };
    MPI_Comm comm;
    MPI_Win win = MPI_WIN_NULL;
    int n, limbs, rank;
    size_t count;
    Header *headers = nullptr;
    mp_limb_t *limb_data = nullptr;
    El::BigFloat *structs = nullptr;
    mp_limb_t dummy[2] = {0, 0};

    size_t index(int i, int j) const
    {
      return size_t(j) * n - size_t(j) * (j - 1) / 2 + (i - j);
    }
    __mpf_struct *element(int i, int j)
    {
      return structs[i + size_t(j) * n].gmp_float.get_mpf_t();
    }

  public:
    El::Matrix<El::BigFloat> matrix;

    Shared_Lower_Matrix(int n_, MPI_Comm comm_)
        : comm(comm_), n(n_), count(size_t(n_) * (n_ + 1) / 2)
    {
      static_assert(sizeof(El::BigFloat) == sizeof(__mpf_struct),
                    "El::BigFloat must be a bare mpf");
      MPI_Comm_rank(comm, &rank);
      mpf_t probe;
      mpf_init2(probe, El::gmp::Precision());
      limbs = probe->_mp_prec + 1; // mpf writes at most _mp_prec + 1 limbs
      mpf_clear(probe);
      const MPI_Aint bytes
        = rank == 0 ? count * (sizeof(Header) + limbs * sizeof(mp_limb_t))
                    : 0;
      void *base;
      MPI_Win_allocate_shared(bytes, 1, MPI_INFO_NULL, comm, &base, &win);
      MPI_Aint size;
      int disp;
      MPI_Win_shared_query(win, 0, &size, &disp, &base);
      headers = static_cast<Header *>(base);
      limb_data = reinterpret_cast<mp_limb_t *>(headers + count);
      MPI_Win_lock_all(MPI_MODE_NOCHECK, win);
      if(rank == 0)
        std::memset(headers, 0, count * sizeof(Header));
      // mpf structs filled in place (no constructors: limbs live in the
      // window, so ~BigFloat must never run on them)
      structs = static_cast<El::BigFloat *>(
        ::operator new(sizeof(El::BigFloat) * size_t(n) * n));
      for(int j = 0; j < n; ++j)
        for(int i = 0; i < n; ++i)
          {
            __mpf_struct *s = element(i, j);
            s->_mp_size = 0;
            s->_mp_exp = 0;
            if(i >= j)
              {
                s->_mp_prec = limbs - 1;
                s->_mp_d = limb_data + index(i, j) * limbs;
              }
            else
              {
                s->_mp_prec = 1;
                s->_mp_d = dummy;
              }
          }
      matrix.Attach(n, n, structs, n);
      sync();
    }
    ~Shared_Lower_Matrix()
    {
      matrix.Empty();
      ::operator delete(structs);
      MPI_Win_unlock_all(win);
      MPI_Win_free(&win);
    }
    int Rank() const { return rank; }

    void publish(int i, int j)
    {
      const __mpf_struct *s = element(i, j);
      headers[index(i, j)] = {s->_mp_size, 0, s->_mp_exp};
    }
    void pull(int i, int j)
    {
      const Header &h = headers[index(i, j)];
      __mpf_struct *s = element(i, j);
      s->_mp_size = h.size;
      s->_mp_exp = h.exp;
    }
    // publish/pull the lower part of rows [i0, i1) x columns [j0, j1)
    void publish(int i0, int i1, int j0, int j1)
    {
      for(int j = j0; j < j1; ++j)
        for(int i = std::max(i0, j); i < i1; ++i)
          publish(i, j);
    }
    void pull(int i0, int i1, int j0, int j1)
    {
      for(int j = j0; j < j1; ++j)
        for(int i = std::max(i0, j); i < i1; ++i)
          pull(i, j);
    }
    void sync()
    {
      MPI_Win_sync(win);
      MPI_Barrier(comm);
      MPI_Win_sync(win);
    }
  };

  // True if all ranks of comm share a node (MPI shared memory works)
  bool on_one_node(MPI_Comm comm)
  {
    MPI_Comm node;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node);
    int node_size, size;
    MPI_Comm_size(node, &node_size);
    MPI_Comm_size(comm, &size);
    MPI_Comm_free(&node);
    return node_size == size;
  }

  // S := lower triangle of the distributed `schur`: each rank writes the
  // entries it owns.
  void fill_shared(const El::DistMatrix<El::BigFloat> &schur,
                   Shared_Lower_Matrix &S)
  {
    const int n = schur.Height();
    for(int jLoc = 0; jLoc < schur.LocalWidth(); ++jLoc)
      for(int iLoc = 0; iLoc < schur.LocalHeight(); ++iLoc)
        {
          const int i = schur.GlobalRow(iLoc), j = schur.GlobalCol(jLoc);
          if(i < j)
            continue;
          mpf_set(S.matrix.Ref(i, j).gmp_float.get_mpf_t(),
                  schur.GetLocalCRef(iLoc, jLoc).gmp_float.get_mpf_t());
          S.publish(i, j);
        }
    S.sync();
    S.pull(0, n, 0, n);
  }

  // Cholesky of rows/columns [r0, r0+n) of the shared matrix, as
  // distributed_multimodular_cholesky but writing in place instead of
  // allgathering. The parts below SDPB_MULTIMODULAR_DIST_MIN are computed by
  // rank 0 of the grid only.
  void shared_multimodular_cholesky(Shared_Lower_Matrix &S, int r0, int n,
                                    int grid_size)
  {
    const auto &config = multimodular_config();
    Multimodular_Blas_Timings timings;
    const int rank = S.Rank();
    if(grid_size == 1 || n < multimodular_distributed_min_size())
      {
        if(rank == 0)
          {
            auto A = S.matrix(El::IR(r0, r0 + n), El::IR(r0, r0 + n));
            multimodular_cholesky_lower(A, config, timings);
            S.publish(r0, r0 + n, r0, r0 + n);
          }
        S.sync();
        if(rank != 0)
          S.pull(r0, r0 + n, r0, r0 + n);
        return;
      }
    const int n1 = n / 2, n2 = n - n1;
    shared_multimodular_cholesky(S, r0, n1, grid_size);
    const auto rows1 = El::IR(r0, r0 + n1), rows2 = El::IR(r0 + n1, r0 + n);
    const auto A11 = S.matrix(rows1, rows1);
    auto A21 = S.matrix(rows2, rows1);
    auto A22 = S.matrix(rows2, rows2);

    // This rank's rows j of A21 (j = rank mod grid_size)
    std::vector<int> mine;
    for(int j = rank; j < n2; j += grid_size)
      mine.push_back(j);
    const int w = mine.size();

    // L21^T = L11^{-1} A21^T for this rank's rows, written in place
    {
      El::Matrix<El::BigFloat> T(n1, w);
      for(int jj = 0; jj < w; ++jj)
        for(int i = 0; i < n1; ++i)
          T.Set(i, jj, A21.CRef(mine[jj], i));
      multimodular_trsm_left_lower(A11, T, config, timings);
      for(int jj = 0; jj < w; ++jj)
        for(int i = 0; i < n1; ++i)
          {
            A21.Set(mine[jj], i, T.CRef(i, jj));
            S.publish(r0 + n1 + mine[jj], r0 + i);
          }
      S.sync();
      S.pull(r0 + n1, r0 + n, r0, r0 + n1);
    }

    // A22 -= L21 L21^T, this rank's columns, lower part, written in place
    {
      El::Matrix<El::BigFloat> C(n2, w), L21_rows(w, n1);
      std::vector<int> row_start(w);
      for(int jj = 0; jj < w; ++jj)
        {
          const int j = mine[jj];
          row_start[jj] = j;
          for(int i = j; i < n2; ++i)
            C.Set(i, jj, A22.CRef(i, j));
          for(int l = 0; l < n1; ++l)
            L21_rows.Set(jj, l, A21.CRef(j, l));
        }
      multimodular_gemm_sub(C, A21, El::TRANSPOSE, L21_rows, row_start,
                            config, timings);
      for(int jj = 0; jj < w; ++jj)
        {
          const int j = mine[jj];
          for(int i = j; i < n2; ++i)
            {
              A22.Set(i, j, C.CRef(i, jj));
              S.publish(r0 + n1 + i, r0 + n1 + j);
            }
        }
      S.sync();
      S.pull(r0 + n1, r0 + n, r0 + n1, r0 + n);
    }

    shared_multimodular_cholesky(S, r0 + n1, n2, grid_size);
  }

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

  void multimodular_cholesky_shared(const El::DistMatrix<El::BigFloat> &schur,
                                    El::DistMatrix<El::BigFloat> &cholesky,
                                    Shared_Lower_Matrix &S, Timers &timers)
  {
    {
      Scoped_Timer timer(timers, "fill_shared");
      fill_shared(schur, S);
    }
    {
      Scoped_Timer timer(timers, "multimodular");
      shared_multimodular_cholesky(S, 0, schur.Height(),
                                   schur.Grid().Size());
    }
    Scoped_Timer timer(timers, "set_local");
    cholesky = schur;
    for(int jLoc = 0; jLoc < cholesky.LocalWidth(); ++jLoc)
      for(int iLoc = 0; iLoc < cholesky.LocalHeight(); ++iLoc)
        {
          const int i = cholesky.GlobalRow(iLoc), j = cholesky.GlobalCol(jLoc);
          if(i >= j)
            cholesky.SetLocal(iLoc, jLoc, S.matrix.CRef(i, j));
        }
  }

  // B := L^{-1} B. The columns of B are split among the ranks of the grid
  // ([STAR,VR] distribution); each rank solves for its own columns with
  // its full copy of L.
  // B := L^{-1} B for a grid on one node, through an MPI-3 shared-memory
  // window holding B (one copy per block): each rank writes the entries of
  // B it owns ([MC,MR]), solves its columns (j = rank mod g) in place in the
  // window, and reads back the entries it owns. No [STAR,VR] copy and no El
  // redistribution buffers. On one rank, B is solved in place.
  void multimodular_solve_shared(const El::Matrix<El::BigFloat> &L,
                                 El::DistMatrix<El::BigFloat> &B,
                                 Timers &timers)
  {
    Multimodular_Blas_Timings timings;
    if(B.Grid().Size() == 1)
      {
        Scoped_Timer timer(timers, "multimodular");
        multimodular_trsm_left_lower(L, B.Matrix(), multimodular_config(),
                                     timings);
        return;
      }
    struct Header
    {
      int size;
      int pad;
      long exp;
    };
    const int P = B.Height(), N = B.Width();
    MPI_Comm comm = B.Grid().Comm().comm;
    int rank, grid_size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &grid_size);
    mpf_t probe;
    mpf_init2(probe, El::gmp::Precision());
    const int limbs = probe->_mp_prec + 1;
    mpf_clear(probe);
    const size_t count = size_t(P) * N;
    const MPI_Aint bytes
      = rank == 0 ? count * (sizeof(Header) + limbs * sizeof(mp_limb_t)) : 0;
    void *base;
    MPI_Win win;
    MPI_Win_allocate_shared(bytes, 1, MPI_INFO_NULL, comm, &base, &win);
    MPI_Aint size;
    int disp;
    MPI_Win_shared_query(win, 0, &size, &disp, &base);
    Header *headers = static_cast<Header *>(base);
    mp_limb_t *limb_data = reinterpret_cast<mp_limb_t *>(headers + count);
    MPI_Win_lock_all(MPI_MODE_NOCHECK, win);
    auto sync = [&] {
      MPI_Win_sync(win);
      MPI_Barrier(comm);
      MPI_Win_sync(win);
    };
    auto view = [&](size_t index) {
      __mpf_struct v;
      v._mp_prec = limbs - 1;
      v._mp_size = headers[index].size;
      v._mp_exp = headers[index].exp;
      v._mp_d = limb_data + index * limbs;
      return v;
    };

    Scoped_Timer to_shared_timer(timers, "to_shared");
    for(int jLoc = 0; jLoc < B.LocalWidth(); ++jLoc)
      for(int iLoc = 0; iLoc < B.LocalHeight(); ++iLoc)
        {
          const size_t index
            = B.GlobalRow(iLoc) + size_t(B.GlobalCol(jLoc)) * P;
          __mpf_struct v = view(index);
          mpf_set(&v, B.GetLocalCRef(iLoc, jLoc).gmp_float.get_mpf_t());
          headers[index] = {v._mp_size, 0, v._mp_exp};
        }
    sync();
    to_shared_timer.stop();

    {
      Scoped_Timer timer(timers, "multimodular");
      std::vector<int> columns;
      for(int j = rank; j < N; j += grid_size)
        columns.push_back(j);
      const int w = columns.size();
      // mpf structs of this rank's columns, pointing into the window
      El::BigFloat *structs = static_cast<El::BigFloat *>(
        ::operator new(sizeof(El::BigFloat) * std::max<size_t>(1, size_t(P) * w)));
      for(int jj = 0; jj < w; ++jj)
        for(int i = 0; i < P; ++i)
          *structs[i + size_t(jj) * P].gmp_float.get_mpf_t()
            = view(i + size_t(columns[jj]) * P);
      {
        El::Matrix<El::BigFloat> X;
        X.Attach(P, w, structs, P);
        multimodular_trsm_left_lower(L, X, multimodular_config(), timings);
      }
      for(int jj = 0; jj < w; ++jj)
        for(int i = 0; i < P; ++i)
          {
            const __mpf_struct *s
              = structs[i + size_t(jj) * P].gmp_float.get_mpf_t();
            headers[i + size_t(columns[jj]) * P] = {s->_mp_size, 0, s->_mp_exp};
          }
      ::operator delete(structs);
      sync();
    }

    Scoped_Timer from_shared_timer(timers, "from_shared");
    for(int jLoc = 0; jLoc < B.LocalWidth(); ++jLoc)
      for(int iLoc = 0; iLoc < B.LocalHeight(); ++iLoc)
        {
          const __mpf_struct v
            = view(B.GlobalRow(iLoc) + size_t(B.GlobalCol(jLoc)) * P);
          mpf_set(B.Matrix().Ref(iLoc, jLoc).gmp_float.get_mpf_t(), &v);
        }
    MPI_Win_unlock_all(win);
    MPI_Win_free(&win);
  }

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
      // The block's factor, on every rank of its grid: one shared copy per
      // node (SDPB_MULTIMODULAR_SHARED=1, default) or one copy per rank.
      std::unique_ptr<Lower_BigFloat_Matrix> L_storage;
      std::unique_ptr<Shared_Lower_Matrix> L_shared;
      const bool use_shared
        = use_multimodular && env_int("SDPB_MULTIMODULAR_SHARED", 1) != 0
          && on_one_node(schur_complement.blocks[block].Grid().Comm().comm);
      if(use_shared)
        L_shared = std::make_unique<Shared_Lower_Matrix>(
          schur_complement.blocks[block].Height(),
          schur_complement.blocks[block].Grid().Comm().comm);
      else if(use_multimodular)
        L_storage = std::make_unique<Lower_BigFloat_Matrix>(
          schur_complement.blocks[block].Height());
      El::Matrix<El::BigFloat> dummy;
      El::Matrix<El::BigFloat> &L_full
        = use_shared ? L_shared->matrix
                     : (use_multimodular ? L_storage->matrix : dummy);
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
            if(use_shared)
              {
                multimodular_cholesky_shared(
                  schur_complement.blocks[block],
                  schur_complement_cholesky.blocks[block], *L_shared, timers);
              }
            else if(use_multimodular)
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
          if(use_shared)
            multimodular_solve_shared(L_full, schur_off_diagonal.blocks[block],
                                      timers);
          else
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