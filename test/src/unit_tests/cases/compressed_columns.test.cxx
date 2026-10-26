#include "catch2/catch_amalgamated.hpp"

#include "sdp_solve/SDP_Solver/run/bigint_syrk/BigInt_Shared_Memory_Syrk_Context.hxx"
#include "sdp_solve/SDP_Solver/run/bigint_syrk/Matrix_Normalizer.hxx"
#include "sdp_solve/SDP_Solver/run/bigint_syrk/fmpz/Fmpz_Comb.hxx"
#include "sdp_solve/SDP_Solver/run/multimodular/Multimodular_Block_Factor.hxx"
#include "sdp_solve/compressed_columns.hxx"
#include "unit_tests/util/util.hxx"

#include <El.hpp>

#include <random>
#include <vector>

// sdpb --compressLinvB (lab #170): a matrix with zero columns, stored with its
// nonzero columns only (compressed_columns.hxx), must give bitwise the same
// results as the dense matrix in
// - Matrix_Normalizer and the bigint syrk Q = P^T P (column maps), with and
//   without splitting the input residue window;
// - the multimodular and El solves L^{-1} B, whose algorithm depends on the
//   width (the compressed solve chooses by the dense width).

namespace
{
  bool same_bits(const El::BigFloat &a, const El::BigFloat &b)
  {
    const auto *x = a.gmp_float.get_mpf_t();
    const auto *y = b.gmp_float.get_mpf_t();
    if(x->_mp_size != y->_mp_size)
      return false;
    if(x->_mp_size == 0)
      return true;
    if(x->_mp_exp != y->_mp_exp)
      return false;
    const int n = std::abs(x->_mp_size);
    for(int i = 0; i < n; ++i)
      if(x->_mp_d[i] != y->_mp_d[i])
        return false;
    return true;
  }

  // Random columns set to zero: about `fraction` of them; seeded per block
  std::vector<int> random_nonzero_columns(const int width, const double fraction,
                                          const unsigned seed)
  {
    std::mt19937 rng(seed);
    std::bernoulli_distribution zero(fraction);
    std::vector<int> nonzero(width);
    for(auto &x : nonzero)
      x = zero(rng) ? 0 : 1;
    return nonzero;
  }

  // compressed column kLoc of `compressed` equals column map[k] of `dense`,
  // padding columns are zero (same rank holds both)
  void require_columns_equal(const El::DistMatrix<El::BigFloat> &dense,
                             const El::DistMatrix<El::BigFloat> &compressed,
                             const std::vector<El::Int> &map)
  {
    REQUIRE(compressed.LocalHeight() == dense.LocalHeight());
    for(int kLoc = 0; kLoc < compressed.LocalWidth(); ++kLoc)
      {
        const El::Int g = map.at(compressed.GlobalCol(kLoc));
        for(int iLoc = 0; iLoc < compressed.LocalHeight(); ++iLoc)
          {
            CAPTURE(g, iLoc);
            if(g < 0)
              REQUIRE(compressed.GetLocalCRef(iLoc, kLoc) == El::BigFloat(0));
            else
              REQUIRE(same_bits(compressed.GetLocalCRef(iLoc, kLoc),
                                dense.GetLocalCRef(iLoc, dense.LocalCol(g))));
          }
      }
  }
}

TEST_CASE("compressLinvB: owner_preserving_column_map")
{
  // W = 3, columns 0..9 with 1, 2, 4, 5, 9 kept:
  // r=0: 9 -> k=0; r=1: 1, 4 -> k=1, 4; r=2: 2, 5 -> k=2, 5
  const std::vector<int> nonzero{0, 1, 1, 0, 1, 1, 0, 0, 0, 1};
  const auto map = owner_preserving_column_map(nonzero, 3);
  REQUIRE(map == std::vector<El::Int>{9, 1, 2, -1, 4, 5});
  for(size_t k = 0; k < map.size(); ++k)
    if(map[k] >= 0)
      REQUIRE(map[k] % 3 == El::Int(k % 3));
  // all kept: identity
  const auto identity = owner_preserving_column_map(std::vector<int>(7, 1), 3);
  REQUIRE(identity == std::vector<El::Int>{0, 1, 2, 3, 4, 5, 6});
  // one kept column: padded to two
  REQUIRE(owner_preserving_column_map({0, 0, 1}, 1)
          == std::vector<El::Int>{2, -1});
  REQUIRE(owner_preserving_column_map({0, 0, 0}, 2).empty());
}

TEST_CASE("compressLinvB: Matrix_Normalizer and bigint_syrk_blas")
{
  const El::mpi::Comm comm_world = El::mpi::COMM_WORLD;
  {
    El::mpi::Comm comm_shmem;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                        &comm_shmem.comm);
    INFO("The test runs on one node");
    REQUIRE(El::mpi::Congruent(comm_shmem, comm_world));
  }
  const int group_size = GENERATE(1, 2, 3, 4, 6);
  if(comm_world.Size() % group_size != 0)
    return;
  const int width = GENERATE(1, 5, 13);
  // 0: no split; 1: input window split (3 rows per group)
  const int split = GENERATE(0, 1);
  DYNAMIC_SECTION("group_size=" << group_size << " width=" << width
                                << " split=" << split)
  {
    const int bits = El::gmp::Precision();
    El::mpi::Comm group_comm;
    MPI_Comm_split(comm_world.comm, comm_world.Rank() / group_size, 0,
                   &group_comm.comm);
    const int group_index = comm_world.Rank() / group_size;
    const int num_groups = comm_world.Size() / group_size;
    const El::Grid grid(group_comm);

    // Three blocks per group, of heights 7, 20, 3; zero columns: ~0, 50%, 100%
    const std::vector<int> heights{7, 20, 3};
    const std::vector<double> zero_fraction{0.0, 0.5, 1.0};
    std::vector<El::DistMatrix<El::BigFloat>> dense, compressed;
    std::vector<std::vector<El::Int>> maps;
    std::vector<size_t> block_indices;
    for(size_t b = 0; b < heights.size(); ++b)
      {
        const unsigned seed = 1000 * group_index + 17 * b + width;
        const auto nonzero = random_nonzero_columns(
          width, b == 1 ? zero_fraction[b] : zero_fraction[b], seed);
        // Same matrix on all ranks of the group: generated on its root
        El::DistMatrix<El::BigFloat, El::CIRC, El::CIRC> root(heights[b],
                                                             width, grid);
        if(root.CrossRank() == root.Root())
          {
            root.Matrix() = Test_Util::random_matrix(heights[b], width);
            for(int j = 0; j < width; ++j)
              if(!nonzero[j])
                for(int i = 0; i < heights[b]; ++i)
                  root.Matrix().Set(i, j, El::BigFloat(0));
          }
        dense.emplace_back(root);
        maps.push_back(owner_preserving_column_map(nonzero, grid.Width()));
        compressed.push_back(compress_columns(dense.back(), maps.back()));
        require_columns_equal(dense.back(), compressed.back(), maps.back());
        block_indices.push_back(group_index * heights.size() + b);
      }

    const int total_height = num_groups * (7 + 20 + 3);
    const auto num_primes
      = Fmpz_Comb(bits, bits, 1, total_height).num_primes;
    size_t max_shared_memory_bytes = 0;
    if(split == 1)
      max_shared_memory_bytes
        = (size_t(width) * width + 3 * size_t(num_groups) * width) * num_primes
          * sizeof(double);
    std::vector<int> group_comm_sizes(num_groups, group_size);
    std::vector<El::Int> heights_per_group(num_groups, 7 + 20 + 3);

    std::vector<El::DistMatrix<El::BigFloat>> Q_results;
    std::vector<std::vector<El::BigFloat>> norms;
    for(const bool use_maps : {false, true})
      {
        auto &P = use_maps ? compressed : dense;
        const auto *column_maps = use_maps ? &maps : nullptr;
        Matrix_Normalizer normalizer(P, width, bits, El::mpi::COMM_WORLD,
                                     column_maps);
        norms.push_back(normalizer.column_norms);
        normalizer.normalize_and_shift_P_blocks(P, column_maps);
        BigInt_Shared_Memory_Syrk_Context context(
          comm_world, group_index, group_comm_sizes, bits,
          max_shared_memory_bytes, heights_per_group, width, block_indices,
          Verbosity::none);
        El::DistMatrix<El::BigFloat> Q(width, width);
        Timers timers;
        El::Matrix<int32_t> block_timings_ms(num_groups * heights.size(), 1);
        El::Zero(block_timings_ms);
        context.bigint_syrk_blas(El::UPPER, P, Q, timers, block_timings_ms,
                                 column_maps);
        normalizer.restore_Q(El::UPPER, Q);
        normalizer.restore_P_blocks(P, column_maps);
        Q_results.push_back(Q);
      }
    INFO("Column norms, Q and the restored P: bitwise equal");
    REQUIRE(norms[0].size() == norms[1].size());
    for(size_t j = 0; j < norms[0].size(); ++j)
      REQUIRE(same_bits(norms[0][j], norms[1][j]));
    for(int iLoc = 0; iLoc < Q_results[0].LocalHeight(); ++iLoc)
      for(int jLoc = 0; jLoc < Q_results[0].LocalWidth(); ++jLoc)
        REQUIRE(same_bits(Q_results[0].GetLocalCRef(iLoc, jLoc),
                          Q_results[1].GetLocalCRef(iLoc, jLoc)));
    for(size_t b = 0; b < dense.size(); ++b)
      require_columns_equal(dense[b], compressed[b], maps[b]);
  }
}

TEST_CASE("compressLinvB: L^{-1} B, multimodular and El")
{
  const bool all_ranks = GENERATE(true, false);
  const int n = GENERATE(20, 40);
  DYNAMIC_SECTION("all_ranks=" << all_ranks << " n=" << n)
  {
    const El::Grid grid(all_ranks ? El::mpi::COMM_WORLD : El::mpi::COMM_SELF);
    // Dense widths around the multimodular left-looking threshold 2n (per
    // rank) and El's Large/Medium threshold 5p; ~60% zero columns, so the
    // compressed width is on the other side of the thresholds
    for(const int width : {2 * n - 1, 2 * n + 1, 2 * n * grid.Size() + 7,
                           5 * grid.Size() + 3, 1, 2})
      {
        CAPTURE(width);
        El::DistMatrix<El::BigFloat, El::CIRC, El::CIRC> S_root(n, n, grid),
          B_root(n, width, grid);
        const auto nonzero = random_nonzero_columns(width, 0.6, 7 * width + n);
        if(S_root.CrossRank() == S_root.Root())
          {
            const auto G = Test_Util::random_matrix(n, n);
            El::Matrix<El::BigFloat> S(n, n);
            El::Gemm(El::NORMAL, El::TRANSPOSE, El::BigFloat(1), G, G,
                     El::BigFloat(0), S);
            for(int i = 0; i < n; ++i)
              S(i, i) += El::BigFloat(n);
            S_root.Matrix() = S;
            B_root.Matrix() = Test_Util::random_matrix(n, width);
            for(int j = 0; j < width; ++j)
              if(!nonzero[j])
                for(int i = 0; i < n; ++i)
                  B_root.Matrix().Set(i, j, El::BigFloat(0));
          }
        const El::DistMatrix<El::BigFloat> S(S_root), B(B_root);
        const auto map = owner_preserving_column_map(nonzero, B.RowStride());

        Timers timers;
        {
          INFO("multimodular");
          Multimodular_Block_Factor factor(S, Multimodular_Blas_Config());
          El::DistMatrix<El::BigFloat> L(grid), X(B);
          factor.cholesky(S, L, timers);
          auto X_compressed = compress_columns(B, map);
          factor.solve(X, timers);
          factor.solve(X_compressed, timers, width);
          require_columns_equal(X, X_compressed, map);
        }
        {
          INFO("El");
          El::DistMatrix<El::BigFloat> L(S), X(B);
          El::Cholesky(El::LOWER, L);
          auto X_compressed = compress_columns(B, map);
          El::Trsm(El::LEFT, El::LOWER, El::NORMAL, El::NON_UNIT,
                   El::BigFloat(1), L, X);
          El::Trsm(El::LEFT, El::LOWER, El::NORMAL, El::NON_UNIT,
                   El::BigFloat(1), L, X_compressed, false,
                   trsm_algorithm_for_dense_width(width, grid.Size()));
          require_columns_equal(X, X_compressed, map);
        }
      }
  }
}
