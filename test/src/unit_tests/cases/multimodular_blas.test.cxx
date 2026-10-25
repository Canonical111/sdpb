#include "catch2/catch_amalgamated.hpp"

#include "sdp_solve/SDP_Solver/run/multimodular/Multimodular_Block_Factor.hxx"
#include "sdp_solve/SDP_Solver/run/multimodular/multimodular_blas.hxx"
#include "unit_tests/util/util.hxx"

#include <El.hpp>

#include <cmath>
#include <random>
#include <vector>

// Tests for the multimodular BLAS routines:
// - multimodular GEMM, Cholesky and Trsm against El's BigFloat routines;
// - Multimodular_Block_Factor (as used by sdpb --multimodularTrsm) on
//   distributed blocks, with the factor in shared memory or on every rank.

namespace
{
  // log2 |x|, or -infinity for x = 0
  double log2_abs(const El::BigFloat &x)
  {
    if(mpf_sgn(x.gmp_float.get_mpf_t()) == 0)
      return -INFINITY;
    long e;
    const double d = mpf_get_d_2exp(&e, x.gmp_float.get_mpf_t());
    return std::log2(std::abs(d)) + e;
  }

  // Random BigFloat in (-1, 1) times 2^e, e uniform in [-max_shift, 0]
  // (entries of very different sizes in one matrix)
  El::BigFloat random_scaled(std::mt19937_64 &rng, const int max_shift)
  {
    El::BigFloat x = Test_Util::random_bigfloat();
    const int shift = std::uniform_int_distribution<int>(0, max_shift)(rng);
    mpf_div_2exp(x.gmp_float.get_mpf_t(), x.gmp_float.get_mpf_t(), shift);
    return x;
  }

  El::Matrix<El::BigFloat>
  random_scaled_matrix(const int height, const int width, std::mt19937_64 &rng,
                       const int max_shift)
  {
    El::Matrix<El::BigFloat> result(height, width);
    for(int j = 0; j < width; ++j)
      for(int i = 0; i < height; ++i)
        result.Set(i, j, random_scaled(rng, max_shift));
    return result;
  }

  // Well-conditioned symmetric positive definite matrix G G^T + n I
  El::Matrix<El::BigFloat> random_spd(const int n)
  {
    const auto G = Test_Util::random_matrix(n, n);
    El::Matrix<El::BigFloat> S(n, n);
    El::Gemm(El::NORMAL, El::TRANSPOSE, El::BigFloat(1), G, G, El::BigFloat(0),
             S);
    for(int i = 0; i < n; ++i)
      S(i, i) += El::BigFloat(n);
    return S;
  }

  El::BigFloat max_abs(const El::Matrix<El::BigFloat> &A)
  {
    El::BigFloat result = 0;
    for(int j = 0; j < A.Width(); ++j)
      for(int i = 0; i < A.Height(); ++i)
        result = std::max(result, El::Abs(A.CRef(i, j)));
    return result;
  }

  // max_ij |A_ij - B_ij| / max_ij |B_ij|, as log2; only i >= j if lower
  double
  log2_rel_diff(const El::Matrix<El::BigFloat> &A,
                const El::Matrix<El::BigFloat> &B, const bool lower = false)
  {
    REQUIRE(A.Height() == B.Height());
    REQUIRE(A.Width() == B.Width());
    El::BigFloat max_diff = 0, max_B = 0;
    for(int j = 0; j < A.Width(); ++j)
      for(int i = lower ? j : 0; i < A.Height(); ++i)
        {
          max_diff = std::max(max_diff, El::Abs(A.CRef(i, j) - B.CRef(i, j)));
          max_B = std::max(max_B, El::Abs(B.CRef(i, j)));
        }
    if(max_diff == El::BigFloat(0))
      return -INFINITY;
    return log2_abs(max_diff) - log2_abs(max_B);
  }

  bool
  bitwise_equal(const El::Matrix<El::BigFloat> &A,
                const El::Matrix<El::BigFloat> &B, const bool lower = false)
  {
    if(A.Height() != B.Height() || A.Width() != B.Width())
      return false;
    for(int j = 0; j < A.Width(); ++j)
      for(int i = lower ? j : 0; i < A.Height(); ++i)
        if(mpf_cmp(A.CRef(i, j).gmp_float.get_mpf_t(),
                   B.CRef(i, j).gmp_float.get_mpf_t())
           != 0)
          return false;
    return true;
  }

  // Backward error of a Cholesky factor: max |S - L L^T| / max |S|, log2
  double log2_cholesky_backward_error(const El::Matrix<El::BigFloat> &S,
                                      const El::Matrix<El::BigFloat> &L_full)
  {
    El::Matrix<El::BigFloat> L(L_full);
    El::MakeTrapezoidal(El::LOWER, L);
    El::Matrix<El::BigFloat> LLT(S.Height(), S.Width());
    El::Gemm(El::NORMAL, El::TRANSPOSE, El::BigFloat(1), L, L, El::BigFloat(0),
             LLT);
    return log2_rel_diff(LLT, S);
  }

  // Backward error of a solve: max |B - L X| / (max |L| max |X|), log2
  double log2_solve_backward_error(const El::Matrix<El::BigFloat> &L_full,
                                   const El::Matrix<El::BigFloat> &X,
                                   const El::Matrix<El::BigFloat> &B)
  {
    El::Matrix<El::BigFloat> L(L_full);
    El::MakeTrapezoidal(El::LOWER, L);
    El::Matrix<El::BigFloat> R(B);
    El::Gemm(El::NORMAL, El::NORMAL, El::BigFloat(-1), L, X, El::BigFloat(1),
             R);
    El::BigFloat max_R = 0;
    for(int j = 0; j < R.Width(); ++j)
      for(int i = 0; i < R.Height(); ++i)
        max_R = std::max(max_R, El::Abs(R.CRef(i, j)));
    if(max_R == El::BigFloat(0))
      return -INFINITY;
    return log2_abs(max_R) - log2_abs(max_abs(L)) - log2_abs(max_abs(X));
  }

}

TEST_CASE("multimodular_gemm_sub")
{
  if(El::mpi::Rank() != 0)
    return;
  std::mt19937_64 rng(2);
  const int precision = El::gmp::Precision();
  Multimodular_Blas_Config config;
  config.column_chunk = 64;

  // m, k, n: one chunk and several chunks of columns
  const auto [m, k, n]
    = GENERATE(std::make_tuple(1, 1, 1), std::make_tuple(7, 33, 5),
               std::make_tuple(40, 70, 150));
  const auto orientation = GENERATE(El::NORMAL, El::TRANSPOSE);
  DYNAMIC_SECTION("m=" << m << " k=" << k << " n=" << n
                       << " transpose=" << (orientation == El::TRANSPOSE))
  {
    const auto A = random_scaled_matrix(m, k, rng, 100);
    const auto B = orientation == El::NORMAL
                     ? random_scaled_matrix(k, n, rng, 100)
                     : random_scaled_matrix(n, k, rng, 100);
    const auto C0 = Test_Util::random_matrix(m, n);

    // El's BigFloat Gemm: its rounding errors, ~k 2^-precision max|A| max|B|
    // (|A|, |B| < 1), dominate the difference
    El::Matrix<El::BigFloat> C_ref(C0);
    El::Gemm(El::NORMAL, orientation, El::BigFloat(-1), A, B, El::BigFloat(1),
             C_ref);

    auto C = C0;
    Multimodular_Blas_Timings timings;
    multimodular_gemm_sub(C, A, orientation, B, false, config, timings);
    // Multimodular: |dC_ij| <~ k 2^-(precision + guard) max|A_i.| max|B_.j|
    // plus the rounding of C_ij
    CHECK(log2_rel_diff(C, C_ref) < -precision + std::log2(k) + 4);

    SECTION("FLINT CRT gives the same bits")
    {
      auto C_flint = C0;
      auto flint_config = config;
      flint_config.flint_crt = true;
      multimodular_gemm_sub(C_flint, A, orientation, B, false, flint_config,
                            timings);
      REQUIRE(bitwise_equal(C, C_flint));
    }
    SECTION("lower_only leaves the strictly upper part")
    {
      auto C_lower = C0;
      multimodular_gemm_sub(C_lower, A, orientation, B, true, config, timings);
      for(int j = 0; j < n; ++j)
        for(int i = 0; i < m; ++i)
          {
            const auto &expected = i >= j ? C.CRef(i, j) : C0.CRef(i, j);
            REQUIRE(mpf_cmp(C_lower.CRef(i, j).gmp_float.get_mpf_t(),
                            expected.gmp_float.get_mpf_t())
                    == 0);
          }
    }
  }
}

TEST_CASE("multimodular Cholesky and Trsm vs El")
{
  if(El::mpi::Rank() != 0)
    return;
  const int precision = El::gmp::Precision();
  // 1: base case only; 32/33: around the base size; 100: recursion
  const int n = GENERATE(1, 32, 33, 100);
  DYNAMIC_SECTION("n=" << n)
  {
    const auto S = random_spd(n);
    El::Matrix<El::BigFloat> L_el(S);
    El::Cholesky(El::LOWER, L_el);

    Multimodular_Blas_Config config;
    Multimodular_Blas_Timings timings;
    El::Matrix<El::BigFloat> L(S);
    multimodular_cholesky_lower(L, config, timings);

    // Backward error at the level of El's, and the same factor up to
    // rounding (S is well conditioned)
    const double backward_el = log2_cholesky_backward_error(S, L_el);
    const double backward_mm = log2_cholesky_backward_error(S, L);
    CAPTURE(backward_el);
    CAPTURE(backward_mm);
    CHECK(backward_mm < -precision + 8);
    CHECK(log2_rel_diff(L, L_el, true) < -precision + 10);
    // The strictly upper triangle is untouched, as in El::Cholesky
    for(int j = 0; j < n; ++j)
      for(int i = 0; i < j; ++i)
        REQUIRE(L.CRef(i, j) == S.CRef(i, j));

    SECTION("Trsm")
    {
      // Few columns (left-looking) and many (recursive)
      const int width = GENERATE_COPY(1, n / 2 + 1, 3 * n);
      const bool left_looking = GENERATE(true, false);
      DYNAMIC_SECTION("width=" << width << " left_looking=" << left_looking)
      {
        const auto B = Test_Util::random_matrix(n, width);
        El::Matrix<El::BigFloat> X_el(B);
        El::Trsm(El::LEFT, El::LOWER, El::NORMAL, El::NON_UNIT,
                 El::BigFloat(1), L, X_el);

        auto trsm_config = config;
        trsm_config.left_looking = left_looking;
        El::Matrix<El::BigFloat> X(B);
        multimodular_trsm_left_lower(L, X, trsm_config, timings);

        CHECK(log2_solve_backward_error(L, X, B) < -precision + 8);
        CHECK(log2_rel_diff(X, X_el) < -precision + 10);

        auto flint_config = trsm_config;
        flint_config.flint_crt = true;
        El::Matrix<El::BigFloat> X_flint(B);
        multimodular_trsm_left_lower(L, X_flint, flint_config, timings);
        REQUIRE(bitwise_equal(X, X_flint));
      }
    }
  }
}

TEST_CASE("Multimodular_Block_Factor")
{
  // A block on all ranks (one node in tests), or on each rank alone
  const bool all_ranks = GENERATE(true, false);
  // above Multimodular_Block_Factor::distributed_min_size: the Cholesky
  // is split over the grid
  const int n = GENERATE(20, 300);
  const int width = 40;
  DYNAMIC_SECTION("all_ranks=" << all_ranks << " n=" << n)
  {
    const El::Grid grid(all_ranks ? El::mpi::COMM_WORLD : El::mpi::COMM_SELF);
    const int precision = El::gmp::Precision();

    // Same S and B on all ranks of the grid: generated on its root
    El::DistMatrix<El::BigFloat, El::CIRC, El::CIRC> S_root(n, n, grid),
      B_root(n, width, grid);
    if(S_root.CrossRank() == S_root.Root())
      {
        S_root.Matrix() = random_spd(n);
        B_root.Matrix() = Test_Util::random_matrix(n, width);
      }
    const El::DistMatrix<El::BigFloat> S(S_root), B(B_root);

    // El reference
    El::DistMatrix<El::BigFloat> L_el(S), X_el(B);
    El::Cholesky(El::LOWER, L_el);
    El::Trsm(El::LEFT, El::LOWER, El::NORMAL, El::NON_UNIT, El::BigFloat(1),
             L_el, X_el);

    Timers timers;
    std::vector<El::Matrix<El::BigFloat>> L_results, X_results;
    for(const bool allow_shared : {true, false})
      {
        CAPTURE(allow_shared);
        Multimodular_Block_Factor factor(S, Multimodular_Blas_Config(),
                                         allow_shared);
        REQUIRE(factor.shared() == allow_shared);
        El::DistMatrix<El::BigFloat> L(grid), X(B);
        factor.cholesky(S, L, timers);
        factor.solve(X, timers);

        El::DistMatrix<El::BigFloat, El::STAR, El::STAR> L_star(L), X_star(X),
          L_el_star(L_el), X_el_star(X_el), S_star(S), B_star(B);
        // Upper triangle as El::Cholesky leaves it
        for(int j = 0; j < n; ++j)
          for(int i = 0; i < j; ++i)
            REQUIRE(L_star.Matrix().CRef(i, j) == S_star.Matrix().CRef(i, j));
        CHECK(log2_cholesky_backward_error(S_star.Matrix(), L_star.Matrix())
              < -precision + 8);
        CHECK(log2_rel_diff(L_star.Matrix(), L_el_star.Matrix(), true)
              < -precision + 10);
        CHECK(log2_solve_backward_error(L_star.Matrix(), X_star.Matrix(),
                                        B_star.Matrix())
              < -precision + 8);
        CHECK(log2_rel_diff(X_star.Matrix(), X_el_star.Matrix())
              < -precision + 10);
        L_results.push_back(L_star.Matrix());
        X_results.push_back(X_star.Matrix());
      }
    // Shared-memory and per-rank factors: the same arithmetic
    REQUIRE(bitwise_equal(L_results.at(0), L_results.at(1)));
    REQUIRE(bitwise_equal(X_results.at(0), X_results.at(1)));
  }
}
