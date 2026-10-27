// #184 (bootstrap-speed): S = P^T P without ever holding P.
//
// P = (vec(G_1) ... vec(G_m)), G_p = L_X^{-1} F_p L_Y, is sum(dim^2) x m: at 512 bits on the
// 12_min benchmark it is the largest object of the solve (~0.5 GB packed, ~0.7 GB as BigFloat).
// The default path builds it chunk-of-p by chunk-of-p (the trmm pipeline wants all of F_p),
// while the syrk wants it band-of-rows by band-of-rows (it needs all p for each row). P is the
// transpose buffer between the two orders.
//
// Here the factorization is reassociated so that rows can be produced directly:
//   rows of vec(G_p) for columns J of G_p  =  vec( L_X^{-1} (F_p L_Y[:, J]) ).
// F_p L_Y[:, J] is formed straight from the sparse entries; the triangular product runs only
// over the nonzero rows of that slab. Each band of rows is then column-normalized on its own,
// squared by the existing residue syrk, restored, and added to S. Nothing of size P exists.
//
// Results agree with the default path to working precision but not bit for bit: the
// association order and the per-band normalization both change roundings.
//
// A block spread over several ranks (an MPI group): every rank of the group gets the whole
// L_X^{-1} and L_Y of the block (small) and forms only its own entries of each band.

#include "Compute_S_Context.hxx"
#include "sdp_solve/Block_Matrix/Block_Matrix.hxx"
#include "sdp_solve/SDP_Solver/run/bigint_syrk/Matrix_Normalizer.hxx"
#include "sdp_solve/SDP_Solver/run/bigint_syrk/streaming_syrk.hxx"
#include "sdpa_solve/SDP.hxx"
#include "sdpb_util/Timers/Timers.hxx"

#include <gmp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

namespace Sdpb::Sdpa
{
  bool stream_S_enabled()
  {
    return Streaming_Syrk::enabled();
  }

  namespace
  {
    // A contiguous range of columns [col_begin, col_end) of G_p for one local block.
    // Its rows in P are vec(G_p[:, col_begin:col_end]): dim * (col_end - col_begin) of them.
    struct Slab
    {
      size_t local_block;
      El::Int col_begin;
      El::Int col_end;
    };
    using Band = std::vector<Slab>;

    // Greedy packing in block order: each band holds at most `height` rows and at most one
    // (contiguous) slab per block, so it maps onto one matrix per local block.
    std::vector<Band>
    plan_bands(const std::vector<El::Int> &dims, const El::Int height)
    {
      std::vector<Band> bands;
      Band current;
      El::Int current_height = 0;
      for(size_t block = 0; block < dims.size(); ++block)
        {
          const El::Int dim = dims.at(block);
          ASSERT(dim <= height, "SDPB_STREAM_P: one column of block ", block,
                 " (", dim, " rows) does not fit the syrk band height ",
                 height, "; raise --maxSharedMemory");
          El::Int col = 0;
          while(col < dim)
            {
              const El::Int fit = (height - current_height) / dim;
              if(fit <= 0)
                {
                  bands.push_back(std::move(current));
                  current = {};
                  current_height = 0;
                  continue;
                }
              const El::Int width = std::min(fit, dim - col);
              current.push_back({block, col, col + width});
              current_height += dim * width;
              col += width;
            }
        }
      if(!current.empty())
        bands.push_back(std::move(current));
      return bands;
    }

    using Streaming_Syrk::mul_2exp;

    // Exponent e with |x| < 2^e (x != 0).
    long upper_exponent(mpf_srcptr x)
    {
      long e;
      mpf_get_d_2exp(&e, x);
      return e;
    }

    // mantissa := trunc(|x| 2^(64 limbs - exponent)) for |x| < 2^exponent; returns sign(x).
    int8_t to_fixed(mpf_srcptr x, const long exponent, const size_t limbs,
                    mp_limb_t *mantissa, mpf_ptr tmp, mpz_ptr z)
    {
      std::fill(mantissa, mantissa + limbs, mp_limb_t(0));
      const int sign = mpf_sgn(x);
      if(sign == 0)
        return 0;
      mpf_set(tmp, x);
      mul_2exp(tmp, static_cast<long>(64 * limbs) - exponent);
      mpz_set_f(z, tmp);
      const size_t size = mpz_size(z);
      ASSERT(size <= limbs, DEBUG_STRING(size), DEBUG_STRING(limbs));
      std::copy_n(mpz_limbs_read(z), size, mantissa);
      return sign > 0 ? 1 : -1;
    }

    // L_X^{-1} of one block in fixed point, row by row:
    //   L(i,k) = sign(i,k) * mantissa(i,k) * 2^(exponent(i) - 64 limbs),
    // with exponent(i) bounding row i (the rows normalization of the default path's trmm).
    struct Fixed_Lower
    {
      El::Int dim = 0;
      size_t limbs = 0;
      std::vector<mp_limb_t> mantissas; // (i dim + k) limbs
      std::vector<int8_t> signs; // i dim + k
      std::vector<long> exponents; // per row

      Fixed_Lower(const El::Matrix<El::BigFloat> &L, const size_t limbs)
          : dim(L.Height()),
            limbs(limbs),
            mantissas(dim * dim * limbs, 0),
            signs(dim * dim, 0),
            exponents(dim, 0)
      {
        mpf_t tmp;
        mpz_t z;
        mpf_init2(tmp, El::gmp::Precision());
        mpz_init(z);
        for(El::Int i = 0; i < dim; ++i)
          {
            long exponent = std::numeric_limits<long>::min();
            for(El::Int k = 0; k <= i; ++k)
              if(mpf_sgn(L.CRef(i, k).gmp_float.get_mpf_t()) != 0)
                exponent = std::max(
                  exponent, upper_exponent(L.CRef(i, k).gmp_float.get_mpf_t()));
            if(exponent == std::numeric_limits<long>::min())
              continue;
            exponents[i] = exponent;
            for(El::Int k = 0; k <= i; ++k)
              signs[i * dim + k]
                = to_fixed(L.CRef(i, k).gmp_float.get_mpf_t(), exponent, limbs,
                           mantissas.data() + (i * dim + k) * limbs, tmp, z);
          }
        mpf_clear(tmp);
        mpz_clear(z);
      }
      [[nodiscard]] const mp_limb_t *mantissa(const El::Int i, const El::Int k) const
      {
        return mantissas.data() + (i * dim + k) * limbs;
      }
    };

    // out(:, col) = vec( Linv (F_p LY[:, j0:j1]) ) for p = columns[col], one local block.
    // Linv and LY are lower triangular; their strict upper parts are ignored (they hold
    // stale values from the in-place Cholesky / inverse).
    // B = F_p LY[:, j0:j1] is formed in BigFloat from the sparse entries. With Linv_fixed,
    // Linv B is then summed exactly in fixed point (B's columns normalized like the
    // default path's G) and rounded once per element; without it, in BigFloat.
    void compute_slab(const std::vector<std::vector<Sparse_F_Entry>> &F_block,
                      const std::vector<El::BigFloat> &values,
                      const El::Matrix<El::BigFloat> &Linv,
                      const Fixed_Lower *Linv_fixed,
                      const El::Matrix<El::BigFloat> &LY, const El::Int j0,
                      const El::Int j1, const std::vector<El::Int> &columns,
                      El::Matrix<El::BigFloat> &out)
    {
      const El::Int dim = Linv.Height();
      const El::Int width = j1 - j0;
      const auto prec = El::gmp::Precision();
      ASSERT_EQUAL(out.Height(), dim * width);
      ASSERT_EQUAL(static_cast<size_t>(out.Width()), columns.size());

      El::Matrix<El::BigFloat> B(dim, width);
      std::vector<char> is_nonzero_row(dim, 0);
      std::vector<El::Int> nonzero_rows;
      nonzero_rows.reserve(dim);

      mpf_t tmp, acc;
      mpf_init2(tmp, prec);
      mpf_init2(acc, prec);
      mpz_t z;
      mpz_init(z);

      // Fixed-point scratch: B's columns, and pos/neg sums of 2 limbs + 1 limbs.
      const size_t limbs = Linv_fixed != nullptr ? Linv_fixed->limbs : 0;
      const size_t sum_limbs = 2 * limbs + 1;
      std::vector<mp_limb_t> B_mantissas, product(2 * limbs), pos(sum_limbs),
        neg(sum_limbs), result(sum_limbs);
      std::vector<int8_t> B_signs;
      std::vector<long> B_exponents(width);

      for(size_t col = 0; col < columns.size(); ++col)
        {
          // B := F_p LY[:, j0:j1], only rows where F_p has entries.
          for(const auto row : nonzero_rows)
            is_nonzero_row[row] = 0;
          nonzero_rows.clear();
          for(const auto &e : F_block.at(columns[col]))
            {
              const El::Int i = e.row;
              const El::Int k = e.col;
              if(!is_nonzero_row[i])
                {
                  is_nonzero_row[i] = 1;
                  nonzero_rows.push_back(i);
                  for(El::Int jj = 0; jj < width; ++jj)
                    mpf_set_ui(B(i, jj).gmp_float.get_mpf_t(), 0);
                }
              const auto *v = values[e.value_index].gmp_float.get_mpf_t();
              // LY(k, j) is zero for j > k.
              const El::Int j_end = std::min(j1, k + 1);
              for(El::Int j = j0; j < j_end; ++j)
                {
                  mpf_mul(tmp, v, LY.CRef(k, j).gmp_float.get_mpf_t());
                  auto *b = B(i, j - j0).gmp_float.get_mpf_t();
                  mpf_add(b, b, tmp);
                }
            }
          std::sort(nonzero_rows.begin(), nonzero_rows.end());

          if(Linv_fixed == nullptr)
            {
              // C := Linv B, written column by column into out(:, col) in vec order.
              for(El::Int jj = 0; jj < width; ++jj)
                {
                  for(El::Int i = 0; i < dim; ++i)
                    {
                      mpf_set_ui(acc, 0);
                      for(const auto k : nonzero_rows)
                        {
                          if(k > i)
                            break;
                          mpf_mul(tmp, Linv.CRef(i, k).gmp_float.get_mpf_t(),
                                  B.CRef(k, jj).gmp_float.get_mpf_t());
                          mpf_add(acc, acc, tmp);
                        }
                      mpf_set(out(i + jj * dim, col).gmp_float.get_mpf_t(), acc);
                    }
                }
              continue;
            }

          // B's columns in fixed point, over its nonzero rows (index t).
          const size_t r = nonzero_rows.size();
          B_mantissas.assign(width * r * limbs, 0);
          B_signs.assign(width * r, 0);
          for(El::Int jj = 0; jj < width; ++jj)
            {
              long exponent = std::numeric_limits<long>::min();
              for(const auto k : nonzero_rows)
                if(mpf_sgn(B.CRef(k, jj).gmp_float.get_mpf_t()) != 0)
                  exponent = std::max(
                    exponent, upper_exponent(B.CRef(k, jj).gmp_float.get_mpf_t()));
              if(exponent == std::numeric_limits<long>::min())
                exponent = 0;
              B_exponents[jj] = exponent;
              for(size_t t = 0; t < r; ++t)
                B_signs[jj * r + t] = to_fixed(
                  B.CRef(nonzero_rows[t], jj).gmp_float.get_mpf_t(), exponent,
                  limbs, B_mantissas.data() + (jj * r + t) * limbs, tmp, z);
            }

          // C(i, jj) = 2^(e_i + e_jj - 128 limbs) sum_k L~(i,k) B~(k,jj), exactly.
          for(El::Int jj = 0; jj < width; ++jj)
            for(El::Int i = 0; i < dim; ++i)
              {
                std::fill(pos.begin(), pos.end(), mp_limb_t(0));
                std::fill(neg.begin(), neg.end(), mp_limb_t(0));
                for(size_t t = 0; t < r; ++t)
                  {
                    const El::Int k = nonzero_rows[t];
                    if(k > i)
                      break;
                    const int8_t sign_L = Linv_fixed->signs[i * dim + k];
                    const int8_t sign_B = B_signs[jj * r + t];
                    if(sign_L == 0 || sign_B == 0)
                      continue;
                    mpn_mul_n(product.data(), Linv_fixed->mantissa(i, k),
                              B_mantissas.data() + (jj * r + t) * limbs, limbs);
                    auto &sum = sign_L == sign_B ? pos : neg;
                    sum[2 * limbs]
                      += mpn_add_n(sum.data(), sum.data(), product.data(), 2 * limbs);
                  }
                int sign = 1;
                if(mpn_cmp(pos.data(), neg.data(), sum_limbs) >= 0)
                  mpn_sub_n(result.data(), pos.data(), neg.data(), sum_limbs);
                else
                  {
                    mpn_sub_n(result.data(), neg.data(), pos.data(), sum_limbs);
                    sign = -1;
                  }
                mp_size_t size = sum_limbs;
                while(size > 0 && result[size - 1] == 0)
                  --size;
                auto *c = out(i + jj * dim, col).gmp_float.get_mpf_t();
                if(size == 0)
                  {
                    mpf_set_ui(c, 0);
                    continue;
                  }
                mpz_t view;
                mpf_set_z(c, mpz_roinit_n(view, result.data(), sign * size));
                mul_2exp(c, Linv_fixed->exponents[i] + B_exponents[jj]
                              - static_cast<long>(128 * limbs));
              }
        }
      mpf_clear(tmp);
      mpf_clear(acc);
      mpz_clear(z);
    }
  }

  void compute_S_streaming(const SDP &sdp, const Block_Info &block_info,
                           const Block_Diagonal_Matrix &X_cholesky,
                           const Block_Diagonal_Matrix &Y_cholesky,
                           BigInt_Shared_Memory_Syrk_Context &syrk_context,
                           std::vector<El::Int> &column_exponents,
                           El::DistMatrix<El::BigFloat> &S, Timers &timers,
                           El::Matrix<int32_t> &block_timings_ms)
  {
    Scoped_Timer timer(timers, "S_stream");
    const size_t m = block_info.primal_dimension;
    const size_t num_blocks = block_info.num_blocks_local();
    const auto bits = El::gmp::Precision();
    constexpr auto uplo = El::UPPER;

    // Local copies of L_X^{-1} and L_Y (small: sum(dim^2) each). A block on several
    // ranks is gathered whole on each of them (LY_gathered).
    Scoped_Timer L_timer(timers, "L");
    std::vector<El::Matrix<El::BigFloat>> Linv(num_blocks), LY_gathered(num_blocks);
    std::vector<const El::Matrix<El::BigFloat> *> LY(num_blocks);
    std::vector<El::Int> dims(num_blocks);
    for(size_t b = 0; b < num_blocks; ++b)
      {
        const auto &X_block = X_cholesky.blocks.at(b);
        const auto &Y_block = Y_cholesky.blocks.at(b);
        if(X_block.Grid().Size() == 1)
          {
            Linv.at(b) = X_block.LockedMatrix();
            LY.at(b) = &Y_block.LockedMatrix();
          }
        else
          {
            const El::DistMatrix<El::BigFloat, El::STAR, El::STAR> X_star(
              X_block),
              Y_star(Y_block);
            Linv.at(b) = X_star.LockedMatrix();
            LY_gathered.at(b) = Y_star.LockedMatrix();
            LY.at(b) = &LY_gathered.at(b);
          }
        El::TriangularInverse(El::LOWER, El::NON_UNIT, Linv.at(b));
        dims.at(b) = X_block.Height();
      }
    // Fixed-point rows kernel unless SDPB_STREAM_P_MPF is set (BigFloat, for checks).
    std::vector<Fixed_Lower> Linv_fixed;
    if(std::getenv("SDPB_STREAM_P_MPF") == nullptr)
      {
        const size_t limbs = (bits + 63) / 64;
        Linv_fixed.reserve(num_blocks);
        for(size_t b = 0; b < num_blocks; ++b)
          Linv_fixed.emplace_back(Linv.at(b), limbs);
      }
    L_timer.stop();

    const auto bands = plan_bands(dims, syrk_context.input_band_height());
    // Every rank must take part in every band's syrk (shared-window fences, reductions).
    const auto num_iterations = static_cast<size_t>(El::mpi::AllReduce(
      static_cast<El::Int>(bands.size()), El::mpi::MAX, El::mpi::COMM_WORLD));

    // Rows of P for band t: one matrix per local block (zero height when the block is
    // not in this band). Each rank forms its own columns p of the band; on a grid with
    // more than one row of ranks, all rows of those columns, of which it keeps its own.
    const auto fill_band = [&](const size_t t, Block_Matrix &band) {
      band.width = m;
      band.blocks.clear();
      band.blocks.reserve(num_blocks);
      for(size_t b = 0; b < num_blocks; ++b)
        band.blocks.emplace_back(0, m, X_cholesky.blocks.at(b).Grid());
      if(t >= bands.size())
        return;
      for(const auto &slab : bands.at(t))
        {
          const auto b = slab.local_block;
          auto &block = band.blocks.at(b);
          block.Resize(dims.at(b) * (slab.col_end - slab.col_begin), m);
          std::vector<El::Int> columns(block.LocalWidth());
          for(El::Int jLoc = 0; jLoc < block.LocalWidth(); ++jLoc)
            columns.at(jLoc) = block.GlobalCol(jLoc);
          const auto compute = [&](El::Matrix<El::BigFloat> &out) {
            compute_slab(sdp.sparse_F.entries.at(b), sdp.sparse_F.values,
                         Linv.at(b),
                         Linv_fixed.empty() ? nullptr : &Linv_fixed.at(b),
                         *LY.at(b), slab.col_begin, slab.col_end, columns,
                         out);
          };
          if(block.ColStride() == 1)
            {
              compute(block.Matrix());
              continue;
            }
          El::Matrix<El::BigFloat> all_rows(block.Height(), block.LocalWidth());
          compute(all_rows);
          for(El::Int jLoc = 0; jLoc < block.LocalWidth(); ++jLoc)
            for(El::Int iLoc = 0; iLoc < block.LocalHeight(); ++iLoc)
              block.SetLocal(iLoc, jLoc,
                             all_rows.CRef(block.GlobalRow(iLoc), jLoc));
        }
    };

    El::Zero(S);
    using clock = std::chrono::steady_clock;
    const bool print_profile = std::getenv("SDPB_STREAM_P_PROFILE") != nullptr;

    if(syrk_context.can_accumulate_bands(S))
      {
        Scoped_Timer bands_timer(timers, "bands");
        Streaming_Syrk::Profile profile;
        Streaming_Syrk::syrk(
          syrk_context, num_iterations,
          [&](const size_t t, std::vector<El::DistMatrix<El::BigFloat>> &blocks) {
            Block_Matrix band;
            fill_band(t, band);
            blocks = std::move(band.blocks);
          },
          [&] {
            // The L copies are no longer needed; free them before the restore.
            Linv.clear();
            Linv.shrink_to_fit();
            LY_gathered.clear();
            LY_gathered.shrink_to_fit();
            Linv_fixed.clear();
            Linv_fixed.shrink_to_fit();
          },
          column_exponents, S, block_timings_ms, profile);
        if(print_profile && El::mpi::Rank(El::mpi::COMM_WORLD) == 0)
          {
            const auto &sp = syrk_context.band_accumulation_profile;
            El::Output("S_stream: ", num_iterations, " bands, ", profile.passes,
                       " passes; rows ", profile.produce.count(), " s, scale ",
                       profile.scale.count(), " s, syrk ", profile.syrk.count(),
                       " s (last pass: residues ", sp.residues, ", clear ",
                       sp.clear, ", blas ", sp.blas, ", reduce ", sp.reduce,
                       "), restore ", profile.restore.count(),
                       " s; log2(norm/2^E) in [", profile.lowest, ", ",
                       profile.highest, "]");
          }
        return;
      }

    // Fallback (several nodes): each band is normalized on its own
    // and restored into S separately.
    // On a single node, band contributions are restored straight into S (no second
    // m x m matrix); across nodes the residues must be reduced first, via S_band.
    const bool single_node = block_info.num_nodes() == 1;
    El::DistMatrix<El::BigFloat> S_band(S.Grid());
    if(!single_node)
      S_band.Resize(S.Height(), S.Width());
    const El::BigFloat one(1);
    const El::BigFloat eps = one >> static_cast<int>(bits / 2);

    // Timers keeps one list node per Scoped_Timer for the whole run, and the syrk driver
    // opens dozens per call: with ~100-200 bands per iteration that grew the process by
    // ~2 MB per iteration. Per-band internals therefore go to a scratch Timers, and the
    // phases are accumulated here; set SDPB_STREAM_P_PROFILE to print them.
    std::chrono::duration<double> t_rows{0}, t_normalize{0}, t_syrk{0};
    Scoped_Timer bands_timer(timers, "bands");

    for(size_t t = 0; t < num_iterations; ++t)
      {
        Timers band_timers;
        Block_Matrix band;
        const auto t0 = clock::now();
        fill_band(t, band);
        const auto t1 = clock::now();
        // Per-band column normalization (global over ranks: all rows of this band
        // enter one syrk accumulation).
        const auto normalizer
          = normalize_and_shift<Matrix_Normalization_Kind::COLUMNS>(
            band, bits, El::mpi::COMM_WORLD);
        const auto t2 = clock::now();

        const BigInt_Shared_Memory_Syrk_Context::Band_Provider provider
          = [&band](El::Int, El::Int, const El::Range<El::Int> &col_range,
                    std::vector<El::DistMatrix<El::BigFloat>> &views) {
              views.clear();
              for(const auto &block : band.blocks)
                {
                  auto &view = views.emplace_back();
                  El::LockedView(view, block,
                                 El::Range<El::Int>(0, block.Height()),
                                 col_range);
                }
            };
        if(single_node)
          {
            // S(i,j) += (S~(i,j) / 2^(2 bits)) * norm_i * norm_j, with the same
            // sanity checks as restore_syrk_output().
            const auto &norms = normalizer.norms;
            El::BigFloat value;
            syrk_context.restore_callback = [&](const El::Int i, const El::Int j,
                                                const Fmpz_BigInt &bigint) {
              bigint.to_BigFloat(value);
              value >>= static_cast<int>(2 * bits);
              ASSERT(El::Abs(value) <= one + eps,
                     "normalized band contribution out of range: ",
                     DEBUG_STRING(i), DEBUG_STRING(j), DEBUG_STRING(value));
              if(i == j)
                ASSERT((norms.at(i) == El::BigFloat(0) && value == El::BigFloat(0))
                         || El::Abs(value - one) <= eps,
                       "normalized band diagonal is not 0 or 1: ",
                       DEBUG_STRING(i), DEBUG_STRING(value));
              value *= norms.at(i) * norms.at(j);
              S.UpdateLocal(S.LocalRow(i), S.LocalCol(j), value);
            };
            syrk_context.bigint_syrk_blas_from_bands(uplo, provider, S, band_timers,
                                                     block_timings_ms, 1);
            syrk_context.restore_callback = nullptr;
          }
        else
          {
            El::Zero(S_band);
            syrk_context.bigint_syrk_blas_from_bands(
              uplo, provider, S_band, band_timers, block_timings_ms, 1);
            restore_syrk_output(uplo, normalizer, S_band);
            for(El::Int jLoc = 0; jLoc < S.LocalWidth(); ++jLoc)
              {
                const El::Int j = S.GlobalCol(jLoc);
                for(El::Int iLoc = 0; iLoc < S.LocalHeight(); ++iLoc)
                  {
                    if(S.GlobalRow(iLoc) > j)
                      continue;
                    S.UpdateLocal(iLoc, jLoc, S_band.GetLocalCRef(iLoc, jLoc));
                  }
              }
          }
        t_rows += t1 - t0;
        t_normalize += t2 - t1;
        t_syrk += clock::now() - t2;
      }
    if(print_profile && El::mpi::Rank(El::mpi::COMM_WORLD) == 0)
      El::Output("S_stream: ", num_iterations, " bands; rows ", t_rows.count(),
                 " s, normalize ", t_normalize.count(), " s, syrk+restore ",
                 t_syrk.count(), " s");
  }
}
