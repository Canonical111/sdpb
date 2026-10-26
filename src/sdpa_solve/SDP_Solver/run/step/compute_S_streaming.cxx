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
// Scope (v1): every local block must live on a single rank (always true at 1 rank, and
// whenever #blocks >= #ranks per group).

#include "Compute_S_Context.hxx"
#include "sdp_solve/Block_Matrix/Block_Matrix.hxx"
#include "sdp_solve/SDP_Solver/run/bigint_syrk/Matrix_Normalizer.hxx"
#include "sdpa_solve/SDP.hxx"
#include "sdpb_util/Timers/Timers.hxx"

#include <gmp.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <string>
#include <vector>

namespace Sdpb::Sdpa
{
  bool stream_S_enabled()
  {
    const char *value = std::getenv("SDPB_STREAM_P");
    return value != nullptr && std::string(value) != "0";
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

    // out(:, p) = vec( Linv (F_p LY[:, j0:j1]) ) for every p, one local block.
    // Linv and LY are lower triangular; their strict upper parts are ignored (they hold
    // stale values from the in-place Cholesky / inverse).
    void compute_slab(const std::vector<std::vector<Sparse_F_Entry>> &F_block,
                      const std::vector<El::BigFloat> &values,
                      const El::Matrix<El::BigFloat> &Linv,
                      const El::Matrix<El::BigFloat> &LY, const El::Int j0,
                      const El::Int j1, El::Matrix<El::BigFloat> &out)
    {
      const El::Int dim = Linv.Height();
      const El::Int width = j1 - j0;
      const auto prec = El::gmp::Precision();
      ASSERT_EQUAL(out.Height(), dim * width);
      ASSERT_EQUAL(static_cast<size_t>(out.Width()), F_block.size());

      El::Matrix<El::BigFloat> B(dim, width);
      std::vector<char> is_nonzero_row(dim, 0);
      std::vector<El::Int> nonzero_rows;
      nonzero_rows.reserve(dim);

      mpf_t tmp, acc;
      mpf_init2(tmp, prec);
      mpf_init2(acc, prec);

      for(size_t p = 0; p < F_block.size(); ++p)
        {
          // B := F_p LY[:, j0:j1], only rows where F_p has entries.
          for(const auto row : nonzero_rows)
            is_nonzero_row[row] = 0;
          nonzero_rows.clear();
          for(const auto &e : F_block[p])
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

          // C := Linv B, written column by column into out(:, p) in vec order.
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
                  mpf_set(out(i + jj * dim, p).gmp_float.get_mpf_t(), acc);
                }
            }
        }
      mpf_clear(tmp);
      mpf_clear(acc);
    }
  }

  void compute_S_streaming(const SDP &sdp, const Block_Info &block_info,
                           const Block_Diagonal_Matrix &X_cholesky,
                           const Block_Diagonal_Matrix &Y_cholesky,
                           BigInt_Shared_Memory_Syrk_Context &syrk_context,
                           El::DistMatrix<El::BigFloat> &S, Timers &timers,
                           El::Matrix<int32_t> &block_timings_ms)
  {
    Scoped_Timer timer(timers, "S_stream");
    const size_t m = block_info.primal_dimension;
    const size_t num_blocks = block_info.num_blocks_local();
    const auto bits = El::gmp::Precision();
    constexpr auto uplo = El::UPPER;

    // Local copies of L_X^{-1} and L_Y (small: sum(dim^2) each).
    Scoped_Timer L_timer(timers, "L");
    std::vector<El::Matrix<El::BigFloat>> Linv(num_blocks);
    std::vector<const El::Matrix<El::BigFloat> *> LY(num_blocks);
    std::vector<El::Int> dims(num_blocks);
    for(size_t b = 0; b < num_blocks; ++b)
      {
        const auto &X_block = X_cholesky.blocks.at(b);
        ASSERT(X_block.Grid().Size() == 1,
               "SDPB_STREAM_P (v1) needs every block on a single rank; block ",
               block_info.block_indices.at(b), " spans ",
               X_block.Grid().Size(), " ranks");
        Linv.at(b) = X_block.LockedMatrix();
        El::TriangularInverse(El::LOWER, El::NON_UNIT, Linv.at(b));
        LY.at(b) = &Y_cholesky.blocks.at(b).LockedMatrix();
        dims.at(b) = X_block.Height();
      }
    L_timer.stop();

    const auto bands = plan_bands(dims, syrk_context.input_band_height());
    // Every rank must take part in every band's syrk (shared-window fences, reductions).
    const auto num_iterations = static_cast<size_t>(El::mpi::AllReduce(
      static_cast<El::Int>(bands.size()), El::mpi::MAX, El::mpi::COMM_WORLD));

    El::Zero(S);
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
    using clock = std::chrono::steady_clock;
    std::chrono::duration<double> t_rows{0}, t_normalize{0}, t_syrk{0};
    const bool print_profile = std::getenv("SDPB_STREAM_P_PROFILE") != nullptr;
    Scoped_Timer bands_timer(timers, "bands");

    for(size_t t = 0; t < num_iterations; ++t)
      {
        Timers band_timers;
        // One matrix per local block (zero height when the block is not in this band).
        Block_Matrix band;
        band.width = m;
        band.blocks.reserve(num_blocks);
        for(size_t b = 0; b < num_blocks; ++b)
          band.blocks.emplace_back(0, m, X_cholesky.blocks.at(b).Grid());
        const auto t0 = clock::now();
        {
          if(t < bands.size())
            for(const auto &slab : bands.at(t))
              {
                const auto b = slab.local_block;
                auto &block = band.blocks.at(b);
                block.Resize(dims.at(b) * (slab.col_end - slab.col_begin), m);
                compute_slab(sdp.sparse_F.entries.at(b), sdp.sparse_F.values,
                             Linv.at(b), *LY.at(b), slab.col_begin,
                             slab.col_end, block.Matrix());
              }
        }

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
