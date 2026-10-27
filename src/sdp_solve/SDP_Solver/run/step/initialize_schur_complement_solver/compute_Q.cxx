#include "sdp_solve/Block_Matrix/Block_Diagonal_Matrix.hxx"
#include "sdp_solve/SDP.hxx"
#include "sdp_solve/SDP_Solver/run/bigint_syrk/BigInt_Shared_Memory_Syrk_Context.hxx"
#include "sdp_solve/SDP_Solver/run/bigint_syrk/Matrix_Normalizer.hxx"
#include "sdp_solve/SDP_Solver/run/bigint_syrk/streaming_syrk.hxx"
#include "sdp_solve/SDP_Solver/run/multimodular/Multimodular_Block_Factor.hxx"
#include "sdp_solve/SDP_Solver/run/multimodular/multimodular_blas.hxx"
#include "sdp_solve/memory_estimates.hxx"
#include "sdpb_util/memory_trace.hxx"
#include "sdpb_util/Timers/Timers.hxx"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <map>
#include <memory>
#include <sstream>
#include <string>

// schur_complement_cholesky := its Cholesky factor L (in place: it holds the Schur
// complement on entry), and schur_off_diagonal = L^{-1} B.
// With streaming_factors (#184, streaming Q), only the factors: L^{-1} B is left to
// syrk_Q_streaming(), and with --multimodularTrsm the factor objects of blocks on
// several ranks are kept there (nullptr for blocks on one rank).
void initialize_schur_off_diagonal(
  const Environment &env, const SDP &sdp, const Block_Info &block_info,
  Block_Matrix &schur_off_diagonal,
  Block_Diagonal_Matrix &schur_complement_cholesky, Timers &timers,
  El::Matrix<int32_t> &block_timings_ms, const bool multimodular_trsm,
  const Verbosity verbosity,
  std::vector<std::unique_ptr<Multimodular_Block_Factor>> *streaming_factors
  = nullptr)
{
  schur_off_diagonal.blocks.clear();
  // Setting correct width is needed for Block_Matrix_Normalizer.
  // TODO: refactor Block_Matrix to ensure correct block width for all blocks
  schur_off_diagonal.width = sdp.dual_objective_b.Height();
  schur_off_diagonal.blocks.reserve(schur_complement_cholesky.blocks.size());

  size_t num_blocks_local = schur_complement_cholesky.blocks.size();
  for(size_t block = 0; block < num_blocks_local; ++block)
    {
      const auto global_block_index = block_info.block_indices.at(block);
      auto block_index_string = std::to_string(global_block_index);
      // --multimodularTrsm: Cholesky and L^{-1} B with multimodular BLAS,
      // see SDP_Solver/run/multimodular/Readme.md. #184: the factor is formed in
      // place (schur_complement_cholesky holds S on entry); a block on one rank needs
      // no factor object, its local matrix is factored directly.
      auto &cholesky = schur_complement_cholesky.blocks[block];
      const bool one_rank = cholesky.Grid().Size() == 1;
      const Multimodular_Blas_Config multimodular_config;
      Multimodular_Blas_Timings multimodular_timings;
      std::unique_ptr<Multimodular_Block_Factor> multimodular_factor;
      if(multimodular_trsm && !one_rank)
        multimodular_factor = std::make_unique<Multimodular_Block_Factor>(
          cholesky, multimodular_config);
      {
        Scoped_Timer cholesky_timer(timers, "cholesky_" + block_index_string);
        try
          {
            if(multimodular_factor)
              multimodular_factor->cholesky(cholesky, cholesky, timers);
            else if(multimodular_trsm)
              multimodular_cholesky_lower(cholesky.Matrix(), multimodular_config,
                                          multimodular_timings);
            else
              Cholesky(El::UpperOrLowerNS::LOWER, cholesky);
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

      if(streaming_factors != nullptr)
        {
          streaming_factors->push_back(std::move(multimodular_factor));
          continue;
        }
      // schur_off_diagonal = L^{-1} B
      Scoped_Timer solve_timer(timers, "solve_" + block_index_string);

      schur_off_diagonal.blocks.push_back(sdp.free_var_block_copy(block));
      auto &L_inv_B = schur_off_diagonal.blocks[block];
      if(multimodular_factor)
        multimodular_factor->solve(L_inv_B, timers);
      else if(multimodular_trsm)
        multimodular_trsm_left_lower(cholesky.LockedMatrix(), L_inv_B.Matrix(),
                                     multimodular_config, multimodular_timings);
      else
        El::Trsm(El::LeftOrRightNS::LEFT, El::UpperOrLowerNS::LOWER,
                 El::OrientationNS::NORMAL, El::UnitOrNonUnitNS::NON_UNIT,
                 El::BigFloat(1), cholesky, L_inv_B);
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

// Q = P^T P = (L^{-1} B)^T (L^{-1} B) = schur_off_diagonal^T schur_off_diagonal
void syrk_Q(const Environment &env, Block_Matrix &schur_off_diagonal,
            BigInt_Shared_Memory_Syrk_Context &bigint_syrk_context,
            El::DistMatrix<El::BigFloat> &Q, Timers &timers,
            El::Matrix<int32_t> &block_timings_ms, const Verbosity verbosity)
{
  Scoped_Timer syrk_timer(timers, "syrk");

  // Normalize P columns and multiply by 2^N
  Scoped_Timer normalizer_timer(timers, "normalize_P");
  const auto normalizer
    = normalize_and_shift<Matrix_Normalization_Kind::COLUMNS>(
      schur_off_diagonal, El::gmp::Precision(), El::mpi::COMM_WORLD);
  if(verbosity >= Verbosity::trace)
    {
      print_allocation_message_per_node(env, "Matrix_Normalizer",
                                        get_allocated_bytes(normalizer.norms));
    }
  normalizer_timer.stop();

  // Calculate Q = P^T P
  constexpr auto uplo = El::UPPER;
  bigint_syrk_context.bigint_syrk_blas(uplo, schur_off_diagonal.blocks, Q,
                                       timers, block_timings_ms);

  // Remove normalization
  Scoped_Timer restore_timer(timers, "restore_P_Q");
  normalizer.restore(schur_off_diagonal);
  restore_syrk_output(uplo, normalizer, Q);
}

// #184 (SDPB_STREAM_P): Q = (L^{-1} B)^T (L^{-1} B) without holding L^{-1} B. The rows of
// T_j = L_j^{-1} B_j are formed panel by panel as the bands need them (forward
// substitution: T[r0:r1] = L[r0:r1,r0:r1]^{-1} (B[r0:r1] - L[r0:r1,0:r0] T[0:r0])), go to
// the residue syrk, and T_j is freed after its last band (streaming_syrk.hxx). Bands are
// collective over the node, so the panels are paced to keep each rank's work per band
// even (below). The Schur complement equation is then solved with B and L
// (solve_schur_complement_equation()).
// A block may span several ranks: they form one MPI group, share all its blocks, and so
// have identical band schedules.
// With --multimodularTrsm the panels are solved with multimodular BLAS, each rank for its
// own columns of T_j ([STAR,VR]) against the whole factor: the block's local matrix on one
// rank, or the factor object's lower triangle (multimodular_factors[block]) on several.
void syrk_Q_streaming(
  const SDP &sdp, const Block_Info &block_info,
  const Block_Diagonal_Matrix &schur_complement_cholesky,
  const bool multimodular_trsm,
  const std::vector<std::unique_ptr<Multimodular_Block_Factor>>
    &multimodular_factors,
  BigInt_Shared_Memory_Syrk_Context &bigint_syrk_context,
                      El::DistMatrix<El::BigFloat> &Q, Timers &timers,
                      El::Matrix<int32_t> &block_timings_ms)
{
  Scoped_Timer syrk_timer(timers, "syrk_stream");
  const auto &B = sdp.free_var_matrix;
  const size_t num_blocks = B.blocks.size();
  const El::Int N = Q.Width();
  const El::Int band_height = bigint_syrk_context.input_band_height();

  // Bands of rows [row_begin, row_end) of the blocks, in block order, at most one piece
  // per block per band.
  struct Piece
  {
    size_t block;
    El::Int row_begin;
    El::Int row_end;
  };
  std::vector<std::vector<Piece>> bands;
  std::vector<size_t> last_band(num_blocks, 0);
  {
    std::vector<Piece> current;
    El::Int height = 0;
    for(size_t block = 0; block < num_blocks; ++block)
      {
        const El::Int block_height = sdp.free_var_block_height(block);
        for(El::Int row = 0; row < block_height;)
          {
            if(height == band_height)
              {
                bands.push_back(std::move(current));
                current = {};
                height = 0;
              }
            const El::Int take
              = std::min(band_height - height, block_height - row);
            current.push_back({block, row, row + take});
            height += take;
            row += take;
            last_band.at(block) = bands.size();
          }
      }
    if(!current.empty())
      bands.push_back(std::move(current));
  }
  // Every rank takes part in every band's syrk.
  const auto num_bands = static_cast<size_t>(El::mpi::AllReduce(
    static_cast<El::Int>(bands.size()), El::mpi::MAX, El::mpi::COMM_WORLD));

  // Bands run in lockstep, so what counts is the slowest rank's work in each band, summed
  // over bands. Row r of T_j costs ~r (it is solved against the r rows above it), so
  // computing rows only as the bands take them loads the late bands of a rank holding one
  // big block, while ranks of many small blocks work evenly: each band then waits for a
  // different rank. Instead each rank computes T at an even pace, a fraction t/num_bands
  // of its total cost by band t, and at least the rows the bands take. That runs ahead of
  // the bands, never behind (the cost of the first r rows grows as r^2); a block whose T_j
  // is started early is held whole until its last band either way. Rows are computed
  // every bands_per_panel bands, on the same bands on every rank: fewer, taller panels
  // cut the per-call cost of Gemm/Trsm (communication when a block spans several ranks).
  const size_t bands_per_panel = [] {
    const char *value = std::getenv("SDPB_STREAM_P_PANEL_BANDS");
    return value == nullptr ? size_t(8)
                            : std::max<size_t>(1, std::stoul(value));
  }();
  // Cost of the first r rows of T_j, in units of N multiply-adds.
  const auto rows_cost = [](const El::Int r) {
    return 0.5 * static_cast<double>(r) * static_cast<double>(r);
  };
  double total_cost = 0;
  for(size_t block = 0; block < num_blocks; ++block)
    total_cost += rows_cost(sdp.free_var_block_height(block));

  // The grid of each band block. With --multimodularTrsm a block on a grid of several
  // rows gets a grid of one row over the same ranks in VR order: each rank then owns the
  // same whole columns of the band as of T_j ([STAR,VR]), so the copy into the band is
  // local and the residues are computed column by column. On the block's own grid,
  // [MC,MR] splits each column over the grid rows: a redistribution per band, and the
  // residues written to the window one by one.
  std::vector<const El::Grid *> band_grids(num_blocks);
  std::map<const El::Grid *, std::unique_ptr<El::Grid>> column_grids;
  for(size_t block = 0; block < num_blocks; ++block)
    {
      const auto &grid = B.blocks.at(block).Grid();
      band_grids.at(block) = &grid;
      if(multimodular_trsm && grid.Height() > 1)
        {
          auto &column_grid = column_grids[&grid];
          if(!column_grid)
            column_grid = std::make_unique<El::Grid>(grid.VRComm(), 1);
          band_grids.at(block) = column_grid.get();
        }
    }

  // T_j: [MC,MR] like B_j (El), or this rank's columns of it (multimodular).
  std::vector<std::unique_ptr<El::DistMatrix<El::BigFloat>>> L_inv_B(
    num_blocks);
  std::vector<std::unique_ptr<El::DistMatrix<El::BigFloat, El::STAR, El::VR>>>
    L_inv_B_columns(num_blocks);
  const Multimodular_Blas_Config multimodular_config;
  Multimodular_Blas_Timings multimodular_timings;
  // Rows of T_j computed so far in this pass (kept after T_j is freed).
  std::vector<El::Int> rows_done(num_blocks, 0);
  const auto produce_band
    = [&](const size_t t, std::vector<El::DistMatrix<El::BigFloat>> &blocks) {
        if(t == 0)
          std::fill(rows_done.begin(), rows_done.end(), 0);
        for(size_t block = 0; block < num_blocks; ++block)
          if(t == 0 || last_band.at(block) < t)
            {
              L_inv_B.at(block).reset();
              L_inv_B_columns.at(block).reset();
            }
        blocks.clear();
        blocks.reserve(num_blocks);
        for(size_t block = 0; block < num_blocks; ++block)
          blocks.emplace_back(0, N, *band_grids.at(block));
        if(t >= bands.size())
          return;
        if(t % bands_per_panel == 0)
          {
            // T[r0:r1] = L[r0:r1,r0:r1]^{-1} (B[r0:r1] - L[r0:r1,0:r0] T[0:r0]) for the
            // rows the next bands_per_panel bands take, and ahead up to the pace.
            const size_t next_t = std::min(t + bands_per_panel, bands.size());
            std::vector<El::Int> rows_needed(rows_done);
            for(size_t u = t; u < next_t; ++u)
              for(const auto &piece : bands.at(u))
                rows_needed.at(piece.block) = std::max(
                  rows_needed.at(piece.block), piece.row_end);
            double pace = total_cost * static_cast<double>(next_t)
                          / static_cast<double>(bands.size());
            for(size_t block = 0; block < num_blocks; ++block)
              {
                const El::Int height = sdp.free_var_block_height(block);
                const El::Int pace_rows
                  = pace <= 0 ? 0
                    : pace >= rows_cost(height)
                      ? height
                      : static_cast<El::Int>(std::sqrt(2 * pace));
                pace -= rows_cost(height);
                const El::Int r0 = rows_done.at(block);
                const El::Int r1 = std::max(rows_needed.at(block),
                                            std::min(pace_rows, height));
                if(r1 <= r0)
                  continue;
                Timer trsm_timer;
                const auto &L = schur_complement_cholesky.blocks.at(block);
                const El::IR rows(r0, r1);
                const El::IR above(0, r0);
                if(multimodular_trsm)
                  {
                    auto &T = L_inv_B_columns.at(block);
                    if(!T)
                      {
                        El::DistMatrix<El::BigFloat> B_scratch; // --sparseB
                        T = std::make_unique<
                          El::DistMatrix<El::BigFloat, El::STAR, El::VR>>(
                          B.blocks.at(block).Grid());
                        *T = sdp.free_var_block(block, B_scratch);
                      }
                    const auto &factor = multimodular_factors.at(block);
                    const El::Matrix<El::BigFloat> &L_local
                      = factor ? factor->lower() : L.LockedMatrix();
                    auto &X = T->Matrix();
                    El::Matrix<El::BigFloat> panel;
                    El::View(panel, X, rows, El::ALL);
                    if(r0 > 0)
                      multimodular_gemm_sub(
                        panel, El::LockedView(L_local, rows, above),
                        El::NORMAL, El::LockedView(X, above, El::ALL), false,
                        multimodular_config, multimodular_timings);
                    multimodular_trsm_left_lower(
                      El::LockedView(L_local, rows, rows), panel,
                      multimodular_config, multimodular_timings);
                    rows_done.at(block) = r1;
                    block_timings_ms(block_info.block_indices.at(block), 0)
                      += trsm_timer.elapsed_milliseconds();
                    continue;
                  }
                // T_j starts as B_j: rows below rows_done are B's until solved.
                auto &T = L_inv_B.at(block);
                if(!T)
                  T = std::make_unique<El::DistMatrix<El::BigFloat>>(
                    sdp.free_var_block_copy(block));
                El::DistMatrix<El::BigFloat> panel;
                El::View(panel, *T, rows, El::ALL);
                if(r0 > 0)
                  El::Gemm(El::NORMAL, El::NORMAL, El::BigFloat(-1),
                           El::LockedView(L, rows, above),
                           El::LockedView(*T, above, El::ALL),
                           El::BigFloat(1), panel);
                El::Trsm(El::LeftOrRightNS::LEFT, El::UpperOrLowerNS::LOWER,
                         El::OrientationNS::NORMAL,
                         El::UnitOrNonUnitNS::NON_UNIT, El::BigFloat(1),
                         El::LockedView(L, rows, rows), panel);
                rows_done.at(block) = r1;
                block_timings_ms(block_info.block_indices.at(block), 0)
                  += trsm_timer.elapsed_milliseconds();
              }
          }
        // The syrk scales the band in place, and later panels still need these rows of
        // T unscaled: hand it a copy.
        for(const auto &piece : bands.at(t))
          {
            const El::IR rows(piece.row_begin, piece.row_end);
            if(multimodular_trsm)
              {
                // The same columns on the same ranks (band_grids): a local copy.
                const auto &T = *L_inv_B_columns.at(piece.block);
                auto &band_block = blocks.at(piece.block);
                band_block.Resize(piece.row_end - piece.row_begin, N);
                ASSERT_EQUAL(band_block.LocalWidth(), T.LocalWidth());
                ASSERT(T.LocalWidth() == 0
                         || band_block.GlobalCol(0) == T.GlobalCol(0),
                       DEBUG_STRING(band_block.GlobalCol(0)),
                       DEBUG_STRING(T.GlobalCol(0)));
                El::Copy(El::LockedView(T.LockedMatrix(), rows, El::ALL),
                         band_block.Matrix());
              }
            else
              El::Copy(El::LockedView(*L_inv_B.at(piece.block), rows, El::ALL),
                       blocks.at(piece.block));
          }
      };

  Streaming_Syrk::Profile profile;
  Streaming_Syrk::syrk(
    bigint_syrk_context, num_bands, produce_band,
    [&] {
      for(auto &block : L_inv_B)
        block.reset();
      for(auto &block : L_inv_B_columns)
        block.reset();
    },
    bigint_syrk_context.band_column_exponents, Q, block_timings_ms, profile);

  if(Streaming_Syrk::profiling() && El::mpi::Rank(El::mpi::COMM_WORLD) == 0)
    {
      const auto &sp = bigint_syrk_context.band_accumulation_profile;
      const auto &by_rank = profile.produce_by_rank;
      std::ostringstream ranks;
      for(size_t r = 0; r < by_rank.size(); ++r)
        ranks << (r == 0 ? "" : " ") << std::lround(by_rank[r]);
      El::Output("Q_stream: ", num_bands, " bands, ", bands_per_panel,
                 " per panel, ", profile.passes, " passes; trsm ",
                 profile.produce.count(), " s (slowest rank ",
                 *std::max_element(by_rank.begin(), by_rank.end()),
                 " s, per-band slowest summed ", profile.produce_band_max,
                 " s), scale ", profile.scale.count(), " s, syrk ",
                 profile.syrk.count(), " s (last pass: residues ",
                 sp.residues, ", blas ", sp.blas, ", reduce ", sp.reduce,
                 ", fence ", sp.fence, "), restore ",
                 profile.restore.count(), " s; log2(norm/2^E) in [",
                 profile.lowest, ", ", profile.highest, "]; trsm by rank [",
                 ranks.str(), "]",
                 multimodular_trsm
                   ? El::BuildString("; multimodular (rank 0): to_int ",
                                     multimodular_timings.to_int_seconds,
                                     " s, blas ",
                                     multimodular_timings.blas_seconds,
                                     " s, from_int ",
                                     multimodular_timings.from_int_seconds,
                                     " s, base ",
                                     multimodular_timings.base_seconds, " s")
                   : std::string());
    }
}

void compute_Q(const Environment &env, const SDP &sdp,
               const Block_Info &block_info, Block_Matrix &schur_off_diagonal,
               Block_Diagonal_Matrix &schur_complement_cholesky,
               BigInt_Shared_Memory_Syrk_Context &bigint_syrk_context,
               El::DistMatrix<El::BigFloat> &Q, Timers &timers,
               El::Matrix<int32_t> &block_timings_ms,
               const bool multimodular_trsm, const Verbosity verbosity)
{
  Scoped_Timer timer(timers, "Q");

  // #184: streaming leaves schur_off_diagonal empty.
  const bool streaming = Streaming_Syrk::enabled()
                         && bigint_syrk_context.can_accumulate_bands(Q);
  std::vector<std::unique_ptr<Multimodular_Block_Factor>> streaming_factors;
  initialize_schur_off_diagonal(env, sdp, block_info, schur_off_diagonal,
                                schur_complement_cholesky, timers,
                                block_timings_ms, multimodular_trsm, verbosity,
                                streaming ? &streaming_factors : nullptr);
  memory_trace(streaming ? "Cholesky done" : "Cholesky and L^{-1} B done");
  if(streaming)
    syrk_Q_streaming(sdp, block_info, schur_complement_cholesky,
                     multimodular_trsm, streaming_factors, bigint_syrk_context,
                     Q, timers, block_timings_ms);
  else
    syrk_Q(env, schur_off_diagonal, bigint_syrk_context, Q, timers,
           block_timings_ms, verbosity);
}