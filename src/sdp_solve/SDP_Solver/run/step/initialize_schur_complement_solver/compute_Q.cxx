#include "sdp_solve/Block_Matrix/Block_Diagonal_Matrix.hxx"
#include "sdp_solve/SDP.hxx"
#include "sdp_solve/SDP_Solver/run/bigint_syrk/BigInt_Shared_Memory_Syrk_Context.hxx"
#include "sdp_solve/SDP_Solver/run/bigint_syrk/Matrix_Normalizer.hxx"
#include "sdp_solve/SDP_Solver/run/bigint_syrk/streaming_syrk.hxx"
#include "sdp_solve/memory_estimates.hxx"
#include "sdpb_util/Timers/Timers.hxx"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <sstream>
#include <string>

// schur_off_diagonal = L^{-1} B
void initialize_schur_off_diagonal(
  const Environment &env, const SDP &sdp, const Block_Info &block_info,
  Block_Diagonal_Matrix &schur_complement,
  Block_Matrix &schur_off_diagonal,
  Block_Diagonal_Matrix &schur_complement_cholesky, Timers &timers,
  El::Matrix<int32_t> &block_timings_ms, const Verbosity verbosity,
  const bool cholesky_only = false)
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
      {
        Scoped_Timer cholesky_timer(timers, "cholesky_" + block_index_string);
        schur_complement_cholesky.blocks[block]
          = schur_complement.blocks[block];
        // #184: the Schur complement is workspace, not used after this copy; free each
        // block now instead of holding both through compute_Q().
        schur_complement.blocks[block].Empty();

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

      if(cholesky_only)
        continue;
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
void syrk_Q_streaming(const SDP &sdp, const Block_Info &block_info,
                      const Block_Diagonal_Matrix &schur_complement_cholesky,
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
        const El::Int block_height = B.blocks.at(block).Height();
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
    total_cost += rows_cost(B.blocks.at(block).Height());

  std::vector<std::unique_ptr<El::DistMatrix<El::BigFloat>>> L_inv_B(
    num_blocks);
  // Rows of T_j computed so far in this pass (kept after T_j is freed).
  std::vector<El::Int> rows_done(num_blocks, 0);
  const auto produce_band
    = [&](const size_t t, std::vector<El::DistMatrix<El::BigFloat>> &blocks) {
        if(t == 0)
          std::fill(rows_done.begin(), rows_done.end(), 0);
        for(size_t block = 0; block < num_blocks; ++block)
          if(L_inv_B.at(block) && (t == 0 || last_band.at(block) < t))
            L_inv_B.at(block).reset();
        blocks.clear();
        blocks.reserve(num_blocks);
        for(size_t block = 0; block < num_blocks; ++block)
          blocks.emplace_back(0, N, B.blocks.at(block).Grid());
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
                const El::Int height = B.blocks.at(block).Height();
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
                const auto &B_block = B.blocks.at(block);
                const auto &L = schur_complement_cholesky.blocks.at(block);
                auto &T = L_inv_B.at(block);
                if(!T)
                  T = std::make_unique<El::DistMatrix<El::BigFloat>>(
                    B_block.Height(), N, B_block.Grid());
                const El::IR rows(r0, r1);
                const El::IR above(0, r0);
                El::DistMatrix<El::BigFloat> panel;
                El::View(panel, *T, rows, El::ALL);
                El::Copy(El::LockedView(B_block, rows, El::ALL), panel);
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
          El::Copy(El::LockedView(*L_inv_B.at(piece.block),
                                  El::IR(piece.row_begin, piece.row_end),
                                  El::ALL),
                   blocks.at(piece.block));
      };

  Streaming_Syrk::Profile profile;
  Streaming_Syrk::syrk(
    bigint_syrk_context, num_bands, produce_band,
    [&] {
      for(auto &block : L_inv_B)
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
                 sp.residues, ", clear ", sp.clear, ", blas ", sp.blas,
                 ", reduce ", sp.reduce, "), restore ",
                 profile.restore.count(), " s; log2(norm/2^E) in [",
                 profile.lowest, ", ", profile.highest, "]; trsm by rank [",
                 ranks.str(), "]");
    }
}

void compute_Q(const Environment &env, const SDP &sdp,
               const Block_Info &block_info,
               Block_Diagonal_Matrix &schur_complement,
               Block_Matrix &schur_off_diagonal,
               Block_Diagonal_Matrix &schur_complement_cholesky,
               BigInt_Shared_Memory_Syrk_Context &bigint_syrk_context,
               El::DistMatrix<El::BigFloat> &Q, Timers &timers,
               El::Matrix<int32_t> &block_timings_ms,
               const Verbosity verbosity)
{
  Scoped_Timer timer(timers, "Q");

  // #184: streaming leaves schur_off_diagonal empty.
  const bool streaming = Streaming_Syrk::enabled()
                         && bigint_syrk_context.can_accumulate_bands(Q);
  initialize_schur_off_diagonal(env, sdp, block_info, schur_complement,
                                schur_off_diagonal, schur_complement_cholesky,
                                timers, block_timings_ms, verbosity, streaming);
  if(streaming)
    syrk_Q_streaming(sdp, block_info, schur_complement_cholesky,
                     bigint_syrk_context, Q, timers, block_timings_ms);
  else
    syrk_Q(env, schur_off_diagonal, bigint_syrk_context, Q, timers,
           block_timings_ms, verbosity);
}