#include "Compute_S_Context.hxx"
#include "P_Spill.hxx"
#include "sdp_solve/Block_Matrix/Block_Matrix.hxx"
#include "sdp_solve/SDP_Solver/run/bigint_syrk/Matrix_Normalizer.hxx"
#include "sdpa_solve/SDP.hxx"
#include "sdpa_solve/memory_estimates.hxx"
#include "sdpb_util/split_range.hxx"
#include "sdpb_util/Timers/Timers.hxx"

#include <cstdlib>

namespace Sdpb::Sdpa
{
  // The normalized-and-shifted P, packed (#184), plus its column norms:
  // everything the syrk step needs, at a fraction of full BigFloat P.
  struct Spilled_P
  {
    P_Spill spill;
    std::vector<El::BigFloat> column_norms;
  };

  // Shim for restore_syrk_output(): a column normalizer whose norms were
  // collected chunk by chunk while P was being spilled.
  struct Precomputed_Column_Normalizer final
      : Abstract_Matrix_Normalizer<Matrix_Normalization_Kind::COLUMNS,
                                   Block_Matrix>
  {
    Precomputed_Column_Normalizer(const int bits,
                                  std::vector<El::BigFloat> &&_norms)
        : Abstract_Matrix_Normalizer(
            Block_Matrix{}, bits, std::nullopt,
            [&_norms](const Block_Matrix &, const auto &) {
              return std::move(_norms);
            })
    {}
  };

  // compute P = (vec(G_1) vec(G_2) vec(G_3) ... )
  // G_p = L_X_Inv F_p L_Y
  // P is split vertically into blocks
  //
  // #184: P is normalized, shifted to integers and packed into the spill
  // chunk by chunk; the full BigFloat P (the second-largest resident object
  // of the solve) never exists.
  Spilled_P initialize_P(const Environment &env, const SDP &sdp,
                         const Block_Info &block_info, const El::Grid &grid,
                         const Block_Diagonal_Matrix &X_cholesky,
                         const Block_Diagonal_Matrix &Y_cholesky,
                         Initialize_P_Context &ctx, Timers &timers,
                         El::Matrix<int32_t> &block_timings_ms,
                         const Verbosity verbosity)
  {
    // TODO add timers, update block_timings_ms
    Scoped_Timer timer(timers, "initialize_P");
    const size_t primal_dimension = block_info.primal_dimension;

    // Each (virtual) P block is a (dim*dim) x primal_dimension matrix
    std::vector<size_t> P_block_heights;
    P_block_heights.reserve(block_info.num_blocks());
    for(const auto &dim : block_info.block_dimensions)
      {
        P_block_heights.emplace_back(dim * dim);
      }
    const size_t P_width = primal_dimension;

    Spilled_P result;
    {
      std::vector<El::Int> local_heights;
      for(const auto block_index : block_info.block_indices)
        local_heights.push_back(P_block_heights.at(block_index));
      std::optional<std::filesystem::path> spill_dir;
      if(const char *dir = std::getenv("SDPB_P_SPILL_DIR"))
        spill_dir = std::filesystem::path(dir);
      result.spill.initialize(local_heights, P_width, grid,
                              El::gmp::Precision(), spill_dir);
      result.column_norms.resize(P_width);
      if(verbosity >= Verbosity::trace)
        {
          size_t total_height = 0;
          for(const auto h : local_heights)
            total_height += h;
          print_allocation_message_per_node(env, "P (packed spill)",
                                            result.spill.record_bytes()
                                              * total_height * P_width);
        }
    }

    const auto bits = El::gmp::Precision();
    constexpr auto uplo = El::LOWER;
    constexpr auto diag = El::NON_UNIT;
    constexpr auto orientation = El::NORMAL;

    // Prepare L_X_inv and L_Y
    Scoped_Timer L_timer(timers, "L");
    auto L_X_inv = X_cholesky;
    for(auto &block : L_X_inv.blocks)
      {
        El::TriangularInverse(uplo, diag, block);
      }
    auto L_Y = Y_cholesky;
    if(verbosity >= Verbosity::trace)
      {
        print_allocation_message_per_node(env, "L_X_inv",
                                          get_allocated_bytes(L_X_inv));
        print_allocation_message_per_node(env, "L_Y",
                                          get_allocated_bytes(L_Y));
      }

    // Make L_X_inv and L_Y bigint matrices
    const auto L_X_inv_normalizer
      = normalize_and_shift<Matrix_Normalization_Kind::ROWS>(L_X_inv, bits,
                                                             uplo);
    const auto L_Y_normalizer
      = normalize_and_shift<Matrix_Normalization_Kind::COLUMNS>(L_Y, bits,
                                                                uplo);

    ctx.compute_residues(L_X_inv, ctx.L_X_inv_residues, block_timings_ms);
    ctx.compute_residues(L_Y, ctx.L_Y_residues, block_timings_ms);

    L_timer.stop();

    // Finished L_X_inv and L_Y

    // Compute G_p = L_X_inv F_p L_Y and copy to P
    Scoped_Timer G_timer(timers, "G");
    for(const auto &p_range :
        split_range(El::IR(0, primal_dimension), ctx.cfg.split_factor))
      {
        const size_t p_begin = p_range.beg;
        const size_t p_end = p_range.end;
        Scoped_Timer p_timer(timers,
                             El::BuildString("p_", p_begin, "_", p_end));
        // #184: densify this chunk of F from the sparse triplets. The chunk is the only
        // dense copy of F data that ever exists now, and it is bounded by the split factor.
        std::vector<Block_Diagonal_Matrix> G;
        G.reserve(p_end - p_begin);
        for(size_t p = p_begin; p < p_end; ++p)
          {
            auto &G_p = G.emplace_back(block_info.block_dimensions,
                                       block_info.block_indices, grid);
            for(size_t local_block = 0;
                local_block < sdp.sparse_F.entries.size(); ++local_block)
              {
                sparse_densify_local(
                  sdp.sparse_F.entries.at(local_block).at(p),
                  G_p.blocks.at(local_block));
              }
          }
        if(verbosity >= Verbosity::trace && p_begin == 0)
          {
            print_allocation_message_per_node(env, "G",
                                              get_allocated_bytes(G));
          }
        // G := L_X_inv G
        {
          Scoped_Timer norm_timer(timers, "normalize_G");
          const auto G_normalizer
            = normalize_and_shift<Matrix_Normalization_Kind::COLUMNS>(
              G, bits, std::nullopt);
          norm_timer.stop();

          Scoped_Timer compute_residues_timer(timers, "compute_residues");
          auto &G_residues = ctx.G_residues(p_end - p_begin, El::HORIZONTAL);
          ctx.compute_residues(G, G_residues, block_timings_ms);
          compute_residues_timer.stop();

          constexpr auto side = El::LEFT;
          ctx.trmm(side, uplo, orientation, diag, ctx.L_X_inv_residues,
                   G_residues, G, verbosity, timers, block_timings_ms);

          Scoped_Timer remove_normalization_timer(timers,
                                                  "remove_normalization");
          restore_trmm_output(side, uplo, orientation, L_X_inv_normalizer,
                              G_normalizer, G);
        }
        // G := G L_Y
        {
          Scoped_Timer norm_timer(timers, "normalize_G");
          const auto G_normalizer
            = normalize_and_shift<Matrix_Normalization_Kind::ROWS>(
              G, bits, std::nullopt);
          norm_timer.stop();

          Scoped_Timer compute_residues_timer(timers, "compute_residues");
          auto &G_residues = ctx.G_residues(p_end - p_begin, El::VERTICAL);
          ctx.compute_residues(G, G_residues, block_timings_ms);
          compute_residues_timer.stop();

          constexpr auto side = El::RIGHT;
          ctx.trmm(side, uplo, orientation, diag, ctx.L_Y_residues, G_residues,
                   G, verbosity, timers, block_timings_ms);
          Scoped_Timer remove_normalization_timer(timers,
                                                  "remove_normalization");
          restore_trmm_output(side, uplo, orientation, L_Y_normalizer,
                              G_normalizer, G);
        }

        // Copy each G_p into a column of a P chunk (columns [p_begin, p_end)),
        // aligned so that element ownership matches the virtual full-width P.
        Scoped_Timer reshape_timer(timers, "reshape");
        const size_t chunk_width = p_end - p_begin;
        Block_Matrix P_chunk(P_block_heights, chunk_width,
                             block_info.block_indices, grid);
        for(auto &chunk_block : P_chunk.blocks)
          {
            const auto height = chunk_block.Height();
            chunk_block.Empty();
            chunk_block.Align(0, p_begin % grid.Width());
            chunk_block.Resize(height, chunk_width);
          }
        for(size_t block = 0; block < block_info.num_blocks_local(); ++block)
          {
            for(size_t p = p_begin; p < p_end; ++p)
              {
                const auto &G_p_block = G.at(p - p_begin).blocks.at(block);
                auto &P_block = P_chunk.blocks.at(block);
                const int vec_height = P_block.Height();
                constexpr int vec_width = 1;
                const auto out_I = El::Range<int>(0, vec_height);
                const auto out_J
                  = El::Range<int>(p - p_begin, p - p_begin + vec_width);

                auto vec_G_p_view = El::View(P_block, out_I, out_J);
                El::Reshape(vec_height, vec_width, G_p_block, vec_G_p_view);
              }
          }
        reshape_timer.stop();

        // Normalize this chunk's columns (norms are per column, so chunking
        // changes nothing), shift to integers, and spill.
        Scoped_Timer spill_timer(timers, "normalize_and_spill");
        const auto chunk_normalizer
          = normalize_and_shift<Matrix_Normalization_Kind::COLUMNS>(
            P_chunk, static_cast<int>(bits), El::mpi::COMM_WORLD);
        for(size_t j = 0; j < chunk_width; ++j)
          result.column_norms.at(p_begin + j) = chunk_normalizer.norms.at(j);
        for(size_t block = 0; block < block_info.num_blocks_local(); ++block)
          result.spill.append_chunk_block(block, p_begin,
                                          P_chunk.blocks.at(block));

        if(std::getenv("SDPB_P_SPILL_DEBUG") != nullptr && p_begin == 0)
          {
            // Probe A: the normalized-and-shifted column 0 should have
            // sum(v^2) / 2^(2 bits) ~= 1 (up to truncation).
            El::BigFloat sum_sq(0);
            for(auto &chunk_block : P_chunk.blocks)
              for(El::Int iLoc = 0; iLoc < chunk_block.LocalHeight(); ++iLoc)
                if(chunk_block.IsLocal(chunk_block.GlobalRow(iLoc), 0))
                  {
                    const auto &v = chunk_block.GetLocalCRef(
                      iLoc, chunk_block.LocalCol(0));
                    sum_sq += v * v;
                  }
            El::BigFloat scaled = sum_sq;
            scaled >>= 2 * static_cast<int>(bits);
            El::Output("SPILL_DEBUG A: col0 sum_sq >> 2bits = ", scaled);

            // Probe B: round-trip the spill for the first block, cols [0, chunk).
            for(size_t block = 0; block < block_info.num_blocks_local();
                ++block)
              {
                const auto &chunk_block = P_chunk.blocks.at(block);
                El::DistMatrix<El::BigFloat> rt(grid);
                result.spill.read_band(
                  block, El::Range<El::Int>(0, chunk_block.Height()),
                  El::Range<El::Int>(0, static_cast<El::Int>(chunk_width)),
                  rt);
                El::BigFloat max_diff(0);
                bool first_reported = false;
                for(El::Int jLoc = 0; jLoc < rt.LocalWidth(); ++jLoc)
                  for(El::Int iLoc = 0; iLoc < rt.LocalHeight(); ++iLoc)
                    {
                      const auto diff
                        = rt.GetLocalCRef(iLoc, jLoc)
                          - chunk_block.GetLocalCRef(iLoc, jLoc);
                      if(El::Abs(diff) > El::BigFloat(1) && !first_reported)
                        {
                          first_reported = true;
                          El::Output(
                            "SPILL_DEBUG B1: block ", block, " first bad (i=",
                            iLoc, ",j=", jLoc, ") old=",
                            chunk_block.GetLocalCRef(iLoc, jLoc),
                            " new=", rt.GetLocalCRef(iLoc, jLoc),
                            " old(i+1)=",
                            iLoc + 1 < rt.LocalHeight()
                              ? chunk_block.GetLocalCRef(iLoc + 1, jLoc)
                              : El::BigFloat(-999),
                            " old(j+1)=",
                            jLoc + 1 < rt.LocalWidth()
                              ? chunk_block.GetLocalCRef(iLoc, jLoc + 1)
                              : El::BigFloat(-999));
                        }
                      if(El::Abs(diff) > max_diff)
                        max_diff = El::Abs(diff);
                    }
                if(block < 3 || max_diff > El::BigFloat(1))
                  El::Output("SPILL_DEBUG B: block ", block, " h=",
                             chunk_block.Height(),
                             " roundtrip max |new - old| = ", max_diff);
              }
          }
      }

    return result;
  }

  void
  compute_S(const Environment &env, const SDP &sdp,
            const Block_Info &block_info,
            const Block_Diagonal_Matrix &X_cholesky,
            const Block_Diagonal_Matrix &Y_cholesky, const El::Grid &grid,
            Compute_S_Context &compute_S_context,
            El::DistMatrix<El::BigFloat> &S, Timers &timers,
            El::Matrix<int32_t> &block_timings_ms, const Verbosity verbosity)
  {
    Scoped_Timer timer(timers, "S");
    Spilled_P P
      = initialize_P(env, sdp, block_info, grid, X_cholesky, Y_cholesky,
                     compute_S_context.initialize_P_context, timers,
                     block_timings_ms, verbosity);

    // Band provider: one matrix per local block, zero-height outside the band,
    // heights summing to min(max_height, remaining rows of the stack).
    const auto &spill = P.spill;
    const BigInt_Shared_Memory_Syrk_Context::Band_Provider band_provider
      = [&spill](const El::Int skip_rows, const El::Int max_height,
                 const El::Range<El::Int> &col_range,
                 std::vector<El::DistMatrix<El::BigFloat>> &bands) {
          bands.clear();
          El::Int skip = skip_rows;
          El::Int budget = max_height;
          const bool debug = std::getenv("SDPB_P_SPILL_DEBUG") != nullptr;
          for(size_t b = 0; b < spill.block_heights.size(); ++b)
            {
              const auto height = spill.block_heights.at(b);
              El::Int row_begin = 0, row_end = 0;
              if(skip >= height)
                {
                  skip -= height;
                }
              else if(budget > 0)
                {
                  row_begin = skip;
                  skip = 0;
                  row_end = std::min(height, row_begin + budget);
                  budget -= row_end - row_begin;
                }
              auto &band = bands.emplace_back();
              spill.read_band(b, El::Range<El::Int>(row_begin, row_end),
                              col_range, band);
            }
          if(debug)
            {
              El::Int total = 0;
              for(const auto &band : bands)
                total += band.Height();
              El::Output("SPILL_DEBUG C: band skip=", skip_rows,
                         " max_h=", max_height, " cols=[", col_range.beg, ",",
                         col_range.end, ") total_band_height=", total);
            }
        };

    constexpr auto uplo = El::UPPER;
    compute_S_context.syrk_S_context.bigint_syrk_blas_from_bands(
      uplo, band_provider, S, timers, block_timings_ms);

    Scoped_Timer restore_timer(timers, "restore_S");
    const Precomputed_Column_Normalizer normalizer(
      static_cast<int>(El::gmp::Precision()), std::move(P.column_norms));
    restore_syrk_output(uplo, normalizer, S);
  }
}
