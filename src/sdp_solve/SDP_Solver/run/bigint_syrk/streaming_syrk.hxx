#pragma once

// #184 (bootstrap-speed): Q = P^T P for a P that is produced band by band and never held
// whole (SDPB_STREAM_P). Used by the SDPA solver (P = vec(L_X^{-1} F_p L_Y)) and by the PMP
// solver (P = L^{-1} B).
//
// All bands share one column scaling: entry (r, p) enters the residue syrk as the integer
// trunc(P(r,p) 2^(bits + extra - E_p)), so the residues of all bands add up and Q is
// restored once. E_p is ceil(log2) of the previous iteration's norm of column p; this
// iteration's exact norms are summed on the way. The syrk primes cover 2(bits + extra)
// bits plus headroom, so the scaling is exact (no overflow, and no fewer bits than an exact
// column normalization) while
//   -extra <= log2(norm_p) - E_p <= max_log2_ratio   for every column.
// Otherwise the pass is repeated with E from the norms just computed, which always fit.
// Without a prediction (first iteration) a norm-only pass runs first.

#include "BigInt_Shared_Memory_Syrk_Context.hxx"
#include "sdpb_util/Timers/Timers.hxx"
#include "sdpb_util/memory_trace.hxx"
#include "sdpb_util/assert.hxx"

#include <El.hpp>
#include <gmp.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <limits>
#include <string>
#include <vector>

namespace Streaming_Syrk
{
  inline bool enabled()
  {
    const char *value = std::getenv("SDPB_STREAM_P");
    return value != nullptr && std::string(value) != "0";
  }

  // The streaming syrk's primes cover this many extra bits per factor, so a predicted
  // scale that overshoots a column norm by up to 2^12 keeps every bit.
  constexpr int extra_bits = 12;

  inline void mul_2exp(mpf_ptr x, const long shift)
  {
    if(shift >= 0)
      mpf_mul_2exp(x, x, static_cast<mp_bitcnt_t>(shift));
    else
      mpf_div_2exp(x, x, static_cast<mp_bitcnt_t>(-shift));
  }

  // Squared norm of one column, summed as d^2 2^(2e) (x = d 2^e) relative to the largest
  // exponent seen: any range, ~double accuracy.
  struct Column_Norm
  {
    double sum = 0;
    long exp = std::numeric_limits<long>::min();

    void add(mpf_srcptr x)
    {
      long e;
      const double d = mpf_get_d_2exp(&e, x);
      if(d == 0)
        return;
      if(e > exp)
        {
          sum = exp == std::numeric_limits<long>::min()
                  ? 0
                  : std::ldexp(sum, static_cast<int>(
                                      2 * std::max(exp - e, -4096L)));
          exp = e;
        }
      sum += std::ldexp(d * d,
                        static_cast<int>(2 * std::max(e - exp, -4096L)));
    }
  };

  // log2 of the column norms summed over all ranks (-inf for zero columns).
  inline std::vector<double> global_log2_norms(const std::vector<Column_Norm> &norms)
  {
    const size_t m = norms.size();
    std::vector<El::Int> exps(m);
    for(size_t p = 0; p < m; ++p)
      exps[p] = norms[p].sum == 0 ? std::numeric_limits<El::Int>::min()
                                  : static_cast<El::Int>(norms[p].exp);
    El::mpi::AllReduce(exps.data(), static_cast<int>(m), El::mpi::MAX,
                       El::mpi::COMM_WORLD);
    std::vector<double> sums(m, 0.0);
    for(size_t p = 0; p < m; ++p)
      if(norms[p].sum != 0)
        sums[p] = std::ldexp(
          norms[p].sum,
          static_cast<int>(
            2 * std::max<El::Int>(norms[p].exp - exps[p], -4096)));
    El::mpi::AllReduce(sums.data(), static_cast<int>(m), El::mpi::SUM,
                       El::mpi::COMM_WORLD);
    std::vector<double> result(m);
    for(size_t p = 0; p < m; ++p)
      result[p] = sums[p] == 0
                    ? -std::numeric_limits<double>::infinity()
                    : static_cast<double>(exps[p]) + 0.5 * std::log2(sums[p]);
    return result;
  }

  // E_p = ceil(log2 norm_p): the norm then lies in (2^(E_p-1), 2^E_p].
  inline std::vector<El::Int>
  exponents_from_log2_norms(const std::vector<double> &log2_norms)
  {
    std::vector<El::Int> exponents(log2_norms.size(), 0);
    for(size_t p = 0; p < log2_norms.size(); ++p)
      if(std::isfinite(log2_norms[p]))
        exponents[p] = static_cast<El::Int>(std::ceil(log2_norms[p]));
    return exponents;
  }

  inline bool profiling()
  {
    return std::getenv("SDPB_STREAM_P_PROFILE") != nullptr;
  }

  struct Profile
  {
    std::chrono::duration<double> produce{0}, scale{0}, syrk{0}, restore{0};
    size_t passes = 0;
    double lowest = 0, highest = 0;
    // With profiling(): the bands run in lockstep, so the produce time that counts is
    // the slowest rank's in each band, summed over bands; and each rank's own total.
    double produce_band_max = 0;
    std::vector<double> produce_by_rank;
  };

  // Q := P^T P (upper triangle), P given band by band.
  // produce_band(t, blocks) fills band t = 0 .. num_bands-1 (called in order, once per
  // pass): one matrix per local block, each of width Q.Width(), zero height for blocks
  // not in the band, at most input_band_height() rows in total. The entries are scaled in
  // place afterwards. num_bands must be the same on all ranks.
  // column_exponents carries the prediction from one call to the next (empty at first).
  // before_restore() runs once the passes are done.
  inline void
  syrk(BigInt_Shared_Memory_Syrk_Context &syrk_context, const size_t num_bands,
       const std::function<void(size_t, std::vector<El::DistMatrix<El::BigFloat>> &)>
         &produce_band,
       const std::function<void()> &before_restore,
       std::vector<El::Int> &column_exponents, El::DistMatrix<El::BigFloat> &Q,
       El::Matrix<int32_t> &block_timings_ms, Profile &profile)
  {
    using clock = std::chrono::steady_clock;
    const size_t m = Q.Width();
    const long bits = El::gmp::Precision();
    const double max_log2_ratio
      = (static_cast<double>(syrk_context.primes_product_bits()) - 3
         - 2.0 * static_cast<double>(bits + extra_bits))
        / 2;
    ASSERT(max_log2_ratio >= 2, "SDPB_STREAM_P: syrk primes leave no headroom",
           DEBUG_STRING(syrk_context.primes_product_bits()), DEBUG_STRING(bits));

    // Q is needed only for the restore: lay out the accumulator for Q's distribution,
    // then free Q's storage until then.
    syrk_context.begin_band_accumulation(Q);
    Q.Empty();
    size_t syrk_passes = 0;
    const bool measure_bands = profiling();
    profile.produce_band_max = 0;

    // with_syrk = false: norms only.
    const auto pass = [&](const bool with_syrk) -> std::vector<double> {
      std::vector<Column_Norm> norms(m);
      if(with_syrk && ++syrk_passes > 1)
        syrk_context.reset_band_accumulation();
      for(size_t t = 0; t < num_bands; ++t)
        {
          // Per-band syrk internals go to a scratch Timers: Timers keeps every node.
          Timers band_timers;
          std::vector<El::DistMatrix<El::BigFloat>> blocks;
          const auto t0 = clock::now();
          produce_band(t, blocks);
          const auto t1 = clock::now();
          if(measure_bands)
            profile.produce_band_max += El::mpi::AllReduce(
              std::chrono::duration<double>(t1 - t0).count(), El::mpi::MAX,
              El::mpi::COMM_WORLD);
          for(auto &block : blocks)
            {
              auto &local = block.Matrix();
              for(El::Int jLoc = 0; jLoc < block.LocalWidth(); ++jLoc)
                {
                  const El::Int p = block.GlobalCol(jLoc);
                  const long shift
                    = bits + extra_bits
                      - static_cast<long>(with_syrk ? column_exponents[p] : 0);
                  for(El::Int iLoc = 0; iLoc < block.LocalHeight(); ++iLoc)
                    {
                      auto *x = local(iLoc, jLoc).gmp_float.get_mpf_t();
                      norms[p].add(x);
                      if(with_syrk)
                        mul_2exp(x, shift);
                    }
                }
            }
          const auto t2 = clock::now();
          if(with_syrk)
            syrk_context.accumulate_band(blocks, band_timers, block_timings_ms);
          if(t == num_bands / 2)
            memory_trace(with_syrk ? "streaming syrk: middle band"
                                   : "streaming norms: middle band");
          profile.produce += t1 - t0;
          profile.scale += t2 - t1;
          profile.syrk += clock::now() - t2;
        }
      return global_log2_norms(norms);
    };

    profile.passes = 0;
    if(column_exponents.size() != m)
      {
        column_exponents = exponents_from_log2_norms(pass(false));
        ++profile.passes;
      }
    std::vector<double> log2_norms;
    while(true)
      {
        log2_norms = pass(true);
        ++profile.passes;
        profile.lowest = std::numeric_limits<double>::infinity();
        profile.highest = -profile.lowest;
        for(size_t p = 0; p < m; ++p)
          if(std::isfinite(log2_norms[p]))
            {
              const double r
                = log2_norms[p] - static_cast<double>(column_exponents[p]);
              profile.lowest = std::min(profile.lowest, r);
              profile.highest = std::max(profile.highest, r);
            }
        if(profile.highest <= max_log2_ratio && profile.lowest >= -extra_bits)
          break;
        ASSERT(profile.passes < 4,
               "SDPB_STREAM_P: column scaling does not settle",
               DEBUG_STRING(profile.lowest), DEBUG_STRING(profile.highest));
        column_exponents = exponents_from_log2_norms(log2_norms);
      }

    if(before_restore)
      before_restore();
    const auto t0 = clock::now();
    Q.Resize(m, m);
    El::Zero(Q);
    El::BigFloat value;
    Timers restore_timers;
    syrk_context.restore_band_accumulation(
      [&](const El::Int i, const El::Int j, const Fmpz_BigInt &bigint) {
        bigint.to_BigFloat(value);
        auto *v = value.gmp_float.get_mpf_t();
        if(i == j && std::isfinite(log2_norms[i]))
          {
            // Q(i,i) is the squared norm summed above, in units of
            // 2^(2(bits + extra) - 2 E_i).
            long e;
            const double d = mpf_get_d_2exp(&e, v);
            const double expected
              = 2
                * (log2_norms[i] - static_cast<double>(column_exponents[i])
                   + static_cast<double>(bits + extra_bits));
            ASSERT(d > 0 && std::abs(std::log2(d) + e - expected) < 1e-6,
                   "streaming syrk: diagonal disagrees with the column norm",
                   DEBUG_STRING(i), DEBUG_STRING(e), DEBUG_STRING(expected));
          }
        mul_2exp(v, static_cast<long>(column_exponents[i] + column_exponents[j])
                      - 2 * (bits + extra_bits));
        Q.SetLocal(Q.LocalRow(i), Q.LocalCol(j), value);
      },
      restore_timers);
    profile.restore += clock::now() - t0;
    if(measure_bands)
      {
        profile.produce_by_rank.assign(El::mpi::Size(El::mpi::COMM_WORLD), 0.0);
        profile.produce_by_rank.at(El::mpi::Rank(El::mpi::COMM_WORLD))
          = profile.produce.count();
        El::mpi::AllReduce(profile.produce_by_rank.data(),
                           static_cast<int>(profile.produce_by_rank.size()),
                           El::mpi::SUM, El::mpi::COMM_WORLD);
      }
    // Prediction for the next call.
    column_exponents = exponents_from_log2_norms(log2_norms);
  }
}
