#pragma once

#include "sdp_solve/SDP_Solver/run/bigint_syrk/BigInt_Shared_Memory_Syrk_Context.hxx"
#include "sdp_solve/SDP_Solver/run/bigint_syrk/streaming_syrk.hxx"
#include "sdpa_solve/Block_Info.hxx"
#include "sdpa_solve/SDP_Solver/run/step/Initialize_P_Context.hxx"
#include "Solver_Run_Config.hxx"

#include <optional>

namespace Sdpb::Sdpa
{
  // #184: SDPB_STREAM_P=1 forms S band by band without P (compute_S_streaming.cxx);
  // the trmm windows of the default path are then never allocated.
  bool stream_S_enabled();
  // Extra bits per factor covered by the streaming syrk's primes (streaming_syrk.hxx).
  constexpr int stream_S_extra_bits = Streaming_Syrk::extra_bits;

  struct Compute_S_Context
  {
    std::optional<Initialize_P_Context> initialize_P_context;
    BigInt_Shared_Memory_Syrk_Context syrk_S_context;
    // Streaming: ceil(log2) of each column norm of P in the previous iteration
    // (empty before the first S).
    std::vector<El::Int> stream_column_exponents{};
    // TODO
    // Compute_S_Context(const Initialize_P_Config &initialize_p_config,
    //                   const Bigint_Syrk_Config &syrk_S_config)
    //     : initialize_P_context(initialize_p_config),
    //       syrk_S_context(syrk_S_config)
    // {}
    static Compute_S_Context
    create(const Block_Info &block_info, const Solver_Run_Config &cfg,
           const Verbosity verbosity)
    {
      // TODO: reuse shared buffers for trmm and syrk?
      if(stream_S_enabled())
        return {std::nullopt, BigInt_Shared_Memory_Syrk_Context(
                                cfg.syrk_S_config, block_info.node_group_index(),
                                block_info.block_indices, verbosity)};
      return {std::optional<Initialize_P_Context>(std::in_place,
                                                  cfg.initialize_P_config),
              BigInt_Shared_Memory_Syrk_Context(
                cfg.syrk_S_config, block_info.node_group_index(),
                block_info.block_indices, verbosity)};
    }
  };
}
