#pragma once

#include "Bigint_Syrk_Config.hxx"
#include "blas_jobs/create_blas_job_schedule.hxx"
#include "sdpb_util/bigint_shared_memory/fmpz/Fmpz_BigInt.hxx"
#include "sdpb_util/bigint_shared_memory/fmpz/Fmpz_Comb.hxx"
#include "sdpb_util/bigint_shared_memory/Vertical_Block_Matrix_Residues_Window.hxx"
#include "sdpb_util/bigint_shared_memory/Matrix_Residues_Window.hxx"
#include "sdpb_util/Timers/Timers.hxx"

#include <functional>
#include <memory>
#include <optional>
#include <vector>

class BigInt_Shared_Memory_Syrk_Context : boost::noncopyable
{
public:
  using Job_Schedule = Blas_Job_Schedule<Blas_Job>;

  BigInt_Shared_Memory_Syrk_Context(
    const Bigint_Syrk_Config &cfg, size_t group_index,
    const std::vector<size_t> &block_index_local_to_global,
    Verbosity verbosity,
    const std::function<Job_Schedule(
      Blas_Job::Kind kind, El::UpperOrLower uplo, size_t num_ranks,
      size_t num_primes, int output_height, int output_width,
      Verbosity _verbosity)> &create_job_schedule
    = create_blas_job_schedule);

  // Calculate Q := P^T P
  //
  // P and Q are distributed among shared memory communicator, context.shared_memory_comm
  // (in practice - among all processes on a single machine).
  //
  // bigint_input_matrix_blocks - horizontal bands of P matrix
  // bigint_output - Q matrix
  //
  // Each process has some blocks of P, and their indices in communicator are stored in
  // block_indices_per_shared_memory_comm (vector of the same size as bigint_input_matrix_blocks)
  //
  // Both P and Q elements should be (big) integers;
  // We calculate residues of P modulo set of primes,
  // then multiply residue matrices via BLAS,
  // and restore Q from residues using Chinese Remainder Theorem
  //
  // If you want to square arbitrary BigFloat matrix P,
  // then use Matrix_Normalizer before and after calling this bigint_syrk_blas()
  void bigint_syrk_blas(El::UpperOrLower uplo,
                        const std::vector<El::DistMatrix<El::BigFloat>>
                          &bigint_input_matrix_blocks,
                        El::DistMatrix<El::BigFloat> &bigint_output,
                        Timers &timers, El::Matrix<int32_t> &block_timings_ms);

  // #184: same as bigint_syrk_blas(), but the input P is never held as a whole.
  // The provider materializes, on demand, the row band [skip_rows, skip_rows + max_height)
  // of the vertical block stack, restricted to columns col_range: one matrix per local
  // block (zero-height for blocks outside the band), heights summing to
  // min(max_height, remaining). Bands are freed after their residues are computed,
  // so peak memory is one band instead of all of P.
  using Band_Provider = std::function<void(
    El::Int skip_rows, El::Int max_height, const El::Range<El::Int> &col_range,
    std::vector<El::DistMatrix<El::BigFloat>> &bands)>;
  void bigint_syrk_blas_from_bands(El::UpperOrLower uplo,
                                   const Band_Provider &band_provider,
                                   El::DistMatrix<El::BigFloat> &bigint_output,
                                   Timers &timers,
                                   El::Matrix<int32_t> &block_timings_ms,
                                   std::optional<size_t> input_iterations
                                   = std::nullopt);

  // #184: when set (single node only), restore_and_reduce() hands each restored element
  // (global row, global column, value) to this callback instead of writing it into the
  // output matrix -- streaming S adds band contributions into S without a second m x m
  // matrix. The callback must be cleared after use.
  std::function<void(El::Int, El::Int, const Fmpz_BigInt &)> restore_callback;

  // Rows per band that one input-window fill holds for this rank's MPI group.
  [[nodiscard]] El::Int input_band_height() const
  {
    return input_group_height_per_prime();
  }

  // #184: Q summed over many bands that share one integer scaling, restored once.
  // Every band runs the same BLAS jobs, so each rank sums the products of its own jobs,
  // reduced mod their prime, in private tiles; the output window is used only by the
  // restore, which gathers there all primes of each element for the CRT. The CRT thus
  // runs once per accumulation instead of once per band, and a band costs no clearing,
  // reduction or fences on the shared output window.
  // Single node, UPPER, full_width_input; every band spans all columns of Q.
  // Collective: all ranks call begin, then accumulate_band the same number of times,
  // then restore_band_accumulation.
  [[nodiscard]] bool
  can_accumulate_bands(const El::DistMatrix<El::BigFloat> &output) const;
  // Lays out the accumulator for output's distribution (output's storage is not used
  // afterwards and may be freed until the restore).
  void begin_band_accumulation(const El::DistMatrix<El::BigFloat> &output);
  // Zero the accumulator for another pass over the same bands.
  void reset_band_accumulation();
  void accumulate_band(
    const std::vector<El::DistMatrix<El::BigFloat>> &bigint_band_blocks,
    Timers &timers, El::Matrix<int32_t> &block_timings_ms);
  // Calls callback(global row, global column, value) for each element of this rank,
  // then frees the accumulator.
  void restore_band_accumulation(
    const std::function<void(El::Int, El::Int, const Fmpz_BigInt &)> &callback,
    Timers &timers);
  // Streaming (PMP solver): ceil(log2) of each column norm of P = L^{-1} B in the
  // previous iteration, the next scaling prediction (empty before the first Q).
  std::vector<El::Int> band_column_exponents;
  // Seconds spent in accumulate_band() since the last begin_band_accumulation().
  struct Band_Accumulation_Profile
  {
    double residues = 0, blas = 0, reduce = 0, fence = 0;
  } band_accumulation_profile;
  // Restored values are exact while |Q(i,j)| < 2^(primes_product_bits() - 2).
  [[nodiscard]] flint_bitcnt_t primes_product_bits() const
  {
    return fmpz_bits(comb.primes_product.value);
  }

private:
  El::mpi::Comm shared_memory_comm;
  // Index of MPI group on a node
  size_t group_index;
  // Number of MPI groups on a node
  size_t num_groups;
  int total_block_height_per_node;
  Fmpz_Comb comb;
  const Verbosity verbosity;
  // All blocks from each MPI group are combined
  // into a single block in Block_Residue_Matrices_Window
  std::unique_ptr<Vertical_Block_Matrix_Residues_Window<double>>
    input_grouped_block_residues_window_A;
  std::unique_ptr<Vertical_Block_Matrix_Residues_Window<double>>
    input_grouped_block_residues_window_B;
  // How many times we should fill input window
  // to process all blocks:
  // (should be same for all ranks)
  size_t input_window_split_factor = 0;
  size_t output_window_split_factor = 0;
  // #184: window A spans all input columns (band accumulation only).
  bool full_width_input = false;
  std::unique_ptr<Matrix_Residues_Window<double>> output_residues_window;
  const std::vector<size_t> block_index_local_to_global;
  std::function<Job_Schedule(Blas_Job::Kind kind, El::UpperOrLower uplo,
                             size_t num_ranks, size_t num_primes,
                             int output_height, int output_width,
                             Verbosity verbosity)>
    create_blas_job_schedule_func;
  std::map<std::tuple<Blas_Job::Kind, El::UpperOrLower, El::Int, El::Int>,
           Job_Schedule>
    blas_job_schedule_cache;

  [[nodiscard]] const Job_Schedule &
  get_blas_job_schedule(Blas_Job::Kind kind, El::UpperOrLower uplo,
                        El::Int output_height, El::Int output_width);

  void clear_residues(const Job_Schedule &blas_job_schedule);
  void compute_block_residues(
    Vertical_Block_Matrix_Residues_Window<double>
      &grouped_block_residues_window,
    const std::vector<El::DistMatrix<El::BigFloat>> &bigint_input_matrix_blocks,
    El::Int skip_rows, El::Range<El::Int> col_range, Timers &timers,
    El::Matrix<int32_t> &block_timings_ms);

  void bigint_syrk_blas_shmem_submatrix(
    El::UpperOrLower uplo,
    const std::vector<El::DistMatrix<El::BigFloat>> &bigint_input_matrix_blocks,
    const El::Range<El::Int> &output_I, const El::Range<El::Int> &output_J,
    Timers &timers, El::Matrix<int32_t> &block_timings_ms);

  void bigint_syrk_blas_shmem_submatrix_from_bands(
    El::UpperOrLower uplo, const Band_Provider &band_provider,
    const El::Range<El::Int> &output_I, const El::Range<El::Int> &output_J,
    Timers &timers, El::Matrix<int32_t> &block_timings_ms,
    size_t input_iterations);

  void
  restore_and_reduce(std::optional<El::UpperOrLower> uplo,
                     El::DistMatrix<El::BigFloat> &output, Timers &timers);
  // Global offsets of the output submatrix passed to restore_and_reduce().
  El::Int restore_row_offset = 0;
  El::Int restore_col_offset = 0;

  // Band accumulation state (#184). A run is a contiguous piece of one column of an
  // output block that this rank restores: rows [row_begin, row_begin + length) of
  // global column col.
  struct Accumulator_Run
  {
    El::Int row_begin;
    El::Int col;
    El::Int length;
    size_t window_offset; // in each prime's output matrix
  };
  struct Accumulator_Block
  {
    El::Range<El::Int> I;
    El::Range<El::Int> J;
    std::vector<Accumulator_Run> runs;
  };
  // One BLAS job of this rank in the schedule of the whole output (job.I, job.J global,
  // as for an output split factor 1: the output blocks only bound the restore's window,
  // and one schedule balances the ranks' work better than one per block), summed column
  // by column from accumulator[offset]: the upper triangle of a syrk job, all of a gemm
  // job.
  struct Accumulator_Tile
  {
    Blas_Job job;
    size_t offset;
  };
  std::vector<Accumulator_Block> accumulator_blocks;
  std::vector<Accumulator_Tile> accumulator_tiles;
  // Residues in [0, prime). Primes are < 2^21 (capped for full_width_input, see
  // Bigint_Syrk_Config::comb()), so float is exact.
  std::vector<float> accumulator;
  // BLAS output of one tile.
  std::vector<double> tile_scratch;
  // The band's values, staged by their owners at the input window's rows and columns
  // (records of band_value_record limbs: size, exponent, limbs), so that all ranks share
  // computing the residues: a rank's own part of a band is as uneven as the block
  // mapping makes it (a rank holding several blocks whole has many times the entries of
  // one holding some columns of a big block). They are kept in the output window, idle
  // until the restore, or in a window of their own if it is too small
  // (Bigint_Syrk_Config::band_values_bytes()).
  mp_limb_t *band_values = nullptr;
  std::unique_ptr<Shared_Window_Array<mp_limb_t>> band_values_window;
  size_t band_value_limbs = 0;
  size_t band_value_record = 0;
  void fence_band_values();
  void stage_band_values(
    const std::vector<El::DistMatrix<El::BigFloat>> &bigint_band_blocks);
  void compute_staged_band_residues(
    const std::vector<El::DistMatrix<El::BigFloat>> &bigint_band_blocks,
    El::Matrix<int32_t> &block_timings_ms);

  [[nodiscard]] El::Int input_group_height_per_prime() const;
};
