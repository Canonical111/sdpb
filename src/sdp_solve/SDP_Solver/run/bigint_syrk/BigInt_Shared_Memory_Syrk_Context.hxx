#pragma once

#include "blas_jobs/create_blas_jobs_schedule.hxx"
#include "blas_jobs/Blas_Job_Schedule.hxx"
#include "fmpz/Fmpz_Comb.hxx"
#include "Block_Residue_Matrices_Window.hxx"
#include "Residue_Matrices_Window.hxx"
#include "sdpb_util/Timers/Timers.hxx"

#include <memory>
#include <optional>
#include <vector>

// sdpb --compressLinvB (lab #170): the nonzero columns of one block of P, as
// sorted disjoint ranges of global columns, and the block's height
struct Block_Column_Runs
{
  El::Int height = 0;
  std::vector<El::Range<El::Int>> runs;
};

// A range of rows of the input residue window whose nonzero columns are in
// runs (no runs: zero padding rows)
struct Window_Row_Segment
{
  El::Range<El::Int> rows;
  std::vector<El::Range<El::Int>> runs;
};

class BigInt_Shared_Memory_Syrk_Context : boost::noncopyable
{
public:
  BigInt_Shared_Memory_Syrk_Context(
    const El::mpi::Comm &shared_memory_comm, size_t group_index,
    const std::vector<int> &group_comm_sizes, mp_bitcnt_t precision,
    size_t max_shared_memory_bytes,
    const std::vector<El::Int> &blocks_height_per_group, int block_width,
    const std::vector<size_t> &block_index_local_to_global,
    Verbosity verbosity,
    const std::function<Blas_Job_Schedule(
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
  //
  // column_maps (optional, sdpb --compressLinvB): block b holds the columns
  // (*column_maps)[b] of its band of P (-1: zero padding); the other
  // columns are zero. Their residues are written as zeros, so the residue
  // windows, and Q, are those of the dense P.
  void bigint_syrk_blas(
    El::UpperOrLower uplo,
    const std::vector<El::DistMatrix<El::BigFloat>> &bigint_input_matrix_blocks,
    El::DistMatrix<El::BigFloat> &bigint_output, Timers &timers,
    El::Matrix<int32_t> &block_timings_ms,
    const std::vector<std::vector<El::Int>> *column_maps = nullptr);

private:
  El::mpi::Comm shared_memory_comm;
  // Index of MPI group on a node
  size_t group_index;
  // Sizes of MPI groups on a node
  const std::vector<int> &group_comm_sizes;
  // Number of MPI groups on a node
  size_t num_groups;
  int total_block_height_per_node;
  Fmpz_Comb comb;
  const Verbosity verbosity;
  // All blocks from each MPI group are combined
  // into a single block in Block_Residue_Matrices_Window
  std::unique_ptr<Block_Residue_Matrices_Window<double>>
    input_grouped_block_residues_window_A;
  std::unique_ptr<Block_Residue_Matrices_Window<double>>
    input_grouped_block_residues_window_B;
  // How many times we should fill input window
  // to process all blocks:
  // (should be same for all ranks)
  size_t input_window_split_factor = 0;
  size_t output_window_split_factor = 0;
  std::unique_ptr<Residue_Matrices_Window<double>> output_residues_window;
  const std::vector<size_t> block_index_local_to_global;
  std::function<Blas_Job_Schedule(Blas_Job::Kind kind, El::UpperOrLower uplo,
                                  size_t num_ranks, size_t num_primes,
                                  int output_height, int output_width,
                                  Verbosity verbosity)>
    create_blas_job_schedule_func;
  std::map<std::tuple<Blas_Job::Kind, El::UpperOrLower, El::Int, El::Int>,
           std::shared_ptr<Blas_Job_Schedule>>
    blas_job_schedule_cache;
  // --compressLinvB: schedules with the work on the column runs as job
  // costs, by (kind, uplo, output_I, output_J)
  std::map<std::tuple<Blas_Job::Kind, El::UpperOrLower, El::Int, El::Int,
                      El::Int, El::Int>,
           std::shared_ptr<Blas_Job_Schedule>>
    weighted_blas_job_schedule_cache;

  [[nodiscard]] std::shared_ptr<Blas_Job_Schedule>
  get_blas_job_schedule(Blas_Job::Kind kind, El::UpperOrLower uplo,
                        El::Int output_height, El::Int output_width);
  [[nodiscard]] std::shared_ptr<Blas_Job_Schedule>
  get_weighted_blas_job_schedule(
    Blas_Job::Kind kind, El::UpperOrLower uplo,
    const El::Range<El::Int> &output_I, const El::Range<El::Int> &output_J,
    const std::vector<std::vector<Block_Column_Runs>> &node_column_runs);

  void clear_residues(const Blas_Job_Schedule &blas_job_schedule);
  // column_inverse_maps (optional): for block b, the column of the block
  // holding global column g of P, or -1 if that column is zero
  void compute_block_residues(
    Block_Residue_Matrices_Window<double> &grouped_block_residues_window,
    const std::vector<El::DistMatrix<El::BigFloat>> &bigint_input_matrix_blocks,
    El::Int skip_rows, El::Range<El::Int> col_range, Timers &timers,
    El::Matrix<int32_t> &block_timings_ms,
    const std::vector<std::vector<El::Int>> *column_inverse_maps = nullptr);

  void bigint_syrk_blas_shmem_submatrix(
    El::UpperOrLower uplo,
    const std::vector<El::DistMatrix<El::BigFloat>> &bigint_input_matrix_blocks,
    const El::Range<El::Int> &output_I, const El::Range<El::Int> &output_J,
    Timers &timers, El::Matrix<int32_t> &block_timings_ms,
    const std::vector<std::vector<El::Int>> *column_inverse_maps = nullptr,
    const std::vector<std::vector<Block_Column_Runs>> *node_column_runs
    = nullptr);

  // --compressLinvB: for each MPI group on the node, the column runs of its
  // blocks, in the order of the window (gathered over shared_memory_comm)
  std::vector<std::vector<Block_Column_Runs>> gather_node_column_runs(
    const std::vector<El::DistMatrix<El::BigFloat>> &bigint_input_matrix_blocks,
    const std::vector<std::vector<El::Int>> &column_maps, El::Int width) const;
  // Row segments of the input window filled at split iteration iter
  std::vector<Window_Row_Segment> window_row_segments(
    const std::vector<std::vector<Block_Column_Runs>> &node_column_runs,
    size_t iter) const;

  void
  restore_and_reduce(std::optional<El::UpperOrLower> uplo,
                     El::DistMatrix<El::BigFloat> &output, Timers &timers);

  [[nodiscard]] El::Int input_group_height_per_prime() const;
};
