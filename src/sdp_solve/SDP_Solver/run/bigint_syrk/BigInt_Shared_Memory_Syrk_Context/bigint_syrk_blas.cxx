#include "../BigInt_Shared_Memory_Syrk_Context.hxx"
#include "sdpb_util/bigint_shared_memory/fmpz/Fmpz_BigInt.hxx"
#include "sdpb_util/bigint_shared_memory/fmpz/residues.hxx"
#include "sdpb_util/assert.hxx"
#include "sdpb_util/split_range.hxx"

#include <cblas.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>

namespace
{
  // output = inputA^T * inputB + beta * output
  void gemm(const El::Matrix<double> &input_A,
            const El::Matrix<double> &input_B, El::Matrix<double> &output,
            const double beta = 1.0)
  {
    // A: KxN matrix
    // A: KxM matrix
    // output = input^T * input: NxM matrix
    ASSERT_EQUAL(input_A.Height(), input_B.Height());
    ASSERT_EQUAL(output.Height(), input_A.Width());
    ASSERT_EQUAL(output.Width(), input_B.Width());

    CBLAS_LAYOUT layout = CblasColMajor;
    CBLAS_TRANSPOSE TransA = CblasTrans;
    CBLAS_TRANSPOSE TransB = CblasNoTrans;
    const CBLAS_INDEX M = input_A.Width();
    const CBLAS_INDEX N = input_B.Width();
    const CBLAS_INDEX K = input_A.Height();
    const double alpha = 1.0;
    const double *A = input_A.LockedBuffer();
    const double *B = input_B.LockedBuffer();
    const CBLAS_INDEX lda = input_A.LDim();
    const CBLAS_INDEX ldb = input_B.LDim();
    double *C = output.Buffer();
    const CBLAS_INDEX ldc = output.LDim();
    // C := alpha * A^T * B + beta * C
    cblas_dgemm(layout, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta,
                C, ldc);
  }

  // output = input^T * input + beta * output (the uplo triangle)
  void syrk(const El::UpperOrLower uplo, const El::Matrix<double> &input,
            El::Matrix<double> &output, const double beta = 1.0)
  {
    // input: KxN matrix
    // output = input^T * input: NxN matrix
    ASSERT_EQUAL(input.Width(), output.Width());
    ASSERT_EQUAL(output.Height(), output.Width());

    CBLAS_LAYOUT layout = CblasColMajor;
    CBLAS_UPLO Uplo
      = uplo == El::UpperOrLowerNS::UPPER ? CblasUpper : CblasLower;
    CBLAS_TRANSPOSE Trans = CblasTrans;
    const CBLAS_INDEX N = input.Width();
    const CBLAS_INDEX K = input.Height();
    const double alpha = 1.0;
    const double *A = input.LockedBuffer();
    const CBLAS_INDEX lda = input.LDim();
    double *C = output.Buffer();
    const CBLAS_INDEX ldc = output.LDim();
    // C := alpha * A^T * A + beta * C
    cblas_dsyrk(layout, Uplo, Trans, N, K, alpha, A, lda, beta, C, ldc);
  }

  // job: calculate submatrix Q_IJ = P_I^T * P_J (module some prime)
  // I and J are column ranges of P
  void do_blas_job(const Blas_Job &job, const El::UpperOrLower uplo,
                   const Vertical_Block_Matrix_Residues_Window<double>
                     &input_block_residues_window_A,
                   const Vertical_Block_Matrix_Residues_Window<double>
                     &input_block_residues_window_B,
                   Matrix_Residues_Window<double> &output_residues_window)
  {
    const auto prime_index = job.prime_index;
    const auto I = job.I;
    const auto J = job.J;

    auto output_matrix
      = El::View(output_residues_window.residues.at(prime_index), I, J);

    switch(job.kind)
      {
        case Blas_Job::syrk: {
          // take whole columns
          const auto input_matrix = El::LockedView(
            input_block_residues_window_A.residues.at(prime_index), El::ALL,
            I);

          syrk(uplo, input_matrix, output_matrix);
          break;
        }
        case Blas_Job::gemm: {
          const auto input_A = El::LockedView(
            input_block_residues_window_A.residues.at(prime_index), El::ALL,
            I);
          const auto input_B = El::LockedView(
            input_block_residues_window_B.residues.at(prime_index), El::ALL,
            J);
          gemm(input_A, input_B, output_matrix);
          break;
        }
        default: {
          El::RuntimeError("Unexpected Blas_Job::Kind=", job.kind);
        }
      }
  }

  void do_blas_jobs(
    const El::UpperOrLower uplo, const Blas_Job::Kind kind,
    const BigInt_Shared_Memory_Syrk_Context::Job_Schedule &blas_job_schedule,
    const std::unique_ptr<Vertical_Block_Matrix_Residues_Window<double>>
      &input_grouped_block_residues_window_A,
    const std::unique_ptr<Vertical_Block_Matrix_Residues_Window<double>>
      &input_grouped_block_residues_window_B,
    const std::unique_ptr<Matrix_Residues_Window<double>>
      &output_residues_window,
    const El::mpi::Comm &shared_memory_comm, Timers &timers)
  {
    // Square each residue matrix
    {
      Scoped_Timer blas_timer(timers, "blas_jobs");
      const auto shmem_rank = shared_memory_comm.Rank();
      for(const auto &job : blas_job_schedule.jobs_by_rank.at(shmem_rank))
        {
          if(kind == Blas_Job::syrk && job.I.beg != job.J.beg)
            ASSERT_EQUAL(job.I.beg < job.J.beg, uplo == El::UPPER);
          do_blas_job(job, uplo, *input_grouped_block_residues_window_A,
                      kind == Blas_Job::syrk
                        ? *input_grouped_block_residues_window_A
                        : *input_grouped_block_residues_window_B,
                      *output_residues_window);
        }
    }
    {
      Scoped_Timer fence_timer(timers, "fence");
      output_residues_window->Fence();
    }
  }

  void update_block_timings_with_syrk(
    El::Matrix<int32_t> &block_timings_ms, const Scoped_Timer &syrk_timer,
    const std::vector<El::DistMatrix<El::BigFloat>> &bigint_input_matrix_blocks,
    const std::vector<size_t> &block_index_local_to_global,
    const El::mpi::Comm &shared_memory_comm,
    const Vertical_Block_Matrix_Residues_Window<double>
      &input_grouped_block_residues_window,
    const size_t total_block_height)
  {
    // Update block_timings_ms with syrk time.
    // We split total syrk time to individual blocks contributions, which are proportional to the block size.
    // To avoid double-counting, each core accounts only for the elements that it owns.
    // If block is distributed among several ranks,
    // then total block timing is a sum of contributions from all its ranks.
    constexpr auto get_size
      = [](auto &block) { return block.LocalHeight() * block.LocalWidth(); };

    // Estimate total CPU time spent on syrk for all blocks on the node,
    // assuming that time is similar for all ranks.
    const auto total_syrk_time
      = syrk_timer.elapsed_milliseconds() * shared_memory_comm.Size();
    // Total number of block elements on the node
    // NB: total_block_height > input_grouped_block_residues_window.height,
    // if split_factor > 1.
    const auto total_size
      = total_block_height * input_grouped_block_residues_window.width;
    // Average time spent on one block element
    const double time_per_element
      = static_cast<double>(total_syrk_time) / total_size;

    for(size_t local_block_index = 0;
        local_block_index < block_index_local_to_global.size();
        ++local_block_index)
      {
        const auto block_size
          = get_size(bigint_input_matrix_blocks.at(local_block_index));
        const double time = block_size * time_per_element;
        const auto global_block_index
          = block_index_local_to_global.at(local_block_index);
        block_timings_ms(global_block_index, 0) += std::round(time);
      }
  }
}

void BigInt_Shared_Memory_Syrk_Context::bigint_syrk_blas(
  El::UpperOrLower uplo,
  const std::vector<El::DistMatrix<El::BigFloat>> &bigint_input_matrix_blocks,
  El::DistMatrix<El::BigFloat> &bigint_output, Timers &timers,
  El::Matrix<int32_t> &block_timings_ms)
{
  // - Calculate Q = P^T P from all P blocks on the node.
  // - If we have only one node, then it's the global result (=bigint_output).
  // - If there are several nodes, we sum Q's from all nodes
  //   and write result to a global matrix bigint_output.

  Scoped_Timer timer(timers, "bigint_syrk_blas");

  ASSERT_EQUAL(bigint_output.Height(), bigint_output.Width());
  ASSERT(output_residues_window->width * output_window_split_factor
         >= bigint_output.Height());

  const auto output_ranges
    = split_range({0, bigint_output.Height()}, output_window_split_factor);

  for(size_t i = 0; i < output_window_split_factor; ++i)
    {
      const auto &I = output_ranges.at(i);
      for(size_t j = i; j < output_window_split_factor; ++j)
        {
          Scoped_Timer ij_timer(timers, El::BuildString("Q_", i, "_", j));
          const auto &J = output_ranges.at(j);
          auto bigint_output_submatrix = bigint_output(I, J);
          // Call BLAS to calculate residues for Q(I,J)
          bigint_syrk_blas_shmem_submatrix(uplo, bigint_input_matrix_blocks, I,
                                           J, timers, block_timings_ms);
          std::optional<El::UpperOrLower> uplo_opt;
          if(I.beg == J.beg)
            uplo_opt = uplo;
          // Restore Q from residues and reduce-scatter over all nodes
          restore_and_reduce(uplo_opt, bigint_output_submatrix, timers);
        }
    }
}

// Calculate contribution to Q_IJ = P_I^T P_J from all ranks of a single node
// (i.e. single shared memory window).
// bigint_output_shmem_submatrix is Q_IJ on the node
void BigInt_Shared_Memory_Syrk_Context::bigint_syrk_blas_shmem_submatrix(
  const El::UpperOrLower uplo,
  const std::vector<El::DistMatrix<El::BigFloat>> &bigint_input_matrix_blocks,
  const El::Range<El::Int> &output_I, const El::Range<El::Int> &output_J,
  Timers &timers, El::Matrix<int32_t> &block_timings_ms)
{
  Scoped_Timer timer(timers, "shmem");

  ASSERT(El::mpi::Congruent(shared_memory_comm,
                            input_grouped_block_residues_window_A->Comm()));
  ASSERT(
    El::mpi::Congruent(shared_memory_comm, output_residues_window->Comm()));

  ASSERT(input_window_split_factor > 0);

  const auto output_height = output_I.end - output_I.beg;
  const auto output_width = output_J.end - output_J.beg;

  ASSERT(input_grouped_block_residues_window_A->width >= output_height,
         DEBUG_STRING(input_grouped_block_residues_window_A->width),
         DEBUG_STRING(output_height));
  ASSERT(input_grouped_block_residues_window_A->width >= output_width,
         DEBUG_STRING(input_grouped_block_residues_window_A->width),
         DEBUG_STRING(output_width));

  // syrk for diagonal blocks of Q (Q_II = P_I^T P_I),
  // gemm for off-diagonal (Q_IJ = P_I^T P_J)
  const auto kind = output_I == output_J ? Blas_Job::syrk : Blas_Job::gemm;
  const auto &blas_job_schedule
    = get_blas_job_schedule(kind, uplo, output_height, output_width);

  // Clear input and output windows
  clear_residues(blas_job_schedule);

  // If input window is not big enough, we should fill input window
  // several times (taking different input block rows)
  // and call BLAS each time to update output window.
  for(size_t iter = 0; iter < input_window_split_factor; ++iter)
    {
      Scoped_Timer iter_timer(timers, "split_P_" + std::to_string(iter));

      // Compute block residues
      {
        // For each block group, compute residues and write them
        // to input residues window,
        // skipping first skip_rows rows
        const auto skip_rows = iter * input_group_height_per_prime();
        compute_block_residues(*input_grouped_block_residues_window_A,
                               bigint_input_matrix_blocks, skip_rows, output_I,
                               timers, block_timings_ms);
        if(output_I.beg != output_J.beg)
          {
            ASSERT(output_I.end != output_J.end);
            compute_block_residues(*input_grouped_block_residues_window_B,
                                   bigint_input_matrix_blocks, skip_rows,
                                   output_J, timers, block_timings_ms);
          }
        // TODO if input_split_factor == 1, we can reuse
        // same P_I for all P_J
      }

      // Square each residue matrix
      {
        Scoped_Timer syrk_timer(timers, "syrk");
        do_blas_jobs(uplo, kind, blas_job_schedule,
                     input_grouped_block_residues_window_A,
                     input_grouped_block_residues_window_B,
                     output_residues_window, shared_memory_comm, timers);
        update_block_timings_with_syrk(
          block_timings_ms, syrk_timer, bigint_input_matrix_blocks,
          block_index_local_to_global, shared_memory_comm,
          *input_grouped_block_residues_window_A, total_block_height_per_node);
      }
    }

  output_residues_window->Fence();
}

// #184: drivers that take a Band_Provider instead of persistent P blocks.
// Mirrors bigint_syrk_blas()/bigint_syrk_blas_shmem_submatrix() above; the only
// difference is that each input-window fill materializes its band on demand and
// frees it afterwards, so P's BigFloat form never exists whole.
void BigInt_Shared_Memory_Syrk_Context::bigint_syrk_blas_from_bands(
  const El::UpperOrLower uplo, const Band_Provider &band_provider,
  El::DistMatrix<El::BigFloat> &bigint_output, Timers &timers,
  El::Matrix<int32_t> &block_timings_ms,
  const std::optional<size_t> input_iterations)
{
  Scoped_Timer timer(timers, "bigint_syrk_blas");

  ASSERT_EQUAL(bigint_output.Height(), bigint_output.Width());
  ASSERT(output_residues_window->width * output_window_split_factor
         >= bigint_output.Height());

  const auto output_ranges
    = split_range({0, bigint_output.Height()}, output_window_split_factor);

  for(size_t i = 0; i < output_window_split_factor; ++i)
    {
      const auto &I = output_ranges.at(i);
      for(size_t j = i; j < output_window_split_factor; ++j)
        {
          Scoped_Timer ij_timer(timers, El::BuildString("Q_", i, "_", j));
          const auto &J = output_ranges.at(j);
          auto bigint_output_submatrix = bigint_output(I, J);
          bigint_syrk_blas_shmem_submatrix_from_bands(
            uplo, band_provider, I, J, timers, block_timings_ms,
            input_iterations.value_or(input_window_split_factor));
          std::optional<El::UpperOrLower> uplo_opt;
          if(I.beg == J.beg)
            uplo_opt = uplo;
          restore_row_offset = I.beg;
          restore_col_offset = J.beg;
          restore_and_reduce(uplo_opt, bigint_output_submatrix, timers);
          restore_row_offset = 0;
          restore_col_offset = 0;
        }
    }
}

void BigInt_Shared_Memory_Syrk_Context::
  bigint_syrk_blas_shmem_submatrix_from_bands(
    const El::UpperOrLower uplo, const Band_Provider &band_provider,
    const El::Range<El::Int> &output_I, const El::Range<El::Int> &output_J,
    Timers &timers, El::Matrix<int32_t> &block_timings_ms,
    const size_t input_iterations)
{
  Scoped_Timer timer(timers, "shmem");

  ASSERT(El::mpi::Congruent(shared_memory_comm,
                            input_grouped_block_residues_window_A->Comm()));
  ASSERT(
    El::mpi::Congruent(shared_memory_comm, output_residues_window->Comm()));

  ASSERT(input_window_split_factor > 0);

  const auto output_height = output_I.end - output_I.beg;
  const auto output_width = output_J.end - output_J.beg;

  ASSERT(input_grouped_block_residues_window_A->width >= output_height,
         DEBUG_STRING(input_grouped_block_residues_window_A->width),
         DEBUG_STRING(output_height));
  ASSERT(input_grouped_block_residues_window_A->width >= output_width,
         DEBUG_STRING(input_grouped_block_residues_window_A->width),
         DEBUG_STRING(output_width));

  const auto kind = output_I == output_J ? Blas_Job::syrk : Blas_Job::gemm;
  const auto &blas_job_schedule
    = get_blas_job_schedule(kind, uplo, output_height, output_width);

  clear_residues(blas_job_schedule);

  for(size_t iter = 0; iter < input_iterations; ++iter)
    {
      Scoped_Timer iter_timer(timers, "split_P_" + std::to_string(iter));

      // Materialize this band and compute its residues.
      // The band matrices are already sliced to (rows x col_range),
      // so compute_block_residues() consumes them with no extra offsets.
      std::vector<El::DistMatrix<El::BigFloat>> bands;
      {
        const auto skip_rows
          = static_cast<El::Int>(iter * input_group_height_per_prime());
        {
          Scoped_Timer band_timer(timers, "materialize_band");
          band_provider(skip_rows, input_group_height_per_prime(), output_I,
                        bands);
        }
        compute_block_residues(*input_grouped_block_residues_window_A, bands,
                               0, El::Range<El::Int>(0, output_height),
                               timers, block_timings_ms);
        if(output_I.beg != output_J.beg)
          {
            ASSERT(output_I.end != output_J.end);
            std::vector<El::DistMatrix<El::BigFloat>> bands_B;
            {
              Scoped_Timer band_timer(timers, "materialize_band");
              band_provider(skip_rows, input_group_height_per_prime(),
                            output_J, bands_B);
            }
            compute_block_residues(*input_grouped_block_residues_window_B,
                                   bands_B, 0,
                                   El::Range<El::Int>(0, output_width), timers,
                                   block_timings_ms);
          }
      }

      {
        Scoped_Timer syrk_timer(timers, "syrk");
        do_blas_jobs(uplo, kind, blas_job_schedule,
                     input_grouped_block_residues_window_A,
                     input_grouped_block_residues_window_B,
                     output_residues_window, shared_memory_comm, timers);
        update_block_timings_with_syrk(
          block_timings_ms, syrk_timer, bands, block_index_local_to_global,
          shared_memory_comm, *input_grouped_block_residues_window_A,
          total_block_height_per_node);
      }
    }

  output_residues_window->Fence();
}

// #184: band accumulation (see the header). Q's residues are summed over all bands in
// private tiles instead of being restored band by band.

namespace
{
  El::Int tile_column_length(const Blas_Job &job, const El::Int col)
  {
    return job.kind == Blas_Job::syrk ? col + 1 : job.I.end - job.I.beg;
  }
}

bool BigInt_Shared_Memory_Syrk_Context::can_accumulate_bands(
  const El::DistMatrix<El::BigFloat> &output) const
{
  return full_width_input
         && El::mpi::Congruent(shared_memory_comm, output.DistComm());
}

void BigInt_Shared_Memory_Syrk_Context::begin_band_accumulation(
  const El::DistMatrix<El::BigFloat> &output)
{
  ASSERT(can_accumulate_bands(output),
         "band accumulation needs a single node and a full-width input window");
  ASSERT_EQUAL(output.Height(), output.Width());
  // The float accumulator holds a residue exactly only below 2^24.
  for(const auto prime : comb.primes)
    ASSERT(prime < (mp_limb_t(1) << 24), "band accumulation: prime ", prime,
           " >= 2^24 cannot be stored as float");
  constexpr auto uplo = El::UPPER;
  const auto ranges
    = split_range({0, output.Height()}, output_window_split_factor);
  const auto window_height
    = static_cast<size_t>(output_residues_window->height);
  const auto rank = shared_memory_comm.Rank();

  accumulator_blocks.clear();
  accumulator_tiles.clear();
  for(size_t i = 0; i < ranges.size(); ++i)
    for(size_t j = i; j < ranges.size(); ++j)
      {
        auto &block = accumulator_blocks.emplace_back();
        block.I = ranges.at(i);
        block.J = ranges.at(j);
        // This rank's elements of the upper triangle, column by column.
        for(El::Int col = block.J.beg; col < block.J.end; ++col)
          {
            if(!output.IsLocalCol(col))
              continue;
            const El::Int row_end = std::min(block.I.end, col + 1);
            for(El::Int row = block.I.beg; row < row_end; ++row)
              {
                if(!output.IsLocalRow(row))
                  continue;
                auto &runs = block.runs;
                if(!runs.empty() && runs.back().col == col
                   && runs.back().row_begin + runs.back().length == row)
                  {
                    ++runs.back().length;
                  }
                else
                  {
                    runs.push_back(
                      {row, col, 1,
                       static_cast<size_t>(row - block.I.beg)
                         + static_cast<size_t>(col - block.J.beg)
                             * window_height});
                  }
              }
          }
      }
  // This rank's BLAS jobs (Accumulator_Tile). A job's product goes through the scratch
  // in strips of columns (accumulate_band()), at most scratch_cap entries but at
  // least a whole column.
  constexpr size_t scratch_cap = size_t(1) << 20;
  size_t accumulator_size = 0;
  size_t scratch_size = 0;
  const auto &schedule = get_blas_job_schedule(
    Blas_Job::syrk, uplo, output.Height(), output.Width());
  for(const auto &job : schedule.jobs_by_rank.at(rank))
    {
      const El::Int height = job.I.end - job.I.beg;
      const El::Int width = job.J.end - job.J.beg;
      if(job.kind == Blas_Job::syrk)
        ASSERT_EQUAL(height, width);
      accumulator_tiles.push_back({job, accumulator_size});
      for(El::Int col = 0; col < width; ++col)
        accumulator_size += static_cast<size_t>(tile_column_length(job, col));
      scratch_size = std::max(
        {scratch_size, std::min(static_cast<size_t>(height * width), scratch_cap),
         static_cast<size_t>(height)});
    }
  accumulator.assign(accumulator_size, 0.0f);
  tile_scratch.assign(scratch_size, 0.0);

  // Where the band's values are staged (stage_band_values()): mpf writes at most
  // _mp_prec + 1 limbs.
  {
    mpf_t probe;
    mpf_init2(probe, El::gmp::Precision());
    band_value_limbs = probe->_mp_prec + 1;
    mpf_clear(probe);
    band_value_record = 2 + band_value_limbs;
    const auto &input_residues
      = input_grouped_block_residues_window_A->residues.at(0);
    const size_t size = static_cast<size_t>(input_residues.Height())
                        * input_residues.Width() * band_value_record;
    static_assert(sizeof(double) == sizeof(mp_limb_t));
    if(output_residues_window->size() >= size)
      band_values = reinterpret_cast<mp_limb_t *>(
        output_residues_window->window_view.data());
    else
      {
        band_values_window = std::make_unique<Shared_Window_Array<mp_limb_t>>(
          shared_memory_comm, size);
        band_values = band_values_window->data();
      }
  }
  band_accumulation_profile = {};
}

void BigInt_Shared_Memory_Syrk_Context::fence_band_values()
{
  if(band_values_window)
    band_values_window->Fence();
  else
    output_residues_window->Fence();
}

// The values of this rank's part of the band at their rows and columns of the input
// window: the rows of the rank's MPI group, its blocks in order, then zeros to the
// group's height (as compute_block_residues()).
void BigInt_Shared_Memory_Syrk_Context::stage_band_values(
  const std::vector<El::DistMatrix<El::BigFloat>> &bigint_band_blocks)
{
  const auto &input = *input_grouped_block_residues_window_A;
  const auto &residues = input.residues.at(0);
  const size_t height = residues.Height();
  mp_limb_t *const values = band_values;
  const auto record = [&](const El::Int row, const El::Int col) {
    return values
           + (static_cast<size_t>(row) + static_cast<size_t>(col) * height)
               * band_value_record;
  };
  const El::Int group_begin
    = input.block_residues.at(0).at(group_index).LockedBuffer()
      - residues.LockedBuffer();
  const El::Int group_end = group_begin + input_group_height_per_prime();
  El::Int row = group_begin;
  for(const auto &block : bigint_band_blocks)
    {
      for(El::Int jLoc = 0; jLoc < block.LocalWidth(); ++jLoc)
        {
          const El::Int col = block.GlobalCol(jLoc);
          for(El::Int iLoc = 0; iLoc < block.LocalHeight(); ++iLoc)
            {
              mpf_srcptr x = block.GetLocalCRef(iLoc, jLoc).gmp_float.get_mpf_t();
              const size_t limbs = std::abs(x->_mp_size);
              ASSERT(limbs <= band_value_limbs, DEBUG_STRING(limbs),
                     DEBUG_STRING(band_value_limbs));
              mp_limb_t *r = record(row + block.GlobalRow(iLoc), col);
              r[0] = static_cast<mp_limb_t>(static_cast<long>(x->_mp_size));
              r[1] = static_cast<mp_limb_t>(static_cast<long>(x->_mp_exp));
              std::copy(x->_mp_d, x->_mp_d + limbs, r + 2);
            }
        }
      row += block.Height();
    }
  ASSERT(row <= group_end, DEBUG_STRING(row), DEBUG_STRING(group_end));
  // Zeros: written for each column by its owners in the first grid row.
  const auto &first = bigint_band_blocks.at(0);
  if(first.ColRank() == 0)
    for(El::Int jLoc = 0; jLoc < first.LocalWidth(); ++jLoc)
      for(El::Int zero_row = row; zero_row < group_end; ++zero_row)
        {
          mp_limb_t *r = record(zero_row, first.GlobalCol(jLoc));
          r[0] = 0;
          r[1] = 0;
        }
  fence_band_values();
}

// Residues of this rank's share of the staged band's columns, into the input window.
// Block timings: each local block is charged the time per entry measured here times its
// own entries, as if the rank had computed the residues of its own part.
void BigInt_Shared_Memory_Syrk_Context::compute_staged_band_residues(
  const std::vector<El::DistMatrix<El::BigFloat>> &bigint_band_blocks,
  El::Matrix<int32_t> &block_timings_ms)
{
  const auto start = std::chrono::steady_clock::now();
  auto &input = *input_grouped_block_residues_window_A;
  const El::Int height = input.residues.at(0).Height();
  const El::Int width = input.residues.at(0).Width();
  const mp_limb_t *const values = band_values;
  const auto rank = static_cast<El::Int>(shared_memory_comm.Rank());
  const auto num_ranks = static_cast<El::Int>(shared_memory_comm.Size());
  const El::Int col_begin = width * rank / num_ranks;
  const El::Int col_end = width * (rank + 1) / num_ranks;
  std::vector<double> column(static_cast<size_t>(height) * comb.num_primes);
  Fmpz_BigInt value;
  for(El::Int col = col_begin; col < col_end; ++col)
    {
      for(El::Int row = 0; row < height; ++row)
        {
          const mp_limb_t *r
            = values
              + (static_cast<size_t>(row) + static_cast<size_t>(col) * height)
                  * band_value_record;
          __mpf_struct x;
          x._mp_prec = static_cast<int>(band_value_limbs) - 1;
          x._mp_size = static_cast<int>(static_cast<long>(r[0]));
          x._mp_exp = static_cast<long>(r[1]);
          x._mp_d = const_cast<mp_limb_t *>(r + 2);
          fmpz_set_mpf(value.value, &x);
          bigint_to_residues(value, comb, column.data() + row, height);
        }
      for(size_t prime_index = 0; prime_index < comb.num_primes; ++prime_index)
        std::memcpy(input.residues.at(prime_index).Buffer(0, col),
                    column.data() + prime_index * height,
                    sizeof(double) * height);
    }
  const double entries = static_cast<double>(height) * (col_end - col_begin);
  if(entries > 0)
    {
      const double ms_per_entry
        = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start)
            .count()
          / entries;
      ASSERT_EQUAL(bigint_band_blocks.size(),
                   block_index_local_to_global.size());
      for(size_t block = 0; block < bigint_band_blocks.size(); ++block)
        {
          const auto &matrix = bigint_band_blocks.at(block);
          block_timings_ms(block_index_local_to_global.at(block), 0)
            += std::round(ms_per_entry * matrix.LocalHeight()
                          * matrix.LocalWidth());
        }
    }
  input.Fence();
}

void BigInt_Shared_Memory_Syrk_Context::reset_band_accumulation()
{
  ASSERT(!accumulator_blocks.empty(), "begin_band_accumulation() not called");
  std::fill(accumulator.begin(), accumulator.end(), 0.0f);
  band_accumulation_profile = {};
}

void BigInt_Shared_Memory_Syrk_Context::accumulate_band(
  const std::vector<El::DistMatrix<El::BigFloat>> &bigint_band_blocks,
  Timers &timers, El::Matrix<int32_t> &block_timings_ms)
{
  Scoped_Timer timer(timers, "accumulate_band");
  ASSERT(!accumulator_blocks.empty(), "begin_band_accumulation() not called");
  constexpr auto uplo = El::UPPER;
  const auto &input = *input_grouped_block_residues_window_A;

  using clock = std::chrono::steady_clock;
  auto &profile = band_accumulation_profile;
  const auto seconds_since = [](const clock::time_point &t) {
    return std::chrono::duration<double>(clock::now() - t).count();
  };

  // Residues of the whole band, once, into the full-width input window: its values
  // staged by their owners, the residues computed by all ranks evenly.
  auto t0 = clock::now();
  stage_band_values(bigint_band_blocks);
  compute_staged_band_residues(bigint_band_blocks, block_timings_ms);
  profile.residues += seconds_since(t0);

  {
    Scoped_Timer syrk_timer(timers, "syrk_and_reduce");
    constexpr double round_magic = 6755399441055744.0; // 2^52 + 2^51
    for(const auto &tile : accumulator_tiles)
      {
        const auto &job = tile.job;
        const auto &residues = input.residues.at(job.prime_index);
        const El::Int height = job.I.end - job.I.beg;
        const El::Int tile_width = job.J.end - job.J.beg;
        const double prime = comb.primes.at(job.prime_index);
        const double prime_inverse = 1.0 / prime;
        float *acc = accumulator.data() + tile.offset;
        // As do_blas_job(), into this rank's scratch, in strips of columns [c0, c1)
        // of the job: of a syrk job, rows [0, c0) by gemm and the square [c0, c1)
        // by syrk.
        const El::Int strip = std::max<El::Int>(
          1, static_cast<El::Int>(tile_scratch.size()) / height);
        for(El::Int c0 = 0; c0 < tile_width; c0 += strip)
          {
            const El::Int c1 = std::min(tile_width, c0 + strip);
            const El::Int rows = job.kind == Blas_Job::syrk ? c1 : height;
            t0 = clock::now();
            El::Matrix<double> product;
            product.Attach(rows, c1 - c0, tile_scratch.data(), rows);
            if(job.kind == Blas_Job::syrk)
              {
                const El::IR columns(job.I.beg + c0, job.I.beg + c1);
                if(c0 > 0)
                  {
                    El::Matrix<double> above;
                    El::View(above, product, El::IR(0, c0), El::ALL);
                    gemm(El::LockedView(residues, El::ALL,
                                        El::IR(job.I.beg, job.I.beg + c0)),
                         El::LockedView(residues, El::ALL, columns), above,
                         0.0);
                  }
                El::Matrix<double> square;
                El::View(square, product, El::IR(c0, c1), El::ALL);
                syrk(uplo, El::LockedView(residues, El::ALL, columns), square,
                     0.0);
              }
            else
              gemm(El::LockedView(residues, El::ALL, job.I),
                   El::LockedView(residues, El::ALL,
                                  El::IR(job.J.beg + c0, job.J.beg + c1)),
                   product, 0.0);
            profile.blas += seconds_since(t0);
            t0 = clock::now();
            // acc := (acc + product mod prime) mod prime. Per band, |product| <=
            // (input window height) * prime^2 / 4, far below 2^51, so the rounding
            // below is exact.
            for(El::Int col = c0; col < c1; ++col)
              {
                const double *x = product.LockedBuffer(0, col - c0);
                const El::Int length = tile_column_length(job, col);
                for(El::Int k = 0; k < length; ++k)
                  {
                    const double q
                      = (x[k] * prime_inverse + round_magic) - round_magic;
                    double sum = acc[k] + (x[k] - q * prime);
                    sum = sum < 0 ? sum + prime : sum;
                    sum = sum >= prime ? sum - prime : sum;
                    acc[k] = static_cast<float>(sum);
                  }
                acc += length;
              }
            profile.reduce += seconds_since(t0);
          }
      }
  }
  // Nobody writes the next band's residues before everyone has read this band.
  t0 = clock::now();
  {
    Scoped_Timer fence_timer(timers, "fence");
    input_grouped_block_residues_window_A->Fence();
  }
  profile.fence += seconds_since(t0);
}

void BigInt_Shared_Memory_Syrk_Context::restore_band_accumulation(
  const std::function<void(El::Int, El::Int, const Fmpz_BigInt &)> &callback,
  Timers &timers)
{
  Scoped_Timer timer(timers, "restore_band_accumulation");
  constexpr int sign = 1; // negative values allowed
  std::vector<mp_limb_t> residues(comb.num_primes);
  Fmpz_BigInt value;
  for(size_t block_index = 0; block_index < accumulator_blocks.size();
      ++block_index)
    {
      const auto &block = accumulator_blocks.at(block_index);
      // This rank's tiles, where they overlap the block, into the output window...
      for(const auto &tile : accumulator_tiles)
        {
          const auto &job = tile.job;
          if(job.I.end <= block.I.beg || job.I.beg >= block.I.end
             || job.J.end <= block.J.beg || job.J.beg >= block.J.end)
            continue;
          auto &window = output_residues_window->residues.at(job.prime_index);
          const float *acc = accumulator.data() + tile.offset;
          for(El::Int col = 0; col < job.J.end - job.J.beg; ++col)
            {
              const El::Int global_col = job.J.beg + col;
              const El::Int length = tile_column_length(job, col);
              if(global_col >= block.J.beg && global_col < block.J.end)
                {
                  const El::Int row_begin = std::max(job.I.beg, block.I.beg);
                  const El::Int row_end
                    = std::min(job.I.beg + length, block.I.end);
                  if(row_begin < row_end)
                    {
                      double *x = window.Buffer(row_begin - block.I.beg,
                                                global_col - block.J.beg);
                      const float *a = acc + (row_begin - job.I.beg);
                      for(El::Int k = 0; k < row_end - row_begin; ++k)
                        x[k] = a[k];
                    }
                }
              acc += length;
            }
        }
      output_residues_window->Fence();
      // ...and the elements of Q that this rank holds, out of it.
      for(const auto &run : accumulator_blocks.at(block_index).runs)
        for(El::Int k = 0; k < run.length; ++k)
          {
            for(size_t prime_index = 0; prime_index < comb.num_primes;
                ++prime_index)
              residues.at(prime_index) = static_cast<mp_limb_t>(
                output_residues_window->residues.at(prime_index)
                  .LockedBuffer()[run.window_offset + k]);
            fmpz_multi_CRT_ui(value.value, residues.data(), comb.comb,
                              comb.comb_temp, sign);
            callback(run.row_begin + k, run.col, value);
          }
      // Nobody writes the next block before everyone has read this one.
      output_residues_window->Fence();
    }
  accumulator_blocks.clear();
  accumulator_tiles.clear();
  accumulator.clear();
  accumulator.shrink_to_fit();
  tile_scratch.clear();
  tile_scratch.shrink_to_fit();
  band_values = nullptr;
  band_values_window.reset();
}
