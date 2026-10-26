#include "../BigInt_Shared_Memory_Syrk_Context.hxx"
#include "sdpb_util/bigint_shared_memory/fmpz/Fmpz_BigInt.hxx"
#include "sdpb_util/assert.hxx"
#include "sdpb_util/split_range.hxx"

#include <cblas.h>

#include <chrono>

namespace
{
  // output = inputA^T * inputB
  void gemm(const El::Matrix<double> &input_A,
            const El::Matrix<double> &input_B, El::Matrix<double> &output)
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
    const double beta = 1.0;
    double *C = output.Buffer();
    const CBLAS_INDEX ldc = output.LDim();
    // C := alpha * A^T * B + beta * C = (in our case) A^T * B + C
    cblas_dgemm(layout, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta,
                C, ldc);
  }

  // output = input^T * input
  void syrk(const El::UpperOrLower uplo, const El::Matrix<double> &input,
            El::Matrix<double> &output)
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
    const double beta = 1.0;
    double *C = output.Buffer();
    const CBLAS_INDEX ldc = output.LDim();
    // C := alpha * A^T * A + beta * C = (in our case) A^T * A + C
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

// #184: band accumulation (see the header). Q's residues are summed over all bands in a
// private accumulator instead of being restored band by band.

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
  const auto ranges
    = split_range({0, output.Height()}, output_window_split_factor);
  const auto window_height
    = static_cast<size_t>(output_residues_window->height);

  accumulator_blocks.clear();
  accumulator_elements = 0;
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
                             * window_height,
                       accumulator_elements});
                  }
                ++accumulator_elements;
              }
          }
      }
  accumulator.assign(accumulator_elements * comb.num_primes, 0.0f);
  band_accumulation_profile = {};
}

void BigInt_Shared_Memory_Syrk_Context::reset_band_accumulation()
{
  ASSERT(!accumulator_blocks.empty(), "begin_band_accumulation() not called");
  std::fill(accumulator.begin(), accumulator.end(), 0.0f);
  band_accumulation_profile = {};
}

// Zero this rank's parts of the output window (the Q_IJ of its BLAS jobs).
void BigInt_Shared_Memory_Syrk_Context::clear_output_residues(
  const Job_Schedule &blas_job_schedule)
{
  const auto rank = shared_memory_comm.Rank();
  for(const auto &job : blas_job_schedule.jobs_by_rank.at(rank))
    {
      auto &output_matrix
        = output_residues_window->residues.at(job.prime_index);
      El::Matrix<double> submatrix;
      El::View(submatrix, output_matrix, job.I, job.J);
      El::Zero(submatrix);
    }
  output_residues_window->Fence();
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

  // Residues of the whole band, once, into the full-width input window.
  auto t0 = clock::now();
  const El::Int width = accumulator_blocks.back().J.end;
  compute_block_residues(*input_grouped_block_residues_window_A,
                         bigint_band_blocks, 0, {0, width}, timers,
                         block_timings_ms);
  profile.residues += seconds_since(t0);

  const auto rank = shared_memory_comm.Rank();
  for(const auto &block : accumulator_blocks)
    {
      const auto height = block.I.end - block.I.beg;
      const auto block_width = block.J.end - block.J.beg;
      const auto kind = block.I == block.J ? Blas_Job::syrk : Blas_Job::gemm;
      const auto &schedule
        = get_blas_job_schedule(kind, uplo, height, block_width);
      t0 = clock::now();
      {
        Scoped_Timer clear_timer(timers, "clear_output");
        clear_output_residues(schedule);
      }
      profile.clear += seconds_since(t0);
      t0 = clock::now();
      {
        // As do_blas_job(), with the job's columns offset to the block's place in
        // the full-width input.
        Scoped_Timer syrk_timer(timers, "syrk");
        for(const auto &job : schedule.jobs_by_rank.at(rank))
          {
            const auto &residues = input.residues.at(job.prime_index);
            auto output_matrix = El::View(
              output_residues_window->residues.at(job.prime_index), job.I,
              job.J);
            const El::Range<El::Int> I(block.I.beg + job.I.beg,
                                       block.I.beg + job.I.end);
            const El::Range<El::Int> J(block.J.beg + job.J.beg,
                                       block.J.beg + job.J.end);
            if(job.kind == Blas_Job::syrk)
              syrk(uplo, El::LockedView(residues, El::ALL, I), output_matrix);
            else
              gemm(El::LockedView(residues, El::ALL, I),
                   El::LockedView(residues, El::ALL, J), output_matrix);
          }
        output_residues_window->Fence();
      }
      profile.blas += seconds_since(t0);
      t0 = clock::now();
      {
        // acc := (acc + Q_IJ mod prime) mod prime. Per band, |Q_IJ| <= height * prime^2 / 4,
        // far below 2^51, so the rounding below is exact.
        Scoped_Timer reduce_timer(timers, "reduce");
        constexpr double round_magic = 6755399441055744.0; // 2^52 + 2^51
        for(size_t prime_index = 0; prime_index < comb.num_primes;
            ++prime_index)
          {
            const double prime = comb.primes.at(prime_index);
            const double prime_inverse = 1.0 / prime;
            const double *window
              = output_residues_window->residues.at(prime_index)
                  .LockedBuffer();
            float *acc
              = accumulator.data() + prime_index * accumulator_elements;
            for(const auto &run : block.runs)
              {
                const double *x = window + run.window_offset;
                float *a = acc + run.accumulator_offset;
                for(El::Int k = 0; k < run.length; ++k)
                  {
                    const double q
                      = (x[k] * prime_inverse + round_magic) - round_magic;
                    double sum = a[k] + (x[k] - q * prime);
                    sum = sum < 0 ? sum + prime : sum;
                    sum = sum >= prime ? sum - prime : sum;
                    a[k] = static_cast<float>(sum);
                  }
              }
          }
        // Nobody clears the window for the next block before everyone has read it.
        output_residues_window->Fence();
      }
      profile.reduce += seconds_since(t0);
    }
}

void BigInt_Shared_Memory_Syrk_Context::restore_band_accumulation(
  const std::function<void(El::Int, El::Int, const Fmpz_BigInt &)> &callback,
  Timers &timers)
{
  Scoped_Timer timer(timers, "restore_band_accumulation");
  constexpr int sign = 1; // negative values allowed
  std::vector<mp_limb_t> residues(comb.num_primes);
  Fmpz_BigInt value;
  for(const auto &block : accumulator_blocks)
    for(const auto &run : block.runs)
      for(El::Int k = 0; k < run.length; ++k)
        {
          const size_t element = run.accumulator_offset + k;
          for(size_t prime_index = 0; prime_index < comb.num_primes;
              ++prime_index)
            residues.at(prime_index) = static_cast<mp_limb_t>(
              accumulator.at(prime_index * accumulator_elements + element));
          fmpz_multi_CRT_ui(value.value, residues.data(), comb.comb,
                            comb.comb_temp, sign);
          callback(run.row_begin + k, run.col, value);
        }
  accumulator_blocks.clear();
  accumulator.clear();
  accumulator.shrink_to_fit();
  accumulator_elements = 0;
}
