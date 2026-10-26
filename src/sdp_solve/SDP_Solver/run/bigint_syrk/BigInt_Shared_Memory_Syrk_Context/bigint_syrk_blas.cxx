#include "../BigInt_Shared_Memory_Syrk_Context.hxx"
#include "../fmpz/Fmpz_BigInt.hxx"
#include "sdpb_util/assert.hxx"
#include "sdpb_util/split_range.hxx"

#include <cblas.h>

#include <algorithm>
#include <cstdlib>
#include <string>

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
  void do_blas_job(
    const Blas_Job &job, const El::UpperOrLower uplo,
    const Block_Residue_Matrices_Window<double> &input_block_residues_window_A,
    const Block_Residue_Matrices_Window<double> &input_block_residues_window_B,
    Residue_Matrices_Window<double> &output_residues_window)
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

  El::Range<El::Int>
  intersect(const El::Range<El::Int> &a, const El::Range<El::Int> &b)
  {
    return {std::max(a.beg, b.beg), std::min(a.end, b.end)};
  }

  // --compressLinvB: the same job, but each row segment of the window is
  // multiplied on its nonzero column runs only. The products left out are
  // products of zero residues. All sums are exact integers in double (the
  // primes are chosen for the full window height), so Q_IJ is unchanged.
  // offset_A, offset_B: global column of window column 0 of A, B.
  void do_blas_job_on_segments(
    const Blas_Job &job, const El::UpperOrLower uplo,
    const Block_Residue_Matrices_Window<double> &input_block_residues_window_A,
    const Block_Residue_Matrices_Window<double> &input_block_residues_window_B,
    Residue_Matrices_Window<double> &output_residues_window,
    const std::vector<Window_Row_Segment> &segments, const El::Int offset_A,
    const El::Int offset_B)
  {
    const auto prime_index = job.prime_index;
    const auto &residues_A
      = input_block_residues_window_A.residues.at(prime_index);
    const auto &residues_B
      = input_block_residues_window_B.residues.at(prime_index);
    auto &output = output_residues_window.residues.at(prime_index);
    const El::Range<El::Int> I(job.I.beg + offset_A, job.I.end + offset_A);
    const El::Range<El::Int> J(job.J.beg + offset_B, job.J.end + offset_B);
    // global column range -> window columns
    const auto local = [](const El::Range<El::Int> &range,
                          const El::Int offset) -> El::Range<El::Int> {
      return {range.beg - offset, range.end - offset};
    };
    for(const auto &segment : segments)
      {
        const auto &runs = segment.runs;
        for(size_t r = 0; r < runs.size(); ++r)
          {
            const auto a = intersect(I, runs[r]);
            if(a.beg >= a.end)
              continue;
            const auto input_a
              = El::LockedView(residues_A, segment.rows, local(a, offset_A));
            if(job.kind == Blas_Job::syrk)
              {
                // Q_II: the diagonal block of each run, and the products of
                // two runs on the uplo side
                auto output_aa
                  = El::View(output, local(a, offset_A), local(a, offset_A));
                syrk(uplo, input_a, output_aa);
                for(size_t s_run = 0; s_run < runs.size(); ++s_run)
                  {
                    if(uplo == El::UPPER ? s_run <= r : s_run >= r)
                      continue;
                    const auto b = intersect(I, runs[s_run]);
                    if(b.beg >= b.end)
                      continue;
                    const auto input_b = El::LockedView(
                      residues_A, segment.rows, local(b, offset_A));
                    auto output_ab = El::View(output, local(a, offset_A),
                                              local(b, offset_A));
                    gemm(input_a, input_b, output_ab);
                  }
              }
            else
              {
                ASSERT_EQUAL(job.kind, Blas_Job::gemm);
                for(const auto &run_s : runs)
                  {
                    const auto b = intersect(J, run_s);
                    if(b.beg >= b.end)
                      continue;
                    const auto input_b = El::LockedView(
                      residues_B, segment.rows, local(b, offset_B));
                    auto output_ab = El::View(output, local(a, offset_A),
                                              local(b, offset_B));
                    gemm(input_a, input_b, output_ab);
                  }
              }
          }
      }
  }

  void
  do_blas_jobs(const El::UpperOrLower uplo, const Blas_Job::Kind kind,
               const Blas_Job_Schedule &blas_job_schedule,
               const std::unique_ptr<Block_Residue_Matrices_Window<double>>
                 &input_grouped_block_residues_window_A,
               const std::unique_ptr<Block_Residue_Matrices_Window<double>>
                 &input_grouped_block_residues_window_B,
               const std::unique_ptr<Residue_Matrices_Window<double>>
                 &output_residues_window,
               const El::mpi::Comm &shared_memory_comm, Timers &timers,
               const std::vector<Window_Row_Segment> *segments = nullptr,
               const El::Int offset_A = 0, const El::Int offset_B = 0)
  {
    // Square each residue matrix
    {
      Scoped_Timer blas_timer(timers, "blas_jobs");
      const auto shmem_rank = shared_memory_comm.Rank();
      for(const auto &job : blas_job_schedule.jobs_by_rank.at(shmem_rank))
        {
          if(kind == Blas_Job::syrk && job.I.beg != job.J.beg)
            ASSERT_EQUAL(job.I.beg < job.J.beg, uplo == El::UPPER);
          const auto &window_B = kind == Blas_Job::syrk
                                   ? *input_grouped_block_residues_window_A
                                   : *input_grouped_block_residues_window_B;
          if(segments != nullptr)
            do_blas_job_on_segments(job, uplo,
                                    *input_grouped_block_residues_window_A,
                                    window_B, *output_residues_window,
                                    *segments, offset_A, offset_B);
          else
            do_blas_job(job, uplo, *input_grouped_block_residues_window_A,
                        window_B, *output_residues_window);
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
    const Block_Residue_Matrices_Window<double>
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
  El::Matrix<int32_t> &block_timings_ms,
  const std::vector<std::vector<El::Int>> *column_maps)
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

  // --compressLinvB: global column of P -> column of each block, or -1
  std::vector<std::vector<El::Int>> column_inverse_maps;
  if(column_maps != nullptr)
    {
      ASSERT_EQUAL(column_maps->size(), bigint_input_matrix_blocks.size());
      for(const auto &column_map : *column_maps)
        {
          auto &inverse = column_inverse_maps.emplace_back(
            bigint_output.Height(), -1);
          for(size_t k = 0; k < column_map.size(); ++k)
            if(column_map[k] >= 0)
              inverse.at(column_map[k]) = k;
        }
    }
  const auto *inverse_maps
    = column_maps == nullptr ? nullptr : &column_inverse_maps;
  // ... and the BLAS jobs multiply only the nonzero column runs of each
  // block (SDPB_COMPRESS_SYRK=0: full width, for comparison)
  std::vector<std::vector<Block_Column_Runs>> node_column_runs;
  const char *compress_syrk_env = std::getenv("SDPB_COMPRESS_SYRK");
  const bool compress_syrk
    = column_maps != nullptr
      && !(compress_syrk_env != nullptr
           && std::string(compress_syrk_env) == "0");
  if(compress_syrk)
    {
      Scoped_Timer runs_timer(timers, "gather_column_runs");
      node_column_runs = gather_node_column_runs(
        bigint_input_matrix_blocks, *column_maps, bigint_output.Height());
    }

  for(size_t i = 0; i < output_window_split_factor; ++i)
    {
      const auto &I = output_ranges.at(i);
      for(size_t j = i; j < output_window_split_factor; ++j)
        {
          Scoped_Timer ij_timer(timers, El::BuildString("Q_", i, "_", j));
          const auto &J = output_ranges.at(j);
          auto bigint_output_submatrix = bigint_output(I, J);
          // Call BLAS to calculate residues for Q(I,J)
          bigint_syrk_blas_shmem_submatrix(
            uplo, bigint_input_matrix_blocks, I, J, timers, block_timings_ms,
            inverse_maps, compress_syrk ? &node_column_runs : nullptr);
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
  Timers &timers, El::Matrix<int32_t> &block_timings_ms,
  const std::vector<std::vector<El::Int>> *column_inverse_maps,
  const std::vector<std::vector<Block_Column_Runs>> *node_column_runs)
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
  // --compressLinvB: balance the work on the column runs (job costs);
  // SDPB_COMPRESS_SYRK_SCHEDULE=0 keeps the dense schedule
  const char *weighted_env = std::getenv("SDPB_COMPRESS_SYRK_SCHEDULE");
  const bool weighted
    = node_column_runs != nullptr
      && !(weighted_env != nullptr && std::string(weighted_env) == "0");
  const auto blas_job_schedule
    = weighted ? get_weighted_blas_job_schedule(kind, uplo, output_I, output_J,
                                                *node_column_runs)
               : get_blas_job_schedule(kind, uplo, output_height, output_width);

  // Clear input and output windows
  clear_residues(*blas_job_schedule);

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
                               timers, block_timings_ms, column_inverse_maps);
        if(output_I.beg != output_J.beg)
          {
            ASSERT(output_I.end != output_J.end);
            compute_block_residues(*input_grouped_block_residues_window_B,
                                   bigint_input_matrix_blocks, skip_rows,
                                   output_J, timers, block_timings_ms,
                                   column_inverse_maps);
          }
        // TODO if input_split_factor == 1, we can reuse
        // same P_I for all P_J
      }

      // Square each residue matrix
      {
        Scoped_Timer syrk_timer(timers, "syrk");
        std::vector<Window_Row_Segment> segments;
        if(node_column_runs != nullptr)
          segments = window_row_segments(*node_column_runs, iter);
        do_blas_jobs(uplo, kind, *blas_job_schedule,
                     input_grouped_block_residues_window_A,
                     input_grouped_block_residues_window_B,
                     output_residues_window, shared_memory_comm, timers,
                     node_column_runs == nullptr ? nullptr : &segments,
                     output_I.beg,
                     kind == Blas_Job::syrk ? output_I.beg : output_J.beg);
        update_block_timings_with_syrk(
          block_timings_ms, syrk_timer, bigint_input_matrix_blocks,
          block_index_local_to_global, shared_memory_comm,
          *input_grouped_block_residues_window_A, total_block_height_per_node);
      }
    }

  output_residues_window->Fence();
}

std::vector<std::vector<Block_Column_Runs>>
BigInt_Shared_Memory_Syrk_Context::gather_node_column_runs(
  const std::vector<El::DistMatrix<El::BigFloat>> &bigint_input_matrix_blocks,
  const std::vector<std::vector<El::Int>> &column_maps,
  const El::Int width) const
{
  ASSERT_EQUAL(column_maps.size(), bigint_input_matrix_blocks.size());
  // This rank's group: group_index, number of blocks, then for each block:
  // height, number of runs, (begin, end) of each run
  std::vector<long long> local{static_cast<long long>(group_index),
                               static_cast<long long>(column_maps.size())};
  for(size_t b = 0; b < column_maps.size(); ++b)
    {
      std::vector<El::Int> columns;
      for(const auto g : column_maps[b])
        if(g >= 0)
          columns.push_back(g);
      std::sort(columns.begin(), columns.end());
      std::vector<std::pair<El::Int, El::Int>> runs;
      for(const auto g : columns)
        {
          ASSERT(g < width, DEBUG_STRING(g), DEBUG_STRING(width));
          if(!runs.empty() && runs.back().second == g)
            ++runs.back().second;
          else
            runs.emplace_back(g, g + 1);
        }
      local.push_back(bigint_input_matrix_blocks[b].Height());
      local.push_back(runs.size());
      for(const auto &[beg, end] : runs)
        {
          local.push_back(beg);
          local.push_back(end);
        }
    }

  const int num_ranks = shared_memory_comm.Size();
  int local_size = local.size();
  std::vector<int> sizes(num_ranks), offsets(num_ranks, 0);
  MPI_Allgather(&local_size, 1, MPI_INT, sizes.data(), 1, MPI_INT,
                shared_memory_comm.comm);
  for(int rank = 1; rank < num_ranks; ++rank)
    offsets[rank] = offsets[rank - 1] + sizes[rank - 1];
  std::vector<long long> all(offsets.back() + sizes.back());
  MPI_Allgatherv(local.data(), local_size, MPI_LONG_LONG, all.data(),
                 sizes.data(), offsets.data(), MPI_LONG_LONG,
                 shared_memory_comm.comm);

  std::vector<std::vector<Block_Column_Runs>> result(num_groups);
  std::vector<bool> filled(num_groups, false);
  for(int rank = 0; rank < num_ranks; ++rank)
    {
      size_t pos = offsets[rank];
      const auto group = all.at(pos++);
      ASSERT(group >= 0 && size_t(group) < num_groups, DEBUG_STRING(group));
      const auto num_blocks = all.at(pos++);
      if(filled[group])
        continue;
      filled[group] = true;
      for(long long b = 0; b < num_blocks; ++b)
        {
          auto &block = result[group].emplace_back();
          block.height = all.at(pos++);
          const auto num_runs = all.at(pos++);
          for(long long r = 0; r < num_runs; ++r)
            {
              const El::Int beg = all.at(pos++);
              const El::Int end = all.at(pos++);
              block.runs.emplace_back(beg, end);
            }
        }
    }
  for(size_t group = 0; group < num_groups; ++group)
    ASSERT(filled[group], DEBUG_STRING(group));
  return result;
}

// As compute_block_residues(): group g fills rows [iter * h_g, (iter+1) * h_g)
// of its stacked blocks into its h_g window rows, then zero padding.
std::vector<Window_Row_Segment>
BigInt_Shared_Memory_Syrk_Context::window_row_segments(
  const std::vector<std::vector<Block_Column_Runs>> &node_column_runs,
  const size_t iter) const
{
  ASSERT_EQUAL(node_column_runs.size(), num_groups);
  const auto same_runs = [](const std::vector<El::Range<El::Int>> &a,
                            const std::vector<El::Range<El::Int>> &b) {
    if(a.size() != b.size())
      return false;
    for(size_t i = 0; i < a.size(); ++i)
      if(a[i].beg != b[i].beg || a[i].end != b[i].end)
        return false;
    return true;
  };
  std::vector<Window_Row_Segment> segments;
  El::Int window_row = 0;
  for(size_t group = 0; group < num_groups; ++group)
    {
      const El::Int h
        = input_grouped_block_residues_window_A->block_residues.at(0)
            .at(group)
            .Height();
      const El::Int skip = iter * h;
      El::Int block_begin = 0;
      for(const auto &block : node_column_runs[group])
        {
          const El::Int lo = std::max(block_begin, skip);
          const El::Int hi = std::min(block_begin + block.height, skip + h);
          block_begin += block.height;
          if(lo >= hi || block.runs.empty())
            continue;
          const El::Range<El::Int> rows(window_row + lo - skip,
                                        window_row + hi - skip);
          if(!segments.empty() && segments.back().rows.end == rows.beg
             && same_runs(segments.back().runs, block.runs))
            segments.back().rows.end = rows.end;
          else
            segments.push_back({rows, block.runs});
        }
      window_row += h;
    }
  return segments;
}
