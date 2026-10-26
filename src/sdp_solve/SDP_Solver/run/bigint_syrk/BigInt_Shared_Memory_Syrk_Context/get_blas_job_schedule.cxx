#include "../BigInt_Shared_Memory_Syrk_Context.hxx"

std::shared_ptr<Blas_Job_Schedule>
BigInt_Shared_Memory_Syrk_Context::get_blas_job_schedule(Blas_Job::Kind kind,
                                                         El::UpperOrLower uplo,
                                                         El::Int output_height,
                                                         El::Int output_width)
{
  const auto key = std::tie(kind, uplo, output_height, output_width);
  if(blas_job_schedule_cache.find(key) != blas_job_schedule_cache.end())
    return blas_job_schedule_cache.at(key);

  auto [it, res] = blas_job_schedule_cache.emplace(
    key, std::make_shared<Blas_Job_Schedule>(create_blas_job_schedule_func(
           kind, uplo, shared_memory_comm.Size(), comb.num_primes,
           output_height, output_width, verbosity)));
  ASSERT(res);
  return it->second;
}

namespace
{
  El::Int overlap(const El::Range<El::Int> &a, const El::Range<El::Int> &b)
  {
    return std::max<El::Int>(0,
                             std::min(a.end, b.end) - std::max(a.beg, b.beg));
  }
}

// --compressLinvB: the cost of a job is its BLAS work on the column runs,
// sum over blocks of (rows x output elements that the block contributes to)
std::shared_ptr<Blas_Job_Schedule>
BigInt_Shared_Memory_Syrk_Context::get_weighted_blas_job_schedule(
  const Blas_Job::Kind kind, const El::UpperOrLower uplo,
  const El::Range<El::Int> &output_I, const El::Range<El::Int> &output_J,
  const std::vector<std::vector<Block_Column_Runs>> &node_column_runs)
{
  const auto key = std::make_tuple(kind, uplo, output_I.beg, output_I.end,
                                   output_J.beg, output_J.end);
  if(const auto it = weighted_blas_job_schedule_cache.find(key);
     it != weighted_blas_job_schedule_cache.end())
    return it->second;

  const El::Int offset_B
    = kind == Blas_Job::syrk ? output_I.beg : output_J.beg;
  const auto job_cost = [&](const Blas_Job &job) -> Blas_Job::Cost {
    const El::Range<El::Int> I(job.I.beg + output_I.beg,
                               job.I.end + output_I.beg);
    const El::Range<El::Int> J(job.J.beg + offset_B, job.J.end + offset_B);
    size_t work = 0, blas_calls = 0;
    for(const auto &group : node_column_runs)
      for(const auto &block : group)
        {
          const auto &runs = block.runs;
          size_t touched = 0;
          for(size_t r = 0; r < runs.size(); ++r)
            {
              const El::Int a = overlap(I, runs[r]);
              if(a == 0)
                continue;
              if(job.kind == Blas_Job::syrk)
                {
                  touched += a * (a + 1) / 2;
                  ++blas_calls;
                  for(size_t s = 0; s < runs.size(); ++s)
                    if((uplo == El::UPPER ? s > r : s < r)
                       && overlap(I, runs[s]) > 0)
                      {
                        touched += a * overlap(I, runs[s]);
                        ++blas_calls;
                      }
                }
              else
                for(const auto &run_s : runs)
                  if(overlap(J, run_s) > 0)
                    {
                      touched += a * overlap(J, run_s);
                      ++blas_calls;
                    }
            }
          work += block.height * touched;
        }
    return {work, blas_calls};
  };
  auto [it, inserted] = weighted_blas_job_schedule_cache.emplace(
    key, std::make_shared<Blas_Job_Schedule>(create_weighted_blas_job_schedule(
           kind, uplo, shared_memory_comm.Size(), comb.num_primes,
           output_I.end - output_I.beg, output_J.end - output_J.beg, verbosity,
           job_cost)));
  ASSERT(inserted);
  return it->second;
}
