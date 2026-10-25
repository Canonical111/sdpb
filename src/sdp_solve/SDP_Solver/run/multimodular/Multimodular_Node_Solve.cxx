#include "Multimodular_Node_Solve.hxx"

#include "Multimodular_Block_Factor.hxx"
#include "Shared_BigFloat_Array.hxx"
#include "sdp_solve/SDP_Solver/run/bigint_syrk/blas_jobs/LPT_scheduling.hxx"
#include "sdpb_util/Shared_Window_Array.hxx"
#include "sdpb_util/assert.hxx"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>

namespace
{
  using Clock = std::chrono::steady_clock;
  int64_t milliseconds_since(const Clock::time_point &start)
  {
    return std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now()
                                                                 - start)
      .count();
  }

  // A block whose solve is shared by the node's ranks
  struct Node_Block
  {
    int64_t index;          // global block index
    int64_t n;              // rows of L and B
    int64_t N;              // columns of B
    int64_t grid_size;      // ranks of the block's grid
    int64_t decision_width; // see Multimodular_Blas_Config::decision_width
    size_t L_offset = 0;    // first record in the L window
    size_t B_offset = 0;    // first record in the B window
    size_t num_jobs = 1;
  };
  constexpr int node_block_fields = 5;

  // Job: columns [j0, j1) of block `block` (index into the node's blocks)
  struct Solve_Job
  {
    size_t block;
    int j0, j1;
    size_t cost;
  };

  // Cost model of a left-looking or recursive multimodular solve of
  // `width` columns: GEMM work ~ n^2 width, output conversion and CRT
  // ~ n width (fitted to the per-block solve times of a 128-rank nmax10
  // run: about 12 rows' worth).
  size_t solve_cost(const int64_t n, const int64_t width)
  { return size_t(n) * size_t(n + 12) * size_t(width); }

  size_t lower_index(const int64_t n, const int64_t i, const int64_t j)
  { return size_t(j) * n - size_t(j) * (j - 1) / 2 + (i - j); }

  // The left-looking / recursive choice the grid path makes for this
  // block: grid rank 0 solves columns j = 0 mod g, i.e. ceil(N / g) of them
  // (Multimodular_Block_Factor::solve).
  int64_t grid_decision_width(const int64_t N, const int64_t grid_size)
  { return (N + grid_size - 1) / grid_size; }

  // All blocks of the node whose grid is on the node, sorted by index.
  // Each block is reported by rank 0 of its grid.
  std::vector<Node_Block>
  gather_node_blocks(const El::mpi::Comm &node_comm,
                     const std::vector<int64_t> &local_fields)
  {
    const int size = El::mpi::Size(node_comm);
    int count = local_fields.size();
    std::vector<int> counts(size), displacements(size);
    MPI_Allgather(&count, 1, MPI_INT, counts.data(), 1, MPI_INT,
                  node_comm.comm);
    int total = 0;
    for(int r = 0; r < size; ++r)
      {
        displacements[r] = total;
        total += counts[r];
      }
    std::vector<int64_t> fields(std::max(total, 1));
    MPI_Allgatherv(local_fields.data(), count, MPI_INT64_T, fields.data(),
                   counts.data(), displacements.data(), MPI_INT64_T,
                   node_comm.comm);
    std::vector<Node_Block> blocks;
    for(int k = 0; k < total; k += node_block_fields)
      blocks.push_back({fields[k], fields[k + 1], fields[k + 2],
                        fields[k + 3], fields[k + 4]});
    std::sort(blocks.begin(), blocks.end(),
              [](const Node_Block &a, const Node_Block &b) {
                return a.index < b.index;
              });
    return blocks;
  }

  // n x n El::Matrix of mpf structs whose lower triangle points into a
  // Shared_BigFloat_Array (records offset + lower_index(n, i, j)); the
  // strictly upper part points to a dummy zero and is never read.
  class Lower_View
  {
    El::BigFloat *structs = nullptr;
    mp_limb_t dummy[2] = {0, 0};

  public:
    El::Matrix<El::BigFloat> matrix;
    Lower_View(Shared_BigFloat_Array &array, const size_t offset,
               const int64_t n)
    {
      structs = static_cast<El::BigFloat *>(::operator new(
        sizeof(El::BigFloat) * std::max<size_t>(1, size_t(n) * n)));
      for(int64_t j = 0; j < n; ++j)
        for(int64_t i = 0; i < n; ++i)
          {
            __mpf_struct *s = structs[i + size_t(j) * n].gmp_float.get_mpf_t();
            if(i >= j)
              *s = array.view(offset + lower_index(n, i, j));
            else
              {
                s->_mp_prec = 1;
                s->_mp_size = 0;
                s->_mp_exp = 0;
                s->_mp_d = dummy;
              }
          }
      matrix.Attach(n, n, structs, n);
    }
    ~Lower_View()
    {
      matrix.Empty();
      ::operator delete(structs);
    }
    Lower_View(const Lower_View &) = delete;
    Lower_View &operator=(const Lower_View &) = delete;
  };

  // B[:, j0:j1) := L^{-1} B[:, j0:j1) in place in the B window
  void run_job(const Solve_Job &job, const Node_Block &block,
               Shared_BigFloat_Array &L_window,
               Shared_BigFloat_Array &B_window,
               std::unique_ptr<Lower_View> &L_view, size_t &L_view_block,
               const Multimodular_Blas_Config &config)
  {
    const int64_t n = block.n;
    if(!L_view || L_view_block != job.block)
      {
        L_view.reset();
        L_view = std::make_unique<Lower_View>(L_window, block.L_offset, n);
        L_view_block = job.block;
      }
    const int w = job.j1 - job.j0;
    El::BigFloat *structs = static_cast<El::BigFloat *>(::operator new(
      sizeof(El::BigFloat) * std::max<size_t>(1, size_t(n) * w)));
    for(int jj = 0; jj < w; ++jj)
      for(int64_t i = 0; i < n; ++i)
        *structs[i + size_t(jj) * n].gmp_float.get_mpf_t() = B_window.view(
          block.B_offset + i + size_t(job.j0 + jj) * n);
    {
      El::Matrix<El::BigFloat> X;
      X.Attach(n, w, structs, n);
      Multimodular_Blas_Config job_config = config;
      job_config.decision_width = block.decision_width;
      Multimodular_Blas_Timings timings;
      multimodular_trsm_left_lower(L_view->matrix, X, job_config, timings);
    }
    for(int jj = 0; jj < w; ++jj)
      for(int64_t i = 0; i < n; ++i)
        B_window.publish(block.B_offset + i + size_t(job.j0 + jj) * n,
                         *structs[i + size_t(jj) * n].gmp_float.get_mpf_t());
    ::operator delete(structs);
  }

  // SDPB_MULTIMODULAR_NODE_PERSIST=1 (diagnostic): keep the node windows
  // from one call to the next (same sizes), instead of allocating them in
  // every iteration. Freed at the start of MPI_Finalize, through an
  // attribute of MPI_COMM_SELF.
  struct Persistent_Windows
  {
    size_t L_records = 0, B_records = 0;
    std::unique_ptr<Shared_BigFloat_Array> L, B;
  };
  Persistent_Windows *persistent_windows = nullptr;
  int free_persistent_windows(MPI_Comm, int, void *, void *)
  {
    delete persistent_windows;
    persistent_windows = nullptr;
    return MPI_SUCCESS;
  }

  size_t env_size(const char *name, const size_t default_value)
  {
    const char *value = std::getenv(name);
    return value && *value ? std::strtoul(value, nullptr, 10) : default_value;
  }
}

Multimodular_Node_Schedule multimodular_node_schedule_from_env()
{
  const char *value = std::getenv("SDPB_MULTIMODULAR_NODE_SCHEDULE");
  if(value == nullptr || *value == 0 || std::strcmp(value, "grid") == 0)
    return Multimodular_Node_Schedule::grid;
  if(std::strcmp(value, "static") == 0)
    return Multimodular_Node_Schedule::static_lpt;
  if(std::strcmp(value, "dynamic") == 0)
    return Multimodular_Node_Schedule::dynamic;
  RUNTIME_ERROR("SDPB_MULTIMODULAR_NODE_SCHEDULE: expected grid, static or "
                "dynamic, got ",
                value);
}

void multimodular_node_cholesky_solve(
  const El::mpi::Comm &node_comm, const std::vector<size_t> &block_indices,
  const std::vector<El::DistMatrix<El::BigFloat>> &schur,
  std::vector<El::DistMatrix<El::BigFloat>> &cholesky,
  std::vector<El::DistMatrix<El::BigFloat>> &B,
  const Multimodular_Node_Schedule schedule,
  const Multimodular_Blas_Config &config, Timers &timers,
  El::Matrix<int32_t> &block_timings_ms)
{
  ASSERT(schedule != Multimodular_Node_Schedule::grid);
  const size_t num_local = schur.size();
  ASSERT_EQUAL(cholesky.size(), num_local);
  ASSERT_EQUAL(B.size(), num_local);
  const int node_rank = El::mpi::Rank(node_comm);
  const int node_size = El::mpi::Size(node_comm);

  // Which local blocks are shared by the node: those whose grid is on it
  std::vector<bool> pooled(num_local);
  std::vector<int64_t> local_fields;
  for(size_t k = 0; k < num_local; ++k)
    {
      const El::Grid &grid = schur[k].Grid();
      pooled[k] = grid.Size() == 1 || on_one_node(grid.Comm());
      if(pooled[k] && grid.Rank() == 0)
        {
          const int64_t N = B[k].Width();
          local_fields.insert(
            local_fields.end(),
            {int64_t(block_indices.at(k)), int64_t(schur[k].Height()), N,
             int64_t(grid.Size()), grid_decision_width(N, grid.Size())});
        }
    }
  Scoped_Timer setup_timer(timers, "node_setup");
  std::vector<Node_Block> blocks = gather_node_blocks(node_comm, local_fields);
  const size_t num_blocks = blocks.size();
  size_t L_records = 0, B_records = 0, total_cost = 0;
  for(auto &block : blocks)
    {
      block.L_offset = L_records;
      block.B_offset = B_records;
      L_records += size_t(block.n) * (block.n + 1) / 2;
      B_records += size_t(block.n) * block.N;
      total_cost += solve_cost(block.n, block.N);
    }
  // Jobs: each block split into column ranges of about
  // total_cost / (chunks_per_rank * ranks), at least min_columns wide
  const size_t chunks_per_rank
    = std::max<size_t>(1, env_size("SDPB_MULTIMODULAR_NODE_CHUNKS", 2));
  const size_t min_columns
    = std::max<size_t>(1, env_size("SDPB_MULTIMODULAR_NODE_MIN_COLUMNS", 64));
  const double target
    = std::max(1.0, double(total_cost) / (chunks_per_rank * node_size));
  std::vector<Solve_Job> jobs;
  std::vector<std::vector<size_t>> block_jobs(num_blocks);
  for(size_t i = 0; i < num_blocks; ++i)
    {
      auto &block = blocks[i];
      const size_t max_jobs = std::max<size_t>(1, block.N / min_columns);
      block.num_jobs = std::clamp<size_t>(
        std::ceil(solve_cost(block.n, block.N) / target), 1, max_jobs);
      for(size_t c = 0; c < block.num_jobs; ++c)
        {
          const int j0 = block.N * c / block.num_jobs;
          const int j1 = block.N * (c + 1) / block.num_jobs;
          block_jobs[i].push_back(jobs.size());
          jobs.push_back({i, j0, j1, solve_cost(block.n, j1 - j0)});
        }
    }
  std::vector<size_t> position(num_local, num_blocks);
  for(size_t k = 0; k < num_local; ++k)
    if(pooled[k])
      {
        const auto index = int64_t(block_indices.at(k));
        for(size_t i = 0; i < num_blocks; ++i)
          if(blocks[i].index == index)
            position[k] = i;
        ASSERT(position[k] < num_blocks, "block ", index, " not found");
      }

  std::unique_ptr<Shared_BigFloat_Array> L_owned, B_owned;
  if(env_size("SDPB_MULTIMODULAR_NODE_PERSIST", 0) != 0)
    {
      if(persistent_windows == nullptr)
        {
          persistent_windows = new Persistent_Windows;
          int keyval;
          MPI_Comm_create_keyval(MPI_COMM_NULL_COPY_FN,
                                 free_persistent_windows, &keyval, nullptr);
          MPI_Comm_set_attr(MPI_COMM_SELF, keyval, nullptr);
        }
      auto &p = *persistent_windows;
      if(!p.L || p.L_records != L_records || p.B_records != B_records)
        {
          p.L.reset();
          p.B.reset();
          p.L = std::make_unique<Shared_BigFloat_Array>(node_comm, L_records,
                                                        false);
          p.B = std::make_unique<Shared_BigFloat_Array>(node_comm, B_records,
                                                        false);
          p.L_records = L_records;
          p.B_records = B_records;
        }
    }
  else
    {
      L_owned = std::make_unique<Shared_BigFloat_Array>(node_comm, L_records,
                                                        false);
      B_owned = std::make_unique<Shared_BigFloat_Array>(node_comm, B_records,
                                                        false);
    }
  Shared_BigFloat_Array &L_window
    = L_owned ? *L_owned : *persistent_windows->L;
  Shared_BigFloat_Array &B_window
    = B_owned ? *B_owned : *persistent_windows->B;
  // dynamic: ready[i] counts the grid ranks that have written block i;
  // next[i] is the next job of block i to take
  Shared_Window_Array<int64_t> control(node_comm, 2 * num_blocks + 1);
  if(node_rank == 0)
    std::fill_n(control.data, 2 * num_blocks + 1, 0);
  control.Fence();
  int64_t *ready = control.data;
  int64_t *next = control.data + num_blocks;
  setup_timer.stop();

  // Phase 1: Cholesky of each local block on its grid; write L and B of
  // the shared blocks into the node windows
  for(size_t k = 0; k < num_local; ++k)
    {
      const auto global_index = block_indices.at(k);
      const auto index_string = std::to_string(global_index);
      Multimodular_Block_Factor factor(schur[k], config);
      {
        Scoped_Timer cholesky_timer(timers, "cholesky_" + index_string);
        try
          {
            factor.cholesky(schur[k], cholesky[k], timers);
          }
        catch(std::exception &e)
          {
            RUNTIME_ERROR(
              "Error when computing Cholesky decomposition of block_",
              global_index, ": ", e.what());
          }
        block_timings_ms(global_index, 0)
          += cholesky_timer.elapsed_milliseconds();
      }
      if(!pooled[k])
        {
          Scoped_Timer solve_timer(timers, "solve_" + index_string);
          factor.solve(B[k], timers);
          block_timings_ms(global_index, 0)
            += solve_timer.elapsed_milliseconds();
          continue;
        }
      const auto start = Clock::now();
      Scoped_Timer fill_timer(timers, "to_node_" + index_string);
      const Node_Block &block = blocks[position[k]];
      const auto &L = cholesky[k];
      for(int jLoc = 0; jLoc < L.LocalWidth(); ++jLoc)
        for(int iLoc = 0; iLoc < L.LocalHeight(); ++iLoc)
          {
            const int i = L.GlobalRow(iLoc), j = L.GlobalCol(jLoc);
            if(i >= j)
              L_window.set(block.L_offset + lower_index(block.n, i, j),
                           L.GetLocalCRef(iLoc, jLoc));
          }
      const auto &Bk = B[k];
      for(int jLoc = 0; jLoc < Bk.LocalWidth(); ++jLoc)
        for(int iLoc = 0; iLoc < Bk.LocalHeight(); ++iLoc)
          B_window.set(block.B_offset + Bk.GlobalRow(iLoc)
                         + size_t(Bk.GlobalCol(jLoc)) * block.n,
                       Bk.GetLocalCRef(iLoc, jLoc));
      __atomic_fetch_add(&ready[position[k]], 1, __ATOMIC_RELEASE);
      block_timings_ms(global_index, 0) += milliseconds_since(start);
    }

  // Phase 2: the solve jobs
  std::unique_ptr<Lower_View> L_view;
  size_t L_view_block = num_blocks;
  auto run = [&](const size_t job_index) {
    const auto start = Clock::now();
    const Solve_Job &job = jobs[job_index];
    run_job(job, blocks[job.block], L_window, B_window, L_view, L_view_block,
            config);
    block_timings_ms(blocks[job.block].index, 0) += milliseconds_since(start);
  };
  if(schedule == Multimodular_Node_Schedule::static_lpt)
    {
      {
        Scoped_Timer timer(timers, "node_wait_factors");
        L_window.Fence();
        B_window.Fence();
      }
      Scoped_Timer timer(timers, "node_solve");
      std::vector<size_t> job_ids(jobs.size());
      for(size_t j = 0; j < jobs.size(); ++j)
        job_ids[j] = j;
      const std::function<size_t(const size_t &)> get_cost
        = [&jobs](const size_t &j) { return jobs[j].cost; };
      const auto jobs_by_rank
        = LPT_scheduling<size_t, size_t>(node_size, job_ids, get_cost);
      for(const auto j : jobs_by_rank.at(node_rank))
        run(j);
    }
  else
    {
      // Largest blocks first; take a job of the first block whose factor
      // is ready and that has jobs left
      Scoped_Timer timer(timers, "node_solve");
      std::vector<size_t> order(num_blocks);
      for(size_t i = 0; i < num_blocks; ++i)
        order[i] = i;
      std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return solve_cost(blocks[a].n, blocks[a].N)
               > solve_cost(blocks[b].n, blocks[b].N);
      });
      while(true)
        {
          bool remaining = false, took = false;
          for(const auto i : order)
            {
              if(__atomic_load_n(&next[i], __ATOMIC_RELAXED)
                 >= int64_t(blocks[i].num_jobs))
                continue;
              remaining = true;
              if(__atomic_load_n(&ready[i], __ATOMIC_ACQUIRE)
                 < blocks[i].grid_size)
                continue;
              const int64_t c = __atomic_fetch_add(&next[i], 1, __ATOMIC_ACQ_REL);
              if(c >= int64_t(blocks[i].num_jobs))
                continue;
              run(block_jobs[i][c]);
              took = true;
              break;
            }
          if(!remaining)
            break;
          if(!took)
            std::this_thread::yield();
        }
    }
  L_view.reset();

  {
    Scoped_Timer timer(timers, "node_wait_solves");
    B_window.Fence();
  }
  // Read back the entries of P = L^{-1} B this rank owns
  Scoped_Timer from_timer(timers, "from_node");
  for(size_t k = 0; k < num_local; ++k)
    {
      if(!pooled[k])
        continue;
      const Node_Block &block = blocks[position[k]];
      auto &Bk = B[k];
      for(int jLoc = 0; jLoc < Bk.LocalWidth(); ++jLoc)
        for(int iLoc = 0; iLoc < Bk.LocalHeight(); ++iLoc)
          {
            const __mpf_struct v = B_window.view(
              block.B_offset + Bk.GlobalRow(iLoc)
              + size_t(Bk.GlobalCol(jLoc)) * block.n);
            mpf_set(Bk.Matrix().Ref(iLoc, jLoc).gmp_float.get_mpf_t(), &v);
          }
    }
}
