// lab #175: apply sdpa-solver's multimodular trmm (compute_S.cxx,
// Initialize_P_Context.cxx at 16b1a86f) to an SDPB Schur block.
//
// SDPB's compute_Q needs L = chol(S) and X = L^{-1} B for each block.
// sdpa-solver computes L_X^{-1} once with El::TriangularInverse and then
// multiplies by it through residues: normalize_and_shift (rows of L^{-1},
// columns of the right factor), FLINT residues per element into
// Vector_Matrix_Residues_Window's, one cblas_dtrmm per prime (BLAS job
// schedule), FLINT CRT per element, restore_trmm_output. This harness runs
// exactly those functions (Initialize_P_Context.cxx is included verbatim to
// reach its file-local helpers) on one rank, with B in the role of G.
//
// run:  sdpa_trmm_bench run <schur.bin> <B.bin> --method el|elinv|sdpa
//         [--precision P] [--n N] [--cols C] [--guard G] [--out-L f] [--out-X f]
//       el:    El::Cholesky + El::Trsm (SDPB master)
//       elinv: El::Cholesky + El::TriangularInverse + El::Trmm (BigFloat)
//       sdpa:  El::Cholesky + El::TriangularInverse + sdpa-solver's
//              multimodular trmm (--guard G: fixed point at precision + G
//              bits instead of sdpa-solver's precision)
// Matrix files: the SDPB_DUMP_BLOCK / mm_trsm_bench format of lab #145
// (int64 height, width, precision; El::BigFloat::Serialize records,
// column-major). Reader/writer copied from #145's mm_trsm_bench.

#include "sdpa_solve/SDP_Solver/run/step/Initialize_P_Context.cxx"
#include "sdp_solve/SDP_Solver/run/bigint_syrk/Matrix_Normalizer.hxx"

#include <El.hpp>

#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace
{
  using Clock = std::chrono::steady_clock;
  double seconds_since(const Clock::time_point &start)
  {
    return std::chrono::duration<double>(Clock::now() - start).count();
  }

  El::Matrix<El::BigFloat>
  read_matrix(const std::string &path, int max_height = -1,
              int max_width = -1)
  {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if(!in)
      throw std::runtime_error("cannot open " + path);
    const size_t file_size = in.tellg();
    in.seekg(0);
    int64_t header[3];
    in.read(reinterpret_cast<char *>(header), sizeof(header));
    const int64_t height = header[0], width = header[1];
    const size_t record = (file_size - sizeof(header)) / (height * width);
    const size_t max_limbs
      = (record - sizeof(int) - sizeof(mp_exp_t)) / sizeof(mp_limb_t);
    const int h
      = max_height < 0 ? height : std::min<int64_t>(max_height, height);
    const int w = max_width < 0 ? width : std::min<int64_t>(max_width, width);

    El::Matrix<El::BigFloat> result(h, w);
    std::vector<char> buffer(record);
    std::vector<mp_limb_t> limbs(max_limbs + 1);
    for(int64_t j = 0; j < width; ++j)
      for(int64_t i = 0; i < height; ++i)
        {
          in.read(buffer.data(), record);
          if(i >= h || j >= w)
            continue;
          int size;
          mp_exp_t exp;
          std::memcpy(&size, buffer.data(), sizeof(int));
          std::memcpy(&exp, buffer.data() + sizeof(int), sizeof(mp_exp_t));
          std::memcpy(limbs.data(),
                      buffer.data() + sizeof(int) + sizeof(mp_exp_t),
                      max_limbs * sizeof(mp_limb_t));
          __mpf_struct view;
          view._mp_prec = max_limbs;
          view._mp_size = size;
          view._mp_exp = exp;
          view._mp_d = limbs.data();
          mpf_set(result.Ref(i, j).gmp_float.get_mpf_t(), &view);
        }
    return result;
  }

  void write_matrix(const El::Matrix<El::BigFloat> &matrix,
                    const std::string &path)
  {
    std::ofstream out(path, std::ios::binary);
    int64_t header[3] = {matrix.Height(), matrix.Width(),
                         static_cast<int64_t>(El::gmp::Precision())};
    out.write(reinterpret_cast<const char *>(header), sizeof(header));
    std::vector<El::byte> buffer;
    for(int j = 0; j < matrix.Width(); ++j)
      for(int i = 0; i < matrix.Height(); ++i)
        {
          const auto &value = matrix.CRef(i, j);
          buffer.resize(value.SerializedSize());
          value.Serialize(buffer.data());
          out.write(reinterpret_cast<const char *>(buffer.data()),
                    buffer.size());
        }
  }

  mp_bitcnt_t file_precision(const std::string &path)
  {
    std::ifstream in(path, std::ios::binary);
    int64_t header[3];
    in.read(reinterpret_cast<char *>(header), sizeof(header));
    return header[2];
  }

  // Same as Initialize_P_Context::do_blas_job, for a right factor that is
  // a general n x N matrix (Vector_Matrix_Residues_Window) instead of a
  // vector<Block_Diagonal_Matrix> window.
  void do_blas_job_general(const Sdpb::Sdpa::Trmm::Blas_Job &job,
                           Vector_Matrix_Residues_Window<double> &L_window,
                           Vector_Matrix_Residues_Window<double> &G_window)
  {
    const auto &L
      = L_window.matrices_residues.at(job.prime_index).at(job.block_index);
    auto &G
      = G_window.matrices_residues.at(job.prime_index).at(job.block_index);
    auto G_view = G(job.I, job.J);
    Sdpb::Sdpa::dtrmm(job.side, job.uplo, job.orientation, job.diag, L,
                      G_view);
  }

  // X := L^{-1} X by sdpa-solver's route. L_inv is the explicit inverse
  // (lower triangle significant), X is n x N. Both are overwritten.
  // guard_bits = 0 is sdpa-solver's choice (bits = El::gmp::Precision()).
  void sdpa_trmm(El::DistMatrix<El::BigFloat> &L_inv,
                 El::DistMatrix<El::BigFloat> &X, const int num_ranks,
                 const int guard_bits, std::map<std::string, double> &t)
  {
    using namespace Sdpb::Sdpa;
    const int bits = El::gmp::Precision() + guard_bits;
    const size_t n = L_inv.Height(), N = X.Width();
    constexpr auto uplo = El::LOWER;
    constexpr auto diag = El::NON_UNIT;
    constexpr auto orientation = El::NORMAL;
    constexpr auto side = El::LEFT;

    auto start = Clock::now();
    const auto L_normalizer
      = normalize_and_shift<Matrix_Normalization_Kind::ROWS>(L_inv, bits,
                                                             uplo);
    const auto X_normalizer
      = normalize_and_shift<Matrix_Normalization_Kind::COLUMNS>(X, bits);
    t["normalize"] = seconds_since(start);

    // As Initialize_P_Config::comb(): max additions = max block dim.
    start = Clock::now();
    Fmpz_Comb comb(bits, bits, 1, n);
    const size_t num_primes = comb.num_primes;
    Vector_Matrix_Residues_Window<double> L_window(
      Shared_Window_Array_View<double>(El::mpi::COMM_SELF,
                                       n * n * num_primes),
      num_primes, {n}, {n});
    Vector_Matrix_Residues_Window<double> X_window(
      Shared_Window_Array_View<double>(El::mpi::COMM_SELF,
                                       n * N * num_primes),
      num_primes, {n}, {N});
    t["setup"] = seconds_since(start);
    t["num_primes"] = num_primes;

    start = Clock::now();
    for(size_t j = 0; j < n; ++j)
      compute_column_residues_elementwise(L_inv, 0, j, comb, L_window);
    L_window.Fence();
    t["residues_L"] = seconds_since(start);

    start = Clock::now();
    for(size_t j = 0; j < N; ++j)
      compute_column_residues_elementwise(X, 0, j, comb, X_window);
    X_window.Fence();
    t["residues_X"] = seconds_since(start);

    start = Clock::now();
    const auto schedule = Trmm::create_blas_job_schedule(
      side, uplo, orientation, diag, num_ranks, num_primes, {n}, {N},
      Verbosity::regular);
    // One rank does every rank's jobs (single-process harness).
    size_t num_jobs = 0;
    for(const auto &rank_jobs : schedule.jobs_by_rank)
      for(const auto &job : rank_jobs)
        {
          do_blas_job_general(job, L_window, X_window);
          ++num_jobs;
        }
    t["blas"] = seconds_since(start);
    t["blas_jobs"] = num_jobs;

    start = Clock::now();
    restore_distmatrix_from_residues(
      std::nullopt, X_window.matrices_residues.at(0).at(0),
      X_window.prime_stride, comb, X);
    t["restore"] = seconds_since(start);

    start = Clock::now();
    restore_trmm_output(side, uplo, orientation, L_normalizer, X_normalizer,
                        X);
    t["remove_normalization"] = seconds_since(start);
  }

  int run(int argc, char **argv)
  {
    std::map<std::string, std::string> opts;
    std::vector<std::string> positional;
    for(int a = 2; a < argc; ++a)
      {
        std::string s = argv[a];
        if(s.rfind("--", 0) == 0 && a + 1 < argc)
          opts[s.substr(2)] = argv[++a];
        else
          positional.push_back(s);
      }
    if(positional.size() != 2)
      throw std::runtime_error("run: need <schur.bin> <B.bin>");
    auto get = [&](const std::string &key, const std::string &dflt) {
      auto it = opts.find(key);
      return it == opts.end() ? dflt : it->second;
    };
    const std::string method = get("method", "sdpa");
    const mp_bitcnt_t precision = std::stoul(get("precision", "0"));
    const int n = std::stoi(get("n", "-1"));
    const int cols = std::stoi(get("cols", "-1"));
    const int num_ranks = std::stoi(get("ranks", "1"));
    const int guard_bits = std::stoi(get("guard", "0"));

    El::gmp::SetPrecision(precision == 0 ? file_precision(positional[0])
                                         : precision);
    const auto S_local = read_matrix(positional[0], n, n);
    const auto B_local = read_matrix(positional[1], S_local.Height(), cols);
    std::cout << "S " << S_local.Height() << "x" << S_local.Width() << ", B "
              << B_local.Height() << "x" << B_local.Width() << ", precision "
              << El::gmp::Precision() << ", method " << method << std::endl;

    // SDPB keeps every block as a DistMatrix; one rank here.
    const El::Grid grid(El::mpi::COMM_SELF);
    El::DistMatrix<El::BigFloat> L(grid), X(grid);
    L.Resize(S_local.Height(), S_local.Width());
    El::Copy(S_local, L.Matrix());
    X.Resize(B_local.Height(), B_local.Width());
    El::Copy(B_local, X.Matrix());

    std::map<std::string, double> t;
    auto start = Clock::now();
    El::Cholesky(El::LOWER, L);
    t["cholesky"] = seconds_since(start);

    if(method == "el")
      {
        start = Clock::now();
        El::Trsm(El::LEFT, El::LOWER, El::NORMAL, El::NON_UNIT,
                 El::BigFloat(1), L, X);
        t["trsm"] = seconds_since(start);
      }
    else if(method == "elinv" || method == "sdpa")
      {
        El::DistMatrix<El::BigFloat> L_inv(L);
        // El::Cholesky leaves S's values in the strict upper triangle.
        El::MakeTrapezoidal(El::LOWER, L_inv);
        start = Clock::now();
        El::TriangularInverse(El::LOWER, El::NON_UNIT, L_inv);
        t["triangular_inverse"] = seconds_since(start);
        start = Clock::now();
        if(method == "elinv")
          El::Trmm(El::LEFT, El::LOWER, El::NORMAL, El::NON_UNIT,
                   El::BigFloat(1), L_inv, X);
        else
          sdpa_trmm(L_inv, X, num_ranks, guard_bits, t);
        t["trmm"] = seconds_since(start);
      }
    else
      throw std::runtime_error("unknown method " + method);

    for(const auto &[key, value] : t)
      std::cout << "  " << key << " " << value << std::endl;

    if(opts.count("out-L"))
      write_matrix(L.Matrix(), opts["out-L"]);
    if(opts.count("out-X"))
      write_matrix(X.Matrix(), opts["out-X"]);
    return 0;
  }
}

int main(int argc, char **argv)
{
  El::Environment env(argc, argv);
  try
    {
      if(argc < 2 || std::string(argv[1]) != "run")
        throw std::runtime_error("usage: sdpa_trmm_bench run ...");
      return run(argc, argv);
    }
  catch(std::exception &e)
    {
      std::cerr << "error: " << e.what() << std::endl;
      return 1;
    }
}
