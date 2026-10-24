// Benchmark and accuracy check for the multimodular blocked Cholesky + Trsm
// (lab #145). Reads matrices written by the SDPB_DUMP_BLOCK hook in
// compute_Q.cxx.
//
// run:     mm_trsm_bench run <schur.bin> <B.bin> [--method el|mm]
//            [--precision P] [--n N] [--cols C] [--base B] [--guard G]
//            [--chunk K] [--out-L file] [--out-X file]
//          Computes L = chol(S) and X = L^{-1} B (first N rows/columns of S,
//          first C columns of B), prints the timings.
// compare: mm_trsm_bench compare <X_ref.bin> <X.bin>
//          Prints log2 of the max elementwise relative difference and of
//          the max over columns of max_i|dX_ij| / max_i|X_ref_ij|
//          (lower triangle only if both are square).

#include "sdp_solve/SDP_Solver/run/multimodular/multimodular_blas.hxx"

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

  // Read a matrix written by El::BigFloat::Serialize() at any precision
  // (record: int size, mp_exp_t exp, limbs) into BigFloats of the current
  // El::gmp::Precision().
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

  double log2_abs(const El::BigFloat &x)
  {
    if(mpf_sgn(x.gmp_float.get_mpf_t()) == 0)
      return -1e9;
    long e;
    double d = mpf_get_d_2exp(&e, x.gmp_float.get_mpf_t());
    return std::log2(std::abs(d)) + e;
  }

  int compare(const std::string &ref_path, const std::string &path)
  {
    El::gmp::SetPrecision(
      std::max(file_precision(ref_path), file_precision(path)) + 128);
    auto ref = read_matrix(ref_path);
    auto x = read_matrix(path);
    if(ref.Height() != x.Height() || ref.Width() != x.Width())
      throw std::runtime_error("size mismatch");
    const bool lower = ref.Height() == ref.Width();
    double max_rel = -1e9, max_col = -1e9, max_row = -1e9;
    std::vector<double> col_max_ref(ref.Width(), -1e9),
      col_max_diff(ref.Width(), -1e9);
    std::vector<double> row_max_ref(ref.Height(), -1e9),
      row_max_diff(ref.Height(), -1e9);
    El::BigFloat diff;
    size_t nonzero = 0, zero_mismatch = 0;
    for(int j = 0; j < ref.Width(); ++j)
      for(int i = lower ? j : 0; i < ref.Height(); ++i)
        {
          diff = x.CRef(i, j) - ref.CRef(i, j);
          const double ld = log2_abs(diff), lr = log2_abs(ref.CRef(i, j));
          if(lr > -1e8)
            {
              ++nonzero;
              max_rel = std::max(max_rel, ld - lr);
            }
          else if(ld > -1e8)
            ++zero_mismatch;
          col_max_ref[j] = std::max(col_max_ref[j], lr);
          col_max_diff[j] = std::max(col_max_diff[j], ld);
          row_max_ref[i] = std::max(row_max_ref[i], lr);
          row_max_diff[i] = std::max(row_max_diff[i], ld);
        }
    for(int j = 0; j < ref.Width(); ++j)
      if(col_max_ref[j] > -1e8)
        max_col = std::max(max_col, col_max_diff[j] - col_max_ref[j]);
    for(int i = 0; i < ref.Height(); ++i)
      if(row_max_ref[i] > -1e8)
        max_row = std::max(max_row, row_max_diff[i] - row_max_ref[i]);
    std::cout << "compare " << ref.Height() << "x" << ref.Width()
              << (lower ? " (lower triangle)" : "") << " nonzero=" << nonzero
              << " zero_mismatch=" << zero_mismatch
              << "\n  log2 max_ij |d_ij|/|ref_ij|              = " << max_rel
              << "\n  log2 max_j max_i|d_ij| / max_i|ref_ij|   = " << max_col
              << "\n  log2 max_i max_j|d_ij| / max_j|ref_ij|   = " << max_row
              << std::endl;
    // magnitude spread of the reference, for context
    double lo = 1e9, hi = -1e9;
    for(int j = 0; j < ref.Width(); ++j)
      for(int i = lower ? j : 0; i < ref.Height(); ++i)
        {
          double l = log2_abs(ref.CRef(i, j));
          if(l > -1e8)
            {
              lo = std::min(lo, l);
              hi = std::max(hi, l);
            }
        }
    std::cout << "  log2|ref| range: [" << lo << ", " << hi << "]"
              << std::endl;
    return 0;
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
    const std::string method = get("method", "mm");
    const mp_bitcnt_t precision = std::stoul(get("precision", "0"));
    const int n = std::stoi(get("n", "-1"));
    const int cols = std::stoi(get("cols", "-1"));
    Multimodular_Blas_Config config;
    config.base_size = std::stoi(get("base", "32"));
    config.guard_bits = std::stoi(get("guard", "64"));
    config.column_chunk = std::stoi(get("chunk", "256"));
    config.flint_crt = std::stoi(get("flint-crt", "0")) != 0;
    config.left_looking = std::stoi(get("left-looking", "1")) != 0;
    config.growth_bits = std::stoi(get("growth", "32"));
    config.left_looking_bytes = size_t(std::stoi(get("ll-mb", "384"))) << 20;

    El::gmp::SetPrecision(precision == 0 ? file_precision(positional[0])
                                         : precision);
    auto L = read_matrix(positional[0], n, n);
    auto X = read_matrix(positional[1], L.Height(), cols);
    std::cout << "S " << L.Height() << "x" << L.Width() << ", B "
              << X.Height() << "x" << X.Width()
              << ", precision " << El::gmp::Precision() << ", method "
              << method << ", base " << config.base_size << ", guard "
              << config.guard_bits << ", chunk " << config.column_chunk
              << ", crt " << (config.flint_crt ? "flint" : "blas")
              << ", trsm " << (config.left_looking ? "left-looking" : "recursive")
              << ", growth " << config.growth_bits
              << std::endl;

    Multimodular_Blas_Timings chol_t, trsm_t;
    auto start = Clock::now();
    if(method == "el")
      El::Cholesky(El::LOWER, L);
    else
      multimodular_cholesky_lower(L, config, chol_t);
    const double chol_s = seconds_since(start);

    start = Clock::now();
    if(method == "el")
      El::Trsm(El::LEFT, El::LOWER, El::NORMAL, El::NON_UNIT,
               El::BigFloat(1), L, X);
    else
      multimodular_trsm_left_lower(L, X, config, trsm_t);
    const double trsm_s = seconds_since(start);

    auto print_t = [](const char *name, double total,
                      const Multimodular_Blas_Timings &t) {
      std::cout << name << " " << total << " s";
      if(t.num_gemms > 0)
        std::cout << "  [base " << t.base_seconds << ", to_int "
                  << t.to_int_seconds << ", blas " << t.blas_seconds
                  << ", from_int " << t.from_int_seconds << "; gemms "
                  << t.num_gemms << ", primes " << t.max_num_primes
                  << ", rescales " << t.num_rescales << "]";
      std::cout << std::endl;
    };
    print_t("cholesky", chol_s, chol_t);
    print_t("trsm", trsm_s, trsm_t);

    if(opts.count("out-L"))
      write_matrix(L, opts["out-L"]);
    if(opts.count("out-X"))
      write_matrix(X, opts["out-X"]);
    return 0;
  }
}

int main(int argc, char **argv)
{
  El::Environment env(argc, argv);
  try
    {
      if(argc < 2)
        throw std::runtime_error("usage: mm_trsm_bench run|compare ...");
      std::string mode = argv[1];
      if(mode == "run")
        return run(argc, argv);
      if(mode == "compare" && argc == 4)
        return compare(argv[2], argv[3]);
      throw std::runtime_error("bad arguments");
    }
  catch(std::exception &e)
    {
      std::cerr << "error: " << e.what() << std::endl;
      return 1;
    }
}
