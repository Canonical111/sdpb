#pragma once

// lab (agent), bfss #74. Opt-in diagnostic, off by default: if LAB_DUMP_S=<dir> is set, the first
// call of step() writes the Schur complement S (m x m, upper triangle) at the working precision to
// <dir>/S.bin, factors it with the same unpivoted Cholesky(UPPER) as step(), writes the diagonal
// of the factor to <dir>/cholS_diag.bin and a summary to <dir>/cholS.txt, and SDPB stops.
//
// File format (little-endian, all fields 8 bytes): header "LABSDMP1", m, precision bits,
// limbs per record NL, layout (1 = upper triangle by columns: j = 0..m-1, i = 0..j; 2 = vector
// of length m), number of records; then records {int64 size, int64 exp, uint64 limb[NL]} with
// the GMP mpf fields: value = sign(size) * sum_{k<|size|} limb[k] * 2^(64 (k - |size| + exp)).

#include "sdpb_util/assert.hxx"

#include <El.hpp>
#include <gmp.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace Sdpb::Sdpa::lab
{
  struct Mpf_Record_Writer
  {
    std::FILE *file = nullptr;
    int64_t limbs = 0;
    std::vector<uint64_t> buffer;
    int64_t written = 0;

    Mpf_Record_Writer(const std::filesystem::path &path, const int64_t m,
                      const int64_t layout, const int64_t records)
    {
      El::BigFloat probe;
      limbs = probe.gmp_float.get_mpf_t()->_mp_prec + 1;
      file = std::fopen(path.c_str(), "wb");
      ASSERT(file != nullptr, "LAB_DUMP_S: cannot open ", path);
      const char magic[8] = {'L', 'A', 'B', 'S', 'D', 'M', 'P', '1'};
      const int64_t header[5]
        = {m, static_cast<int64_t>(El::gmp::Precision()), limbs, layout,
           records};
      ASSERT(std::fwrite(magic, 1, 8, file) == 8);
      ASSERT(std::fwrite(header, 8, 5, file) == 5);
      buffer.resize(2 + limbs);
    }
    void write(const El::BigFloat &x)
    {
      const __mpf_struct *p = x.gmp_float.get_mpf_t();
      const int64_t size = p->_mp_size;
      const int64_t n = size < 0 ? -size : size;
      ASSERT(n <= limbs, "LAB_DUMP_S: mpf with ", n, " limbs > ", limbs);
      std::fill(buffer.begin(), buffer.end(), 0);
      buffer[0] = static_cast<uint64_t>(size);
      buffer[1] = static_cast<uint64_t>(n == 0 ? 0 : p->_mp_exp);
      for(int64_t k = 0; k < n; ++k)
        buffer[2 + k] = p->_mp_d[k];
      ASSERT(std::fwrite(buffer.data(), 8, buffer.size(), file)
               == buffer.size(),
             "LAB_DUMP_S: short write");
      ++written;
    }
    ~Mpf_Record_Writer()
    {
      if(file != nullptr)
        std::fclose(file);
    }
  };

  inline bool dump_S_requested()
  {
    const char *dir = std::getenv("LAB_DUMP_S");
    return dir != nullptr && dir[0] != '\0';
  }

  // Collective: all ranks call it. Writes on rank 0 only.
  inline void dump_S_and_cholesky(El::DistMatrix<El::BigFloat> &S)
  {
    const std::filesystem::path dir(std::getenv("LAB_DUMP_S"));
    const int64_t m = S.Height();
    const bool root = El::mpi::Rank() == 0;
    const int64_t panel_width = 64;
    if(root)
      {
        std::filesystem::create_directories(dir);
        El::Output("LAB_DUMP_S: writing S (", m, " x ", m, ", upper, ",
                   El::gmp::Precision(), " bits) to ", dir);
      }
    {
      std::unique_ptr<Mpf_Record_Writer> writer;
      if(root)
        writer = std::make_unique<Mpf_Record_Writer>(dir / "S.bin.partial",
                                                     m, 1, m * (m + 1) / 2);
      for(int64_t j0 = 0; j0 < m; j0 += panel_width)
        {
          const int64_t j1 = std::min(m, j0 + panel_width);
          auto view = El::LockedView(S, El::IR(0, j1), El::IR(j0, j1));
          El::DistMatrix<El::BigFloat, El::CIRC, El::CIRC> panel(view);
          if(root)
            {
              const auto &local = panel.LockedMatrix();
              for(int64_t j = j0; j < j1; ++j)
                for(int64_t i = 0; i <= j; ++i)
                  writer->write(local.CRef(i, j - j0));
            }
        }
      if(root)
        {
          ASSERT_EQUAL(writer->written, m * (m + 1) / 2);
          writer.reset();
          std::filesystem::rename(dir / "S.bin.partial", dir / "S.bin");
          El::Output("LAB_DUMP_S: S.bin written");
        }
    }
    El::mpi::Barrier(El::mpi::COMM_WORLD);

    // The same factorization as step(); on failure the diagonal is valid up to the failing
    // column (El reports the index within the failing diagonal block).
    std::string status = "ok";
    try
      {
        El::Cholesky(El::UpperOrLowerNS::UPPER, S);
      }
    catch(std::exception &e)
      {
        status = std::string("failed: ") + e.what();
      }
    El::DistMatrix<El::BigFloat, El::CIRC, El::CIRC> diag(S.Grid(), 0);
    {
      El::DistMatrix<El::BigFloat> d(S.Grid());
      El::GetDiagonal(S, d);
      diag = d;
    }
    if(root)
      {
        {
          Mpf_Record_Writer writer(dir / "cholS_diag.bin", m, 2, m);
          const auto &local = diag.LockedMatrix();
          for(int64_t i = 0; i < m; ++i)
            writer.write(local.CRef(i, 0));
        }
        std::ofstream summary(dir / "cholS.txt");
        summary << "m " << m << "\nprecision " << El::gmp::Precision()
                << "\ncholesky " << status << "\n";
        El::Output("LAB_DUMP_S: Cholesky(S) ", status,
                   "; cholS_diag.bin written");
      }
    El::mpi::Barrier(El::mpi::COMM_WORLD);
  }
}
