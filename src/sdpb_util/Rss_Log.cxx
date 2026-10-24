#include "Rss_Log.hxx"

#include <El.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>

namespace
{
  // Read "VmRSS:" and "VmHWM:" (kB) from /proc/self/status
  void read_status(long &rss_kb, long &hwm_kb)
  {
    rss_kb = -1;
    hwm_kb = -1;
    std::ifstream f("/proc/self/status");
    std::string key;
    long value;
    std::string unit;
    while(f >> key)
      {
        if(key == "VmRSS:" || key == "VmHWM:")
          {
            f >> value >> unit;
            (key == "VmRSS:" ? rss_kb : hwm_kb) = value;
          }
        else
          f.ignore(1 << 20, '\n');
      }
  }
}

void rss_log(const char *event, const std::string &name)
{
  static const char *dir = std::getenv("SDPB_RSS_LOG");
  if(dir == nullptr || dir[0] == '\0')
    return;
  static std::FILE *out = [] {
    int initialized = 0;
    MPI_Initialized(&initialized);
    int rank = -1;
    if(initialized)
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    const std::string path = std::string(dir) + "/rss-rank"
                             + std::to_string(rank) + ".tsv";
    return std::fopen(path.c_str(), "a");
  }();
  if(out == nullptr)
    return;
  static const auto t0 = std::chrono::steady_clock::now();
  const double t = std::chrono::duration<double>(
                     std::chrono::steady_clock::now() - t0)
                     .count();
  long rss_kb, hwm_kb;
  read_status(rss_kb, hwm_kb);
  std::fprintf(out, "%.3f\t%s\t%s\t%ld\t%ld\n", t, event, name.c_str(),
               rss_kb, hwm_kb);
  std::fflush(out);
}
