#pragma once

// #184: with SDPB_MEM_TRACE set, memory_trace(where) prints the proportional set size (PSS:
// shared pages split among the processes mapping them) summed over all ranks and the largest
// rank's, with a Unix timestamp to line up with an external sampler, and with glibc the
// malloc heap summed over all ranks: in use (also mmap'd chunks) and free but kept.
// Collective over COMM_WORLD: every rank must reach the same calls.

#include <El.hpp>

#ifdef __GLIBC__
#include <malloc.h>
#endif

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <string>

inline void memory_trace(const std::string &where)
{
  static const bool enabled = std::getenv("SDPB_MEM_TRACE") != nullptr;
  if(!enabled)
    return;
  double pss_kB = 0;
  {
    std::ifstream smaps("/proc/self/smaps_rollup");
    std::string key;
    while(smaps >> key)
      {
        if(key == "Pss:")
          {
            smaps >> pss_kB;
            break;
          }
        smaps.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
      }
  }
  const double sum
    = El::mpi::AllReduce(pss_kB, El::mpi::SUM, El::mpi::COMM_WORLD);
  const double max
    = El::mpi::AllReduce(pss_kB, El::mpi::MAX, El::mpi::COMM_WORLD);
  double heap[2] = {0, 0}; // in use, free (bytes)
#ifdef __GLIBC__
  {
    // mallinfo's int fields hold one process's heap (well below 2 GiB here).
    const struct mallinfo info = mallinfo();
    heap[0] = static_cast<double>(static_cast<unsigned>(info.uordblks))
              + static_cast<unsigned>(info.hblkhd);
    heap[1] = static_cast<unsigned>(info.fordblks);
  }
#endif
  El::mpi::AllReduce(heap, 2, El::mpi::SUM, El::mpi::COMM_WORLD);
  if(El::mpi::Rank(El::mpi::COMM_WORLD) == 0)
    {
      const double now = std::chrono::duration<double>(
                           std::chrono::system_clock::now().time_since_epoch())
                           .count();
      El::Output("MEMTRACE ", std::to_string(now), " ", where, ": PSS ",
                 sum / 1048576, " GiB (largest rank ", max / 1024,
                 " MiB); heap in use ", heap[0] / 1073741824, " GiB, free ",
                 heap[1] / 1073741824, " GiB");
    }
}
