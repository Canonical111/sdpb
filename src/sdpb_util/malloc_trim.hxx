#pragma once

#include "Environment.hxx"
#include "assert.hxx"
#include "Timers/Timers.hxx"

#include <cstdlib>
#include <string>

#if defined(__GLIBC__)
#include <malloc.h>
#endif

// Force libc allocator to release deallocated memory back to OS.
// See https://man7.org/linux/man-pages/man3/malloc_trim.3.html
// NB: this is glibc-only! For example, MUSL (used in alpine-based SDPB Docker image
// ) does not have such function.
// TODO: detect other libc versions providing malloc_trim() .
inline void malloc_trim(const Environment &env, Timers &timers)
{
#if defined(__GLIBC__)
  El::mpi::Barrier(env.comm_shared_mem);

  Scoped_Timer timer(timers, "malloc_trim");
  const bool success = malloc_trim(0);
  timer.stop();

  if(!success)
    PRINT_WARNING("rank=", El::mpi::Rank(), ": malloc_trim(0) failed.");

  El::mpi::Barrier(env.comm_shared_mem);
#endif
}

// #184: return this rank's free heap pages to the OS (glibc; no barrier). A solver step
// frees and reallocates GBs of BigFloats phase after phase (S, the bilinear pairings,
// the Cholesky factors, L^{-1} B, the search direction's temporaries), and glibc keeps
// the freed pages: the resident heap grows to the union of all of them, well above what
// is ever in use at once. Called between the phases; SDPB_HEAP_TRIM=0 disables it.
inline void release_free_heap()
{
#if defined(__GLIBC__)
  static const bool enabled = [] {
    const char *value = std::getenv("SDPB_HEAP_TRIM");
    return value == nullptr || std::string(value) != "0";
  }();
  if(enabled)
    malloc_trim(0);
#endif
}