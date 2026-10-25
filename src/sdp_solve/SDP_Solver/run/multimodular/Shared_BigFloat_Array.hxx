#pragma once

#include "sdpb_util/Shared_Window_Array.hxx"
#include "sdpb_util/assert.hxx"

#include <El.hpp>

// count BigFloat values in an MPI-3 shared-memory window
// (Shared_Window_Array), one record of 2 + limbs mp_limb_t each:
// _mp_size, _mp_exp, then the limbs, as in an mpf_t of precision
// El::gmp::Precision(). A rank writes a value through an mpf struct whose
// limb pointer points into the window (view()), then stores its size and
// exponent with publish(). After Fence(), other ranks read it with view().
// The window may map at different addresses in different processes, so
// each rank makes its own structs.
class Shared_BigFloat_Array
{
  Shared_Window_Array<mp_limb_t> window;
  int limbs; // mpf writes at most _mp_prec + 1 limbs
  size_t record;

  static int value_limbs()
  {
    mpf_t probe;
    mpf_init2(probe, El::gmp::Precision());
    const int result = probe->_mp_prec + 1;
    mpf_clear(probe);
    return result;
  }

public:
  // zero = false: records are left uninitialized (each must be written
  // and published before it is read)
  Shared_BigFloat_Array(const El::mpi::Comm &comm, const size_t count,
                        const bool zero = true)
      : window(comm, count * (2 + value_limbs())),
        limbs(value_limbs()),
        record(2 + limbs)
  {
    static_assert(sizeof(mp_limb_t) == sizeof(long)
                    && sizeof(mp_limb_t) >= sizeof(int),
                  "size and exponent are stored as limbs");
    if(zero && El::mpi::Rank(comm) == 0)
      for(size_t index = 0; index < count; ++index)
        {
          window[index * record] = 0;
          window[index * record + 1] = 0;
        }
    Fence();
  }

  // An mpf struct for the value at index: limbs in the window, size and
  // exponent as last published
  __mpf_struct view(const size_t index)
  {
    __mpf_struct v;
    v._mp_prec = limbs - 1;
    v._mp_size = static_cast<int>(static_cast<long>(window[index * record]));
    v._mp_exp = static_cast<long>(window[index * record + 1]);
    v._mp_d = &window[index * record + 2];
    return v;
  }
  // Store size and exponent of v, whose limbs are those of view(index)
  void publish(const size_t index, const __mpf_struct &v)
  {
    ASSERT(v._mp_d == &window[index * record + 2]);
    window[index * record] = static_cast<mp_limb_t>(long(v._mp_size));
    window[index * record + 1] = static_cast<mp_limb_t>(long(v._mp_exp));
  }
  // value at index := x
  void set(const size_t index, const El::BigFloat &x)
  {
    __mpf_struct v = view(index);
    mpf_set(&v, x.gmp_float.get_mpf_t());
    publish(index, v);
  }
  void Fence() const { window.Fence(); }
};

// True if all ranks of comm share a node (MPI shared memory works)
inline bool on_one_node(const El::mpi::Comm &comm)
{
  MPI_Comm node;
  MPI_Comm_split_type(comm.comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                      &node);
  int node_size, size;
  MPI_Comm_size(node, &node_size);
  MPI_Comm_size(comm.comm, &size);
  MPI_Comm_free(&node);
  return node_size == size;
}
