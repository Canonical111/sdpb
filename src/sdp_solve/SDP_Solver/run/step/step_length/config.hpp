#pragma once
#include <El.hpp>
#include <cstdlib>
#include <stdexcept>
#include <string>

// Selected-step configuration, validated collectively on COMM_WORLD.
// SDPB_SELECTED_STEP chooses which collective schedule a rank enters, so a
// rank-local getenv is not enough: every rank must agree before any rank
// takes the branch. On a mismatch or a malformed value, every rank throws the
// same exception before entering either schedule (deterministic collective
// rejection); no rank silently falls back.
//
//   SDPB_SELECTED_STEP         unset, "" or "0": pristine; "1": selected.
//   SDPB_SELECTED_INJECT       test seam: unset, "", exception, uncertain,
//                              range, work, width.
//   SDPB_SELECTED_INJECT_RANK  unset (=0) or a decimal world rank in [0,size).
//
// The injection variables only alter root-local status ahead of its
// broadcast, so they cannot change collective order; they are still required
// to be well formed and identical when the selected branch is on. When it is
// off they are ignored. Values are read once per process, on the first call.
namespace selected {
enum Inject { none=0, inject_exception, inject_uncertain, inject_range,
              inject_work, inject_width };
struct Config { bool enabled=false; int inject=none; int target=0; };

inline int parse_step(const char* s) {
  if(!s) return 0;
  const std::string v(s);
  if(v.empty()||v=="0") return 0;
  if(v=="1") return 1;
  return -1;
}
inline int parse_inject(const char* s) {
  if(!s) return none;
  const std::string v(s);
  if(v.empty()) return none;
  const char* names[]={"exception","uncertain","range","work","width"};
  for(int i=0;i<5;++i) if(v==names[i]) return i+1;
  return -1;
}
inline int parse_rank(const char* s,int size) {
  if(!s) return 0;
  const std::string v(s);
  if(v.empty()||v.size()>9||v.find_first_not_of("0123456789")!=std::string::npos)
    return -1;
  const int r=std::stoi(v);
  return r<size?r:-1;
}

// Collective on COMM_WORLD: every rank must call it, the same number of times.
inline Config configure() {
  const auto world=El::mpi::COMM_WORLD;
  const int local[3]={parse_step(std::getenv("SDPB_SELECTED_STEP")),
                      parse_inject(std::getenv("SDPB_SELECTED_INJECT")),
                      parse_rank(std::getenv("SDPB_SELECTED_INJECT_RANK"),
                                 El::mpi::Size(world))};
  // One reduction yields both max and min of every field on every rank.
  int send[6],recv[6];
  for(int i=0;i<3;++i) {send[i]=local[i];send[i+3]=-local[i];}
  El::mpi::AllReduce(send,recv,6,El::mpi::MAX,world);
  const auto field=[&](int i,const char* name)->int {
    const int hi=recv[i],lo=-recv[i+3];
    if(lo!=hi)
      throw std::runtime_error(std::string("selected step: ")+name
        +" differs between MPI ranks (min code "+std::to_string(lo)
        +", max code "+std::to_string(hi)+"); set it identically on all ranks");
    if(hi<0)
      throw std::runtime_error(std::string("selected step: malformed ")+name);
    return hi;
  };
  Config c;
  c.enabled=field(0,"SDPB_SELECTED_STEP")==1;
  if(c.enabled) {
    c.inject=field(1,"SDPB_SELECTED_INJECT");
    c.target=field(2,"SDPB_SELECTED_INJECT_RANK");
  }
  return c;
}
} // namespace selected
