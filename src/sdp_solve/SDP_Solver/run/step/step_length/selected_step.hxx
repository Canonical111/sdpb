#pragma once
#include "partitioned.hpp"
#include "step.hpp"
#include "sdp_solve/Block_Diagonal_Matrix.hxx"
#include <chrono>
#include <cstdlib>
#include <iostream>

namespace selected {
enum Status { certified=0, uncertain=1, range=2, work_limit=3,
              width=4, exception=5 };
inline double seconds() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
// Constrain BOTH the MPFR interval and the final GMP conversion. The upward
// threshold and downward imported alpha make the comparison conservative.
inline bool usable(const StepResult& s, int bits) {
  R factor(bits),delta(bits),threshold(bits);
  mpfr_set_ui_2exp(delta.x,1,-32,MPFR_RNDN);
  mpfr_ui_sub(factor.x,1,delta.x,MPFR_RNDU);
  mpfr_mul(threshold.x,factor.x,s.alpha_upper.x,MPFR_RNDU);
  auto actual=input(s.alpha,bits);
  return s.alpha>B(0) && mpfr_cmp(s.alpha_lower.x,threshold.x)>=0
       && mpfr_cmp(actual.l.x,threshold.x)>=0
       && mpfr_cmp(actual.h.x,s.alpha_lower.x)<=0;
}
struct Outcome { int status=certified; B alpha=1; double reduction=0,kernel=0; };
// The pristine congruence is retained unmodified. Each trial reduction owns a
// copy; pristine fallback must receive A, not its overwritten trial.
inline Outcome attempt(const Block_Diagonal_Matrix& A,const B& gamma) {
  Outcome out;
  const int bits=El::gmp::Precision();
  const char* injection=std::getenv("SDPB_SELECTED_INJECT");
  const char* rank_text=std::getenv("SDPB_SELECTED_INJECT_RANK");
  const int target=rank_text?std::atoi(rank_text):0;
  for(const auto& original:A.blocks) {
    if(original.Height()==0) continue;
    auto comm=original.Grid().Comm();
    int status=certified; B alpha=1;
    double start=seconds();
    El::DistMatrix<B> trial(original),t(original.Grid());
    El::HermitianTridiag(El::LOWER,trial,t);
    El::DistMatrix<B,El::MD,El::STAR>d(original.Grid()),e(original.Grid());
    El::GetDiagonal(trial,d); El::GetDiagonal(trial,e,-1);
    El::DistMatrix<B,El::STAR,El::STAR>dr(d),er(e);
    out.reduction+=seconds()-start;
    start=seconds();
    if(El::mpi::Rank(comm)==0) {
      mpfr_clear_flags();
      try {
        const std::string inject=(injection && El::mpi::Rank()==target)?injection:"";
        if(inject=="exception") throw std::runtime_error("injected root exception");
        std::vector<B>diag(original.Height()),off(original.Height()-1);
        for(int i=0;i<original.Height();++i)diag[i]=dr.GetLocal(i,0);
        for(int i=0;i<original.Height()-1;++i)off[i]=er.GetLocal(i,0);
        auto result=partitioned(diag,off,bits);
        if(result.status!="certified") status=result.status=="iteration_limit"?work_limit:uncertain;
        else {
          auto s=step(result,gamma,bits);
          if(!usable(s,bits)) status=width;
          else alpha=s.alpha;
        }
        if(inject=="uncertain") status=uncertain;
        if(inject=="range") status=range;
        if(inject=="work") status=work_limit;
        if(inject=="width") status=width;
      } catch(const std::exception& ex) {
        const std::string message=ex.what();
        status=message.find("range")!=std::string::npos?range:exception;
      } catch(...) { status=exception; }
      mpfr_clear_flags();
    }
    // Status always precedes a conditional bound broadcast, on EVERY group.
    El::mpi::Broadcast(status,0,comm);
    if(status==certified) El::mpi::Broadcast(alpha,0,comm);
    out.kernel+=seconds()-start;
    out.status=std::max(out.status,status);
    if(status==certified) out.alpha=El::Min(out.alpha,alpha);
  }
  // All groups finish their local loop before any peer enters pristine fallback.
  out.status=El::mpi::AllReduce(out.status,El::mpi::MAX,El::mpi::COMM_WORLD);
  if(out.status==certified)
    out.alpha=El::mpi::AllReduce(out.alpha,El::mpi::MIN,El::mpi::COMM_WORLD);
  return out;
}
} // namespace selected
