#pragma once
#include "selected.hpp"
namespace selected {
// Supplementary extension: solve exact-zero-separated tridiagonal blocks
// independently. min_b lower_b <= min spectrum <= min_b upper_b. Bounds are
// returned in physical units (exponent=0), with the same working-bit arithmetic.
inline Result partitioned(const std::vector<B>&d,const std::vector<B>&e,int bits) {
    if(d.empty()||e.size()+1!=d.size())throw std::runtime_error("partition shape");
    Result out(bits);out.status="certified";
    mpfr_set_inf(out.lo.x,1);mpfr_set_inf(out.hi.x,1);
    size_t start=0;
    for(size_t end=0;end<d.size();++end) {
        if(end+1<d.size()&&e[end]!=B(0))continue;
        std::vector<B>diag(d.begin()+start,d.begin()+end+1),off(e.begin()+start,e.begin()+end);
        Kernel k(diag,off,bits);auto r=k.solve();
        if(r.status!="certified") {out.status=r.status;return out;}
        mpfr_mul_2si(r.lo.x,r.lo.x,r.exponent,MPFR_RNDD);
        mpfr_mul_2si(r.hi.x,r.hi.x,r.exponent,MPFR_RNDU);finite(r.lo);finite(r.hi);
        mpfr_min(out.lo.x,out.lo.x,r.lo.x,MPFR_RNDD);
        mpfr_min(out.hi.x,out.hi.x,r.hi.x,MPFR_RNDU);
        out.counts+=r.counts;out.ambiguous+=r.ambiguous;out.newtons+=r.newtons;
        start=end+1;
    }
    return out;
}
} // namespace selected
