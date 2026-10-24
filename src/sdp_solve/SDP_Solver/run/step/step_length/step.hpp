#pragma once
#include "selected.hpp"
namespace selected {
struct StepResult {
    B alpha;
    R alpha_lower,alpha_upper;
    explicit StepResult(mpfr_prec_t p):alpha_lower(p),alpha_upper(p){}
};
// gamma is the exact represented BigFloat input. For a certified spectral
// bracket of A, compute alpha <= min(1,gamma/(-lambda_min(A))) with all rounding
// toward the conservative side. Margin >=1-gamma for exact I+alpha*A.
// This makes no assertion about a floating congruence's backward error.
inline StepResult step(const Result&r,const B&gamma,mpfr_prec_t p) {
    if(r.status!="certified")throw std::runtime_error("step needs certified bracket");
    auto g=input(gamma,p);
    if(mpfr_sgn(g.l.x)<=0||mpfr_cmp_ui(g.h.x,1)>=0)throw std::runtime_error("gamma outside (0,1)");
    R lo=r.lo,hi=r.hi,den(p);
    mpfr_mul_2si(lo.x,lo.x,r.exponent,MPFR_RNDD);
    mpfr_mul_2si(hi.x,hi.x,r.exponent,MPFR_RNDU);
    finite(lo);finite(hi);
    StepResult out(p);mpfr_set_ui(out.alpha_lower.x,1,MPFR_RNDN);
    mpfr_set_ui(out.alpha_upper.x,1,MPFR_RNDN);
    if(mpfr_sgn(lo.x)<0) {
        mpfr_neg(den.x,lo.x,MPFR_RNDU);
        mpfr_div(out.alpha_lower.x,g.l.x,den.x,MPFR_RNDD);
        mpfr_min(out.alpha_lower.x,out.alpha_lower.x,out.alpha_upper.x,MPFR_RNDD);
    }
    if(mpfr_sgn(hi.x)<0) {
        mpfr_neg(den.x,hi.x,MPFR_RNDD);
        mpfr_div(out.alpha_upper.x,g.h.x,den.x,MPFR_RNDU);
        if(mpfr_cmp_ui(out.alpha_upper.x,1)>0)mpfr_set_ui(out.alpha_upper.x,1,MPFR_RNDN);
    }
    finite(out.alpha_lower);finite(out.alpha_upper);
    mpfr_get_f(out.alpha.gmp_float.get_mpf_t(),out.alpha_lower.x,MPFR_RNDD);
    return out;
}
} // namespace selected
