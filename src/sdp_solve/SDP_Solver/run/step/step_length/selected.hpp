#pragma once
#include <El.hpp>
#include <mpfr.h>
#include <vector>
#include <stdexcept>
#include <string>

// All interval arithmetic uses MPFR at the caller's BigFloat working precision.
// GMP inputs are exact dyadics, enclosed on import, including GMP guard limbs.
// Exceptions mean no certified answer; never silently substitute a midpoint.
namespace selected {
using B = El::BigFloat;
struct R {
    mpfr_t x;
    explicit R(mpfr_prec_t p) { mpfr_init2(x,p); mpfr_set_zero(x,0); }
    R(const R& a) : R(mpfr_get_prec(a.x)) { mpfr_set(x,a.x,MPFR_RNDN); }
    R& operator=(const R& a) { mpfr_set(x,a.x,MPFR_RNDN); return *this; }
    ~R() { mpfr_clear(x); }
};
inline void finite(const R& a) {
    if (!mpfr_number_p(a.x) || mpfr_underflow_p() || mpfr_overflow_p())
        throw std::runtime_error("nonfinite/range");
}
struct I {
    R l,h;
    explicit I(mpfr_prec_t p) : l(p),h(p) {}
    mpfr_prec_t prec() const { return mpfr_get_prec(l.x); }
};
inline I integer(long n,mpfr_prec_t p) {
    I a(p); mpfr_set_si(a.l.x,n,MPFR_RNDD); mpfr_set_si(a.h.x,n,MPFR_RNDU); return a;
}
inline I point(const R& x) { I a(mpfr_get_prec(x.x)); a.l=x; a.h=x; return a; }
inline I input(const B& b,mpfr_prec_t p) {
    I a(p); mpfr_set_f(a.l.x,b.gmp_float.get_mpf_t(),MPFR_RNDD);
    mpfr_set_f(a.h.x,b.gmp_float.get_mpf_t(),MPFR_RNDU);
    finite(a.l); finite(a.h);
    // Reject underflow as well as overflow: an unreported zero is not an enclosure.
    if (b != B(0) && (mpfr_zero_p(a.l.x)||mpfr_zero_p(a.h.x)))
        throw std::runtime_error("input underflow");
    return a;
}
inline I add(const I&a,const I&b) {
    I c(a.prec()); mpfr_add(c.l.x,a.l.x,b.l.x,MPFR_RNDD);
    mpfr_add(c.h.x,a.h.x,b.h.x,MPFR_RNDU); finite(c.l);finite(c.h);return c;
}
inline I sub(const I&a,const I&b) {
    I c(a.prec()); mpfr_sub(c.l.x,a.l.x,b.h.x,MPFR_RNDD);
    mpfr_sub(c.h.x,a.h.x,b.l.x,MPFR_RNDU); finite(c.l);finite(c.h);return c;
}
inline I mul(const I&a,const I&b) {
    I c(a.prec()); R v(a.prec());
    if(mpfr_sgn(a.l.x)>=0&&mpfr_sgn(b.l.x)>=0) {
        mpfr_mul(c.l.x,a.l.x,b.l.x,MPFR_RNDD);
        mpfr_mul(c.h.x,a.h.x,b.h.x,MPFR_RNDU);
        finite(c.l);finite(c.h);return c;
    }
    mpfr_set_inf(c.l.x,1); mpfr_set_inf(c.h.x,-1);
    for(auto x:{a.l.x,a.h.x}) for(auto y:{b.l.x,b.h.x}) {
        mpfr_mul(v.x,x,y,MPFR_RNDD); mpfr_min(c.l.x,c.l.x,v.x,MPFR_RNDD);
        mpfr_mul(v.x,x,y,MPFR_RNDU); mpfr_max(c.h.x,c.h.x,v.x,MPFR_RNDU);
    }
    finite(c.l);finite(c.h);return c;
}
inline bool zero(const I&a) { return mpfr_sgn(a.l.x)<=0 && mpfr_sgn(a.h.x)>=0; }
inline I div(const I&a,const I&b) {
    if(zero(b)) throw std::runtime_error("zero interval divisor");
    if(mpfr_sgn(a.l.x)>=0) {
        I c(a.prec());
        if(mpfr_sgn(b.l.x)>0) {
            mpfr_div(c.l.x,a.l.x,b.h.x,MPFR_RNDD);
            mpfr_div(c.h.x,a.h.x,b.l.x,MPFR_RNDU);
        } else {
            mpfr_div(c.l.x,a.h.x,b.h.x,MPFR_RNDD);
            mpfr_div(c.h.x,a.l.x,b.l.x,MPFR_RNDU);
        }
        finite(c.l);finite(c.h);return c;
    }
    I inv(b.prec()); mpfr_ui_div(inv.l.x,1,b.h.x,MPFR_RNDD);
    mpfr_ui_div(inv.h.x,1,b.l.x,MPFR_RNDU); return mul(a,inv);
}
inline I absolute(const I&a) {
    I c(a.prec());
    if(mpfr_sgn(a.l.x)>=0) return a;
    if(mpfr_sgn(a.h.x)<=0) {
        mpfr_neg(c.l.x,a.h.x,MPFR_RNDD);mpfr_neg(c.h.x,a.l.x,MPFR_RNDU);
    } else {
        mpfr_set_zero(c.l.x,0);mpfr_neg(c.h.x,a.l.x,MPFR_RNDU);
        mpfr_max(c.h.x,c.h.x,a.h.x,MPFR_RNDU);
    }
    return c;
}
inline void scale(I&a,long e) {
    mpfr_mul_2si(a.l.x,a.l.x,e,MPFR_RNDD);
    mpfr_mul_2si(a.h.x,a.h.x,e,MPFR_RNDU);
    finite(a.l);finite(a.h);
}
struct Result {
    R lo,hi; long exponent=0; int counts=0,ambiguous=0,newtons=0,iterations=0;
    std::string status="uninitialized";
    explicit Result(mpfr_prec_t p) : lo(p),hi(p) {}
};
class Kernel {
    mpfr_prec_t p;
    std::vector<I> d,e,e2;
    long exponent=0;
public:
    Kernel(const std::vector<B>& diag,const std::vector<B>& off,mpfr_prec_t bits) : p(bits) {
        mpfr_clear_flags();
        if(p<64 || diag.empty() || off.size()+1!=diag.size())
            throw std::runtime_error("invalid size/precision");
        R largest(p);
        for(const auto& b:diag) {d.push_back(input(b,p));auto a=absolute(d.back());mpfr_max(largest.x,largest.x,a.h.x,MPFR_RNDU);}
        for(const auto& b:off) {e.push_back(input(b,p));auto a=absolute(e.back());mpfr_max(largest.x,largest.x,a.h.x,MPFR_RNDU);}
        if(!mpfr_zero_p(largest.x)) exponent=mpfr_get_exp(largest.x);
        // MPFR's configured exponent range is intentionally not changed globally.
        for(auto& a:d) scale(a,-exponent);
        for(auto& a:e) { scale(a,-exponent); e2.push_back(mul(a,a)); }
    }
    // Returns # eigenvalues strictly below x if every LDL sign is certified.
    // Exact zero offdiagonals split independent blocks. Uncertain pivots and
    // exact zero internal pivots are undecided. Exact zero terminal pivots
    // contribute a zero eigenvalue, with no division or arbitrary sign.
    int count(const R&x, I* trace=nullptr,I* trace2=nullptr) const {
        if((trace2&&!trace)||(trace&&trace->prec()!=p)||(trace2&&trace2->prec()!=p))
            throw std::runtime_error("invalid moment output precision/shape");
        std::vector<I> piv,h; int negatives=0;
        for(size_t j=0;j<d.size();++j) {
            I q=sub(d[j],point(x));
            if(j && !(mpfr_zero_p(e[j-1].l.x)&&mpfr_zero_p(e[j-1].h.x)))
                q=sub(q,div(e2[j-1],piv.back()));
            if(zero(q)) {
                bool exact=mpfr_zero_p(q.l.x)&&mpfr_zero_p(q.h.x);
                bool terminal=j+1==d.size()||(mpfr_zero_p(e[j].l.x)&&mpfr_zero_p(e[j].h.x));
                // A zero last LDL pivot contributes one zero eigenvalue. It
                // cannot be divided through; inverse moments remain undecided.
                if(exact&&terminal&&!trace) {piv.push_back(q);continue;}
                return -1;
            }
            if(mpfr_sgn(q.h.x)<0) ++negatives;
            if(trace2) {
                I v=integer(1,p);
                if(j) v=add(v,mul(e2[j-1],div(h.back(),mul(piv.back(),piv.back()))));
                h.push_back(v); // h_j = -dq_j/dx, positive below the spectrum
            }
            piv.push_back(q);
        }
        if(trace && negatives==0) {
            I s=integer(0,p),sum=integer(0,p),one=integer(1,p),ds=integer(0,p),sum2=integer(0,p);
            for(size_t j=d.size();j-->0;) {
                I inv=div(one,piv[j]);
                if(trace2) {
                    I dinv=mul(h[j],mul(inv,inv));
                    ds=add(dinv,j+1<d.size()?mul(e2[j],add(mul(mul(integer(2,p),mul(inv,dinv)),s),mul(mul(inv,inv),ds))):integer(0,p));
                    sum2=add(sum2,ds);
                }
                s=add(inv,j+1<d.size()?mul(mul(e2[j],mul(inv,inv)),s):integer(0,p));
                sum=add(sum,s);
            }
            *trace=sum;
            if(trace2)*trace2=sum2;
        }
        return negatives;
    }
    // Sufficient full-step certificate, including indefinite directions whose
    // minimum is >=-gamma. A false return means only "not certified".
    bool full_step(const B&gamma) const {
        auto g=input(gamma,p);
        if(mpfr_sgn(g.l.x)<=0||mpfr_cmp_ui(g.h.x,1)>=0)
            throw std::runtime_error("gamma outside (0,1)");
        R x(p);mpfr_neg(x.x,g.l.x,MPFR_RNDU);
        mpfr_mul_2si(x.x,x.x,-exponent,MPFR_RNDU);finite(x);
        return count(x)==0;
    }
    Result solve(bool accelerated=true,int guard_bits=32,int max_iterations=0,bool second_moment=true) const {
        if(guard_bits<8 || guard_bits>=p-8) throw std::runtime_error("invalid tolerance");
        Result out(p);out.exponent=exponent;
        R lo(p),hi(p),width(p),tol(p);
        mpfr_set_inf(lo.x,1);mpfr_set_inf(hi.x,-1);
        for(size_t j=0;j<d.size();++j) {
            I r=integer(0,p);
            if(j) r=add(r,absolute(e[j-1]));
            if(j+1<d.size()) r=add(r,absolute(e[j]));
            I row=add(d[j],r),left=sub(d[j],r);
            mpfr_min(lo.x,lo.x,left.l.x,MPFR_RNDD);
            mpfr_max(hi.x,hi.x,row.h.x,MPFR_RNDU);
        }
        mpfr_set_ui_2exp(tol.x,1,-p+guard_bits,MPFR_RNDU);
        if(mpfr_cmp(lo.x,hi.x)!=0) {
            R origin(p);++out.counts;int c=count(origin);
            if(c==0)mpfr_max(lo.x,lo.x,origin.x,MPFR_RNDD);
            else if(c>0)mpfr_min(hi.x,hi.x,origin.x,MPFR_RNDU);
            else ++out.ambiguous;
        }
        if(!max_iterations) max_iterations=4*p+128;
        for(int k=0;k<max_iterations;++k) {
            out.iterations=k;
            mpfr_sub(width.x,hi.x,lo.x,MPFR_RNDU);
            if(mpfr_cmp(width.x,tol.x)<=0) {out.status="certified";break;}
            bool moved=false;
            if(accelerated) {
                I trace(p),trace2(p);R x=lo;
                ++out.counts;
                int c=count(x,&trace,second_moment?&trace2:nullptr);
                if(c<0) {
                    ++out.ambiguous;
                    // At a rounding-limited endpoint, retreat by a fraction of
                    // bracket width. The trace ratio can still contract the
                    // upper endpoint quadratically from this safe exterior point.
                    if(second_moment) {
                        R delta(p);mpfr_div_2ui(delta.x,width.x,3,MPFR_RNDU);
                        mpfr_sub(x.x,lo.x,delta.x,MPFR_RNDD);++out.counts;
                        c=count(x,&trace,&trace2);if(c<0)++out.ambiguous;
                    }
                }
                if(c==0 && mpfr_sgn(trace.l.x)>0) {
                    // For x < lambda_min: 1/tr((T-xI)^-1) <= lambda_min-x
                    //                         <= n/tr((T-xI)^-1).
                    I lower=add(point(x),div(integer(1,p),trace));
                    // mu_max >= sum(mu_i^2)/sum(mu_i), mu_i>0. Thus
                    // lambda_min-x <= tr(inv)/tr(inv^2), a tighter upper bound.
                    I upper=add(point(x),second_moment?div(trace,trace2):div(integer(long(d.size()),p),trace));
                    if(mpfr_cmp(upper.h.x,hi.x)<0) {hi=upper.h;moved=true;}
                    if(mpfr_cmp(lower.l.x,lo.x)>0) {lo=lower.l;moved=true;}
                    if(moved) ++out.newtons;
                }
            }
            // Always attempt inertia bisection, even if Newton advanced a little.
            bool bisected=false;
            for(int fraction:{2,1,3}) {
                R x(p),delta(p);
                mpfr_sub(delta.x,hi.x,lo.x,MPFR_RNDN);
                mpfr_mul_ui(delta.x,delta.x,fraction,MPFR_RNDN);
                mpfr_div_2ui(delta.x,delta.x,2,MPFR_RNDN);
                mpfr_add(x.x,lo.x,delta.x,MPFR_RNDN);
                if(mpfr_cmp(x.x,lo.x)<=0 || mpfr_cmp(x.x,hi.x)>=0) continue;
                ++out.counts;int c=count(x);
                if(c<0) {++out.ambiguous;continue;}
                if(c==0) lo=x; else hi=x;
                moved=bisected=true;break;
            }
            (void)bisected;
            if(mpfr_cmp(lo.x,hi.x)>0) throw std::runtime_error("inverted bracket");
            if(!moved) {out.status="undecided";break;}
        }
        if(out.status=="uninitialized") out.status="iteration_limit";
        out.lo=lo;out.hi=hi;
        return out;
    }
};
inline std::string decimal(const R&a) {
    char* s=nullptr;mpfr_asprintf(&s,"%.12Re",a.x);std::string t(s);mpfr_free_str(s);return t;
}
} // namespace selected
