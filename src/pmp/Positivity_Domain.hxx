#pragma once

#include "Polynomial.hxx"
#include "sdpb_util/assert.hxx"

#include <El.hpp>

#include <optional>
#include <utility>

// Domain on which a polynomial matrix M(x) must be positive semidefinite:
// the half-line x >= 0 (default), or a closed interval a <= x <= b.
//
// Positivity is imposed through a Markov-Lukacs representation with the two
// Gram matrices Y_1, Y_2 of a Dual_Constraint_Group,
//
//   M(x) = w_0(x) q_0(x)^T Y_1 q_0(x) + w_1(x) q_1(x)^T Y_2 q_1(x),
//
// where q_0(x), q_1(x) are the bilinear bases of degrees
// delta1 = degree/2 and delta2 = (degree+1)/2 - 1, and
//
//   [0, inf):            w_0 = 1,      w_1 = x,
//   [a, b], even degree: w_0 = 1,      w_1 = (x - a)(b - x),
//   [a, b], odd degree:  w_0 = x - a,  w_1 = b - x.
//
// Both kinds of domain give the same delta1, delta2, hence the same SDP block
// sizes; only the sampled bilinear bases differ.
//
// On an interval, the coefficients of the polynomials (and of user-supplied
// bilinear bases) may be given in the Chebyshev basis T_i(u) of [a, b],
// u = (2x - a - b)/(b - a), instead of in monomials x^i. Monomial
// coefficients of a polynomial of size O(1) on [a, b] can grow exponentially
// with the degree (like (3 + 2 sqrt 2)^degree on [0, 1]), and evaluating them
// at the sample points then loses that many digits; Chebyshev coefficients
// stay O(1) and are evaluated by Clenshaw's recurrence without cancellation.
enum class Polynomial_Basis
{
  monomial,
  chebyshev
};

struct Positivity_Domain
{
  // Endpoints (a, b) of a closed interval; empty for the half-line [0, inf).
  std::optional<std::pair<El::BigFloat, El::BigFloat>> interval;
  // Basis of the polynomial coefficients; chebyshev requires an interval.
  Polynomial_Basis polynomial_basis = Polynomial_Basis::monomial;

  [[nodiscard]] bool is_interval() const { return interval.has_value(); }

  [[nodiscard]] const El::BigFloat &a() const
  {
    ASSERT(is_interval());
    return interval->first;
  }
  [[nodiscard]] const El::BigFloat &b() const
  {
    ASSERT(is_interval());
    return interval->second;
  }

  // Weight w_parity(x) of the Gram matrix Y_{parity+1} for a polynomial
  // matrix of the given degree (degree = number of sample points - 1).
  [[nodiscard]] El::BigFloat
  weight(const size_t parity, const size_t degree, const El::BigFloat &x) const
  {
    ASSERT(parity == 0 || parity == 1, DEBUG_STRING(parity));
    if(!is_interval())
      return parity == 0 ? El::BigFloat(1) : x;
    if(degree % 2 == 0)
      return parity == 0 ? El::BigFloat(1) : (x - a()) * (b() - x);
    return parity == 0 ? x - a() : b() - x;
  }

  // p(x), with the coefficients of p in polynomial_basis
  [[nodiscard]] El::BigFloat
  evaluate(const Polynomial &p, const El::BigFloat &x) const
  {
    if(polynomial_basis == Polynomial_Basis::monomial)
      return p(x);
    // Clenshaw: b_i = c_i + 2u b_{i+1} - b_{i+2}, p = c_0 + u b_1 - b_2
    const auto &c = p.coefficients;
    ASSERT(!c.empty());
    const El::BigFloat u = (2 * x - a() - b()) / (b() - a());
    El::BigFloat b1 = 0, b2 = 0;
    for(size_t i = c.size() - 1; i >= 1; --i)
      {
        El::BigFloat b0 = c[i] + 2 * u * b1 - b2;
        b2 = std::move(b1);
        b1 = std::move(b0);
      }
    return c[0] + u * b1 - b2;
  }

  [[nodiscard]] bool contains(const El::BigFloat &x) const
  {
    if(!is_interval())
      return x >= 0;
    return a() <= x && x <= b();
  }

  void validate() const
  {
    if(is_interval())
      ASSERT(a() < b(), "domain: interval [a, b] requires a < b",
             DEBUG_STRING(a()), DEBUG_STRING(b()));
    else
      ASSERT(polynomial_basis == Polynomial_Basis::monomial,
             "domain: polynomialBasis \"chebyshev\" requires an interval");
  }
};
