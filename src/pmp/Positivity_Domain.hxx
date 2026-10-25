#pragma once

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
struct Positivity_Domain
{
  // Endpoints (a, b) of a closed interval; empty for the half-line [0, inf).
  std::optional<std::pair<El::BigFloat, El::BigFloat>> interval;

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
  }
};
