#pragma once

#include "Polynomial.hxx"
#include "Positivity_Domain.hxx"
#include "sdpb_util/Damped_Rational.hxx"
#include "sdpb_util/Simple_Matrix.hxx"

#include <optional>

// Let M(x) be a matrix whose entries are vectors of polynomials:
//
//   M(x) = ( \vec P^{00}(x) ... \vec P^{m0}(x) )
//          ( ...                               )
//          ( \vec P^{0n}(x) ... \vec P^{mn}(x) )
//
// where each vector has length N+1:
//
//   \vec P^{rs}(x) = (P^{rs}_{-1}(x), P^{rs}_0, ... , P^{rs}_{N-1}(x))
//
// Consider a vector y = (y_0, ..., y_{N-1}) of length N, and let
// (1,y) denote the vector of length N+1 whose components are 1,
// followed by the components of y.  As explained in the manual, the
// constraint
//
//   (1,y) . M(x) is positive semidefinite
//
// is equivalent to a Dual_Constraint_Group
//
//   Tr(A_p Y) + (B y)_p = c_p
//
// A Polynomial_Vector_Matrix contains the data needed to construct this
// Dual_Constraint_Group.
// It also may contain Damped_Rational prefactor
// and the domain of x (by default x >= 0, see Positivity_Domain.hxx).
struct Polynomial_Vector_Matrix
{
  // TODO: This is actually a symmetric matrix (dim x dim) of a vector
  // of polynomials.  So we should use a proper symmetric matrix class.
  Simple_Matrix<Polynomial_Vector> polynomials;

  Damped_Rational prefactor;
  Damped_Rational reduced_prefactor;

  // A list of real numbers x_k (0 <= k <= degree(M)) at which to
  // sample M(x) to construct the v_{b,k}.
  std::vector<El::BigFloat> sample_points;

  // A list of real numbers s_k (0 <= k <= degree(M)) to scale M(x_k)
  // and the corresponding v_{b,k}.
  std::vector<El::BigFloat> sample_scalings;
  std::vector<El::BigFloat> reduced_sample_scalings;

  // bilinearBasis[m] = q_m(x) (0 <= m <= degree/2), where q_m is a
  // polynomial with degree deg(q_m) = m.
  // For an interval domain without user-supplied bilinear bases, both are
  // empty: pmp2sdp then samples bases orthonormal at the sample points
  // directly, without monomial coefficients (see sample_bilinear_basis).
  std::array<Polynomial_Vector, 2> bilinear_basis;

  // Where (1,y) . M(x) must be positive semidefinite: x >= 0 or [a, b].
  Positivity_Domain domain;

  Polynomial_Vector_Matrix(
    const Simple_Matrix<Polynomial_Vector> &polynomials,
    const std::optional<Damped_Rational> &prefactor_opt,
    const std::optional<Damped_Rational> &reduced_prefactor_opt,
    const std::optional<int64_t> &max_num_poles_opt,
    const std::optional<std::vector<El::BigFloat>> &sample_points_opt,
    const std::optional<std::vector<El::BigFloat>> &sample_scalings_opt,
    const std::optional<std::vector<El::BigFloat>> &reduced_sample_scalings_opt,
    const std::optional<std::array<Polynomial_Vector, 2>> &bilinear_basis_opt,
    const std::optional<Positivity_Domain> &domain_opt) noexcept(false);

  // True if bilinear_basis is empty and pmp2sdp builds orthonormal bases
  // at the sample points (interval domain without user-supplied bases).
  [[nodiscard]] bool has_default_interval_basis() const
  {
    return domain.is_interval() && bilinear_basis[0].empty();
  }

  void validate(int64_t num_points) const;
};
