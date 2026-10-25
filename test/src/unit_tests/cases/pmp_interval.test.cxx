#include "pmp/Polynomial_Vector_Matrix.hxx"
#include "pmp/Positivity_Domain.hxx"
#include "sdpb_util/Boost_Float.hxx"

#include <catch2/catch_amalgamated.hpp>

#include "unit_tests/util/util.hxx"

using Test_Util::REQUIRE_Equal::diff;

std::vector<Boost_Float>
interval_sample_points(const size_t &num_points, const Boost_Float &a,
                       const Boost_Float &b);

std::array<El::Matrix<El::BigFloat>, 2>
sample_bilinear_basis(const std::array<Polynomial_Vector, 2> &bilinear_basis,
                      const std::vector<El::BigFloat> &sample_points,
                      const std::vector<El::BigFloat> &sample_scalings,
                      const Positivity_Domain &domain);

namespace
{
  Positivity_Domain interval(const El::BigFloat &a, const El::BigFloat &b)
  {
    Positivity_Domain domain;
    domain.interval.emplace(a, b);
    return domain;
  }

  Polynomial polynomial(const std::vector<El::BigFloat> &coefficients)
  {
    Polynomial p;
    p.coefficients = coefficients;
    return p;
  }

  // Value at x of the polynomial of degree xs.size()-1 through (xs, ys)
  El::BigFloat lagrange(const std::vector<El::BigFloat> &xs,
                        const std::vector<El::BigFloat> &ys,
                        const El::BigFloat &x)
  {
    El::BigFloat result = 0;
    for(size_t i = 0; i < xs.size(); ++i)
      {
        El::BigFloat term = ys.at(i);
        for(size_t j = 0; j < xs.size(); ++j)
          if(j != i)
            term *= (x - xs.at(j)) / (xs.at(i) - xs.at(j));
        result += term;
      }
    return result;
  }

  // 1x1 polynomial matrix with the vector (1 + x^4, x^2 + x^4/12)
  Simple_Matrix<Polynomial_Vector> polynomials_1d()
  {
    Simple_Matrix<Polynomial_Vector> matrix(1, 1);
    matrix(0, 0) = {polynomial({1, 0, 0, 0, 1}),
                    polynomial({0, 0, 1, 0, El::BigFloat(1) / 12})};
    return matrix;
  }
}

TEST_CASE("pmp_interval")
{
  if(El::mpi::Rank() != 0)
    return;

  // Unit tests run at 768 bits; allow a few bits of rounding error.
  const int diff_precision = 740;
  Test_Util::REQUIRE_Equal::Diff_Precision diff_prec(diff_precision);
  const El::BigFloat eps = El::BigFloat(1) >>= diff_precision;

  SECTION("Positivity_Domain weights")
  {
    const El::BigFloat x = El::BigFloat(3) / 4;
    const Positivity_Domain half_line;
    for(size_t degree : {0, 1, 2, 7})
      {
        CAPTURE(degree);
        REQUIRE(half_line.weight(0, degree, x) == El::BigFloat(1));
        REQUIRE(half_line.weight(1, degree, x) == x);
      }

    const auto domain = interval(-1, 2);
    REQUIRE(domain.weight(0, 4, x) == El::BigFloat(1));
    REQUIRE(domain.weight(1, 4, x) == (x + 1) * (2 - x));
    REQUIRE(domain.weight(0, 5, x) == x + 1);
    REQUIRE(domain.weight(1, 5, x) == 2 - x);
    REQUIRE(domain.contains(El::BigFloat(-1)));
    REQUIRE(domain.contains(El::BigFloat(2)));
    REQUIRE(!domain.contains(El::BigFloat(-1) - eps));
    REQUIRE_THROWS(interval(1, 1).validate());
  }

  SECTION("Chebyshev sample points")
  {
    const auto points = to_BigFloat_Vector(interval_sample_points(3, 0, 1));
    const El::BigFloat c = El::Sqrt(El::BigFloat(3)) / 4;
    const std::vector<El::BigFloat> expected{El::BigFloat(1) / 2 - c,
                                             El::BigFloat(1) / 2,
                                             El::BigFloat(1) / 2 + c};
    DIFF(points, expected);

    for(size_t num_points = 1; num_points <= 20; ++num_points)
      {
        CAPTURE(num_points);
        const El::BigFloat a = -2, b = 5;
        const auto xs
          = to_BigFloat_Vector(interval_sample_points(num_points, -2, 5));
        REQUIRE(xs.size() == num_points);
        for(size_t k = 0; k < num_points; ++k)
          {
            CAPTURE(k);
            REQUIRE(a < xs.at(k));
            REQUIRE(xs.at(k) < b);
            if(k > 0)
              REQUIRE(xs.at(k - 1) < xs.at(k));
            // symmetric about the midpoint
            DIFF(xs.at(k) + xs.at(num_points - 1 - k), a + b);
          }
      }
  }

  SECTION("Default bases are orthonormal, weighted polynomials")
  {
    for(const auto &[a, b] : std::vector<std::pair<El::BigFloat, El::BigFloat>>{
          {0, 1}, {-2, 3}, {1, 21}})
      for(size_t degree = 0; degree <= 14; ++degree)
        {
          CAPTURE(a);
          CAPTURE(b);
          CAPTURE(degree);
          const auto domain = interval(a, b);
          const auto points = to_BigFloat_Vector(interval_sample_points(
            degree + 1, to_Boost_Float(a), to_Boost_Float(b)));
          // A positive scaling, as from a prefactor 1/(x+3)
          std::vector<El::BigFloat> scalings;
          for(const auto &x : points)
            scalings.push_back(1 / (x + 3));

          const auto bases
            = sample_bilinear_basis({}, points, scalings, domain);
          const std::array<size_t, 2> sizes{degree / 2 + 1, (degree + 1) / 2};

          for(const size_t parity : {0, 1})
            {
              CAPTURE(parity);
              const auto &v = bases.at(parity);
              REQUIRE(static_cast<size_t>(v.Height()) == sizes.at(parity));
              REQUIRE(static_cast<size_t>(v.Width()) == degree + 1);

              // Orthonormal rows
              for(int i = 0; i < v.Height(); ++i)
                for(int j = 0; j <= i; ++j)
                  {
                    El::BigFloat dot = 0;
                    for(int k = 0; k < v.Width(); ++k)
                      dot += v.CRef(i, k) * v.CRef(j, k);
                    CAPTURE(i);
                    CAPTURE(j);
                    CAPTURE(dot);
                    if(i == j)
                      {
                        DIFF(dot, El::BigFloat(1));
                      }
                    else
                      {
                        REQUIRE(El::Abs(dot) < eps);
                      }
                  }

              // Row i is sqrt(s_k w(x_k)) p_i(x_k) with deg p_i = i:
              // the polynomial through the first i+1 points predicts the rest.
              for(int i = 0; i < v.Height(); ++i)
                {
                  CAPTURE(i);
                  std::vector<El::BigFloat> ys;
                  for(int k = 0; k < v.Width(); ++k)
                    {
                      const auto &x = points.at(k);
                      ys.push_back(
                        v.CRef(i, k)
                        / El::Sqrt(scalings.at(k)
                                   * domain.weight(parity, degree, x)));
                    }
                  const std::vector<El::BigFloat> xs_fit(points.begin(),
                                                         points.begin() + i
                                                           + 1),
                    ys_fit(ys.begin(), ys.begin() + i + 1);
                  for(int k = i + 1; k < v.Width(); ++k)
                    {
                      CAPTURE(k);
                      // Lagrange interpolation in x loses precision for
                      // wide intervals; compare at a lower precision.
                      Test_Util::REQUIRE_Equal::Diff_Precision lower(600);
                      DIFF(lagrange(xs_fit, ys_fit, points.at(k)), ys.at(k));
                    }
                }
            }
        }
  }

  SECTION("User-supplied bases are weighted on an interval")
  {
    const auto domain = interval(1, 3);
    const size_t degree = 3; // odd: weights x - 1 and 3 - x
    const std::vector<El::BigFloat> points{1, El::BigFloat(3) / 2, 2, 3};
    const std::vector<El::BigFloat> scalings{1, 2, 3, 4};
    const Polynomial_Vector basis{polynomial({1}), polynomial({-1, 1})};
    const auto bases
      = sample_bilinear_basis({basis, basis}, points, scalings, domain);
    for(const size_t parity : {0, 1})
      {
        CAPTURE(parity);
        REQUIRE(bases.at(parity).Height() == 2);
        REQUIRE(bases.at(parity).Width() == 4);
        for(int i = 0; i < 2; ++i)
          for(int k = 0; k < 4; ++k)
            {
              CAPTURE(i);
              CAPTURE(k);
              const auto &x = points.at(k);
              const auto expected
                = El::Sqrt(scalings.at(k) * domain.weight(parity, degree, x))
                  * basis.at(i)(x);
              DIFF(bases.at(parity).CRef(i, k), expected);
            }
      }
  }

  SECTION("Polynomial_Vector_Matrix on an interval")
  {
    const auto polys = polynomials_1d();
    const auto domain = interval(El::BigFloat(1) / 2, 2);
    const auto make = [&](const std::optional<Damped_Rational> &prefactor,
                          const std::optional<Damped_Rational> &reduced,
                          const std::optional<std::vector<El::BigFloat>>
                            &sample_points,
                          const std::optional<Positivity_Domain> &d) {
      return Polynomial_Vector_Matrix(polys, prefactor, reduced, std::nullopt,
                                      sample_points, std::nullopt,
                                      std::nullopt, std::nullopt, d);
    };

    {
      INFO("Defaults: prefactor 1, Chebyshev points, sampled bases");
      const auto pvm
        = make(std::nullopt, std::nullopt, std::nullopt, domain);
      REQUIRE(pvm.domain.is_interval());
      REQUIRE(pvm.prefactor.is_constant());
      REQUIRE(pvm.prefactor.constant == 1);
      REQUIRE(pvm.has_default_interval_basis());
      REQUIRE(pvm.sample_points.size() == 5);
      DIFF(pvm.sample_points,
           to_BigFloat_Vector(interval_sample_points(
             5, to_Boost_Float(domain.a()), to_Boost_Float(domain.b()))));
      for(const auto &s : pvm.sample_scalings)
        REQUIRE(s == El::BigFloat(1));
    }
    {
      INFO("Without a domain, the half-line defaults are unchanged");
      const auto pvm
        = make(std::nullopt, std::nullopt, std::nullopt, std::nullopt);
      REQUIRE(!pvm.domain.is_interval());
      REQUIRE(!pvm.has_default_interval_basis());
      REQUIRE(!pvm.prefactor.is_constant());
    }
    {
      INFO("A prefactor positive on [1/2, 2] is accepted");
      const Damped_Rational prefactor{1, 3, {-1, 3, 4}};
      const auto pvm = make(prefactor, std::nullopt, std::nullopt, domain);
      REQUIRE(pvm.sample_scalings.size() == 5);
      for(size_t k = 0; k < 5; ++k)
        {
          const auto x = to_Boost_Float(pvm.sample_points.at(k));
          DIFF(pvm.sample_scalings.at(k), to_BigFloat(prefactor.evaluate(x)));
        }
    }

    INFO("Invalid inputs are refused");
    // pole inside [1/2, 2]
    REQUIRE_THROWS(
      make(Damped_Rational{1, 1, {1}}, std::nullopt, std::nullopt, domain));
    // odd number of poles above b: negative on the domain
    REQUIRE_THROWS(
      make(Damped_Rational{1, 1, {3}}, std::nullopt, std::nullopt, domain));
    // negative constant
    REQUIRE_THROWS(
      make(Damped_Rational{-1, 1, {}}, std::nullopt, std::nullopt, domain));
    // reduced prefactor
    REQUIRE_THROWS(make(Damped_Rational{1, 1, {-1}},
                        Damped_Rational{1, 1, {}}, std::nullopt, domain));
    // sample point outside [1/2, 2]
    REQUIRE_THROWS(make(std::nullopt, std::nullopt,
                        std::vector<El::BigFloat>{0, 1, 1.5, 1.75, 2},
                        domain));
  }
}
