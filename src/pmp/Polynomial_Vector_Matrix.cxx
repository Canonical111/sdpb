#include "Polynomial_Vector_Matrix.hxx"

#include "sdpb_util/Boost_Float.hxx"
#include "sdpb_util/assert.hxx"

#include <boost/math/constants/constants.hpp>
#include <algorithm>
#include <iterator>

std::vector<Boost_Float>
sample_points(const size_t &num_points, const Damped_Rational &prefactor);

std::vector<Boost_Float>
interval_sample_points(const size_t &num_points, const Boost_Float &a,
                       const Boost_Float &b);

std::vector<Boost_Float>
sample_scalings(const std::vector<Boost_Float> &points,
                const Damped_Rational &damped_rational);

std::array<Polynomial_Vector, 2>
bilinear_basis(const std::vector<El::BigFloat> &sample_points,
               const std::vector<El::BigFloat> &sample_scalings);

namespace
{
  int64_t get_max_degree(const Simple_Matrix<Polynomial_Vector> &pvm)
  {
    int64_t max_degree = 0;
    for(int i = 0; i < pvm.Height(); ++i)
      for(int j = 0; j < pvm.Width(); ++j)
        for(auto &polynomial : pvm(i, j))
          max_degree = std::max(max_degree, polynomial.degree());
    return max_degree;
  }

  // We use XXX_or_default() functions instead of std::optional::value_or()
  // to prevent unnecessary computation of default value.

  Damped_Rational
  prefactor_or_default(const std::optional<Damped_Rational> &prefactor_opt,
                       const int64_t max_degree)
  {
    if(prefactor_opt.has_value())
      return prefactor_opt.value();

    // For constants, we can use constant prefactor.
    // Otherwise, we need prefactor.base < 1 to ensure integrability
    if(max_degree == 0)
      return {1, 1, {}};

    // NB: we cannot initialize exp_minus_one as a global constant,
    // because we don't know required MPFR precision at that moment.
    // Thus, we do it inside the function.
    const Boost_Float exp_minus_one =
#if (BOOST_VERSION >= 106900)
      boost::math::constants::exp_minus_one<Boost_Float>();
#else
      Boost_Float(1) / boost::math::constants::e<Boost_Float>();
#endif

    // Default prefactor is e^-x
    const Damped_Rational exp_minus_x = {1, exp_minus_one, {}};
    return exp_minus_x;
  }

  // On an interval, the default prefactor is 1:
  // the polynomials need no damping on a compact domain
  // (and the default sample scalings are equilibrated, see above).
  Damped_Rational interval_prefactor_or_default(
    const std::optional<Damped_Rational> &prefactor_opt,
    const Positivity_Domain &domain)
  {
    if(!prefactor_opt.has_value())
      return {1, 1, {}};

    // Positivity on [a,b] is imposed on the polynomials,
    // so the prefactor must be positive on [a,b].
    const auto &prefactor = prefactor_opt.value();
    const Boost_Float a = to_Boost_Float(domain.a());
    const Boost_Float b = to_Boost_Float(domain.b());
    ASSERT(prefactor.constant > 0 && prefactor.base > 0,
           "prefactor must be positive on the domain [a, b]",
           DEBUG_STRING(prefactor));
    size_t num_poles_above = 0;
    for(const auto &pole : prefactor.poles)
      {
        ASSERT(pole < a || pole > b,
               "prefactor has a pole inside the domain [a, b]",
               DEBUG_STRING(pole), DEBUG_STRING(a), DEBUG_STRING(b));
        if(pole > b)
          ++num_poles_above;
      }
    ASSERT(num_poles_above % 2 == 0,
           "prefactor must be positive on the domain [a, b]: an odd number "
           "of poles above b makes it negative",
           DEBUG_STRING(prefactor), DEBUG_STRING(a), DEBUG_STRING(b));
    return prefactor;
  }

  std::vector<El::BigFloat> sample_points_or_default(
    const std::optional<std::vector<El::BigFloat>> &sample_points_opt,
    const Damped_Rational &prefactor, const Positivity_Domain &domain,
    const size_t num_points)
  {
    if(sample_points_opt.has_value())
      return sample_points_opt.value();

    if(domain.is_interval())
      return to_BigFloat_Vector(
        interval_sample_points(num_points, to_Boost_Float(domain.a()),
                               to_Boost_Float(domain.b())));

    return to_BigFloat_Vector(sample_points(num_points, prefactor));
  }

  std::vector<El::BigFloat> sample_scalings_or_default(
    const std::optional<std::vector<El::BigFloat>> &sample_scalings_opt,
    const std::vector<El::BigFloat> &sample_points,
    const Damped_Rational &damped_rational)
  {
    if(sample_scalings_opt.has_value())
      return sample_scalings_opt.value();
    return to_BigFloat_Vector(
      sample_scalings(to_Boost_Float_Vector(sample_points), damped_rational));
  }

  // Default sample scalings on an interval without prefactor and
  // sampleScalings: s_k = 1 / max |P^{rs}_n(x_k)| over the entries and
  // components of the block, so that every constraint row (c, B) at x_k has
  // largest entry 1. Then the dual residuals that SDPB compares with its
  // thresholds, its initial point Y = Omega I (in bilinear bases orthonormal
  // at the points with these weights) and its rounding errors are on the same
  // scale at every point of [a, b], whatever the size of the polynomials
  // there. A row smaller than sqrt(epsilon) times the largest one (a common
  // zero of the block) is scaled as if it were that size.
  std::vector<El::BigFloat>
  equilibrated_sample_scalings(const Simple_Matrix<Polynomial_Vector> &pvm,
                               const std::vector<El::BigFloat> &sample_points,
                               const Positivity_Domain &domain)
  {
    ASSERT(!sample_points.empty());
    std::vector<El::BigFloat> row_size(sample_points.size(), 0);
    for(size_t k = 0; k < sample_points.size(); ++k)
      for(int i = 0; i < pvm.Height(); ++i)
        for(int j = 0; j <= i; ++j)
          for(const auto &polynomial : pvm(i, j))
            row_size[k] = std::max(
              row_size[k],
              El::Abs(domain.evaluate(polynomial, sample_points[k])));

    const El::BigFloat largest
      = *std::max_element(row_size.begin(), row_size.end());
    std::vector<El::BigFloat> scalings(sample_points.size(), 1);
    if(largest == El::BigFloat(0))
      return scalings;
    const El::BigFloat floor
      = largest * El::Sqrt(El::limits::Epsilon<El::BigFloat>());
    for(size_t k = 0; k < sample_points.size(); ++k)
      scalings[k] = 1 / std::max(row_size[k], floor);
    return scalings;
  }

  // NB: result is truncated to (delta1+1, delta2+1)
  std::array<Polynomial_Vector, 2> bilinear_basis_or_default(
    const std::optional<std::array<Polynomial_Vector, 2>> &bilinear_basis_opt,
    const std::vector<El::BigFloat> &sample_points,
    const std::vector<El::BigFloat> &sample_scalings)
  {
    if(!bilinear_basis_opt.has_value())
      {
        return bilinear_basis(sample_points, sample_scalings);
      }

    std::array<Polynomial_Vector, 2> basis;
    ASSERT(!sample_points.empty());
    const size_t degree = sample_points.size() - 1;

    for(const size_t parity : {0, 1})
      {
        // basis_size = delta + 1
        const size_t basis_size
          = parity == 0 ? degree / 2 + 1 : (degree + 1) / 2;
        const auto &input_basis = bilinear_basis_opt.value()[parity];

        // Check input size.
        if(input_basis.size() < basis_size)
          {
            RUNTIME_ERROR("PMP: bilinearBasis_", parity,
                          " size=", input_basis.size(), ", required at least ",
                          basis_size);
          }
        if(input_basis.size() > basis_size)
          {
            PRINT_WARNING("PMP: bilinearBasis_", parity,
                          " size=", input_basis.size(),
                          " is too large, only the first ", basis_size,
                          " polynomials will be used");
          }
        basis[parity] = Polynomial_Vector(input_basis.begin(),
                                          input_basis.begin() + basis_size);
      }
    return basis;
  }
}

Polynomial_Vector_Matrix::Polynomial_Vector_Matrix(
  const Simple_Matrix<Polynomial_Vector> &polynomials,
  const std::optional<Damped_Rational> &prefactor_opt,
  const std::optional<Damped_Rational> &reduced_prefactor_opt,
  const std::optional<int64_t> &max_num_poles_opt,
  const std::optional<std::vector<El::BigFloat>> &sample_points_opt,
  const std::optional<std::vector<El::BigFloat>> &sample_scalings_opt,
  const std::optional<std::vector<El::BigFloat>> &reduced_sample_scalings_opt,
  const std::optional<std::array<Polynomial_Vector, 2>> &bilinear_basis_opt,
  const std::optional<Positivity_Domain> &domain_opt)
    : polynomials(polynomials), domain(domain_opt.value_or(Positivity_Domain{}))
{
  domain.validate();

  const int64_t max_degree = get_max_degree(polynomials);

  prefactor = domain.is_interval()
                ? interval_prefactor_or_default(prefactor_opt, domain)
                : prefactor_or_default(prefactor_opt, max_degree);

  // Set and validate reduced_prefactor
  {
    if(reduced_prefactor_opt.has_value())
      PRINT_WARNING(prefactor_opt.has_value(),
                    "reducedPrefactor is specified, but prefactor is not!");
    reduced_prefactor = reduced_prefactor_opt.value_or(prefactor);

    // Leave the rightmost max_num_poles in reduced_prefactor
    const int64_t max_num_poles
      = max_num_poles_opt.has_value() && max_num_poles_opt.value() >= 0
          ? max_num_poles_opt.value()
          : reduced_prefactor.poles.size();
    if(max_num_poles < reduced_prefactor.poles.size())
      {
        auto poles = reduced_prefactor.poles;
        std::sort(poles.begin(), poles.end());
        reduced_prefactor.poles
          = {std::prev(poles.end(), max_num_poles), poles.end()};
      }

    if(domain.is_interval()
       && (reduced_prefactor_opt.has_value()
           || reduced_prefactor.poles.size() != prefactor.poles.size()))
      {
        RUNTIME_ERROR("reducedPrefactor, or maxNumPoles below the number of "
                      "prefactor poles, is not supported for an interval "
                      "domain",
                      DEBUG_STRING(prefactor.poles.size()),
                      DEBUG_STRING(reduced_prefactor.poles.size()));
      }

    if(reduced_prefactor.poles.size() > prefactor.poles.size())
      {
        PRINT_WARNING("reducedPrefactor has more poles than prefactor, the "
                      "number of sample points will be increased!\n\t",
                      DEBUG_STRING(prefactor.poles.size()),
                      DEBUG_STRING(reduced_prefactor.poles.size()));
      }
  }

  const int64_t num_points
    = max_degree + 1 + reduced_prefactor.poles.size() - prefactor.poles.size();
  ASSERT(num_points > 0, DEBUG_STRING(num_points), DEBUG_STRING(max_degree),
         DEBUG_STRING(prefactor), DEBUG_STRING(reduced_prefactor));

  sample_points = sample_points_or_default(
    sample_points_opt, reduced_prefactor, domain, num_points);
  // Checked before the scalings and bases are computed from the points
  ASSERT_EQUAL(sample_points.size(), num_points);

  sample_scalings
    = domain.is_interval() && !prefactor_opt.has_value()
          && !sample_scalings_opt.has_value()
        ? equilibrated_sample_scalings(polynomials, sample_points, domain)
        : sample_scalings_or_default(sample_scalings_opt, this->sample_points,
                                     prefactor);

  reduced_sample_scalings = [&] {
    if(reduced_sample_scalings_opt.has_value()
       || reduced_prefactor_opt.has_value()
       || reduced_prefactor.poles.size() != prefactor.poles.size())
      {
        return sample_scalings_or_default(
          reduced_sample_scalings_opt, this->sample_points, reduced_prefactor);
      }
    return sample_scalings;
  }();

  // s_k P(x_k) = s'_k (sum of squares)(x_k) at the D+1 points is the
  // polynomial identity P = sum of squares only if s' = s. On the half-line
  // s' != s is what reducedPrefactor is for, which intervals don't support.
  if(domain.is_interval() && reduced_sample_scalings != sample_scalings)
    RUNTIME_ERROR("reducedSampleScalings different from sampleScalings is "
                  "not supported for an interval domain");

  // On an interval, the default bases are built by pmp2sdp
  // at the sample points (see has_default_interval_basis()).
  if(!domain.is_interval() || bilinear_basis_opt.has_value())
    bilinear_basis = bilinear_basis_or_default(
      bilinear_basis_opt, sample_points, reduced_sample_scalings);

  validate(num_points);
}

void Polynomial_Vector_Matrix::validate(const int64_t num_points) const
{
  const int64_t degree = num_points - 1;
  ASSERT_EQUAL(sample_points.size(), num_points);
  ASSERT_EQUAL(reduced_sample_scalings.size(), sample_points.size());

  if(domain.is_interval())
    {
      for(const auto &x : sample_points)
        ASSERT(domain.contains(x),
               "sample points must lie in the domain [a, b]", DEBUG_STRING(x),
               DEBUG_STRING(domain.a()), DEBUG_STRING(domain.b()));
      // D+1 equations determine a degree-D polynomial only at D+1
      // distinct points.
      auto sorted_points = sample_points;
      std::sort(sorted_points.begin(), sorted_points.end());
      for(size_t k = 1; k < sorted_points.size(); ++k)
        ASSERT(sorted_points[k] != sorted_points[k - 1],
               "sample points must be distinct",
               DEBUG_STRING(sorted_points[k]));
      // The sampled bilinear bases carry sqrt(s_k) (sample_bilinear_basis).
      for(const auto &s : sample_scalings)
        ASSERT(s > 0, "sample scalings must be positive", DEBUG_STRING(s));
    }

  if(has_default_interval_basis())
    {
      ASSERT(bilinear_basis[1].empty());
    }
  else
    {
      const size_t delta1 = degree / 2;
      ASSERT_EQUAL(bilinear_basis[0].size(), delta1 + 1, DEBUG_STRING(degree));
      if(degree == 0)
        {
          ASSERT_EQUAL(bilinear_basis[1].size(), 0);
        }
      else
        {
          const size_t delta2 = (degree + 1) / 2 - 1;
          ASSERT_EQUAL(bilinear_basis[1].size(), delta2 + 1,
                       DEBUG_STRING(degree));
        }
    }

  ASSERT_EQUAL(polynomials.Height(), polynomials.Width());

  // TODO check if this is fast enough
  for(int i = 0; i < polynomials.Height(); ++i)
    for(int j = 0; j < polynomials.Width(); ++j)
      {
        if(i == j)
          continue;
        ASSERT_EQUAL(polynomials(i, j), polynomials(j, i), DEBUG_STRING(i),
                     DEBUG_STRING(j));
      }
}
