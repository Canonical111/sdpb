// Given a vector of polynomials {q_0(x), q_1(x), ..., q_n(x)}
// of degree deg q_m(x) = m,
// a list of points {x_0 .. x_K}
// and scaling factors {s_0 .. s_K}, form the Matrix
//
//   {{ \sqrt(s_0) q_0(x_0), ..., \sqrt(s_K) q_0(x_K) },
//    { \sqrt(s_0) q_1(x_0), ..., \sqrt(s_K) q_1(x_K) },
//    ...
//    { \sqrt(s_0) q_n(x_0), ..., \sqrt(s_K) q_n(x_K) }}
//
// Input:
// - bilinearBasis: the vector {q_0(x), q_1(x), ..., q_n(x)}
// - samplePoints: the points {x_0, x_1, ... x_K}
// - sampleScalings: the scale factors {s_0, s_1, ... s_K}

#include "pmp/Polynomial.hxx"
#include "pmp/Positivity_Domain.hxx"
#include "sdpb_util/assert.hxx"

El::Matrix<El::BigFloat>
sample_bilinear_basis(const Polynomial_Vector &bilinearBasis,
                      const std::vector<El::BigFloat> &samplePoints,
                      const std::vector<El::BigFloat> &sampleScalings)
{
  El::Matrix<El::BigFloat> b(bilinearBasis.size(), samplePoints.size());
  for(int k = 0; k < samplePoints.size(); k++)
    {
      const auto &x = samplePoints.at(k);
      const auto scale = Sqrt(sampleScalings.at(k));
      for(int i = 0; i < bilinearBasis.size(); i++)
        {
          b.Set(i, k, scale * bilinearBasis.at(i)(x));
        }
    }
  return b;
}

std::array<El::Matrix<El::BigFloat>, 2>
sample_bilinear_basis(const std::array<Polynomial_Vector, 2> &bilinear_basis,
                      const std::vector<El::BigFloat> &sample_points,
                      const std::vector<El::BigFloat> &sample_scalings)
{
  // The matrix Y has two blocks Y_1, Y_2.  The bilinear_bases for the
  // constraint matrices A_p are given by sampling the following
  // vectors for each block:
  //
  //   Y_1: {q_0(x), ..., q_delta1(x)}
  //   Y_2: {\sqrt(x) q_0(x), ..., \sqrt(x) q_delta2(x)

  std::array<El::Matrix<El::BigFloat>, 2> bilinear_bases;

  bilinear_bases[0]
    = sample_bilinear_basis(bilinear_basis[0], sample_points, sample_scalings);

  // The \sqrt(x) factors can be accounted for by replacing the
  // scale factors s_k with x_k s_k.
  std::vector<El::BigFloat> scaled_samples;
  for(size_t ii = 0; ii < sample_points.size(); ++ii)
    {
      scaled_samples.emplace_back(sample_points[ii] * sample_scalings[ii]);
    }
  bilinear_bases[1]
    = sample_bilinear_basis(bilinear_basis[1], sample_points, scaled_samples);
  return bilinear_bases;
}

namespace
{
  // Rows i = 0..size-1 of the result are sqrt(s_k) p_i(x_k), k = 0..K, where
  // p_0, p_1, ... (deg p_i = i) are orthonormal for the discrete inner product
  //
  //   <p, q> = sum_k s_k p(x_k) q(x_k).
  //
  // They are built from the Chebyshev polynomials T_i(u), u = (2x-a-b)/(b-a),
  // by modified Gram-Schmidt with one reorthogonalization. No monomial
  // coefficients are formed: the values at x_k stay well conditioned for any
  // degree, unlike Horner evaluation of orthonormal polynomials in x.
  El::Matrix<El::BigFloat>
  sample_orthonormal_basis(const size_t size,
                           const std::vector<El::BigFloat> &sample_points,
                           const std::vector<El::BigFloat> &sample_scalings,
                           const El::BigFloat &a, const El::BigFloat &b)
  {
    const size_t num_points = sample_points.size();
    El::Matrix<El::BigFloat> v(size, num_points);
    if(size == 0)
      return v;

    for(size_t k = 0; k < num_points; ++k)
      {
        const auto sqrt_scale = El::Sqrt(sample_scalings.at(k));
        const El::BigFloat u = (2 * sample_points.at(k) - a - b) / (b - a);
        El::BigFloat t_prev = 1, t = u;
        v.Set(0, k, sqrt_scale);
        for(size_t i = 1; i < size; ++i)
          {
            v.Set(i, k, sqrt_scale * t);
            const El::BigFloat t_next = 2 * u * t - t_prev;
            t_prev = t;
            t = t_next;
          }
      }

    const auto dot = [&v, num_points](const size_t i, const size_t j) {
      El::BigFloat result = 0;
      for(size_t k = 0; k < num_points; ++k)
        result += v.CRef(i, k) * v.CRef(j, k);
      return result;
    };
    for(size_t i = 0; i < size; ++i)
      {
        for(int pass = 0; pass < 2; ++pass)
          for(size_t j = 0; j < i; ++j)
            {
              const El::BigFloat r = dot(i, j);
              for(size_t k = 0; k < num_points; ++k)
                v.Set(i, k, v.CRef(i, k) - r * v.CRef(j, k));
            }
        const El::BigFloat norm = El::Sqrt(dot(i, i));
        ASSERT(norm > 0,
               "bilinear bases: sampled basis is degenerate; too few sample "
               "points with positive weight",
               DEBUG_STRING(i), DEBUG_STRING(size), DEBUG_STRING(num_points));
        for(size_t k = 0; k < num_points; ++k)
          v.Set(i, k, v.CRef(i, k) / norm);
      }
    return v;
  }
}

// Sampled bilinear bases for a domain (see Positivity_Domain.hxx).
// On [0, inf) this is sample_bilinear_basis() above.
// On [a, b], the Gram matrix Y_{parity+1} carries the weight w_parity(x):
// the scalings of its rows are w_parity(x_k) s_k.
// If bilinear_basis is empty (no user-supplied bases on [a, b]), the rows
// are orthonormal at the sample points, see sample_orthonormal_basis().
std::array<El::Matrix<El::BigFloat>, 2>
sample_bilinear_basis(const std::array<Polynomial_Vector, 2> &bilinear_basis,
                      const std::vector<El::BigFloat> &sample_points,
                      const std::vector<El::BigFloat> &sample_scalings,
                      const Positivity_Domain &domain)
{
  if(!domain.is_interval())
    return sample_bilinear_basis(bilinear_basis, sample_points,
                                 sample_scalings);

  ASSERT(!sample_points.empty());
  ASSERT_EQUAL(sample_points.size(), sample_scalings.size());
  const size_t degree = sample_points.size() - 1;
  const std::array<size_t, 2> basis_sizes{degree / 2 + 1, (degree + 1) / 2};
  const bool default_basis = bilinear_basis[0].empty();

  std::array<El::Matrix<El::BigFloat>, 2> bilinear_bases;
  for(const size_t parity : {0, 1})
    {
      std::vector<El::BigFloat> weighted_scalings;
      weighted_scalings.reserve(sample_points.size());
      for(size_t k = 0; k < sample_points.size(); ++k)
        {
          const auto &x = sample_points.at(k);
          const auto weight = domain.weight(parity, degree, x);
          ASSERT(weight >= 0, "sample point outside the domain [a, b]",
                 DEBUG_STRING(x), DEBUG_STRING(domain.a()),
                 DEBUG_STRING(domain.b()));
          weighted_scalings.push_back(weight * sample_scalings.at(k));
        }
      bilinear_bases[parity]
        = default_basis
            ? sample_orthonormal_basis(basis_sizes[parity], sample_points,
                                       weighted_scalings, domain.a(),
                                       domain.b())
            : sample_bilinear_basis(bilinear_basis[parity], sample_points,
                                    weighted_scalings);
    }
  return bilinear_bases;
}
