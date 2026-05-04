#pragma once

#include "integration_tools/integration_tools.h"
#include <armadillo>
#include <array>
#include <cmath>
#include <ostream>
#include <stdexcept>
#include <iomanip>
#include <limits>
#include <math.h>
#include <numeric>
#include <vector>

namespace gaussian_lib {

struct GaussianPrimitive {
  double center;
  double exponent;
  char momentum;

  double operator()(double x) const;
};

struct NormedGaussianPrimitive3d {
private:
  std::array<GaussianPrimitive, 3> components_;

public:
  std::array<double, 3> center;
  double exponent;
  std::array<char, 3> momentum;
  double norm;

  NormedGaussianPrimitive3d(const std::array<double, 3> center, double exponent,
                            const std::array<char, 3> momentum);

  double operator()(std::array<double, 3> x) const;
  const GaussianPrimitive &component(int i) const { return components_[i]; };

  arma::cube operator()(const arma::vec &x_points, const arma::vec &y_points,
                        const arma::vec &z_points) const;
};

struct GaussianContracted {
private:
  std::vector<NormedGaussianPrimitive3d> components_;
  std::vector<double> weights_;

public:
  std::array<double, 3> center;
  std::array<char, 3> momentum;
  char shell;

  GaussianContracted(const std::array<double, 3> &center,
                     const std::vector<double> &alphas,
                     const std::vector<double> &weights,
                     const std::array<char, 3> &momentum);

  const std::vector<double> &get_weights() const { return weights_; };

  const std::vector<NormedGaussianPrimitive3d> &get_components() const {
    return components_;
  };
  size_t size() const { return components_.size(); };

  double operator()(std::array<double, 3> x) const;

  arma::cube operator()(const arma::vec &x_points, const arma::vec &y_points,
                        const arma::vec &z_points) const;

  void recenter(std::array<double, 3> new_position);
};

/** One term in a derivative or (r−origin) factorization; used for L integrals. */
struct DerivativeGaussian {
  GaussianContracted gaussian;
  double prefactor;
};

std::vector<DerivativeGaussian>
get_gaussian_derivative(const GaussianContracted &g, int direction);

std::vector<DerivativeGaussian>
multiply_by_coords(const GaussianContracted &g, int direction, double origin);

struct GaussianTemplate {
  std::vector<double> alphas;
  std::vector<double> weights;

  GaussianTemplate(std::vector<double> alphas, std::vector<double> weights)
      : alphas(alphas), weights(weights) {
    if (alphas.size() != weights.size()) {
      throw std::runtime_error(
          "Weights and coefficients must be same length.");
    }
  };

  GaussianContracted toFunction(std::array<double, 3> center,
                                std::array<char, 3> momentum) const {
    return GaussianContracted(center, this->alphas, this->weights, momentum);
  };
};

std::ostream &operator<<(std::ostream &os, const GaussianPrimitive &g);

double numerically_integrate_product(const GaussianPrimitive &ga,
                                     const GaussianPrimitive &gb, double tol);
double integrate_product(const GaussianPrimitive &ga,
                         const GaussianPrimitive &gb);
double integrate_product_dxa(const GaussianPrimitive &ga,
                             const GaussianPrimitive &gb);

double integrate_product(const NormedGaussianPrimitive3d &ga,
                         const NormedGaussianPrimitive3d &gb);
std::array<double, 3>
integrate_product_dRa(const NormedGaussianPrimitive3d &ga,
                      const NormedGaussianPrimitive3d &gb);

double integrate_product(const GaussianContracted &ga,
                         const GaussianContracted &gb);
std::array<double, 3> integrate_product_dRa(const GaussianContracted &ga,
                                            const GaussianContracted &gb);

double calculate_gamma_base_term(double sigma_A, double sigma_B,
                                 const std::array<double, 3> &RA,
                                 const std::array<double, 3> &RB);
arma::vec3 calculate_gamma_base_term_dRa(double sigma_A, double sigma_B,
                                         const std::array<double, 3> &RA,
                                         const std::array<double, 3> &RB);
double calculate_gamma(const GaussianContracted &ga,
                       const GaussianContracted &gb);
arma::vec3 calculate_gamma_dRa(const GaussianContracted &ga,
                               const GaussianContracted &gb);

} // namespace gaussian_lib
