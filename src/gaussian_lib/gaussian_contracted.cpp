#include "gaussian_lib.h"

namespace gaussian_lib {
double GaussianContracted::operator()(std::array<double, 3> x) const {
  double f_x = 0;
  for (size_t i = 0; i < weights_.size(); i++) {
    f_x += weights_.at(i) * components_.at(i)(x);
  }
  return f_x;
}

arma::cube GaussianContracted::operator()(const arma::vec &x_points,
                                          const arma::vec &y_points,
                                          const arma::vec &z_points) const {
  arma::cube evaluations =
      arma::cube(x_points.size(), y_points.size(), z_points.size());
  for (size_t i = 0; i < weights_.size(); i++) {
    evaluations +=
        weights_.at(i) * components_.at(i)(x_points, y_points, z_points);
  }
  return evaluations;
}

double integrate_product(const GaussianContracted &ga,
                         const GaussianContracted &gb) {
  double integral = 0;
  for (size_t i = 0; i < ga.size(); i++) {
    for (size_t j = 0; j < gb.size(); j++) {
      double d_ai = ga.get_weights()[i];
      double d_bj = gb.get_weights()[j];
      integral +=
          d_ai * d_bj *
          integrate_product(ga.get_components()[i], gb.get_components()[j]);
    }
  }
  return integral;
}

std::array<double, 3> integrate_product_dRa(const GaussianContracted &ga,
                                            const GaussianContracted &gb) {
  std::array<double, 3> gradient{0, 0, 0};
  for (size_t i = 0; i < ga.size(); i++) {
    for (size_t j = 0; j < gb.size(); j++) {
      double d_ai = ga.get_weights()[i];
      double d_bj = gb.get_weights()[j];
      std::array<double, 3> gradient_term =
          integrate_product_dRa(ga.get_components()[i], gb.get_components()[j]);
      for (size_t idim = 0; idim < 3; ++idim) {
        gradient[idim] += d_ai * d_bj * gradient_term[idim];
      }
    }
  }
  return gradient;
}

double calculate_gamma_base_term(double sigma_A, double sigma_B,
                                 const std::array<double, 3> &RA,
                                 const std::array<double, 3> &RB) {
  arma::vec3 R_A = arma::vec3(RA.data());
  arma::vec3 R_B = arma::vec3(RB.data());
  double UA = std::pow(M_PI * sigma_A, 1.5);
  double UB = std::pow(M_PI * sigma_B, 1.5);
  double V2 = 1 / (sigma_A + sigma_B);
  double dist = arma::norm(R_A - R_B);
  if (dist <= std::numeric_limits<double>::epsilon()) {
    return UA * UB * 2 * std::sqrt(V2 / M_PI);
  } else {
    return (UA * UB * std::erf(sqrt(V2) * dist)) / dist;
  }
};

arma::vec3 calculate_gamma_base_term_dRa(double sigma_A, double sigma_B,
                                         const std::array<double, 3> &RA,
                                         const std::array<double, 3> &RB) {
  arma::vec3 R_A = arma::vec3(RA.data());
  arma::vec3 R_B = arma::vec3(RB.data());
  double UA = std::pow(M_PI * sigma_A, 1.5);
  double UB = std::pow(M_PI * sigma_B, 1.5);
  double V2 = 1 / (sigma_A + sigma_B);
  double dist = arma::norm(R_A - R_B);
  double T = V2 * dist * dist;

  if (dist <= std::numeric_limits<double>::epsilon()) {
    return arma::zeros(3);
  } else {
    return (UA * UB / (dist * dist)) *
           (-std::erf(std::sqrt(T)) / dist +
            (2 * std::sqrt(V2) / std::sqrt(M_PI) * std::exp(-T))) *
           (R_A - R_B);
  }
}

double calculate_gamma(const GaussianContracted &ga,
                       const GaussianContracted &gb) {
  double sum = 0;
  for (size_t i = 0; i < ga.size(); i++) {
    double da_i = ga.get_weights().at(i) / ga.get_components().at(i).norm;
    for (size_t j = 0; j < ga.size(); j++) {
      double da_j = ga.get_weights().at(j) / ga.get_components().at(j).norm;
      double sigma_A = 1 / (ga.get_components().at(i).exponent +
                            ga.get_components().at(j).exponent);
      for (size_t k = 0; k < gb.size(); k++) {
        double db_k = gb.get_weights().at(k) / gb.get_components().at(k).norm;
        for (size_t l = 0; l < gb.size(); l++) {
          double db_l = gb.get_weights().at(l) / gb.get_components().at(l).norm;
          double sigma_B = 1 / (gb.get_components().at(k).exponent +
                                gb.get_components().at(l).exponent);

          double base_term =
              calculate_gamma_base_term(sigma_A, sigma_B, ga.center, gb.center);
          sum += da_i * da_j * db_k * db_l * base_term;
        }
      }
    }
  }
  return sum * 27.211324570273;
}

arma::vec3 calculate_gamma_dRa(const GaussianContracted &ga,
                               const GaussianContracted &gb) {
  arma::vec3 gradient = arma::zeros(3);
  for (size_t i = 0; i < ga.size(); i++) {
    double da_i = ga.get_weights().at(i) / ga.get_components().at(i).norm;
    for (size_t j = 0; j < ga.size(); j++) {
      double da_j = ga.get_weights().at(j) / ga.get_components().at(j).norm;
      double sigma_A = 1 / (ga.get_components().at(i).exponent +
                            ga.get_components().at(j).exponent);
      for (size_t k = 0; k < gb.size(); k++) {
        double db_k = gb.get_weights().at(k) / gb.get_components().at(k).norm;
        for (size_t l = 0; l < gb.size(); l++) {
          double db_l = gb.get_weights().at(l) / gb.get_components().at(l).norm;
          double sigma_B = 1 / (gb.get_components().at(k).exponent +
                                gb.get_components().at(l).exponent);

          arma::vec3 base_term = calculate_gamma_base_term_dRa(
              sigma_A, sigma_B, ga.center, gb.center);
          gradient -= da_i * da_j * db_k * db_l * base_term;
        }
      }
    }
  }
  return gradient * 27.211324570273;
}

std::vector<DerivativeGaussian>
GaussianContracted::get_gaussian_derivative(int direction) const {
  // Info from this contracted gaussian to generated DerivGaussians
  std::vector<double> current_alphas;
  current_alphas.reserve(this->components_.size());
  for (const auto &prim : this->components_) {
    current_alphas.push_back(prim.exponent);
  }

  // Vector to store DerivativeGaussians
  std::vector<DerivativeGaussian> gauss_derivs;

  // example for deriv with respect to x
  // d/dx X(l_x) = l_x * X * (l_x - 1) - 2 * alpha * X * (l_x + 1)
  // First get term one where momentum is reduced by 1
  if (this->momentum[direction] > 0) {
    std::array<char, 3> new_momentum = this->momentum;
    new_momentum[direction] -= 1;
    GaussianContracted term1(this->center, current_alphas, this->weights_,
                             new_momentum);
    // prefactor for term 1 is the original l value
    double prefactor = static_cast<double>(this->momentum[direction]);
    gauss_derivs.push_back(DerivativeGaussian{term1, prefactor});
  }

  // Next get term two where momentum is inreased by 1
  // No need to check if momentum is greater than 1 in this case
  std::vector<double> new_weights;
  new_weights.reserve(this->components_.size());
  for (size_t iprim = 0; iprim < this->components_.size(); ++iprim) {
    double weight_factor = -2.0 * this->components_[iprim].exponent;
    double new_weight = weight_factor * this->weights_[iprim];
    new_weights.push_back(new_weight);
  }

  double prefactor = 1.0;
  std::array<char, 3> new_momentum2 = this->momentum;
  new_momentum2[direction] += 1;
  GaussianContracted term2(this->center, current_alphas, new_weights,
                           new_momentum2);
  gauss_derivs.push_back(DerivativeGaussian{term2, prefactor});

  return gauss_derivs;
}

std::vector<DerivativeGaussian>
GaussianContracted::multiply_by_coords(int direction, double origin) const {
  std::vector<DerivativeGaussian> components;
  // Extract current alphas first
  std::vector<double> current_alphas;
  current_alphas.reserve(this->components_.size());
  for (const auto &prim : this->components_) {
    current_alphas.push_back(prim.exponent);
  }

  // To get ang momentum op, multiply deriv by coord relative to gauge origin
  // Example for operator in Z direction
  // Lz = -i((x - origin_x) * d/dy - (y - origin_y) * d/dx)
  // (x - origin_x) = (x - A_x) + (A_x - origin_x)

  // Term 1: (x - A_x) term
  // Increase angular momentum in direction by 1
  // Prefactor is 1.0
  std::array<char, 3> new_momentum = this->momentum;
  new_momentum[direction] += 1;
  double prefactor1 = 1.0 GaussianContracted term1(
      this->center, current_alphas, this->weights_, new_momentum);
  components.push_back(DerivativeGaussian{term1, prefactor1});

  // Term 2: (A_x - origin_x) term
  // Same orbital but scaled by distance
  // Prefactor is the distance factor
  double dist_factor = this->center[direction] - origin;

  // Only save Deriv Gaussian if distance greater than epsilon
  if (std::abs(dist_factor > 1e-12)) {
    GaussianContracted term2(*this);
    components.push_back(DerivativeGaussian{term2, dist_factor});
  }

  return components;
}

GaussianContracted::GaussianContracted(const std::array<double, 3> &center,
                                       const std::vector<double> &alphas,
                                       const std::vector<double> &weights,
                                       const std::array<char, 3> &momentum)
    : center(center), weights_(weights), momentum(momentum) {
  shell = std::accumulate(momentum.begin(), momentum.end(), 0);
  components_.reserve(alphas.size());
  for (size_t i = 0; i < alphas.size(); i++) {
    components_.push_back(
        NormedGaussianPrimitive3d(center, alphas.at(i), momentum));
  }
}
} // namespace gaussian_lib
