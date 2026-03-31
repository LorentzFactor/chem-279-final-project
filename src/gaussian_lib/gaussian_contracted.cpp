#include "gaussian_lib.h"

namespace gaussian_lib {
    double GaussianContracted::operator()(std::array<double, 3> x) const {
        double f_x = 0;
        for(size_t i=0; i < weights_.size(); i++) {
            f_x += weights_.at(i) * components_.at(i)(x);
        }
        return f_x;
    }

    arma::cube GaussianContracted::operator()(
        const arma::vec& x_points,
        const arma::vec& y_points,
        const arma::vec& z_points
    ) const {
        arma::cube evaluations = arma::cube(x_points.size(), y_points.size(), z_points.size());
        for(size_t i=0; i < weights_.size(); i++) {
            evaluations += weights_.at(i) * components_.at(i)(x_points, y_points, z_points);
        }
        return evaluations;
    }

    double integrate_product(const GaussianContracted& ga, const GaussianContracted& gb) {
        double integral = 0;
        for (size_t i = 0; i < ga.size(); i++) {
            for(size_t j = 0; j < gb.size(); j++) {
                double d_ai = ga.get_weights()[i];
                double d_bj = gb.get_weights()[j];
                integral += d_ai*d_bj*integrate_product(ga.get_components()[i], gb.get_components()[j]);
            }
        }
        return integral;
    }

    double calculate_gamma_base_term(
        double sigma_A, double sigma_B,
        const std::array<double,3>& RA, const std::array<double,3>& RB
    ) {
        arma::vec3 R_A = arma::vec3(RA.data());
        arma::vec3 R_B = arma::vec3(RB.data());
        double UA = std::pow(M_PI*sigma_A, 1.5);
        double UB = std::pow(M_PI*sigma_B, 1.5);
        double V2 = 1/(sigma_A + sigma_B);
        double dist = arma::norm(R_A - R_B);
        if (dist <= std::numeric_limits<double>::epsilon()) {
            return UA * UB * 2 * std::sqrt(V2/M_PI);
        }
        else {
            return (UA * UB * std::erf(sqrt(V2) * dist))/dist;
        }
    };

    double calculate_gamma(const GaussianContracted& ga, const GaussianContracted& gb) {
        double sum = 0;
        for (size_t i = 0; i < ga.size(); i++) {
            double da_i = ga.get_weights().at(i) / ga.get_components().at(i).norm;
            for(size_t j = 0; j < ga.size(); j++) {
                double da_j = ga.get_weights().at(j) / ga.get_components().at(j).norm;
                double sigma_A = 1/(ga.get_components().at(i).exponent + ga.get_components().at(j).exponent);
                for (size_t k = 0; k < gb.size(); k++) {
                    double db_k = gb.get_weights().at(k) / gb.get_components().at(k).norm;
                    for(size_t l = 0; l < gb.size(); l++) {
                        double db_l = gb.get_weights().at(l) / gb.get_components().at(l).norm;
                        double sigma_B = 1/(gb.get_components().at(k).exponent + gb.get_components().at(l).exponent);

                        double base_term = calculate_gamma_base_term(sigma_A, sigma_B, ga.center, gb.center);
                        sum += da_i * da_j * db_k * db_l * base_term;
                    }
                }
            }
        }
        return sum * 27.211324570273;
    }

    GaussianContracted::GaussianContracted(
        const std::array<double,3>& center,
        const std::vector<double>& alphas,
        const std::vector<double>& weights,
        const std::array<char,3>& momentum
    )
        : center(center), weights_(weights), momentum(momentum)
    {
        shell = std::accumulate(momentum.begin(), momentum.end(), 0);
        components_.reserve(alphas.size());
        for(size_t i=0; i < alphas.size(); i++) {
            components_.push_back(NormedGaussianPrimitive3d(center, alphas.at(i), momentum));
        }
    }
}
