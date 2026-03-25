#include "gaussian_lib.h"

namespace gaussian_lib {
    double GaussianContracted::operator()(std::array<double, 3> x) const {
        double f_x = 0;
        for(size_t i=0; i < weights_.size(); i++) {
            f_x += weights_.at(i) * components_.at(i)(x);
        }
        return f_x;
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

    GaussianContracted::GaussianContracted(
        const std::array<double,3>& center,
        const std::vector<double>& alphas,
        const std::vector<double>& weights,
        const std::array<char,3>& momentum
    )
        : center(center), weights_(weights), momentum(momentum)
    {
        components_.reserve(alphas.size());
        for(size_t i=0; i < alphas.size(); i++) {
            components_.push_back(NormedGaussianPrimitive3d(center, alphas.at(i), momentum));
        }
    }
}
