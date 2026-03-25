#include "gaussian_lib.h"

namespace gaussian_lib {
    double NormedGaussianPrimitive3d::operator()(std::array<double, 3> x) const {
        double result = 1.0;
        for(size_t i=0; i<x.size(); i++) {
            result *= components_[i](x[i]);
        }
        return result / norm;
    }

    double integrate_product(const NormedGaussianPrimitive3d& ga, const NormedGaussianPrimitive3d& gb) {
        double integral = 1;
        for(size_t i=0; i < 3; i++) {
            integral *= integrate_product(ga.component(i), gb.component(i));
        }
        integral /= ga.norm;
        integral /= gb.norm;
        return integral;
    }

    NormedGaussianPrimitive3d::NormedGaussianPrimitive3d(const std::array<double,3> center, double exponent, const std::array<char,3> momentum)
        : center(center), exponent(exponent), momentum(momentum)
    {
        norm = 1;
        for(size_t i = 0; i < center.size(); i++) {
            components_[i] = GaussianPrimitive{center[i], exponent, momentum[i]};
            norm *= integrate_product(components_[i], components_[i]);
        }
        norm = std::sqrt(norm);
    }
}
