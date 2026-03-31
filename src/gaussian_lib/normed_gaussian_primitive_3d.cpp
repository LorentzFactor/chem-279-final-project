#include "gaussian_lib.h"

namespace gaussian_lib {
    double NormedGaussianPrimitive3d::operator()(std::array<double, 3> x) const {
        double result = 1.0;
        for(size_t i=0; i<x.size(); i++) {
            result *= components_[i](x[i]);
        }
        return result / norm;
    }

    arma::cube NormedGaussianPrimitive3d::operator()(
                const arma::vec& x_points,
                const arma::vec& y_points,
                const arma::vec& z_points
    ) const {
        // Initialize evaluated values to correct shape
        arma::cube evaluations = arma::cube(x_points.size(), y_points.size(), z_points.size(), arma::fill::none);

        // Individually evaluate components along each dimension
        arma::vec x_component = x_points.for_each(components_[0]);
        arma::vec y_component = y_points.for_each(components_[1]);
        // Note that we apply the norm to the last dimension to reduce number of division operations
        arma::vec z_component = z_points.for_each(components_[2]) / norm;

        // slice along z dimension, take tensor product of other two for parallelism
        for(size_t iz=0; iz<z_points.size(); ++iz) {
            const double& z_component_value = z_component.at(iz);
            evaluations.slice(iz) = z_component_value * (x_component * y_component.t());
        }

        return evaluations;
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
