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
        arma::vec x_component = arma::vec(x_points.size());
        for (int ix = 0; ix < x_points.size(); ++ix) {
            x_component(ix) = components_[0](x_points[ix]);
        }

        arma::vec y_component = arma::vec(y_points.size());
        for (int iy = 0; iy < y_points.size(); ++iy) {
            y_component(iy) = components_[1](y_points[iy]);
        }

        // Note that we apply the norm to the last dimension to reduce number of division operations
        arma::vec z_component = arma::vec(z_points.size());
        for (int iz = 0; iz < z_points.size(); ++iz) {
            z_component(iz) = components_[2](z_points[iz]);
        }

        // slice along z dimension, take tensor product of other two for parallelism
        arma::mat x_y_grid = (x_component * y_component.t());
        for(size_t iz=0; iz<z_points.size(); ++iz) {
            const double& z_component_value = z_component.at(iz);
            evaluations.slice(iz) = z_component_value * x_y_grid;
        }

        return evaluations / norm;
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

    std::array<double,3> integrate_product_dRa(const NormedGaussianPrimitive3d& ga, const NormedGaussianPrimitive3d& gb) {
        double Ix = integrate_product(ga.component(0), gb.component(0));
        double Iy = integrate_product(ga.component(1), gb.component(1));
        double Iz = integrate_product(ga.component(2), gb.component(2));

        std::array<double,3> gradient {
            integrate_product_dxa(ga.component(0), gb.component(0)) * Iy * Iz,
            integrate_product_dxa(ga.component(1), gb.component(1)) * Ix * Iz,
            integrate_product_dxa(ga.component(2), gb.component(2)) * Ix * Iy,
        };

        for (auto& elm : gradient) {
            elm /= ga.norm;
            elm /= gb.norm;
        }

        return gradient;
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
