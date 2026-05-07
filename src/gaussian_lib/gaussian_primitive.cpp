#include "gaussian_lib.h"

namespace {
    /*
    Note - it's probably more efficient to compute these using a loop than recursively,
    but this is more straightforward to write and understand.
    Since n is likely small, it may be even better to use templated functions and compute
    these at compile time.But I'm not 100% certain we will not have to compute these for
    large n at some point, so I will just write them as recursive functions for now.
    */

    /*Compute double factorial*/
    int double_factorial(int n) {

        // Check for invalid input
        if (n < -1) {
            throw std::runtime_error("Input less than -1 not allowed for double factorial");
        }

        // Base cases
        if (n == -1) {return 1;}
        if (n == 0) {return 1;}

        return double_factorial(n - 2) * n;
    }

    /*Compute factorial*/
    int factorial(int n) {

        // Check for invalid input
        if (n < 0) {
            throw std::runtime_error("Negative input not allowed for factorial");
        }

        // Base case
        if (n == 0) {return 1;}

        return factorial(n - 1) * n;
    }

    /*
        Computes the binomial coefficient "m choose n"
        Note that this has issues with potentially overflowing
        for large m or n when it could be avoided.
    */
    int binomial_coefficient(int m, int n) {
        // Check for invalid inputs
        if (m < 0 || n < 0 || n > m) {
            throw std::runtime_error("Invalid input for binomial coefficient");
        }

        // Use the symmetry (m choose n = m choose m-n)
        // to reduce the number of multiplications when n is small compared to m
        if (m-n > n) {
            n = m - n;
        }

        int prod = 1;
        for (int i = n+1; i <= m; i++) {
            prod *= i;
        }
        prod /= factorial(m - n);
        return prod;
    }
}

namespace gaussian_lib {
    double GaussianPrimitive::operator()(double x) const {
        return std::pow(x-center, momentum) * std::exp(-exponent * std::pow(x - center, 2));
    }

    /*Given two Gaussian primitives, compute the center of the product Gaussian,
    which is just the exponent-weighted average of the input centers.*/
    double compute_rp(const GaussianPrimitive& g1, const GaussianPrimitive& g2) {
        double R_p = (g1.exponent * g1.center + g2.exponent * g2.center) \
            / (g1.exponent + g2.exponent);
        return R_p;
    }

    double integrate_product(const GaussianPrimitive& ga, const GaussianPrimitive& gb) {
        double outer_prefactor = std::sqrt(M_PI / (ga.exponent + gb.exponent)) * \
            std::exp(-ga.exponent * gb.exponent * std::pow(ga.center - gb.center, 2) / (ga.exponent + gb.exponent));
        
        double rp = compute_rp(ga, gb); // the center of the product Gaussian
        double summation = 0;
        // For now, just have a simple nested for-loop
        for (int i = 0; i <= ga.momentum; i++) {
            for (int j = 0; j <= gb.momentum; j++) {
                if ((i + j) % 2 != 0) {
                    continue; // skip odd terms since they will integrate to zero
                }

                summation += binomial_coefficient(ga.momentum, i) * \
                    binomial_coefficient(gb.momentum, j) * \
                    std::pow(rp - ga.center, ga.momentum - i) * \
                    std::pow(rp - gb.center, gb.momentum - j) * \
                    double_factorial(i + j - 1) / std::pow(2 * (ga.exponent + gb.exponent), (i + j) / 2);
            }
        }
        summation *= outer_prefactor;
        return summation;
    }

    /* Compute the partial derivative of the integral of the product w.r.t x_a */
    double integrate_product_dxa(const GaussianPrimitive& ga, const GaussianPrimitive& gb) {
        double result = 0;
        // First term is 0 if momentum = 0... else it's the following
        if (ga.momentum != 0) {
            GaussianPrimitive temp_a_term1(ga.center, ga.exponent, ga.momentum-1);
            result = -ga.momentum * integrate_product(temp_a_term1, gb);
        }
        GaussianPrimitive temp_a_term2(ga.center, ga.exponent, ga.momentum+1);
        result += 2*ga.exponent*integrate_product(temp_a_term2, gb);

        return result;
    }

    double numerically_integrate_product(const GaussianPrimitive& ga, const GaussianPrimitive& gb, double tol) {
        auto product = [ga, gb](double x) {
            double g1 = ga(x);
            double g2 = gb(x);
            return g1 * g2;
        };

        // Determine integration limits based on the centroids and exponents
        double lower_limit = std::min(ga.center - 5 / std::sqrt(ga.exponent), gb.center - 5 / std::sqrt(gb.exponent));
        double upper_limit = std::max(ga.center + 5 / std::sqrt(ga.exponent), gb.center + 5 / std::sqrt(gb.exponent));

        // Integrate the product of the two Gaussians
        return integration_tools::qtrap(product, lower_limit, upper_limit, tol);
    }

    std::ostream& operator<<(std::ostream& os, const GaussianPrimitive& g) {
        os << "GaussianPrimitive(center=" << g.center << ", exponent=" << g.exponent << ", momentum=" << static_cast<int>(g.momentum) << ")";
        return os;
    }
}
