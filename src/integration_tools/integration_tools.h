#pragma once

#include <concepts>
#include <functional>
#include <cmath>
#include <iostream>
#include <limits>

namespace integration_tools {

template<typename Func>
concept RealValuedFunction =
    std::regular_invocable<Func, double> && std::same_as<std::invoke_result_t<Func, double>, double>;

// Base class for quadrature methods from numerical recipes.
struct Quadrature {
    int n;
    virtual double next() = 0;
    virtual ~Quadrature() = default;
};

// Implementation of the trapezoidal rule for numerical integration.
template<typename Func>
requires RealValuedFunction<Func>
struct Trapzd : Quadrature {
    double a, b, s;
    Func &func;

    Trapzd() = default;
    Trapzd(Func &f, double a_, double b_) : func(f), a(a_), b(b_) { n = 0; }

    double next() override {
        double tnm, del;
        int it, j;
        ++n;
        if (n == 1) {
            s = 0.5 * (b - a) * (func(a) + func(b));
        } else {
            for (it = 1, j = 1; j < n - 1; j++) {
                it <<= 1;
            }
            tnm = it;
            del = (b - a) / tnm;
            double x_base = a + 0.5 * del;
            double sum = 0;
            # pragma omp parallel for reduction(+:sum)
            for(int k = 1; k <= it; k++) {
                double x = x_base + del * (k-1);
                double f_x = func(x);
                sum += f_x;
            }
            s = 0.5 * (s + (b - a) * sum / tnm);
        }
        return s;
    }
};

template<typename Func>
requires RealValuedFunction<Func>
double qtrap(Func &func, const double a, const double b, const double eps = 1.0e-6, const int n = 1) {
    //std::cout << "eps: " << eps << " n: " << n << std::endl;
    const int JMAX = 20;
    double s, olds = 0.0;
    Trapzd<Func> t(func, a, b);
    for (int j = 0; j < JMAX; j++) {
        s = t.next();
        //std::cout << "Step: " << j << " Estimate: " << s << std::endl;
        if (j > 5) {
            // Check for convergence. We check that the difference between the current and previous estimate is less than the specified tolerance times the absolute value of the previous estimate.
            // We also check if both estimates are very small (less than machine epsilon) to avoid issues with floating-point precision.
            if (std::abs(s - olds) < eps * std::abs(olds) || (s <= std::numeric_limits<double>::epsilon() && olds <= std::numeric_limits<double>::epsilon())) {
                return s;
            }
            //std::cout << "Difference: " << std::abs(s - olds) << " Tolerance: " << eps * std::abs(olds) << " Old S: " << std::abs(olds) << " Eps: " << eps << " S: " << s << std::endl;
        }
        olds = s;
    }
    throw("Too many steps in routine qtrap");
}

}