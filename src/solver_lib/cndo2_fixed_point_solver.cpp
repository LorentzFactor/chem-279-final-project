#include "solver_lib.h"

namespace fixed_point {
    int solve_cndo(CNDO2System& sys, int max_iters, double tol) {
        int p = sys.get_nalpha();
        int q = sys.get_nbeta();

        RealMat p_alpha =
            RealMat(sys.num_orbitals(), sys.num_orbitals(), arma::fill::zeros);
        RealMat p_beta =
            RealMat(sys.num_orbitals(), sys.num_orbitals(), arma::fill::zeros);
        RealMat p_tot = p_alpha + p_beta;
        sys.set_p(p_alpha, p_beta);

        RealMat p_a_old;
        RealMat p_b_old;
        RealMat p_tot_old;

        for(size_t i = 0; i < max_iters; ++i) {

            p_a_old = sys.get_p_alpha();
            p_b_old = sys.get_p_beta();
            p_tot_old = p_a_old + p_b_old;

            p_alpha = sys.get_occupied_MOs_alpha() * sys.get_occupied_MOs_alpha().t();
            p_beta = sys.get_occupied_MOs_beta() * sys.get_occupied_MOs_beta().t();
            
            sys.set_p(p_alpha, p_beta);
            
            p_tot = p_alpha + p_beta;

            if (
                arma::approx_equal(p_alpha, p_a_old, "absdiff", tol) &&
                arma::approx_equal(p_beta, p_b_old, "absdiff", tol)
            ) {
                return i+1;
            }
        }

        throw std::runtime_error("Failed to converge!");
    }

    int solve_cndo(CNDO2SystemComplex& sys, int max_iters, double tol) {
        ComplexMat p_alpha =
            ComplexMat(sys.num_orbitals(), sys.num_orbitals(), arma::fill::zeros);
        ComplexMat p_beta =
            ComplexMat(sys.num_orbitals(), sys.num_orbitals(), arma::fill::zeros);
        sys.set_p(p_alpha, p_beta);

        ComplexMat p_a_old;
        ComplexMat p_b_old;

        for (size_t i = 0; i < max_iters; ++i) {
            p_a_old = sys.get_p_alpha();
            p_b_old = sys.get_p_beta();

            p_alpha = sys.get_occupied_MOs_alpha() * sys.get_occupied_MOs_alpha().t();
            p_beta = sys.get_occupied_MOs_beta() * sys.get_occupied_MOs_beta().t();

            sys.set_p(p_alpha, p_beta);

            if (arma::approx_equal(p_alpha, p_a_old, "absdiff", tol) &&
                arma::approx_equal(p_beta, p_b_old, "absdiff", tol)) {
                return static_cast<int>(i + 1);
            }
        }

        throw std::runtime_error("Failed to converge (complex fixed-point)!");
    }
}