#include "solver_lib.h"

namespace fixed_point {
    void solve_cndo(CNDO2System& sys, int max_iters, double tol) {
        arma::mat S = sys.compute_overlap_matrix();
        arma::mat reduced_gamma = sys.compute_reduced_gamma_matrix();
        arma::mat gamma = sys.compute_gamma_matrix();
        arma::mat beta = sys.compute_beta_matrix();

        int p = sys.get_nalpha();
        int q = sys.get_nbeta();

        arma::mat p_alpha = arma::mat(sys.num_orbitals(), sys.num_orbitals(), arma::fill::zeros) * (p/(sys.num_orbitals()*sys.num_orbitals()));
        arma::mat p_beta = arma::mat(sys.num_orbitals(), sys.num_orbitals(), arma::fill::zeros) *(q/(sys.num_orbitals()*sys.num_orbitals()));
        arma::mat p_tot = p_alpha + p_beta;
        sys.set_p(p_alpha, p_beta);

        arma::mat p_a_old;
        arma::mat p_b_old;
        arma::mat p_tot_old;

        const arma::mat& f_alpha = sys.get_f_alpha();
        const arma::mat& f_beta = sys.get_f_beta();

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
                return;
            }
        }

        throw std::runtime_error("Failed to converge!");
    }
}