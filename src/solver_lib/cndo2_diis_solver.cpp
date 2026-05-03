#include "solver_lib.h"

namespace diis {

    void build_solutions_mat(
        arma::mat& solution_mat,
        const std::list<arma::mat>& errors_a
    ) {
        solution_mat = arma::zeros(errors_a.size()+1, errors_a.size()+1);
        
        // Initialize bottom row and rightmost column
        for (int i=0; i < errors_a.size(); ++i) {
            solution_mat(i, errors_a.size()) = -1;
            solution_mat(errors_a.size(), i) = -1;
        }

        // Initialize remaining errors mat
        auto it_a_i = errors_a.begin();
        for (int i=0; i < errors_a.size(); ++i) {
            auto it_a_j = errors_a.begin();
            for (int j=0; j < errors_a.size(); ++j) {
                solution_mat(i,j) = arma::accu((*it_a_i) % (*it_a_j));
                ++it_a_j;
            }
            ++it_a_i;
        }
    }

    void build_target_vec(
        arma::vec& target_vec,
        const std::list<arma::mat>& errors_a
    ) {
        target_vec = arma::zeros(errors_a.size()+1);
        target_vec(errors_a.size()) = -1;
    }

    /*
        Extrapolate a new F matrix from a vector of coefficients
        (including the lagrange multiplier, which will be ignored)
        and the previous F matrices.
    */
    void extrapolate_f(
        arma::mat& new_f,
        const arma::vec& coefficients,
        const std::list<arma::mat>& prev_fs
    ) {
        size_t f_size = (*prev_fs.begin()).n_cols;
        new_f = arma::zeros(f_size, f_size);

        auto it_cs = coefficients.begin();
        for(auto it_fs = prev_fs.begin(); it_fs != prev_fs.end(); ++it_fs) {
            new_f += (*it_cs) * (*it_fs);
            ++it_cs;
        }
    }

    int solve_cndo(CNDO2System& sys, int max_iters, double tol) {
        arma::mat p_alpha = arma::mat(sys.num_orbitals(), sys.num_orbitals(), arma::fill::randu);
        arma::mat p_beta = arma::mat(sys.num_orbitals(), sys.num_orbitals(), arma::fill::randu);
        sys.set_p(p_alpha, p_beta);

        size_t error_lengths = 10; //sys.num_orbitals();
        std::list<arma::mat> errors_a{};
        std::list<arma::mat> errors_b{};
        std::list<arma::mat> f_alphas{};
        std::list<arma::mat> f_betas{};

        for(size_t i = 0; i < max_iters; ++i) {
            arma::mat error_a_prev = p_alpha * sys.get_f_alpha() - sys.get_f_alpha() * p_alpha;
            arma::mat error_b_prev = p_beta * sys.get_f_beta() - sys.get_f_beta() * p_beta;

            std::cout << "e_a: " << arma::norm(error_a_prev) << "\n"
                      << "e_b: " << arma::norm(error_b_prev) << std::endl;

            arma::mat p_a_old = p_alpha;
            arma::mat p_b_old = p_beta;

            // If errors list is full, pop oldest entry
            if(errors_a.size() == error_lengths) {
                errors_a.pop_front();
                errors_b.pop_front();
                f_alphas.pop_front();
                f_betas.pop_front();
            }
            errors_a.push_back(error_a_prev);
            errors_b.push_back(error_b_prev);
            f_alphas.push_back(sys.get_f_alpha());
            f_betas.push_back(sys.get_f_beta());

            arma::mat solution_mat_a;
            build_solutions_mat(solution_mat_a, errors_a);
            arma::vec target_vec_a;
            build_target_vec(target_vec_a, errors_a);
            arma::vec solution_vec_a;
            arma::solve(solution_vec_a, solution_mat_a, target_vec_a);

            arma::mat solution_mat_b;
            build_solutions_mat(solution_mat_b, errors_b);
            arma::vec target_vec_b;
            build_target_vec(target_vec_b, errors_b);
            arma::vec solution_vec_b;
            arma::solve(solution_vec_b, solution_mat_b, target_vec_b);

            arma::mat new_f_a;
            arma::mat new_f_b;
            extrapolate_f(new_f_a, solution_vec_a, f_alphas);
            extrapolate_f(new_f_b, solution_vec_b, f_betas);
            sys.set_f(new_f_a, new_f_b);

            p_alpha = sys.get_occupied_MOs_alpha() * sys.get_occupied_MOs_alpha().t();
            p_beta = sys.get_occupied_MOs_beta() * sys.get_occupied_MOs_beta().t();
            sys.set_p(p_alpha, p_beta);

            if (
                arma::approx_equal(p_alpha, p_a_old, "absdiff", tol) &&
                arma::approx_equal(p_beta, p_b_old, "absdiff", tol)
            ) {
                return i+1;
            }
        }

        throw std::runtime_error("Failed to converge!");
    }
}