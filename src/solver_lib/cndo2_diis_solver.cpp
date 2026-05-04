#include "solver_lib.h"

namespace diis {
namespace {

bool solve_diis_coeffs(RealVec &coeffs, const RealMat &solution_mat,
                       const RealVec &target_vec) {
  bool ok = arma::solve(coeffs, solution_mat, target_vec, arma::solve_opts::no_approx);
  if (ok && coeffs.is_finite()) {
    return true;
  }

  RealMat regularized = solution_mat;
  for (arma::uword i = 0; i + 1 < regularized.n_rows; ++i) {
    regularized(i, i) += 1.0e-10;
  }

  ok = arma::solve(coeffs, regularized, target_vec, arma::solve_opts::no_approx);
  if (ok && coeffs.is_finite()) {
    return true;
  }

  coeffs = arma::zeros<RealVec>(target_vec.n_elem);
  if (target_vec.n_elem >= 2) {
    // Fall back to the newest stored Fock matrix coefficient.
    coeffs(target_vec.n_elem - 2) = 1.0;
  }
  return false;
}

} // namespace

void build_solutions_mat(RealMat &solution_mat,
                         const std::list<RealMat> &errors_a) {
  solution_mat = arma::zeros(errors_a.size() + 1, errors_a.size() + 1);

  // Initialize bottom row and rightmost column
  for (int i = 0; i < errors_a.size(); ++i) {
    solution_mat(i, errors_a.size()) = -1;
    solution_mat(errors_a.size(), i) = -1;
  }

  // Initialize remaining errors mat
  auto it_a_i = errors_a.begin();
  for (int i = 0; i < errors_a.size(); ++i) {
    auto it_a_j = errors_a.begin();
    for (int j = 0; j < errors_a.size(); ++j) {
      solution_mat(i, j) = arma::accu((*it_a_i) % (*it_a_j));
      ++it_a_j;
    }
    ++it_a_i;
  }
}

// Complex analogue: build the real DIIS B-matrix from complex error matrices.
// We use the Frobenius inner product <Ei, Ej> = Re( sum(conj(Ei) ⊙ Ej) ),
// which yields a real symmetric system in the common case of nearly-Hermitian
// errors from commutators.
void build_solutions_mat(RealMat &solution_mat,
                         const std::list<ComplexMat> &errors_a) {
  solution_mat = arma::zeros(errors_a.size() + 1, errors_a.size() + 1);

  for (int i = 0; i < static_cast<int>(errors_a.size()); ++i) {
    solution_mat(i, errors_a.size()) = -1;
    solution_mat(errors_a.size(), i) = -1;
  }

  auto it_a_i = errors_a.begin();
  for (int i = 0; i < static_cast<int>(errors_a.size()); ++i) {
    auto it_a_j = errors_a.begin();
    for (int j = 0; j < static_cast<int>(errors_a.size()); ++j) {
      const arma::cx_double ip = arma::accu(arma::conj(*it_a_i) % (*it_a_j));
      solution_mat(i, j) = std::real(ip);
      ++it_a_j;
    }
    ++it_a_i;
  }
}

void build_target_vec(RealVec &target_vec,
                      const std::list<RealMat> &errors_a) {
  target_vec = arma::zeros(errors_a.size() + 1);
  target_vec(errors_a.size()) = -1;
}

void build_target_vec(RealVec &target_vec,
                      const std::list<ComplexMat> &errors_a) {
  target_vec = arma::zeros(errors_a.size() + 1);
  target_vec(errors_a.size()) = -1;
}

/*
    Extrapolate a new F matrix from a vector of coefficients
    (including the lagrange multiplier, which will be ignored)
    and the previous F matrices.
*/
void extrapolate_f(RealMat &new_f, const RealVec &coefficients,
                   const std::list<RealMat> &prev_fs) {
  size_t f_size = (*prev_fs.begin()).n_cols;
  new_f = RealMat(f_size, f_size, arma::fill::zeros);

  auto it_cs = coefficients.begin();
  for (auto it_fs = prev_fs.begin(); it_fs != prev_fs.end(); ++it_fs) {
    new_f += (*it_cs) * (*it_fs);
    ++it_cs;
  }
}

void extrapolate_f(ComplexMat &new_f, const RealVec &coefficients,
                   const std::list<ComplexMat> &prev_fs) {
  size_t f_size = (*prev_fs.begin()).n_cols;
  new_f = arma::zeros<ComplexMat>(f_size, f_size);

  auto it_cs = coefficients.begin();
  for (auto it_fs = prev_fs.begin(); it_fs != prev_fs.end(); ++it_fs) {
    new_f += (*it_cs) * (*it_fs);
    ++it_cs;
  }
}

int solve_cndo(CNDO2System &sys, int max_iters, double tol) {
  RealMat p_alpha =
      RealMat(sys.num_orbitals(), sys.num_orbitals(), arma::fill::randu);
  RealMat p_beta =
      RealMat(sys.num_orbitals(), sys.num_orbitals(), arma::fill::randu);
  sys.set_p(p_alpha, p_beta);

  size_t error_lengths = 10; // sys.num_orbitals();
  std::list<RealMat> errors_a{};
  std::list<RealMat> errors_b{};
  std::list<RealMat> f_alphas{};
  std::list<RealMat> f_betas{};

  for (size_t i = 0; i < max_iters; ++i) {
    RealMat error_a_prev =
        p_alpha * sys.get_f_alpha() - sys.get_f_alpha() * p_alpha;
    RealMat error_b_prev =
        p_beta * sys.get_f_beta() - sys.get_f_beta() * p_beta;

    RealMat p_a_old = p_alpha;
    RealMat p_b_old = p_beta;

    // If errors list is full, pop oldest entry
    if (errors_a.size() == error_lengths) {
      errors_a.pop_front();
      errors_b.pop_front();
      f_alphas.pop_front();
      f_betas.pop_front();
    }
    errors_a.push_back(error_a_prev);
    errors_b.push_back(error_b_prev);
    f_alphas.push_back(sys.get_f_alpha());
    f_betas.push_back(sys.get_f_beta());

    RealMat solution_mat_a;
    build_solutions_mat(solution_mat_a, errors_a);
    RealVec target_vec_a;
    build_target_vec(target_vec_a, errors_a);
    RealVec solution_vec_a;
    solve_diis_coeffs(solution_vec_a, solution_mat_a, target_vec_a);

    RealMat solution_mat_b;
    build_solutions_mat(solution_mat_b, errors_b);
    RealVec target_vec_b;
    build_target_vec(target_vec_b, errors_b);
    RealVec solution_vec_b;
    solve_diis_coeffs(solution_vec_b, solution_mat_b, target_vec_b);

    RealMat new_f_a;
    RealMat new_f_b;
    extrapolate_f(new_f_a, solution_vec_a, f_alphas);
    extrapolate_f(new_f_b, solution_vec_b, f_betas);
    sys.set_f(new_f_a, new_f_b);

    p_alpha = sys.get_occupied_MOs_alpha() * sys.get_occupied_MOs_alpha().t();
    p_beta = sys.get_occupied_MOs_beta() * sys.get_occupied_MOs_beta().t();
    sys.set_p(p_alpha, p_beta);

    if (arma::approx_equal(p_alpha, p_a_old, "absdiff", tol) &&
        arma::approx_equal(p_beta, p_b_old, "absdiff", tol)) {
      return i + 1;
    }
  }

  throw std::runtime_error("Failed to converge!");
}

int solve_cndo(CNDO2SystemComplex &sys, int max_iters, double tol) {
  ComplexMat p_alpha =
      ComplexMat(sys.num_orbitals(), sys.num_orbitals(), arma::fill::randu);
  ComplexMat p_beta =
      ComplexMat(sys.num_orbitals(), sys.num_orbitals(), arma::fill::randu);
  sys.set_p(p_alpha, p_beta);

  size_t error_lengths = 10; // sys.num_orbitals();
  std::list<ComplexMat> errors_a{};
  std::list<ComplexMat> errors_b{};
  std::list<ComplexMat> f_alphas{};
  std::list<ComplexMat> f_betas{};

  for (size_t i = 0; i < max_iters; ++i) {
    ComplexMat error_a_prev =
        p_alpha * sys.get_f_alpha() - sys.get_f_alpha() * p_alpha;
    ComplexMat error_b_prev =
        p_beta * sys.get_f_beta() - sys.get_f_beta() * p_beta;

    ComplexMat p_a_old = p_alpha;
    ComplexMat p_b_old = p_beta;

    // If errors list is full, pop oldest entry
    if (errors_a.size() == error_lengths) {
      errors_a.pop_front();
      errors_b.pop_front();
      f_alphas.pop_front();
      f_betas.pop_front();
    }
    errors_a.push_back(error_a_prev);
    errors_b.push_back(error_b_prev);
    f_alphas.push_back(sys.get_f_alpha());
    f_betas.push_back(sys.get_f_beta());

    RealMat solution_mat_a;
    build_solutions_mat(solution_mat_a, errors_a);
    RealVec target_vec_a;
    build_target_vec(target_vec_a, errors_a);
    RealVec solution_vec_a;
    solve_diis_coeffs(solution_vec_a, solution_mat_a, target_vec_a);

    RealMat solution_mat_b;
    build_solutions_mat(solution_mat_b, errors_b);
    RealVec target_vec_b;
    build_target_vec(target_vec_b, errors_b);
    RealVec solution_vec_b;
    solve_diis_coeffs(solution_vec_b, solution_mat_b, target_vec_b);

    ComplexMat new_f_a;
    ComplexMat new_f_b;
    extrapolate_f(new_f_a, solution_vec_a, f_alphas);
    extrapolate_f(new_f_b, solution_vec_b, f_betas);
    sys.set_f(new_f_a, new_f_b);

    p_alpha = sys.get_occupied_MOs_alpha() * sys.get_occupied_MOs_alpha().t();
    p_beta = sys.get_occupied_MOs_beta() * sys.get_occupied_MOs_beta().t();
    sys.set_p(p_alpha, p_beta);

    if (arma::approx_equal(p_alpha, p_a_old, "absdiff", tol) &&
        arma::approx_equal(p_beta, p_b_old, "absdiff", tol)) {
      return i + 1;
    }
  }

  throw std::runtime_error("Failed to converge!");
}
} // namespace diis