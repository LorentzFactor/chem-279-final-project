#include "solver_lib.h"

namespace diis {
namespace {

bool solve_diis_coeffs(RealVec &coeffs, const RealMat &solution_mat,
                       const RealVec &target_vec) {
  bool ok = arma::solve(coeffs, solution_mat, target_vec, arma::solve_opts::force_sym);
  if (ok && coeffs.is_finite()) {
    return true;
  }
  //return true;

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

template <typename SystemType, typename DensityMatrixType>
void initialize_p_atomic(DensityMatrixType &p_alpha, DensityMatrixType &p_beta,
                         const SystemType &sys) {
  p_alpha = DensityMatrixType(sys.num_orbitals(), sys.num_orbitals(),
                              arma::fill::randu) /
            10.0;
  p_beta = DensityMatrixType(sys.num_orbitals(), sys.num_orbitals(),
                             arma::fill::randu) /
           10.0;

  for (size_t iatom = 0; iatom < sys.num_atoms(); ++iatom) {
    const auto &atom_orbital_idx = sys.get_atom_orbital_idxs().at(iatom);
    const auto &atom = sys.get_atom(iatom);
    int num_valence_electrons = atom.get_atomic_number();
    if (num_valence_electrons > 2) {
      num_valence_electrons -= 2;
    }
    if (num_valence_electrons > 10) {
      num_valence_electrons -= 10;
    }
    int alpha_occupation =
        num_valence_electrons / 2 + (iatom + num_valence_electrons) % 2;
    int beta_occupation =
        num_valence_electrons / 2 + (iatom + num_valence_electrons) % 2;
    for (int i = 0; i < alpha_occupation; ++i) {
      p_alpha(atom_orbital_idx[0] + i, atom_orbital_idx[0] + i) = 1.0;
    }
    for (int i = 0; i < beta_occupation; ++i) {
      p_beta(atom_orbital_idx[0] + i, atom_orbital_idx[0] + i) = 1.0;
    }
  }
}

template <typename DensityMatrixType>
DensityMatrixType build_density_from_core_mos(const DensityMatrixType &core_mos,
                                              int num_occupied) {
  if (core_mos.n_rows != core_mos.n_cols) {
    throw std::runtime_error(
        "Extended Huckel initialization requires square MO coefficients.");
  }
  if (num_occupied < 0) {
    throw std::runtime_error(
        "Extended Huckel initialization got a negative occupation.");
  }
  if (static_cast<arma::uword>(num_occupied) > core_mos.n_cols) {
    throw std::runtime_error(
        "Extended Huckel initialization occupation exceeds orbital count.");
  }

  if (num_occupied == 0) {
    return DensityMatrixType(core_mos.n_rows, core_mos.n_rows,
                             arma::fill::zeros);
  }

  const DensityMatrixType occupied_mos =
      core_mos.cols(0, static_cast<arma::uword>(num_occupied - 1));
  return occupied_mos * occupied_mos.t();
}

double get_extended_huckel_diagonal_element(const Atom &atom,
                                            const GaussianContracted &orbital,
                                            bool use_indo) {
  if (orbital.shell == 0) {
    return -atom.get_atom_constant("sI+A/2");
  }
  if (orbital.shell == 1) {
    return -atom.get_atom_constant("pI+A/2");
  }
  throw std::runtime_error(
      "Extended Huckel initialization only supports s and p orbitals.");
}

template <typename SystemType>
RealMat build_extended_huckel_matrix(const SystemType &sys) {
  constexpr double kWolfsbergHelmholtzK = 1.75;

  const RealMat overlap = sys.compute_overlap_matrix();
  if (overlap.n_rows != sys.num_orbitals() ||
      overlap.n_cols != sys.num_orbitals()) {
    throw std::runtime_error(
        "Extended Huckel initialization got an invalid overlap matrix size.");
  }

  RealVec diagonal(overlap.n_rows, arma::fill::zeros);
  arma::uword iorbital = 0;
  for (size_t iatom = 0; iatom < sys.num_atoms(); ++iatom) {
    const Atom &atom = sys.get_atom(iatom);
    for (int ilocal = 0; ilocal < atom.num_orbitals(); ++ilocal) {
      diagonal(iorbital) = get_extended_huckel_diagonal_element(
          atom, atom.get_atomic_orbital(ilocal), sys.use_indo());
      ++iorbital;
    }
  }

  if (iorbital != overlap.n_rows) {
    throw std::runtime_error(
        "Extended Huckel initialization orbital indexing mismatch.");
  }

  RealMat h_eh = arma::zeros<RealMat>(overlap.n_rows, overlap.n_cols);
  h_eh.diag() = diagonal;
  for (arma::uword i = 0; i < overlap.n_rows; ++i) {
    for (arma::uword j = i + 1; j < overlap.n_cols; ++j) {
      const double hij =
          kWolfsbergHelmholtzK * overlap(i, j) * 0.5 * (diagonal(i) + diagonal(j));
      h_eh(i, j) = hij;
      h_eh(j, i) = hij;
    }
  }
  return h_eh;
}

void initialize_p_extended_huckel(RealMat &p_alpha, RealMat &p_beta,
                                  CNDO2System &sys) {
  RealMat h_eh = build_extended_huckel_matrix(sys);
  h_eh = 0.5 * (h_eh + h_eh.t());

  RealVec eh_energies;
  RealMat eh_mos;
  const bool ok = arma::eig_sym(eh_energies, eh_mos, h_eh);
  if (!ok || !eh_mos.is_finite()) {
    throw std::runtime_error(
        "Failed to construct Extended Huckel guess for DIIS (real)."
    );
  }

  p_alpha = build_density_from_core_mos(eh_mos, sys.get_nalpha());
  p_beta = build_density_from_core_mos(eh_mos, sys.get_nbeta());
}

void initialize_p_extended_huckel(ComplexMat &p_alpha, ComplexMat &p_beta,
                                  CNDO2SystemComplex &sys) {
  ComplexMat h_eh = arma::conv_to<ComplexMat>::from(build_extended_huckel_matrix(sys));
  h_eh = 0.5 * (h_eh + h_eh.t());

  RealVec eh_energies;
  ComplexMat eh_mos;
  const bool ok = arma::eig_sym(eh_energies, eh_mos, h_eh);
  if (!ok || !eh_mos.is_finite()) {
    throw std::runtime_error(
        "Failed to construct Extended Huckel guess for DIIS (complex)."
    );
  }

  p_alpha = build_density_from_core_mos(eh_mos, sys.get_nalpha());
  p_beta = build_density_from_core_mos(eh_mos, sys.get_nbeta());
}

template <typename SystemType, typename DensityMatrixType>
void initialize_p(DensityMatrixType &p_alpha, DensityMatrixType &p_beta,
                  SystemType &sys, InitialGuess initial_guess) {
  switch (initial_guess) {
  case InitialGuess::kAtomic:
    initialize_p_atomic(p_alpha, p_beta, sys);
    return;
  case InitialGuess::kExtendedHuckel:
    initialize_p_extended_huckel(p_alpha, p_beta, sys);
    return;
  }

  throw std::runtime_error("Unknown DIIS initial guess option.");
}

int solve_cndo(CNDO2System &sys, int max_iters, double tol, bool keep_p,
               InitialGuess initial_guess) {
  RealMat p_alpha, p_beta;
  if (!keep_p) {
    initialize_p(p_alpha, p_beta, sys, initial_guess);
  } else {
    p_alpha = sys.get_p_alpha();
    p_beta = sys.get_p_beta();
  }

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

    RealMat solution_mat_b;
    build_solutions_mat(solution_mat_b, errors_b);
    RealVec target_vec_b;
    build_target_vec(target_vec_b, errors_b);
    RealVec solution_vec_b;

    RealMat new_f_a;
    RealMat new_f_b;

    // Skip and do fixed point until we have enough error vectors to do DIIS, then switch to DIIS.
    if (true)  {
      solve_diis_coeffs(solution_vec_a, solution_mat_a, target_vec_a);
      solve_diis_coeffs(solution_vec_b, solution_mat_b, target_vec_b);
      extrapolate_f(new_f_a, solution_vec_a, f_alphas);
      extrapolate_f(new_f_b, solution_vec_b, f_betas);
      sys.set_f(new_f_a, new_f_b);
    }

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

int solve_cndo(CNDO2SystemComplex &sys, int max_iters, double tol, bool keep_p,
               InitialGuess initial_guess) {
  ComplexMat p_alpha, p_beta;
  if (!keep_p) {
    initialize_p(p_alpha, p_beta, sys, initial_guess);
  } else {
    p_alpha = sys.get_p_alpha();
    p_beta = sys.get_p_beta();
  }
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