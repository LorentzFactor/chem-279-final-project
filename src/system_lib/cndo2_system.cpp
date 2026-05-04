#include "system_lib.h"

#include <complex>

using namespace gaussian_lib;
namespace fs = std::filesystem;
using json = nlohmann::json;

namespace system_lib {

CNDO2System::CNDO2System(const std::vector<Atom> &atoms, int p, int q)
    : System(atoms) {
  p_ = (double)p;
  q_ = (double)q;
  set_p(arma::zeros(num_orbitals_, num_orbitals_),
        arma::zeros(num_orbitals_, num_orbitals_));
  I_.beta_ = arma::zeros(0, 0);
  I_.gamma_ = arma::zeros(0, 0);
  I_.gamma_reduced_ = arma::zeros(0, 0);
}

CNDO2System CNDO2System::from_files(std::string atoms_filepath,
                                    std::string basis_directory, int p, int q,
                                    DistanceUnits distance_units) {
  auto atoms =
      atoms_from_files_(atoms_filepath, basis_directory, distance_units);
  return CNDO2System(atoms, p, q);
}

void CNDO2System::set_p(const arma::mat &new_p_alpha,
                        const arma::mat &new_p_beta) {
  // Ensure density matrices are properly shaped
  if (!(new_p_alpha.n_cols == num_orbitals_) ||
      !(new_p_alpha.n_rows == num_orbitals_) ||
      !(new_p_beta.n_cols == num_orbitals_) ||
      !(new_p_beta.n_rows == num_orbitals_)) {
    throw std::runtime_error("Invalid shape for p_alpha/beta");
  }

  // Set internal density matrices to new values
  p_alpha_ = new_p_alpha;
  p_beta_ = new_p_beta;

  // update fock matrices with new densities
  auto new_f = CNDO2System::compute_cndo_f_matrix_internal(p_alpha_, p_beta_);
  set_f(new_f.first, new_f.second);
}

void CNDO2System::set_f(const arma::mat &new_f_alpha,
                        const arma::mat &new_f_beta) {
  f_alpha_ = new_f_alpha;
  f_beta_ = new_f_beta;

  // update molecular orbitals
  arma::eig_sym(E_alpha_, mos_alpha_, new_f_alpha);
  arma::eig_sym(E_beta_, mos_beta_, new_f_beta);
}

RealMat compute_gamma_matrix(RealMat &gamma, System &sys,
                             const std::vector<Atom> &atoms_) {
  if (gamma.n_cols > 0) {
    return gamma;
  }

  gamma = arma::zeros(sys.num_orbitals(), sys.num_orbitals());

  std::vector<Atom> atoms{};
  atoms.reserve(sys.num_orbitals());
  for (const auto &atom : atoms_) {
    for (const auto &orbital : atom.get_atomic_orbitals()) {
      atoms.emplace_back(atom);
    }
  }

  for (int iatom = 0; iatom < atoms.size(); iatom++) {
    const Atom &atom_i = atoms.at(iatom);
    for (int jatom = iatom; jatom < atoms.size(); jatom++) {
      const Atom &atom_j = atoms.at(jatom);
      // Compute gamma using the s orbitals of each atom
      double gamma_ij = calculate_gamma(atom_i.get_atomic_orbitals().at(0),
                                        atom_j.get_atomic_orbitals().at(0));
      gamma(iatom, jatom) = gamma_ij;
      gamma(jatom, iatom) = gamma_ij;
    }
  }
  return gamma;
};

/* Compute the gamma matrix indexed by atoms rather than orbitals */
RealMat compute_reduced_gamma_matrix(RealMat &gamma_reduced, System &sys) {
  if (gamma_reduced.n_cols > 0) {
    return gamma_reduced;
  }

  gamma_reduced = arma::zeros(sys.num_atoms(), sys.num_atoms());

  for (int iatom = 0; iatom < sys.num_atoms(); iatom++) {
    const Atom &atom_i = sys.get_atom(iatom);
    for (int jatom = iatom; jatom < sys.num_atoms(); jatom++) {
      const Atom &atom_j = sys.get_atom(jatom);
      // Compute gamma using the s orbitals of each atom
      double gamma_ij = calculate_gamma(atom_i.get_atomic_orbitals().at(0),
                                        atom_j.get_atomic_orbitals().at(0));
      gamma_reduced(iatom, jatom) = gamma_ij;
      gamma_reduced(jatom, iatom) = gamma_ij;
    }
  }
  return gamma_reduced;
}

RealMat compute_beta_matrix(RealMat &beta, System &sys,
                            const std::vector<Atom> &atoms) {
  if (beta.n_cols > 0) {
    return beta;
  }

  beta = arma::zeros(sys.num_orbitals(), sys.num_orbitals());
  std::vector<double> orbital_betas{};
  orbital_betas.reserve(sys.num_orbitals());
  for (const auto &atom : atoms) {
    double beta = atom.get_atom_constant("neg_beta");
    for (size_t i = 0; i < atom.num_orbitals(); ++i) {
      orbital_betas.push_back(beta);
    }
  }
  RealVec vec_betas = RealVec(orbital_betas.data(), sys.num_orbitals());
  vec_betas /= 2;
  beta.each_col() += vec_betas;
  beta.each_row() += vec_betas.as_row();

  return beta;
}

std::pair<RealMat, RealMat>
CNDO2System::compute_cndo_f_matrix_internal(const RealMat &p_alpha,
                                            const RealMat &p_beta) {
  return build_cndo2_fock(*this, p_alpha, p_beta);
}

std::pair<arma::mat, arma::mat> CNDO2System::compute_cndo_f_matrix() {
  return {f_alpha_, f_beta_};
}

arma::mat CNDO2System::compute_h_core() {
  return CNDO2System::compute_cndo_f_matrix_internal(
             arma::zeros(num_orbitals(), num_orbitals()),
             arma::zeros(num_orbitals(), num_orbitals()))
      .first;
}

double CNDO2System::compute_electronic_energy() {
  arma::mat h_core = compute_h_core();

  return 0.5 * (arma::accu(p_alpha_ % (h_core + f_alpha_)) +
                arma::accu(p_beta_ % (h_core + f_beta_)));
}

double CNDO2System::get_electron_density(size_t atom_idx) const {
  double density = 0;
  density += arma::accu(arma::square(get_occupied_MOs_alpha().rows(
      atom_orbital_idxs[atom_idx][0], atom_orbital_idxs[atom_idx][1] - 1)));
  density += arma::accu(arma::square(get_occupied_MOs_beta().rows(
      atom_orbital_idxs[atom_idx][0], atom_orbital_idxs[atom_idx][1] - 1)));
  return density;
}

arma::mat CNDO2System::get_occupied_MOs_alpha() const {
  if (p_ > 0)
    return mos_alpha_.cols(arma::span(0, p_ - 1));
  else
    return arma::zeros(num_orbitals_, num_orbitals_);
}

arma::mat CNDO2System::get_occupied_MOs_beta() const {
  if (q_ > 0)
    return mos_beta_.cols(arma::span(0, q_ - 1));
  else
    return arma::zeros(num_orbitals_, num_orbitals_);
}

CNDO2SystemComplex::CNDO2SystemComplex(const std::vector<Atom> &atoms, int p,
                                       int q)
    : System(atoms) {
  p_ = (double)p;
  q_ = (double)q;
  set_p(ComplexMat(num_orbitals_, num_orbitals_, arma::fill::zeros),
        ComplexMat(num_orbitals_, num_orbitals_, arma::fill::zeros));
  I_.beta_ = arma::zeros(0, 0);
  I_.gamma_ = arma::zeros(0, 0);
  I_.gamma_reduced_ = arma::zeros(0, 0);
}

CNDO2SystemComplex
CNDO2SystemComplex::from_files(std::string atoms_filepath,
                               std::string basis_directory, int p, int q,
                               DistanceUnits distance_units) {
  auto atoms =
      atoms_from_files_(atoms_filepath, basis_directory, distance_units);
  return CNDO2SystemComplex(atoms, p, q);
}

void CNDO2SystemComplex::set_p(const ComplexMat &new_p_alpha,
                               const ComplexMat &new_p_beta) {
  if (!(new_p_alpha.n_cols == num_orbitals_) ||
      !(new_p_alpha.n_rows == num_orbitals_) ||
      !(new_p_beta.n_cols == num_orbitals_) ||
      !(new_p_beta.n_rows == num_orbitals_)) {
    throw std::runtime_error("Invalid shape for complex p_alpha/beta");
  }
  p_alpha_ = new_p_alpha;
  p_beta_ = new_p_beta;
  auto new_f =
      CNDO2SystemComplex::compute_cndo_f_matrix_internal(p_alpha_, p_beta_);
  set_f(new_f.first, new_f.second);
}

void CNDO2SystemComplex::set_f(const ComplexMat &new_f_alpha,
                               const ComplexMat &new_f_beta) {
  f_alpha_ = new_f_alpha;
  f_beta_ = new_f_beta;
  arma::eig_sym(E_alpha_, mos_alpha_, new_f_alpha);
  arma::eig_sym(E_beta_, mos_beta_, new_f_beta);
}

std::pair<ComplexMat, ComplexMat>
CNDO2SystemComplex::compute_cndo_f_matrix_internal(const ComplexMat &p_alpha,
                                                   const ComplexMat &p_beta) {
  return build_cndo2_fock(*this, p_alpha, p_beta);
}

std::pair<ComplexMat, ComplexMat> CNDO2SystemComplex::compute_cndo_f_matrix() {
  return {f_alpha_, f_beta_};
}

ComplexMat CNDO2SystemComplex::compute_h_core() {
  return CNDO2SystemComplex::compute_cndo_f_matrix_internal(
             ComplexMat(num_orbitals(), num_orbitals(), arma::fill::zeros),
             ComplexMat(num_orbitals(), num_orbitals(), arma::fill::zeros))
      .first;
}

double CNDO2SystemComplex::compute_electronic_energy() {
  ComplexMat h_core = compute_h_core();
  return std::real(0.5 * (arma::accu(p_alpha_ % (h_core + f_alpha_)) +
                          arma::accu(p_beta_ % (h_core + f_beta_))));
}

ComplexMat CNDO2SystemComplex::get_occupied_MOs_alpha() const {
  if (p_ > 0)
    return mos_alpha_.cols(arma::span(0, p_ - 1));
  return ComplexMat(num_orbitals_, num_orbitals_, arma::fill::zeros);
}

ComplexMat CNDO2SystemComplex::get_occupied_MOs_beta() const {
  if (q_ > 0)
    return mos_beta_.cols(arma::span(0, q_ - 1));
  return ComplexMat(num_orbitals_, num_orbitals_, arma::fill::zeros);
}

// RealMat CNDO2SystemComplex::get_ang_mom_matrix(int direction) const {
//   return arma::zeros(num_orbitals_, num_orbitals_);
// }

double CNDO2SystemComplex::calc_angular_momentum_term(
    const gaussian_lib::GaussianContracted &u,
    const gaussian_lib::GaussianContracted &v,
    const std::array<double, 3> &gauge_origin, int coord_dir, int deriv_dir) {
  double total_integral = 0.0;
  // Take derivative in specific direction
  auto deriv_components = v.get_gaussian_derivative(deriv_dir);

  // Loop through the returned derivatives
  for (const auto &deriv : deriv_components) {
    // Multiply derivative by coords
    auto coord_components =
        deriv.multiply_by_coords(coord_dir, gauge_origin[coord_dir]);

    // Get combined prefactors, multiply by overlap, add to total integral
    for (const auto &final_deriv : coord_components) {
      double combined_prefactor = deriv.prefactor * final_deriv.prefactor;
      double overlap = integrate_product(u, final_deriv.gaussian);
      total_integral += combined_prefactor * overlap;
    }
  }
  return total_integral;
}

RealMat CNDO2SystemComplex::compute_angular_momentum_matrix(
    int direction, const std::array<double, 3> &gauge_origin) const {
  RealMat M;
  M.zeros(num_orbitals(), num_orbitals());

  // Figure out the axes for cross-product based on direction
  // Example for x you need y coord and z deriv
  int coord1 = (direction + 1) % 3;
  int deriv1 = (direction + 2) % 3;

  // flipped for second term
  int coord2 = deriv1;
  int deriv2 = coord1;

  // Flatten overlap list so that we can build the matrix
  std::vector<gaussian_lib::GaussianContracted> basis_functions;
  for (const auto &atom : atoms_) {
    for (const auto &orbital : atom.get_atomic_orbitals()) {
      basis_functions.push_back(orbital);
    }
  }

  // Iterate through all the AOs and fill out ang momentum matrix
  for (size_t i = 0; i < basis_functions.size(); ++i) {
    for (size_t j = 0; j < basis_functions.size(); ++j) {

      const auto &u = basis_functions[i];
      const auto &v = basis_functions[j];

      // Positive term: +(r1 * d1)
      double term_positive =
          calc_angular_momentum_term(u, v, guage_origin, coord1, deriv1);

      // Negative term: -(r1 * d1)
      double term_negative =
          calc_angular_momentum_term(u, v, guage_origin, coord2, deriv2);

      M(i, j) = term_positive - term_negative;
    }
  }

  return M;
}

} // namespace system_lib