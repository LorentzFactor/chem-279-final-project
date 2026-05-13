#include "solver_lib/solver_lib.h"
#include "system_lib.h"

#include <bit>
#include <complex>

using namespace gaussian_lib;
using json = nlohmann::json;

namespace system_lib {
namespace {

inline void hash_combine(size_t &seed, size_t value) {
  seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

RealMat
central_difference_density_derivative_wrt_B(const ComplexMat &density_B_plus,
                                            const ComplexMat &density_B_minus,
                                            double epsilon_B) {
  if (density_B_plus.n_rows != density_B_minus.n_rows ||
      density_B_plus.n_cols != density_B_minus.n_cols) {
    throw std::runtime_error(
        "density_B_plus and density_B_minus dimension mismatch");
  }

  if (epsilon_B <= 0.0) {
    throw std::runtime_error("epsilon_B must be greater than zero");
  }

  const RealMat imag_plus = arma::imag(density_B_plus);
  const RealMat imag_minus = arma::imag(density_B_minus);
  return (imag_plus - imag_minus) / (2.0 * epsilon_B);
}

} // namespace

size_t CNDO2SystemComplex::AngularMomentumCacheKeyHash::operator()(
    const AngularMomentumCacheKey &key) const {
  size_t seed = 0;

  for (size_t orb = 0; orb < 2; ++orb) {
    for (size_t dim = 0; dim < 3; ++dim) {
      hash_combine(seed, std::hash<uint64_t>{}(key.centers_bits[orb][dim]));
      hash_combine(seed, std::hash<int>{}(key.momenta[orb][dim]));
    }
    hash_combine(seed, std::hash<int>{}(key.shells[orb]));
  }

  for (size_t dim = 0; dim < 3; ++dim) {
    hash_combine(seed, std::hash<uint64_t>{}(key.gauge_origin_bits[dim]));
  }

  hash_combine(seed, std::hash<int>{}(key.coord_dir));
  hash_combine(seed, std::hash<int>{}(key.deriv_dir));
  return seed;
}

CNDO2System::CNDO2System(const std::vector<Atom> &atoms, int p, int q, bool use_indo)
    : System(atoms), use_indo_(use_indo) {
  p_ = (double)p;
  q_ = (double)q;
  set_p(RealMat(num_orbitals_, num_orbitals_, arma::fill::zeros),
        RealMat(num_orbitals_, num_orbitals_, arma::fill::zeros));
  I_.beta_ = arma::zeros(0, 0);
  I_.gamma_ = arma::zeros(0, 0);
  I_.gamma_reduced_ = arma::zeros(0, 0);
}

CNDO2System CNDO2System::from_files(std::string atoms_filepath,
                                    std::string basis_directory, int p, int q,
                                    DistanceUnits distance_units, bool use_indo) {
  auto atoms =
      atoms_from_files_(atoms_filepath, basis_directory, distance_units);
  return CNDO2System(atoms, p, q, use_indo);
}

void CNDO2System::set_p(const RealMat &new_p_alpha,
                        const RealMat &new_p_beta) {
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

void CNDO2System::set_f(const RealMat &new_f_alpha,
                        const RealMat &new_f_beta) {
  // Numerical noise (and DIIS extrapolation) can introduce tiny asymmetry;
  // enforce symmetry before eig_sym.
  f_alpha_ = 0.5 * (new_f_alpha + new_f_alpha.t());
  f_beta_ = 0.5 * (new_f_beta + new_f_beta.t());

  // update molecular orbitals
  arma::eig_sym(E_alpha_, mos_alpha_, f_alpha_);
  arma::eig_sym(E_beta_, mos_beta_, f_beta_);
}

RealMat compute_gamma_matrix(RealMat &gamma, System &sys,
                             const std::vector<Atom> &atoms_) {
  if (gamma.n_cols > 0) {
    return gamma;
  }

  gamma = RealMat(sys.num_orbitals(), sys.num_orbitals(), arma::fill::zeros);

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

  gamma_reduced =
      RealMat(sys.num_atoms(), sys.num_atoms(), arma::fill::zeros);

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

  beta = RealMat(sys.num_orbitals(), sys.num_orbitals(), arma::fill::zeros);
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

bool same_orbital(const GaussianContracted& u, const GaussianContracted& v) {
  if(u.shell != v.shell)
    return false;
  for(size_t idim = 0; idim < 3; ++idim) {
    if (u.momentum[idim] != v.momentum[idim])
      return false;
  }
  return true;
}

double get_integral(const GaussianContracted &u, const GaussianContracted &v,
                    const GaussianContracted &l, const GaussianContracted &s,
                    const Atom &atom) {

  double F0 = atom.get_atom_constant("F0");

  int num_s_shells = (u.shell == 0) + (v.shell == 0) + (l.shell == 0) + (s.shell == 0);

  if (num_s_shells == 4) {
    // ssss
    return F0; 
  } else if (num_s_shells == 2) {
    // Mixed s and p orbitals
    if (same_orbital(u, v) && same_orbital(l, s)) {
      // ssxx
      return F0; 
    } else if ((same_orbital(u, l) && same_orbital(v, s)) ||
               (same_orbital(u, s) && same_orbital(v, l))) {
      // sxsx
      double G1 = atom.get_atom_constant("G1");
      return G1 / 3.0; 
    }

  } else if (num_s_shells == 0) {
    double F2 = atom.get_atom_constant("F2");

    if (same_orbital(u, v) && same_orbital(l, s)) {
      if (same_orbital(u, l)) {
        // xxxx
        return F0 + (4.0 / 25.0) * F2; 
      } else {
        // xxyy
        return F0 - (2.0 / 25.0) * F2; 
      }
    } else if ((same_orbital(u, l) && same_orbital(v, s)) ||
               (same_orbital(u, s) && same_orbital(v, l))) {
      // xyxy
      return (3.0 / 25.0) * F2; 
    }
  }

  throw std::runtime_error("Invalid orbital combination under INDO approximation.");
}

std::pair<RealMat, RealMat>
CNDO2System::compute_cndo_f_matrix_internal(const RealMat &p_alpha,
                                            const RealMat &p_beta) {
  return build_cndo2_fock(*this, p_alpha, p_beta);
}

std::pair<RealMat, RealMat> CNDO2System::compute_cndo_f_matrix() {
  return {f_alpha_, f_beta_};
}

RealMat CNDO2System::compute_h_core() {
  return CNDO2System::compute_cndo_f_matrix_internal(
             RealMat(num_orbitals(), num_orbitals(), arma::fill::zeros),
             RealMat(num_orbitals(), num_orbitals(), arma::fill::zeros))
      .first;
}

double CNDO2System::compute_electronic_energy() {
  RealMat h_core = compute_h_core();

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

RealMat CNDO2System::get_occupied_MOs_alpha() const {
  if (p_ > 0)
    return mos_alpha_.cols(arma::span(0, p_ - 1));
  else
    return RealMat(num_orbitals_, num_orbitals_, arma::fill::zeros);
}

RealMat CNDO2System::get_occupied_MOs_beta() const {
  if (q_ > 0)
    return mos_beta_.cols(arma::span(0, q_ - 1));
  else
    return RealMat(num_orbitals_, num_orbitals_, arma::fill::zeros);
}

CNDO2SystemComplex::CNDO2SystemComplex(const std::vector<Atom> &atoms, int p,
                                       int q, bool use_indo)
    : System(atoms), use_indo_(use_indo) {
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
                               DistanceUnits distance_units, bool use_indo) {
  auto atoms =
      atoms_from_files_(atoms_filepath, basis_directory, distance_units);
  return CNDO2SystemComplex(atoms, p, q, use_indo);
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
  // Enforce Hermiticity before eig_sym to avoid warnings on tiny anti-Hermitian
  // numerical residue.
  f_alpha_ = 0.5 * (new_f_alpha + new_f_alpha.t());
  f_beta_ = 0.5 * (new_f_beta + new_f_beta.t());
  arma::eig_sym(E_alpha_, mos_alpha_, f_alpha_);
  arma::eig_sym(E_beta_, mos_beta_, f_beta_);
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

double CNDO2SystemComplex::calc_angular_momentum_term(
    const gaussian_lib::GaussianContracted &u,
    const gaussian_lib::GaussianContracted &v,
    const std::array<double, 3> &gauge_origin,
    int coord_dir, int deriv_dir) const {
  AngularMomentumCacheKey cache_key{};
  const std::array<const gaussian_lib::GaussianContracted *, 2> orbitals{&u,
                                                                          &v};
  for (size_t orb = 0; orb < orbitals.size(); ++orb) {
    const auto &gaussian = *orbitals[orb];
    for (size_t dim = 0; dim < 3; ++dim) {
      cache_key.centers_bits[orb][dim] =
          std::bit_cast<uint64_t>(gaussian.center[dim]);
      cache_key.momenta[orb][dim] = gaussian.momentum[dim];
    }
    cache_key.shells[orb] = gaussian.shell;
  }
  for (size_t dim = 0; dim < 3; ++dim) {
    cache_key.gauge_origin_bits[dim] =
        std::bit_cast<uint64_t>(gauge_origin[dim]);
  }
  cache_key.coord_dir = coord_dir;
  cache_key.deriv_dir = deriv_dir;

  const auto cached = angular_momentum_term_cache_.find(cache_key);
  if (cached != angular_momentum_term_cache_.end()) {
    return cached->second;
  }

  double total_integral = 0.0;
  // Take derivative in specific direction
  auto deriv_components = gaussian_lib::get_gaussian_derivative(v, deriv_dir);

  // Loop through the returned derivatives
  for (const auto &deriv : deriv_components) {
    auto coord_components = gaussian_lib::multiply_by_coords(
        deriv.gaussian, coord_dir, gauge_origin[coord_dir]);

    // Get combined prefactors, multiply by overlap, add to total integral
    for (const auto &final_deriv : coord_components) {
      double combined_prefactor = deriv.prefactor * final_deriv.prefactor;
      double overlap = integrate_product(u, final_deriv.gaussian);
      total_integral += combined_prefactor * overlap;
    }
  }

  angular_momentum_term_cache_.emplace(cache_key, total_integral);
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
          calc_angular_momentum_term(u, v, gauge_origin, coord1, deriv1);

      double term_negative =
          calc_angular_momentum_term(u, v, gauge_origin, coord2, deriv2);

      M(i, j) = term_positive - term_negative;
    }
  }

  return M;
}

RealMat CNDO2SystemComplex::compute_shielding_operator_matrix(
    int direction, size_t target_proton_idx) const {
  RealMat shield_M(num_orbitals(), num_orbitals(), arma::fill::zeros);

  const Atom &proton_A = get_atom(target_proton_idx);

  for (size_t ib = 0; ib < num_atoms(); ++ib) {
    if (ib == target_proton_idx) {
      continue;
    }

    const Atom &atom_B = get_atom(ib);
    const auto &pos_B = atom_B.get_position();

    const double R_AB = distance(proton_A, atom_B);
    const double R3_inv = 1.0 / std::pow(R_AB, 3);

    const RealMat M_total = compute_angular_momentum_matrix(direction, pos_B);

    const auto &b_idxs = atom_orbital_idxs[ib];
    const arma::span b_span(b_idxs[0], b_idxs[1] - 1);

    shield_M(b_span, b_span) = M_total(b_span, b_span) * R3_inv;
  }

  return shield_M;
}

RealMat
CNDO2SystemComplex::compute_proton_shielding_tensor(size_t target_proton_idx,
                                                    double epsilon) {
  RealMat sigma_tensor(3, 3, arma::fill::zeros);
  constexpr int scf_max_iters = 1000;
  constexpr double scf_tol = 1e-6;

  ComplexMat p_alpha_unperturbed = get_p_alpha();
  ComplexMat p_beta_unperturbed = get_p_beta();

  // loop over magnetic field directions
  for (int dir = 0; dir < 3; ++dir) {
    // Positive perturbation
    set_magnetic_field(dir, epsilon);
    diis::solve_cndo(*this, scf_max_iters, scf_tol, true);
    // Get the total density after DIIS
    ComplexMat p_plus = p_alpha_ + p_beta_;

    // Reset for next run
    set_p(p_alpha_unperturbed, p_beta_unperturbed);

    // Negative perturbation
    set_magnetic_field(dir, -epsilon);
    diis::solve_cndo(*this, scf_max_iters, scf_tol, true);
    // Get the total density after DIIS
    ComplexMat p_minus = p_alpha_ + p_beta_;

    // Reset for next run
    set_p(p_alpha_unperturbed, p_beta_unperturbed);

    // Calculate the dertivative with respect to field
    RealMat dP_dB =
        central_difference_density_derivative_wrt_B(p_plus, p_minus, epsilon);

    // Contract with the shielding operator
    for (int response_dir = 0; response_dir < 3; ++response_dir) {
      RealMat H11 =
          compute_shielding_operator_matrix(response_dir, target_proton_idx);
      sigma_tensor(response_dir, dir) = arma::trace(dP_dB * H11);
    }
  }
  // Clean field state before exit
  set_magnetic_field(0, 0.0);

  return sigma_tensor;
}

} // namespace system_lib