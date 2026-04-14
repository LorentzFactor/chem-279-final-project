#include "orbitals.h"
#include "integrate.h"

#include <armadillo>
#include <cmath>
#include <fstream>
#include <highfive/H5File.hpp>
#include <iostream>
#include <map>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace fs = std::filesystem;

/**
 * @brief Load primitive Gaussian basis functions from JSON files on disk.
 *
 * Creates the basis function mapping used later for parsing atoms.
 * Structure is {atomic num: {momentum num: vec of primitive gaussians}}
 *
 */
std::map<int, std::map<int, std::vector<PrimitiveGaussian>>>
extract_basis(const fs::path &basis_path) {
  if (!fs::exists(basis_path)) {
    std::cerr << "Directory path: " << basis_path << " does not exist"
              << std::endl;
    // throw an error here
    throw std::runtime_error("Invalid basis set path error.");
  }

  std::map<int, std::map<int, std::vector<PrimitiveGaussian>>> basis_map;

  for (const auto &entry : fs::directory_iterator(basis_path)) {
    std::ifstream file_stream(entry.path());
    json basis = json::parse(file_stream);
    std::vector<PrimitiveGaussian> gaussians;
    int mom = basis["shell_momentum"];
    int num = basis["atomic_number"];
    double exp;
    double con;
    for (int i = 0; i < 3; ++i) {
      exp = basis["contracted_gaussians"][i]["exponent"];
      con = basis["contracted_gaussians"][i]["contraction_coefficient"];
      PrimitiveGaussian gauss(exp, con, mom, num);
      gaussians.push_back(gauss);
    }
    // basis_map.insert({mom, gaussians});
    basis_map[num][mom] = gaussians;
  }

  return basis_map;
}

/**
 * @brief Set each primitive's `norm` so the contracted AO is normalized.
 * @param atom AO whose `prim_gauss` overlaps are used to compute norms.
 */
void calculateNormalization(AtomicOrbital &atom) {
  for (int prim_i = 0; prim_i < atom.prim_gauss.size(); ++prim_i) {
    double overlap = calc3DOverlap(atom, atom.prim_gauss.at(prim_i), atom,
                                   atom.prim_gauss.at(prim_i));
    double normalization = 1 / std::sqrt(overlap);
    atom.prim_gauss[prim_i].norm = normalization;
  }
}

/**
 * @brief Construct a molecule and precompute the AO overlap matrix.
 *
 * Parses atoms and basis sets to build contracted AOs, computes the overlap
 * matrix \(S\), and initializes density and Fock matrix storage. The
 * CNDO/2 gamma matrix, core Hamiltonian, and SCF are performed in `SCF()`.
 *
 */
Molecule::Molecule(const fs::path &atoms_path, const fs::path &basis_path,
                   const int num_alpha, const int num_beta) {
  basis_funcs_ = parse_atoms(atoms_path, basis_path);
  N_basis_funcs_ = basis_funcs_.size();
  n_alpha_ = num_alpha;
  n_beta_ = num_beta;
  S_ = contractedOverlapMatrix(*this);
  P_total_.zeros(N_basis_funcs_, N_basis_funcs_);
  P_alpha_.zeros(N_basis_funcs_, N_basis_funcs_);
  P_beta_.zeros(N_basis_funcs_, N_basis_funcs_);
  gamma_.zeros(n_atoms_, n_atoms_);
  F_alpha_.zeros(N_basis_funcs_, N_basis_funcs_);
  F_beta_.zeros(N_basis_funcs_, N_basis_funcs_);
  H_core_.zeros(N_basis_funcs_, N_basis_funcs_);
  grad_overlap_term_.zeros(3, N_basis_funcs_ * N_basis_funcs_);
  grad_repulsion_term_.zeros(3, n_atoms_ * n_atoms_);
  zeroGradient(gradient_electronic_);
  zeroGradient(gradient_nuclear_);
  zeroGradient(gradient_total_);
}

// === SCF Pipeline ===

/**
 * @brief Build the spin density matrix from occupied molecular orbitals.
 *
 * Occupies the first `n_electrons` columns of `C` and forms
 * \(P = C_{occ} C_{occ}^T\).
 *
 * @param C Molecular orbital coefficient matrix.
 * @param n_electrons Number of occupied orbitals for this spin block.
 * @return Spin density matrix.
 */
arma::mat Molecule::solveDensity(const arma::mat &C, int n_electrons) {
  arma::mat C_occ =
      C.cols(0, n_electrons - 1); // update to take either alpha or beta e-
  arma::mat P = C_occ * C_occ.t();
  return P;
}

/**
 * @brief Compute the CNDO/2 two-center helper integral for s-like charge
 * clouds.
 *
 * @param aoA First orbital center.
 * @param aoB Second orbital center.
 * @param sigmaA Effective Gaussian width parameter for center A.
 * @param sigmaB Effective Gaussian width parameter for center B.
 * @return Integral contribution in atomic units.
 */
double Molecule::calc00(const AtomicOrbital &aoA, const AtomicOrbital &aoB,
                        double sigmaA, double sigmaB) {
  double result_00 = 0.0;
  double UA = std::pow(M_PI * sigmaA, 1.5);
  double UB = std::pow(M_PI * sigmaB, 1.5);
  double V2 = 1 / (sigmaA + sigmaB);
  double Rd = arma::norm(aoA.coords - aoB.coords, 2);
  if (std::abs(Rd) < 1e-12) {
    result_00 = (UA * UB) * 2 * std::sqrt(V2 / M_PI);
  } else {
    result_00 = (UA * UB) / Rd * std::erf(std::sqrt(V2) * Rd);
  }
  return result_00;
}

/**
 * @brief Compute contracted CNDO/2 \f$\gamma_{AB}\f$ between two s-orbitals.
 *
 * Performs the full primitive-pair contraction and converts to eV.
 *
 * @param aoA First atom-centered s-orbital.
 * @param aoB Second atom-centered s-orbital.
 * @return Contracted gamma value in electron-volts.
 */
double Molecule::calcGamma(const AtomicOrbital &aoA, const AtomicOrbital &aoB) {
  double ev_AU_conv = 27.211324570273;
  double gamma_val = 0.0;
  double sigmaA, sigmaB;
  double term_00, term_contract;
  double c_k, c_kprime, c_l, c_lprime;

  // calculate the 2-electron integral between AOs
  for (int idx_k = 0; idx_k < 3; ++idx_k) {
    for (int idx_kprime = 0; idx_kprime < 3; ++idx_kprime) {
      sigmaA = 1 / ((aoA.prim_gauss[idx_k].exponent +
                     aoA.prim_gauss[idx_kprime].exponent));
      for (int idx_l = 0; idx_l < 3; ++idx_l) {
        for (int idx_lprime = 0; idx_lprime < 3; ++idx_lprime) {
          sigmaB = 1 / ((aoB.prim_gauss[idx_l].exponent +
                         aoB.prim_gauss[idx_lprime].exponent));
          term_00 = calc00(aoA, aoB, sigmaA, sigmaB);
          c_k = aoA.prim_gauss[idx_k].contraction * aoA.prim_gauss[idx_k].norm;
          c_kprime = aoA.prim_gauss[idx_kprime].contraction *
                     aoA.prim_gauss[idx_kprime].norm;
          c_l = aoB.prim_gauss[idx_l].contraction * aoB.prim_gauss[idx_l].norm;
          c_lprime = aoB.prim_gauss[idx_lprime].contraction *
                     aoB.prim_gauss[idx_lprime].norm;
          term_contract = c_k * c_kprime * c_l * c_lprime;
          gamma_val += term_contract * term_00 * ev_AU_conv;
        }
      }
    }
  }
  return gamma_val;
}

/** @brief Assemble the full atom-pair gamma matrix used in CNDO/2. */
void Molecule::gammaMatrix() {
  int gamma_A_count = 0;
  for (int idx_A = 0; idx_A < N_basis_funcs_; ++idx_A) {
    int gamma_B_count = 0;
    if (basis_funcs_[idx_A].momentum_num != 0) {
      continue;
    }
    for (int idx_B = 0; idx_B < N_basis_funcs_; ++idx_B) {
      if (basis_funcs_[idx_B].momentum_num != 0) {
        continue;
      }
      double gamma_val = calcGamma(basis_funcs_[idx_A], basis_funcs_[idx_B]);
      gamma_(gamma_A_count, gamma_B_count) = gamma_val;
      gamma_B_count += 1;
    }
    gamma_A_count += 1;
  }
}

/**
 * @brief Compute Mulliken-like local population on a single atom.
 *
 * @param atom_id Atom index.
 * @param P Density matrix to sample.
 * @return Sum of on-atom diagonal AO densities.
 */
double Molecule::localAtomDensity(int atom_id, const arma::mat &P) {
  double local_density = 0.0;
  for (int i = 0; i < P.n_cols; ++i) {
    if (atom_id == basis_funcs_[i].atom_id) {
      local_density += P(i, i);
    }
  }
  return local_density;
}

/**
 * @brief Compute electrostatic interaction contribution for one atom.
 *
 * @param atom_id Atom whose environment term is requested.
 * @return Electrostatic CNDO/2 contribution for diagonal Fock assembly.
 */
double Molecule::electrostaticInteract(int atom_id) {
  double electro = 0.0;
  for (int i = 0; i < n_atoms_; ++i) {
    if (i != atom_id) {
      int C_atomic_num = id_to_atomic_num_[i];
      double C_density =
          localAtomDensity(i, P_total_) - V_.valence[C_atomic_num];
      C_density *= gamma_(atom_id, i);
      electro += C_density;
    }
  }
  return electro;
}

/**
 * @brief Compute one diagonal element of the spin-specific Fock matrix.
 *
 * @param u AO index.
 * @param P_self Density matrix for the current spin channel.
 * @return Diagonal Fock value.
 */
double Molecule::fockDiagonal(int u, const arma::mat &P_self) {
  double fock_energy = 0.0;
  // Get info from current atom in fock matrix
  int atomic_num = basis_funcs_[u].atomic_number;
  int momentum_num = basis_funcs_[u].momentum_num;
  int atom_id = basis_funcs_[u].atom_id;
  // Subtract the core energy for this atomic orbital
  fock_energy -= CE_.energy[atomic_num][momentum_num];
  // Calculate and add the electrostatic interaction of the AOs
  double local_electro =
      localAtomDensity(atom_id, P_total_) - V_.valence[atomic_num];
  fock_energy +=
      (local_electro - (P_self(u, u) - 0.5)) * gamma_(atom_id, atom_id);
  // Add the electrostatic interactions with all other atoms in molecule
  fock_energy += electrostaticInteract(atom_id);
  return fock_energy;
}

/**
 * @brief Compute one off-diagonal element of the spin-specific Fock matrix.
 *
 * @param u First AO index.
 * @param v Second AO index.
 * @param P_self Density matrix for the current spin channel.
 * @return Off-diagonal Fock value.
 */
double Molecule::fockOffDiagonal(int u, int v, const arma::mat &P_self) {
  double fock_energy = 0.0;
  int u_id = basis_funcs_[u].atom_id;
  int u_num = basis_funcs_[u].atomic_number;
  int v_id = basis_funcs_[v].atom_id;
  int v_num = basis_funcs_[v].atomic_number;

  fock_energy += 0.5 * (B_.binding[u_num] + B_.binding[v_num]) * S_(u, v);
  fock_energy -= P_self(u, v) * gamma_(u_id, v_id);

  return fock_energy;
}

/**
 * @brief Build the full spin-specific Fock matrix from a spin density matrix.
 *
 * @param F_self Output Fock matrix.
 * @param P_self Input spin density matrix.
 */
void Molecule::fockMatrix(arma::mat &F_self, const arma::mat &P_self) {

  for (int u = 0; u < N_basis_funcs_; ++u) {
    for (int v = 0; v < N_basis_funcs_; ++v) {
      if (u == v) {
        F_self(u, u) = fockDiagonal(u, P_self);
      } else {
        F_self(u, v) = fockOffDiagonal(u, v, P_self);
      }
    }
  }
}

/**
 * @brief Compute one diagonal core Hamiltonian element.
 * @param u AO index.
 * @return Core Hamiltonian diagonal value.
 */
double Molecule::coreHamiltonianDiagonal(int u) {
  double ham_energy = 0.0;
  // Get info from current atom in fock matrix
  int atomic_num = basis_funcs_[u].atomic_number;
  int momentum_num = basis_funcs_[u].momentum_num;
  int atom_id = basis_funcs_[u].atom_id;
  // Subtract the core energy for this atomic orbital
  ham_energy -= CE_.energy[atomic_num][momentum_num];

  ham_energy -= (V_.valence[atomic_num] - 0.5) * gamma_(atom_id, atom_id);

  for (int i = 0; i < n_atoms_; ++i) {
    if (i != atom_id) {
      ham_energy -= V_.valence[id_to_atomic_num_[i]] * gamma_(atom_id, i);
    }
  }

  return ham_energy;
}

/**
 * @brief Compute one off-diagonal core Hamiltonian element.
 * @param u First AO index.
 * @param v Second AO index.
 * @return Core Hamiltonian off-diagonal value.
 */
double Molecule::coreHamiltonianOffDiagonal(int u, int v) {
  double ham_energy = 0.0;
  int u_num = basis_funcs_[u].atomic_number;
  int v_num = basis_funcs_[v].atomic_number;

  ham_energy += 0.5 * (B_.binding[u_num] + B_.binding[v_num]) * S_(u, v);

  return ham_energy;
}

/** @brief Build the full core Hamiltonian matrix. */
void Molecule::coreHamiltonianMatrix() {
  for (int u = 0; u < N_basis_funcs_; ++u) {
    for (int v = 0; v < N_basis_funcs_; ++v) {
      if (u == v) {
        H_core_(u, u) = coreHamiltonianDiagonal(u);
      } else {
        H_core_(u, v) = coreHamiltonianOffDiagonal(u, v);
      }
    }
  }
}

/**
 * @brief Diagonalize a Fock matrix to obtain orbital energies and MOs.
 * @param F_self Real symmetric Fock matrix.
 * @param n_electrons Reserved for occupancy handling.
 * @return Orbital energies and coefficient matrix.
 */
MoleculeEnergy Molecule::solveEnergy(const arma::mat &F_self, int n_electrons) {
  MoleculeEnergy system;
  // solve the eigen problem
  arma::eig_sym(system.energies, system.C, F_self);

  return system;
}

/**
 * @brief Compute classical nuclear repulsion from current geometry.
 * @return Nuclear repulsion energy in eV.
 */
double Molecule::nuclearRepulsion() {
  double ev_AU_conv = 27.211324570273;
  double nuc_rep = 0.0;

  for (int A = 0; A < n_atoms_; ++A) {
    for (int B = 0; B < A; ++B) {
      int Z_A = V_.valence[id_to_atomic_num_[A]];
      int Z_B = V_.valence[id_to_atomic_num_[B]];
      double R_AB = arma::norm(id_to_coords_[A] - id_to_coords_[B], 2);
      nuc_rep += (Z_A * Z_B) / R_AB;
    }
  }

  return nuc_rep * ev_AU_conv;
}

/**
 * @brief Compute electronic energy from current densities and Fock/core terms.
 * @return Total electronic contribution in eV.
 */
double Molecule::electronEnergy() {
  double e_energy_alpha = 0.0;
  double e_energy_beta = 0.0;

  for (int u = 0; u < N_basis_funcs_; ++u) {
    for (int v = 0; v < N_basis_funcs_; ++v) {
      e_energy_alpha += P_alpha_(u, v) * (H_core_(u, v) + F_alpha_(u, v));
      e_energy_beta += P_beta_(u, v) * (H_core_(u, v) + F_beta_(u, v));
    }
  }

  return (0.5 * e_energy_alpha) + (0.5 * e_energy_beta);
}

/** @brief Store electronic, nuclear, and total CNDO/2 energy components. */
void Molecule::cndo2Energy() {
  CNDO2_.electron_energy = electronEnergy();
  CNDO2_.nuclear_repulsion = nuclearRepulsion();
  CNDO2_.total_energy = CNDO2_.electron_energy + CNDO2_.nuclear_repulsion;
}

/**
 * @brief Run unrestricted CNDO/2 SCF iterations to self-consistency.
 * @param verbose If true, print convergence and final energy information.
 */
void Molecule::SCF(bool verbose) {
  // Initialize gamma and the core hamiltonian
  gammaMatrix();
  coreHamiltonianMatrix();
  double converge_tol = 1.0e-6;
  arma::mat P_alpha_old;
  arma::mat P_beta_old;

  bool convergence = false;
  int iterations = 0;
  int iterations_max = 50; // Max iterations safety net
  while ((!convergence) && (iterations < iterations_max)) {

    // Calculate both alpha and beta fock matrices
    fockMatrix(F_alpha_, P_alpha_);
    fockMatrix(F_beta_, P_beta_);

    // Save initial fock matrices for test cases
    if (iterations == 0) {
      F_alpha_initial_ = F_alpha_;
      F_beta_initial_ = F_beta_;
    }

    // Solve the eigen problem for alpha and beta
    system_alpha_ = solveEnergy(F_alpha_, n_alpha_);
    system_beta_ = solveEnergy(F_beta_, n_beta_);

    // Copy the current densities to old
    P_alpha_old = P_alpha_;
    P_beta_old = P_beta_;

    // Determine new densities
    P_alpha_ = solveDensity(system_alpha_.C, n_alpha_);
    P_beta_ = solveDensity(system_beta_.C, n_beta_);

    // Update total density
    P_total_ = P_alpha_ + P_beta_;

    double dP_alpha = arma::abs(P_alpha_ - P_alpha_old).max();
    double dP_beta = arma::abs(P_beta_ - P_beta_old).max();

    // Check convergence of both alpha and beta
    bool alpha_converged =
        arma::approx_equal(P_alpha_, P_alpha_old, "absdiff", converge_tol);
    bool beta_converged =
        arma::approx_equal(P_beta_, P_beta_old, "absdiff", converge_tol);

    if (alpha_converged && beta_converged) {
      convergence = true;
      cndo2Energy(); // Store the final energies
      if (verbose) {
        std::cout << "Iterations to converge: " << iterations << '\n';
        std::cout << "Nuclear Repulsion Energy is " << CNDO2_.nuclear_repulsion
                  << " eV.\n";
        std::cout << "Electron Energy is " << CNDO2_.electron_energy
                  << " eV.\n";
        std::cout << "The molecule in file " << config_file_path_
                  << " has energy " << CNDO2_.total_energy << " eV.\n";
      }
    }

    iterations += 1;
  }
}

// === Analytic Gradient Terms ===

/**
 * @brief Zero all Cartesian components of a gradient container.
 * @param g Gradient vectors to reset.
 */
void Molecule::zeroGradient(Gradient &g) {
  g.x.zeros(n_atoms_);
  g.y.zeros(n_atoms_);
  g.z.zeros(n_atoms_);
}

/**
 * @brief Evaluate \f$\partial S_{uv}/\partial R_{u,\mathrm{dim}}\f$ for one
 * Cartesian direction.
 *
 * Performs primitive-pair contraction and multiplies by overlaps in the other
 * two Cartesian directions.
 *
 * @param dim Cartesian index (0=x, 1=y, 2=z).
 * @param u First atomic orbital.
 * @param v Second atomic orbital.
 * @return Directional derivative of the AO overlap integral.
 */
double Molecule::calcDerivative3D(int dim, AtomicOrbital &u, AtomicOrbital &v) {
  int num_prim_gauss = u.prim_gauss.size();
  int num_dims = 3;
  double total_dS_dDIM = 0.0;
  // Iterate through all prim gaussian pairs
  for (int k = 0; k < num_prim_gauss; ++k) {
    for (int l = 0; l < num_prim_gauss; ++l) {
      // Grab a reference to current primitive gaussians
      PrimitiveGaussian &u_p = u.prim_gauss.at(k);
      PrimitiveGaussian &v_p = v.prim_gauss.at(l);

      int l_A = u.momentum.at(dim);
      int l_B = v.momentum.at(dim);

      // Calculate the center between these two primitives
      arma::rowvec center = gaussianCenter(u, v, u_p.exponent, v_p.exponent);
      // Calculate the derivative for this specific dimension
      double term1 = -l_A * calcTotal1DOverlap(
                                u_p.exponent, v_p.exponent, u.coords.at(dim),
                                v.coords.at(dim), center.at(dim), l_A - 1, l_B);
      double term2 =
          2 * u_p.exponent *
          calcTotal1DOverlap(u_p.exponent, v_p.exponent, u.coords.at(dim),
                             v.coords.at(dim), center.at(dim), l_A + 1, l_B);
      double dS_dDIM = term1 + term2;

      double dS_3D_dDIM = dS_dDIM;
      // Iterate through the other dimensions and calculate total 1D overlap
      // Multiply overlap with dS in current dimension evaluating
      for (int xyz = 0; xyz < num_dims; ++xyz) {
        if (xyz == dim)
          continue;
        dS_3D_dDIM *= calcTotal1DOverlap(
            u_p.exponent, v_p.exponent, u.coords.at(xyz), v.coords.at(xyz),
            center.at(xyz), u.momentum.at(xyz), v.momentum.at(xyz));
      }

      // Get constants term for these prim gaussians
      double constants =
          u_p.contraction * v_p.contraction * u_p.norm * v_p.norm;

      total_dS_dDIM += constants * dS_3D_dDIM;
    }
  }
  return total_dS_dDIM;
}

/**
 * @brief Accumulate the overlap-driven contribution to the electronic gradient.
 *
 * Fills the cached overlap derivative tensor for an atom pair and applies
 * equal-and-opposite force updates to both atoms.
 *
 * @param id_A First atom index.
 * @param id_B Second atom index.
 */
void Molecule::gradOverlapTerm(int id_A, int id_B) {

  // pack up the AOs for each Atom
  std::vector<globalAO> &AOs_A = id_to_global_AO_[id_A];
  std::vector<globalAO> &AOs_B = id_to_global_AO_[id_B];

  double bind_A = B_.binding[id_to_atomic_num_[id_A]];
  double bind_B = B_.binding[id_to_atomic_num_[id_B]];

  int x = 0, y = 1, z = 2;
  for (int u = 0; u < AOs_A.size(); ++u) {
    for (int v = 0; v < AOs_B.size(); ++v) {
      int global_u = AOs_A.at(u).globalIDX;
      int global_v = AOs_B.at(v).globalIDX;
      double x_uv = P_total_.at(global_u, global_v) * (bind_A + bind_B);

      double dS_x =
          calcDerivative3D(x, AOs_A.at(u).globalAO, AOs_B.at(v).globalAO);
      double dS_y =
          calcDerivative3D(y, AOs_A.at(u).globalAO, AOs_B.at(v).globalAO);
      double dS_z =
          calcDerivative3D(z, AOs_A.at(u).globalAO, AOs_B.at(v).globalAO);

      // Save results to grad overlap matrix term using global idxs
      grad_overlap_term_.at(x, ((global_u * N_basis_funcs_) + global_v)) = dS_x;
      grad_overlap_term_.at(y, ((global_u * N_basis_funcs_) + global_v)) = dS_y;
      grad_overlap_term_.at(z, ((global_u * N_basis_funcs_) + global_v)) = dS_z;
      // Apply translational invariance as well, swap global idxs
      grad_overlap_term_.at(x, ((global_v * N_basis_funcs_) + global_u)) =
          -dS_x;
      grad_overlap_term_.at(y, ((global_v * N_basis_funcs_) + global_u)) =
          -dS_y;
      grad_overlap_term_.at(z, ((global_v * N_basis_funcs_) + global_u)) =
          -dS_z;

      // Force on Atom A
      gradient_electronic_.x.at(id_A) += x_uv * dS_x;
      gradient_electronic_.y.at(id_A) += x_uv * dS_y;
      gradient_electronic_.z.at(id_A) += x_uv * dS_z;

      // Equal and opposite force on Atom B
      gradient_electronic_.x.at(id_B) -= x_uv * dS_x;
      gradient_electronic_.y.at(id_B) -= x_uv * dS_y;
      gradient_electronic_.z.at(id_B) -= x_uv * dS_z;
    }
  }
}

/**
 * @brief Compute the derivative of the CNDO/2 \f$(00|00)\f$ helper integral.
 *
 * @param aoA First orbital center.
 * @param aoB Second orbital center.
 * @param sigmaA Effective Gaussian width parameter for center A.
 * @param sigmaB Effective Gaussian width parameter for center B.
 * @return Cartesian derivative vector of the helper integral.
 */
arma::rowvec Molecule::calc00Derivative(const AtomicOrbital &aoA,
                                        const AtomicOrbital &aoB, double sigmaA,
                                        double sigmaB) {
  arma::rowvec deriv_00(3);
  int num_dims = 3;
  double UA = std::pow(M_PI * sigmaA, 1.5);
  double UB = std::pow(M_PI * sigmaB, 1.5);
  double V2 = 1 / (sigmaA + sigmaB);
  double V = std::sqrt(V2);

  arma::rowvec R_diff = aoA.coords - aoB.coords;
  double R_dist = arma::norm(R_diff);
  double T = V2 * (R_dist * R_dist);
  arma::rowvec term1;
  double term2;
  if (R_dist < 1e-12) {
    term1 = 0.0;
    term2 = 0.0;
  } else {
    term1 = (UA * UB * R_diff) / (R_dist * R_dist);
    term2 = -std::erf(std::sqrt(T)) / R_dist;
  }
  double term3 = ((2 * V) / std::sqrt(M_PI)) * std::exp(-T);
  deriv_00 = term1 * (term2 + term3);

  return deriv_00;
}

/**
 * @brief Compute Cartesian derivatives of contracted CNDO/2
 * \f$\gamma_{AB}\f$.
 *
 * Performs full primitive-pair contraction and returns the derivative vector
 * in eV.
 *
 * @param aoA First atom-centered s-orbital.
 * @param aoB Second atom-centered s-orbital.
 * @return \f$(d\gamma/dx, d\gamma/dy, d\gamma/dz)\f$ for atom-pair AB.
 */
arma::rowvec Molecule::calcGammaDerivative(const AtomicOrbital &aoA,
                                           const AtomicOrbital &aoB) {
  double ev_AU_conv = 27.211324570273;
  double gamma_val = 0.0;
  double sigmaA, sigmaB;
  double term_contract;
  arma::rowvec term_00(3);
  double c_k, c_kprime, c_l, c_lprime;

  arma::rowvec deriv_gamma(3);

  int x = 0, y = 1, z = 2;

  // calculate the 2-electron integral between AOs
  for (int idx_k = 0; idx_k < 3; ++idx_k) {
    for (int idx_kprime = 0; idx_kprime < 3; ++idx_kprime) {
      sigmaA = 1 / ((aoA.prim_gauss[idx_k].exponent +
                     aoA.prim_gauss[idx_kprime].exponent));
      for (int idx_l = 0; idx_l < 3; ++idx_l) {
        for (int idx_lprime = 0; idx_lprime < 3; ++idx_lprime) {
          sigmaB = 1 / ((aoB.prim_gauss[idx_l].exponent +
                         aoB.prim_gauss[idx_lprime].exponent));
          term_00 = calc00Derivative(aoA, aoB, sigmaA, sigmaB);
          c_k = aoA.prim_gauss[idx_k].contraction * aoA.prim_gauss[idx_k].norm;
          c_kprime = aoA.prim_gauss[idx_kprime].contraction *
                     aoA.prim_gauss[idx_kprime].norm;
          c_l = aoB.prim_gauss[idx_l].contraction * aoB.prim_gauss[idx_l].norm;
          c_lprime = aoB.prim_gauss[idx_lprime].contraction *
                     aoB.prim_gauss[idx_lprime].norm;
          term_contract = c_k * c_kprime * c_l * c_lprime;
          deriv_gamma.at(x) += term_contract * term_00.at(x) * ev_AU_conv;
          deriv_gamma.at(y) += term_contract * term_00.at(y) * ev_AU_conv;
          deriv_gamma.at(z) += term_contract * term_00.at(z) * ev_AU_conv;
        }
      }
    }
  }
  return deriv_gamma;
}

/**
 * @brief Accumulate the gamma/repulsion contribution to the electronic
 * gradient.
 *
 * Builds the CNDO/2 \f$Y_{AB}\f$ prefactor from atomic populations, evaluates
 * \f$\nabla\gamma_{AB}\f$, stores the pair derivative cache, and applies
 * equal-and-opposite force updates.
 *
 * @param id_A First atom index.
 * @param id_B Second atom index.
 */
void Molecule::gradRepulsionTerm(int id_A, int id_B) {
  // pack up the AOs for each Atom
  std::vector<globalAO> &AOs_A = id_to_global_AO_[id_A];
  std::vector<globalAO> &AOs_B = id_to_global_AO_[id_B];

  // Get global idx start and end for grabbing correct densities
  int A_start_idx = AOs_A.front().globalIDX;
  int A_end_idx = AOs_A.back().globalIDX;
  int B_start_idx = AOs_B.front().globalIDX;
  int B_end_idx = AOs_B.back().globalIDX;

  // Take submatrix of total density for just atom A or B densities
  // Trace to sum across diagonal
  double P_AA_tot = arma::trace(
      P_total_.submat(A_start_idx, A_start_idx, A_end_idx, A_end_idx));
  double P_BB_tot = arma::trace(
      P_total_.submat(B_start_idx, B_start_idx, B_end_idx, B_end_idx));

  // Perform the double summation for u and v
  // u belonging to A and v belonging to B
  double density_sum = 0.0;
  for (int u = 0; u < AOs_A.size(); ++u) {
    for (int v = 0; v < AOs_B.size(); ++v) {
      int global_u = AOs_A.at(u).globalIDX;
      int global_v = AOs_B.at(v).globalIDX;

      double p_alpha =
          P_alpha_.at(global_u, global_v) * P_alpha_.at(global_v, global_u);
      double p_beta =
          P_beta_.at(global_u, global_v) * P_beta_.at(global_v, global_u);

      density_sum += p_alpha + p_beta;
    }
  }

  // Calculate Y_AB
  int atomic_num_A = id_to_atomic_num_.at(id_A);
  int atomic_num_B = id_to_atomic_num_.at(id_B);
  double Z_A = V_.valence.at(atomic_num_A);
  double Z_B = V_.valence.at(atomic_num_B);
  double y_term1 = P_AA_tot * P_BB_tot;
  double y_term2 = -(Z_B * P_AA_tot) - (Z_A * P_BB_tot);
  double y_AB = y_term1 + y_term2 - density_sum;

  AtomicOrbital s_orbital_A = id_to_global_AO_.at(id_A).front().globalAO;
  AtomicOrbital s_orbital_B = id_to_global_AO_.at(id_B).front().globalAO;

  arma::mat gamma_deriv = calcGammaDerivative(s_orbital_A, s_orbital_B);

  int flat_idx_AB = (id_A * n_atoms_) + id_B;
  int flat_idx_BA = (id_B * n_atoms_) + id_A;
  int x = 0, y = 1, z = 2;

  // Save results to grad overlap matrix term using global idxs
  grad_repulsion_term_.at(x, flat_idx_AB) = gamma_deriv.at(x);
  grad_repulsion_term_.at(y, flat_idx_AB) = gamma_deriv.at(y);
  grad_repulsion_term_.at(z, flat_idx_AB) = gamma_deriv.at(z);
  // Apply translational invariance as well, swap global idxs
  grad_repulsion_term_.at(x, flat_idx_BA) = -gamma_deriv.at(x);
  grad_repulsion_term_.at(y, flat_idx_BA) = -gamma_deriv.at(y);
  grad_repulsion_term_.at(z, flat_idx_BA) = -gamma_deriv.at(z);

  // Repulsion on Atom A
  gradient_electronic_.x.at(id_A) += y_AB * gamma_deriv.at(x);
  gradient_electronic_.y.at(id_A) += y_AB * gamma_deriv.at(y);
  gradient_electronic_.z.at(id_A) += y_AB * gamma_deriv.at(z);

  // Equal and opposite repulsion on Atom B
  gradient_electronic_.x.at(id_B) -= y_AB * gamma_deriv.at(x);
  gradient_electronic_.y.at(id_B) -= y_AB * gamma_deriv.at(y);
  gradient_electronic_.z.at(id_B) -= y_AB * gamma_deriv.at(z);
}

/**
 * @brief Assemble the full electronic force contribution.
 *
 * Loops over unique atom pairs and accumulates both repulsion and overlap
 * terms.
 */
void Molecule::electronicGradient() {
  for (int id_A = 0; id_A < n_atoms_; ++id_A) {
    for (int id_B = id_A + 1; id_B < n_atoms_; ++id_B) {
      gradRepulsionTerm(id_A, id_B);
      gradOverlapTerm(id_A, id_B);
    }
  }
}

/**
 * @brief Compute derivative of classical nuclear repulsion for one atom pair.
 *
 * @param RA Coordinates of atom A.
 * @param RB Coordinates of atom B.
 * @param ZA Valence charge on atom A.
 * @param ZB Valence charge on atom B.
 * @return Cartesian derivative vector in eV/bohr.
 */
arma::rowvec Molecule::nucRepulsionDeriv(arma::rowvec &RA, arma::rowvec &RB,
                                         double ZA, double ZB) {
  double ev_AU_conv = 27.211324570273;
  arma::rowvec nuc_repulse_result;
  nuc_repulse_result.zeros(3);
  arma::rowvec R_diff = RA - RB;
  double R_dist = arma::norm(R_diff);
  double R_dist_cubed = R_dist * R_dist * R_dist;
  if (R_dist < 1e-12) {
    return nuc_repulse_result;
  }
  nuc_repulse_result = -ZA * ZB * (R_diff / R_dist_cubed);
  return nuc_repulse_result * ev_AU_conv;
}

/**
 * @brief Assemble the full nuclear repulsion force contribution.
 *
 * Loops over unique atom pairs and applies equal-and-opposite nuclear forces.
 */
void Molecule::nuclearGradient() {
  int x = 0, y = 1, z = 2;
  for (int id_A = 0; id_A < n_atoms_; ++id_A) {
    for (int id_B = id_A + 1; id_B < n_atoms_; ++id_B) {
      // Get all necessary terms by atom IDs
      int A_atomic_num = id_to_atomic_num_.at(id_A);
      int B_atomic_num = id_to_atomic_num_.at(id_B);
      arma::rowvec RA = id_to_coords_.at(id_A);
      arma::rowvec RB = id_to_coords_.at(id_B);
      double ZA = V_.valence.at(A_atomic_num);
      double ZB = V_.valence.at(B_atomic_num);

      arma::rowvec nuc_rep = nucRepulsionDeriv(RA, RB, ZA, ZB);

      // Repulsion on Atom A
      gradient_nuclear_.x.at(id_A) += nuc_rep.at(x);
      gradient_nuclear_.y.at(id_A) += nuc_rep.at(y);
      gradient_nuclear_.z.at(id_A) += nuc_rep.at(z);

      // Equal and opposite repulsion on Atom B
      gradient_nuclear_.x.at(id_B) -= nuc_rep.at(x);
      gradient_nuclear_.y.at(id_B) -= nuc_rep.at(y);
      gradient_nuclear_.z.at(id_B) -= nuc_rep.at(z);
    }
  }
}

/** @brief Sum electronic and nuclear force components into total gradient. */
void Molecule::totalGradient() {
  gradient_total_.x = gradient_electronic_.x + gradient_nuclear_.x;
  gradient_total_.y = gradient_electronic_.y + gradient_nuclear_.y;
  gradient_total_.z = gradient_electronic_.z + gradient_nuclear_.z;
}

/**
 * @brief Convert gradient component vectors into a stacked matrix.
 * @param g Gradient container with x/y/z vectors.
 * @return Matrix \f$[g_x; g_y; g_z]\f$.
 */
arma::mat Molecule::gradVecsToMat(Gradient &g) {
  arma::mat grad_mat;
  grad_mat = arma::join_vert(g.x, arma::join_vert(g.y, g.z));
  return grad_mat;
}

/**
 * @brief Recompute SCF energy and all analytic force components.
 *
 * Resets gradient and density storage, runs SCF, then assembles electronic,
 * nuclear, and total gradients.
 *
 * @return Current total CNDO/2 energy in eV.
 */
double Molecule::calcEnergyAndForces() {

  // Zero all the gradients to start fresh
  zeroGradient(gradient_electronic_);
  zeroGradient(gradient_nuclear_);
  zeroGradient(gradient_total_);

  S_ = contractedOverlapMatrix(*this);

  // Reset density matrices
  P_total_.zeros(N_basis_funcs_, N_basis_funcs_);
  P_alpha_.zeros(N_basis_funcs_, N_basis_funcs_);
  P_beta_.zeros(N_basis_funcs_, N_basis_funcs_);

  // Perform SCF and calculate gradients
  // SCF call initially recalculates gamma and core hamiltonian
  this->SCF(false);
  this->electronicGradient();
  this->nuclearGradient();
  this->totalGradient();
  // Return the total energy after SCF
  return this->CNDO2_.total_energy;
}

// === Geometry Optimization ===

/** @brief Return a value copy of the current molecule object. */
Molecule Molecule::copyMolecule() {
  Molecule copy(*this);
  return copy;
}

/**
 * @brief Optimize geometry by adaptive steepest-descent steps.
 *
 * Proposes coordinate updates from the current force, accepts steps that lower
 * energy, and adapts step size until the force threshold is reached or the
 * iteration cap is hit.
 *
 * @param step Initial displacement scaling factor.
 * @param force_threshold Convergence threshold on maximum force component.
 */
void Molecule::steepestDescentOptimizer(double step, double force_threshold) {
  Molecule temp_mol = this->copyMolecule();
  bool converged = false;
  int counter = 0;
  int max_iterations = 1000;

  double E_new = 0.0;
  double E_old = calcEnergyAndForces();

  std::cout << "Initial energy: " << E_old << '\n';

  while ((!converged) && (counter < max_iterations)) {

    // sync coordinate storage locations
    temp_mol.basis_funcs_ = this->basis_funcs_;
    temp_mol.id_to_coords_ = this->id_to_coords_;
    temp_mol.id_to_global_AO_ = this->id_to_global_AO_;

    double global_max_force = 0.0;

    // iterate through each atom and determine potential coordinates
    for (int atom_id = 0; atom_id < n_atoms_; ++atom_id) {
      double xforce = -gradient_total_.x.at(atom_id);
      double yforce = -gradient_total_.y.at(atom_id);
      double zforce = -gradient_total_.z.at(atom_id);

      double local_max_force =
          std::max({std::abs(xforce), std::abs(yforce), std::abs(zforce)});
      global_max_force = std::max(global_max_force, local_max_force);

      // Iterate through all AOs and update temp mols coords
      // If condition met will copy temp coords into this molecule
      int x = 0, y = 1, z = 2;

      // Sync nuclei coords used by nuclear repulsion calculation
      temp_mol.id_to_coords_.at(atom_id).at(x) =
          this->id_to_coords_.at(atom_id).at(x) + (xforce * step);
      temp_mol.id_to_coords_.at(atom_id).at(y) =
          this->id_to_coords_.at(atom_id).at(y) + (yforce * step);
      temp_mol.id_to_coords_.at(atom_id).at(z) =
          this->id_to_coords_.at(atom_id).at(z) + (zforce * step);

      std::vector<globalAO> AOs = id_to_global_AO_.at(atom_id);
      for (int i = 0; i < AOs.size(); ++i) {
        int global_idx = AOs.at(i).globalIDX;
        AtomicOrbital &AO = this->basis_funcs_.at(global_idx);
        AtomicOrbital &AO_temp = temp_mol.basis_funcs_.at(global_idx);

        AO_temp.coords.at(x) = AO.coords.at(x) + (xforce * step);
        AO_temp.coords.at(y) = AO.coords.at(y) + (yforce * step);
        AO_temp.coords.at(z) = AO.coords.at(z) + (zforce * step);

        // Update globalAO copies used by analytical gradients
        temp_mol.id_to_global_AO_.at(atom_id).at(i).globalAO.coords =
            AO_temp.coords;
      }
    }

    if (global_max_force < force_threshold) {
      converged = true;
      std::cout << "Final Energy after " << counter << " iterations: " << E_old
                << '\n';
      break;
    }

    // determine new energy from potential coords
    // only update actual coords if energy decreased
    // increase step size by fixed amount in this case
    E_new = temp_mol.calcEnergyAndForces();
    if (E_new < E_old) {
      this->basis_funcs_ = temp_mol.basis_funcs_;
      this->id_to_coords_ = temp_mol.id_to_coords_;
      this->id_to_global_AO_ = temp_mol.id_to_global_AO_;
      this->gradient_total_ = temp_mol.gradient_total_;
      step *= 1.2;
      E_old = E_new;
      // if energy increased, only decrease step size
    } else {
      step *= 0.8;
    }

    ++counter;
  }
}

/** @brief Print atom atomic-numbers and Cartesian coordinates. */
void Molecule::printCoords() {
  int x = 0, y = 1, z = 2;
  for (int atom_id = 0; atom_id < n_atoms_; ++atom_id) {
    arma::rowvec &coords = id_to_coords_.at(atom_id);
    int atomic_num = id_to_atomic_num_.at(atom_id);
    std::cout << atomic_num << ' ' << coords.at(x) << ' ' << coords.at(y) << ' '
              << coords.at(z) << '\n';
  }
}

/**
 * @brief Print simple geometric observables from current coordinates.
 *
 * Reports the first A-B bond length and, when available, a second A-B bond
 * length plus the included bond angle.
 */
void Molecule::geometricProperties() {
  double bohr_to_angstrom = 0.529177;
  std::cout << "Geometry parameters:\n";

  arma::rowvec R_A = id_to_coords_.at(0);
  arma::rowvec R_B1 = id_to_coords_.at(1);

  // Calculate first bond length
  arma::rowvec vec_A_B1 = R_B1 - R_A;
  double bond_length_1 = arma::norm(vec_A_B1);
  std::cout << "A-B1 bond length (A): " << bond_length_1 * bohr_to_angstrom
            << '\n';

  if (n_atoms_ >= 3) {
    arma::rowvec R_B2 = id_to_coords_.at(2);
    // calculate the bond lengths
    arma::rowvec vec_A_B2 = R_B2 - R_A;
    double bond_length_2 = arma::norm(vec_A_B2);

    // use dot product to calculate the bond angle
    double dot_product = arma::dot(vec_A_B1, vec_A_B2);
    double cos_theta = dot_product / (bond_length_1 * bond_length_2);

    // convert radians to degrees
    double bond_angle_rad = std::acos(cos_theta);
    double bond_angle_deg = bond_angle_rad * (180.0 / M_PI);

    std::cout << "A-B2 bond length (A): " << bond_length_2 * bohr_to_angstrom
              << '\n';
    std::cout << "Bond angle (degrees): " << bond_angle_deg << '\n';
  }
}

// === Parsing Utilities ===

/**
 * @brief Parse atoms and build atomic orbitals from coordinate and basis files.
 *
 * Generates the vector of AOs that will be stored in the Molecule class.
 * Called from the Molecule class during construction.
 *
 */
std::vector<AtomicOrbital> Molecule::parse_atoms(const fs::path &atoms_path,
                                                 const fs::path &basis_path) {
  std::vector<AtomicOrbital> atoms;
  std::ifstream file(atoms_path);
  std::map<int, std::map<int, std::vector<PrimitiveGaussian>>> basis_map =
      extract_basis(basis_path);

  if (!file.is_open()) {
    std::cerr << "Error: could not open " << atoms_path << '\n';
    return atoms;
  }

  std::string line;
  int num_atoms = 0;

  if (std::getline(file, line)) {
    std::stringstream ss(line);
    ss >> num_atoms;
    this->n_atoms_ = num_atoms;
  }

  // skip comment line
  std::getline(file, line);

  // parse atom data
  int atom_id = 0;
  for (int i = 0; i < num_atoms; ++i) {
    double x, y, z;
    int atomic_number;
    if (std::getline(file, line)) {
      std::stringstream ss(line);
      ss >> atomic_number >> x >> y >> z;

      for (const auto &pair : basis_map[atomic_number]) {
        std::vector<PrimitiveGaussian> basis = pair.second;
        Shells shell(pair.first); // momentum from basis map key to get shell
        for (arma::uword i = 0; i < shell.L.n_rows; ++i) {
          arma::rowvec momentum = shell.L.row(i);
          // pair.second is vector of three primitive gaussians
          AtomicOrbital atom(atomic_number, atom_id, x, y, z, pair.first,
                             momentum, basis);
          calculateNormalization(atom);
          // Track global AOs mapped to atom_id
          int global_AO_idx = atoms.size();
          globalAO gao = {global_AO_idx, atom};
          id_to_global_AO_[atom_id].push_back(gao);
          atoms.push_back(atom);
        }
      }
      id_to_atomic_num_[atom_id] = atomic_number;
      id_to_coords_[atom_id] = arma::rowvec{x, y, z};
      atom_id += 1;
    }
  }
  return atoms;
}