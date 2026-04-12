#ifndef ORBITALS_H
#define ORBITALS_H

#include <armadillo>
#include <map>
#include <nlohmann/json.hpp>
#include <unordered_map>

namespace fs = std::filesystem;
using json = nlohmann::json;

/**
 * @brief Cartesian angular-momentum patterns for s, p, or d shells.
 *
 * `L` holds one row per basis function in the shell (exponents of x, y, z in
 * the Cartesian Gaussian). `momentum_number` is the shell angular momentum
 * (0, 1, or 2). The constructor argument `num` is that same shell index
 * (s=0, p=1, d=2).
 */
struct Shells {
  /** @brief Rows are \((l_x,l_y,l_z)\) for each function in the shell. */
  arma::mat L;
  /** @brief Shell angular momentum quantum number. */
  int momentum_number;
  Shells(int num) : momentum_number(num) {
    switch (momentum_number) {
    case 0: {
      L = {{0, 0, 0}};
      break;
    }
    case 1: {
      L = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
      break;
    }
    case 2: {
      L = {{2, 0, 0}, {1, 1, 0}, {1, 0, 1}, {0, 2, 0}, {0, 1, 1}, {0, 0, 2}};
      break;
    }
    }
  }
};

/**
 * @brief One-dimensional Gaussian atomic orbital.
 */
struct Gaussian1DAO {
  double centroid;
  double exponent;
  double momentum;

  /**
   * @brief Construct a 1D Gaussian AO.
   * @param cen Centroid.
   * @param exp Exponent.
   * @param mom Momentum.
   */
  Gaussian1DAO(double cen, double exp, double mom)
      : centroid(cen), exponent(exp), momentum(mom) {}
};

/**
 * @brief One primitive in a contracted STO-NG (or similar) Gaussian expansion.
 */
struct PrimitiveGaussian {
  /** @brief Gaussian exponent \(\alpha\). */
  double exponent;
  /** @brief Contraction coefficient in the contracted AO. */
  double contraction;
  /** @brief Shell angular momentum (same as basis shell key). */
  int momentum;
  /** @brief Atomic number \(Z\) of the center. */
  int atomic_number;
  /** @brief Normalization constant for this primitive (set after overlap). */
  double norm;

  /**
   * @brief Construct a primitive with exponent, coefficient, shell, and
   * element.
   * @param exp Exponent.
   * @param con Contraction coefficient.
   * @param mom Shell angular momentum index.
   * @param num Atomic number.
   */
  PrimitiveGaussian(double exp, double con, int mom, int num)
      : exponent(exp), contraction(con), momentum(mom), atomic_number(num) {}
};

/**
 * @brief Load primitive Gaussian basis functions from disk.
 *
 * @param basis_path Path to directory containing basis set files.
 * @return Nested map keyed by atomic number and angular momentum.
 */
std::map<int, std::map<int, std::vector<PrimitiveGaussian>>>
extract_basis(const fs::path &basis_path);

/**
 * @brief CNDO/2 empirical core (atomic) orbital energies \(U_{\mu\mu}\) (eV).
 *
 * Outer key: atomic number; inner key: shell type (0=s, 1=p, ...).
 */
struct CoreEnergy {
  std::unordered_map<int, std::unordered_map<int, double>> energy = {
      {1, {{0, 7.176}}},                // H
      {6, {{0, 14.051}, {1, 5.572}}},   // C
      {7, {{0, 19.316}, {1, 7.275}}},   // N
      {8, {{0, 25.390}, {1, 9.111}}},   // O
      {9, {{0, 32.272}, {1, 11.080}}}}; // F
};

/**
 * @brief CNDO/2 atomic binding parameters used in off-diagonal core/Fock
 * elements (half-sum \(\beta_A^0 + \beta_B^0\) style terms).
 */
struct AtomicBinding {
  std::unordered_map<int, double> binding = {
      {1, -9.0}, {6, -21.0}, {7, -25.0}, {8, -31.0}, {9, -39.0}};
};

/**
 * @brief Valence electron counts per element for CNDO/2 charge bookkeeping.
 */
struct ValenceElectrons {
  std::unordered_map<int, int> valence = {
      {1, 1}, {6, 4}, {7, 5}, {8, 6}, {9, 7}};
};

/**
 * @brief CNDO/2 energy components (eV) after a converged SCF cycle.
 */
struct CNDO2Energy {
  /** @brief Classical nuclear repulsion in eV. */
  double nuclear_repulsion;
  /** @brief Electronic energy
   * \(\frac12\sum_{\sigma}\mathrm{Tr}\,P_\sigma(H_\mathrm{core}+F_\sigma)\) in
   * eV. */
  double electron_energy;
  /** @brief `electron_energy` + `nuclear_repulsion`. */
  double total_energy;
};

/**
 * @brief Result of diagonalizing a Fock matrix (orbital energies and MOs).
 *
 * Returned by `Molecule::solveEnergy`. `energies` are the eigenvalues
 * (orbital energies); `C` holds the MO coefficients (columns are MOs).
 */
struct MoleculeEnergy {
  /** @brief Orbital eigenvalues from `arma::eig_sym`. */
  arma::vec energies;
  // arma::mat C_prime;
  // double total_energy;
  /** @brief Molecular orbital coefficient matrix. */
  arma::mat C;
};

struct Gradient {
  arma::rowvec x;
  arma::rowvec y;
  arma::rowvec z;
};

/**
 * @brief Contracted atomic orbital (single Cartesian component) on a center.
 */
struct AtomicOrbital {
  /** @brief Cartesian coordinates of the nucleus (bohr). */
  double x, y, z;
  /** @brief Atomic number. */
  int atomic_number;
  /** @brief Index of this atom in the molecule. */
  int atom_id;
  /** @brief Shell angular momentum (0=s, 1=p, ...). */
  int momentum_num;
  /** @brief Powers \((l_x,l_y,l_z)\) for this Cartesian Gaussian. */
  arma::rowvec momentum;
  /** @brief Same as (x,y,z); used for vectorized distance formulas. */
  arma::rowvec coords;
  /** @brief Primitives shared by all Cartesian functions in this shell. */
  std::vector<PrimitiveGaussian> prim_gauss;

  /**
   * @brief Build one AO on an atom with a fixed Cartesian angular part.
   * @param num Atomic number.
   * @param id Atom index.
   * @param xx Nuclear x coordinate.
   * @param yy Nuclear y coordinate.
   * @param zz Nuclear z coordinate.
   * @param mom_num Shell momentum index.
   * @param mom Row of angular powers for this basis function.
   * @param gaus Primitives for the shell (STO-NG contraction).
   */
  AtomicOrbital(int num, int id, double xx, double yy, double zz, int mom_num,
                arma::rowvec mom, std::vector<PrimitiveGaussian> gaus)
      : atomic_number(num), atom_id(id), x(xx), y(yy), z(zz),
        momentum_num(mom_num), momentum(mom), prim_gauss(gaus) {
    coords = {x, y, z};
  }
};

/**
 * @brief Symmetric orthogonalization data for the overlap matrix.
 *
 * Used to hold return values generated during process of construction the
 * orthogonal transformation matrix, X.
 *
 */
struct SValVec {
  arma::vec eigval;
  arma::mat eigvec;
  arma::vec inv_sqrt_eigval;
  arma::mat X;
};

struct globalAO {
  int globalIDX;
  AtomicOrbital globalAO;
};

/**
 * @brief CNDO/2 Molecule built from atomic orbitals.
 *
 * CNDO/2 Theory used to calculate all quantum properties of a
 * molecule. Energy minimized using SCF.
 *
 */
class Molecule {
private:
  std::vector<AtomicOrbital> basis_funcs_;
  std::unordered_map<int, int> id_to_atomic_num_;
  std::unordered_map<int, arma::rowvec> id_to_coords_;
  std::unordered_map<int, std::vector<globalAO>> id_to_global_AO_;
  CoreEnergy CE_;
  AtomicBinding B_;
  ValenceElectrons V_;

  int N_basis_funcs_;
  int n_atoms_;
  int n_alpha_;
  int n_beta_;
  arma::mat S_;
  SValVec ortho_;
  arma::mat orthoH_;
  MoleculeEnergy system_alpha_;
  MoleculeEnergy system_beta_;
  arma::mat P_total_;
  arma::mat P_alpha_;
  arma::mat P_beta_;
  arma::mat gamma_;
  arma::mat F_alpha_;
  arma::mat F_beta_;
  arma::mat F_alpha_initial_;
  arma::mat F_beta_initial_;
  arma::mat H_core_;
  CNDO2Energy CNDO2_;
  arma::mat grad_overlap_term_;
  arma::mat grad_repulsion_term_;
  Gradient gradient_electronic_;

public:
  /**
   * @brief Construct a molecule from atom and basis set files.
   *
   * @param atoms_path Path to atomic coordinates file.
   * @param basis_path Path to basis set directory.
   */
  Molecule(const fs::path &atoms_path, const fs::path &basis_path,
           const int num_alpha, const int num_beta);

  /**
   * @brief Number of contracted basis functions (dimension of AO matrices).
   * @return Basis size \(N\).
   */
  int getN() const { return N_basis_funcs_; }

  int getNumAtoms() const { return n_atoms_; }

  arma::mat getGradOverlapTerm() const { return grad_overlap_term_; }
  arma::mat getGradRepulsionTerm() const { return grad_repulsion_term_; }
  Gradient getGradientElectronic() const { return gradient_electronic_; }

  /**
   * @brief Energies from CNDO/2 calculation.
   * @return CNDO2 struct with energy components and total energy.
   */
  CNDO2Energy getCNDO2() const { return CNDO2_; }

  /**
   * @brief Closed-shell density from occupied MO columns.
   * @param C MO coefficient matrix (columns = MOs).
   * @param n_electrons Number of electrons of this spin; occupies first
   * `n_electrons` columns of `C`.
   * @return Density matrix \(P = C_\mathrm{occ} C_\mathrm{occ}^\top\).
   */
  arma::mat solveDensity(const arma::mat &C, int n_electrons);

  /**
   * @brief Two-center \(s\)-type Coulomb integral helper for CNDO/2
   * \(\gamma\) parameters.
   *
   * Evaluates the \( (ss|ss) \)-type interaction between spherical Gaussian
   * charge clouds with widths `sigmaA`, `sigmaB` (related to primitive
   * exponents) at centers `aoA` and `aoB`.
   *
   * @param aoA First AO (center coordinates).
   * @param aoB Second AO (center coordinates).
   * @param sigmaA Effective Gaussian width parameter for center A.
   * @param sigmaB Effective Gaussian width parameter for center B.
   * @return Two-electron integral contribution in atomic units (before any
   * eV conversion in the caller).
   */
  double calc00(const AtomicOrbital &aoA, const AtomicOrbital &aoB,
                double sigmaA, double sigmaB);

  /**
   * @brief Contracted \(\gamma_{AB}\) between two s-shell AOs (CNDO/2).
   *
   * Sums over primitive pairs on each center, converts result to eV.
   *
   * @param aoA First minimal-basis s AO.
   * @param aoB Second minimal-basis s AO.
   * @return \(\gamma_{AB}\) in electron-volts.
   */
  double calcGamma(const AtomicOrbital &aoA, const AtomicOrbital &aoB);

  arma::rowvec calc00Derivative(const AtomicOrbital &aoA,
                                const AtomicOrbital &aoB, double sigmaA,
                                double sigmaB);

  arma::rowvec calcGammaDerivative(const AtomicOrbital &aoA,
                                   const AtomicOrbital &aoB);

  /**
   * @brief Fill `gamma_` with \(\gamma_{AB}\) for all atom pairs (s orbitals
   * only).
   */
  void gammaMatrix();

  /**
   * @brief Mulliken-like atomic population from diagonal density on that atom.
   * @param atom_id Atom index.
   * @param P Density matrix (alpha, beta, or total depending on caller).
   * @return \(\sum_{\mu\in A} P_{\mu\mu}\).
   */
  double localAtomDensity(int atom_id, const arma::mat &P);

  /**
   * @brief CNDO/2 electrostatic field at `atom_id` from other atoms'
   * net charges.
   * @param atom_id Index of the atom whose environment is summed.
   * @return Sum over \(C \neq A\) of \((P_\mathrm{tot,CC} - Z_C)\gamma_{AC}\)
   * in eV-style bookkeeping used in the Fock diagonal.
   */
  double electrostaticInteract(int atom_id);

  /**
   * @brief Diagonal CNDO/2 Fock matrix element \(F_{\mu\mu}\).
   * @param u AO index \(\mu\).
   * @param P_self Spin density \(P^\alpha\) or \(P^\beta\) for this Fock build.
   * @return Diagonal Fock value.
   */
  double fockDiagonal(int u, const arma::mat &P_self);

  /**
   * @brief Off-diagonal CNDO/2 Fock element \(F_{\mu\nu}\), \(\mu\neq\nu\).
   * @param u First AO index.
   * @param v Second AO index.
   * @param P_self Same-spin density matrix.
   * @return Off-diagonal Fock value.
   */
  double fockOffDiagonal(int u, int v, const arma::mat &P_self);

  /**
   * @brief Assemble full Fock matrix `F_self` from `P_self` and stored
   * \(S\), \(\gamma\), and parameters.
   * @param F_self Output matrix (same dimension as \(S\)).
   * @param P_self Spin density used in Coulomb/exchange terms.
   */
  void fockMatrix(arma::mat &F_self, const arma::mat &P_self);

  /**
   * @brief Diagonal core Hamiltonian \(H^\mathrm{core}_{\mu\mu}\) (CNDO/2).
   * @param u AO index.
   * @return Core diagonal element.
   */
  double coreHamiltonianDiagonal(int u);

  /**
   * @brief Off-diagonal core Hamiltonian \(H^\mathrm{core}_{\mu\nu}\).
   * @param u First AO index.
   * @param v Second AO index.
   * @return Core off-diagonal (proportional to overlap and binding params).
   */
  double coreHamiltonianOffDiagonal(int u, int v);

  /** @brief Fill `H_core_` using `coreHamiltonianDiagonal/OffDiagonal`. */
  void coreHamiltonianMatrix();

  /**
   * @brief Diagonalize a real symmetric Fock matrix.
   * @param F_self Fock matrix to diagonalize.
   * @param n_electrons Unused by implementation; reserved for occupation logic.
   * @return Eigenvalues and eigenvectors (`MoleculeEnergy`).
   */
  MoleculeEnergy solveEnergy(const arma::mat &F_self, int n_electrons);

  /**
   * @brief Nuclear repulsion \(\sum_{A<B} Z_A Z_B / R_{AB}\) in eV.
   * @return Classical repulsion energy (valence \(Z\) from `ValenceElectrons`).
   */
  double nuclearRepulsion();

  /**
   * @brief Electronic energy from current densities and Fock/core matrices.
   * @return \(\frac12\mathrm{Tr}\,P_\alpha(H_\mathrm{core}+F_\alpha) +
   * \frac12\mathrm{Tr}\,P_\beta(H_\mathrm{core}+F_\beta)\).
   */
  double electronEnergy();

  /** @brief Store `electronEnergy`, `nuclearRepulsion`, and total in `CNDO2_`.
   */
  void cndo2Energy();

  /**
   * @brief Self-consistent field: build \(\gamma\) and \(H_\mathrm{core}\),
   * iterate Fock/density until convergence, then fill `CNDO2_` energies.
   * @param filepath Input path (used for log messages only).
   */
  void SCF(fs::path filepath);

  double calcDerivative3D(int dim, AtomicOrbital &u, AtomicOrbital &v);
  void gradOverlapTerm(int id_A, int id_B);
  void electronicGradient();
  void gradRepulsionTerm(int id_A, int id_B);

  /**
   * @brief Access an atomic orbital by index.
   *
   * Used during overlap, core Hamiltonian, and Fock matrix assembly (CNDO/2).
   *
   * @param idx Basis function index.
   * @return Constant reference to the corresponding `AtomicOrbital`.
   */
  const AtomicOrbital &getAtomicOrbital(int idx) const {
    return basis_funcs_.at(idx);
  }

  /**
   * @brief Parse atoms and build atomic orbitals from input files.
   *
   * @param atoms_path Path to atomic coordinates file.
   * @param basis_path Path to basis set directory.
   * @return Vector of constructed atomic orbitals.
   */
  std::vector<AtomicOrbital> parse_atoms(const fs::path &atoms_path,
                                         const fs::path &basis_path);

  /**
   * @brief Export molecule data and matrices to an HDF5 file.
   *
   * @param output_file_path Path to the HDF5 output file.
   */
  void exportMoleculeResults(fs::path output_file_path);

  /**
   * @brief Generate a 1D PES for diatomic oxygen
   *
   * @param config_file_path Path to the JSON setup file.
   * @param atoms_file_path Path to the atoms XYZ file.
   * @param basis_path Path to the directory with basis function info.
   * @param p Num of alpha electrons.
   * @param q Num of beta electrons.
   * @param max_bound Max bond length to test for PES
   * @param step_size Step size between each CNDO/2 calculation.
   */
  static void generatePES(fs::path &config_file_path, fs::path &basis_path,
                          int p, int q, double max_bond, double step_size);
};

#endif