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
  fs::path config_file_path_;

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
  Gradient gradient_nuclear_;
  Gradient gradient_total_;

public:
  /** @name Construction And State */
  ///@{
  /** @brief Construct a molecule from atom and basis set files. */
  Molecule(const fs::path &atoms_path, const fs::path &basis_path,
           const int num_alpha, const int num_beta);
  /** @brief Set the config path used for reporting. */
  void setConfigPath(fs::path config_path) { config_file_path_ = config_path; }
  ///@}

  /** @name Basic Accessors */
  ///@{
  /** @brief Number of contracted basis functions. */
  int getN() const { return N_basis_funcs_; }
  /** @brief Number of atoms in the molecule. */
  int getNumAtoms() const { return n_atoms_; }
  /** @brief Current CNDO/2 energy components. */
  CNDO2Energy getCNDO2() const { return CNDO2_; }
  ///@}

  /** @name SCF Pipeline */
  ///@{
  /** @brief Build spin density from occupied molecular orbitals. */
  arma::mat solveDensity(const arma::mat &C, int n_electrons);
  /** @brief Compute the CNDO/2 gamma helper integral term. */
  double calc00(const AtomicOrbital &aoA, const AtomicOrbital &aoB,
                double sigmaA, double sigmaB);
  /** @brief Compute contracted CNDO/2 atom-pair gamma value. */
  double calcGamma(const AtomicOrbital &aoA, const AtomicOrbital &aoB);
  /** @brief Assemble the atom-pair gamma matrix. */
  void gammaMatrix();
  /** @brief Sum on-atom diagonal density contributions. */
  double localAtomDensity(int atom_id, const arma::mat &P);
  /** @brief Compute electrostatic interaction term for one atom. */
  double electrostaticInteract(int atom_id);
  /** @brief Compute diagonal Fock matrix element. */
  double fockDiagonal(int u, const arma::mat &P_self);
  /** @brief Compute off-diagonal Fock matrix element. */
  double fockOffDiagonal(int u, int v, const arma::mat &P_self);
  /** @brief Build full Fock matrix for one spin block. */
  void fockMatrix(arma::mat &F_self, const arma::mat &P_self);
  /** @brief Compute diagonal core Hamiltonian element. */
  double coreHamiltonianDiagonal(int u);
  /** @brief Compute off-diagonal core Hamiltonian element. */
  double coreHamiltonianOffDiagonal(int u, int v);
  /** @brief Build the core Hamiltonian matrix. */
  void coreHamiltonianMatrix();
  /** @brief Diagonalize Fock matrix to obtain MOs and orbital energies. */
  MoleculeEnergy solveEnergy(const arma::mat &F_self, int n_electrons);
  /** @brief Compute nuclear repulsion energy. */
  double nuclearRepulsion();
  /** @brief Compute electronic energy from densities and Fock/core terms. */
  double electronEnergy();
  /** @brief Store all CNDO/2 energy components. */
  void cndo2Energy();
  /** @brief Run self-consistent field iterations. */
  void SCF(bool verbose);
  ///@}

  /** @name Analytic Gradient Terms */
  ///@{
  /** @brief Reset all gradient vectors to zero. */
  void zeroGradient(Gradient &g);
  /** @brief Primitive overlap derivative for one Cartesian direction. */
  double calcDerivative3D(int dim, AtomicOrbital &u, AtomicOrbital &v);
  /** @brief Accumulate overlap-driven gradient contribution. */
  void gradOverlapTerm(int id_A, int id_B);
  /** @brief Derivative of the CNDO/2 gamma helper integral. */
  arma::rowvec calc00Derivative(const AtomicOrbital &aoA,
                                const AtomicOrbital &aoB, double sigmaA,
                                double sigmaB);
  /** @brief Derivative of contracted CNDO/2 gamma value. */
  arma::rowvec calcGammaDerivative(const AtomicOrbital &aoA,
                                   const AtomicOrbital &aoB);
  /** @brief Accumulate electron-repulsion gradient contribution. */
  void gradRepulsionTerm(int id_A, int id_B);
  /** @brief Build full electronic gradient. */
  void electronicGradient();
  /** @brief Nuclear repulsion derivative between two atoms. */
  arma::rowvec nucRepulsionDeriv(arma::rowvec &RA, arma::rowvec &RB, double ZA,
                                 double ZB);
  /** @brief Build full nuclear gradient. */
  void nuclearGradient();
  /** @brief Combine electronic and nuclear gradients. */
  void totalGradient();
  /** @brief Convert gradient struct to 3xN matrix form. */
  arma::mat gradVecsToMat(Gradient &g);
  /** @brief Recompute SCF energy and all force terms at current geometry. */
  double calcEnergyAndForces();
  ///@}

  /** @name Geometry Optimization */
  ///@{
  /** @brief Return a value copy of this molecule. */
  Molecule copyMolecule();
  /** @brief Steepest-descent geometry optimization loop. */
  void steepestDescentOptimizer(double step, double force_threshold);
  /** @brief Print current atomic coordinates. */
  void printCoords();
  /** @brief Print basic geometric properties for small molecules. */
  void geometricProperties();
  /** @brief Gradient overlap diagnostic matrix. */
  arma::mat getGradOverlapTerm() const { return grad_overlap_term_; }
  /** @brief Gradient repulsion diagnostic matrix. */
  arma::mat getGradRepulsionTerm() const { return grad_repulsion_term_; }
  /** @brief Electronic gradient vector components. */
  Gradient getGradientElectronic() const { return gradient_electronic_; }
  /** @brief Nuclear gradient vector components. */
  Gradient getGradientNuclear() const { return gradient_nuclear_; }
  /** @brief Total gradient vector components. */
  Gradient getGradientTotal() const { return gradient_total_; }
  ///@}
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
};

#endif