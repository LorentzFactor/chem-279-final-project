#ifndef SYSTEM_LIB_H
#define SYSTEM_LIB_H

#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>

#include "gaussian_lib/gaussian_lib.h"
#include <armadillo>
#include <array>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using namespace gaussian_lib;

// Custom type names for clarity
using RealMat = arma::mat;
using RealVec = arma::vec;
using RealRowVec = arma::rowvec;
using RealCube = arma::cube;

using ComplexMat = arma::cx_mat;
using ComplexVec = arma::cx_vec;
using ComplexRowVec = arma::cx_rowvec;
using ComplexCube = arma::cx_cube;

namespace system_lib {

// Units for distance in input files
enum DistanceUnits { BOHR, ANGSTROM };

struct Atom {
private:
  std::array<double, 3> position_; // Position of atom
  short atomic_number_;            // Its atomic number
  std::string atomic_symbol_;      // Atomic symbol
  inline static const std::array<std::string, 14> symbol_list_{
      "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",
      "Ne", "Na", "Mg", "Al", "Si"};
  std::vector<GaussianContracted>
      atomic_orbitals_;                    // A vector of the atom's orbitals
  std::vector<std::string> orbital_names_; // A vector of the same length as
                                           // orbitals with their names
  /** CNDO/2-style parameters (eV); Si from third-row extension (literature). */
  inline static const std::unordered_map<std::string, double> sI_plus_A_{
      {"H", 7.176},  {"C", 14.051}, {"N", 19.361}, {"O", 25.390}, {"F", 32.272},
      {"Si", 10.06},
  };
  inline static const std::unordered_map<std::string, double> pI_plus_A_{
      {"C", 5.572}, {"N", 7.275}, {"O", 9.111}, {"F", 11.080}, {"Si", 5.57},
  };
  inline static const std::unordered_map<std::string, double> neg_beta_{
      {"H", -9}, {"C", -21}, {"N", -25}, {"O", -31}, {"F", -39}, {"Si", -8},
  };
  inline static const std::unordered_map<std::string, double> Z_A{
      {"H", 1}, {"C", 4}, {"N", 5}, {"O", 6}, {"F", 7}, {"Si", 4},
  };

public:
  Atom(std::array<double, 3> position, short atomic_number,
       GaussianTemplate s_basis);
  Atom(std::array<double, 3> position, short atomic_number,
       GaussianTemplate s_basis, GaussianTemplate p_basis);
  static const std::string &get_number_symbol(const short &atomic_number);
  static const short get_symbol_number(const std::string &atomic_symbol);
  double get_atom_constant(const std::string &const_name) const;
  std::string get_symbol() const { return atomic_symbol_; };
  const std::vector<GaussianContracted> &get_atomic_orbitals() const {
    return atomic_orbitals_;
  };
  int num_orbitals() const { return atomic_orbitals_.size(); };
  const std::vector<std::string> &get_orbital_names() const {
    return orbital_names_;
  };
  const std::array<double, 3> &get_position() const { return position_; };
  const short &get_atomic_number() const { return atomic_number_; };
};

struct System {
private:
  // Lazy-cached AO overlap (recomputed when empty)
  mutable arma::mat S_;

protected:
  std::vector<Atom> atoms_;
  size_t num_orbitals_;
  std::vector<std::array<size_t, 2>> atom_orbital_idxs;
  static std::vector<Atom>
  atoms_from_files_(std::string atoms_filepath, std::string basis_directory,
                    DistanceUnits distance_units = DistanceUnits::BOHR);

public:
  System(const std::vector<Atom> &atoms);
  arma::mat compute_overlap_matrix() const;
  size_t num_orbitals() const;
  size_t num_atoms() const;
  virtual double compute_electronic_energy() = 0;
  double compute_nuclear_energy() const;
  double compute_total_energy();
  const Atom &get_atom(size_t atom_idx) const { return atoms_.at(atom_idx); };
  const std::vector<std::array<size_t, 2>> &get_atom_orbital_idxs() const {
    return atom_orbital_idxs;
  };
};

struct CNDO2RealIntegrals {
  RealMat S_;
  RealMat gamma_;
  RealMat gamma_reduced_;
  RealMat beta_;
};

// Shared functions that both CNDO2 classes can use

RealMat compute_gamma_matrix(RealMat &gamma, System &sys,
                             const std::vector<Atom> &atoms_);
RealMat compute_reduced_gamma_matrix(RealMat &gamma_reduced, System &sys);
RealMat compute_beta_matrix(RealMat &beta, System &sys,
                            const std::vector<Atom> &atoms);

// Shared Fock builder used by both real and complex CNDO2 systems.
template <class MatT, class SystemT>
std::pair<MatT, MatT> build_cndo2_fock(SystemT &sys, const MatT &p_alpha,
                                       const MatT &p_beta);

struct CNDO2System : public System { // Real (13C NMR)
private:
  // F matrix + energy states
  // Real matrices
  RealMat p_alpha_, p_beta_;
  RealMat f_alpha_, f_beta_;
  RealMat mos_alpha_, mos_beta_;
  RealVec E_alpha_, E_beta_;

  // Shared real integrals between real and complex structs
  CNDO2RealIntegrals I_;

  // Electron counts
  double p_;
  double q_;

protected:
  std::pair<RealMat, RealMat>
  compute_cndo_f_matrix_internal(const RealMat &p_alpha, const RealMat &p_beta);

public:
  CNDO2System(const std::vector<Atom> &atoms, int p, int q);
  static CNDO2System
  from_files(std::string atoms_filepath, std::string basis_directory, int p,
             int q, DistanceUnits distance_units = DistanceUnits::BOHR);

  void set_p(const RealMat &new_p_alpha, const RealMat &new_p_beta);
  const RealMat &get_p_alpha() const { return p_alpha_; };
  const RealMat &get_p_beta() const { return p_beta_; };

  void set_f(const RealMat &new_f_alpha, const RealMat &new_f_beta);
  const RealMat &get_f_alpha() const { return f_alpha_; };
  const RealMat &get_f_beta() const { return f_beta_; };

  void set_nelectrons(int p, int q) {
    p_ = (double)p;
    q_ = (double)q;
  };
  int get_nalpha() const { return (int)p_; };
  int get_nbeta() const { return (int)q_; };

  const RealMat &get_MOs_alpha() const { return mos_alpha_; };
  const RealMat &get_MOs_beta() const { return mos_beta_; };
  RealMat get_occupied_MOs_alpha() const;
  RealMat get_occupied_MOs_beta() const;

  RealRowVec get_E_alpha() const { return E_alpha_.as_row(); };
  RealRowVec get_E_beta() const { return E_beta_.as_row(); };

  CNDO2RealIntegrals &real_integrals() { return I_; }
  const CNDO2RealIntegrals &real_integrals() const { return I_; }
  const std::vector<Atom> &atoms() const { return atoms_; }

  std::pair<RealMat, RealMat> compute_cndo_f_matrix();
  RealMat compute_h_core();
  double compute_electronic_energy() override;

  /** Mulliken-like population on atom AOs from occupied MOs (used by NMR). */
  double get_electron_density(size_t atom_idx) const;
};

double distance(const Atom &a1, const Atom &a2);

// center of mass helper for getting the gauge origin
std::array<double, 3> molecule_center_of_mass(const std::vector<Atom> &atoms);

struct CNDO2SystemComplex : public System { // Complex (1H NMR)
private:
  // F matrix + energy states
  // Complex types
  ComplexMat p_alpha_, p_beta_;
  ComplexMat f_alpha_, f_beta_;
  ComplexMat mos_alpha_, mos_beta_;
  RealVec E_alpha_, E_beta_;

  // Shared real integrals between real and complex structs
  CNDO2RealIntegrals I_;

  // Electron counts
  double p_;
  double q_;

  // Magnetic perturbation field
  double lambda_ = 0.0;
  int field_dir_ = 0; // x, y, or z

protected:
  std::pair<ComplexMat, ComplexMat>
  compute_cndo_f_matrix_internal(const ComplexMat &p_alpha,
                                 const ComplexMat &p_beta);

public:
  CNDO2SystemComplex(const std::vector<Atom> &atoms, int p, int q);
  static CNDO2SystemComplex
  from_files(std::string atoms_filepath, std::string basis_directory, int p,
             int q, DistanceUnits distance_units = DistanceUnits::BOHR);

  void set_p(const ComplexMat &new_p_alpha, const ComplexMat &new_p_beta);
  const ComplexMat &get_p_alpha() const { return p_alpha_; };
  const ComplexMat &get_p_beta() const { return p_beta_; };

  void set_f(const ComplexMat &new_f_alpha, const ComplexMat &new_f_beta);
  const ComplexMat &get_f_alpha() const { return f_alpha_; };
  const ComplexMat &get_f_beta() const { return f_beta_; };

  void set_nelectrons(int p, int q) {
    p_ = (double)p;
    q_ = (double)q;
  };
  int get_nalpha() const { return (int)p_; };
  int get_nbeta() const { return (int)q_; };

  const ComplexMat &get_MOs_alpha() const { return mos_alpha_; };
  const ComplexMat &get_MOs_beta() const { return mos_beta_; };
  ComplexMat get_occupied_MOs_alpha() const;
  ComplexMat get_occupied_MOs_beta() const;

  RealRowVec get_E_alpha() const { return E_alpha_.as_row(); };
  RealRowVec get_E_beta() const { return E_beta_.as_row(); };

  CNDO2RealIntegrals &real_integrals() { return I_; }
  const CNDO2RealIntegrals &real_integrals() const { return I_; }
  const std::vector<Atom> &atoms() const { return atoms_; }

  std::pair<ComplexMat, ComplexMat> compute_cndo_f_matrix();
  ComplexMat compute_h_core();
  double compute_electronic_energy() override;

  void set_magnetic_field(int direction, double lambda) {
    field_dir_ = direction;
    lambda_ = lambda;
  }

  int get_field_dir() const { return field_dir_; }
  double get_lambda() const { return lambda_; }

  double calc_angular_momentum_term(const gaussian_lib::GaussianContracted &u,
                                    const gaussian_lib::GaussianContracted &v,
                                    const std::array<double, 3> &gauge_origin,
                                    int coord_dir, int deriv_dir) const;

  RealMat compute_angular_momentum_matrix(
      int direction, const std::array<double, 3> &gauge_origin) const;

  /** Point-dipole-style shielding operator H^{(1,1)} in AO basis (Pople); uses L
   *  matrices and neighbor weights. See compute_angular_momentum_matrix. */
  RealMat compute_shielding_operator_matrix(int direction,
                                            size_t target_proton_idx) const;
};

#include "fock.tpp"
} // namespace system_lib

#endif