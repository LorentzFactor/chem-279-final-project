// 1H NMR chemical shifts (CNDO/2 + complex SCF with magnetic-field perturbation
// i·λ·L in the Fock build). Layout mirrors nmr_13C_calculator.cpp; Ramsey σ(H)
// and δ in ppm are wired once nmr_lib exposes proton shielding.

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

#include <armadillo>
#include <nlohmann/json.hpp>

#include "solver_lib/solver_lib.h"
#include "system_lib/system_lib.h"

namespace fs = std::filesystem;
using json = nlohmann::json;

inline system_lib::DistanceUnits extract_config_units(json config) {
  if (config.find("distance_unit") != config.end()) {
    std::cout << "Extracted distance unit from config: "
              << config["distance_unit"] << std::endl;
    return config["distance_unit"] == "angstrom"
               ? system_lib::DistanceUnits::ANGSTROM
               : system_lib::DistanceUnits::BOHR;
  }
  return system_lib::DistanceUnits::BOHR;
}

static void collect_hydrogen_atom_indices(const system_lib::CNDO2System &sys,
                                          std::vector<size_t> *out_indices) {
  out_indices->clear();
  for (size_t i = 0; i < sys.num_atoms(); ++i) {
    if (sys.get_atom(i).get_atomic_number() == 1) {
      out_indices->push_back(i);
    }
  }
}

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " path/to/molecule_config.json"
              << " path/to/reference_config.json\n"
              << "Example (from repo root): " << argv[0]
              << " sample_input/ethane.json sample_input/TMS.json\n"
              << "JSON: atoms_file_path, num_alpha_electrons, num_beta_electrons;\n"
              << "optional: distance_unit, basis_dir (default \"./basis\"), "
                 "delta_e (default 11.30, informational), lambda_probe "
                 "(default 1e-8), field_dir 0|1|2 (default 2=z).\n";
    return EXIT_FAILURE;
  }

  fs::path config_file_path(argv[1]);
  if (!fs::exists(config_file_path)) {
    std::cerr << "Path: " << config_file_path << " does not exist" << std::endl;
    return EXIT_FAILURE;
  }
  std::ifstream config_file(config_file_path);
  json config = json::parse(config_file);

  fs::path reference_config_file_path(argv[2]);
  if (!fs::exists(reference_config_file_path)) {
    std::cerr << "Path: " << reference_config_file_path << " does not exist"
              << std::endl;
    return EXIT_FAILURE;
  }
  std::ifstream reference_config_file(reference_config_file_path);
  json reference_config = json::parse(reference_config_file);

  fs::path atoms_file_path = config["atoms_file_path"];
  fs::path output_file_path = config["output_file_path"];
  int num_alpha_electrons = config["num_alpha_electrons"];
  int num_beta_electrons = config["num_beta_electrons"];
  const double delta_e_main = config.value("delta_e", 11.30);
  system_lib::DistanceUnits distance_units = extract_config_units(config);

  fs::path reference_atoms_file_path = reference_config["atoms_file_path"];
  int reference_num_alpha_electrons = reference_config["num_alpha_electrons"];
  int reference_num_beta_electrons = reference_config["num_beta_electrons"];
  const double delta_e_ref = reference_config.value("delta_e", 11.30);
  system_lib::DistanceUnits reference_distance_units =
      extract_config_units(reference_config);

  std::string basis_dir = "./basis";
  if (config.contains("basis_dir")) {
    basis_dir = config["basis_dir"].get<std::string>();
  }
  if (reference_config.contains("basis_dir")) {
    const std::string br = reference_config["basis_dir"].get<std::string>();
    if (br != basis_dir) {
      std::cout << "Note: reference basis_dir (\"" << br
                << "\") ignored; using main basis_dir \"" << basis_dir << "\".\n";
    }
  }

  double lambda_probe = 1e-8;
  if (config.contains("lambda_probe")) {
    lambda_probe = config["lambda_probe"].get<double>();
  } else if (reference_config.contains("lambda_probe")) {
    lambda_probe = reference_config["lambda_probe"].get<double>();
  }

  int field_dir = 2;
  if (config.contains("field_dir")) {
    field_dir = config["field_dir"].get<int>();
  } else if (reference_config.contains("field_dir")) {
    field_dir = reference_config["field_dir"].get<int>();
  }
  if (field_dir < 0 || field_dir > 2) {
    std::cerr << "field_dir must be 0 (x), 1 (y), or 2 (z).\n";
    return EXIT_FAILURE;
  }

  (void)output_file_path;

  // --- Main: real CNDO/2 (Mulliken / future σ(H) path, same as 13C stack) ---
  system_lib::CNDO2System main_real = system_lib::CNDO2System::from_files(
      atoms_file_path, basis_dir, num_alpha_electrons, num_beta_electrons,
      distance_units);
  fixed_point::solve_cndo(main_real);

  std::vector<size_t> main_h_atoms;
  collect_hydrogen_atom_indices(main_real, &main_h_atoms);

  std::cout << "Delta_E (config, informational) for each hydrogen on main:\n";
  for (size_t k = 0; k < main_h_atoms.size(); ++k) {
    std::cout << "Hydrogen " << k << " (atom " << main_h_atoms[k]
              << ") delta_E: " << delta_e_main << '\n';
  }

  std::cout << "Mulliken-like H electron count per atom (main, real SCF):\n";
  for (size_t k = 0; k < main_h_atoms.size(); ++k) {
    const size_t iatom = main_h_atoms[k];
    const double rho = main_real.get_electron_density(iatom);
    std::cout << std::fixed << std::setprecision(4) << "Atom " << iatom
              << " (H): electron_density = " << rho << '\n';
  }

  // --- Main: complex SCF with magnetic perturbation (λ = 0 then λ = probe) ---
  system_lib::CNDO2SystemComplex main_cx0 =
      system_lib::CNDO2SystemComplex::from_files(
          atoms_file_path, basis_dir, num_alpha_electrons, num_beta_electrons,
          distance_units);
  main_cx0.set_magnetic_field(field_dir, 0.0);
  const int it_main_0 = fixed_point::solve_cndo(main_cx0);
  const double E_tot_main_0 = main_cx0.compute_total_energy();
  const double E_el_main_0 = main_cx0.compute_electronic_energy();

  system_lib::CNDO2SystemComplex main_cx_lam =
      system_lib::CNDO2SystemComplex::from_files(
          atoms_file_path, basis_dir, num_alpha_electrons, num_beta_electrons,
          distance_units);
  main_cx_lam.set_magnetic_field(field_dir, lambda_probe);
  const int it_main_lam = fixed_point::solve_cndo(main_cx_lam);
  const double E_tot_main_lam = main_cx_lam.compute_total_energy();
  const double E_el_main_lam = main_cx_lam.compute_electronic_energy();

  std::cout << std::fixed << std::setprecision(10);
  std::cout << "Main complex SCF: field axis = " << field_dir
            << ", iterations (λ=0 / λ) = " << it_main_0 << " / " << it_main_lam
            << "\n";
  std::cout << "Main E_tot (Re, λ=0):     " << std::real(E_tot_main_0) << " Ha\n";
  std::cout << "Main E_el (Re, λ=0):     " << std::real(E_el_main_0) << " Ha\n";
  std::cout << "Main E_tot (Re, λ):      " << std::real(E_tot_main_lam) << " Ha\n";
  std::cout << "Main E_el (Re, λ):       " << std::real(E_el_main_lam) << " Ha\n";
  std::cout << "|E_el(Re,λ) - E_el(Re,0)| = "
            << std::abs(std::real(E_el_main_lam) - std::real(E_el_main_0))
            << " Ha\n";

  // --- Reference: real (DIIS, same as 13C calculator) ---
  system_lib::CNDO2System reference_real =
      system_lib::CNDO2System::from_files(
          reference_atoms_file_path, basis_dir, reference_num_alpha_electrons,
          reference_num_beta_electrons, reference_distance_units);
  diis::solve_cndo(reference_real);

  std::vector<size_t> ref_h_atoms;
  collect_hydrogen_atom_indices(reference_real, &ref_h_atoms);

  std::cout << std::setprecision(4);
  std::cout << "Delta_E (config, informational) for each hydrogen on reference:\n";
  for (size_t k = 0; k < ref_h_atoms.size(); ++k) {
    std::cout << "Hydrogen " << k << " (atom " << ref_h_atoms[k]
              << ") delta_E: " << delta_e_ref << '\n';
  }

  std::cout << "Mulliken-like H electron count per atom (reference, real SCF):\n";
  for (size_t k = 0; k < ref_h_atoms.size(); ++k) {
    const size_t iatom = ref_h_atoms[k];
    const double rho = reference_real.get_electron_density(iatom);
    std::cout << std::fixed << std::setprecision(4) << "Reference Atom " << iatom
              << " (H): electron_density = " << rho << '\n';
  }

  // --- Reference: complex with perturbation (mirrors main) ---
  system_lib::CNDO2SystemComplex ref_cx0 =
      system_lib::CNDO2SystemComplex::from_files(
          reference_atoms_file_path, basis_dir, reference_num_alpha_electrons,
          reference_num_beta_electrons, reference_distance_units);
  ref_cx0.set_magnetic_field(field_dir, 0.0);
  const int it_ref_0 = fixed_point::solve_cndo(ref_cx0);
  const double E_tot_ref_0 = ref_cx0.compute_total_energy();
  const double E_el_ref_0 = ref_cx0.compute_electronic_energy();

  system_lib::CNDO2SystemComplex ref_cx_lam =
      system_lib::CNDO2SystemComplex::from_files(
          reference_atoms_file_path, basis_dir, reference_num_alpha_electrons,
          reference_num_beta_electrons, reference_distance_units);
  ref_cx_lam.set_magnetic_field(field_dir, lambda_probe);
  const int it_ref_lam = fixed_point::solve_cndo(ref_cx_lam);
  const double E_tot_ref_lam = ref_cx_lam.compute_total_energy();
  const double E_el_ref_lam = ref_cx_lam.compute_electronic_energy();

  std::cout << std::setprecision(10);
  std::cout << "Reference complex SCF: iterations (λ=0 / λ) = " << it_ref_0
            << " / " << it_ref_lam << "\n";
  std::cout << "Reference E_tot (Re, λ=0): " << std::real(E_tot_ref_0) << " Ha\n";
  std::cout << "Reference E_el (Re, λ=0): " << std::real(E_el_ref_0) << " Ha\n";
  std::cout << "Reference E_tot (Re, λ):  " << std::real(E_tot_ref_lam) << " Ha\n";
  std::cout << "Reference E_el (Re, λ):   " << std::real(E_el_ref_lam) << " Ha\n";
  std::cout << "|E_el(Re,λ) - E_el(Re,0)| = "
            << std::abs(std::real(E_el_ref_lam) - std::real(E_el_ref_0))
            << " Ha\n";

  const int num_H_main = static_cast<int>(main_h_atoms.size());
  const int num_H_reference = static_cast<int>(ref_h_atoms.size());
  arma::mat chemical_shifts =
      arma::zeros(num_H_main, num_H_reference);

  const double delta_H_const = 0.0;

  for (int iH_main = 0; iH_main < num_H_main; ++iH_main) {
    for (int iH_ref = 0; iH_ref < num_H_reference; ++iH_ref) {
      // TODO: δ(ppm) = σ_ref(H) - σ_main(H) once nmr_lib exposes σ(H).
      chemical_shifts(iH_main, iH_ref) = 0.0 + delta_H_const;
    }
  }

  std::cout << std::fixed << std::setprecision(4);
  std::cout << "Chemical shifts δ (ppm) — placeholder (σ(H) not in nmr_lib yet):\n";
  for (size_t i = 0; i < chemical_shifts.n_rows; i++) {
    for (size_t j = 0; j < chemical_shifts.n_cols; j++) {
      std::cout << "Main H " << i << " vs Reference H " << j << ": "
                << chemical_shifts(i, j) << " ppm; \n";
    }
    std::cout << "\n";
  }

  return EXIT_SUCCESS;
}
