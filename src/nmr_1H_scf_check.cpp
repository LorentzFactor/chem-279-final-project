// Sanity check: real CNDO/2 SCF vs complex SCF path (magnetic perturbation hook).
// With a dummy zero angular-momentum matrix, i*lambda*L is zero and energies should
// match the real path at lambda=0; a small lambda should not change energy until L is implemented.

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include <armadillo>
#include <nlohmann/json.hpp>

#include "solver_lib/solver_lib.h"
#include "system_lib/system_lib.h"

namespace fs = std::filesystem;
using json = nlohmann::json;

static system_lib::DistanceUnits extract_config_units(const json &config) {
  if (config.contains("distance_unit")) {
    const std::string u = config["distance_unit"].get<std::string>();
    std::cout << "distance_unit: " << u << std::endl;
    return (u == "angstrom") ? system_lib::DistanceUnits::ANGSTROM
                             : system_lib::DistanceUnits::BOHR;
  }
  return system_lib::DistanceUnits::BOHR;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " path/to/molecule_config.json\n"
              << "JSON fields: atoms_file_path, num_alpha_electrons, "
                 "num_beta_electrons;\n"
              << "optional: distance_unit, basis_dir (default \"./basis\"), "
                 "lambda_probe (default 1e-8), field_dir 0|1|2 (default 2=z).\n";
    return EXIT_FAILURE;
  }

  const fs::path config_path(argv[1]);
  if (!fs::exists(config_path)) {
    std::cerr << "Config not found: " << config_path << std::endl;
    return EXIT_FAILURE;
  }

  std::ifstream in(config_path);
  json config = json::parse(in);

  const fs::path atoms_path = config.at("atoms_file_path").get<std::string>();
  const int n_alpha = config.at("num_alpha_electrons").get<int>();
  const int n_beta = config.at("num_beta_electrons").get<int>();
  const system_lib::DistanceUnits units = extract_config_units(config);

  std::string basis_dir = "./basis";
  if (config.contains("basis_dir")) {
    basis_dir = config["basis_dir"].get<std::string>();
  }

  double lambda_probe = 1e-8;
  if (config.contains("lambda_probe")) {
    lambda_probe = config["lambda_probe"].get<double>();
  }

  int field_dir = 2;
  if (config.contains("field_dir")) {
    field_dir = config["field_dir"].get<int>();
    if (field_dir < 0 || field_dir > 2) {
      std::cerr << "field_dir must be 0 (x), 1 (y), or 2 (z).\n";
      return EXIT_FAILURE;
    }
  }

  if (!fs::exists(atoms_path)) {
    std::cerr << "Atoms file not found: " << atoms_path << std::endl;
    return EXIT_FAILURE;
  }

  // --- Real SCF (13C path) ---
  system_lib::CNDO2System real_sys = system_lib::CNDO2System::from_files(
      atoms_path.string(), basis_dir, n_alpha, n_beta, units);
  fixed_point::solve_cndo(real_sys);
  const double E_tot_real = real_sys.compute_total_energy();
  const double E_el_real = real_sys.compute_electronic_energy();

  // --- Complex SCF, lambda = 0 ---
  system_lib::CNDO2SystemComplex cx0 = system_lib::CNDO2SystemComplex::from_files(
      atoms_path.string(), basis_dir, n_alpha, n_beta, units);
  cx0.set_magnetic_field(field_dir, 0.0);
  fixed_point::solve_cndo(cx0);
  const double E_tot_cx0 = cx0.compute_total_energy();
  const double E_el_cx0 = cx0.compute_electronic_energy();

  // --- Complex SCF, small lambda (perturbation is still zero if L_k is zero) ---
  system_lib::CNDO2SystemComplex cx_lam =
      system_lib::CNDO2SystemComplex::from_files(
          atoms_path.string(), basis_dir, n_alpha, n_beta, units);
  cx_lam.set_magnetic_field(field_dir, lambda_probe);
  fixed_point::solve_cndo(cx_lam);
  const double E_tot_lam = cx_lam.compute_total_energy();
  const double E_el_lam = cx_lam.compute_electronic_energy();

  const double tol_abs = 1e-4;
  const double d_el_0 = std::abs(E_el_real - E_el_cx0);
  const double d_tot_0 = std::abs(E_tot_real - E_tot_cx0);
  const double d_el_lam = std::abs(E_el_cx0 - E_el_lam);

  std::cout << std::fixed << std::setprecision(10);
  std::cout << "Electronic energy (real):     " << E_el_real << " Ha\n";
  std::cout << "Electronic energy (cx,λ=0):   " << E_el_cx0 << " Ha\n";
  std::cout << "Electronic energy (cx,λ=" << lambda_probe << "): " << E_el_lam
            << " Ha\n";
  std::cout << "|E_el(real) - E_el(cx,0)| = " << d_el_0 << " Ha\n";
  std::cout << "|E_tot(real) - E_tot(cx,0)| = " << d_tot_0 << " Ha\n";
  std::cout << "|E_el(cx,0) - E_el(cx,λ)| = " << d_el_lam << " Ha\n";

  bool ok = (d_el_0 < tol_abs) && (d_tot_0 < tol_abs) && (d_el_lam < tol_abs);
  if (!ok) {
    std::cerr << "FAIL: energy mismatch exceeds " << tol_abs << " Ha.\n";
    return EXIT_FAILURE;
  }

  std::cout << "PASS: real vs complex (λ=0) and small-λ check within " << tol_abs
            << " Ha.\n";
  return EXIT_SUCCESS;
}
