#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "solver_lib/solver_lib.h"
#include "system_lib/system_lib.h"

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {
constexpr double kEvToKcalPerMol = 23.060547830619;
}

inline system_lib::DistanceUnits extract_config_units(const json &config) {
  if (config.find("distance_unit") != config.end()) {
    return config["distance_unit"] == "angstrom"
               ? system_lib::DistanceUnits::ANGSTROM
               : system_lib::DistanceUnits::BOHR;
  }
  return system_lib::DistanceUnits::BOHR;
}

double solve_isolated_atom_energy(const system_lib::Atom &atom,
                                  bool use_indo,
                                  int scf_max_iters,
                                  double scf_tol) {
  const int z = static_cast<int>(atom.get_atomic_number());
  const int nalpha = (z + 1) / 2;
  const int nbeta = z / 2;

  std::vector<system_lib::Atom> singleton_atoms{atom};
  system_lib::CNDO2System atom_sys(singleton_atoms, nalpha, nbeta, use_indo);
  diis::solve_cndo(atom_sys, scf_max_iters, scf_tol);
  return atom_sys.compute_total_energy();
}

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " [--cndo|--indo] path/to/molecule_config.json\n"
              << "Example (from repo root): " << argv[0]
              << " --cndo sample_input/ethane.json\n"
              << "JSON: atoms_file_path, num_alpha_electrons, num_beta_electrons;\n"
              << "optional: distance_unit, basis_dir (default \"./basis\"),\n"
              << "optional: scf_max_iters (default 1000), scf_tol (default 1e-6).\n";
    return EXIT_FAILURE;
  }

  bool use_indo = false;
  std::string method_flag(argv[1]);
  if (method_flag == "--cndo") {
    use_indo = false;
  } else if (method_flag == "--indo") {
    use_indo = true;
  } else {
    std::cerr << "First argument must be --cndo or --indo.\n";
    return EXIT_FAILURE;
  }

  fs::path config_file_path(argv[2]);
  if (!fs::exists(config_file_path)) {
    std::cerr << "Path: " << config_file_path << " does not exist\n";
    return EXIT_FAILURE;
  }

  std::ifstream config_file(config_file_path);
  if (!config_file) {
    std::cerr << "Could not open config file: " << config_file_path << "\n";
    return EXIT_FAILURE;
  }
  json config = json::parse(config_file);

  fs::path atoms_file_path = config["atoms_file_path"];
  const int num_alpha_electrons = config["num_alpha_electrons"];
  const int num_beta_electrons = config["num_beta_electrons"];
  const system_lib::DistanceUnits distance_units = extract_config_units(config);

  std::string basis_dir = "./basis";
  if (config.contains("basis_dir")) {
    basis_dir = config["basis_dir"].get<std::string>();
  }

  int scf_max_iters = 1000;
  if (config.contains("scf_max_iters")) {
    scf_max_iters = config["scf_max_iters"].get<int>();
  }
  double scf_tol = 1e-6;
  if (config.contains("scf_tol")) {
    scf_tol = config["scf_tol"].get<double>();
  }

  try {
    system_lib::CNDO2System mol_sys = system_lib::CNDO2System::from_files(
        atoms_file_path, basis_dir, num_alpha_electrons, num_beta_electrons,
        distance_units, use_indo);
    diis::solve_cndo(mol_sys, scf_max_iters, scf_tol);

    const double electronic_energy = mol_sys.compute_electronic_energy();
    const double nuclear_energy = mol_sys.compute_nuclear_energy();
    const double total_energy = electronic_energy + nuclear_energy;

    double isolated_atoms_total = 0.0;
    for (size_t i = 0; i < mol_sys.num_atoms(); ++i) {
      isolated_atoms_total += solve_isolated_atom_energy(
          mol_sys.get_atom(i), use_indo, scf_max_iters, scf_tol);
    }
    const double atomization_energy = isolated_atoms_total - total_energy;
    const double atomization_energy_kcal = atomization_energy * kEvToKcalPerMol;

    std::cout << std::fixed << std::setprecision(8);
    std::cout << "Method: " << (use_indo ? "INDO" : "CNDO/2") << "\n";
    std::cout << "Electronic energy (eV): " << electronic_energy << "\n";
    std::cout << "Nuclear energy (eV):    " << nuclear_energy << "\n";
    std::cout << "Total energy (eV):      " << total_energy << "\n";
    std::cout << "Atomization energy (eV, sum(E_atoms)-E_mol): "
              << atomization_energy << "\n";
    std::cout << "Atomization energy (kcal/mol): " << atomization_energy_kcal
          << "\n";
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n";
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}