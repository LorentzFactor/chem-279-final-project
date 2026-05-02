#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

#include <armadillo>
#include <nlohmann/json.hpp>

#include "nmr_lib/nmr_lib.h"
#include "solver_lib/solver_lib.h"
#include "system_lib/system_lib.h"

namespace fs = std::filesystem;
using json = nlohmann::json;

inline system_lib::DistanceUnits extract_config_units(json config) {
    if (config.find("distance_unit") != config.end()) {
      std::cout << "Extracted distance unit from config: " << config["distance_unit"] << std::endl;
      return config["distance_unit"] == "angstrom" ? system_lib::DistanceUnits::ANGSTROM : system_lib::DistanceUnits::BOHR;
    }
    return system_lib::DistanceUnits::BOHR;
}

int main(int argc, char **argv) {
  // check that a config file is supplied
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " path/to/molecule_config.json"
              << " path/to/reference_config.json" << std::endl;
    return EXIT_FAILURE;
  }

  // parse the main molecule config file
  fs::path config_file_path(argv[1]);
  if (!fs::exists(config_file_path)) {
    std::cerr << "Path: " << config_file_path << " does not exist" << std::endl;
    return EXIT_FAILURE;
  }
  std::ifstream config_file(config_file_path);
  json config = json::parse(config_file);

  // parse the reference molecule config file
  fs::path reference_config_file_path(argv[2]);
  if (!fs::exists(reference_config_file_path)) {
    std::cerr << "Path: " << reference_config_file_path << " does not exist"
              << std::endl;
    return EXIT_FAILURE;
  }
  std::ifstream reference_config_file(reference_config_file_path);
  json reference_config = json::parse(reference_config_file);

  // extract the important info from the config file
  fs::path atoms_file_path = config["atoms_file_path"];
  fs::path output_file_path = config["output_file_path"];
  int num_alpha_electrons = config["num_alpha_electrons"];
  int num_beta_electrons = config["num_beta_electrons"];
  double delta_E = config["delta_e"];
  system_lib::DistanceUnits distance_units = extract_config_units(config);

  fs::path reference_atoms_file_path = reference_config["atoms_file_path"];
  int reference_num_alpha_electrons = reference_config["num_alpha_electrons"];
  int reference_num_beta_electrons = reference_config["num_beta_electrons"];
  double reference_delta_E = reference_config["delta_e"];
  system_lib::DistanceUnits reference_distance_units = extract_config_units(reference_config);

  system_lib::CNDO2System main_sys = system_lib::CNDO2System::from_files(
    atoms_file_path, "./basis",
    num_alpha_electrons, num_beta_electrons,
    distance_units
  );
  fixed_point::solve_cndo(main_sys);

  CarbonGraph main_graph = nmr_lib::build_carbon_graph(main_sys);

  // Debug print for carbon-carbon adjacency list
  // std::cout << "Main molecule carbon adjacency:\n";
  // for (size_t i = 0; i < main_graph.adj.size(); ++i) {
  //   std::cout << "C_local " << i << " (atom "
  //             << main_graph.carbon_local_to_atom_idx[i] << "): ";
  //   for (size_t nbr : main_graph.adj[i]) {
  //     std::cout << nbr << " ";
  //   }
  //   std::cout << "\n";
  // }

  // Debug for delta_E on each carbon for main
  std::cout << "Delta_E for each carbon on main:\n";
  for (size_t i = 0; i < main_sys.num_atoms(); ++i) {
    if (main_sys.get_atom(i).get_atomic_number() == 6) {
      const size_t local_atom_idx = main_graph.atom_idx_to_carbon_local.at(i);
      double delta_e = nmr_lib::calculate_delta_E(main_graph, i);
      std::cout << "Carbon " << local_atom_idx << " delta_E: " << delta_e
                << '\n';
    }
  }

  // compute sigma for each carbon atom and print it
  int num_carbons_main = 0;
  for (size_t iatom = 0; iatom < main_sys.num_atoms(); iatom++) {
    const auto &atom = main_sys.get_atom(iatom);
    if (atom.get_symbol() == "C") {
      ++num_carbons_main;
      double sigma_d = nmr_lib::calculate_sigma_d(main_sys, iatom);
      double sigma_p = nmr_lib::calculate_sigma_p(main_sys, main_graph, iatom);
      double sigma = nmr_lib::calculate_sigma(main_sys, main_graph, iatom);
      std::cout << std::format(
          "Atom {}: sigma_d = {:.4f}, sigma_p = {:.4f}, sigma = {:.4f}\n",
          iatom, sigma_d, sigma_p, sigma);
    }
  }

  system_lib::CNDO2System reference_sys = system_lib::CNDO2System::from_files(
    reference_atoms_file_path, "./basis",
    reference_num_alpha_electrons, reference_num_beta_electrons,
    reference_distance_units
  );
  diis::solve_cndo(reference_sys);
  CarbonGraph ref_graph = nmr_lib::build_carbon_graph(reference_sys);

  // Debug for delta_E on each carbon for reference
  std::cout << "Delta_E for each carbon on reference:\n";
  for (size_t i = 0; i < reference_sys.num_atoms(); ++i) {
    if (reference_sys.get_atom(i).get_atomic_number() == 6) {
      const size_t local_atom_idx = ref_graph.atom_idx_to_carbon_local.at(i);
      double delta_e = nmr_lib::calculate_delta_E(ref_graph, i);
      std::cout << "Carbon " << local_atom_idx << " delta_E: " << delta_e
                << '\n';
    }
  }

  // compute sigma for each carbon atom and print it
  int num_carbons_reference = 0;
  for (size_t iatom = 0; iatom < reference_sys.num_atoms(); iatom++) {
    const auto &atom = reference_sys.get_atom(iatom);
    if (atom.get_symbol() == "C") {
      num_carbons_reference++;
      double sigma_d = nmr_lib::calculate_sigma_d(reference_sys, iatom);
      double sigma_p =
          nmr_lib::calculate_sigma_p(reference_sys, ref_graph, iatom);
      double sigma = nmr_lib::calculate_sigma(reference_sys, ref_graph, iatom);
      std::cout << std::format("Reference Atom {}: sigma_d = {:.4f}, sigma_p = "
                               "{:.4f}, sigma = {:.4f}\n",
                               iatom, sigma_d, sigma_p, sigma);
    }
  }

  arma::mat chemical_shifts =
      arma::zeros(num_carbons_main, num_carbons_reference);

  for (size_t iatom_main = 0, iC_main = 0; iatom_main < main_sys.num_atoms();
       iatom_main++) {
    const auto &atom_main = main_sys.get_atom(iatom_main);
    if (atom_main.get_symbol() != "C")
      continue;

    double sigma_main =
        nmr_lib::calculate_sigma(main_sys, main_graph, iatom_main);

    // delta_c for reference -> currently not using!!!
    const double delta_c = 0.0;
    for (size_t iatom_ref = 0, iC_ref = 0;
         iatom_ref < reference_sys.num_atoms(); iatom_ref++) {
      const auto &atom_ref = reference_sys.get_atom(iatom_ref);
      if (atom_ref.get_symbol() != "C")
        continue;

      double sigma_ref =
          nmr_lib::calculate_sigma(reference_sys, ref_graph, iatom_ref);
      chemical_shifts(iC_main, iC_ref) = (sigma_ref - sigma_main) + delta_c;
      ++iC_ref;
    }
    ++iC_main;
  }

  // Print the chemical shifts
  std::cout << "Chemical Shifts (ppm):\n";
  for (size_t i = 0; i < chemical_shifts.n_rows; i++) {
    for (size_t j = 0; j < chemical_shifts.n_cols; j++) {
      std::cout << std::format("Main Atom {} vs Reference Atom {}: {:.4f} ", i,
                               j, chemical_shifts(i, j))
                << "ppm; \n";
    }
    std::cout << "\n";
  }
}