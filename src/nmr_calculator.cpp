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

#include "system_lib/system_lib.h"
#include "solver_lib/solver_lib.h"
#include "nmr_lib/nmr_lib.h"

namespace fs = std::filesystem;
using json = nlohmann::json;

int main(int argc, char **argv) {
  // check that a config file is supplied
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " path/to/molecule_config.json" << " path/to/reference_config.json" << std::endl;
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
    std::cerr << "Path: " << reference_config_file_path << " does not exist" << std::endl;
    return EXIT_FAILURE;
  }
  std::ifstream reference_config_file(reference_config_file_path);
  json reference_config = json::parse(reference_config_file);

  // extract the important info from the config file
  fs::path atoms_file_path = config["atoms_file_path"];
  fs::path output_file_path = config["output_file_path"];
  int num_alpha_electrons = config["num_alpha_electrons"];
  int num_beta_electrons = config["num_beta_electrons"];

  fs::path reference_atoms_file_path = reference_config["atoms_file_path"];
  int reference_num_alpha_electrons = reference_config["num_alpha_electrons"];
  int reference_num_beta_electrons = reference_config["num_beta_electrons"];

  system_lib::CNDO2System main_sys = system_lib::CNDO2System::from_files(atoms_file_path, "./basis", num_alpha_electrons, num_beta_electrons);
  fixed_point::solve_cndo(main_sys);

  // compute sigma for each carbon atom and print it
  int num_carbons_main = 0;
  for (size_t iatom = 0; iatom < main_sys.num_atoms(); iatom++) {
      const auto& atom = main_sys.get_atom(iatom);
      if (atom.get_symbol() == "C") {
          ++num_carbons_main;
          double sigma_d = nmr_lib::calculate_sigma_d(main_sys, iatom);
          double sigma_p = nmr_lib::calculate_sigma_p(main_sys, iatom);
          double sigma = nmr_lib::calculate_sigma(main_sys, iatom);
          std::cout << std::format("Atom {}: sigma_d = {:.4f}, sigma_p = {:.4f}, sigma = {:.4f}\n", iatom, sigma_d, sigma_p, sigma);
      }
  }

  system_lib::CNDO2System reference_sys = system_lib::CNDO2System::from_files(reference_atoms_file_path, "./basis", reference_num_alpha_electrons, reference_num_beta_electrons);
  fixed_point::solve_cndo(reference_sys);

  // compute sigma for each carbon atom and print it
  int num_carbons_reference = 0;
  for (size_t iatom = 0; iatom < reference_sys.num_atoms(); iatom++) {
      const auto& atom = reference_sys.get_atom(iatom);
      if (atom.get_symbol() == "C") {
          num_carbons_reference++;
          double sigma_d = nmr_lib::calculate_sigma_d(reference_sys, iatom);
          double sigma_p = nmr_lib::calculate_sigma_p(reference_sys, iatom);
          double sigma = nmr_lib::calculate_sigma(reference_sys, iatom);
          std::cout << std::format("Reference Atom {}: sigma_d = {:.4f}, sigma_p = {:.4f}, sigma = {:.4f}\n", iatom, sigma_d, sigma_p, sigma);
      }
  }

  arma::mat chemical_shifts = arma::zeros(num_carbons_main, num_carbons_reference);

  for (size_t iatom_main = 0, iC_main = 0; iatom_main < main_sys.num_atoms(); iatom_main++) {
      const auto& atom_main = main_sys.get_atom(iatom_main);
      if (atom_main.get_symbol() != "C") continue;

      double sigma_main = nmr_lib::calculate_sigma(main_sys, iatom_main);

      for (size_t iatom_ref = 0, iC_ref = 0; iatom_ref < reference_sys.num_atoms(); iatom_ref++) {
          const auto& atom_ref = reference_sys.get_atom(iatom_ref);
          if (atom_ref.get_symbol() != "C") continue;

          double sigma_ref = nmr_lib::calculate_sigma(reference_sys, iatom_ref);
          chemical_shifts(iC_main, iC_ref) = sigma_ref - sigma_main;
          ++iC_ref;
      }
      ++iC_main;
  }


  // Print the chemical shifts
  std::cout << "Chemical Shifts (ppm):\n";
  for (size_t i = 0; i < chemical_shifts.n_rows; i++) {
      for (size_t j = 0; j < chemical_shifts.n_cols; j++) {
          std::cout << std::format("Main Atom {} vs Reference Atom {}: {:.4f} ", i, j, chemical_shifts(i, j)) << "ppm; \n";
      }
    std::cout << "\n";
  }
  
}