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
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " path/to/config.json" << std::endl;
    return EXIT_FAILURE;
  }

  // parse the config file
  fs::path config_file_path(argv[1]);
  if (!fs::exists(config_file_path)) {
    std::cerr << "Path: " << config_file_path << " does not exist" << std::endl;
    return EXIT_FAILURE;
  }
  std::ifstream config_file(config_file_path);
  json config = json::parse(config_file);

  // extract the important info from the config file
  fs::path atoms_file_path = config["atoms_file_path"];
  fs::path output_file_path = config["output_file_path"];
  int num_alpha_electrons = config["num_alpha_electrons"];
  int num_beta_electrons = config["num_beta_electrons"];

  system_lib::CNDO2System sys = system_lib::CNDO2System::from_files(atoms_file_path, "./basis", num_alpha_electrons, num_beta_electrons);
  fixed_point::solve_cndo(sys);

  // compute sigma for each carbon atom and print it
    for (size_t iatom = 0; iatom < sys.num_atoms(); iatom++) {
        const auto& atom = sys.get_atom(iatom);
        if (atom.get_symbol() == "C") {
            double sigma_d = nmr_lib::calculate_sigma_d(sys, iatom);
            double sigma_p = nmr_lib::calculate_sigma_p(sys, iatom);
            double sigma = nmr_lib::calculate_sigma(sys, iatom);
            std::cout << std::format("Atom {}: sigma_d = {:.4f}, sigma_p = {:.4f}, sigma = {:.4f}\n", iatom, sigma_d, sigma_p, sigma);
        }
    }
}