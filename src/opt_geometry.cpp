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

#include "integrate.h"
#include "orbitals.h"

#include <armadillo>

#include <highfive/H5File.hpp>
#include <nlohmann/json.hpp>

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

  fs::path basis_path{"basis/"};
  Molecule molecule(atoms_file_path, basis_path, num_alpha_electrons,
                    num_beta_electrons);

  molecule.setConfigPath(config_file_path);

  std::cout << "Results for " << config_file_path << '\n';

  std::cout << "Initial coords:\n";
  molecule.printCoords();

  molecule.steepestDescentOptimizer(0.001, 1e-4);

  std::cout << "Optimized coords:\n";
  molecule.printCoords();

  molecule.geometricProperties();
}