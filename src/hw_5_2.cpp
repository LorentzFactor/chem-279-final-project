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



  int num_atoms = 0; // You will have to replace this with the
                                      // number of atoms in the molecule
  int num_basis_functions = 0; // you will have to replace this with the number of basis
                       // sets in the molecule
  int num_3D_dims = 3;

  // Your answers go in these objects
  // Information about the convention of the requirements 
  std::cout << "Order of columns for Suv_RA is as follows: (u,v)" << std::endl;
  for (int u = 0; u < num_basis_functions; u++) {
    for (int v = 0; v < num_basis_functions; v++) {
      std::cout << std::format("({},{}) ", u, v);
    }
  }
  std::cout << std::endl;

  std::cout << "Order of columns for gammaAB_RA is as follows: (A,B)"
            << std::endl;
  for (int A = 0; A < num_atoms; A++) {
    for (int B = 0; B < num_atoms; B++) {
      std::cout << std::format("({},{}) ", A, B);
    }
  }
  std::cout << std::endl;

  std::cout << "Order of rows is as follows" << std::endl;
  std::cout << "x" << std::endl;
  std::cout << "y" << std::endl;
  std::cout << "z" << std::endl;

  arma::mat Suv_RA(num_3D_dims, num_basis_functions * num_basis_functions);
  // Ideally, this would be (3, n_funcs, n_funcs) rank-3 tensor
  // but we're flattening (n-funcs, n-atoms) into a single dimension (n-funcs ^
  // 2) this is because tensors are not supported in Eigen and I want students
  // to be able to submit their work in a consistent format
  arma::mat gammaAB_RA(num_3D_dims, num_atoms * num_atoms);
  // This is the same story, ideally, this would be (3, num_atoms, num_atoms)
  // instead of (3, num_atoms ^ 2)
  arma::mat gradient_nuclear(num_3D_dims, num_atoms);
  arma::mat gradient_electronic(num_3D_dims, num_atoms);
  arma::mat gradient(num_3D_dims, num_atoms);

  // most of your code will go here



  // You do not need to modify the code below this point

  // Set print configs
  std::cout << std::fixed << std::setprecision(4) << std::setw(8) << std::right;

  // inspect your answer via printing
  Suv_RA.print("Suv_RA");
  gammaAB_RA.print("gammaAB_RA");
  gradient_nuclear.print("gradient_nuclear");
  gradient_electronic.print("gradient_electronic");
  gradient.print("gradient");

  // check that output dir exists
  if (!fs::exists(output_file_path.parent_path())) {
    fs::create_directories(output_file_path.parent_path());
  }

  // delete the file if it does exist (so that no old answers stay there by
  // accident)
  if (fs::exists(output_file_path)) {
    fs::remove(output_file_path);
  }

  // write results to file
  Suv_RA.save(
      arma::hdf5_name(output_file_path, "Suv_RA",
                      arma::hdf5_opts::append + arma::hdf5_opts::trans));
  gammaAB_RA.save(
      arma::hdf5_name(output_file_path, "gammaAB_RA",
                      arma::hdf5_opts::append + arma::hdf5_opts::trans));
  gradient_nuclear.save(
      arma::hdf5_name(output_file_path, "gradient_nuclear",
                      arma::hdf5_opts::append + arma::hdf5_opts::trans));
  gradient_electronic.save(
      arma::hdf5_name(output_file_path, "gradient_electronic",
                      arma::hdf5_opts::append + arma::hdf5_opts::trans));
  gradient.save(
      arma::hdf5_name(output_file_path, "gradient",
                      arma::hdf5_opts::append + arma::hdf5_opts::trans));
}