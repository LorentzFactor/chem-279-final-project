#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <map>
#include <stdexcept>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

#include <armadillo>
#include <highfive/H5File.hpp>
#include <nlohmann/json.hpp> 

#include "system_lib/system_lib.h"
#include "solver_lib/solver_lib.h"

namespace fs = std::filesystem;
using json = nlohmann::json;
using namespace HighFive;

template <typename T>
void write_to_high_five(File output_file, std::string name, T data) {
     auto dataset = output_file.createDataSet<T>(name, DataSpace::From(data));
     dataset.write(data);
};

int main(int argc, char **argv) {
  // check that required args are supplied
  if (argc < 2 || argc > 3) {
    std::cerr << "Usage: " << argv[0]
              << " path/to/config.json [--indo|--cndo]" << std::endl;
    return EXIT_FAILURE;
  }

  bool use_indo = false;
  if (argc == 3) {
    const std::string method_flag = argv[2];
    if (method_flag == "--indo" || method_flag == "indo") {
      use_indo = true;
    } else if (method_flag == "--cndo" || method_flag == "cndo") {
      use_indo = false;
    } else {
      std::cerr << "Unknown method flag: " << method_flag << "\n"
                << "Use one of: --indo, --cndo" << std::endl;
      return EXIT_FAILURE;
    }
  }

  // parse the config file
  fs::path config_file_path(argv[1]);
  if (!fs::exists(config_file_path)) {
    std::cerr << "Path: " << config_file_path << " does not exist" << std::endl;
    return EXIT_FAILURE;
  }
  std::ifstream config_file(config_file_path);
  json config = json::parse(config_file);

  std::cout << "Here" << std::endl;

  // extract the important info from the config file
  fs::path atoms_file_path = config["atoms_file_path"];
  fs::path output_file_path = config["output_file_path"];
  int num_alpha_electrons = config["num_alpha_electrons"];
  int num_beta_electrons = config["num_beta_electrons"];

  system_lib::CNDO2System sys = system_lib::CNDO2System::from_files(
      atoms_file_path, "./basis", num_alpha_electrons, num_beta_electrons,
      system_lib::DistanceUnits::BOHR, use_indo);

  int num_atoms = sys.num_atoms(); // You will have to replace this with the
                                      // number of atoms in the molecule
  int num_basis_functions = sys.num_orbitals(); // you will have to replace this with the number of basis
                       // sets in the molecule
  int num_3D_dims = 3;

  // Your answers go in these objects
  // Information about the convention of the requirements 
  /*std::cout << "Order of columns for Suv_RA is as follows: (u,v)" << std::endl;
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
  */

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

  // Solve for ground state using fixed point method
  //fixed_point::solve_cndo(sys);

  // Solve for ground state using diis method
  diis::solve_cndo(sys, 1000, 1e-6, false, diis::InitialGuess::kAtomic);

  // compute Suv_RA
  //arma::cube Suv_RA_cube = sys.compute_overlap_matrix_gradient();

  // Convert it to expected (2d) output format
  //Suv_RA = Suv_RA_cube.reshape(3, num_basis_functions*num_basis_functions, 1).slice(0);

  // Do the same for gamma_RA
  //arma::cube gamma_RA_cube = sys.compute_gamma_matrix_gradient();
  //gammaAB_RA = gamma_RA_cube.reshape(3, num_atoms*num_atoms, 1).slice(0);

  // Compute electronic gradient
  //gradient_electronic = sys.E_electronic_dRA();

  // Compute nuclear gradient
  //gradient_nuclear = sys.E_nuclear_dRA();

  // Compute total gradient
  //gradient = gradient_electronic + gradient_nuclear;

  // You do not need to modify the code below this point

  // Set print configs
  std::cout << std::fixed << std::setprecision(4) << std::setw(8) << std::right;

  // check that output dir exists
  if (!fs::exists(output_file_path.parent_path())) {
    fs::create_directories(output_file_path.parent_path());
  }

  // delete the file if it does exist (so that no old answers stay there by
  // accident)
  if (fs::exists(output_file_path)) {
    fs::remove(output_file_path);
  }

  // Create high five file
  File output_file(output_file_path.string(), File::Overwrite);

  // write results to file

  
  arma::mat h_core = sys.compute_h_core();

  double electronic_energy = sys.compute_electronic_energy();
  double nuclear_energy = sys.compute_nuclear_energy();
  double total_energy = sys.compute_total_energy();

  arma::mat S = sys.compute_overlap_matrix();
  //arma::mat reduced_gamma = sys.compute_reduced_gamma_matrix();
  //arma::mat gamma = sys.compute_gamma_matrix();
  //arma::mat beta = sys.compute_beta_matrix();

  arma::rowvec Ea = sys.get_E_alpha();
  arma::rowvec Eb = sys.get_E_beta();
  Ea.save(arma::hdf5_name(output_file_path, "Ea", arma::hdf5_opts::replace));
  Eb.save(arma::hdf5_name(output_file_path, "Eb", arma::hdf5_opts::replace));
  h_core.save(arma::hdf5_name(output_file_path, "H_core", arma::hdf5_opts::replace));
  S.save(arma::hdf5_name(output_file_path, "S", arma::hdf5_opts::replace));
  //reduced_gamma.save(arma::hdf5_name(output_file_path, "gamma", arma::hdf5_opts::replace));    
  write_to_high_five(output_file, "electronic_energy", electronic_energy);
  //write_to_high_five(output_file, "nuclear_energy", nuclear_energy);
  write_to_high_five(output_file, "total_energy", total_energy);
}