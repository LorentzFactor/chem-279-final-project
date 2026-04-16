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

  int num_atoms = sys.num_atoms(); // You will have to replace this with the
                                      // number of atoms in the molecule
  int num_basis_functions = sys.num_orbitals(); // you will have to replace this with the number of basis
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

  // Converge output

    arma::mat S = sys.compute_overlap_matrix();
    arma::mat reduced_gamma = sys.compute_reduced_gamma_matrix();
    arma::mat gamma = sys.compute_gamma_matrix();
    arma::mat beta = sys.compute_beta_matrix();

    int p = num_alpha_electrons;
    int q = num_beta_electrons;

    arma::mat p_alpha = arma::mat(sys.num_orbitals(), sys.num_orbitals(), arma::fill::zeros) * (p/(sys.num_orbitals()*sys.num_orbitals()));
    arma::mat p_beta = arma::mat(sys.num_orbitals(), sys.num_orbitals(), arma::fill::zeros) *(q/(sys.num_orbitals()*sys.num_orbitals()));
    arma::mat p_tot = p_alpha + p_beta;
    sys.set_p(p_alpha, p_beta);

    arma::mat p_a_old;
    arma::mat p_b_old;
    arma::mat p_tot_old;

    const arma::mat& f_alpha = sys.get_f_alpha();
    const arma::mat& f_beta = sys.get_f_beta();

    f_alpha.save(arma::hdf5_name(output_file_path, "Fa_initial", arma::hdf5_opts::replace));
    f_beta.save(arma::hdf5_name(output_file_path, "Fb_initial", arma::hdf5_opts::replace));

    for(size_t i = 0; i < 1e3; ++i) {

      p_a_old = sys.get_p_alpha();
      p_b_old = sys.get_p_beta();
      p_tot_old = p_a_old + p_b_old;

      p_alpha = sys.get_occupied_MOs_alpha() * sys.get_occupied_MOs_alpha().t();
      p_beta = sys.get_occupied_MOs_beta() * sys.get_occupied_MOs_beta().t();
      
      sys.set_p(p_alpha, p_beta);
      
      p_tot = p_alpha + p_beta;

      std::cout << "Step: " << i+1 << std::endl;
      std::cout << "Energy: " << std::setprecision(9) << sys.compute_total_energy() << std::endl;

      if (
        arma::approx_equal(p_alpha, p_a_old, "absdiff", 1e-6) &&
        arma::approx_equal(p_beta, p_b_old, "absdiff", 1e-6)
        ) {
        std::cout <<"n iters: " << i << std::endl;
        break;
      }
    }

  // compute Suv_RA
  arma::cube Suv_RA_cube = sys.compute_overlap_matrix_gradient();

  // Convert it to expected (2d) output format
  Suv_RA = Suv_RA_cube.reshape(3, num_basis_functions*num_basis_functions, 1).slice(0);

  // Do the same for gamma_RA
  arma::cube gamma_RA_cube = sys.compute_gamma_matrix_gradient();
  gammaAB_RA = gamma_RA_cube.reshape(3, num_atoms*num_atoms, 1).slice(0);

  // Compute electronic gradient
  gradient_electronic = sys.E_electronic_dRA();

  // Compute nuclear gradient
  gradient_nuclear = sys.E_nuclear_dRA();

  // Compute total gradient
  gradient = gradient_electronic + gradient_nuclear;

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