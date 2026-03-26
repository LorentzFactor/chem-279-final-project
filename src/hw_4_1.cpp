#include <fstream>
#include <iostream>
#include <stdexcept>

#include <armadillo>

#include <highfive/H5File.hpp>
#include <nlohmann/json.hpp>

#include "system_lib/system_lib.h"

namespace fs = std::filesystem;
using json = nlohmann::json;
using namespace HighFive;

int main(int argc, char *argv[]) {
  // Check args
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " path/to/config" << std::endl;
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

  fs::path atoms_file_path = config["atoms_file_path"];
  fs::path output_file_path = config["output_file_path"];
  int p = config["num_alpha_electrons"];
  int q = config["num_beta_electrons"];

  system_lib::System sys = system_lib::System::from_files(atoms_file_path, "./basis");

  arma::mat S = sys.compute_overlap_matrix();
  std::cout << S << std::endl;

  arma::mat reduced_gamma = sys.compute_reduced_gamma_matrix();
  arma::mat gamma = sys.compute_gamma_matrix();
  std::cout << gamma << std::endl;

  arma::mat beta = sys.compute_beta_matrix();
  std::cout << beta << std::endl;

  
  arma::mat p_alpha = arma::zeros(sys.num_orbitals(), sys.num_orbitals());
  arma::mat p_beta = arma::zeros(sys.num_orbitals(), sys.num_orbitals());

  std::pair<arma::mat, arma::mat> f_mats = sys.compute_cndo_f_matrix(p_alpha, p_beta);
  arma::mat& f_alpha = f_mats.first;
  arma::mat& f_beta = f_mats.second;
  std::cout << f_alpha << std::endl;
  std::cout << f_beta << std::endl;

  // check that output dir exists
  if (!fs::exists(output_file_path.parent_path())){
      fs::create_directories(output_file_path.parent_path()); 
  }
  
  // delete the file if it does exist (so that no old answers stay there by accident)
  if (fs::exists(output_file_path)){
      fs::remove(output_file_path); 
  }

  // Create high five file
  File output_file(output_file_path.string(), File::Overwrite);

  // Write out relevant data
  S.save(arma::hdf5_name(output_file_path, "S", arma::hdf5_opts::replace));
  f_alpha.save(arma::hdf5_name(output_file_path, "Fa_initial", arma::hdf5_opts::replace));
  f_beta.save(arma::hdf5_name(output_file_path, "Fb_initial", arma::hdf5_opts::replace));
  reduced_gamma.save(arma::hdf5_name(output_file_path, "gamma", arma::hdf5_opts::replace));

  return EXIT_SUCCESS;
}