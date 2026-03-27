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


template <typename T>
void write_to_high_five(File output_file, std::string name, T data) {
     auto dataset = output_file.createDataSet<T>(name, DataSpace::From(data));
     dataset.write(data);
};

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

  system_lib::CNDO2System sys = system_lib::CNDO2System::from_files(atoms_file_path, "./basis", p, q);

  arma::mat S = sys.compute_overlap_matrix();
  std::cout << S << std::endl;

  arma::mat reduced_gamma = sys.compute_reduced_gamma_matrix();
  arma::mat gamma = sys.compute_gamma_matrix();
  std::cout << gamma << std::endl;

  arma::mat beta = sys.compute_beta_matrix();
  std::cout << beta << std::endl;

  arma::mat p_alpha = arma::zeros(sys.num_orbitals(), sys.num_orbitals());
  arma::mat p_beta = arma::zeros(sys.num_orbitals(), sys.num_orbitals());
  arma::mat p_tot = p_alpha + p_beta;

  arma::mat p_a_old;
  arma::mat p_b_old;
  arma::mat p_tot_old;

  arma::mat f_alpha = sys.get_f_alpha();
  arma::mat f_beta = sys.get_f_beta();

  f_alpha.save(arma::hdf5_name(output_file_path, "Fa_initial", arma::hdf5_opts::replace));
  f_beta.save(arma::hdf5_name(output_file_path, "Fb_initial", arma::hdf5_opts::replace));
  
  std::cout << f_alpha << std::endl;
  std::cout << f_beta << std::endl;

  arma::vec E_alpha;
  arma::mat C_alpha;
  arma::vec E_beta;
  arma::mat C_beta;

  for(size_t i = 0; i < 1e3; ++i) {
    arma::eig_sym(E_alpha, C_alpha,  sys.get_f_alpha());
    arma::eig_sym(E_beta, C_beta, sys.get_f_beta());

    p_a_old = sys.get_p_alpha();
    p_b_old = sys.get_p_beta();
    p_tot_old = p_a_old + p_b_old;

    p_alpha = C_alpha.cols(arma::span(0, p-1)) * C_alpha.cols(arma::span(0, p-1)).t();
    p_beta = C_beta.cols(arma::span(0, q-1)) * C_beta.cols(arma::span(0, q-1)).t();
    sys.set_p(p_alpha, p_beta);
    p_tot = p_alpha + p_beta;

    std::cout << "Step: " << i+1 << std::endl;
    std::cout << "Energy: " << sys.compute_total_energy() << std::endl;

    /*std::cout << "C alpha: \n" << C_alpha << std::endl;
    std::cout << "p alpha: \n" << p_alpha << std::endl;
    std::cout << "p tot: \n" << p_tot << std::endl;*/
    if (arma::approx_equal(p_tot, p_tot_old, "absdiff", 1e-6)) {
      std::cout <<"n iters: " << i << std::endl;
      break;
    }
  }


  std::cout << "E alpha: \n" << E_alpha << std::endl;
  std::cout << "P alpha prev: \n" <<p_a_old<<std::endl;
  std::cout<< "P alpha new: \n" <<p_alpha<<std::endl;

  arma::mat h_core = sys.compute_h_core();

  double electronic_energy = sys.compute_electronic_energy();
  double nuclear_energy = sys.compute_nuclear_energy();
  double total_energy = sys.compute_total_energy();

  arma::rowvec Ea = E_alpha.as_row();
  arma::rowvec Eb = E_beta.as_row();
  Ea.save(arma::hdf5_name(output_file_path, "Ea", arma::hdf5_opts::replace));
  Eb.save(arma::hdf5_name(output_file_path, "Eb", arma::hdf5_opts::replace));
  h_core.save(arma::hdf5_name(output_file_path, "H_core", arma::hdf5_opts::replace));
  S.save(arma::hdf5_name(output_file_path, "S", arma::hdf5_opts::replace));
  reduced_gamma.save(arma::hdf5_name(output_file_path, "gamma", arma::hdf5_opts::replace));
  write_to_high_five(output_file, "electronic_energy", electronic_energy);
  write_to_high_five(output_file, "nuclear_energy", nuclear_energy);
  write_to_high_five(output_file, "total_energy", total_energy);

  return EXIT_SUCCESS;
}