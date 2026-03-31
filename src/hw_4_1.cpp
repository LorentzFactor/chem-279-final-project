#include <fstream>
#include <iostream>
#include <stdexcept>
#include <iomanip>
#include <limits>

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
  arma::mat reduced_gamma = sys.compute_reduced_gamma_matrix();
  arma::mat gamma = sys.compute_gamma_matrix();
  arma::mat beta = sys.compute_beta_matrix();

  arma::mat p_alpha = 5*arma::mat(sys.num_orbitals(), sys.num_orbitals(), arma::fill::randu) * (p/(sys.num_orbitals()*sys.num_orbitals()));
  arma::mat p_beta = arma::mat(sys.num_orbitals(), sys.num_orbitals(), arma::fill::randu) *(q/(sys.num_orbitals()*sys.num_orbitals()));
  arma::mat p_tot = p_alpha + p_beta;
  sys.set_p(p_alpha, p_beta);

  arma::mat p_a_old;
  arma::mat p_b_old;
  arma::mat p_tot_old;

  const arma::mat& f_alpha = sys.get_f_alpha();
  const arma::mat& f_beta = sys.get_f_beta();

  f_alpha.save(arma::hdf5_name(output_file_path, "Fa_initial", arma::hdf5_opts::replace));
  f_beta.save(arma::hdf5_name(output_file_path, "Fb_initial", arma::hdf5_opts::replace));

  std::cout << "P: " << p << " Q: " << q << std::endl;

  for(size_t i = 0; i < 1e3; ++i) {

    p_a_old = sys.get_p_alpha();
    p_b_old = sys.get_p_beta();
    p_tot_old = p_a_old + p_b_old;

    p_alpha = sys.get_occupied_MOs_alpha() * sys.get_occupied_MOs_alpha().t();
    p_beta = sys.get_occupied_MOs_beta() * sys.get_occupied_MOs_beta().t();
    
    sys.set_p(p_alpha, p_beta);

    /*double energy_neutral = sys.compute_total_energy();
    int delta = 0;
    double energy_best = energy_neutral;

    if (p != 0 && q != sys.num_orbitals() && p/2 > q/2) {
      sys.set_nelectrons(p-1, q+1);
      p_alpha = sys.get_occupied_MOs_alpha() * sys.get_occupied_MOs_alpha().t();
      p_beta = sys.get_occupied_MOs_beta() * sys.get_occupied_MOs_beta().t();
      sys.set_p(p_alpha, p_beta);
      double energy_min = sys.compute_total_energy();
      if (energy_min + std::numeric_limits<double>::epsilon() < energy_best) { 
        delta = -1;
      }
      sys.set_nelectrons(p+1, q-1);
    }

    if (q != 0 && p != sys.num_orbitals()) {
      sys.set_nelectrons(p+1, q-1);
      p_alpha = sys.get_occupied_MOs_alpha() * sys.get_occupied_MOs_alpha().t();
      p_beta = sys.get_occupied_MOs_beta() * sys.get_occupied_MOs_beta().t();
      sys.set_p(p_alpha, p_beta);
      double energy_plus = sys.compute_total_energy();
      // <= here instead of < because hund's rule
      if (energy_plus <= energy_best + std::numeric_limits<double>::epsilon()) {
        delta = 1;
      }
      sys.set_nelectrons(p-1, q+1);
    }
    p += delta;
    q -= delta;
    

    std::cout << "P: " << p << " Q: " << q << std::endl;

    sys.set_p(p_alpha, p_beta);
    sys.set_nelectrons(sys.get_nalpha() + delta, sys.get_nbeta() - 1);
    */
    
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

  arma::mat h_core = sys.compute_h_core();

  double electronic_energy = sys.compute_electronic_energy();
  double nuclear_energy = sys.compute_nuclear_energy();
  double total_energy = sys.compute_total_energy();

  std::cout << "total energy" << sys.compute_total_energy()<< std::endl;
  std::cout << "P" << p << "Q" << q << std::endl;

  std::cout << "density at 0.0 1.426 -0.8876 " << sys.get_electron_density({0.0, 1.426, -0.8876}) << std::endl;

  arma::cube spatial_electron_density = sys.get_electron_density_3d_grid({-3, 3}, {-3, 3}, {-3, 3}, 100);

  arma::rowvec Ea = sys.get_E_alpha();
  arma::rowvec Eb = sys.get_E_beta();
  Ea.save(arma::hdf5_name(output_file_path, "Ea", arma::hdf5_opts::replace));
  Eb.save(arma::hdf5_name(output_file_path, "Eb", arma::hdf5_opts::replace));
  h_core.save(arma::hdf5_name(output_file_path, "H_core", arma::hdf5_opts::replace));
  S.save(arma::hdf5_name(output_file_path, "S", arma::hdf5_opts::replace));
  reduced_gamma.save(arma::hdf5_name(output_file_path, "gamma", arma::hdf5_opts::replace));
  spatial_electron_density.save(arma::hdf5_name(output_file_path, "spatial_density", arma::hdf5_opts::replace));
  write_to_high_five(output_file, "electronic_energy", electronic_energy);
  write_to_high_five(output_file, "nuclear_energy", nuclear_energy);
  write_to_high_five(output_file, "total_energy", total_energy);

  return EXIT_SUCCESS;
}