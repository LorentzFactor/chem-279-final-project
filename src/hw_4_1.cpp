#include <fstream>
#include <iostream>
#include <stdexcept>

#include <armadillo>

#include <highfive/H5File.hpp>
#include <nlohmann/json.hpp>

#include "system_lib/system_lib.h"

namespace fs = std::filesystem;
using json = nlohmann::json;

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

  system_lib::System::from_files(atoms_file_path, "./basis");

  // check that output dir exists
  if (!fs::exists(output_file_path.parent_path())){
      fs::create_directories(output_file_path.parent_path()); 
  }
  
  // delete the file if it does exist (so that no old answers stay there by accident)
  if (fs::exists(output_file_path)){
      fs::remove(output_file_path); 
  }

  return EXIT_SUCCESS;
}