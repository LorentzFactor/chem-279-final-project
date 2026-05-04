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

#include "nmr_lib/nmr_lib.h"
#include "solver_lib/solver_lib.h"
#include "system_lib/system_lib.h"

namespace fs = std::filesystem;
using json = nlohmann::json;

inline system_lib::DistanceUnits extract_config_units(json config) {
    if (config.find("distance_unit") != config.end()) {
      std::cout << "Extracted distance unit from config: " << config["distance_unit"] << std::endl;
      return config["distance_unit"] == "angstrom" ? system_lib::DistanceUnits::ANGSTROM : system_lib::DistanceUnits::BOHR;
    }
    return system_lib::DistanceUnits::BOHR;
}

system_lib::CNDO2System load_from_config(json config) {
    fs::path atoms_file_path = config["atoms_file_path"];
    int num_alpha_electrons = config["num_alpha_electrons"];
    int num_beta_electrons = config["num_beta_electrons"];
    system_lib::DistanceUnits distance_units = extract_config_units(config);

    return system_lib::CNDO2System::from_files(
        atoms_file_path, "./basis",
        num_alpha_electrons, num_beta_electrons,
        distance_units
    );
}

int main(int argc, char **argv) {
  // check that a config file is supplied
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " path/to/test_config.json"
              << std::endl;
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

  fs::path methane_config_path = config["methane_reference_file_path"];
  std::ifstream methane_config_file = std::ifstream(methane_config_path);
  json methane_config = json::parse(methane_config_file);
  system_lib::CNDO2System methane = load_from_config(methane_config);
  fixed_point::solve_cndo(methane);

  CarbonGraph main_graph = nmr_lib::build_carbon_graph(methane);
  double sigma_d_methane = nmr_lib::calculate_sigma_d(methane, 0);
  double sigma_p_methane = nmr_lib::calculate_sigma_p(methane, main_graph, 0);
  double sigma_methane = nmr_lib::calculate_sigma(methane, main_graph, 0);
  static const double methane_benzene_chem_shift = 130.8;

  std::vector<double> chem_shifts;
  chem_shifts.reserve(config["test_molecules"].size()*3); // assuming max 3 spectrally distinct carbons per molecule
  for (const auto& test_molecule_data : config["test_molecules"]) {
    std::string molecule_name = test_molecule_data["name"];
    fs::path test_molecule_config_path = test_molecule_data["path"];
    std::vector<double> expected_peaks = test_molecule_data["chem_shifts"];
    std::ifstream test_molecule_config_file = std::ifstream(test_molecule_config_path);
    json molecule_config = json::parse(test_molecule_config_file);
    system_lib::CNDO2System test_molecule = load_from_config(molecule_config);
    diis::solve_cndo(test_molecule);
    std::vector<double> peaks = nmr_lib::get_nmr_peaks(test_molecule);

    std::vector<double> absolute_errors;
    absolute_errors.reserve(expected_peaks.size());

    for (size_t i = 0; i < expected_peaks.size() && i < peaks.size(); ++i) {
      double error = std::abs(peaks[i] - expected_peaks[i]);
      absolute_errors.push_back(error);
    }
    std::vector<double> relative_errors;
    relative_errors.reserve(expected_peaks.size());
    for (size_t i = 0; i < expected_peaks.size() && i < peaks.size(); ++i) {
      double error = std::abs(peaks[i] - expected_peaks[i]) / std::abs(expected_peaks[i]);
      relative_errors.push_back(error);
    }

    std::cout << "NMR peaks for " << molecule_name << ": ";
    for (const auto& peak : peaks) {
      std::cout << peak << " ppm, ";
      chem_shifts.push_back(peak);
    }
    
    for (size_t i = 0; i < expected_peaks.size() && i < peaks.size(); ++i) {
      std::cout << "\n\tExpected peak: " << expected_peaks[i] << " ppm, "
                << "Absolute error: " << absolute_errors[i] << " ppm, "
                << "Relative error: " << relative_errors[i] * 100 << "%";
    }
    std::cout << std::endl;
  }
}