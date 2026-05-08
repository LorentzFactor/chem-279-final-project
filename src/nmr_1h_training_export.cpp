// Batch-export ¹H shielding features (σ_para, σ_dia, etc.) for empirical δ fitting.
// Usage: nmr_1h_training_export [--cndo|--indo|--mindo] path/to/training_list.json
// JSON schema: see sample_input/nmr_1h_training_export.json

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <armadillo>
#include <nlohmann/json.hpp>

#include "solver_lib/solver_lib.h"
#include "system_lib/system_lib.h"

namespace fs = std::filesystem;
using json = nlohmann::json;

struct ShieldingScaling {
  double para_ppm_factor = 0.0;
  double dia_ppm_factor = 0.0;
  bool include_dia = true;
};

static ShieldingScaling build_default_scaling() {
  constexpr double alpha_inv = 137.035999084;
  const double alpha2 = 1.0 / (alpha_inv * alpha_inv);
  ShieldingScaling s;
  s.para_ppm_factor = alpha2 * 1.0e6;
  s.dia_ppm_factor = s.para_ppm_factor / 3.0;
  return s;
}

static system_lib::DistanceUnits parse_distance_unit(const json &mol) {
  if (!mol.contains("distance_unit")) {
    return system_lib::DistanceUnits::BOHR;
  }
  const std::string u = mol["distance_unit"].get<std::string>();
  if (u == "angstrom") {
    return system_lib::DistanceUnits::ANGSTROM;
  }
  return system_lib::DistanceUnits::BOHR;
}

static void collect_hydrogen_atom_indices(const system_lib::CNDO2System &sys,
                                          std::vector<size_t> *out_indices) {
  out_indices->clear();
  for (size_t i = 0; i < sys.num_atoms(); ++i) {
    if (sys.get_atom(i).get_atomic_number() == 1) {
      out_indices->push_back(i);
    }
  }
}

static std::string csv_escape(const std::string &s) {
  if (s.find_first_of(",\"\n\r") != std::string::npos) {
    std::ostringstream oss;
    oss << '"';
    for (char c : s) {
      if (c == '"') {
        oss << "\"\"";
      } else {
        oss << c;
      }
    }
    oss << '"';
    return oss.str();
  }
  return s;
}

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr
        << "Usage: " << argv[0]
        << " [--cndo|--indo|--mindo] path/to/training_list.json\n"
        << "Writes CSV path from JSON key \"output_csv\" (see "
           "sample_input/nmr_1h_training_export.json).\n";
    return EXIT_FAILURE;
  }

  bool use_indo = false;
  std::string method_flag(argv[1]);
  if (method_flag == "--cndo") {
    use_indo = false;
  } else if (method_flag == "--indo") {
    use_indo = true;
  } else {
    std::cerr << "First argument must be --cndo, --indo, or --mindo.\n";
    return EXIT_FAILURE;
  }

  const fs::path list_path(argv[2]);
  if (!fs::exists(list_path)) {
    std::cerr << "File not found: " << list_path << '\n';
    return EXIT_FAILURE;
  }

  std::ifstream in(list_path);
  const json root = json::parse(in);

  if (!root.contains("output_csv")) {
    std::cerr << "JSON must contain \"output_csv\" (path to write).\n";
    return EXIT_FAILURE;
  }
  if (!root.contains("molecules") || !root["molecules"].is_array()) {
    std::cerr << "JSON must contain \"molecules\": [ ... ] array.\n";
    return EXIT_FAILURE;
  }

  std::string basis_dir = "./basis";
  if (root.contains("basis_dir")) {
    basis_dir = root["basis_dir"].get<std::string>();
  }
  int field_dir = 2;
  if (root.contains("field_dir")) {
    field_dir = root["field_dir"].get<int>();
  }
  if (field_dir < 0 || field_dir > 2) {
    std::cerr << "field_dir must be 0, 1, or 2.\n";
    return EXIT_FAILURE;
  }

  double epsilon = 1.0e-5;
  if (root.contains("epsilon")) {
    epsilon = root["epsilon"].get<double>();
  }

  ShieldingScaling scaling = build_default_scaling();
  if (root.contains("para_ppm_factor")) {
    scaling.para_ppm_factor = root["para_ppm_factor"].get<double>();
  }
  if (root.contains("dia_ppm_factor")) {
    scaling.dia_ppm_factor = root["dia_ppm_factor"].get<double>();
  }
  if (root.contains("include_dia")) {
    scaling.include_dia = root["include_dia"].get<bool>();
  }

  const fs::path csv_path = root["output_csv"].get<std::string>();
  const fs::path csv_parent = csv_path.parent_path();
  if (!csv_parent.empty()) {
    fs::create_directories(csv_parent);
  }

  std::ofstream out(csv_path);
  if (!out) {
    std::cerr << "Could not open for write: " << csv_path << '\n';
    return EXIT_FAILURE;
  }

  out << "molecule,h_rank,atom_index,raw_para_iso,sigma_para_ppm,sigma_dia_ppm,"
         "sigma_total_ppm,rho_H,delta_exp_ppm\n";

  const json &mols = root["molecules"];
  for (size_t mi = 0; mi < mols.size(); ++mi) {
    const json &mol = mols[mi];
    const std::string label = mol.value("label", std::string("molecule_") + std::to_string(mi));
    const fs::path atoms_path = mol.at("atoms_file_path").get<std::string>();
    const int n_alpha = mol.at("num_alpha_electrons").get<int>();
    const int n_beta = mol.at("num_beta_electrons").get<int>();
    const system_lib::DistanceUnits du = parse_distance_unit(mol);

    std::cout << "Molecule " << (mi + 1) << "/" << mols.size() << " " << label
              << " (" << atoms_path << ") ..." << std::endl;

    system_lib::CNDO2System real_sys = system_lib::CNDO2System::from_files(
      atoms_path.string(), basis_dir, n_alpha, n_beta, du, use_indo);
    // Same as nmr_1H_calculator: DIIS for real SCF; plain fixed-point often fails
    // for larger molecules (e.g. n-butane).
    diis::solve_cndo(real_sys);

    std::vector<size_t> h_atoms;
    collect_hydrogen_atom_indices(real_sys, &h_atoms);

    system_lib::CNDO2SystemComplex cx =
        system_lib::CNDO2SystemComplex::from_files(
        atoms_path.string(), basis_dir, n_alpha, n_beta, du, use_indo);
    cx.set_magnetic_field(field_dir, 0.0);
    diis::solve_cndo(cx);

    std::vector<double> delta_exp;
    if (mol.contains("experimental_shift_ppm")) {
      const auto &arr = mol["experimental_shift_ppm"];
      if (!arr.is_array()) {
        std::cerr << label << ": experimental_shift_ppm must be an array.\n";
        return EXIT_FAILURE;
      }
      if (arr.size() != h_atoms.size()) {
        std::cerr << label << ": experimental_shift_ppm length " << arr.size()
                  << " != hydrogen count " << h_atoms.size() << '\n';
        return EXIT_FAILURE;
      }
      for (size_t j = 0; j < arr.size(); ++j) {
        delta_exp.push_back(arr[j].get<double>());
      }
    }

    for (size_t h_rank = 0; h_rank < h_atoms.size(); ++h_rank) {
      const size_t atom_idx = h_atoms[h_rank];
      const RealMat tensor = cx.compute_proton_shielding_tensor(atom_idx, epsilon);
      const double raw_para_iso = arma::trace(tensor) / 3.0;
      const double sigma_para = raw_para_iso * scaling.para_ppm_factor;
      const double rho_H = real_sys.get_electron_density(atom_idx);
      const double sigma_dia =
          scaling.include_dia ? rho_H * scaling.dia_ppm_factor : 0.0;
      const double sigma_total = sigma_para + sigma_dia;

      std::ostringstream dexp_cell;
      if (!delta_exp.empty()) {
        dexp_cell << std::setprecision(10) << delta_exp[h_rank];
      }

      out << csv_escape(label) << ',' << h_rank << ',' << atom_idx << ','
          << std::setprecision(12) << raw_para_iso << ',' << sigma_para << ','
          << sigma_dia << ',' << sigma_total << ',' << rho_H << ','
          << dexp_cell.str() << '\n';
    }
  }

  std::cout << "Wrote " << csv_path << std::endl;
  return EXIT_SUCCESS;
}
