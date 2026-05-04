// 1H NMR chemical shifts (CNDO/2 + complex SCF with magnetic-field perturbation
// i·λ·L in the Fock build). Layout mirrors nmr_13C_calculator.cpp; Ramsey σ(H)
// and δ in ppm are wired once nmr_lib exposes proton shielding.

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <optional>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

#include <armadillo>
#include <nlohmann/json.hpp>

#include "solver_lib/solver_lib.h"
#include "system_lib/system_lib.h"

namespace fs = std::filesystem;
using json = nlohmann::json;

struct ShieldingScaling {
  // Atomic-units to ppm scaling for the paramagnetic trace contribution.
  double para_ppm_factor = 0.0;
  // Local diamagnetic proton term (multiplies 1s population proxy).
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

inline system_lib::DistanceUnits extract_config_units(json config) {
  if (config.find("distance_unit") != config.end()) {
    return config["distance_unit"] == "angstrom"
               ? system_lib::DistanceUnits::ANGSTROM
               : system_lib::DistanceUnits::BOHR;
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

// Merge protons whose δ (ppm) differs by at most tol in sorted-δ order (1D
// single-linkage). Works well for symmetric alkane peaks; tune tol via JSON.
static void print_grouped_delta_summary(const std::vector<double> &delta_ppm,
                                        const std::vector<size_t> &main_h_atoms,
                                        double tol_ppm) {
  if (main_h_atoms.empty() || tol_ppm <= 0.0) {
    return;
  }
  const int n = static_cast<int>(delta_ppm.size());
  if (n != static_cast<int>(main_h_atoms.size())) {
    return;
  }
  std::vector<int> order(static_cast<size_t>(n));
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int a, int b) {
    return delta_ppm[static_cast<size_t>(a)] <
           delta_ppm[static_cast<size_t>(b)];
  });
  std::vector<std::vector<int>> groups;
  for (int k = 0; k < n; ++k) {
    const int ih = order[static_cast<size_t>(k)];
    if (k == 0) {
      groups.push_back({ih});
      continue;
    }
    const int prev = order[static_cast<size_t>(k - 1)];
    if (delta_ppm[static_cast<size_t>(ih)] -
            delta_ppm[static_cast<size_t>(prev)] <=
        tol_ppm) {
      groups.back().push_back(ih);
    } else {
      groups.push_back({ih});
    }
  }

  std::cout << "\nGrouped delta summary (sorted delta, merge if neighbor gap "
               "<= "
            << std::fixed << std::setprecision(3) << tol_ppm
            << " ppm; set \"shift_grouping_tol_ppm\" <= 0 on main JSON to "
               "disable):\n";
  for (size_t g = 0; g < groups.size(); ++g) {
    const auto &idxs = groups[g];
    double sum = 0.0;
    double dmin = delta_ppm[static_cast<size_t>(idxs.front())];
    double dmax = dmin;
    for (int ih : idxs) {
      const double d = delta_ppm[static_cast<size_t>(ih)];
      sum += d;
      dmin = std::min(dmin, d);
      dmax = std::max(dmax, d);
    }
    const double avg = sum / static_cast<double>(idxs.size());
    std::cout << "  Peak " << (g + 1) << ": " << idxs.size()
              << "H, delta_avg=" << std::fixed << std::setprecision(4) << avg
              << " ppm (span " << dmin << " - " << dmax << " ppm), atoms";
    for (size_t i = 0; i < idxs.size(); ++i) {
      std::cout << (i == 0 ? ' ' : ',')
                << main_h_atoms[static_cast<size_t>(idxs[i])];
    }
    std::cout << '\n';
  }
}

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr
        << "Usage: " << argv[0] << " path/to/molecule_config.json"
        << " path/to/reference_config.json\n"
        << "Example (from repo root): " << argv[0]
        << " sample_input/ethane.json sample_input/TMS.json\n"
        << "JSON: atoms_file_path, num_alpha_electrons, num_beta_electrons;\n"
        << "optional: distance_unit, basis_dir (default \"./basis\"), "
           "delta_e (default 11.30, informational), lambda_probe "
           "(default 1e-8), field_dir 0|1|2 (default 2=z);\n"
        << "optional empirical δ from σ_para fit (main or ref JSON): "
           "shift_calibration_beta0, shift_calibration_beta1; "
           "optional shift_calibration_relative (bool): if true, "
           "δ = beta1 * (sigma_para_main - mean(sigma_para_ref));\n"
        << "optional shift_grouping_tol_ppm (main JSON, default 0.12): "
           "after per-proton δ, average symmetric peaks; use <=0 to skip.\n";
    return EXIT_FAILURE;
  }

  fs::path config_file_path(argv[1]);
  if (!fs::exists(config_file_path)) {
    std::cerr << "Path: " << config_file_path << " does not exist" << std::endl;
    return EXIT_FAILURE;
  }
  std::ifstream config_file(config_file_path);
  json config = json::parse(config_file);

  fs::path reference_config_file_path(argv[2]);
  if (!fs::exists(reference_config_file_path)) {
    std::cerr << "Path: " << reference_config_file_path << " does not exist"
              << std::endl;
    return EXIT_FAILURE;
  }
  std::ifstream reference_config_file(reference_config_file_path);
  json reference_config = json::parse(reference_config_file);

  fs::path atoms_file_path = config["atoms_file_path"];
  fs::path output_file_path = config["output_file_path"];
  int num_alpha_electrons = config["num_alpha_electrons"];
  int num_beta_electrons = config["num_beta_electrons"];
  system_lib::DistanceUnits distance_units = extract_config_units(config);

  fs::path reference_atoms_file_path = reference_config["atoms_file_path"];
  int reference_num_alpha_electrons = reference_config["num_alpha_electrons"];
  int reference_num_beta_electrons = reference_config["num_beta_electrons"];
  system_lib::DistanceUnits reference_distance_units =
      extract_config_units(reference_config);

  std::string basis_dir = "./basis";
  if (config.contains("basis_dir")) {
    basis_dir = config["basis_dir"].get<std::string>();
  }
  if (reference_config.contains("basis_dir")) {
    const std::string br = reference_config["basis_dir"].get<std::string>();
    if (br != basis_dir) {
      std::cout << "Note: reference basis_dir (\"" << br
                << "\") ignored; using main basis_dir \"" << basis_dir
                << "\".\n";
    }
  }

  double lambda_probe = 1e-8;
  if (config.contains("lambda_probe")) {
    lambda_probe = config["lambda_probe"].get<double>();
  } else if (reference_config.contains("lambda_probe")) {
    lambda_probe = reference_config["lambda_probe"].get<double>();
  }

  int field_dir = 2;
  if (config.contains("field_dir")) {
    field_dir = config["field_dir"].get<int>();
  } else if (reference_config.contains("field_dir")) {
    field_dir = reference_config["field_dir"].get<int>();
  }
  if (field_dir < 0 || field_dir > 2) {
    std::cerr << "field_dir must be 0 (x), 1 (y), or 2 (z).\n";
    return EXIT_FAILURE;
  }

  ShieldingScaling scaling = build_default_scaling();
  if (config.contains("para_ppm_factor")) {
    scaling.para_ppm_factor = config["para_ppm_factor"].get<double>();
  }
  if (config.contains("dia_ppm_factor")) {
    scaling.dia_ppm_factor = config["dia_ppm_factor"].get<double>();
  }
  if (config.contains("include_dia")) {
    scaling.include_dia = config["include_dia"].get<bool>();
  }

  std::optional<double> shift_cal_beta0;
  std::optional<double> shift_cal_beta1;
  bool shift_cal_relative = false;
  if (config.contains("shift_calibration_beta0") &&
      config.contains("shift_calibration_beta1")) {
    shift_cal_beta0 = config["shift_calibration_beta0"].get<double>();
    shift_cal_beta1 = config["shift_calibration_beta1"].get<double>();
  } else if (reference_config.contains("shift_calibration_beta0") &&
             reference_config.contains("shift_calibration_beta1")) {
    shift_cal_beta0 = reference_config["shift_calibration_beta0"].get<double>();
    shift_cal_beta1 = reference_config["shift_calibration_beta1"].get<double>();
  }
  if (config.contains("shift_calibration_relative")) {
    shift_cal_relative = config["shift_calibration_relative"].get<bool>();
  } else if (reference_config.contains("shift_calibration_relative")) {
    shift_cal_relative =
        reference_config["shift_calibration_relative"].get<bool>();
  }
  const bool use_shift_calibration =
      shift_cal_beta0.has_value() && shift_cal_beta1.has_value();

  double shift_grouping_tol_ppm = 0.12;
  if (config.contains("shift_grouping_tol_ppm")) {
    shift_grouping_tol_ppm = config["shift_grouping_tol_ppm"].get<double>();
  }

  (void)output_file_path;

  // --- Main: real CNDO/2 (Mulliken / future σ(H) path, same as 13C stack) ---
  system_lib::CNDO2System main_real = system_lib::CNDO2System::from_files(
      atoms_file_path, basis_dir, num_alpha_electrons, num_beta_electrons,
      distance_units);
  diis::solve_cndo(main_real);

  std::vector<size_t> main_h_atoms;
  collect_hydrogen_atom_indices(main_real, &main_h_atoms);

  // --- Main: complex SCF with magnetic perturbation (λ = 0 then λ = probe) ---
  // DIIS (not plain fixed-point): larger molecules (e.g. n-butane) often fail
  // to converge the complex SCF otherwise.
  system_lib::CNDO2SystemComplex main_cx0 =
      system_lib::CNDO2SystemComplex::from_files(
          atoms_file_path, basis_dir, num_alpha_electrons, num_beta_electrons,
          distance_units);
  main_cx0.set_magnetic_field(field_dir, 0.0);
  diis::solve_cndo(main_cx0);

  system_lib::CNDO2SystemComplex main_cx_lam =
      system_lib::CNDO2SystemComplex::from_files(
          atoms_file_path, basis_dir, num_alpha_electrons, num_beta_electrons,
          distance_units);
  main_cx_lam.set_magnetic_field(field_dir, lambda_probe);
  diis::solve_cndo(main_cx_lam);

  // --- Reference: real (DIIS, same as 13C calculator) ---
  system_lib::CNDO2System reference_real = system_lib::CNDO2System::from_files(
      reference_atoms_file_path, basis_dir, reference_num_alpha_electrons,
      reference_num_beta_electrons, reference_distance_units);
  diis::solve_cndo(reference_real);

  std::vector<size_t> ref_h_atoms;
  collect_hydrogen_atom_indices(reference_real, &ref_h_atoms);

  // --- Reference: complex with perturbation (mirrors main) ---
  system_lib::CNDO2SystemComplex ref_cx0 =
      system_lib::CNDO2SystemComplex::from_files(
          reference_atoms_file_path, basis_dir, reference_num_alpha_electrons,
          reference_num_beta_electrons, reference_distance_units);
  ref_cx0.set_magnetic_field(field_dir, 0.0);
  diis::solve_cndo(ref_cx0);

  system_lib::CNDO2SystemComplex ref_cx_lam =
      system_lib::CNDO2SystemComplex::from_files(
          reference_atoms_file_path, basis_dir, reference_num_alpha_electrons,
          reference_num_beta_electrons, reference_distance_units);
  ref_cx_lam.set_magnetic_field(field_dir, lambda_probe);
  diis::solve_cndo(ref_cx_lam);

  const int num_H_main = static_cast<int>(main_h_atoms.size());
  const int num_H_reference = static_cast<int>(ref_h_atoms.size());
  std::vector<double> ref_sigma_total(num_H_reference, 0.0);
  std::vector<double> ref_sigma_para(num_H_reference, 0.0);

  const double epsilon = 1.0e-5;

  for (int iH_ref = 0; iH_ref < num_H_reference; ++iH_ref) {
    size_t ref_proton = ref_h_atoms[iH_ref];
    std::cout << "\rReference: processing H " << (iH_ref + 1) << "/"
              << num_H_reference << " (atom " << ref_proton << ")..."
              << std::string(24, ' ') << std::flush;
    RealMat ref_tensor =
        ref_cx0.compute_proton_shielding_tensor(ref_proton, epsilon);
    const double raw_para_iso = arma::trace(ref_tensor) / 3.0;
    const double sigma_para = raw_para_iso * scaling.para_ppm_factor;
    const double p_1s = reference_real.get_electron_density(ref_proton);
    const double sigma_dia =
        scaling.include_dia ? p_1s * scaling.dia_ppm_factor : 0.0;
    ref_sigma_total[iH_ref] = sigma_para + sigma_dia;
    ref_sigma_para[iH_ref] = sigma_para;
  }
  std::cout << "\rReference: processing H " << num_H_reference << "/"
            << num_H_reference << " — done." << std::string(40, ' ') << '\n';

  const double sigma_ref_para_avg =
      arma::mean(arma::conv_to<RealVec>::from(ref_sigma_para));

  const double sigma_ref_avg =
      arma::mean(arma::conv_to<RealVec>::from(ref_sigma_total));

  std::vector<double> main_sigma_total(num_H_main, 0.0);
  std::vector<double> main_sigma_para(num_H_main, 0.0);
  std::vector<double> delta_ppm(num_H_main, 0.0);

  for (int iH_main = 0; iH_main < num_H_main; ++iH_main) {
    size_t main_proton = main_h_atoms[iH_main];
    std::cout << "\rMain: processing H " << (iH_main + 1) << "/" << num_H_main
              << " (atom " << main_proton << ")..." << std::string(24, ' ')
              << std::flush;
    RealMat main_tensor =
        main_cx0.compute_proton_shielding_tensor(main_proton, epsilon);
    const double raw_para_iso = arma::trace(main_tensor) / 3.0;
    const double sigma_para = raw_para_iso * scaling.para_ppm_factor;
    const double p_1s = main_real.get_electron_density(main_proton);
    const double sigma_dia =
        scaling.include_dia ? p_1s * scaling.dia_ppm_factor : 0.0;
    main_sigma_total[iH_main] = sigma_para + sigma_dia;
    main_sigma_para[iH_main] = sigma_para;
    if (use_shift_calibration) {
      if (shift_cal_relative) {
        delta_ppm[iH_main] =
            (*shift_cal_beta1) * (sigma_para - sigma_ref_para_avg);
      } else {
        delta_ppm[iH_main] = *shift_cal_beta0 + (*shift_cal_beta1) * sigma_para;
      }
    } else {
      delta_ppm[iH_main] = sigma_ref_avg - main_sigma_total[iH_main];
    }
  }
  std::cout << "\rMain: processing H " << num_H_main << "/" << num_H_main
            << " — done." << std::string(40, ' ') << "\n\n";

  std::cout << std::fixed << std::setprecision(4);
  std::cout << "Reference sigma average (ppm): " << sigma_ref_avg << '\n';
  if (use_shift_calibration) {
    std::cout << "Per-proton (calibrated δ from σ_para; see JSON "
                 "shift_calibration_*):\n";
  } else {
    std::cout << "Per-proton totals (uncalibrated δ = σ_ref,avg − σ_total):\n";
  }
  for (int iH_main = 0; iH_main < num_H_main; ++iH_main) {
    const size_t main_proton = main_h_atoms[iH_main];
    std::cout << "Main H " << iH_main << " (atom " << main_proton
              << "): sigma_total=" << main_sigma_total[iH_main]
              << ", sigma_para=" << main_sigma_para[iH_main]
              << ", delta=" << delta_ppm[iH_main] << " ppm\n";
  }

  print_grouped_delta_summary(delta_ppm, main_h_atoms, shift_grouping_tol_ppm);

  return EXIT_SUCCESS;
}
