// 13C NMR shieldings and chemical shifts (CNDO/2 + Takaishi-style carbon graph).

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
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
    return config["distance_unit"] == "angstrom"
               ? system_lib::DistanceUnits::ANGSTROM
               : system_lib::DistanceUnits::BOHR;
  }
  return system_lib::DistanceUnits::BOHR;
}

static void collect_carbon_atom_indices(const system_lib::CNDO2System &sys,
                                        std::vector<size_t> *out_indices) {
  out_indices->clear();
  for (size_t i = 0; i < sys.num_atoms(); ++i) {
    if (sys.get_atom(i).get_atomic_number() == 6) {
      out_indices->push_back(i);
    }
  }
}

// Indices 0..n-1 into parallel per-carbon arrays; sorted by δ, merged when
// neighbor gap <= tol. If tol <= 0, each index is its own group.
static std::vector<std::vector<int>>
build_shift_groups(const std::vector<double> &delta_ppm, double tol_ppm) {
  std::vector<std::vector<int>> groups;
  const int n = static_cast<int>(delta_ppm.size());
  if (n == 0) {
    return groups;
  }
  if (tol_ppm <= 0.0) {
    for (int i = 0; i < n; ++i) {
      groups.push_back({i});
    }
    return groups;
  }

  std::vector<int> order(static_cast<size_t>(n));
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int a, int b) {
    return delta_ppm[static_cast<size_t>(a)] <
           delta_ppm[static_cast<size_t>(b)];
  });
  for (int k = 0; k < n; ++k) {
    const int ic = order[static_cast<size_t>(k)];
    if (k == 0) {
      groups.push_back({ic});
      continue;
    }
    const int prev = order[static_cast<size_t>(k - 1)];
    if (delta_ppm[static_cast<size_t>(ic)] -
            delta_ppm[static_cast<size_t>(prev)] <=
        tol_ppm) {
      groups.back().push_back(ic);
    } else {
      groups.push_back({ic});
    }
  }
  return groups;
}

static void print_grouped_delta_summary(const std::vector<double> &delta_ppm,
                                        const std::vector<size_t> &main_c_atoms,
                                        double tol_ppm) {
  if (main_c_atoms.empty() || tol_ppm <= 0.0) {
    return;
  }
  const int n = static_cast<int>(delta_ppm.size());
  if (n != static_cast<int>(main_c_atoms.size())) {
    return;
  }
  const auto groups = build_shift_groups(delta_ppm, tol_ppm);

  std::cout << "\nGrouped delta summary (sorted δ, merge if neighbor gap <= "
            << std::fixed << std::setprecision(3) << tol_ppm
            << " ppm; set \"shift_grouping_tol_ppm\" <= 0 on main JSON to "
               "disable):\n";
  for (size_t g = 0; g < groups.size(); ++g) {
    const auto &idxs = groups[g];
    double sum = 0.0;
    double dmin = delta_ppm[static_cast<size_t>(idxs.front())];
    double dmax = dmin;
    for (int ic : idxs) {
      const double d = delta_ppm[static_cast<size_t>(ic)];
      sum += d;
      dmin = std::min(dmin, d);
      dmax = std::max(dmax, d);
    }
    const double avg = sum / static_cast<double>(idxs.size());
    std::cout << "  Peak " << (g + 1) << ": " << idxs.size()
              << "C, delta_avg=" << std::fixed << std::setprecision(4) << avg
              << " ppm (span " << dmin << " - " << dmax << " ppm), atoms";
    for (size_t i = 0; i < idxs.size(); ++i) {
      std::cout << (i == 0 ? ' ' : ',')
                << main_c_atoms[static_cast<size_t>(idxs[i])];
    }
    std::cout << '\n';
  }
}

static void write_results_json(
    const fs::path &path, const json &root) {
  std::ofstream out(path);
  if (!out) {
    std::cerr << "Could not write JSON: " << path << '\n';
    return;
  }
  out << root.dump(2) << '\n';
  std::cout << "Wrote results JSON: " << path << '\n';
}

static void append_results_jsonl(const fs::path &path, const json &root) {
  std::ofstream out(path, std::ios::app);
  if (!out) {
    std::cerr << "Could not append JSONL: " << path << '\n';
    return;
  }
  out << root.dump() << '\n';
  std::cout << "Appended results JSONL: " << path << '\n';
}

static json build_results_json(const std::string &method,
    const std::string &molecule_label,
    double reference_sigma_avg_ppm, double shift_grouping_tol_ppm,
    const std::vector<size_t> &main_c_atoms,
    const std::vector<double> &main_sigma_d,
    const std::vector<double> &main_sigma_p,
    const std::vector<double> &main_sigma_total,
    const std::vector<double> &delta_ppm) {
  json root;
  root["method"] = method;
  root["molecule_label"] = molecule_label;
  root["reference_sigma_avg_ppm"] = reference_sigma_avg_ppm;
  root["shift_grouping_tol_ppm"] = shift_grouping_tol_ppm;

  json carbons = json::array();
  for (size_t i = 0; i < main_c_atoms.size(); ++i) {
    json row;
    row["c_rank"] = i;
    row["atom_index"] = main_c_atoms[i];
    row["sigma_d"] = main_sigma_d[i];
    row["sigma_p"] = main_sigma_p[i];
    row["sigma_total"] = main_sigma_total[i];
    row["delta_ppm"] = delta_ppm[i];
    carbons.push_back(std::move(row));
  }
  root["carbons"] = std::move(carbons);

  const auto groups = build_shift_groups(delta_ppm, shift_grouping_tol_ppm);
  json jgroups = json::array();
  for (size_t g = 0; g < groups.size(); ++g) {
    const auto &idxs = groups[g];
    double sum = 0.0;
    for (int ic : idxs) {
      sum += delta_ppm[static_cast<size_t>(ic)];
    }
    const double avg = sum / static_cast<double>(idxs.size());
    json jg;
    jg["peak_index"] = static_cast<int>(g + 1);
    jg["n_carbons"] = static_cast<int>(idxs.size());
    jg["delta_avg_ppm"] = avg;
    json slots = json::array();
    for (int ic : idxs) {
      slots.push_back(ic);
    }
    jg["carbon_slot_indices"] = std::move(slots);
    json atoms = json::array();
    for (int ic : idxs) {
      atoms.push_back(main_c_atoms[static_cast<size_t>(ic)]);
    }
    jg["atom_indices"] = std::move(atoms);
    jgroups.push_back(std::move(jg));
  }
  root["groups"] = std::move(jgroups);
  return root;
}

int main(int argc, char **argv) {
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0]
              << " [--cndo|--indo|--mindo] path/to/molecule_config.json"
              << " path/to/reference_config.json [results.json]"
              << " [--jsonl-out path/to/combined_results.jsonl]\n"
              << "Example (from repo root): " << argv[0]
              << " --cndo sample_input/ethane.json sample_input/methane.json\n"
              << "Optional third argument: write machine-readable summary "
                 "(for scripts/plot_nmr_13c_calc_peaks.py).\n"
              << "Optional --jsonl-out: append one JSON object per run to a "
                 "combined JSONL file.\n"
              << "JSON: atoms_file_path, num_alpha_electrons, num_beta_electrons;\n"
              << "optional: distance_unit, basis_dir (default \"./basis\"), "
                 "delta_e (optional, unused by σ pipeline), cc_cutoff_bohr "
                 "(default 3.2), label (string for plot titles);\n"
              << "optional experimental_13c_shift_ppm on main JSON: array of "
                 "literature δ (ppm) per carbon, same order as increasing atom "
                 "index (used by plot script only);\n"
              << "optional shift_grouping_tol_ppm (main JSON, default 0.2): "
                 "after per-carbon δ, average symmetric peaks; use <=0 to skip "
                 "grouped text summary (JSON still uses per-carbon singleton "
                 "groups when tol<=0).\n"
              << "δ (uncalibrated) = σ_ref,avg − σ_total,main.\n";
    return EXIT_FAILURE;
  }

  bool use_indo = false;
  std::string method_flag(argv[1]);
  std::string method_name;
  if (method_flag == "--cndo") {
    use_indo = false;
    method_name = "cndo";
  } else if (method_flag == "--indo") {
    use_indo = true;
    method_name = "indo";
  } else if (method_flag == "--mindo") {
    use_indo = true;
    method_name = "mindo";
  } else {
    std::cerr << "First argument must be --cndo, --indo, or --mindo.\n";
    return EXIT_FAILURE;
  }

  fs::path results_json_path;
  bool write_results_file = false;
  fs::path results_jsonl_path;
  bool write_jsonl = false;
  for (int i = 4; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg == "--jsonl-out") {
      if ((i + 1) >= argc) {
        std::cerr << "Missing path after --jsonl-out.\n";
        return EXIT_FAILURE;
      }
      results_jsonl_path = fs::path(argv[++i]);
      write_jsonl = true;
      continue;
    }
    if (!write_results_file) {
      results_json_path = fs::path(arg);
      write_results_file = true;
      continue;
    }
    std::cerr << "Unrecognized extra argument: " << arg << '\n';
    return EXIT_FAILURE;
  }

  fs::path config_file_path(argv[2]);
  if (!fs::exists(config_file_path)) {
    std::cerr << "Path: " << config_file_path << " does not exist" << std::endl;
    return EXIT_FAILURE;
  }
  std::ifstream config_file(config_file_path);
  json config = json::parse(config_file);

  fs::path reference_config_file_path(argv[3]);
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
  (void)config.value("delta_e", 11.30);
  system_lib::DistanceUnits distance_units = extract_config_units(config);

  fs::path reference_atoms_file_path = reference_config["atoms_file_path"];
  int reference_num_alpha_electrons = reference_config["num_alpha_electrons"];
  int reference_num_beta_electrons = reference_config["num_beta_electrons"];
  (void)reference_config.value("delta_e", 11.30);
  system_lib::DistanceUnits reference_distance_units =
      extract_config_units(reference_config);

  std::string basis_dir = "./basis";
  if (config.contains("basis_dir")) {
    basis_dir = config["basis_dir"].get<std::string>();
  } else if (reference_config.contains("basis_dir")) {
    basis_dir = reference_config["basis_dir"].get<std::string>();
  }

  double cc_cutoff_bohr = 3.2;
  if (config.contains("cc_cutoff_bohr")) {
    cc_cutoff_bohr = config["cc_cutoff_bohr"].get<double>();
  } else if (reference_config.contains("cc_cutoff_bohr")) {
    cc_cutoff_bohr = reference_config["cc_cutoff_bohr"].get<double>();
  }

  double shift_grouping_tol_ppm = 0.2;
  if (config.contains("shift_grouping_tol_ppm")) {
    shift_grouping_tol_ppm = config["shift_grouping_tol_ppm"].get<double>();
  }

  (void)output_file_path;

  std::string molecule_label =
      config.value("label", atoms_file_path.stem().string());

  system_lib::CNDO2System main_sys = system_lib::CNDO2System::from_files(
      atoms_file_path, basis_dir, num_alpha_electrons, num_beta_electrons,
      distance_units, use_indo);
  diis::solve_cndo(main_sys);
  CarbonGraph main_graph = nmr_lib::build_carbon_graph(main_sys, cc_cutoff_bohr);

  std::vector<size_t> main_c_atoms;
  collect_carbon_atom_indices(main_sys, &main_c_atoms);

  system_lib::CNDO2System reference_sys = system_lib::CNDO2System::from_files(
      reference_atoms_file_path, basis_dir, reference_num_alpha_electrons,
      reference_num_beta_electrons, reference_distance_units, use_indo);
  diis::solve_cndo(reference_sys);
  CarbonGraph ref_graph =
      nmr_lib::build_carbon_graph(reference_sys, cc_cutoff_bohr);

  std::vector<size_t> ref_c_atoms;
  collect_carbon_atom_indices(reference_sys, &ref_c_atoms);

  const int num_C_main = static_cast<int>(main_c_atoms.size());
  const int num_C_ref = static_cast<int>(ref_c_atoms.size());

  if (ref_c_atoms.empty()) {
    std::cerr << "Reference molecule must contain at least one carbon atom.\n";
    return EXIT_FAILURE;
  }
  if (main_c_atoms.empty()) {
    std::cerr << "Main molecule must contain at least one carbon atom.\n";
    return EXIT_FAILURE;
  }

  std::vector<double> ref_sigma_total(num_C_ref, 0.0);

  for (int iC_ref = 0; iC_ref < num_C_ref; ++iC_ref) {
    const size_t atom_idx = ref_c_atoms[iC_ref];
    std::cout << "\rReference: processing C " << (iC_ref + 1) << "/" << num_C_ref
              << " (atom " << atom_idx << ")..." << std::string(24, ' ')
              << std::flush;
    ref_sigma_total[iC_ref] =
        nmr_lib::calculate_sigma(reference_sys, ref_graph, atom_idx);
  }
  std::cout << "\rReference: processing C " << num_C_ref << "/" << num_C_ref
            << " — done." << std::string(40, ' ') << '\n';

  const double sigma_ref_avg =
      std::accumulate(ref_sigma_total.begin(), ref_sigma_total.end(), 0.0) /
      static_cast<double>(num_C_ref);

  std::vector<double> main_sigma_d(num_C_main, 0.0);
  std::vector<double> main_sigma_p(num_C_main, 0.0);
  std::vector<double> main_sigma_total(num_C_main, 0.0);
  std::vector<double> delta_ppm(num_C_main, 0.0);

  for (int iC_main = 0; iC_main < num_C_main; ++iC_main) {
    const size_t atom_idx = main_c_atoms[iC_main];
    std::cout << "\rMain: processing C " << (iC_main + 1) << "/" << num_C_main
              << " (atom " << atom_idx << ")..." << std::string(24, ' ')
              << std::flush;
    main_sigma_d[iC_main] = nmr_lib::calculate_sigma_d(main_sys, atom_idx);
    main_sigma_p[iC_main] =
        nmr_lib::calculate_sigma_p(main_sys, main_graph, atom_idx);
    main_sigma_total[iC_main] =
        nmr_lib::calculate_sigma(main_sys, main_graph, atom_idx);
    delta_ppm[iC_main] = sigma_ref_avg - main_sigma_total[iC_main];
  }
  std::cout << "\rMain: processing C " << num_C_main << "/" << num_C_main
            << " — done." << std::string(40, ' ') << "\n\n";

  std::cout << std::fixed << std::setprecision(4);
  std::cout << "Reference sigma average (ppm): " << sigma_ref_avg << '\n';
  std::cout << "Per-carbon totals (δ = σ_ref,avg − σ_total):\n";
  for (int iC_main = 0; iC_main < num_C_main; ++iC_main) {
    const size_t atom_idx = main_c_atoms[iC_main];
    std::cout << "Main C " << iC_main << " (atom " << atom_idx
              << "): sigma_d=" << main_sigma_d[iC_main]
              << ", sigma_p=" << main_sigma_p[iC_main]
              << ", sigma_total=" << main_sigma_total[iC_main]
              << ", delta=" << delta_ppm[iC_main] << " ppm\n";
  }

  print_grouped_delta_summary(delta_ppm, main_c_atoms, shift_grouping_tol_ppm);

  const json results =
      build_results_json(method_name, molecule_label, sigma_ref_avg,
                         shift_grouping_tol_ppm, main_c_atoms, main_sigma_d,
                         main_sigma_p, main_sigma_total, delta_ppm);

  if (write_results_file) {
    if (results_json_path.has_parent_path()) {
      fs::create_directories(results_json_path.parent_path());
    }
    write_results_json(results_json_path, results);
  }

  if (write_jsonl) {
    if (results_jsonl_path.has_parent_path()) {
      fs::create_directories(results_jsonl_path.parent_path());
    }
    append_results_jsonl(results_jsonl_path, results);
  }

  return EXIT_SUCCESS;
}
