#pragma once

#include <armadillo>
#include <cmath>
#include <cstdlib>
#include <unordered_map>
#include <vector>

#include "system_lib/system_lib.h"

using namespace system_lib;

struct CarbonGraph {
  std::vector<size_t> carbon_local_to_atom_idx;
  std::unordered_map<size_t, size_t> atom_idx_to_carbon_local;
  std::vector<std::vector<size_t>> adj;
};

struct AlphaBetaGammaCounts {
  int a = 0;
  int b = 0;
  int c = 0;
};

struct SMotifCounts {
  int n_1_4 = 0;
  int n_2_3 = 0;
  int n_2_4 = 0;
  int n_3_2 = 0;
  int n_3_3 = 0;
  int n_4_2 = 0;
};

namespace nmr_lib {
inline const double BOHR_RADIUS = 0.52917721067; // in angstroms

// TODO - implement the following - currently just returns delta_E for methane
// (in eV) inline double calculate_delta_E(const CNDO2System& system)
// {return 11.30;};
// double calculate_delta_E(const CNDO2System &system);
double calculate_delta_E(const CarbonGraph &graph, const size_t atom_A_idx);

/* Calculate sigma_d, the diamagnetic shielding constant for atom A */
double calculate_sigma_d(const CNDO2System &system, const size_t atom_A_idx);

// TODO - implement the following
/* Calculate sigma_p, the paramagnetic shielding constant for atom A */
// double calculate_sigma_p(const CNDO2System &system, const size_t atom_A_idx,
//                          double delta_E = 11.30);
double calculate_sigma_p(const CNDO2System &system, const CarbonGraph &graph,
                         const size_t atom_A_idx);
/* Calculate sigma, the total shielding constant for atom A */
inline double calculate_sigma(const CNDO2System &system,
                              const CarbonGraph &graph,
                              const size_t atom_A_idx) {
  return calculate_sigma_d(system, atom_A_idx) +
         calculate_sigma_p(system, graph, atom_A_idx);
};

CarbonGraph build_carbon_graph(const CNDO2System &system,
                               double cc_cutoff_bohr = 3.2);

AlphaBetaGammaCounts count_alpha_beta_gamma(const CarbonGraph &graph,
                                            size_t local_idx);

SMotifCounts count_motifs(const CarbonGraph &graph, size_t local_idx);

RealMat
central_difference_density_derivative_wrt_B(const ComplexMat &density_B_plus,
                                            const ComplexMat &density_B_minus,
                                            double epsilon_B);

} // namespace nmr_lib