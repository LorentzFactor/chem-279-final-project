#include "nmr_lib.h"
#include <array>
#include <cstddef>
#include <queue>
#include <unordered_map>
#include <string>
#include <vector>

namespace nmr_lib {

double calculate_sigma_d(const CNDO2System &system, const size_t atom_A_idx) {
  const Atom &atom_A = system.get_atom(atom_A_idx);
  if (atom_A.get_symbol() != "C") {
    throw std::runtime_error(
        "Only carbon atoms are supported for sigma_d calculation.");
  }

  /* Calculate q - the effective electron density surrounding atom A.
   *  We add 2 for the 2 electrons in the 1s orbital of the (carbon) itself,
   *  which are not included in the density matrix but do contribute to
   * shielding.
   */
  double q = system.get_electron_density(atom_A_idx) + 2;
  // constants for z_star given by Takaishi 1974
  double z_star = 3.25 - 0.35 * (q - 4);
  double sigma_d = 4.45 * z_star * q;
  return sigma_d;
}

double get_system_orbital_idx_by_name(const CNDO2System &system,
                                      size_t atom_idx,
                                      const std::string &orbital_name) {
  const Atom &atom = system.get_atom(atom_idx);
  const auto &orbital_names = atom.get_orbital_names();
  const auto &orbital_idxs = system.get_atom_orbital_idxs().at(atom_idx);
  for (size_t i = 0; i < orbital_names.size(); i++) {
    if (orbital_names.at(i) == orbital_name) {
      return orbital_idxs[0] + i;
    }
  }
  throw std::runtime_error("Orbital name not found for atom " +
                           atom.get_symbol() + ": " + orbital_name);
}

double calculate_q2p(const CNDO2System &system, size_t atom_A_idx) {
  const Atom &atom_A = system.get_atom(atom_A_idx);
  arma::mat P = system.get_p_alpha() + system.get_p_beta();

  double q2p = 0;
  for (auto &dim : std::array<std::string, 3>{"x", "y", "z"}) {
    double idx_A =
        get_system_orbital_idx_by_name(system, atom_A_idx, "p" + dim);
    q2p += P(idx_A, idx_A);
  }
  return q2p;
}

/* See equation 8 of the paper */
double calculate_Q_AB(const CNDO2System &system, const size_t atom_A_idx,
                      const size_t atom_B_idx, double q2p) {
  const Atom &atom_A = system.get_atom(atom_A_idx);
  const Atom &atom_B = system.get_atom(atom_B_idx);

  const arma::mat &P = system.get_p_alpha() + system.get_p_beta();

  double Q_AB = 0.0;
  // First term
  if (atom_A_idx == atom_B_idx) {
    Q_AB += 4.0 / 3.0 * q2p;
  }

  std::array<std::string, 3> dims = {"x", "y", "z"};

  // Compute second & third terms together
  for (size_t idim = 0; idim < 3; ++idim) {
    double idx_A =
        get_system_orbital_idx_by_name(system, atom_A_idx, "p" + dims[idim]);
    double idx_B =
        get_system_orbital_idx_by_name(system, atom_B_idx, "p" + dims[idim]);
    for (size_t jdim = idim + 1; jdim < 3; ++jdim) {
      double idx_A_j =
          get_system_orbital_idx_by_name(system, atom_A_idx, "p" + dims[jdim]);
      double idx_B_j =
          get_system_orbital_idx_by_name(system, atom_B_idx, "p" + dims[jdim]);
      Q_AB -= 2.0 / 3.0 * P(idx_A, idx_B) * P(idx_A_j, idx_B_j);
      Q_AB += 2.0 / 3.0 * P(idx_A, idx_B_j) * P(idx_A_j, idx_B);
    }
  }
  return Q_AB;
}

double calculate_sigma_p(const CNDO2System &system, const size_t atom_A_idx,
                         double delta_E) {
  double q2p = calculate_q2p(system, atom_A_idx);

  double sigma_p = 0;
  for (size_t jatom = 0; jatom < system.num_atoms(); ++jatom) {
    if (system.get_atom(jatom).get_atomic_number() < 3)
      continue;
    double Q_AB = calculate_Q_AB(system, atom_A_idx, jatom, q2p);
    sigma_p += Q_AB;
  }
  double r3_2p = 1 / (24 * BOHR_RADIUS * BOHR_RADIUS * BOHR_RADIUS) *
                 std::pow(3.25 - 0.35 * (q2p - 3), 3);

  // e^2*h_bar^2/(2m^2*c^2 \delta E)
  double prefactor = 1.0736e2; // scaled to ppm by 1e6, assumes distances are in
                               // angstroms and delta_E is in eV

  sigma_p *= r3_2p * prefactor / delta_E;

  return -sigma_p;
}

double calculate_sigma_p(const CNDO2System &system, const CarbonGraph &graph,
                         const size_t atom_A_idx) {
  return calculate_sigma_p(system, atom_A_idx,
                           calculate_delta_E(graph, atom_A_idx));
}

double calculate_delta_E(const CarbonGraph &graph, const size_t atom_A_idx) {
  const size_t local_atom_A_idx = graph.atom_idx_to_carbon_local.at(atom_A_idx);
  AlphaBetaGammaCounts counts = count_alpha_beta_gamma(graph, local_atom_A_idx);
  SMotifCounts motifs = count_motifs(graph, local_atom_A_idx);
  double a = pow(1.015, counts.a);
  double b = pow(0.958, counts.b);
  double c = pow(1.010, counts.c);

  double term_A = 11.30 * a * b * c;

  double d = pow(1.011, motifs.n_1_4);
  double e = pow(1.009, motifs.n_2_3);
  double f = pow(1.029, motifs.n_2_4);
  double g = pow(1.009, motifs.n_3_2);
  double h = pow(1.030, motifs.n_3_3);
  double i = pow(1.031, motifs.n_4_2);

  double term_S = d * e * f * g * h * i;

  return term_A * term_S;
}

CarbonGraph build_carbon_graph(const CNDO2System &system,
                               double cc_cutoff_bohr) {
  CarbonGraph graph;

  // Build the carbon idx to global idx mapping
  for (size_t global_idx = 0; global_idx < system.num_atoms(); ++global_idx) {
    if (system.get_atom(global_idx).get_atomic_number() == 6) {
      size_t local_idx = graph.carbon_local_to_atom_idx.size();
      graph.carbon_local_to_atom_idx.push_back(global_idx);
      graph.atom_idx_to_carbon_local[global_idx] = local_idx;
    }
  }

  // Build the carbon-carbon adjacency list
  size_t num_carbons = graph.carbon_local_to_atom_idx.size();
  graph.adj.resize(num_carbons, std::vector<size_t>(0, 0));

  for (size_t idxA = 0; idxA < num_carbons; ++idxA) {
    for (size_t idxB = idxA + 1; idxB < num_carbons; ++idxB) {
      size_t carbA = graph.carbon_local_to_atom_idx.at(idxA);
      size_t carbB = graph.carbon_local_to_atom_idx.at(idxB);
      const Atom &atomA = system.get_atom(carbA);
      const Atom &atomB = system.get_atom(carbB);
      double dist = distance(atomA, atomB);
      if (dist <= cc_cutoff_bohr) {
        graph.adj[idxA].push_back(idxB);
        graph.adj[idxB].push_back(idxA);
      }
    }
  }

  return graph;
}

AlphaBetaGammaCounts count_alpha_beta_gamma(const CarbonGraph &graph,
                                            size_t local_idx) {
  AlphaBetaGammaCounts counts;

  const size_t n = graph.adj.size();
  std::vector<int> dist(n, -1);
  std::queue<size_t> q;

  dist[local_idx] = 0;
  q.push(local_idx);

  // Perform BFS to get counts on neighboring carbons
  while (!q.empty()) {
    size_t u = q.front();
    q.pop();

    // Only search up to depth 3 (a=1, b=2 and c=3)
    if (dist[u] >= 3)
      continue;

    for (size_t v : graph.adj[u]) {
      if (dist[v] != -1)
        continue; // visited already
      dist[v] = dist[u] + 1;

      if (dist[v] == 1)
        counts.a++;
      else if (dist[v] == 2)
        counts.b++;
      else if (dist[v] == 3)
        counts.c++;

      q.push(v);
    }
  }

  return counts;
}

SMotifCounts count_motifs(const CarbonGraph &graph, size_t local_idx) {
  SMotifCounts m{};

  // BFS distances from target carbon
  const size_t n = graph.adj.size();
  std::vector<int> dist(n, -1);
  std::queue<size_t> q;
  dist[local_idx] = 0;
  q.push(local_idx);

  while (!q.empty()) {
    size_t u = q.front();
    q.pop();
    if (dist[u] >= 3)
      continue;

    for (size_t v : graph.adj[u]) {
      if (dist[v] != -1)
        continue;
      dist[v] = dist[u] + 1;
      q.push(v);
    }
  }

  // Lambda for getting the carbon class based on n neighbors in adj list
  auto carbon_class = [&](size_t local) {
    int deg = static_cast<int>(graph.adj.at(local).size());
    if (deg <= 1)
      return 1;
    if (deg == 2)
      return 2;
    if (deg == 3)
      return 3;
    return 4;
  };

  for (size_t u = 0; u < n; ++u) {
    if (dist[u] == -1 || dist[u] > 3)
      continue;
    int cu = carbon_class(u);

    for (size_t v : graph.adj.at(u)) {
      if (u >= v)
        continue; // only count edges once
      if (dist[v] == -1 || dist[v] > 3)
        continue;
      int cv = carbon_class(v);

      // Count ordered motifs
      // [cu(cv)] and [cv(cu)]
      auto add_ordered = [&](int a, int b) {
        if (a == 1 && b == 4)
          m.n_1_4++;
        if (a == 2 && b == 3)
          m.n_2_3++;
        if (a == 2 && b == 4)
          m.n_2_4++;
        if (a == 3 && b == 2)
          m.n_3_2++;
        if (a == 3 && b == 3)
          m.n_3_3++;
        if (a == 4 && b == 2)
          m.n_4_2++;
      };

      add_ordered(cu, cv);
      add_ordered(cv, cu);
    }
  }
  return m;
}

} // namespace nmr_lib