#include "system_lib.h"

using namespace gaussian_lib;
namespace fs = std::filesystem;
using json = nlohmann::json;

namespace system_lib {
System::System(const std::vector<Atom> &atoms) : atoms_(atoms) {
  num_orbitals_ = 0;
  atom_orbital_idxs = {};
  atom_orbital_idxs.reserve(atoms.size());
  for (const auto &atom : atoms) {
    size_t before = num_orbitals_;
    num_orbitals_ += atom.num_orbitals();
    atom_orbital_idxs.push_back({before, num_orbitals_});
  }

  S_ = arma::zeros(0, 0);
}

/* Get total number of atomic orbitals in the system */
size_t System::num_orbitals() const { return num_orbitals_; }

/* Get total number of atoms in the system */
size_t System::num_atoms() const { return atoms_.size(); }

/* Load a system of atoms from a set of atom position and basis files */
std::vector<Atom> System::atoms_from_files_(std::string atoms_filepath,
                                            std::string basis_directory,
                                            DistanceUnits distance_units) {
  std::ifstream atoms_file(atoms_filepath);
  std::string line = "";

  // Validate basis_directory
  if (!fs::exists(basis_directory) || !fs::is_directory(basis_directory)) {
    std::cerr << "Invalid directory " << basis_directory << std::endl;
    throw std::runtime_error("Could not load basis files");
  }

  // Read in num elements from first line
  std::getline(atoms_file, line);
  std::stringstream sstream(line);
  int count;
  sstream >> count;
  std::vector<Atom> atoms;
  atoms.reserve(count);

  // Skip comment line
  std::getline(atoms_file, line);

  // Iterate through atoms and their positions
  while (std::getline(atoms_file, line)) {
    sstream = std::stringstream(line);
    std::string atom_repr; // Can be either atomic symbol or number - we will
                           // try to parse both
    short elm;             // Atomic number
    double x, y, z;        // Coordinates
    sstream >> atom_repr >> x >> y >> z;

    // Check if the element is represented by atomic symbol or number
    if (atom_repr.size() == 0) {
      std::cerr << "Invalid line in atoms file: " << line << std::endl;
      throw std::runtime_error("Could not parse atoms file");
    }
    if (std::isdigit(atom_repr[0])) {
      elm = std::stoi(atom_repr);
    } else {
      elm = Atom::get_symbol_number(atom_repr);
    }

    // Convert atomic number to symbol to try to find
    // a matching basis file by name.
    std::string atomic_symbol = Atom::get_number_symbol(elm);
    std::unordered_map<std::string, GaussianTemplate> orbital_templates{};

    for (const auto &entry : fs::directory_iterator(basis_directory)) {
      std::string name = entry.path().filename();
      std::string symbol = "";
      std::string orbital = "";
      u_short n_underscores = 0;

      // Extract element symbol and orbital from file name
      for (const auto &letter : name) {
        if (letter == '_') {
          n_underscores += 1;
        } else if (n_underscores == 0) {
          symbol += letter;
        } else if (n_underscores == 1) {
          orbital = letter;
        } else {
          break;
        }
      }

      // Extract template information
      if (symbol == atomic_symbol) {
        json orbital_info = json::parse(std::ifstream(entry.path()));
        std::vector<double> alphas{};
        std::vector<double> weights{};
        for (const auto &primitive : orbital_info["contracted_gaussians"]) {
          alphas.emplace_back(primitive["exponent"]);
          weights.emplace_back(primitive["contraction_coefficient"]);
        }
        orbital_templates.emplace(orbital, GaussianTemplate(alphas, weights));
      }
    }
    std::array<double, 3> position = {x, y, z};
    if (distance_units == DistanceUnits::ANGSTROM) {
      position = {x / 0.529177, y / 0.529177, z / 0.529177};
    }
    if (elm < 3) { // Hydrogen + Helium
      atoms.push_back(Atom(position, elm, orbital_templates.at("s")));
    } else { // All other elements (in second row)
      atoms.push_back(Atom(position, elm, orbital_templates.at("s"),
                           orbital_templates.at("p")));
    }
  }

  // Final, construct and return system
  return atoms;
}

double System::compute_nuclear_energy() const {
  double nuclear_energy = 0;
  for (size_t iatom = 0; iatom < atoms_.size(); ++iatom) {
    const Atom &atom_i = atoms_[iatom];
    double ZA = atom_i.get_atom_constant("Z_A");
    for (size_t jatom = iatom + 1; jatom < atoms_.size(); ++jatom) {
      const Atom &atom_j = atoms_[jatom];
      double ZB = atom_j.get_atom_constant("Z_A");
      nuclear_energy += (ZA * ZB) / distance(atom_i, atom_j);
    }
  }
  return nuclear_energy * 27.211324570273;
}

std::array<double, 3> molecule_center_of_mass(const std::vector<Atom> &atoms) {
  // define map for masses
  static const std::unordered_map<short, double> kMassAmu{
      {1, 1.008},        {2, 4.002602}, {3, 6.94},         {4, 9.0121831},
      {5, 10.81},        {6, 12.011},   {7, 14.007},       {8, 15.999},
      {9, 18.998403163}, {10, 20.180},  {11, 22.98976928}, {12, 24.305},
      {13, 26.9815385},  {14, 28.085}};

  // weight sum and radius sum
  double wsum = 0.0;
  std::array<double, 3> rsum{0.0, 0.0, 0.0};

  for (const Atom &a : atoms) {
    const short z = a.get_atomic_number();
    // quick safety check to make sure we have the mass in our mapping
    auto it = kMassAmu.find(z);
    if (it == kMassAmu.end()) {
      throw std::runtime_error(
          "molecule_center_of_mass: unsupported atomic number " +
          std::to_string(z));
    }

    const double w = it->second;
    const auto &r = a.get_position();
    wsum += w;
    rsum[0] += w * r[0];
    rsum[1] += w * r[1];
    rsum[2] += w * r[2];
  }
  if (wsum <= 0.0) {
    throw std::runtime_error(
        "molecule_center_of_mass: empty or zero total mass");
  }
  return {rsum[0] / wsum, rsum[1] / wsum, rsum[2] / wsum};
}

arma::mat System::compute_overlap_matrix() const {

  if (S_.n_cols > 0) {
    return S_;
  }

  std::vector<GaussianContracted> basis_functions{};
  basis_functions.reserve(num_orbitals_);
  for (const auto &atom : atoms_) {
    for (const auto &orbital : atom.get_atomic_orbitals()) {
      basis_functions.push_back(orbital);
    }
  }

  // Initialize overlap matrix - we initialize to ones since we will
  // be performing multiplicative operations on the data
  arma::mat S = arma::ones(basis_functions.size(), basis_functions.size());

  // Iterate through each combination of momentums to generate the matrix
  // element-wise
  for (int i = 0; i < basis_functions.size(); i++) {
    for (int j = i; j < basis_functions.size(); j++) {
      auto orbital_i = basis_functions.at(i);
      auto orbital_j = basis_functions.at(j);
      S(i, j) = integrate_product(orbital_i, orbital_j);
      S(j, i) = S(i, j);
    }
  }
  S_ = S;
  return S_;
}

double System::compute_total_energy() {
  return compute_nuclear_energy() + compute_electronic_energy();
}

} // namespace system_lib