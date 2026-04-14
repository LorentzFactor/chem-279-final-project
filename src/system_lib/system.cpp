#include "system_lib.h"

using namespace gaussian_lib;
namespace fs = std::filesystem;
using json = nlohmann::json;

namespace system_lib {
    System::System(const std::vector<Atom>& atoms)
        : atoms_(atoms)
    {
        num_orbitals_ = 0;
        atom_orbital_idxs = {};
        atom_orbital_idxs.reserve(atoms.size());
        for(const auto& atom: atoms) {
            size_t before = num_orbitals_;
            num_orbitals_ += atom.num_orbitals();
            atom_orbital_idxs.push_back({before, num_orbitals_});
        }
    }

    /* Get total number of atomic orbitals in the system */
    size_t System::num_orbitals() const {
        return num_orbitals_;
    }    
    
    /* Get total number of atoms in the system */
    size_t System::num_atoms() const {
        return atoms_.size();
    }

    /* Load a system of atoms from a set of atom position and basis files */
    std::vector<Atom> System::atoms_from_files_(std::string atoms_filepath, std::string basis_directory) {
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
        while(std::getline(atoms_file, line)) {
            sstream = std::stringstream(line);
            short elm = 0;
            double x,y,z=0;
            sstream >> elm >> x >> y >> z;

            // Convert atomic number to symbol to try to find
            // a matching basis file by name.
            std::string atomic_symbol = Atom::get_number_symbol(elm);
            std::unordered_map<std::string, GaussianTemplate> orbital_templates{};

            for (const auto& entry : fs::directory_iterator(basis_directory)) {
                std::string name = entry.path().filename();
                std::string symbol = "";
                std::string orbital = "";
                u_short n_underscores = 0;

                // Extract element symbol and orbital from file name
                for (const auto& letter : name) {
                    if (letter == '_') {
                        n_underscores += 1;
                    }
                    else if (n_underscores == 0) {
                        symbol += letter;
                    }
                    else if (n_underscores == 1) {
                        orbital = letter;
                    }
                    else {
                        break;
                    }
                }
                
                // Extract template information
                if (symbol == atomic_symbol) {
                    json orbital_info = json::parse(
                        std::ifstream(entry.path())
                    );
                    std::vector<double> alphas{};
                    std::vector<double> weights{};
                    for(const auto& primitive: orbital_info["contracted_gaussians"]) {
                        alphas.emplace_back(primitive["exponent"]);
                        weights.emplace_back(primitive["contraction_coefficient"]);
                    }
                    orbital_templates.emplace(orbital, GaussianTemplate(alphas, weights));
                }
            }
            if (elm < 3) { // Hydrogen + Helium
                atoms.push_back(Atom({x,y,z}, elm, orbital_templates.at("s")));
            }
            else { // All other elements (in second row)
                atoms.push_back(Atom(
                    {x,y,z},
                    elm,
                    orbital_templates.at("s"),
                    orbital_templates.at("p")
                ));
            }
        }

        // Final, construct and return system
        return atoms;
    }

    double System::compute_nuclear_energy() const {
        double nuclear_energy = 0;
        for(size_t iatom=0; iatom < atoms_.size(); ++iatom) {
            const Atom& atom_i = atoms_[iatom];
            double ZA = atom_i.get_atom_constant("Z_A");
            for(size_t jatom=iatom+1; jatom < atoms_.size(); ++jatom) {
                const Atom& atom_j = atoms_[jatom];
                double ZB = atom_j.get_atom_constant("Z_A");
                nuclear_energy += (ZA*ZB)/distance(atom_i, atom_j);
            }
        }
        return nuclear_energy * 27.211324570273;
    }

    arma::mat System::compute_overlap_matrix() const {

        std::vector<GaussianContracted> basis_functions{};
        basis_functions.reserve(num_orbitals_);
        for(const auto& atom: atoms_) {
            for (const auto& orbital: atom.get_atomic_orbitals()) {
                basis_functions.push_back(orbital);
            }
        }

        // Initialize overlap matrix - we initialize to ones since we will
        // be performing multiplicative operations on the data
        arma::mat S = arma::ones(basis_functions.size(), basis_functions.size());

        // Iterate through each combination of momentums to generate the matrix element-wise
        for (int i = 0; i < basis_functions.size(); i++) {
            for (int j = i; j < basis_functions.size(); j++) {
                auto orbital_i = basis_functions.at(i);
                auto orbital_j = basis_functions.at(j);
                S(i, j) = integrate_product(orbital_i, orbital_j);
                S(j, i) = S(i, j);
            }
        }
        return S;
    }

    arma::cube System::compute_overlap_matrix_gradient() const {
        std::vector<GaussianContracted> basis_functions{};
        basis_functions.reserve(num_orbitals_);
        for(const auto& atom: atoms_) {
            for (const auto& orbital: atom.get_atomic_orbitals()) {
                basis_functions.push_back(orbital);
            }
        }

        // Initialize the total derivative of S -- dS -- to 0s 
        arma::cube dS = arma::zeros(3, basis_functions.size(), basis_functions.size());

        // Iterate through pairs of orbitals - make sure to not compare orbitals in the same atom
        for (const auto& [begin, end] : atom_orbital_idxs) {
            for (int i = begin; i < end; ++i) {
                for (int j = end; j < basis_functions.size(); ++j) {
                    
                    auto orbital_i = basis_functions.at(i);
                    auto orbital_j = basis_functions.at(j);

                    // Compute partial derivative of overlap between orbitals w.r.t. Ra
                    std::array<double, 3> local_gradient = integrate_product_dRa(orbital_i, orbital_j);
                    for(size_t idim=0; idim<3; ++idim) {
                        dS(idim, i, j) -= local_gradient[idim];
                        // Use symmetry to compute derivative w.r.t. Rb
                        dS(idim, j, i) += local_gradient[idim];
                    }
                }
            }
        }
        return dS;
    }

    double System::compute_total_energy() const {
        return compute_nuclear_energy() + compute_electronic_energy();
    }
}