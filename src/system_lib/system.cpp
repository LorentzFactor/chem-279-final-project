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

    /* Load a system of atoms from a set of atom position and basis files */
    System System::from_files(std::string atoms_filepath, std::string basis_directory) {
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
            std::cout << line << std::endl;
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
        return System(atoms);
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

    arma::mat System::compute_gamma_matrix() const {
        arma::mat gamma = arma::zeros(num_orbitals_, num_orbitals_);

        std::vector<Atom> atoms{};
        atoms.reserve(num_orbitals_);
        for(const auto& atom: atoms_) {
            for (const auto& orbital: atom.get_atomic_orbitals()) {
                atoms.emplace_back(atom);
            }
        }

        for (int iatom = 0; iatom < atoms.size(); iatom++) {
            const Atom& atom_i = atoms.at(iatom);
            for (int jatom = iatom; jatom < atoms.size(); jatom++) {
                const Atom& atom_j = atoms.at(jatom);
                // Compute gamma using the s orbitals of each atom
                double gamma_ij = calculate_gamma(
                    atom_i.get_atomic_orbitals().at(0),
                    atom_j.get_atomic_orbitals().at(0)
                );
                gamma(iatom, jatom) = gamma_ij;
                gamma(jatom, iatom) = gamma_ij;
            }
        }
        return gamma;
    }

    /* Compute the gamma matrix indexed by atoms rather than orbitals */
    arma::mat System::compute_reduced_gamma_matrix() const {
        arma::mat gamma = arma::zeros(atoms_.size(), atoms_.size());

        for (int iatom = 0; iatom < atoms_.size(); iatom++) {
            const Atom& atom_i = atoms_.at(iatom);
            for (int jatom = iatom; jatom < atoms_.size(); jatom++) {
                const Atom& atom_j = atoms_.at(jatom);
                // Compute gamma using the s orbitals of each atom
                double gamma_ij = calculate_gamma(
                    atom_i.get_atomic_orbitals().at(0),
                    atom_j.get_atomic_orbitals().at(0)
                );
                gamma(iatom, jatom) = gamma_ij;
                gamma(jatom, iatom) = gamma_ij;
            }
        }
        return gamma;
    }

    arma::mat System::compute_beta_matrix() const {
        arma::mat beta = arma::zeros(num_orbitals_, num_orbitals_);
        std::vector<double> orbital_betas{};
        orbital_betas.reserve(num_orbitals_);
        for (const auto& atom : atoms_) {
            double beta = atom.get_atom_constant("neg_beta");
            for(size_t i = 0; i < atom.num_orbitals(); ++i) {
                orbital_betas.push_back(beta);
            }
        }
        arma::vec vec_betas = arma::vec(orbital_betas.data(), num_orbitals_);
        vec_betas /= 2;
        beta.each_col() += vec_betas;
        beta.each_row() += vec_betas.as_row();
        return beta;
    }

    std::pair<arma::mat,arma::mat> System::compute_cndo_f_matrix(const arma::mat& p_alpha, const arma::mat& p_beta) const {

        // Compute all off-diagonal elements
        arma::mat gamma = compute_gamma_matrix();
        arma::mat reduced_gamma = compute_reduced_gamma_matrix();
        arma::mat S = compute_overlap_matrix();
        arma::mat beta = compute_beta_matrix();
        arma::mat f_alpha = S%beta - p_alpha%gamma;
        arma::mat f_beta = S%beta - p_beta%gamma;

        arma::mat p_tot = p_alpha + p_beta;

        // Compute the density across each atom
        arma::vec p_AA = arma::zeros(atoms_.size());
        arma::vec p_diag = p_tot.diag();
        for (size_t iatom = 0; iatom < atoms_.size(); iatom++) {
            std::array<size_t,2> indices = atom_orbital_idxs[iatom];
            p_AA(iatom) = arma::sum(p_diag.subvec(indices[0], indices[1]-1));
        }

        // Compute diagonal terms
        int iorbital = 0;
        int iatom = 0;
        for (const auto& atom: atoms_) {
            for (const auto& orbital: atom.get_atomic_orbitals()) {
                double diag_term = 0;

                // first term: -1/2*(I_mu + A_mu)
                if (orbital.shell == 0) {
                    diag_term -= atom.get_atom_constant("sI+A/2");
                }
                else if(orbital.shell == 1) {
                    diag_term -= atom.get_atom_constant("pI+A/2");
                }
                else {
                    throw std::runtime_error("unsupported shell");
                }

                // third term: \sum_{C \ne A} (p^tot_CC - Z_C)*gamma_AC
                for (size_t jatom = 0; jatom < atoms_.size(); jatom++) {
                    if (jatom != iatom) {
                        auto& atom_c = atoms_.at(jatom);
                        double gamma_AC = reduced_gamma(iatom, jatom);
                        diag_term += (p_AA.at(jatom)-atom_c.get_atom_constant("Z_A"))*gamma_AC;
                    }
                }

                // second term (differs for alpha/beta): [(p^tot_AA - Z_A) - (p^alpha_{mu mu} -1/2)]*gamma_AA
                double diag_term_a = diag_term + ((p_AA.at(iatom)-atom.get_atom_constant("Z_A")) - (p_alpha(iorbital, iorbital)-0.5))*gamma(iorbital, iorbital);
                double diag_term_b = diag_term + ((p_AA.at(iatom)-atom.get_atom_constant("Z_A")) - (p_beta(iorbital, iorbital)-0.5))*gamma(iorbital, iorbital);

                // set f_alpha to term
                f_alpha(iorbital, iorbital) = diag_term_a;
                f_beta(iorbital, iorbital) = diag_term_b;
                iorbital++;
            }
            iatom++;
        }

        return {f_alpha, f_beta};
    }
}