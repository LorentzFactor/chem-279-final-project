#include "system_lib.h"

using namespace gaussian_lib;
namespace fs = std::filesystem;
using json = nlohmann::json;

namespace system_lib {

    CNDO2System::CNDO2System(const std::vector<Atom>& atoms, int p, int q)
        : System(atoms)
    {
        p_ = (double) p;
        q_ = (double) q;
        set_p(arma::zeros(num_orbitals_, num_orbitals_), arma::zeros(num_orbitals_, num_orbitals_));
        beta_, gamma_, gamma_reduced_ = arma::zeros(0,0);
    }

    CNDO2System CNDO2System::from_files(std::string atoms_filepath, std::string basis_directory, int p, int q) {
        auto atoms = atoms_from_files_(atoms_filepath, basis_directory);
        return CNDO2System(atoms, p, q);
    }

    void CNDO2System::set_p(const arma::mat& new_p_alpha, const arma::mat& new_p_beta) {
        // Ensure density matrices are properly shaped
        if (
            !(new_p_alpha.n_cols == num_orbitals_) ||
            !(new_p_alpha.n_rows == num_orbitals_) ||
            !(new_p_beta.n_cols  == num_orbitals_) ||
            !(new_p_beta.n_rows  == num_orbitals_)
        ) {
            throw std::runtime_error("Invalid shape for p_alpha/beta");
        }

        // Set internal density matrices to new values
        p_alpha_ = new_p_alpha;
        p_beta_ = new_p_beta;

       // update fock matrices with new densities
       auto new_f = compute_cndo_f_matrix_internal(p_alpha_, p_beta_);
       set_f(new_f.first, new_f.second);
    }

    void CNDO2System::set_f(const arma::mat& new_f_alpha, const arma::mat& new_f_beta) {
       f_alpha_ = new_f_alpha;
       f_beta_ = new_f_beta;

       // update molecular orbitals
       arma::eig_sym(E_alpha_, mos_alpha_, new_f_alpha);
       arma::eig_sym(E_beta_, mos_beta_, new_f_beta);
    }

    arma::mat CNDO2System::compute_gamma_matrix() {
        if (gamma_.n_cols > 0) {
            return gamma_;
        }

        gamma_ = arma::zeros(num_orbitals_, num_orbitals_);

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
                gamma_(iatom, jatom) = gamma_ij;
                gamma_(jatom, iatom) = gamma_ij;
            }
        }
        return gamma_;
    }

    /* Compute the gamma matrix indexed by atoms rather than orbitals */
    arma::mat CNDO2System::compute_reduced_gamma_matrix() {
        if (gamma_reduced_.n_cols > 0) {
            return gamma_reduced_;
        }

        arma::mat gamma_reduced_ = arma::zeros(atoms_.size(), atoms_.size());

        for (int iatom = 0; iatom < atoms_.size(); iatom++) {
            const Atom& atom_i = atoms_.at(iatom);
            for (int jatom = iatom; jatom < atoms_.size(); jatom++) {
                const Atom& atom_j = atoms_.at(jatom);
                // Compute gamma using the s orbitals of each atom
                double gamma_ij = calculate_gamma(
                    atom_i.get_atomic_orbitals().at(0),
                    atom_j.get_atomic_orbitals().at(0)
                );
                gamma_reduced_(iatom, jatom) = gamma_ij;
                gamma_reduced_(jatom, iatom) = gamma_ij;
            }
        }
        return gamma_reduced_;
    }

    arma::mat CNDO2System::compute_beta_matrix() {
        if(beta_.n_cols > 0) {
            return beta_;
        }

        arma::mat beta_ = arma::zeros(num_orbitals_, num_orbitals_);
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
        beta_.each_col() += vec_betas;
        beta_.each_row() += vec_betas.as_row();
        
        return beta_;
    }

    std::pair<arma::mat,arma::mat> CNDO2System::compute_cndo_f_matrix_internal(
        const arma::mat& p_alpha,
        const arma::mat& p_beta
    ) {

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

    std::pair<arma::mat,arma::mat> CNDO2System::compute_cndo_f_matrix() {
        return {f_alpha_, f_beta_};
    }

    arma::mat CNDO2System::compute_h_core() {
        return compute_cndo_f_matrix_internal(
            arma::zeros(num_orbitals(), num_orbitals()),
            arma::zeros(num_orbitals(), num_orbitals())
        ).first;
    }

    double CNDO2System::compute_electronic_energy() {
        arma::mat h_core = compute_h_core();

        return 0.5*(arma::accu(p_alpha_%(h_core + f_alpha_)) +\
                             arma::accu(p_beta_%(h_core+f_beta_)));
    }

    arma::cube CNDO2System::compute_overlap_matrix_gradient() {
        std::vector<GaussianContracted> basis_functions{};
        basis_functions.reserve(num_orbitals_);
        for(const auto& atom: atoms_) {
            for (const auto& orbital: atom.get_atomic_orbitals()) {
                basis_functions.push_back(orbital);
            }
        }

        // Initialize the total derivative of S -- dS -- to zeros 
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

    arma::cube CNDO2System::compute_gamma_matrix_gradient() {
        arma::cube dGamma = arma::zeros(3, atoms_.size(), atoms_.size());

        for(size_t iatom = 0; iatom < atoms_.size(); ++iatom) {
            const auto& atom_i = atoms_.at(iatom);
            for(size_t jatom = iatom+1; jatom < atoms_.size(); ++jatom) {
                const auto& atom_j = atoms_.at(jatom);
                arma::vec3 term = calculate_gamma_dRa(atom_i.get_atomic_orbitals().at(0), atom_j.get_atomic_orbitals().at(0));
                for(size_t idim = 0; idim < 3; ++idim) {
                    dGamma(idim, iatom, jatom) = term(idim);
                    dGamma(idim, jatom, iatom) = -term(idim);
                }
            }
        }

        return dGamma;
    }

    arma::mat CNDO2System::E_electronic_dRA() {
        arma::mat gradient = arma::zeros(3, atoms_.size());

        arma::cube dS = compute_overlap_matrix_gradient();
        arma::cube dGamma = compute_gamma_matrix_gradient();
        arma::mat p_tot = p_alpha_ + p_beta_;

        // Compute the density across each atom
        arma::vec p_AA = arma::zeros(atoms_.size());
        arma::vec p_diag = p_tot.diag();
        for (size_t iatom = 0; iatom < atoms_.size(); iatom++) {
            std::array<size_t,2> indices = atom_orbital_idxs[iatom];
            p_AA(iatom) = arma::sum(p_diag.subvec(indices[0], indices[1]-1));
        }

       for(size_t iatom = 0; iatom < atoms_.size(); ++iatom) {
            const Atom& atom_i = atoms_.at(iatom);

            // Compute the x_mu,nu dS_mu,nu term
            for(size_t iao = atom_orbital_idxs[iatom][0]; iao < atom_orbital_idxs[iatom][1]; ++iao) {
                for(size_t jatom = 0; jatom < atoms_.size(); ++jatom) {
                    if (jatom == iatom) continue;

                    double prefactor = (atom_i.get_atom_constant("neg_beta")+atoms_.at(jatom).get_atom_constant("neg_beta"));

                    for(size_t jao = atom_orbital_idxs[jatom][0]; jao < atom_orbital_idxs[jatom][1]; ++jao) {
                        for(size_t idim = 0; idim < 3; ++idim) {
                            gradient(idim, iatom) += prefactor * (p_alpha_(iao, jao) + p_beta_(iao, jao)) * dS(idim, iao, jao);
                        }
                    }
                }
            }
        
            // Compute the y_A,B dGamma_A,B term
            for(size_t jatom = 0; jatom < atoms_.size(); ++jatom) {
                if (jatom == iatom) continue;

                const Atom& atom_j = atoms_.at(jatom);

                double prefactor = p_AA(iatom)*p_AA(jatom);
                prefactor -= p_AA(iatom)*atoms_.at(jatom).get_atom_constant("Z_A");
                prefactor -= p_AA(jatom)*atoms_.at(iatom).get_atom_constant("Z_A");

                for(size_t iao = atom_orbital_idxs[iatom][0]; iao < atom_orbital_idxs[iatom][1]; ++iao) {
                    for(size_t jao = atom_orbital_idxs[jatom][0]; jao < atom_orbital_idxs[jatom][1]; ++jao) {
                        prefactor -= p_alpha_(iao, jao) * p_alpha_(iao, jao) + p_beta_(iao, jao) * p_beta_(iao, jao);
                    }
                }
                
                for(size_t idim = 0; idim < 3; ++idim) {
                    gradient(idim, iatom) += prefactor * dGamma(idim, iatom, jatom);
                }
            }
        }

       return -gradient;
    }

    arma::mat CNDO2System::E_nuclear_dRA() {
        arma::mat gradient = arma::zeros(3, atoms_.size());

        for(size_t iatom = 0; iatom < atoms_.size(); ++iatom) {
            const Atom& atom_i = atoms_.at(iatom);
            arma::vec3 RA;
            std::copy(
                std::begin(atom_i.get_position()),
                std::end(atom_i.get_position()),
                std::begin(RA)
            );

            for(size_t jatom = iatom+1; jatom < atoms_.size(); ++jatom) {

                const Atom& atom_j = atoms_.at(jatom);
                double nuclear_prefactor = atom_i.get_atom_constant("Z_A");
                nuclear_prefactor *= atom_j.get_atom_constant("Z_A");
                arma::vec3 RB;
                std::copy(
                    std::begin(atom_j.get_position()),
                    std::end(atom_j.get_position()),
                    std::begin(RB)
                );

                arma::vec3 delta = RA-RB;
                nuclear_prefactor *= 1/std::pow(arma::norm(delta), 3);
                nuclear_prefactor *=  27.211324570273;
                gradient.col(iatom) -= nuclear_prefactor * delta;
                gradient.col(jatom) += nuclear_prefactor * delta;
            }
        }
        return gradient;
    }
    
    arma::mat CNDO2System::get_occupied_MOs_alpha() const {
        if (p_>0)
            return mos_alpha_.cols(arma::span(0, p_-1));
        else
            return arma::zeros(num_orbitals_, num_orbitals_);
    }

    arma::mat CNDO2System::get_occupied_MOs_beta() const {
        if (q_>0)
            return mos_beta_.cols(arma::span(0, q_-1));
        else
            return arma::zeros(num_orbitals_, num_orbitals_);
    }

    double CNDO2System::get_electron_density(const std::array<double,3>& position) const {
        double density = 0;
        arma::mat occupied_mos = arma::join_rows(get_occupied_MOs_alpha(), get_occupied_MOs_beta());
        for (int i = 0; i < 1; ++i) {
            arma::vec mo = occupied_mos.col(i);
            double mo_value = 0;
            int icur_atom = 0;
            for (int j = 0; j < occupied_mos.n_rows; ++j) {
                if (atom_orbital_idxs[icur_atom][1] == j) {
                    ++icur_atom;
                }
                GaussianContracted ao = atoms_[icur_atom].get_atomic_orbitals()[j-atom_orbital_idxs[icur_atom][0]];
                mo_value += mo.at(j) * ao(position);
            }
            density += mo_value * mo_value;
        }
        return density;
    }

    arma::cube CNDO2System::get_electron_density_3d_grid(
        const std::array<double,2>& x_range,
        const std::array<double,2>& y_range,
        const std::array<double,2>& z_range,
        int nsamples_per_side
    ) const {
        // Validate nsamples input
        if(nsamples_per_side < 2) {
            throw std::runtime_error("nsamples must be > 1");
        }

        // Generate sample positions along each dimension
        arma::vec sample_x = arma::linspace(x_range[0], x_range[1], nsamples_per_side);
        arma::vec sample_y = arma::linspace(y_range[0], y_range[1], nsamples_per_side);
        arma::vec sample_z = arma::linspace(z_range[0], z_range[1], nsamples_per_side);

        arma::cube values = arma::cube(nsamples_per_side, nsamples_per_side, nsamples_per_side);
        arma::mat occupied_mos = arma::join_rows(get_occupied_MOs_alpha(), get_occupied_MOs_beta());
        for (int i = 0; i < occupied_mos.n_cols; ++i) {
            arma::vec mo = occupied_mos.col(i);
            arma::cube mo_value = arma::cube(nsamples_per_side, nsamples_per_side, nsamples_per_side);
            int icur_atom = 0;
            for (int j = 0; j < occupied_mos.n_rows; ++j) {
                if (atom_orbital_idxs[icur_atom][1] == j) {
                    ++icur_atom;
                }
                GaussianContracted ao = atoms_[icur_atom].get_atomic_orbitals()[j-atom_orbital_idxs[icur_atom][0]];
                mo_value += mo.at(j) * ao(sample_x, sample_y, sample_z);
            }
            values += arma::square(mo_value);
        }
        return values;
    }

    
}