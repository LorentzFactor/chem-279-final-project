#include "nmr_lib.h"

namespace nmr_lib {
    double calculate_sigma_d(const CNDO2System& system, const size_t atom_A_idx) {
        const Atom& atom_A = system.get_atom(atom_A_idx);
        if (atom_A.get_symbol() != "C") {
            throw std::runtime_error("Only carbon atoms are supported for sigma_d calculation.");
        }

        /* Calculate q - the effective electron density surrounding atom A.
        *  We add 2 for the 2 electrons in the 1s orbital of the (carbon) itself,
        *  which are not included in the density matrix but do contribute to shielding.
        */
        double q = system.get_electron_density(atom_A_idx) + 2;
        // constants for z_star given by Takaishi 1974
        double z_star = 3.25 - 0.35 * (q-4);
        double sigma_d = 4.45*z_star*q;
        return sigma_d;
    }

    double get_system_orbital_idx_by_name(const CNDO2System& system, size_t atom_idx, const std::string& orbital_name) {
        const Atom& atom = system.get_atom(atom_idx);
        const auto& orbital_names = atom.get_orbital_names();
        const auto& orbital_idxs = system.get_atom_orbital_idxs().at(atom_idx);
        for (size_t i = 0; i < orbital_names.size(); i++) {
            if (orbital_names.at(i) == orbital_name) {
                return orbital_idxs[0] + i;
            }
        }
        throw std::runtime_error("Orbital name not found for atom " + atom.get_symbol() + ": " + orbital_name);
    }

    double calculate_q2p(const CNDO2System& system, size_t atom_A_idx) {
        const Atom& atom_A = system.get_atom(atom_A_idx);
        arma::mat P = system.get_p_alpha() + system.get_p_beta();

        double q2p = 0;
        for(auto& dim : std::array<std::string, 3>{"x", "y", "z"}) {
            double idx_A = get_system_orbital_idx_by_name(system, atom_A_idx, "p"+dim);
            q2p += P(idx_A, idx_A);
        }
        return q2p;
     }

    /* See equation 8 of the paper */
    double calculate_Q_AB(const CNDO2System& system, const size_t atom_A_idx, const size_t atom_B_idx, double q2p) {
        const Atom& atom_A = system.get_atom(atom_A_idx);
        const Atom& atom_B = system.get_atom(atom_B_idx);

        const arma::mat& P = system.get_p_alpha() + system.get_p_beta();

        double Q_AB = 0.0;
        // First term
        if (atom_A_idx == atom_B_idx) {
            Q_AB += 4.0 / 3.0 * q2p;
        }

        std::array<std::string, 3> dims = {"x", "y", "z"};

        // Compute second & third terms together
        for(size_t idim = 0; idim < 3; ++idim) {
            double idx_A = get_system_orbital_idx_by_name(system, atom_A_idx, "p"+dims[idim]);
            double idx_B = get_system_orbital_idx_by_name(system, atom_B_idx, "p"+dims[idim]);
            for (size_t jdim = idim+1; jdim < 3; ++jdim) {
                double idx_A_j = get_system_orbital_idx_by_name(system, atom_A_idx, "p"+dims[jdim]);
                double idx_B_j = get_system_orbital_idx_by_name(system, atom_B_idx, "p"+dims[jdim]);
                Q_AB -= 2.0/3.0 * P(idx_A, idx_B) * P(idx_A_j, idx_B_j);
                Q_AB += 2.0/3.0 * P(idx_A, idx_B_j) * P(idx_A_j, idx_B);
            }
        }
        return Q_AB;
     }

    double calculate_sigma_p(const CNDO2System& system, const size_t atom_A_idx) {
        double q2p = calculate_q2p(system, atom_A_idx);
        
        double sigma_p = 0;
        for(size_t jatom = 0; jatom < system.num_atoms(); ++jatom) {
            // cb - Open question - should this skip the atom itself? Takaishi's paper doesn't specify.
            // Raw formula wold include the atom itself...
            //if (jatom == atom_A_idx) continue;
            if (system.get_atom(jatom).get_atomic_number() < 3) continue;
            double Q_AB = calculate_Q_AB(system, atom_A_idx, jatom, q2p);
            sigma_p += Q_AB;
        }
        double r3_2p = 1/(24*BOHR_RADIUS*BOHR_RADIUS*BOHR_RADIUS)*std::pow(3.25 - 0.35 * (q2p-3), 3);
        double delta_E = calculate_delta_E(system);

        // e^2*h_bar^2/(2m^2*c^2 \delta E)
        double prefactor = 1.0736e2; // scaled to ppm by 1e6, assumes distances are in angstroms and delta_E is in eV
    
        sigma_p *= r3_2p * prefactor / delta_E;
        return -sigma_p;
    }
}