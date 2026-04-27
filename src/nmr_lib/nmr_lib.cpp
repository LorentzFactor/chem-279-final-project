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
}