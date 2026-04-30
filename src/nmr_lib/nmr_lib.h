#pragma once

#include <cmath>
#include <armadillo>

#include "system_lib/system_lib.h"

using namespace system_lib;

namespace nmr_lib {
    inline const double BOHR_RADIUS = 0.52917721067; // in angstroms

    // TODO - implement the following - currently just returns delta_E for methane (in eV)
    inline double calculate_delta_E(const CNDO2System& system) {return 11.30;};

    /* Calculate sigma_d, the diamagnetic shielding constant for atom A */
    double calculate_sigma_d(const CNDO2System& system, const size_t atom_A_idx);

    // TODO - implement the following
    /* Calculate sigma_p, the paramagnetic shielding constant for atom A */
    double calculate_sigma_p(const CNDO2System& system, const size_t atom_A_idx, double delta_E=11.30);
    /* Calculate sigma, the total shielding constant for atom A */
    inline double calculate_sigma(const CNDO2System& system, const size_t atom_A_idx, double delta_E=11.30) {
        return calculate_sigma_d(system, atom_A_idx) + calculate_sigma_p(system, atom_A_idx, delta_E);
    };
}