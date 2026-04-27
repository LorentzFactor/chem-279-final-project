#pragma once

#include "system_lib/system_lib.h"

using namespace system_lib;

namespace nmr_lib {
    /* Calculate sigma_d, the diamagnetic shielding constant for atom A */
    double calculate_sigma_d(const CNDO2System& system, const size_t atom_A_idx);

    // TODO - implement the following
    /* Calculate sigma_p, the paramagnetic shielding constant for atom A */
    inline double calculate_sigma_p(const CNDO2System& system, const size_t atom_A_idx) { return 0; };
    /* Calculate sigma, the total shielding constant for atom A */
    inline double calculate_sigma(const CNDO2System& system, const size_t atom_A_idx) { return 0; };
}