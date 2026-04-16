#pragma once

#include <armadillo>

#include "system_lib/system_lib.h"

using namespace system_lib;

namespace fixed_point {
    void solve_cndo(CNDO2System& sys, int max_iters=1000, double tol=1e-6);
}