#pragma once

#include <armadillo>
#include <list>

#include "system_lib/system_lib.h"

using namespace system_lib;

namespace fixed_point {
int solve_cndo(CNDO2System &sys, int max_iters = 1000, double tol = 1e-6);
int solve_cndo(CNDO2SystemComplex &sys, int max_iters = 1000,
               double tol = 1e-6);
} // namespace fixed_point

namespace diis {
int solve_cndo(CNDO2System &sys, int max_iters = 1000, double tol = 1e-6);
int solve_cndo(CNDO2SystemComplex &sys, int max_iters = 1000,
               double tol = 1e-6);
} // namespace diis