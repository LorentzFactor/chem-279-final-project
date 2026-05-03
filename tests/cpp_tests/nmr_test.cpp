#include <gtest/gtest.h>
#include <filesystem>
#include <optional>
#include <system_lib/system_lib.h>
#include <solver_lib/solver_lib.h>
#include <nmr_lib/nmr_lib.h>

using namespace system_lib;
using namespace nmr_lib;

namespace fs = std::filesystem;

/** Repo root: tests/cpp_tests/nmr_test.cpp -> .. -> .. -> .. */
static fs::path repo_root() {
  return fs::path(__FILE__).parent_path().parent_path().parent_path();
}

class MethaneFixture: public testing::Test {
    protected:

        void SetUp() override {
          const fs::path root = repo_root();
          methane_system_wrapper.emplace(CNDO2System::from_files(
              (root / "atoms" / "methane.xyz").string(),
              (root / "basis").string(),
              4, 4,
              DistanceUnits::BOHR
          ));
          fixed_point::solve_cndo(methane_system_wrapper.value());
        }

        std::optional<CNDO2System> methane_system_wrapper;
};

TEST_F(MethaneFixture, SigmaCalculation) {
  CNDO2System methane_system = methane_system_wrapper.value();
  CarbonGraph methane_graph = build_carbon_graph(methane_system);
  double sigma_p = calculate_sigma_p(methane_system, methane_graph, 0);
  double sigma_d = calculate_sigma_d(methane_system, 0);
  double sigma = calculate_sigma(methane_system, methane_graph, 0);

  EXPECT_NEAR(sigma_p, -185.4674, 1e-2);
  // Diamagnetic term depends on converged density; values below match fixed-point
  // SCF + current CNDO/2 stack (see methane regression in CI/local builds).
  EXPECT_NEAR(sigma_d, 68.1632, 1e-2);
  EXPECT_NEAR(sigma, -117.3042, 1e-2);
}