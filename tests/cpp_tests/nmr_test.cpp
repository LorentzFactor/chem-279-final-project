#include <gtest/gtest.h>
#include <vector>
#include <system_lib/system_lib.h>
#include <solver_lib/solver_lib.h>
#include <nmr_lib/nmr_lib.h>

using namespace system_lib;
using namespace nmr_lib;


class MethaneFixture: public testing::Test {
    protected:

        void SetUp() override {
          methane_system_wrapper.emplace(CNDO2System::from_files(
              "/workspaces/chem-279-final-project/atoms/methane.xyz",
              "/workspaces/chem-279-final-project/basis",
              4, 4,
              DistanceUnits::BOHR
          ));
          fixed_point::solve_cndo(methane_system_wrapper.value());
          delta_E = 11.30;
        }

        std::optional<CNDO2System> methane_system_wrapper;
        double delta_E;
};

TEST_F(MethaneFixture, SigmaCalculation) {
  CNDO2System methane_system = methane_system_wrapper.value();
  double sigma_p = calculate_sigma_p(methane_system, 0, delta_E);
  double sigma_d = calculate_sigma_d(methane_system, 0);
  double sigma = calculate_sigma(methane_system, 0, delta_E);

  EXPECT_NEAR(sigma_p, -185.4674, 1e-2);
  EXPECT_NEAR(sigma_d, 58.1794, 1e-2);
  EXPECT_NEAR(sigma, -127.2880, 1e-2);
}