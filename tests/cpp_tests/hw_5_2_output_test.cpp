#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include <highfive/H5File.hpp>

namespace fs = std::filesystem;
using namespace HighFive;

namespace {

constexpr double kComparisonTolerance = 3e-3;

static fs::path repo_root() {
  return fs::path(__FILE__).parent_path().parent_path().parent_path();
}

static std::vector<unsigned char> dataset_bytes(const DataSet &dataset) {
  const DataType data_type = dataset.getDataType();
  const size_t element_count = dataset.getElementCount();
  const size_t num_bytes = element_count * data_type.getSize();

  std::vector<unsigned char> bytes(num_bytes);
  if (num_bytes > 0) {
    dataset.read_raw(bytes.data());
  }
  return bytes;
}

static void compare_hdf5_files(const fs::path &expected_path,
                               const fs::path &actual_path) {
  ASSERT_TRUE(fs::exists(expected_path)) << "Missing expected file: " << expected_path;
  ASSERT_TRUE(fs::exists(actual_path)) << "Missing actual file: " << actual_path;

  File expected_file(expected_path.string(), File::ReadOnly);
  File actual_file(actual_path.string(), File::ReadOnly);

  auto expected_names = expected_file.listObjectNames();
  auto actual_names = actual_file.listObjectNames();
  std::sort(expected_names.begin(), expected_names.end());
  std::sort(actual_names.begin(), actual_names.end());

  std::vector<std::string> shared_names;
  std::set_intersection(expected_names.begin(), expected_names.end(),
                        actual_names.begin(), actual_names.end(),
                        std::back_inserter(shared_names));

  ASSERT_FALSE(shared_names.empty())
      << "No shared datasets/objects between files: "
      << expected_path.filename().string();

  for (const auto &name : shared_names) {
    ASSERT_TRUE(expected_file.exist(name));
    ASSERT_TRUE(actual_file.exist(name));

    const DataSet expected_dataset = expected_file.getDataSet(name);
    const DataSet actual_dataset = actual_file.getDataSet(name);

    EXPECT_EQ(expected_dataset.getSpace().getDimensions(),
              actual_dataset.getSpace().getDimensions())
        << "Shape mismatch for dataset '" << name << "'";

    bool compared_as_numeric = false;
    try {
      const size_t element_count = expected_dataset.getElementCount();
      if (element_count == actual_dataset.getElementCount()) {
        std::vector<double> expected_values(element_count);
        std::vector<double> actual_values(element_count);
        expected_dataset.read_raw(expected_values.data());
        actual_dataset.read_raw(actual_values.data());

        double max_abs_diff = 0.0;
        size_t max_abs_diff_index = 0;
        for (size_t i = 0; i < element_count; ++i) {
          const double lhs = expected_values[i];
          const double rhs = actual_values[i];
          if (std::isnan(lhs) && std::isnan(rhs)) {
            continue;
          }
          const double abs_diff = std::fabs(lhs - rhs);
          if (abs_diff > max_abs_diff) {
            max_abs_diff = abs_diff;
            max_abs_diff_index = i;
          }
        }

        EXPECT_LE(max_abs_diff, kComparisonTolerance)
            << "Max abs diff " << max_abs_diff << " exceeds tolerance "
            << kComparisonTolerance << " for dataset '" << name
            << "' at flat index " << max_abs_diff_index;
        compared_as_numeric = true;
      }
    } catch (const std::exception &) {
      compared_as_numeric = false;
    }

    if (!compared_as_numeric) {
      const auto expected_data = dataset_bytes(expected_dataset);
      const auto actual_data = dataset_bytes(actual_dataset);

      EXPECT_EQ(expected_data, actual_data)
          << "Data mismatch for non-numeric dataset '" << name << "'";
    }
  }
}

class Hw52OutputFixture : public testing::Test {
 protected:
  void SetUp() override {
    const fs::path root = repo_root();
    const fs::path student_output = root / "student_output";

    if (fs::exists(student_output)) {
      fs::remove_all(student_output);
    }
    fs::create_directories(student_output);
  }

  int run_hw_5_2(const std::string &input_json) {
    const fs::path root = repo_root();
    const fs::path input_path = root / "sample_input" / input_json;

    if (!fs::exists(input_path)) {
      ADD_FAILURE() << "Missing input: " << input_path;
      return -1;
    }

    const std::string command =
        "cd \"" + root.string() + "\" && \"" + std::string(HW_5_2_EXECUTABLE_PATH) +
        "\" \"" + input_path.string() + "\"";

    return std::system(command.c_str());
  }

  void assert_matches_sample_output(const std::string &stem) {
    const fs::path root = repo_root();
    const fs::path expected = root / "sample_output" / (stem + ".hdf5");
    const fs::path actual = root / "student_output" / (stem + ".hdf5");

    compare_hdf5_files(expected, actual);
  }
};

TEST_F(Hw52OutputFixture, H2MatchesSample) {
  ASSERT_EQ(run_hw_5_2("H2.json"), 0) << "hw_5_2 failed for input H2.json";
  assert_matches_sample_output("H2");
}

TEST_F(Hw52OutputFixture, HFMatchesSample) {
  ASSERT_EQ(run_hw_5_2("HF.json"), 0) << "hw_5_2 failed for input HF.json";
  assert_matches_sample_output("HF");
}

TEST_F(Hw52OutputFixture, HOMatchesSample) {
  ASSERT_EQ(run_hw_5_2("HO.json"), 0) << "hw_5_2 failed for input HO.json";
  assert_matches_sample_output("HO");
}

TEST_F(Hw52OutputFixture, H2OMatchesSample) {
  ASSERT_EQ(run_hw_5_2("H2O.json"), 0) << "hw_5_2 failed for input H2O.json";
  assert_matches_sample_output("H2O");
}

}  // namespace
