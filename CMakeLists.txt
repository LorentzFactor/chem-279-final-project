cmake_minimum_required (VERSION 3.31)

project (cndo2_nmr_calculator LANGUAGES CXX)

# Open MP Flag 
option(USE_OMP "Set to true if open mp parallelism should be enabled" OFF)

# Profiling Flag
option(ENABLE_PROFILING "Enables instrumentation of executable with gprof" OFF)

# Compiler requirements
set(CMAKE_CXX_STANDARD 20) 
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Configure profiling if enabled
# Add this to your CMakeLists.txt
if(ENABLE_PROFILING)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pg")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -pg")
endif()

# Fetch GoogleTest
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/d72f9c8aea6817cdf1ca0ac10887f328de7f3da2.zip
)
FetchContent_MakeAvailable(googletest)

# Find Libraries
find_package(nlohmann_json REQUIRED)
find_package(HDF5 REQUIRED)
find_package(HighFive REQUIRED)
find_package(Armadillo REQUIRED)
find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
if(USE_OMP)
  find_package(OpenMP REQUIRED)
endif()

add_library(common INTERFACE)

target_link_libraries(common INTERFACE
    Eigen3::Eigen
    armadillo
    nlohmann_json::nlohmann_json
    HDF5::HDF5
)

target_compile_definitions(common INTERFACE 
    ARMA_USE_HDF5
    H5_USE_110_API
)

if(USE_OMP)
  target_link_libraries(common INTERFACE OpenMP::OpenMP_CXX)
  target_compile_definitions(common INTERFACE
    ARMA_USE_OPENMP
  )
endif()

#target_compile_options(common INTERFACE ${OpenMP_CXX_FLAGS})
# Some toolchains' assemblers reject "-Wa,--gsframe=no"; enable only when needed (e.g. course Docker image).
option(CNDO2_ASM_GSFRAME "Pass -Wa,--gsframe=no to the assembler (quiet warning on some x86 setups)" OFF)
if(CNDO2_ASM_GSFRAME)
  add_compile_options(-Wa,--gsframe=no)
endif()

link_libraries(common)

add_library(integration_tools src/integration_tools/integration_tools.cpp)
add_library(gaussian_lib
    src/gaussian_lib/gaussian_primitive.cpp
    src/gaussian_lib/normed_gaussian_primitive_3d.cpp
    src/gaussian_lib/gaussian_contracted.cpp
)
add_library(system_lib
    src/system_lib/atom.cpp
    src/system_lib/system.cpp
    src/system_lib/cndo2_system.cpp
)
add_library(solver_lib
    src/solver_lib/cndo2_fixed_point_solver.cpp
    src/solver_lib/cndo2_diis_solver.cpp
)
add_library(nmr_lib
    src/nmr_lib/nmr_lib.cpp
)

target_include_directories(integration_tools PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_include_directories(gaussian_lib PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_include_directories(system_lib PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_include_directories(solver_lib PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_include_directories(nmr_lib PUBLIC ${PROJECT_SOURCE_DIR}/src)
add_executable(nmr_13c_calc src/nmr_13C_calculator.cpp)
add_executable(nmr_1h_calc src/nmr_1H_calculator.cpp)
add_executable(nmr_1h_training_export src/nmr_1h_training_export.cpp)
add_executable(molecule_energy_calc src/molecule_energy_calculator.cpp)

target_link_libraries(gaussian_lib PRIVATE integration_tools)
target_link_libraries(system_lib PRIVATE gaussian_lib)
target_link_libraries(solver_lib PRIVATE system_lib)
target_link_libraries(nmr_lib PRIVATE system_lib)

target_link_libraries(nmr_13c_calc PRIVATE system_lib)
target_link_libraries(nmr_13c_calc PRIVATE solver_lib)
target_link_libraries(nmr_13c_calc PRIVATE nmr_lib)

target_link_libraries(nmr_1h_calc PRIVATE system_lib)
target_link_libraries(nmr_1h_calc PRIVATE solver_lib)

target_link_libraries(nmr_1h_training_export PRIVATE system_lib)
target_link_libraries(nmr_1h_training_export PRIVATE solver_lib)

target_link_libraries(molecule_energy_calc PRIVATE system_lib)
target_link_libraries(molecule_energy_calc PRIVATE solver_lib)

enable_testing()

add_executable(
  nmr_test
  tests/cpp_tests/nmr_test.cpp
)

target_link_libraries(
  nmr_test
  GTest::gtest_main
  system_lib
  solver_lib
  nmr_lib
)

include(GoogleTest)
gtest_discover_tests(nmr_test)