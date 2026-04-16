cmake_minimum_required (VERSION 3.31)

project (homework_5 LANGUAGES CXX)

# Compiler requirements
set(CMAKE_CXX_STANDARD 20) 
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find JSON Libraries
find_package(nlohmann_json REQUIRED)
find_package(HDF5 REQUIRED)
find_package(HighFive REQUIRED)
find_package(Armadillo REQUIRED)
find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(OpenMP REQUIRED)

add_library(common INTERFACE)

target_link_libraries(common INTERFACE
    Eigen3::Eigen
    armadillo
    nlohmann_json::nlohmann_json
    HDF5::HDF5
    OpenMP::OpenMP_CXX
)

target_compile_definitions(common INTERFACE 
    ARMA_USE_HDF5
    ARMA_USE_OPENMP
    H5_USE_110_API
)

target_compile_options(common INTERFACE ${OpenMP_CXX_FLAGS})
add_compile_options(-Wa,--gsframe=no) # This is to suppress a useless warning on x86 machines


link_libraries(common)

# If Needed Make Modifications below this line 

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
)

target_include_directories(integration_tools PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_include_directories(gaussian_lib PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_include_directories(system_lib PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_include_directories(solver_lib PUBLIC ${PROJECT_SOURCE_DIR}/src)

add_executable(hw_5_2 src/hw_5_2.cpp)

target_link_libraries(gaussian_lib PRIVATE integration_tools)
target_link_libraries(system_lib PRIVATE gaussian_lib)
target_link_libraries(solver_lib PRIVATE system_lib)
target_link_libraries(hw_5_2 PRIVATE system_lib)
target_link_libraries(hw_5_2 PRIVATE solver_lib)
