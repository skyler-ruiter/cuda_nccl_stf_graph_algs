cmake_minimum_required(VERSION 3.18)
project(cuda_nccl_stf_graph_algs CUDA CXX)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# Find required packages
find_package(MPI REQUIRED)
find_package(CUDA REQUIRED)

# Set path to CCCL/STF
set(CCCL_HOME $ENV{HOME}/lib/cccl)

include(cmake/CPM.cmake)
set(CCCL_REPOSITORY "https://github.com/NVIDIA/cccl" CACHE STRING "Git repository to fetch CCCL from")
set(CCCL_TAG "main" CACHE STRING "Git tag/branch to fetch from CCCL repository")

CPMAddPackage(
  NAME CCCL
  GIT_REPOSITORY "${CCCL_REPOSITORY}"
  GIT_TAG ${CCCL_TAG}
  # The following is required to make the `CCCL::cudax` target available:
  OPTIONS "CCCL_ENABLE_UNSTABLE ON"
)

# Set path to NCCL
set(NCCL_HOME /N/soft/sles15/nccl/2.25.1.1)

# Include directories
include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${MPI_CXX_INCLUDE_PATH}
    ${NCCL_HOME}/include
)

# CUDA flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --extended-lambda")

# Add the executable (single target since both files are now using CuSTF)
add_executable(mpi_nccl_testing tests/mpi_nccl_testing.cpp)
target_link_libraries(mpi_nccl_testing 
    ${MPI_CXX_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${NCCL_HOME}/lib/libnccl.so
    cuda
    cudart
    CCCL::CCCL
    CCCL::cudax
)

# Add a custom target for running with MPI
add_custom_target(run
    COMMAND mpirun -np 2 ${CMAKE_BINARY_DIR}/mpi_nccl_testing
    DEPENDS mpi_nccl_testing
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running mpi_nccl_testing with 2 MPI processes"
)