cmake_minimum_required(VERSION 3.18)
project(dist_graph_algos CUDA CXX)

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

# Find the exact path to libcuda.so using find_library (added)
find_library(CUDA_DRIVER_LIBRARY
    NAMES cuda
    PATHS ${CUDA_TOOLKIT_ROOT_DIR}/lib64
          ${CUDA_TOOLKIT_ROOT_DIR}/lib
          /usr/lib64
          /usr/lib
          /usr/local/cuda/lib64
          /usr/local/cuda/lib
          ${CUDA_HOME}/lib64
          ${CUDA_HOME}/lib
    PATH_SUFFIXES stubs)

if(NOT CUDA_DRIVER_LIBRARY)
    message(FATAL_ERROR "Could not find libcuda.so. Please specify the correct location with -DCUDA_HOME=/path/to/cuda")
endif()

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

#### MPI+CUDASTF+NCCL DIST BFS ####
add_executable(dist_bfs src/dist_bfs/distributed_bfs.cpp)

set_source_files_properties(src/dist_bfs/distributed_bfs.cpp PROPERTIES LANGUAGE CUDA)

target_link_libraries(dist_bfs 
    ${MPI_CXX_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${NCCL_HOME}/lib/libnccl.so
    ${CUDA_DRIVER_LIBRARY}
    cudart
    CCCL::CCCL
    CCCL::cudax
)

#### MPI+NCCL DIST SSSP ####
add_executable(dist_pagerank src/pagerank/pagerank.cpp)

set_source_files_properties(src/pagerank/pagerank.cpp PROPERTIES LANGUAGE CUDA)

target_link_libraries(dist_pagerank 
    ${MPI_CXX_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${NCCL_HOME}/lib/libnccl.so
    ${CUDA_DRIVER_LIBRARY}
    cudart
    CCCL::CCCL
    CCCL::cudax
)

# #### MPI+NCCL DIST BFS ####
add_executable(no_stf_dist_bfs src/dist_bfs/no_stf_dist_bfs.cpp)

set_source_files_properties(src/dist_bfs/no_stf_dist_bfs.cpp PROPERTIES LANGUAGE CUDA)

target_link_libraries(no_stf_dist_bfs 
    ${MPI_CXX_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${NCCL_HOME}/lib/libnccl.so
    ${CUDA_DRIVER_LIBRARY}
    cudart
    CCCL::CCCL
    CCCL::cudax
)