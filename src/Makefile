NP ?= 1
NVCC = nvcc
MPICXX = mpicxx
MPIRUN ?= mpirun
CUDA_HOME ?= /N/soft/sles15/cuda/12.2
NCCL_HOME ?= /N/soft/sles15/nccl/2.25.1.1
NVCC_FLAGS = -O3 -std=c++17
MPICXX_FLAGS += -I$(CUDA_HOME)/include -I$(NCCL_HOME)/include -std=c++17
LD_FLAGS = -L$(CUDA_HOME)/lib64 -L$(NCCL_HOME)/lib -lnccl -ldl -lcudart

mpi_nccl_testing: mpi_nccl_testing.cpp
	$(MPICXX) $(MPICXX_FLAGS) mpi_nccl_testing.cpp $(LD_FLAGS) -o mpi_nccl_testing