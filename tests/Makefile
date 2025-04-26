NP ?= 1
NVCC = nvcc
MPICXX = mpicxx
MPIRUN ?= mpirun
CUDA_HOME ?= /N/soft/sles15/cuda/12.2
CUDASTF_HOME ?= $(HOME)/lib/cccl/cudax/include
NCCL_HOME ?= /N/soft/sles15/nccl/2.25.1.1
NVCC_FLAGS = -O3 -std=c++17 --expt-relaxed-constexpr --extended-lambda -I$(CUDASTF_HOME)
MPICXX_FLAGS += -I$(CUDA_HOME)/include -I$(NCCL_HOME)/include -std=c++17
LD_FLAGS = -L$(CUDA_HOME)/lib64 -L$(NCCL_HOME)/lib -lnccl -ldl -lcudart -lcuda

mpi_nccl_testing: Makefile mpi_nccl_testing.cpp stf_testing.o
        $(MPICXX) $(MPICXX_FLAGS) mpi_nccl_testing.cpp stf_testing.o $(LD_FLAGS) -o mpi_nccl_testing

stf_testing.o: Makefile stf_testing.cu
        $(NVCC) $(NVCC_FLAGS) stf_testing.cu -c

mpi_nccl_stf_test: Makefile mpi_nccl_testing.cpp
        $(MPICXX) -I$(CUDASTF_HOME) $(MPICXX_FLAGS) mpi_nccl_testing.cpp $(LD_FLAGS) -o mpi_nccl_stf_test

clean:
        rm -f mpi_nccl_testing mpi_nccl_stf_test *.o