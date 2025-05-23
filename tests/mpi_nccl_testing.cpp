#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>
#include <vector>
#include <iostream>
#include <string>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda/experimental/stf.cuh> // CUDASTF

using namespace cuda::experimental::stf;

#define CHECK(cmd)                                                  \
  do {                                                              \
    cudaError_t e = cmd;                                            \
    if (e != cudaSuccess) {                                         \
      printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, \
             cudaGetErrorString(e));                                \
      exit(EXIT_FAILURE);                                           \
    }                                                               \
  } while (0)

#define NCCL_CHECK(cmd)                                             \
  do {                                                              \
    ncclResult_t r = cmd;                                           \
    if (r != ncclSuccess) {                                         \
      printf("Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__, \
             ncclGetErrorString(r));                                \
      exit(EXIT_FAILURE);                                           \
    }                                                               \
  } while (0)

int main(int argc, char* argv[]) {
  
  context ctx;

  MPI_Init(&argc, &argv);

  int world_size, world_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  int local_rank = world_rank % 2;  // 2 GPUs per node
  CHECK(cudaSetDevice(local_rank));

  // Initialize NCCL
  ncclUniqueId id;
  ncclComm_t comm;
  if (world_rank == 0) ncclGetUniqueId(&id);
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
  NCCL_CHECK(ncclCommInitRank(&comm, world_size, id, world_rank));

  // Allocate and initialize local data using CuSTF
  const int N = 16;
  float h_senddata[N];
  std::vector<float> h_recvdata(N * world_size);

  // Initialize data
  for (int i = 0; i < N; i++) {
    h_senddata[i] = world_rank * 10.0f + sin(i);
  }

    // Create logical data objects
    auto l_senddata = ctx.logical_data(h_senddata);
    auto l_recvdata = ctx.logical_data(h_recvdata.data(), h_recvdata.size());

    // Instead, use the logical data objects directly in the task:
    ctx.task(l_senddata.read(), l_recvdata.write())->*[&]
        (cudaStream_t stream, auto send_slice, auto recv_slice) {
        NCCL_CHECK(ncclAllGather(
            send_slice.data_handle(), recv_slice.data_handle(), 
            N, ncclFloat, comm, stream
        ));
    };

  // Execute all tasks
  ctx.finalize();

  // Print results
  MPI_Barrier(MPI_COMM_WORLD);
  for (int r = 0; r < world_size; ++r) {
    if (r == world_rank) {
      printf("Rank %d received: ", world_rank);
      for (int i = 0; i < N * world_size; i++) {
        printf("%.2f ", h_recvdata[i]);
      }
      printf("\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  // Cleanup
  ncclCommDestroy(comm);

  MPI_Finalize();
  return 0;
}