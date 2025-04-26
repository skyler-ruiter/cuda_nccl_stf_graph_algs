#include <cstdio>
#include <cstdlib>
#include <mpi.h>
#include <nccl.h>
#include <cuda_runtime.h>

#define CHECK(cmd) do {                                \
    cudaError_t e = cmd;                               \
    if( e != cudaSuccess ) {                           \
        printf("Failed: Cuda error %s:%d '%s'\n",      \
            __FILE__,__LINE__,cudaGetErrorString(e));  \
        exit(EXIT_FAILURE);                            \
    }                                                  \
} while(0)

#define NCCL_CHECK(cmd) do {                           \
    ncclResult_t r = cmd;                              \
    if (r != ncclSuccess) {                            \
        printf("Failed, NCCL error %s:%d '%s'\n",      \
            __FILE__,__LINE__,ncclGetErrorString(r));  \
        exit(EXIT_FAILURE);                            \
    }                                                  \
} while(0)

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int local_rank = world_rank % 2; // 2 GPUs per node
    CHECK(cudaSetDevice(local_rank));

    // Allocate and initialize local data
    const int N = 4;
    float* sendbuff;
    CHECK(cudaMalloc(&sendbuff, N * sizeof(float)));
    float h_data[N];
    for (int i = 0; i < N; i++) h_data[i] = world_rank * 10 + i;
    CHECK(cudaMemcpy(sendbuff, h_data, N * sizeof(float), cudaMemcpyHostToDevice));

    // Allocate receive buffer
    float* recvbuff;
    CHECK(cudaMalloc(&recvbuff, N * world_size * sizeof(float)));

    // Initialize NCCL
    ncclUniqueId id;
    ncclComm_t comm;
    if (world_rank == 0) ncclGetUniqueId(&id);
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    NCCL_CHECK(ncclCommInitRank(&comm, world_size, id, world_rank));

    // NCCL AllGather
    NCCL_CHECK(ncclAllGather((const void*)sendbuff, (void*)recvbuff, N, ncclFloat, comm, 0));

    // Copy result back to host
    float* h_recv = (float*)malloc(N * world_size * sizeof(float));
    CHECK(cudaMemcpy(h_recv, recvbuff, N * world_size * sizeof(float), cudaMemcpyDeviceToHost));

    // Print results
    MPI_Barrier(MPI_COMM_WORLD);
    for (int r = 0; r < world_size; ++r) {
        if (r == world_rank) {
            printf("Rank %d received: ", world_rank);
            for (int i = 0; i < N * world_size; i++) {
                printf("%.1f ", h_recv[i]);
            }
            printf("\n");
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // Cleanup
    ncclCommDestroy(comm);
    CHECK(cudaFree(sendbuff));
    CHECK(cudaFree(recvbuff));
    free(h_recv);

    MPI_Finalize();
    return 0;
}