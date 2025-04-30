#pragma once

#include <cuda_runtime.h>
#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;
using vertex_t = uint32_t;
using edge_t = uint32_t;

__global__ void copy_kernel(slice<const vertex_t> src, slice<vertex_t> dst) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        dst(0) = src(0);
    }
}

__global__ void copy_array_kernel(vertex_t* src, slice<vertex_t> dst, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dst(idx) = src[idx];
    }
}

// Count vertices destined for each rank
__global__ void count_by_destination_kernel(
    slice<const vertex_t> next_frontier,
    slice<const vertex_t> next_frontier_size,
    slice<int> send_counts,
    int world_size) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = tid; i < next_frontier_size(0); i += stride) {
        vertex_t vertex = next_frontier(i);
        int dest = vertex % world_size;
        atomicAdd(&send_counts(dest), 1);
    }
}

// Sort vertices by destination rank
__global__ void sort_by_destination_kernel(
    slice<const vertex_t> next_frontier,
    slice<const vertex_t> next_frontier_size,
    slice<vertex_t> send_buffer,
    slice<int> send_counts,
    slice<const int> send_displs,
    int world_size) {
    
    // Skip processing if there are no vertices to process
    if (next_frontier_size(0) == 0) return;
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0) {
        // Reset counts for position tracking
        for (int i = 0; i < world_size; i++) {
            send_counts(i) = 0;
        }
    }
    __syncthreads();
    
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < next_frontier_size(0); i += stride) {
        vertex_t vertex = next_frontier(i);
        int dest = vertex % world_size;
        int pos = atomicAdd(&send_counts(dest), 1);
        send_buffer(send_displs(dest) + pos) = vertex;
    }
}
