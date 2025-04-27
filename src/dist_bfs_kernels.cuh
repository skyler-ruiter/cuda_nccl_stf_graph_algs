#pragma once

#include <cuda_runtime.h>
#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;
using vertex_t = uint32_t;
using edge_t = uint32_t;

// Reset the next_frontier_size counter
__global__ void reset_counter_kernel(slice<vertex_t> counter) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        counter(0) = 0;
    }
}

// process current frontier and populate next one
template <typename T>
__global__ void process_frontier_kernel(
  slice<const vertex_t> row_offsets,
  slice<const vertex_t> col_indices,
  slice<const vertex_t> frontier,
  slice<vertex_t> next_frontier,
  slice<int> visited,
  slice<int> distances,
  T frontier_size,
  slice<vertex_t> next_frontier_size,
  int world_rank,
  int world_size,
  int level) {
  
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  // init next frontier size
  if (tid == 0) {
    next_frontier_size(0) = 0;
  }

  __syncthreads();

  for (int i = tid; i < frontier_size; i += stride) {
    vertex_t global_vertex = frontier(i);

    // if vertex is local
    if (global_vertex % world_size == world_rank) {
      vertex_t local_vertex = global_vertex / world_size;

      // iterate through all neighbors
      for (vertex_t edge_idx = row_offsets(local_vertex); edge_idx < row_offsets(local_vertex + 1); edge_idx++) {
        vertex_t neighbor = col_indices(edge_idx);

        // check partition owner
        int owner = neighbor % world_size;
        vertex_t local_neighbor = neighbor / world_size;

        // check if neighbor is visted
        if (owner == world_rank) {
          if (atomicCAS(&visited(local_neighbor), 0, 1) == 0) {
            // if not visited add to frontier
            int pos = atomicAdd(&next_frontier_size(0), 1);
            next_frontier(pos) = neighbor;
            distances(local_neighbor) = level + 1;
          }
        } else {
          // if in other partition
          int pos = atomicAdd(&next_frontier_size(0), 1);
          next_frontier(pos) = neighbor;
        }
      }
    }
  }
}