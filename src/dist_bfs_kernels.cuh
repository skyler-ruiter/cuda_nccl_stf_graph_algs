#pragma once

#include <cuda_runtime.h>
#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;
using vertex_t = uint32_t;
using edge_t = uint32_t;

// Kernel to initialize the source data frontier
__global__ void init_frontier_kernel(
  slice<vertex_t> frontier,
  slice<int> visited,
  slice<int> distances,
  slice<vertex_t> frontier_size,
  vertex_t source,
  int world_size,
  int world_rank) {
    
  if (threadIdx.x == 0 && blockIdx.x == 0) {

    if (source % world_size == world_rank) {
      vertex_t local_source = source / world_size;
      frontier(0) = source;
      frontier_size(0) = 1;
      visited(local_source) = 1;
      distances(local_source) = 0;
    } else {
      frontier_size(0) = 0;
    }

  }
}

// process current frontier and populate next one
__global__ void process_frontier_kernel(
  slice<const vertex_t> row_offsets,
  slice<const vertex_t> col_indices,
  slice<const vertex_t> frontier,
  slice<vertex_t> next_frontier,
  slice<int> visited,
  slice<int> distances,
  slice<const vertex_t> frontier_size,
  slice<vertex_t> next_frontier_size,
  slice<int> sent_vertices,
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

  for (int i = tid; i < frontier_size(0); i += stride) {
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
          // Remote vertex - only send if not sent before
          vertex_t global_neighbor_idx = neighbor / world_size;
          if (atomicCAS(&sent_vertices(global_neighbor_idx), 0, 1) == 0) {
            // Only add to next_frontier if we haven't sent it before
            int pos = atomicAdd(&next_frontier_size(0), 1);
            next_frontier(pos) = neighbor;
          }
        }
      }
    }
  }
}

// Process received vertices
__global__ void filter_received_vertices_kernel(
    vertex_t* recv_buffer,
    int total_recv,
    slice<int> visited,
    slice<int> distances,
    slice<vertex_t> frontier,
    slice<vertex_t> frontier_size,
    int level,
    int world_rank,
    int world_size) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = tid; i < total_recv; i += stride) {
        vertex_t global_vertex = recv_buffer[i];
        
        // Check if this vertex belongs to me
        if (global_vertex % world_size == world_rank) {
            // Convert to local vertex ID
            vertex_t local_vertex = global_vertex / world_size;
            
            // Try to mark as visited (atomically)
            if (atomicCAS(&visited(local_vertex), 0, 1) == 0) {
                // If newly visited, set distance
                distances(local_vertex) = level;
                
                // Add to frontier for next level
                int pos = atomicAdd(&frontier_size(0), 1);
                frontier(pos) = global_vertex;
            }
        }
    }
}