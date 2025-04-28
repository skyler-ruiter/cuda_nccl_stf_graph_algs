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
    int* send_counts,
    int world_size) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = tid; i < next_frontier_size(0); i += stride) {
        vertex_t vertex = next_frontier(i);
        int dest = vertex % world_size;
        atomicAdd(&send_counts[dest], 1);
    }
}

// Sort vertices by destination rank
__global__ void sort_by_destination_kernel(
    slice<const vertex_t> next_frontier,
    slice<const vertex_t> next_frontier_size,
    vertex_t* send_buffer,
    int* send_counts,
    int* send_displs,
    int world_size) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0) {
        // Reset counts for position tracking
        for (int i = 0; i < world_size; i++) {
            send_counts[i] = 0;
        }
    }
    __syncthreads();
    
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < next_frontier_size(0); i += stride) {
        vertex_t vertex = next_frontier(i);
        int dest = vertex % world_size;
        int pos = atomicAdd(&send_counts[dest], 1);
        send_buffer[send_displs[dest] + pos] = vertex;
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