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
  slice<vertex_t> sent_vertices,
  slice<int> send_counts,
  int num_vertices,
  int world_rank,
  int world_size,
  int level) {
  
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  // init next frontier size
  if (tid == 0) {
    next_frontier_size(0) = 0;
    for (int i = 0; i < world_size; i++) {
      send_counts(i) = 0;
    }
  }

  __syncthreads();

  for (int i = tid; i < frontier_size(0); i += stride) {
    vertex_t global_vertex = frontier(i);

    vertex_t local_vertex = global_vertex / world_size;

    vertex_t start = row_offsets(local_vertex);
    vertex_t end = row_offsets(local_vertex + 1);

    // Iterate over neighbors
    for (vertex_t edge = start; edge < end; edge++) {
      vertex_t neighbor = col_indices(edge);
      int owner = neighbor % world_size;

      // if belonging to this rank
      if (owner == world_rank) {
        vertex_t local_neighbor = neighbor / world_size;

        // mark as visited if needed
        if (atomicCAS(&visited(local_neighbor), 0, 1) == 0) {
          distances(local_neighbor) = level + 1;

          // add to next frontier
          int pos = atomicAdd(&next_frontier_size(0), 1);
          next_frontier(pos) = neighbor;
        }
      } else {
          // Remote vertex - ADD THIS CHECK
          vertex_t local_neighbor = neighbor / world_size;
          
        if (atomicCAS(&visited(local_neighbor), 0, 1) == 0) {
            // Only send if this is the first time we've seen it
            int pos = atomicAdd(&send_counts(owner), 1);
            if (pos < num_vertices) {
                sent_vertices(owner * num_vertices + pos) = neighbor;
            }
        }
      }
    }
  }
}

// Process received vertices
__global__ void filter_received_vertices_kernel(
    slice<const vertex_t> recv_buffer,
    slice<const vertex_t> total_recv,
    slice<int> visited,
    slice<int> distances,
    slice<vertex_t> next_frontier,
    slice<vertex_t> next_frontier_size,
    int level,
    int num_vertices,
    int world_rank,
    int world_size) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = tid; i < total_recv(0); i += stride) {
      vertex_t global_vertex = recv_buffer(i);
      vertex_t local_vertex = global_vertex / world_size;

      if (local_vertex < num_vertices) {
        if (atomicCAS(&visited(local_vertex), 0, 1) == 0) {
          distances(local_vertex) = level + 1;

          // add to next frontier
          int pos = atomicAdd(&next_frontier_size(0), 1);
          if (pos < num_vertices) {
            next_frontier(pos) = global_vertex;
          }
        }
      }
    }
}

__global__ void extract_recv_counts_kernel(
    slice<const int> all_send_counts,
    slice<int> recv_counts,
    int world_rank,
    int world_size) {

    if (threadIdx.x == 0 && blockIdx.x == 0) {
      int total_recv = 0;
      for (int i = 0; i < world_size; i++) {
        recv_counts(i) = all_send_counts(i * world_size + world_rank);
        total_recv += recv_counts(i);
      }
    }
}

__global__ void calc_displacements_kernel(
  slice<const int> send_counts,
  slice<const int> recv_counts,
  slice<int> send_displs,
  slice<int> recv_displs,
  slice<vertex_t> d_total_send,
  slice<vertex_t> d_total_recv,
  int world_size) {
    
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid == 0) {
    send_displs(0) = 0;
    recv_displs(0) = 0;
    
    vertex_t total_send = send_counts(0);
    vertex_t total_recv = recv_counts(0);
    
    for (int i = 1; i < world_size; i++) {
      send_displs(i) = send_displs(i-1) + send_counts(i-1);
      recv_displs(i) = recv_displs(i-1) + recv_counts(i-1);
      
      total_send += send_counts(i);
      total_recv += recv_counts(i);
    }
    
    d_total_send(0) = total_send;
    d_total_recv(0) = total_recv;
  }
}

// Sort vertices by destination rank
__global__ void sort_by_destination_kernel(
    slice<const vertex_t> remote_vertices,
    slice<const int> remote_counts,
    slice<vertex_t> send_buffer,
    slice<const int> send_counts,
    slice<const int> send_displs,
    int num_vertices,
    int world_size) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int rank = 0; rank < world_size; rank++) {
      int base_offset = send_displs(rank);

      for (int i = tid; i < remote_counts(rank); i += stride) {
        vertex_t v = remote_vertices(rank * num_vertices + i);
        send_buffer(base_offset + i) = v;
      }
    }
    
}

__global__ void swap_frontiers_kernel(
    slice<vertex_t> frontier_in, slice<vertex_t> frontier_out,
    slice<vertex_t> size_in, slice<vertex_t> size_out) {
    
    // Copy size first
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        size_out(0) = size_in(0);
        size_in(0) = 0;  // Reset next frontier size for next iteration
    }
    
    // Copy frontier vertices (use multiple threads)
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = tid; i < size_out(0); i += stride) {
        if (i < size_in(0)) {
            frontier_out(i) = frontier_in(i);
        }
    }
}