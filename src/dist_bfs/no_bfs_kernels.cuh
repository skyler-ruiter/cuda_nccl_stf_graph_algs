#pragma once

#include <cuda_runtime.h>

using vertex_t = uint32_t;
using edge_t = uint32_t;


__global__ void sort_by_destination_kernel(
    vertex_t* remote_vertices,  // Source: vertices grouped by destination rank
    int* remote_counts,         // How many vertices to send to each rank
    vertex_t* send_buffer,      // Destination: compact buffer for sending
    int* send_counts,           // Running count (used with atomics)
    int* send_displs,           // Starting offset for each rank's section
    int num_vertices,
    int world_size) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // For each destination rank
    for (int rank = 0; rank < world_size; rank++) {
        // Start position in the send buffer for this rank
        int base_offset = send_displs[rank];
        
        // Access the section of remote_vertices for this rank
        for (int i = tid; i < remote_counts[rank]; i += stride) {
            // Get vertex from the remote vertices array
            vertex_t v = remote_vertices[rank * num_vertices + i];
            
            // Place it in the send buffer at the correct position
            send_buffer[base_offset + i] = v;
        }
    }
}

// #####################################################

__global__ void init_frontier_kernel(
  vertex_t* frontier,
  int* visited,
  int* distances,
  vertex_t* frontier_size,
  vertex_t source,
  int world_size,
  int world_rank) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    if (source % world_size == world_rank) {
      vertex_t local_source = source / world_size;
      frontier[0] = source;
      frontier_size[0] = 1;
      visited[local_source] = 1;
      distances[local_source] = 0;
    } else {
      frontier_size[0] = 0;
    }
  }
}

__global__ void process_frontier_kernel(
  const vertex_t* row_offsets,
  const vertex_t* col_indices,
  const vertex_t* frontier,
  vertex_t* next_frontier,
  int* visited,
  int* distances,
  const vertex_t* frontier_size,
  vertex_t* next_frontier_size,
  vertex_t* sent_vertices,
  int* send_counts,
  int num_vertices,
  int world_rank,
  int world_size,
  int level) {
  
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  if (tid == 0) {
    next_frontier_size[0] = 0;
    for (int i = 0; i < world_size; i++) {
      send_counts[i] = 0;
    }
  }

  __syncthreads();

  for (int i = tid; i < frontier_size[0]; i += stride) {
    vertex_t global_vertex = frontier[i];

    vertex_t local_vertex = global_vertex / world_size;

    vertex_t start = row_offsets[local_vertex];
    vertex_t end = row_offsets[local_vertex + 1];

    // Iterate over neighbors
    for (vertex_t edge = start; edge < end; edge++) {
      vertex_t neighbor = col_indices[edge];
      int owner = neighbor % world_size;

      // if belonging to this rank
      if (owner == world_rank) {
        vertex_t local_neighbor = neighbor / world_size;

        // mark as visited if needed
        if (atomicCAS(&visited[local_neighbor], 0, 1) == 0) {
          distances[local_neighbor] = level + 1;

          // add to next frontier
          int pos = atomicAdd(&next_frontier_size[0], 1);
          next_frontier[pos] = neighbor;
        }
      } else {
        // vertex belongs to another rank
        int pos = atomicAdd(&send_counts[owner], 1);
        if (pos < num_vertices) { 
            sent_vertices[owner * num_vertices + pos] = neighbor;
        }
      }
    }
  }
}

__global__ void filter_received_vertices_kernel(
  vertex_t* recv_buffer,
  int total_recv,
  int* visited,
  int* distances,
  vertex_t* next_frontier,
  vertex_t* next_frontier_size,
  int level,
  int num_vertices,
  int world_rank,
  int world_size) {
  
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  
  for (int i = tid; i < total_recv; i += stride) {
    vertex_t global_vertex = recv_buffer[i];
    vertex_t local_vertex = global_vertex / world_size;
    
    // Add bounds check to prevent illegal memory access
    if (local_vertex < num_vertices) { // Add num_vertices parameter to kernel
      // Attempt to mark as visited (only process if not already visited)
      if (atomicCAS(&visited[local_vertex], 0, 1) == 0) {
        distances[local_vertex] = level;
        
        // Add to NEXT frontier, not the current frontier
        int pos = atomicAdd(next_frontier_size, 1);
        if (pos < num_vertices) {
          next_frontier[pos] = global_vertex;
        }
      }
    }
  }
}

__global__ void calc_displacements_kernel(
  int* send_counts,
  int* recv_counts,
  int* send_displs,
  int* recv_displs,
  int* d_total_send,
  int* d_total_recv,
  int world_size) {
    
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid == 0) {
    send_displs[0] = 0;
    recv_displs[0] = 0;
    
    int total_send = send_counts[0];
    int total_recv = recv_counts[0];
    
    for (int i = 1; i < world_size; i++) {
      send_displs[i] = send_displs[i-1] + send_counts[i-1];
      recv_displs[i] = recv_displs[i-1] + recv_counts[i-1];
      
      total_send += send_counts[i];
      total_recv += recv_counts[i];
    }
    
    *d_total_send = total_send;
    *d_total_recv = total_recv;
  }
}

__global__ void extract_recv_counts_kernel(
    const int* all_send_counts,
    int* recv_counts,
    int world_rank,
    int world_size) {
    
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    // Explicitly initialize total_recv to 0
    int total_recv = 0;
    
    for (int i = 0; i < world_size; i++) {
      recv_counts[i] = all_send_counts[i * world_size + world_rank];
      total_recv += recv_counts[i];
    }
  }
}