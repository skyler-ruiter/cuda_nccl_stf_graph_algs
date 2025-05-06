#pragma once

#include <cuda_runtime.h>

using vertex_t = uint32_t;
using edge_t = uint32_t;

__global__ void initialize_pagerank_kernel(float *pagerank, int num_vertices, float init_value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_vertices) {
        pagerank[idx] = init_value;
    }
}

// reset to default value
__global__ void reset_pagerank_kernel(float* next_pagerank, int num_vertices, float damping) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_vertices) {
        next_pagerank[idx] = damping;
    }
}

__global__ void pagerank_iterate(vertex_t* row_offsets, vertex_t* col_indices, 
                                float* pagerank, float* next_pagerank, 
                                int num_vertices, float damping_factor) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_vertices) {
        // Get edges for this vertex
        vertex_t start = row_offsets[tid];
        vertex_t end = row_offsets[tid + 1];
        
        // Calculate contributions to neighbors
        float num_edges = end - start;
        float contribution = (num_edges > 0) ? pagerank[tid] / num_edges : 0.0f;
        
        // Send contribution to each neighbor
        for (vertex_t e = start; e < end; e++) {
            vertex_t neighbor = col_indices[e];
            atomicAdd(&next_pagerank[neighbor % num_vertices], contribution);
        }
    }
}