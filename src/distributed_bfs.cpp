#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda/experimental/stf.cuh>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <ctime>

#include "helper_kernels.cuh"
#include "dist_bfs_kernels.cuh"

using namespace cuda::experimental::stf;

using vertex_t = uint32_t;
using edge_t = uint32_t;

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

#define MPI_CHECK(call)                                                                \
    {                                                                                 \
        int mpi_status = call;                                                        \
        if (MPI_SUCCESS != mpi_status) {                                              \
            char mpi_error_string[MPI_MAX_ERROR_STRING];                              \
            int mpi_error_string_length = 0;                                          \
            MPI_Error_string(mpi_status, mpi_error_string, &mpi_error_string_length); \
            if (NULL != mpi_error_string)                                             \
                fprintf(stderr,                                                       \
                        "ERROR: MPI call \"%s\" in line %d of file %s failed "        \
                        "with %s "                                                    \
                        "(%d).\n",                                                    \
                        #call, __LINE__, __FILE__, mpi_error_string, mpi_status);     \
            else                                                                      \
                fprintf(stderr,                                                       \
                        "ERROR: MPI call \"%s\" in line %d of file %s failed "        \
                        "with %d.\n",                                                 \
                        #call, __LINE__, __FILE__, mpi_status);                       \
            exit( mpi_status );                                                       \
        }                                                                             \
    }

// ############################################

struct Graph_CSR {
  std::vector<vertex_t> row_offsets;
  std::vector<vertex_t> col_indices;
  vertex_t num_vertices;
  edge_t num_edges;
};

struct BFS_Data {
  mutable logical_data<slice<vertex_t>> l_frontier;
  mutable logical_data<slice<vertex_t>> l_next_frontier;
  mutable logical_data<slice<int>> l_visited;
  mutable logical_data<slice<int>> l_distances;
  mutable logical_data<slice<vertex_t>> l_frontier_size;
  mutable logical_data<slice<vertex_t>> l_next_frontier_size;
  mutable logical_data<slice<int>> l_sent_vertices;
  
  BFS_Data(context& ctx, vertex_t num_vertices) {
    l_frontier = ctx.logical_data(shape_of<slice<vertex_t>>(num_vertices));
    l_next_frontier = ctx.logical_data(shape_of<slice<vertex_t>>(num_vertices));
    l_visited = ctx.logical_data(shape_of<slice<int>>(num_vertices));
    l_distances = ctx.logical_data(shape_of<slice<int>>(num_vertices));
    l_frontier_size = ctx.logical_data(shape_of<slice<vertex_t>>(1));
    l_next_frontier_size = ctx.logical_data(shape_of<slice<vertex_t>>(1));
    l_sent_vertices = ctx.logical_data(shape_of<slice<int>>(num_vertices));
  }

  BFS_Data(const BFS_Data&) = delete;
  BFS_Data& operator=(const BFS_Data&) = delete;
};

// ############################################

Graph_CSR load_partition_graph(const std::string& fname, int world_rank, int world_size) {
  Graph_CSR graph;
  std::ifstream file(fname);

  if (!file.is_open()) {
    std::cerr << "Error opening file: " << fname << std::endl;
    exit(EXIT_FAILURE);
  }

  std::vector<std::pair<vertex_t, vertex_t>> edges;
  vertex_t src, dst, max_vertex_id = 0;

  while (file >> src >> dst) {
    edges.emplace_back(src, dst);
    max_vertex_id = std::max(max_vertex_id, std::max(src, dst));
  }

  graph.num_vertices = max_vertex_id + 1; //? zero indexed or no?
  graph.num_edges = edges.size();

  // partition the graph where each gpu processes vertices where vertex_id % world_size == world_rank
  graph.row_offsets.resize(graph.num_vertices / world_size + 2, 0); //? extra space for last vertex

  std::vector<int> edge_counts(graph.num_vertices / world_size + 1, 0);
  for (const auto& edge: edges) {
    if (edge.first % world_size == world_rank) {
      // global to local id
      vertex_t local_src = edge.first / world_size;
      edge_counts[local_src]++;
    }
  }

  // row offsets
  edge_t offset = 0;
  for (vertex_t i = 0; i < edge_counts.size(); i++) {
    graph.row_offsets[i] = offset;
    offset += edge_counts[i];
  }

  // spaec for col_indices
  graph.col_indices.resize(offset);

  // reset edge counts
  std::fill(edge_counts.begin(), edge_counts.end(), 0);

  // fill col_indices
  for (const auto& edge: edges) {
    if (edge.first % world_size == world_rank) {
      // global to local id
      vertex_t local_src = edge.first / world_size;
      vertex_t position = graph.row_offsets[local_src] + edge_counts[local_src];
      graph.col_indices[position] = edge.second;
      edge_counts[local_src]++;
    }
  }

  printf("Rank %d: Loaded %ld vertices and %ld edges in local partition\n",
         world_rank, graph.row_offsets.size() - 1, graph.col_indices.size());

  return graph;
}

// ############################################

int main(int argc, char* argv[]) {

  MPI_CHECK(MPI_Init(&argc, &argv));

  int world_size, world_rank;
  MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));
  MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));

  int local_rank = world_rank % 2;  // 2 GPUs per node
  CHECK(cudaSetDevice(local_rank));

  // Initialize NCCL
  ncclUniqueId id;
  ncclComm_t comm;
  if (world_rank == 0) NCCL_CHECK(ncclGetUniqueId(&id));
  MPI_CHECK(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));
  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  NCCL_CHECK(ncclCommInitRank(&comm, world_size, id, world_rank));

  // Load and partition the graph
  std::string graph_file = "../data/graph500-scale19-ef16_adj.edges";
  Graph_CSR graph = load_partition_graph(graph_file, world_rank, world_size);

  context ctx;

  auto l_row_offsets = ctx.logical_data(graph.row_offsets.data(), graph.row_offsets.size());
  auto l_column_indices = ctx.logical_data(graph.col_indices.data(), graph.col_indices.size());

  int level = 0;
  int num_vertices = graph.num_vertices / world_size + 1;

  auto l_global_frontier_sizes = ctx.logical_data(shape_of<slice<vertex_t>>(world_size));
  auto l_send_frontier_sizes = ctx.logical_data(shape_of<slice<vertex_t>>(1));

  BFS_Data bfs_data(ctx, num_vertices);
  vertex_t source_vertex = 1;  // Starting vertex for BFS

  if (world_rank == 0) {
    printf(
      "Rank %d: Source vertex %u (should be handled by rank %u), "
      "num_vertices=%d\n",
      world_rank, source_vertex, source_vertex % world_size, num_vertices);
  }

  ctx.task(bfs_data.l_frontier.write(), bfs_data.l_visited.write(), bfs_data.l_distances.write(), bfs_data.l_frontier_size.write())->*[&](
    cudaStream_t s, auto frontier, auto visited, auto distances, auto frontier_size) {
    // Initialize the BFS data structures
    init_frontier_kernel<<<1, 1, 0, s>>>(
      frontier, visited, distances, frontier_size, source_vertex, world_size, world_rank);
  };

  ctx.task(bfs_data.l_sent_vertices.write())->*[&](cudaStream_t s, auto sent_vertices) {
    // Initialize sent_vertices to 0
    CHECK(cudaMemsetAsync(sent_vertices.data_handle(), 0, num_vertices * sizeof(int), s));
  };

  ctx.host_launch(l_global_frontier_sizes.write(), l_send_frontier_sizes.write())->*[&](
    auto global_frontier_sizes, auto send_frontier_sizes) {
    send_frontier_sizes(0) = 0;
    for (int i = 0; i < world_size; i++) {
      global_frontier_sizes(i) = 0;
    }
  };

  CHECK(cudaStreamSynchronize(ctx.task_fence()));

  // //! #################
  // //! # Main BFS Loop #
  // //! #################

  bool done = false;
  vertex_t total_size = 0;

  ctx.repeat([&]() {return !done;})->*[&](context ctx, size_t) {

    ctx.host_launch(bfs_data.l_frontier_size.read())->*[&](auto frontier_size) {
      if (world_rank == 0) {
        int size = frontier_size(0);
        printf("Level %d: Frontier size %d\n", level, size);
      }
    };

    // current frontier
    ctx.task(
      l_row_offsets.read(), 
      l_column_indices.read(), 
      bfs_data.l_frontier.read(), 
      bfs_data.l_next_frontier.write(), 
      bfs_data.l_visited.write(), 
      bfs_data.l_distances.write(),
      bfs_data.l_frontier_size.read(),
      bfs_data.l_next_frontier_size.write(),
      bfs_data.l_sent_vertices.rw())->*[&](
        cudaStream_t s, 
        auto row_offsets, 
        auto column_indices, 
        auto frontier, 
        auto next_frontier, 
        auto visited, 
        auto distances,
        auto frontier_size, 
        auto next_frontier_size,
        auto sent_vertices) {
      reset_counter_kernel<<<1, 1, 0, s>>>(next_frontier_size);
      process_frontier_kernel<<<256, 256, 0, s>>>(
        row_offsets, column_indices, frontier, next_frontier, visited, distances, frontier_size, next_frontier_size, sent_vertices, world_rank, world_size, level);
    };

    // cuda_safe_call(cudaStreamSynchronize(ctx.task_fence()));

    // Replace this host_launch call
    ctx.task(bfs_data.l_next_frontier_size.read(), l_send_frontier_sizes.write(), l_global_frontier_sizes.write())->*[&](
      cudaStream_t s, auto next_f_size, auto send_f_sizes, auto global_f_sizes) {
      
      // Get the size of the next frontier (do this in device memory)
      // First, use a kernel to copy the value
      copy_kernel<<<1, 1, 0, s>>>(next_f_size, send_f_sizes);
      
      // Allocate temporary buffers for NCCL
      vertex_t* d_recv_buffer;
      CHECK(cudaMalloc(&d_recv_buffer, world_size * sizeof(vertex_t)));
      
      // AllGather the frontier sizes using the stream from the task
      NCCL_CHECK(ncclGroupStart());
      NCCL_CHECK(ncclAllGather(
          send_f_sizes.data_handle(),  // Use the logical data directly
          d_recv_buffer, 
          1, ncclUint32, comm, s));
      NCCL_CHECK(ncclGroupEnd());
      
      // Copy from temp buffer to logical data
      copy_array_kernel<<<(world_size + 255)/256, 256, 0, s>>>(
          d_recv_buffer, global_f_sizes, world_size);
      
      // Free the temporary buffer
      CHECK(cudaFree(d_recv_buffer));
    };


    ctx.task(bfs_data.l_next_frontier.read(), bfs_data.l_next_frontier_size.read(), bfs_data.l_visited.rw(), bfs_data.l_distances.rw(), bfs_data.l_frontier.rw(), bfs_data.l_frontier_size.rw())->*[&](cudaStream_t s, auto next_frontier, auto next_frontier_size, auto visited, auto distances, auto frontier, auto frontier_size) {
      int* d_send_counts;
      CHECK(cudaMalloc(&d_send_counts, world_size * sizeof(int)));
      CHECK(cudaMemset(d_send_counts, 0, world_size * sizeof(int)));

      count_by_destination_kernel<<<256, 256, 0, s>>>(next_frontier, next_frontier_size, d_send_counts, world_size);

      // transfer counts to host
      int* h_send_counts = new int[world_size];
      CHECK(cudaMemcpy(h_send_counts, d_send_counts, world_size * sizeof(int), cudaMemcpyDeviceToHost));

      // exchange send counts and get recv counts
      int* h_recv_counts = new int[world_size];
      MPI_CHECK(MPI_Alltoall(h_send_counts, 1, MPI_INT, h_recv_counts, 1, MPI_INT, MPI_COMM_WORLD));

      // calc send/recv displacements
      int* h_send_displs = new int[world_size];
      int* h_recv_displs = new int[world_size];
      h_send_displs[0] = 0;
      h_recv_displs[0] = 0;

      for (int i = 1; i < world_size; i++) {
        h_send_displs[i] = h_send_displs[i - 1] + h_send_counts[i - 1];
        h_recv_displs[i] = h_recv_displs[i - 1] + h_recv_counts[i - 1];
      }

      // calc total vertices to recieve
      int total_send = h_send_displs[world_size - 1] + h_send_counts[world_size - 1];
      int total_recv = h_recv_displs[world_size - 1] + h_recv_counts[world_size - 1];

      // displacements to device
      int* d_send_displs;
      CHECK(cudaMalloc(&d_send_displs, world_size * sizeof(int)));
      CHECK(cudaMemcpy(d_send_displs, h_send_displs, world_size * sizeof(int), cudaMemcpyHostToDevice));

      vertex_t* d_send_buffer;
      CHECK(cudaMalloc(&d_send_buffer, total_send * sizeof(vertex_t)));

      // sort vertices into send buffer
      sort_by_destination_kernel<<<256, 256, 0, s>>>(next_frontier, next_frontier_size, d_send_buffer, d_send_counts, d_send_displs, world_size);
    
      // allocate recv buffer
      vertex_t* d_recv_buffer;
      CHECK(cudaMalloc(&d_recv_buffer, total_recv * sizeof(vertex_t)));

      cudaStreamSynchronize(s);

      // apparently no cuda-aware-mpi for alltoallv so use host buffers

      // copy send buffer to host
      vertex_t* h_send_buffer = new vertex_t[total_send];
      vertex_t* h_recv_buffer = new vertex_t[total_recv];

      CHECK(cudaMemcpy(h_send_buffer, d_send_buffer, total_send * sizeof(vertex_t), cudaMemcpyDeviceToHost));

      MPI_CHECK(MPI_Alltoallv(h_send_buffer, h_send_counts, h_send_displs, MPI_UINT32_T,
                    h_recv_buffer, h_recv_counts, h_recv_displs, MPI_UINT32_T, MPI_COMM_WORLD));

      // copy recv buffer to device
      CHECK(cudaMemcpy(d_recv_buffer, h_recv_buffer, total_recv * sizeof(vertex_t), cudaMemcpyHostToDevice));

      // process received vertices
      filter_received_vertices_kernel<<<256, 256, 0, s>>>(
          d_recv_buffer, total_recv, visited,
          distances, frontier,
          frontier_size, level + 1, world_rank,
          world_size);

      // Cleanup
      CHECK(cudaFree(d_send_counts));
      CHECK(cudaFree(d_send_displs));
      CHECK(cudaFree(d_send_buffer));
      CHECK(cudaFree(d_recv_buffer));
      delete[] h_send_buffer;
      delete[] h_recv_buffer;
      delete[] h_send_counts;
      delete[] h_recv_counts;
      delete[] h_send_displs;
      delete[] h_recv_displs;
    };

    ctx.task(bfs_data.l_next_frontier_size.rw())->*[&](cudaStream_t s, auto next_frontier_size) {
      // Reset next frontier size for next iteration
      CHECK(cudaMemsetAsync(next_frontier_size.data_handle(), 0, sizeof(vertex_t), s));
    };

    ctx.host_launch(l_global_frontier_sizes.read())->*
      [&](auto global_frontier_sizes) {
        // Increment level for next iteration
        level++;

        // Add safety check to prevent infinite loop
        if (level > 30) {
          done = true;
          if (world_rank == 0) {
            printf("WARNING: Terminated at maximum level limit (30)\n");
          }
          return;
        }

        vertex_t sum = 0;
        for (int i = 0; i < world_size; i++) {
          sum += global_frontier_sizes(i);
        }
        total_size = sum;
        done = (sum == 0);

        if (world_rank == 0) {
          printf("Level %d: Global new vertices discovered: %d\n", level, sum);
        }
    };

  };

  ctx.finalize();

  //! End of BFS Loop

  // Cleanup
  ncclCommDestroy(comm);
  MPI_Finalize();
  return 0;
}