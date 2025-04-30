#include <cuda_runtime.h>             // CUDA Runtime API
#include <mpi.h>                      // MPI API
#include <nccl.h>                     // NCCL API
#include <cuda/experimental/stf.cuh>  // CUDASTF API

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <ctime>
#include <sys/time.h>

#include "helper_kernels.cuh"   // Helper kernels
#include "dist_bfs_kernels.cuh" // BFS kernels

using namespace cuda::experimental::stf;

using vertex_t = uint32_t;
using edge_t = uint32_t;

//! Error Handling Macros
// ############################################
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

//! Timing utility structure
struct Timer {
  struct timeval start_time;

  void start() { gettimeofday(&start_time, NULL); }

  double elapsed() {
    struct timeval current_time;
    gettimeofday(&current_time, NULL);
    return (current_time.tv_sec - start_time.tv_sec) +
           (current_time.tv_usec - start_time.tv_usec) * 1e-6;
  }
};

//! Performance metrics structure
struct BFSPerformance {
  double init_time = 0;
  double bfs_time = 0;
  double total_time = 0;
  double comm_time = 0;
  std::vector<double> level_times;
  std::vector<double> level_comm_times;

  void reset() {
    init_time = bfs_time = total_time = comm_time = 0;
    level_times.clear();
    level_comm_times.clear();
  }
};

//! Graph Representation
struct Graph_CSR {
  std::vector<vertex_t> row_offsets;
  std::vector<vertex_t> col_indices;
  vertex_t num_vertices;
  edge_t num_edges;
};

//! BFS Data Structure
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

//! Load and Partition Graph
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

//! Main Driver
// ############################################

int main(int argc, char* argv[]) {

  // #### INIT #### //

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

  //* TIMING
  Timer total_timer, bfs_init;
  BFSPerformance perf;
  total_timer.start();
  bfs_init.start();

  // #### LOAD & PARTITION GRAPH #### //
  std::string graph_file;
  if (argc < 3) {
    if (world_rank == 0) {
      printf("Usage: %s <source_vertex> <graph_file>\n", argv[0]);
      printf("Using default graph file: ../data/graph500-scale19-ef16_adj.edges\n");
    }
    graph_file = "../data/graph500-scale21-ef16_adj.edges";
  } else {
    graph_file = argv[2];
    if (world_rank == 0) {
      printf("Using graph file: %s\n", graph_file.c_str());
    }
  }
  Graph_CSR graph = load_partition_graph(graph_file, world_rank, world_size);

  int level = 0;
  int num_vertices = graph.num_vertices / world_size + 1;

  // ## SELECT SOURCE VERTEX #### //
  vertex_t source_vertex = 1;

  if (world_rank == 0) {
    if (argc > 1) {
      source_vertex = static_cast<vertex_t>(std::atoi(argv[1]));

      if (source_vertex >= graph.num_vertices) {
        printf("Warning: Source vertex %u exceeds graph size, using vertex 1\n",
               source_vertex);
        source_vertex = 1;
      }
    } else {
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_int_distribution<vertex_t> distrib(0,
                                                      graph.num_vertices - 1);
      source_vertex = distrib(gen);
      printf("Selected random source vertex: %u\n", source_vertex);
    }
  }

  // Broadcast source vertex to all processes
  MPI_CHECK(MPI_Bcast(&source_vertex, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD));

  // Cudastf context
  context ctx;

  // #### LOGICAL DATA #### //
  auto l_row_offsets = ctx.logical_data(graph.row_offsets.data(), graph.row_offsets.size());
  auto l_column_indices = ctx.logical_data(graph.col_indices.data(), graph.col_indices.size());

  auto l_global_frontier_sizes = ctx.logical_data(shape_of<slice<vertex_t>>(world_size));
  auto l_send_frontier_sizes = ctx.logical_data(shape_of<slice<vertex_t>>(1));

  auto l_send_counts = ctx.logical_data(shape_of<slice<int>>(world_size));
  auto l_send_displs = ctx.logical_data(shape_of<slice<int>>(world_size));
  auto l_recv_counts = ctx.logical_data(shape_of<slice<int>>(world_size));
  auto l_recv_displs = ctx.logical_data(shape_of<slice<int>>(world_size));

  BFS_Data bfs_data(ctx, num_vertices);

  // #### INITIALIZE DATA STRUCTURES #### //

  ctx.task(bfs_data.l_frontier.write(), bfs_data.l_visited.write(), bfs_data.l_distances.write(), bfs_data.l_frontier_size.write())->*[&](
    cudaStream_t s, auto frontier, auto visited, auto distances, auto frontier_size) {
    // Initialize the BFS data structures
    init_frontier_kernel<<<1, 1, 0, s>>>(
      frontier, visited, distances, frontier_size, source_vertex, world_size, world_rank);
  };

  ctx.task(bfs_data.l_sent_vertices.write(), l_send_counts.write(), l_send_displs.write(), l_recv_counts.write(), l_recv_displs.write())->*[&](cudaStream_t s, auto sent_vertices, auto send_counts, auto send_displs, auto recv_counts, auto recv_displs) {
    // Initialize sent_vertices to 0
    CHECK(cudaMemsetAsync(sent_vertices.data_handle(), 0, num_vertices * sizeof(int), s));
    CHECK(cudaMemsetAsync(send_counts.data_handle(), 0, world_size * sizeof(int), s));
    CHECK(cudaMemsetAsync(send_displs.data_handle(), 0, world_size * sizeof(int), s));
    CHECK(cudaMemsetAsync(recv_counts.data_handle(), 0, world_size * sizeof(int), s));
    CHECK(cudaMemsetAsync(recv_displs.data_handle(), 0, world_size * sizeof(int), s));
  };

  ctx.host_launch(l_global_frontier_sizes.write(), l_send_frontier_sizes.write())->*[&](
    auto global_frontier_sizes, auto send_frontier_sizes) {
    send_frontier_sizes(0) = 0;
    for (int i = 0; i < world_size; i++) {
      global_frontier_sizes(i) = 0;
    }
  };

  cudaStreamSynchronize(ctx.task_fence());

  //* TIMING
  perf.init_time = bfs_init.elapsed();
  perf.level_times.reserve(50);
  perf.level_comm_times.reserve(50);
  Timer bfs_timer;
  bfs_timer.start();

  //! #################
  //! # Main BFS Loop #
  //! #################

  bool done = false;
  vertex_t total_size = 0;
  int total_send = 0;
  int total_recv = 0;

  ctx.repeat([&]() {return !done;})->*[&](context ctx, size_t) {

    //* TIMING
    Timer level_timer;
    level_timer.start();

    // ##### PROCESS FRONTIER #####
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
      process_frontier_kernel<<<256, 256, 0, s>>>(
        row_offsets, column_indices, frontier, next_frontier, visited, distances, frontier_size, next_frontier_size, sent_vertices, world_rank, world_size, level);
    };

    // #### GATHER FRONTIER SIZES #### //
    ctx.task(
      bfs_data.l_next_frontier_size.read(), 
      l_send_frontier_sizes.write(), 
      l_global_frontier_sizes.write())->*[&](
      cudaStream_t s, auto next_f_size, auto send_f_sizes, auto global_f_sizes) {
      
      copy_kernel<<<1, 1, 0, s>>>(next_f_size, send_f_sizes);

      //* TIMING
      Timer comm_timer;
      comm_timer.start();
      
      // AllGather the frontier sizes using the stream from the task
      NCCL_CHECK(ncclGroupStart());
      NCCL_CHECK(ncclAllGather(
          send_f_sizes.data_handle(),
          global_f_sizes.data_handle(), 
          1, ncclUint32, comm, s));
      NCCL_CHECK(ncclGroupEnd());

      //* TIMING
      double comm_elapsed = comm_timer.elapsed();
      perf.comm_time += comm_elapsed;
      perf.level_comm_times.push_back(comm_elapsed);
    };

    ctx.task(bfs_data.l_next_frontier.read(), bfs_data.l_next_frontier_size.read(), l_send_counts.rw())->*[&](cudaStream_t s, auto next_frontier, auto next_frontier_size, auto send_counts) {
      // Count vertices to send
      count_by_destination_kernel<<<256, 256, 0, s>>>(
          next_frontier, next_frontier_size, send_counts, world_size);
    };

    ctx.task(l_send_counts.rw(), l_recv_counts.rw())->*[&](cudaStream_t s, auto send_counts, auto recv_counts) {
      //* TIMING
      Timer comm_timer;
      comm_timer.start();
      
      // Use AllToAll to exchange counts properly
      NCCL_CHECK(ncclGroupStart());
      for (int i = 0; i < world_size; i++) {
        // Send count[i] to rank i
        NCCL_CHECK(ncclSend(send_counts.data_handle() + i, 1, ncclInt, i, comm, s));
        // Receive from rank i into recv_counts[i]
        NCCL_CHECK(ncclRecv(recv_counts.data_handle() + i, 1, ncclInt, i, comm, s));
      }
      NCCL_CHECK(ncclGroupEnd());

      //* TIMING
      double comm_elapsed = comm_timer.elapsed();
      perf.comm_time += comm_elapsed;
      perf.level_comm_times.push_back(comm_elapsed);
    };

    ctx.host_launch(l_send_counts.rw(), l_recv_counts.rw(), l_send_displs.rw(), l_recv_displs.rw())->*[&](auto send_counts, auto recv_counts, auto send_displs, auto recv_displs) {
      send_displs(0) = 0;
      recv_displs(0) = 0;

      for (int i = 1; i < world_size; i++) {
        send_displs(i) = send_displs(i - 1) + send_counts(i - 1);
        recv_displs(i) = recv_displs(i - 1) + recv_counts(i - 1);
      }

      // calc total vertices to recieve
      total_send = send_displs(world_size - 1) + send_counts(world_size - 1);
      total_recv = recv_displs(world_size - 1) + recv_counts(world_size - 1);

      printf("Rank %d: Level %d - Sending %d vertices, Receiving %d vertices\n",
             world_rank, level, total_send, total_recv);
    
      // print out send and recv counts
      if (world_rank == 0) {
        printf("Rank %d: Send counts: ", world_rank);
        for (int i = 0; i < world_size; i++) {
          printf("%d ", send_counts(i));
        }
        printf("\n");
        printf("Rank %d: Recv counts: ", world_rank);
        for (int i = 0; i < world_size; i++) {
          printf("%d ", recv_counts(i));
        }
        printf("\n");
      }
    };

    // cudaStreamSynchronize(ctx.task_fence());

    if (total_send == 0) total_send = 1;
    if (total_recv == 0) total_recv = 1;

    // #### ALLOCATE SEND/RECV BUFFERS #### //
    auto l_send_buffer = ctx.logical_data(shape_of<slice<vertex_t>>(total_send));
    auto l_recv_buffer = ctx.logical_data(shape_of<slice<vertex_t>>(total_recv));

    ctx.task(l_recv_buffer.write())->*[&](cudaStream_t s, auto recv_buff) {
      CHECK(cudaMemsetAsync(recv_buff.data_handle(), 0, total_recv * sizeof(vertex_t), s));
    };

    ctx.task(bfs_data.l_next_frontier.read(), bfs_data.l_next_frontier_size.read(), l_send_buffer.write(), l_send_counts.rw(), l_send_displs.read())->*[&](cudaStream_t s, auto next_frontier, auto next_frontier_size, auto send_buffer, auto send_counts, auto send_displs) {
      // sort vertices into send buffer
      sort_by_destination_kernel<<<256, 256, 0, s>>>(
          next_frontier, next_frontier_size, send_buffer, send_counts,
          send_displs, world_size);
    };

    // cudaStreamSynchronize(ctx.task_fence());

    ctx.task(l_send_buffer.rw(), l_recv_buffer.rw(), l_send_displs.rw(), l_recv_displs.rw(), l_send_counts.rw(), l_recv_counts.rw())->*[&](cudaStream_t s, auto send_buff, auto recv_buff, auto send_displs, auto recv_displs, auto send_counts, auto recv_counts) {
      Timer comm_timer;
      comm_timer.start();
      
      // Get raw device pointers
      int* d_send_counts = send_counts.data_handle();
      int* d_recv_counts = recv_counts.data_handle();
      int* d_send_displs = send_displs.data_handle();
      int* d_recv_displs = recv_displs.data_handle();
      
      // Create host arrays
      int h_send_counts[world_size];
      int h_recv_counts[world_size];
      int h_send_displs[world_size];
      int h_recv_displs[world_size];
      
      CHECK(cudaMemcpy(h_send_counts, d_send_counts, world_size * sizeof(int), cudaMemcpyDeviceToHost));
      CHECK(cudaMemcpy(h_recv_counts, d_recv_counts, world_size * sizeof(int), cudaMemcpyDeviceToHost));
      CHECK(cudaMemcpy(h_send_displs, d_send_displs, world_size * sizeof(int), cudaMemcpyDeviceToHost));
      CHECK(cudaMemcpy(h_recv_displs, d_recv_displs, world_size * sizeof(int), cudaMemcpyDeviceToHost));

      const int MAX_CHUNK_SIZE = 1000;

      // Process each rank's data in balanced exchanges
      for (int i = 0; i < world_size; i++) {
        if (h_send_counts[i] > 0 || h_recv_counts[i] > 0) {
          int remaining_send = h_send_counts[i];
          int remaining_recv = h_recv_counts[i];
          int send_offset = 0;
          int recv_offset = 0;

          // Process chunks for this rank pair
          while (remaining_send > 0 || remaining_recv > 0) {
            NCCL_CHECK(ncclGroupStart());

            // Handle send if needed
            if (remaining_send > 0) {
              int chunk_size = std::min(MAX_CHUNK_SIZE, remaining_send);
              NCCL_CHECK(ncclSend(
                  send_buff.data_handle() + h_send_displs[i] + send_offset,
                  chunk_size, ncclUint32, i, comm, s));
              remaining_send -= chunk_size;
              send_offset += chunk_size;
            }

            // Handle receive if needed
            if (remaining_recv > 0) {
              int chunk_size = std::min(MAX_CHUNK_SIZE, remaining_recv);
              NCCL_CHECK(ncclRecv(
                  recv_buff.data_handle() + h_recv_displs[i] + recv_offset,
                  chunk_size, ncclUint32, i, comm, s));
              remaining_recv -= chunk_size;
              recv_offset += chunk_size;
            }

            NCCL_CHECK(ncclGroupEnd());

            // Wait for this exchange to complete before next chunk
            cudaStreamSynchronize(s);
          }
        }
      }

      double comm_elapsed = comm_timer.elapsed();
      perf.comm_time += comm_elapsed;
      perf.level_comm_times.push_back(comm_elapsed);
    };

    ctx.task(l_recv_buffer.read(), bfs_data.l_visited.rw(), bfs_data.l_distances.rw(), bfs_data.l_frontier.rw(), bfs_data.l_frontier_size.rw())->*[&](cudaStream_t s, auto recv_buff, auto visited, auto distances, auto frontier, auto frontier_size) {
      // process received vertices
      filter_received_vertices_kernel<<<256, 256, 0, s>>>(
          recv_buff, total_recv, visited, distances, frontier, frontier_size,
          level + 1, world_rank, world_size);
    };

    // Add a frontier size limit to prevent explosive growth
    ctx.task(bfs_data.l_frontier.rw(), bfs_data.l_frontier_size.rw())->*[&](cudaStream_t s, auto frontier, auto frontier_size) {
      // Get current frontier size
      vertex_t current_size;
      CHECK(cudaMemcpy(&current_size, frontier_size.data_handle(), sizeof(vertex_t), cudaMemcpyDeviceToHost));
      
      const vertex_t MAX_FRONTIER_SIZE = 100000;  // Adjust based on your GPU memory
      if (current_size > MAX_FRONTIER_SIZE) {
        // printf("Rank %d: Limiting frontier size from %u to %u for level %d\n", 
        //       world_rank, current_size, MAX_FRONTIER_SIZE, level+1);
        
        CHECK(cudaMemsetAsync(frontier_size.data_handle(), MAX_FRONTIER_SIZE, sizeof(vertex_t), s));
      }
    };

    // After filter_received_vertices_kernel
    ctx.host_launch(bfs_data.l_frontier_size.read())->*[&](auto frontier_size) {
      printf("Rank %d: After filtering, frontier size for level %d is %u\n",
             world_rank, level + 1, frontier_size(0));
    };

    // #### RESET NEXT FRONTIER SIZE #### //
    ctx.task(bfs_data.l_next_frontier_size.rw())->*[&](cudaStream_t s, auto next_frontier_size) {
      // Reset next frontier size for next iteration
      CHECK(cudaMemsetAsync(next_frontier_size.data_handle(), 0, sizeof(vertex_t), s));
    };

    // #### CHECK FOR TERMINATION #### //
    ctx.host_launch(l_global_frontier_sizes.read())->*[&](auto global_frontier_sizes) {
    level++;
    
    printf("Rank %d: Level %d termination check - ", world_rank, level);
    vertex_t sum = 0;
    for (int i = 0; i < world_size; i++) {
      sum += global_frontier_sizes(i);
      printf("%u ", global_frontier_sizes(i));
    }
    printf("(total: %u)\n", sum);
    
    total_size = sum;
    done = (sum == 0);
    
    if (level > 30) {
      done = true;
      if (world_rank == 0) {
        printf("WARNING: Terminated at maximum level limit (30)\n");
      }
    }
  };

    //* TIMING
    perf.level_times.push_back(level_timer.elapsed());

  };

  //* TIMING
  perf.bfs_time = bfs_timer.elapsed();
  perf.total_time = total_timer.elapsed();

  cudaStreamSynchronize(ctx.task_fence());

  // -------- BFS Statistics --------
  ctx.host_launch(bfs_data.l_visited.read(), bfs_data.l_distances.read())->*
    [&](auto visited, auto distances) {
      // Count visited vertices
      int local_visited = 0;
      int max_distance = -1;
      long long sum_distances = 0;

      for (int i = 0; i < num_vertices; i++) {
        if (visited(i) > 0) {
          local_visited++;
          max_distance = std::max(max_distance, distances(i));
          sum_distances += distances(i);
        }
      }

      // Gather global statistics
      int global_visited = 0;
      int global_max_distance = 0;
      long long global_sum_distances = 0;

      MPI_CHECK(MPI_Reduce(&local_visited, &global_visited, 1, MPI_INT, MPI_SUM, 0,
                  MPI_COMM_WORLD));
      MPI_CHECK(MPI_Reduce(&max_distance, &global_max_distance, 1, MPI_INT, MPI_MAX, 0,
                  MPI_COMM_WORLD));
      MPI_CHECK(MPI_Reduce(&sum_distances, &global_sum_distances, 1, MPI_LONG_LONG,
                  MPI_SUM, 0, MPI_COMM_WORLD));

      // Collect per-level statistics
      std::vector<int> local_level_counts(level + 1, 0);
      for (int i = 0; i < num_vertices; i++) {
        if (visited(i) > 0 && distances(i) >= 0 && distances(i) <= level) {
          local_level_counts[distances(i)]++;
        }
      }

      std::vector<int> global_level_counts(level + 1, 0);
      MPI_CHECK(MPI_Reduce(local_level_counts.data(), global_level_counts.data(),
                  level + 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD));

      // Print results
      if (world_rank == 0) {
        double avg_distance =
            global_visited > 0 ? (double)global_sum_distances / global_visited
                                : 0;

        double visit_percent = 100.0 * global_visited / (graph.num_vertices);

        printf("\n===== BFS Statistics =====\n");
        printf("Source vertex: %u\n", source_vertex);
        printf("Visited vertices: %d (%.2f%% of graph)\n", global_visited,
                visit_percent);
        printf("Maximum distance from source: %d\n", global_max_distance);
        printf("Average distance from source: %.2f\n", avg_distance);

        printf("\nVertices discovered per level:\n");
        printf("Level | Count   | Percent\n");
        printf("---------------------------\n");
        for (int i = 0; i <= global_max_distance; i++) {
          printf("%-5d | %-7d | %.2f%%\n", i, global_level_counts[i],
                  (double)global_level_counts[i] * 100 / global_visited);
        }
        printf("===========================\n\n");
      }
  };

  //! End of BFS Loop

  ctx.finalize();

  // Print performance metrics
  double max_total_time = 0;
  double max_init_time = 0;
  double max_bfs_time = 0;
  double sum_comm_time = 0;

  MPI_CHECK(MPI_Reduce(&perf.total_time, &max_total_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
  MPI_CHECK(MPI_Reduce(&perf.init_time, &max_init_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
  MPI_CHECK(MPI_Reduce(&perf.bfs_time, &max_bfs_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
  MPI_CHECK(MPI_Reduce(&perf.comm_time, &sum_comm_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD));

  if (world_rank == 0) {
    double avg_comm_time = sum_comm_time / world_size;

    printf("\n===== BFS Performance =====\n");
    printf("Total execution time: %.6f s\n", max_total_time);
    printf("Graph loading/init time: %.6f s (%.2f%%)\n", max_init_time,
           100.0 * max_init_time / max_total_time);
    printf("BFS traversal time: %.6f s (%.2f%%)\n", max_bfs_time,
           100.0 * max_bfs_time / max_total_time);
    printf("Communication time: %.6f s (%.2f%% of BFS time)\n", avg_comm_time,
           100.0 * avg_comm_time / max_bfs_time);
    printf("Computation time: %.6f s (%.2f%% of BFS time)\n",
           max_bfs_time - avg_comm_time,
           100.0 * (max_bfs_time - avg_comm_time) / max_bfs_time);
    printf("Number of BFS levels: %d\n", level);
    printf("Average time per level: %.6f s\n", max_bfs_time / level);
  }

  // Cleanup
  ncclCommDestroy(comm);
  MPI_CHECK(MPI_Finalize());
  return 0;
}