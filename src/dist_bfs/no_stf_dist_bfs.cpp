#include <cuda_runtime.h>  // CUDA Runtime API
#include <mpi.h>           // MPI API
#include <nccl.h>          // NCCL API
#include <sys/time.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <sstream>

#include "no_bfs_kernels.cuh"  // BFS kernels

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

#define MPI_CHECK(call)                                                   \
  {                                                                       \
    int mpi_status = call;                                                \
    if (MPI_SUCCESS != mpi_status) {                                      \
      char mpi_error_string[MPI_MAX_ERROR_STRING];                        \
      int mpi_error_string_length = 0;                                    \
      MPI_Error_string(mpi_status, mpi_error_string,                      \
                       &mpi_error_string_length);                         \
      if (NULL != mpi_error_string)                                       \
        fprintf(stderr,                                                   \
                "ERROR: MPI call \"%s\" in line %d of file %s failed "    \
                "with %s "                                                \
                "(%d).\n",                                                \
                #call, __LINE__, __FILE__, mpi_error_string, mpi_status); \
      else                                                                \
        fprintf(stderr,                                                   \
                "ERROR: MPI call \"%s\" in line %d of file %s failed "    \
                "with %d.\n",                                             \
                #call, __LINE__, __FILE__, mpi_status);                   \
      exit(mpi_status);                                                   \
    }                                                                     \
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

  vertex_t* d_row_offsets;
  vertex_t* d_column_indices;
};

//! BFS Data Structure
struct BFS_Data {
  vertex_t* d_frontier;
  vertex_t* d_next_frontier;
  int* d_visited;
  int* d_distances;
  vertex_t* d_frontier_size;
  vertex_t* d_next_frontier_size;
  vertex_t* d_remote_vertices;
  int* d_remote_counts;

  BFS_Data(vertex_t num_vertices, vertex_t global_vertices, int world_size) {
    CHECK(cudaMalloc(&d_frontier, num_vertices * sizeof(vertex_t)));
    CHECK(cudaMalloc(&d_next_frontier, num_vertices * sizeof(vertex_t)));
    CHECK(cudaMalloc(&d_visited, num_vertices * sizeof(int)));
    CHECK(cudaMalloc(&d_distances, num_vertices * sizeof(int)));
    CHECK(cudaMalloc(&d_frontier_size, sizeof(vertex_t)));
    CHECK(cudaMalloc(&d_next_frontier_size, sizeof(vertex_t)));
    CHECK(cudaMalloc(&d_remote_vertices, global_vertices * sizeof(vertex_t)));
    CHECK(cudaMalloc(&d_remote_counts, sizeof(int) * world_size));
  }

  ~BFS_Data() {
    CHECK(cudaFree(d_frontier));
    CHECK(cudaFree(d_next_frontier));
    CHECK(cudaFree(d_visited));
    CHECK(cudaFree(d_distances));
    CHECK(cudaFree(d_frontier_size));
    CHECK(cudaFree(d_next_frontier_size));
    CHECK(cudaFree(d_remote_vertices));
    CHECK(cudaFree(d_remote_counts));
  }

  BFS_Data(const BFS_Data&) = delete;
  BFS_Data& operator=(const BFS_Data&) = delete;
};

//! Load and Partition Graph
// ############################################

Graph_CSR load_partition_graph(const std::string& fname, int world_rank,
                               int world_size) {
  Graph_CSR graph;
  std::ifstream file(fname);

  if (!file.is_open()) {
    std::cerr << "Error opening file: " << fname << std::endl;
    exit(EXIT_FAILURE);
  }

  if (fname.find("web-uk") != std::string::npos) {
    // if web-uk data read in first 2 lines as comments and next line as graph statistics
    std::string line;
    std::getline(file, line); // Skip first line
    std::getline(file, line); // Skip second line
    std::getline(file, line); // Read graph statistics
    std::istringstream iss(line);
    vertex_t num_rows, num_cols, num_nnz;
    iss >> num_rows >> num_cols >> num_nnz;
    if (world_rank == 0) {
      std::cout << "Graph statistics: " << num_rows << " rows, " << num_cols << " cols, " << num_nnz << " non-zeros" << std::endl;
    }
  }

  std::vector<std::pair<vertex_t, vertex_t>> edges;
  vertex_t src, dst, max_vertex_id = 0;

  while (file >> src >> dst) {
    edges.emplace_back(src, dst);
    max_vertex_id = std::max(max_vertex_id, std::max(src, dst));
  }

  graph.num_vertices = max_vertex_id + 1;  //? zero indexed or no?
  graph.num_edges = edges.size();

  // partition the graph where each gpu processes vertices where vertex_id %
  // world_size == world_rank
  graph.row_offsets.resize(graph.num_vertices / world_size + 2, 0);

  std::vector<int> edge_counts(graph.num_vertices / world_size + 1, 0);
  for (const auto& edge : edges) {
    if (edge.first % world_size == world_rank) {
      vertex_t local_src = edge.first / world_size;
      edge_counts[local_src]++;
    }
    // Add reverse edge counting
    // if (edge.second % world_size == world_rank) {
    //   vertex_t local_dst = edge.second / world_size;
    //   edge_counts[local_dst]++;
    // }
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
  for (const auto& edge : edges) {
    // Forward edge
    if (edge.first % world_size == world_rank) {
      vertex_t local_src = edge.first / world_size;
      vertex_t position = graph.row_offsets[local_src] + edge_counts[local_src];
      graph.col_indices[position] = edge.second;
      edge_counts[local_src]++;
    }

    // Reverse edge
    // if (edge.second % world_size == world_rank) {
    //   vertex_t local_dst = edge.second / world_size;
    //   vertex_t position = graph.row_offsets[local_dst] + edge_counts[local_dst];
    //   graph.col_indices[position] = edge.first;
    //   edge_counts[local_dst]++;
    // }
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

  // make a cuda stream
  cudaStream_t stream;
  CHECK(cudaStreamCreate(&stream));

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
      printf(
          "Using default graph file: "
          "../data/graph500-scale19-ef16_adj.edges\n");
    }
    graph_file = "../data/graph500-scale19-ef16_adj.edges";
  } else {
    graph_file = argv[2];
    if (world_rank == 0) {
      printf("Using graph file: %s\n", graph_file.c_str());
    }
  }
  Graph_CSR graph = load_partition_graph(graph_file, world_rank, world_size);

  int level = 1;
  int num_vertices = graph.num_vertices / world_size + 1;

  CHECK(cudaMalloc(&graph.d_row_offsets, graph.row_offsets.size() * sizeof(vertex_t)));
  CHECK(cudaMalloc(&graph.d_column_indices, graph.col_indices.size() * sizeof(vertex_t)));
  CHECK(cudaMemcpyAsync(graph.d_row_offsets, graph.row_offsets.data(), graph.row_offsets.size() * sizeof(vertex_t),
                        cudaMemcpyHostToDevice, stream));
  CHECK(cudaMemcpyAsync(graph.d_column_indices, graph.col_indices.data(), graph.col_indices.size() * sizeof(vertex_t),
                        cudaMemcpyHostToDevice, stream));

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

  std::vector<int> h_send_counts(world_size);
  std::vector<int> h_recv_counts(world_size);
  std::vector<int> h_send_displs(world_size);
  std::vector<int> h_recv_displs(world_size);

  // create BFS_data
  BFS_Data bfs_data(num_vertices, graph.num_vertices, world_size);

  init_frontier_kernel<<<1, 1, 0, stream>>>(
      bfs_data.d_frontier, bfs_data.d_visited, bfs_data.d_distances,
      bfs_data.d_frontier_size, source_vertex, world_size, world_rank);

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

  // Allocate fixed-size buffers outside the loop
  int* d_all_send_counts;
  int* d_recv_counts;
  int* d_send_displs;
  int* d_recv_displs;
  int* d_total_send;
  int* d_total_recv;
  int* d_send_counts;
  
  CHECK(cudaMalloc(&d_all_send_counts, world_size * world_size * sizeof(int)));
  CHECK(cudaMalloc(&d_recv_counts, world_size * sizeof(int)));
  CHECK(cudaMalloc(&d_send_displs, world_size * sizeof(int)));
  CHECK(cudaMalloc(&d_recv_displs, world_size * sizeof(int)));
  CHECK(cudaMalloc(&d_total_send, sizeof(int)));
  CHECK(cudaMalloc(&d_total_recv, sizeof(int)));
  CHECK(cudaMalloc(&d_send_counts, world_size * sizeof(int)));

  while (!done) {
    //* TIMING
    Timer level_timer;
    level_timer.start();

    // ##### PROCESS FRONTIER #####
    process_frontier_kernel<<<256, 256, 0, stream>>>(
        graph.d_row_offsets, graph.d_column_indices, bfs_data.d_frontier,
        bfs_data.d_next_frontier, bfs_data.d_visited, bfs_data.d_distances,
        bfs_data.d_frontier_size, bfs_data.d_next_frontier_size,
        bfs_data.d_remote_vertices, bfs_data.d_remote_counts, num_vertices, world_rank, world_size, level);
    
    // #### GATHER SEND COUNTS #### //
    CHECK(cudaMemsetAsync(d_all_send_counts, 0, world_size * world_size * sizeof(int), stream));

    Timer comm_timer;
    comm_timer.start();

    // Gather send counts from all ranks
    NCCL_CHECK(ncclGroupStart());
    NCCL_CHECK(ncclAllGather(bfs_data.d_remote_counts, d_all_send_counts, world_size, ncclInt, comm, stream));
    NCCL_CHECK(ncclGroupEnd());

    double comm_elapsed = comm_timer.elapsed();
    perf.comm_time += comm_elapsed;
    perf.level_comm_times.push_back(comm_elapsed);

    // #### CALC DISPLACEMENTS #### //
    extract_recv_counts_kernel<<<1, 1, 0, stream>>>(
        d_all_send_counts, d_recv_counts, world_rank, world_size);

    // Calculate displacements
    calc_displacements_kernel<<<1, 1, 0, stream>>>(
        bfs_data.d_remote_counts, d_recv_counts, d_send_displs, d_recv_displs,
        d_total_send, d_total_recv, world_size);

    // #### PREPARE COMMUNICATION #### //
    // Get the total send/recv counts to allocate buffers
    vertex_t total_send, total_recv;
    CHECK(cudaMemcpyAsync(&total_send, d_total_send, sizeof(vertex_t),
                          cudaMemcpyDeviceToHost, stream));
    CHECK(cudaMemcpyAsync(&total_recv, d_total_recv, sizeof(vertex_t),
                          cudaMemcpyDeviceToHost, stream));
    CHECK(cudaStreamSynchronize(stream));

    // Allocate buffers
    vertex_t* d_send_buffer = nullptr;
    vertex_t* d_recv_buffer = nullptr;
    if (total_send > 0) {
      CHECK(cudaMalloc(&d_send_buffer, total_send * sizeof(vertex_t)));
    }
    if (total_recv > 0) {
      CHECK(cudaMalloc(&d_recv_buffer, total_recv * sizeof(vertex_t)));
    }

    // init counts
    CHECK(cudaMemsetAsync(d_send_counts, 0, world_size * sizeof(int), stream));
    
    // #### SORT VERTICES INTO SEND BUFFER #### //
    if (total_send > 0) {
      sort_by_destination_kernel<<<256, 256, 0, stream>>>(bfs_data.d_remote_vertices,
        bfs_data.d_remote_counts, d_send_buffer, d_send_counts, d_send_displs, num_vertices, world_size);
    }

    // #### EXCHANGE SEND BUFFERS OF VERTICES #### //
    comm_timer = Timer();
    comm_timer.start();
    
    // Get send counts (what THIS rank will send to others)
    CHECK(cudaMemcpyAsync(h_send_counts.data(), bfs_data.d_remote_counts,
                          world_size * sizeof(int), cudaMemcpyDeviceToHost,
                          stream));

    // Get recv counts (what THIS rank will receive from others)
    int* h_temp_all_counts = new int[world_size * world_size];
    CHECK(cudaMemcpyAsync(h_temp_all_counts, d_all_send_counts,
                          world_size * world_size * sizeof(int),
                          cudaMemcpyDeviceToHost, stream));

    // Extract correct column from the matrix - what others are sending to me
    for (int i = 0; i < world_size; i++) {
      h_recv_counts[i] = h_temp_all_counts[i * world_size + world_rank];
    }
    delete[] h_temp_all_counts;
    CHECK(cudaMemcpyAsync(h_send_displs.data(), d_send_displs,
                          world_size * sizeof(int), cudaMemcpyDeviceToHost, stream)); 
    CHECK(cudaMemcpyAsync(h_recv_displs.data(), d_recv_displs,
                          world_size * sizeof(int), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaStreamSynchronize(stream));

    NCCL_CHECK(ncclGroupStart());
    for (int i = 0; i < world_size; i++) {
      if (i != world_rank) {
        if (h_send_counts[i] > 0) {
          NCCL_CHECK(ncclSend(d_send_buffer + h_send_displs[i],
                              h_send_counts[i], ncclUint32, i, comm, stream));
        }
        if (h_recv_counts[i] > 0) {
          NCCL_CHECK(ncclRecv(d_recv_buffer + h_recv_displs[i],
                              h_recv_counts[i], ncclUint32, i, comm, stream));
        }
      }
    }
    NCCL_CHECK(ncclGroupEnd());

    comm_elapsed = comm_timer.elapsed();
    perf.comm_time += comm_elapsed;
    perf.level_comm_times.push_back(comm_elapsed);

    // #### PROCESS RECEIVED VERTICES #### //

    // Process received vertices first
    if (total_recv > 0) {
      filter_received_vertices_kernel<<<256, 256, 0, stream>>>(
          d_recv_buffer, total_recv, bfs_data.d_visited, bfs_data.d_distances,
          bfs_data.d_next_frontier, bfs_data.d_next_frontier_size, level,
          num_vertices, world_rank, world_size);
    }
    
    vertex_t h_next_frontier_size = 0;
    CHECK(cudaMemcpyAsync(&h_next_frontier_size, bfs_data.d_next_frontier_size,
                          sizeof(vertex_t), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaStreamSynchronize(stream));

    // Sum frontier sizes across all ranks
    vertex_t global_next_frontier_size = 0;
    MPI_CHECK(MPI_Allreduce(&h_next_frontier_size, &global_next_frontier_size,
                            1, MPI_UINT32_T, MPI_SUM, MPI_COMM_WORLD));

    // If no more vertices to process in any rank, we're done
    done = (global_next_frontier_size == 0);

    if (!done) {
      level++;
      // Swap frontier and next_frontier
      std::swap(bfs_data.d_frontier, bfs_data.d_next_frontier);
      std::swap(bfs_data.d_frontier_size, bfs_data.d_next_frontier_size);

      // Reset the next frontier size for the next iteration
      CHECK(cudaMemsetAsync(bfs_data.d_next_frontier_size, 0, sizeof(vertex_t), stream));
      CHECK(cudaMemsetAsync(bfs_data.d_next_frontier, 0, num_vertices * sizeof(vertex_t), stream));
    }

    // Free only the dynamic buffers
    if (d_send_buffer) CHECK(cudaFree(d_send_buffer));
    if (d_recv_buffer) CHECK(cudaFree(d_recv_buffer));

  }

  perf.bfs_time = bfs_timer.elapsed();

  // #### FINALIZE #### //
  // Free the fixed-size buffers after the loop
  CHECK(cudaFree(d_all_send_counts));
  CHECK(cudaFree(d_recv_counts));
  CHECK(cudaFree(d_send_displs));
  CHECK(cudaFree(d_recv_displs));
  CHECK(cudaFree(d_total_send));
  CHECK(cudaFree(d_total_recv));
  CHECK(cudaFree(d_send_counts));
  CHECK(cudaFree(graph.d_row_offsets));
  CHECK(cudaFree(graph.d_column_indices));

  cudaStreamSynchronize(stream);

  perf.total_time = total_timer.elapsed();

  // -------- BFS Statistics --------
  // Count visited vertices
  int local_visited = 0;
  int max_distance = -1;
  long long sum_distances = 0;

  int* h_visited = new int[num_vertices];
  int* h_distances = new int[num_vertices];
  CHECK(cudaMemcpyAsync(h_visited, bfs_data.d_visited, num_vertices *
  sizeof(int), cudaMemcpyDeviceToHost, stream));
  CHECK(cudaMemcpyAsync(h_distances, bfs_data.d_distances, num_vertices *
  sizeof(int), cudaMemcpyDeviceToHost, stream));
  CHECK(cudaStreamSynchronize(stream));

  for (int i = 0; i < num_vertices; i++) {
    if (h_visited[i] > 0) {
      local_visited++;
      max_distance = std::max(max_distance, h_distances[i]);
      sum_distances += h_distances[i];
    }
  }

  // Gather global statistics
  int global_visited = 0;
  int global_max_distance = 0;
  long long global_sum_distances = 0;

  MPI_CHECK(MPI_Reduce(&local_visited, &global_visited, 1, MPI_INT,
                        MPI_SUM, 0, MPI_COMM_WORLD));
  MPI_CHECK(MPI_Reduce(&max_distance, &global_max_distance, 1, MPI_INT,
                        MPI_MAX, 0, MPI_COMM_WORLD));
  MPI_CHECK(MPI_Reduce(&sum_distances, &global_sum_distances, 1,
                        MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD));

  // Collect per-level statistics
  std::vector<int> local_level_counts(level + 1, 0);
  for (int i = 0; i < num_vertices; i++) {
    if (h_visited[i] > 0 && h_distances[i] >= 0 && h_distances[i] <= level) {
      local_level_counts[h_distances[i]]++;
    }
  }

  std::vector<int> global_level_counts(level + 1, 0);
  MPI_CHECK(MPI_Reduce(local_level_counts.data(),
                        global_level_counts.data(), level + 1, MPI_INT,
                        MPI_SUM, 0, MPI_COMM_WORLD));

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

  // Print performance metrics
  double max_total_time = 0;
  double max_init_time = 0;
  double max_bfs_time = 0;
  double sum_comm_time = 0;

  MPI_CHECK(MPI_Reduce(&perf.total_time, &max_total_time, 1, MPI_DOUBLE,
                       MPI_MAX, 0, MPI_COMM_WORLD));
  MPI_CHECK(MPI_Reduce(&perf.init_time, &max_init_time, 1, MPI_DOUBLE,
  MPI_MAX,
                       0, MPI_COMM_WORLD));
  MPI_CHECK(MPI_Reduce(&perf.bfs_time, &max_bfs_time, 1, MPI_DOUBLE, MPI_MAX,
  0,
                       MPI_COMM_WORLD));
  MPI_CHECK(MPI_Reduce(&perf.comm_time, &sum_comm_time, 1, MPI_DOUBLE,
  MPI_SUM,
                       0, MPI_COMM_WORLD));

  if (world_rank == 0) {
    double avg_comm_time = sum_comm_time / world_size;

    printf("\n===== BFS Performance =====\n");
    printf("Total execution time: %.6f s\n", max_total_time);
    printf("Graph loading/init time: %.6f s (%.2f%%)\n", max_init_time,
           100.0 * max_init_time / max_total_time);
    printf("BFS traversal time: %.6f s (%.2f%%)\n", max_bfs_time,
           100.0 * max_bfs_time / max_total_time);
    printf("Communication time: %.6f s (%.2f%% of BFS time)\n",
    avg_comm_time,
           100.0 * avg_comm_time / max_bfs_time);
    printf("Computation time: %.6f s (%.2f%% of BFS time)\n",
           max_bfs_time - avg_comm_time,
           100.0 * (max_bfs_time - avg_comm_time) / max_bfs_time);
    printf("Number of BFS levels: %d\n", level);
    printf("Average time per level: %.6f s\n", max_bfs_time / level);
  }

  // Free NCCL and MPI resources
  NCCL_CHECK(ncclCommDestroy(comm));
  CHECK(cudaStreamDestroy(stream));
  MPI_CHECK(MPI_Finalize());

  return 0;
}