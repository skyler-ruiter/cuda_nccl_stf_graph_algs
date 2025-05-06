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
#include <sstream>
#include <string>
#include <vector>

#include "pagerank_kernels.cuh"  // PageRank kernels

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

//! Pagerank Performance structure
struct PagerankPerformance {
  double init_time = 0;
  double pagerank_time = 0;
  double total_time = 0;
  double comm_time = 0;
  std::vector<double> level_times;
  std::vector<double> level_comm_times;

  void reset() {
    init_time = pagerank_time = total_time = comm_time = 0;
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

//! PageRank Data Structure
struct PageRank_Data {
  float* d_pagerank;      // current pagerank values
  float* d_next_pagerank; // next iterations pagerank values
  float* d_outgoing_contrib; // outgoing contributions

  PageRank_Data(vertex_t num_vertices, vertex_t global_vertices, int world_size) {
    CHECK(cudaMalloc(&d_pagerank, num_vertices * sizeof(float)));
    CHECK(cudaMalloc(&d_next_pagerank, num_vertices * sizeof(float)));
    CHECK(cudaMalloc(&d_outgoing_contrib, num_vertices * sizeof(float)));
  }

  ~PageRank_Data() {
    CHECK(cudaFree(d_pagerank));
    CHECK(cudaFree(d_next_pagerank));
    CHECK(cudaFree(d_outgoing_contrib));
  }

  PageRank_Data(const PageRank_Data&) = delete;
  PageRank_Data& operator=(const PageRank_Data&) = delete;
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
    // if web-uk data read in first 2 lines as comments and next line as graph
    // statistics
    std::string line;
    std::getline(file, line);  // Skip first line
    std::getline(file, line);  // Skip second line
    std::getline(file, line);  // Read graph statistics
    std::istringstream iss(line);
    vertex_t num_rows, num_cols, num_nnz;
    iss >> num_rows >> num_cols >> num_nnz;
    if (world_rank == 0) {
      std::cout << "Graph statistics: " << num_rows << " rows, " << num_cols
                << " cols, " << num_nnz << " non-zeros" << std::endl;
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
  Timer total_timer, pagerank_init;
  PagerankPerformance perf;
  total_timer.start();
  pagerank_init.start();

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

  int num_vertices = graph.num_vertices / world_size + 1;

  CHECK(cudaMalloc(&graph.d_row_offsets, graph.row_offsets.size() * sizeof(vertex_t)));
  CHECK(cudaMalloc(&graph.d_column_indices, graph.col_indices.size() * sizeof(vertex_t)));
  CHECK(cudaMemcpyAsync(graph.d_row_offsets, graph.row_offsets.data(), graph.row_offsets.size() * sizeof(vertex_t), cudaMemcpyHostToDevice, stream));
  CHECK(cudaMemcpyAsync(graph.d_column_indices, graph.col_indices.data(), graph.col_indices.size() * sizeof(vertex_t), cudaMemcpyHostToDevice, stream));

  PageRank_Data pagerank_data(num_vertices, graph.num_vertices, world_size);

  const int max_iterations = 20;
  const float damping_factor = 0.85f;
  float* global_pagerank = nullptr;

  float init_value = 1.0f / graph.num_vertices;
  initialize_pagerank_kernel<<<(num_vertices + 255) / 256, 256, 0, stream>>>(
    pagerank_data.d_pagerank, num_vertices, init_value);

  // ### MAIN PAGE RANK LOOP ### //
  for (int iter = 0; iter < max_iterations; iter++) {

    reset_pagerank_kernel<<<(num_vertices + 255) / 256, 256, 0, stream>>>(
      pagerank_data.d_next_pagerank, num_vertices, (1.0f - damping_factor) / graph.num_vertices);

    // Compute local contributions
    pagerank_iterate<<<256, 256, 0, stream>>>(
        graph.d_row_offsets, graph.d_column_indices, pagerank_data.d_pagerank,
        pagerank_data.d_next_pagerank, num_vertices, damping_factor);

    // Synchronize contributions globally
    NCCL_CHECK(ncclAllReduce(pagerank_data.d_next_pagerank,
                             pagerank_data.d_next_pagerank, num_vertices,
                             ncclFloat, ncclSum, comm, stream));

    // Swap pointers for next iteration
    std::swap(pagerank_data.d_pagerank, pagerank_data.d_next_pagerank);

  }  // ### END PAGE RANK LOOP ### //

  return 0;
}