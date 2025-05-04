#include <cuda_runtime.h>  // CUDA Runtime API
#include <mpi.h>           // MPI API
#include <nccl.h>          // NCCL API
#include <cuda/experimental/stf.cuh>  // CUDASTF API
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

//! Graph Representation
struct Graph_CSR {
  std::vector<vertex_t> row_offsets;
  std::vector<vertex_t> col_indices;
  vertex_t num_vertices;
  edge_t num_edges;
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
  graph.row_offsets.resize(graph.num_vertices / world_size + 2,
                           0);  //? extra space for last vertex

  std::vector<int> edge_counts(graph.num_vertices / world_size + 1, 0);
  for (const auto& edge : edges) {
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
  for (const auto& edge : edges) {
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
  if (argc > 2) {
    graph_file = argv[2];
  }
  Graph_CSR graph = load_partition_graph(graph_file, world_rank, world_size);

  int local_num_vertices = graph.row_offsets.size() - 1;
  int global_num_vertices = graph.num_vertices;

  float init_rank = 1.0f / global_num_vertices;
  float tolerance = 1e-6f;
  int NITER = 100;

  // Local PageRank vectors
  std::vector<float> local_page_rank(local_num_vertices, init_rank);
  std::vector<float> local_new_page_rank(local_num_vertices, 0.0f);

  // Global PageRank vector for gathering results
  std::vector<float> global_page_rank;
  if (world_rank == 0) {
    global_page_rank.resize(global_num_vertices);
  }

  context ctx;

  // STF logical data
  auto loffsets = ctx.logical_data(&graph.row_offsets[0], graph.row_offsets.size());
  auto lnonzeros = ctx.logical_data(&graph.col_indices[0], graph.col_indices.size());
  auto lpage_rank = ctx.logical_data(&local_page_rank[0], local_page_rank.size());
  auto lnew_page_rank = ctx.logical_data(&local_new_page_rank[0], local_new_page_rank.size());
  auto lmax_diff = ctx.logical_data(shape_of<scalar_view<float>>());

  float* d_all_page_ranks;
  CHECK(cudaMalloc(&d_all_page_ranks, global_num_vertices * sizeof(float)));
  CHECK(cudaMemset(d_all_page_ranks, 0, global_num_vertices * sizeof(float)));

  // Create view of all page ranks
  auto l_all_page_ranks = ctx.logical_data(d_all_page_ranks, global_num_vertices);

  for (int iter = 0; iter < NITER; ++iter) {
    // gather all pagerank values

    NCCL_CHECK(ncclGroupStart());
    NCCL_CHECK(ncclAllGather(
      local_page_rank.data(), // send buff
      d_all_page_ranks + (world_rank * local_num_vertices), // recv buff
      local_num_vertices, // count
      ncclFloat,
      comm,
      0));
    NCCL_CHECK(ncclGroupEnd());


    // end loop early
    break;

  }

  ctx.finalize();

  // Cleanup
  ncclCommDestroy(comm);
  MPI_CHECK(MPI_Finalize());
  return 0;
}