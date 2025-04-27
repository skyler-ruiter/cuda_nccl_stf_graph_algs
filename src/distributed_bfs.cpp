#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda/experimental/stf.cuh>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace cuda::experimental::stf;

using vertex_t = int64_t;
using edge_t = int64_t;

// ############################################

struct Graph_CSR {
  std::vector<vertex_t> row_offsets;
  std::vector<vertex_t> col_indices;
  vertex_t num_vertices;
  edge_t num_edges;
};

struct BFS_Data {
  std::vector<vertex_t> frontier;
  std::vector<vertex_t> next_frontier;
  std::vector<int> visited;
  std::vector<int> distances;
  vertex_t frontier_size;
  vertex_t next_frontier_size;
};

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

  graph.num_vertices = max_vertex_id + 1;
  graph.num_edges = edges.size();

  // partition the graph where each gpu processes vertices where vertex_id % world_size == world_rank
  graph.row_offsets.resize(graph.num_vertices / world_size + 2, 0);

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

  context ctx;

  MPI_Init(&argc, &argv);

  int world_size, world_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  int local_rank = world_rank % 2;  // 2 GPUs per node
  CHECK(cudaSetDevice(local_rank));

  // Initialize NCCL
  ncclUniqueId id;
  ncclComm_t comm;
  if (world_rank == 0) ncclGetUniqueId(&id);
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
  NCCL_CHECK(ncclCommInitRank(&comm, world_size, id, world_rank));

  // Load and partition the graph
  std::string graph_file = "../data/graph500-scale21-ef16_adj.edges";
  Graph_CSR graph = load_partition_graph(graph_file, world_rank, world_size);

  auto l_row_offsets = ctx.logical_data(graph.row_offsets.data(), graph.row_offsets.size());
  auto l_column_indices = ctx.logical_data(graph.col_indices.data(), graph.col_indices.size());

  BFS_Data bfs_data;
  vertex_t source_vertex = 0;  // Starting vertex for BFS

  bfs_data.visited.resize(graph.num_vertices, 0);
  bfs_data.distances.resize(graph.num_vertices, -1);
  bfs_data.frontier.resize(graph.num_vertices);
  bfs_data.next_frontier.resize(graph.num_vertices);

  // Initialize BFS from source vertex
  if (source_vertex % world_size == world_rank) {
    // Source vertex is in this partition
    vertex_t local_source = source_vertex / world_size;
    bfs_data.frontier[0] = source_vertex;
    bfs_data.frontier_size = 1;
    bfs_data.visited[local_source] = 1;
    bfs_data.distances[local_source] = 0;
  } else {
    bfs_data.frontier_size = 0;
  }

  // Create logical data objects for BFS data
  auto l_frontier = ctx.logical_data(bfs_data.frontier.data(), bfs_data.frontier.size());
  auto l_next_frontier = ctx.logical_data(bfs_data.next_frontier.data(), bfs_data.next_frontier.size());
  auto l_visited = ctx.logical_data(bfs_data.visited.data(), bfs_data.visited.size());
  auto l_distances = ctx.logical_data(bfs_data.distances.data(), bfs_data.distances.size());

  // TODO: Implement distributed BFS algorithm

  // Cleanup
  ncclCommDestroy(comm);
  MPI_Finalize();
  return 0;
}