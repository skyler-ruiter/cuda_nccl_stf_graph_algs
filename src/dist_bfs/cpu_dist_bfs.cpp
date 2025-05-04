#include <mpi.h>
#include <sys/time.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <queue>
#include <random>
#include <string>
#include <vector>

using vertex_t = uint32_t;
using edge_t = uint32_t;

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

// Simple timer class
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

// Performance metrics
struct BFSPerformance {
  double init_time = 0;
  double bfs_time = 0;
  double total_time = 0;
  double comm_time = 0;
  std::vector<double> level_times;
  std::vector<double> level_comm_times;
};

// Graph representation (CSR format)
struct Graph_CSR {
  std::vector<vertex_t> row_offsets;
  std::vector<vertex_t> col_indices;
  vertex_t num_vertices;
  edge_t num_edges;
};

// Load and partition graph (same as GPU version)
Graph_CSR load_partition_graph(const std::string& fname, int world_rank,
                               int world_size) {
  Graph_CSR graph;
  std::ifstream file(fname);

  if (!file.is_open()) {
    fprintf(stderr, "Error opening file: %s\n", fname.c_str());
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  std::vector<std::pair<vertex_t, vertex_t>> edges;
  vertex_t src, dst, max_vertex_id = 0;

  while (file >> src >> dst) {
    edges.emplace_back(src, dst);
    max_vertex_id = std::max(max_vertex_id, std::max(src, dst));
  }

  graph.num_vertices = max_vertex_id + 1;
  graph.num_edges = edges.size();

  // Partition vertices by modulo
  graph.row_offsets.resize(graph.num_vertices / world_size + 2, 0);

  std::vector<int> edge_counts(graph.num_vertices / world_size + 1, 0);
  for (const auto& edge : edges) {
    if (edge.first % world_size == world_rank) {
      vertex_t local_src = edge.first / world_size;
      edge_counts[local_src]++;
    }
  }

  // Calculate row offsets
  edge_t offset = 0;
  for (vertex_t i = 0; i < edge_counts.size(); i++) {
    graph.row_offsets[i] = offset;
    offset += edge_counts[i];
  }

  graph.col_indices.resize(offset);
  std::fill(edge_counts.begin(), edge_counts.end(), 0);

  for (const auto& edge : edges) {
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

int main(int argc, char* argv[]) {
  int world_size, world_rank;
  MPI_Init(&argc, &argv);
  MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));
  MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));

  Timer total_timer;
  BFSPerformance perf;
  total_timer.start();

  // Load graph
  std::string graph_file = "../data/graph500-scale21-ef16_adj.edges";
  if (argc > 2) {
    graph_file = argv[2];
  }

  Timer init_timer;
  init_timer.start();
  Graph_CSR graph = load_partition_graph(graph_file, world_rank, world_size);
  perf.init_time = init_timer.elapsed();

  // Local BFS data structures
  int num_vertices = graph.num_vertices / world_size + 1;
  std::vector<int> visited(num_vertices, 0);
  std::vector<int> distances(num_vertices, -1);
  std::vector<vertex_t> frontier;
  std::vector<vertex_t> next_frontier;
  std::vector<int> sent_vertices(num_vertices, 0);

  // Initialize source vertex
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

  MPI_CHECK(MPI_Bcast(&source_vertex, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD));

  if (world_rank == 0) {
    printf("Starting BFS from source vertex %u (handled by rank %u)\n",
           source_vertex, source_vertex % world_size);
  }

  // Initialize BFS from source vertex
  if (source_vertex % world_size == world_rank) {
    vertex_t local_src = source_vertex / world_size;
    visited[local_src] = 1;
    distances[local_src] = 0;
    frontier.push_back(source_vertex);
  }

  Timer bfs_timer;
  bfs_timer.start();

  // Main BFS loop
  int level = 0;
  bool done = false;

  while (!done) {
    Timer level_timer;
    level_timer.start();

    // Process current frontier
    next_frontier.clear();
    for (vertex_t v : frontier) {
      // Only process if this vertex belongs to me
      if (v % world_size == world_rank) {
        vertex_t local_v = v / world_size;
        vertex_t start = graph.row_offsets[local_v];
        vertex_t end = graph.row_offsets[local_v + 1];

        for (vertex_t i = start; i < end; i++) {
          vertex_t neighbor = graph.col_indices[i];

          // Check if neighbor belongs to this process
          if (neighbor % world_size == world_rank) {
            vertex_t local_neighbor = neighbor / world_size;
            if (!visited[local_neighbor]) {
              visited[local_neighbor] = 1;
              distances[local_neighbor] = level + 1;
              next_frontier.push_back(neighbor);
            }
          } else {
            // Send to other process
            if (!sent_vertices[neighbor / world_size]) {
              sent_vertices[neighbor / world_size] = 1;
              next_frontier.push_back(neighbor);
            }
          }
        }
      }
    }

    // Exchange frontier size
    int local_size = next_frontier.size();
    std::vector<int> global_sizes(world_size);

    Timer comm_timer;
    comm_timer.start();
    MPI_CHECK(MPI_Allgather(&local_size, 1, MPI_INT, global_sizes.data(), 1,
                            MPI_INT, MPI_COMM_WORLD));

    // Calculate send/recv counts
    std::vector<int> send_counts(world_size, 0);
    for (vertex_t v : next_frontier) {
      send_counts[v % world_size]++;
    }

    std::vector<int> recv_counts(world_size);
    MPI_CHECK(MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(),
                           1, MPI_INT, MPI_COMM_WORLD));

    // Calculate displacements
    std::vector<int> send_displs(world_size, 0);
    std::vector<int> recv_displs(world_size, 0);
    for (int i = 1; i < world_size; i++) {
      send_displs[i] = send_displs[i - 1] + send_counts[i - 1];
      recv_displs[i] = recv_displs[i - 1] + recv_counts[i - 1];
    }

    int total_send = send_displs.back() + send_counts.back();
    int total_recv = recv_displs.back() + recv_counts.back();

    // Prepare send buffer
    std::vector<vertex_t> send_buffer(total_send);
    std::vector<int> send_pos = send_displs;

    for (vertex_t v : next_frontier) {
      int dest = v % world_size;
      send_buffer[send_pos[dest]++] = v;
    }

    // Exchange vertices
    std::vector<vertex_t> recv_buffer(total_recv);
    MPI_CHECK(MPI_Alltoallv(send_buffer.data(), send_counts.data(),
                            send_displs.data(), MPI_UINT32_T,
                            recv_buffer.data(), recv_counts.data(),
                            recv_displs.data(), MPI_UINT32_T, MPI_COMM_WORLD));

    double comm_elapsed = comm_timer.elapsed();
    perf.comm_time += comm_elapsed;
    perf.level_comm_times.push_back(comm_elapsed);

    // Process received vertices
    frontier.clear();
    for (int i = 0; i < total_recv; i++) {
      vertex_t v = recv_buffer[i];
      // Process only if it belongs to me
      if (v % world_size == world_rank) {
        vertex_t local_v = v / world_size;
        if (!visited[local_v]) {
          visited[local_v] = 1;
          distances[local_v] = level + 1;
          frontier.push_back(v);
        }
      }
    }

    // Reset sent vertices for next iteration
    std::fill(sent_vertices.begin(), sent_vertices.end(), 0);

    // Check termination condition
    int local_frontier_size = frontier.size();
    int global_frontier_size = 0;
    MPI_CHECK(MPI_Allreduce(&local_frontier_size, &global_frontier_size, 1,
                            MPI_INT, MPI_SUM, MPI_COMM_WORLD));

    done = (global_frontier_size == 0);

    if (world_rank == 0) {
      printf("Level %d: Global new vertices discovered: %d\n", level + 1,
             global_frontier_size);
    }

    level++;
    if (level > 30) {
      done = true;
      if (world_rank == 0) {
        printf("WARNING: Terminated at maximum level limit (30)\n");
      }
    }

    perf.level_times.push_back(level_timer.elapsed());
  }

  perf.bfs_time = bfs_timer.elapsed();
  perf.total_time = total_timer.elapsed();

  // Calculate BFS statistics
  int local_visited = 0;
  int max_distance = -1;
  long long sum_distances = 0;

  for (int i = 0; i < num_vertices; i++) {
    if (visited[i] > 0) {
      local_visited++;
      max_distance = std::max(max_distance, distances[i]);
      sum_distances += distances[i];
    }
  }

  int global_visited = 0;
  int global_max_distance = 0;
  long long global_sum_distances = 0;

  MPI_Reduce(&local_visited, &global_visited, 1, MPI_INT, MPI_SUM, 0,
             MPI_COMM_WORLD);
  MPI_Reduce(&max_distance, &global_max_distance, 1, MPI_INT, MPI_MAX, 0,
             MPI_COMM_WORLD);
  MPI_Reduce(&sum_distances, &global_sum_distances, 1, MPI_LONG_LONG, MPI_SUM,
             0, MPI_COMM_WORLD);

  // Print performance statistics
  double max_total_time = 0;
  double max_init_time = 0;
  double max_bfs_time = 0;
  double sum_comm_time = 0;

  MPI_Reduce(&perf.total_time, &max_total_time, 1, MPI_DOUBLE, MPI_MAX, 0,
             MPI_COMM_WORLD);
  MPI_Reduce(&perf.init_time, &max_init_time, 1, MPI_DOUBLE, MPI_MAX, 0,
             MPI_COMM_WORLD);
  MPI_Reduce(&perf.bfs_time, &max_bfs_time, 1, MPI_DOUBLE, MPI_MAX, 0,
             MPI_COMM_WORLD);
  MPI_Reduce(&perf.comm_time, &sum_comm_time, 1, MPI_DOUBLE, MPI_SUM, 0,
             MPI_COMM_WORLD);

  if (world_rank == 0) {
    double avg_comm_time = sum_comm_time / world_size;

    printf("\n===== CPU BFS Performance =====\n");
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

    double avg_distance =
        global_visited > 0 ? (double)global_sum_distances / global_visited : 0;
    double visit_percent = 100.0 * global_visited / graph.num_vertices;

    printf("\n===== CPU BFS Statistics =====\n");
    printf("Source vertex: %u\n", source_vertex);
    printf("Visited vertices: %d (%.2f%% of graph)\n", global_visited,
           visit_percent);
    printf("Maximum distance from source: %d\n", global_max_distance);
    printf("Average distance from source: %.2f\n", avg_distance);
  }

  MPI_Finalize();
  return 0;
}
