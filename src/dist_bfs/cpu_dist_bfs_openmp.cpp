#include <mpi.h>
#include <omp.h>  // Add OpenMP header
#include <sys/time.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <queue>
#include <random>
#include <sstream>
#include <string>
#include <unordered_set>
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

  // Get number of available threads for OpenMP
  int num_threads = omp_get_max_threads();
  if (world_rank == 0) {
    printf("Using OpenMP with %d threads per MPI rank\n", num_threads);
  }

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
  // Replace sent_vertices with a vector of sets for better tracking
  std::vector<std::unordered_set<vertex_t>> sent_vertices_by_rank(world_size);

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
  int level = 1;
  bool done = false;

  while (!done) {
    Timer level_timer;
    level_timer.start();

    // Process current frontier - parallelize with OpenMP
    next_frontier.clear();

    // Use a thread-local container to avoid contention
    std::vector<std::vector<vertex_t>> thread_next_frontiers(num_threads);

#pragma omp parallel
    {
      int thread_id = omp_get_thread_num();
      auto& local_next = thread_next_frontiers[thread_id];

#pragma omp for schedule(dynamic, 64)
      for (size_t f = 0; f < frontier.size(); f++) {
        vertex_t v = frontier[f];
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

              // Use atomic update to avoid race conditions
              bool was_visited;
#pragma omp atomic capture
              {
                was_visited = visited[local_neighbor];
                visited[local_neighbor] = 1;
              }

              if (!was_visited) {
                distances[local_neighbor] = level + 1;
                local_next.push_back(neighbor);
              }
            } else {
              // Send to other process - use thread-local tracking
              int dest_rank = neighbor % world_size;
              bool should_send = false;

#pragma omp critical(sent_vertices_update)
              {
                if (sent_vertices_by_rank[dest_rank].find(neighbor) ==
                    sent_vertices_by_rank[dest_rank].end()) {
                  sent_vertices_by_rank[dest_rank].insert(neighbor);
                  should_send = true;
                }
              }

              if (should_send) {
                local_next.push_back(neighbor);
              }
            }
          }
        }
      }
    }

    // Combine thread-local results
    for (auto& thread_frontier : thread_next_frontiers) {
      next_frontier.insert(next_frontier.end(), thread_frontier.begin(),
                           thread_frontier.end());
    }

    // Separate local and remote vertices in next_frontier - can be parallelized
    std::vector<vertex_t> local_next_frontier;
    std::vector<vertex_t> remote_next_frontier;

#pragma omp parallel
    {
      std::vector<vertex_t> thr_local, thr_remote;

#pragma omp for schedule(static)
      for (size_t i = 0; i < next_frontier.size(); i++) {
        vertex_t v = next_frontier[i];
        if (v % world_size == world_rank) {
          thr_local.push_back(v);
        } else {
          thr_remote.push_back(v);
        }
      }

#pragma omp critical(combine_frontiers)
      {
        local_next_frontier.insert(local_next_frontier.end(), thr_local.begin(),
                                   thr_local.end());
        remote_next_frontier.insert(remote_next_frontier.end(),
                                    thr_remote.begin(), thr_remote.end());
      }
    }

    // Exchange frontier size
    int local_size = remote_next_frontier.size();
    std::vector<int> global_sizes(world_size);

    Timer comm_timer;
    comm_timer.start();
    MPI_CHECK(MPI_Allgather(&local_size, 1, MPI_INT, global_sizes.data(), 1,
                            MPI_INT, MPI_COMM_WORLD));

    // Calculate send/recv counts - only for remote vertices
    std::vector<int> send_counts(world_size, 0);
    for (vertex_t v : remote_next_frontier) {
      int dest_rank = v % world_size;
      send_counts[dest_rank]++;
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

    // print total send/recv and counts
    // if (world_rank == 0) {
    //   printf("Total send: %d, Total recv: %d\n", total_send, total_recv);
    //   printf("Send counts: ");
    //   for (int i = 0; i < world_size; i++) {
    //     printf("%d ", send_counts[i]);
    //   }
    //   printf("\n");
    //   printf("Recv counts: ");
    //   for (int i = 0; i < world_size; i++) {
    //     printf("%d ", recv_counts[i]);
    //   }
    //   printf("\n");
    // }

    // Prepare send buffer
    std::vector<vertex_t> send_buffer(total_send);
    std::vector<int> send_pos = send_displs;

    for (vertex_t v : remote_next_frontier) {
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

    // Process received vertices - parallelize this too
    frontier.clear();

    // Add local vertices to frontier first
    frontier.insert(frontier.end(), local_next_frontier.begin(),
                    local_next_frontier.end());

    // Use thread-local storage for received vertices
    std::vector<std::vector<vertex_t>> thread_frontiers(num_threads);

#pragma omp parallel
    {
      int thread_id = omp_get_thread_num();
      auto& thread_frontier = thread_frontiers[thread_id];

#pragma omp for schedule(dynamic, 64)
      for (int i = 0; i < total_recv; i++) {
        vertex_t v = recv_buffer[i];
        // Process only if it belongs to me
        if (v % world_size == world_rank) {
          vertex_t local_v = v / world_size;

          bool was_visited;
#pragma omp atomic capture
          {
            was_visited = visited[local_v];
            visited[local_v] = 1;
          }

          if (!was_visited) {
            distances[local_v] = level + 1;
            thread_frontier.push_back(v);
          }
        }
      }
    }

    // Combine thread-local frontiers
    for (auto& thread_frontier : thread_frontiers) {
      frontier.insert(frontier.end(), thread_frontier.begin(),
                      thread_frontier.end());
    }

    // Reset sent vertices for next iteration
    for (auto& sent_set : sent_vertices_by_rank) {
      sent_set.clear();
    }

    // More accurate timers - make sure to track all MPI operations
    Timer termination_timer;
    termination_timer.start();

    // Check termination condition
    int local_frontier_size = frontier.size();
    int global_frontier_size = 0;
    MPI_CHECK(MPI_Allreduce(&local_frontier_size, &global_frontier_size, 1,
                            MPI_INT, MPI_SUM, MPI_COMM_WORLD));

    double term_comm_time = termination_timer.elapsed();
    perf.comm_time +=
        term_comm_time;  // Add termination check communication time

    done = (global_frontier_size == 0);

    // if (world_rank == 0) {
    //   printf("Level %d: Global new vertices discovered: %d\n", level + 1,
    //          global_frontier_size);
    // }

    level++;
    // if (level > 30) {
    //   done = true;
    //   if (world_rank == 0) {
    //     printf("WARNING: Terminated at maximum level limit (30)\n");
    //   }
    // }

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
