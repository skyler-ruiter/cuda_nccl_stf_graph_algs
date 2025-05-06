#include <mpi.h>
#include <sys/time.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <string>
#include <vector>
#include <iostream>
#include <sstream>

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
struct PageRankPerformance {
  double init_time = 0;
  double pagerank_time = 0;
  double total_time = 0;
  double comm_time = 0;
  std::vector<double> iter_times;
  std::vector<double> iter_comm_times;
};

// Graph representation (CSR format)
struct Graph_CSR {
  std::vector<vertex_t> row_offsets;
  std::vector<vertex_t> col_indices;
  vertex_t num_vertices;
  edge_t num_edges;
};

// Load and partition graph (same as BFS version)
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

  Timer total_timer;
  PageRankPerformance perf;
  total_timer.start();

  // Load graph
  std::string graph_file = "../data/graph500-scale21-ef16_adj.edges";
  if (argc > 1) {
    graph_file = argv[1];
  }

  // PageRank parameters
  double damping_factor = 0.85;
  double convergence_threshold = 1e-5;
  int max_iterations = 100;
  
  if (argc > 2) {
    damping_factor = std::stod(argv[2]);
  }
  if (argc > 3) {
    convergence_threshold = std::stod(argv[3]);
  }
  if (argc > 4) {
    max_iterations = std::stoi(argv[4]);
  }

  if (world_rank == 0) {
    printf("PageRank parameters: damping factor=%.2f, convergence threshold=%.6f, max iterations=%d\n", 
           damping_factor, convergence_threshold, max_iterations);
  }

  Timer init_timer;
  init_timer.start();
  Graph_CSR graph = load_partition_graph(graph_file, world_rank, world_size);
  
  // Gather global vertex count for correct initialization
  vertex_t global_vertex_count = 0;
  MPI_CHECK(MPI_Allreduce(&graph.num_vertices, &global_vertex_count, 1, MPI_UINT32_T,
                          MPI_MAX, MPI_COMM_WORLD));
  
  if (world_rank == 0) {
    printf("Global vertex count: %u\n", global_vertex_count);
  }
  
  // Local PageRank data structures
  int local_vertex_count = graph.num_vertices / world_size + 1;
  std::vector<double> pagerank_current(local_vertex_count, 1.0 / global_vertex_count);
  std::vector<double> pagerank_next(local_vertex_count, 0.0);
  
  // Calculate out-degrees for local vertices
  std::vector<int> out_degrees(local_vertex_count, 0);
  for (vertex_t i = 0; i < local_vertex_count; i++) {
    out_degrees[i] = graph.row_offsets[i+1] - graph.row_offsets[i];
  }
  
  perf.init_time = init_timer.elapsed();
  
  // PageRank algorithm
  Timer pagerank_timer;
  pagerank_timer.start();
  
  int iteration = 0;
  double global_diff = 1.0;
  
  while (iteration < max_iterations && global_diff > convergence_threshold) {
    Timer iter_timer;
    iter_timer.start();
    
    // Reset next PageRank values
    std::fill(pagerank_next.begin(), pagerank_next.end(), 0.0);
    
    // Calculate PageRank contributions
    std::vector<std::vector<std::pair<vertex_t, double>>> send_contributions(world_size);
    
    for (vertex_t i = 0; i < local_vertex_count; i++) {
      vertex_t global_vertex_id = i * world_size + world_rank;
      if (global_vertex_id >= global_vertex_count) continue;
      
      if (out_degrees[i] > 0) {
        double contribution = pagerank_current[i] * damping_factor / out_degrees[i];
        
        for (vertex_t j = graph.row_offsets[i]; j < graph.row_offsets[i+1]; j++) {
          vertex_t neighbor = graph.col_indices[j];
          int target_rank = neighbor % world_size;
          vertex_t local_neighbor = neighbor / world_size;
          
          send_contributions[target_rank].push_back({local_neighbor, contribution});
        }
      }
    }
    
    // Exchange contributions
    Timer comm_timer;
    comm_timer.start();
    
    // First exchange sizes
    std::vector<int> send_sizes(world_size);
    for (int i = 0; i < world_size; i++) {
      send_sizes[i] = send_contributions[i].size();
    }
    
    std::vector<int> recv_sizes(world_size);
    MPI_CHECK(MPI_Alltoall(send_sizes.data(), 1, MPI_INT, recv_sizes.data(), 1, MPI_INT, MPI_COMM_WORLD));
    
    // Calculate displacements
    std::vector<int> send_displs(world_size, 0);
    std::vector<int> recv_displs(world_size, 0);
    for (int i = 1; i < world_size; i++) {
      send_displs[i] = send_displs[i-1] + send_sizes[i-1];
      recv_displs[i] = recv_displs[i-1] + recv_sizes[i-1];
    }
    
    // Prepare send buffer
    int total_send = 0;
    for (int i = 0; i < world_size; i++) {
      total_send += send_sizes[i];
    }
    
    int total_recv = 0;
    for (int i = 0; i < world_size; i++) {
      total_recv += recv_sizes[i];
    }
    
    // Separate vertex IDs and contribution values into different buffers
    std::vector<vertex_t> send_vertices(total_send);
    std::vector<double> send_values(total_send);
    int buffer_pos = 0;
    for (int i = 0; i < world_size; i++) {
      for (const auto& contrib : send_contributions[i]) {
        send_vertices[buffer_pos] = contrib.first;
        send_values[buffer_pos] = contrib.second;
        buffer_pos++;
      }
    }
    
    // Use MPI_Alltoallv to exchange contributions with basic types
    std::vector<vertex_t> recv_vertices(total_recv);
    std::vector<double> recv_values(total_recv);
    
    // Exchange vertex IDs and values separately using basic MPI datatypes
    MPI_CHECK(MPI_Alltoallv(send_vertices.data(), send_sizes.data(), send_displs.data(), MPI_UINT32_T,
                            recv_vertices.data(), recv_sizes.data(), recv_displs.data(), MPI_UINT32_T, MPI_COMM_WORLD));
    
    MPI_CHECK(MPI_Alltoallv(send_values.data(), send_sizes.data(), send_displs.data(), MPI_DOUBLE,
                            recv_values.data(), recv_sizes.data(), recv_displs.data(), MPI_DOUBLE, MPI_COMM_WORLD));
    
    double comm_elapsed = comm_timer.elapsed();
    perf.comm_time += comm_elapsed;
    perf.iter_comm_times.push_back(comm_elapsed);
    
    // Process received contributions
    for (int i = 0; i < total_recv; i++) {
      pagerank_next[recv_vertices[i]] += recv_values[i];
    }
    
    // Add random teleport factor (1-damping_factor)/N
    double teleport = (1.0 - damping_factor) / global_vertex_count;
    for (vertex_t i = 0; i < local_vertex_count; i++) {
      vertex_t global_vertex_id = i * world_size + world_rank;
      if (global_vertex_id < global_vertex_count) {
        pagerank_next[i] += teleport;
      }
    }
    
    // Calculate difference
    double local_diff = 0.0;
    for (vertex_t i = 0; i < local_vertex_count; i++) {
      vertex_t global_vertex_id = i * world_size + world_rank;
      if (global_vertex_id < global_vertex_count) {
        local_diff += std::fabs(pagerank_next[i] - pagerank_current[i]);
        pagerank_current[i] = pagerank_next[i];
      }
    }
    
    // Reduce to get global difference
    MPI_CHECK(MPI_Allreduce(&local_diff, &global_diff, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));
    
    iteration++;
    perf.iter_times.push_back(iter_timer.elapsed());
    
    if (world_rank == 0) {
      printf("Iteration %d: global_diff = %.6f\n", iteration, global_diff);
    }
  }
  
  perf.pagerank_time = pagerank_timer.elapsed();
  perf.total_time = total_timer.elapsed();
  
  // Find highest ranked vertices
  std::vector<std::pair<double, vertex_t>> top_vertices;
  for (vertex_t i = 0; i < local_vertex_count; i++) {
    vertex_t global_vertex_id = i * world_size + world_rank;
    if (global_vertex_id < global_vertex_count) {
      top_vertices.push_back({pagerank_current[i], global_vertex_id});
    }
  }
  
  // Sort locally
  std::sort(top_vertices.begin(), top_vertices.end(), 
            [](const auto& a, const auto& b) { return a.first > b.first; });
  
  // Keep only top 10
  if (top_vertices.size() > 10) {
    top_vertices.resize(10);
  }
  
  // Gather top vertices from all processes
  int local_top_count = top_vertices.size();
  std::vector<int> top_counts(world_size);
  MPI_CHECK(MPI_Gather(&local_top_count, 1, MPI_INT, top_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD));
  
  std::vector<int> top_displs(world_size, 0);
  for (int i = 1; i < world_size; i++) {
    top_displs[i] = top_displs[i-1] + top_counts[i-1];
  }
  
  // Create MPI datatype for rank-vertex pairs
  MPI_Datatype mpi_rank_type;
  MPI_Datatype rank_types[2] = {MPI_DOUBLE, MPI_UINT32_T};
  int rank_blocklengths[2] = {1, 1};
  MPI_Aint rank_offsets[2] = {0, sizeof(double)};
  MPI_Type_create_struct(2, rank_blocklengths, rank_offsets, rank_types, &mpi_rank_type);
  MPI_Type_commit(&mpi_rank_type);
  
  std::vector<std::pair<double, vertex_t>> all_top;
  if (world_rank == 0) {
    all_top.resize(std::accumulate(top_counts.begin(), top_counts.end(), 0));
  }
  
  MPI_CHECK(MPI_Gatherv(top_vertices.data(), local_top_count, mpi_rank_type,
                      all_top.data(), top_counts.data(), top_displs.data(), mpi_rank_type, 0, MPI_COMM_WORLD));
  
  // Print performance stats
  double max_total_time = 0;
  double max_init_time = 0;
  double max_pagerank_time = 0;
  double sum_comm_time = 0;
  
  MPI_Reduce(&perf.total_time, &max_total_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&perf.init_time, &max_init_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&perf.pagerank_time, &max_pagerank_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&perf.comm_time, &sum_comm_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  
  if (world_rank == 0) {
    double avg_comm_time = sum_comm_time / world_size;
    
    // Sort and print top 10 vertices
    std::sort(all_top.begin(), all_top.end(), 
              [](const auto& a, const auto& b) { return a.first > b.first; });
    if (all_top.size() > 10) {
      all_top.resize(10);
    }
    
    printf("\n===== PageRank Top Vertices =====\n");
    for (size_t i = 0; i < all_top.size(); i++) {
      printf("%zu. Vertex %u: PageRank = %.6f\n", i+1, all_top[i].second, all_top[i].first);
    }
    
    printf("\n===== CPU PageRank Performance =====\n");
    printf("Total execution time: %.6f s\n", max_total_time);
    printf("Graph loading/init time: %.6f s (%.2f%%)\n", max_init_time,
           100.0 * max_init_time / max_total_time);
    printf("PageRank computation time: %.6f s (%.2f%%)\n", max_pagerank_time,
           100.0 * max_pagerank_time / max_total_time);
    printf("Communication time: %.6f s (%.2f%% of PageRank time)\n", avg_comm_time,
           100.0 * avg_comm_time / max_pagerank_time);
    printf("Computation time: %.6f s (%.2f%% of PageRank time)\n",
           max_pagerank_time - avg_comm_time,
           100.0 * (max_pagerank_time - avg_comm_time) / max_pagerank_time);
    printf("Number of iterations: %d\n", iteration);
    printf("Average time per iteration: %.6f s\n", max_pagerank_time / iteration);
    printf("Converged: %s\n", global_diff <= convergence_threshold ? "Yes" : "No");
  }
  
  MPI_Type_free(&mpi_rank_type);
  MPI_Finalize();
  return 0;
}
