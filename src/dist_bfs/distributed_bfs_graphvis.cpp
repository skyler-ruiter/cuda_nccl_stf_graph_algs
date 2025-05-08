#include <cuda_runtime.h>  // CUDA Runtime API
#include <mpi.h>           // MPI API
#include <nccl.h>          // NCCL API
#include <sys/time.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cuda/experimental/stf.cuh>  // CUDASTF API
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "dist_bfs_kernels.cuh"  // BFS kernels

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
  mutable logical_data<slice<vertex_t>> l_frontier;
  mutable logical_data<slice<vertex_t>> l_next_frontier;
  mutable logical_data<slice<int>> l_visited;
  mutable logical_data<slice<int>> l_distances;
  mutable logical_data<slice<vertex_t>> l_frontier_size;
  mutable logical_data<slice<vertex_t>> l_next_frontier_size;
  mutable logical_data<slice<vertex_t>> l_remote_vertices;
  mutable logical_data<slice<int>> l_remote_counts;

  BFS_Data(context& ctx, vertex_t num_vertices, vertex_t global_vertices, int world_size) {
    l_frontier = ctx.logical_data(shape_of<slice<vertex_t>>(num_vertices)).set_symbol("frontier");
    l_next_frontier = ctx.logical_data(shape_of<slice<vertex_t>>(num_vertices)).set_symbol("next_frontier");
    l_visited = ctx.logical_data(shape_of<slice<int>>(num_vertices)).set_symbol("visited");
    l_distances = ctx.logical_data(shape_of<slice<int>>(num_vertices)).set_symbol("distances");
    l_frontier_size = ctx.logical_data(shape_of<slice<vertex_t>>(1)).set_symbol("frontier_size");
    l_next_frontier_size = ctx.logical_data(shape_of<slice<vertex_t>>(1)).set_symbol("next_frontier_size");
    l_remote_vertices = ctx.logical_data(shape_of<slice<vertex_t>>(global_vertices)).set_symbol("remote_vertices");
    l_remote_counts = ctx.logical_data(shape_of<slice<int>>(world_size)).set_symbol("remote_counts");
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

  // partition the graph where each gpu processes vertices where vertex_id %
  // world_size == world_rank
  graph.row_offsets.resize(graph.num_vertices / world_size + 2, 0);

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
  auto l_row_offsets = ctx.logical_data(graph.row_offsets.data(), graph.row_offsets.size()).set_symbol("row_offsets");
  auto l_column_indices = ctx.logical_data(graph.col_indices.data(), graph.col_indices.size()).set_symbol("column_indices");

  auto l_all_send_counts = ctx.logical_data(shape_of<slice<int>>(world_size * world_size)).set_symbol("all_send_counts");
  auto l_recv_counts = ctx.logical_data(shape_of<slice<int>>(world_size)).set_symbol("recv_counts");
  auto l_send_displs = ctx.logical_data(shape_of<slice<int>>(world_size)).set_symbol("send_displs");
  auto l_recv_displs = ctx.logical_data(shape_of<slice<int>>(world_size)).set_symbol("recv_displs");
  auto l_total_send = ctx.logical_data(shape_of<slice<vertex_t>>(1)).set_symbol("total_send");
  auto l_total_recv = ctx.logical_data(shape_of<slice<vertex_t>>(1)).set_symbol("total_recv");
  auto l_send_counts = ctx.logical_data(shape_of<slice<int>>(world_size)).set_symbol("send_counts");

  std::vector<int> h_send_counts(world_size, 0);
  std::vector<int> h_recv_counts(world_size, 0);
  std::vector<int> h_send_displs(world_size, 0);
  std::vector<int> h_recv_displs(world_size, 0);
  std::vector<int> h_all_send_counts(world_size * world_size, 0);

  logical_data<slice<vertex_t>> l_send_buffer;
  logical_data<slice<vertex_t>> l_recv_buffer;
  vertex_t total_send = 0; 
  vertex_t total_recv = 0; 

  BFS_Data bfs_data(ctx, num_vertices, graph.num_vertices, world_size);

  // #### INITIALIZE DATA STRUCTURES #### //

  ctx.task(bfs_data.l_frontier.write(), bfs_data.l_visited.write(),
           bfs_data.l_distances.write(), bfs_data.l_frontier_size.write()).set_symbol("init_frontier")->*[&]
           (cudaStream_t s, auto frontier, auto visited, auto distances, auto frontier_size) {
            init_frontier_kernel<<<1, 1, 0, s>>>(
                frontier, visited, distances, frontier_size, source_vertex,
                world_size, world_rank);
  };

  ctx.task(bfs_data.l_remote_vertices.write(), bfs_data.l_remote_counts.write(),
           l_send_displs.write(), l_recv_counts.write(), l_recv_displs.write(),
           l_send_counts.write()).set_symbol("init_remote_data")->*[&]
           (cudaStream_t s, auto remote_vertices, auto remote_counts,
            auto send_displs, auto recv_counts, auto recv_displs, auto send_counts) {
            // Initialize sent_vertices to 0
            CHECK(cudaMemsetAsync(remote_vertices.data_handle(), 0, graph.num_vertices * sizeof(vertex_t), s));
            CHECK(cudaMemsetAsync(remote_counts.data_handle(), 0, world_size * sizeof(int), s));
            CHECK(cudaMemsetAsync(send_displs.data_handle(), 0, world_size * sizeof(int), s));
            CHECK(cudaMemsetAsync(recv_counts.data_handle(), 0, world_size * sizeof(int), s));
            CHECK(cudaMemsetAsync(recv_displs.data_handle(), 0, world_size * sizeof(int), s));
            CHECK(cudaMemsetAsync(send_counts.data_handle(), 0, world_size * sizeof(int), s));
  };

  ctx.task(l_total_send.write(), l_total_recv.write()).set_symbol("init_totals")->*[&](cudaStream_t s, auto total_send, auto total_recv) {
    // Initialize with zeros
    CHECK(cudaMemsetAsync(total_send.data_handle(), 0, sizeof(vertex_t), s));
    CHECK(cudaMemsetAsync(total_recv.data_handle(), 0, sizeof(vertex_t), s));
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
  vertex_t next_f_size = 0;

  ctx.repeat([&]() { return !done; })->*[&](context ctx, size_t) {
    //* TIMING
    Timer level_timer;
    level_timer.start();

    // ##### PROCESS FRONTIER #####
    ctx.task(l_row_offsets.read(), l_column_indices.read(),
             bfs_data.l_frontier.read(), bfs_data.l_next_frontier.write(),
             bfs_data.l_visited.write(), bfs_data.l_distances.write(),
             bfs_data.l_frontier_size.read(),
             bfs_data.l_next_frontier_size.write(),
             bfs_data.l_remote_vertices.rw(), bfs_data.l_remote_counts.rw()
            ).set_symbol("process_frontier")->*[&](cudaStream_t s, auto row_offsets, auto column_indices,
            auto frontier, auto next_frontier, auto visited, auto distances,
            auto frontier_size, auto next_frontier_size, auto remote_vertices,
            auto remote_counts) {
          process_frontier_kernel<<<256, 256, 0, s>>>(
              row_offsets, column_indices, frontier, next_frontier, visited,
              distances, frontier_size, next_frontier_size, remote_vertices,
              remote_counts, num_vertices, world_rank, world_size, level);
    };

    ctx.task(l_all_send_counts.write()).set_symbol("reset_send_counts")->*[&](cudaStream_t s, auto all_send_counts) {
        CHECK(cudaMemsetAsync(all_send_counts.data_handle(), 0,
                              world_size * world_size * sizeof(int), s));
    };
    cudaStreamSynchronize(ctx.task_fence());

    ctx.task(bfs_data.l_remote_counts.read(), l_all_send_counts.rw())
      .set_symbol("allgather_counts")->*[&](cudaStream_t s, auto remote_counts, auto all_send_counts) {
          Timer comm_timer;
          comm_timer.start();

          // Gather send counts from all ranks
          NCCL_CHECK(ncclGroupStart());
          NCCL_CHECK(ncclAllGather(remote_counts.data_handle(),
                                    all_send_counts.data_handle(),
                                    world_size, ncclInt, comm, s));
          NCCL_CHECK(ncclGroupEnd());

          double comm_elapsed = comm_timer.elapsed();
          perf.comm_time += comm_elapsed;
          perf.level_comm_times.push_back(comm_elapsed);
    };

    ctx.task(bfs_data.l_remote_counts.read(), l_send_counts.write())
      .set_symbol("copy_send_counts")->*[&](cudaStream_t s, auto remote_counts, auto send_counts) {
          CHECK(cudaMemcpyAsync(
              send_counts.data_handle(), remote_counts.data_handle(),
              world_size * sizeof(int), cudaMemcpyDeviceToDevice, s));
    };
    
    cudaStreamSynchronize(ctx.task_fence());

    // calc displacements
    ctx.task(l_all_send_counts.read(), l_recv_counts.rw())
            .set_symbol("extract_recv_counts")->*[&](cudaStream_t s, auto all_send_counts, auto recv_counts) {
                  extract_recv_counts_kernel<<<1, 1, 0, s>>>(
                      all_send_counts, recv_counts, world_rank, world_size);
    };

    ctx.task(bfs_data.l_remote_counts.read(), l_recv_counts.read(),
            l_send_displs.rw(), l_recv_displs.rw(), l_total_send.rw(),
            l_total_recv.rw()).set_symbol("calc_displacements")->*[&](cudaStream_t s, auto remote_counts, auto recv_counts,
            auto send_displs, auto recv_displs, auto total_send, auto total_recv) {
              calc_displacements_kernel<<<1, 1, 0, s>>>(
                  remote_counts, recv_counts, send_displs, recv_displs, total_send, total_recv,  world_size);
    };


    ctx.task(l_total_send.read(), l_total_recv.read()).set_symbol("copy_totals")->*[&](cudaStream_t s, auto t_send, auto t_recv) {
      CHECK(cudaMemcpyAsync(&total_send, t_send.data_handle(),
                            sizeof(vertex_t), cudaMemcpyDeviceToHost));
      CHECK(cudaMemcpyAsync(&total_recv, t_recv.data_handle(),
                            sizeof(vertex_t), cudaMemcpyDeviceToHost));
    };

    total_send = std::max<vertex_t>(1, total_send);
    total_recv = std::max<vertex_t>(1, total_recv);

    l_send_buffer = ctx.logical_data(shape_of<slice<vertex_t>>(total_send)).set_symbol("send_buffer");
    l_recv_buffer = ctx.logical_data(shape_of<slice<vertex_t>>(total_recv)).set_symbol("recv_buffer");

    ctx.task(l_send_buffer.write(), l_recv_buffer.write()).set_symbol("init_comm_buffers")->*[&](cudaStream_t s, auto send_buffer, auto recv_buffer) {
      if (total_send > 1) CHECK(cudaMemsetAsync(send_buffer.data_handle(), 0, total_send * sizeof(vertex_t), s));
      if (total_recv > 1) CHECK(cudaMemsetAsync(recv_buffer.data_handle(), 0, total_recv * sizeof(vertex_t), s));
    };

    ctx.task(bfs_data.l_remote_vertices.read(), bfs_data.l_remote_counts.read(),
            l_send_buffer.rw(), l_send_counts.read(), l_send_displs.read()).set_symbol("sort_vertices")->*[&]
            (cudaStream_t s, auto remote_vertices, auto remote_counts,
          auto send_buffer, auto send_counts, auto send_displs) {
        if (total_send > 0) {
          sort_by_destination_kernel<<<256, 256, 0, s>>>(
              remote_vertices, remote_counts, send_buffer, send_counts,
              send_displs, num_vertices, world_size);
        }
    };

    ctx.task(bfs_data.l_remote_counts.rw(), l_recv_counts.rw(), l_send_displs.rw(), l_recv_displs.rw(), l_all_send_counts.rw()).set_symbol("copy_to_host")->*[&](cudaStream_t s, auto send_counts, auto recv_counts, auto send_displs, auto recv_displs, auto all_send_counts) {
      CHECK(cudaMemcpy(h_send_counts.data(), send_counts.data_handle(), world_size * sizeof(int), cudaMemcpyDeviceToHost));
      CHECK(cudaMemcpy(h_recv_counts.data(), recv_counts.data_handle(), world_size * sizeof(int), cudaMemcpyDeviceToHost));
      CHECK(cudaMemcpy(h_send_displs.data(), send_displs.data_handle(), world_size * sizeof(int), cudaMemcpyDeviceToHost));
      CHECK(cudaMemcpy(h_recv_displs.data(), recv_displs.data_handle(), world_size * sizeof(int), cudaMemcpyDeviceToHost));
      CHECK(cudaMemcpy(h_all_send_counts.data(), all_send_counts.data_handle(), world_size * world_size * sizeof(int), cudaMemcpyDeviceToHost));
    };

    cudaStreamSynchronize(ctx.task_fence());

    ctx.host_launch().set_symbol("update_recv_counts")->*[&]() {
      for (int i = 0; i < world_size; i++) {
        h_recv_counts[i] = h_all_send_counts[i * world_size + world_rank];
      }
    };

    cudaStreamSynchronize(ctx.task_fence());

    ctx.task(l_send_buffer.rw(), l_recv_buffer.rw()).set_symbol("exchange_vertices")->*[&](cudaStream_t s, auto send_buffer, auto recv_buffer) {
      Timer comm_timer;
      comm_timer.start();

      NCCL_CHECK(ncclGroupStart());
      for (int i = 0; i < world_size; i++) {
        if (i != world_rank) {
          if (h_send_counts[i] > 0) {
            NCCL_CHECK(ncclSend(send_buffer.data_handle() + h_send_displs[i],
                                h_send_counts[i], ncclUint32, i, comm, s));
          }
          if (h_recv_counts[i] > 0) {
            NCCL_CHECK(ncclRecv(recv_buffer.data_handle() + h_recv_displs[i],
                                h_recv_counts[i], ncclUint32, i, comm, s));
          }
        }
      }
      NCCL_CHECK(ncclGroupEnd());

      double comm_elapsed = comm_timer.elapsed();
      perf.comm_time += comm_elapsed;
      perf.level_comm_times.push_back(comm_elapsed);
    };

    ctx.task(l_recv_buffer.read(), l_total_recv.read(), bfs_data.l_visited.rw(),
             bfs_data.l_distances.rw(), bfs_data.l_next_frontier.rw(),
             bfs_data.l_next_frontier_size.rw()).set_symbol("filter_received")->*
            [&](cudaStream_t s, auto recv_buffer, auto t_recv, auto visited,
            auto distances, auto next_frontier, auto next_frontier_size) 
    {
      if (total_recv > 0) {
        filter_received_vertices_kernel<<<256, 256, 0, s>>>(
            recv_buffer, t_recv, visited, distances, next_frontier,
            next_frontier_size, level, num_vertices, world_rank, world_size);
      }
    };

    ctx.task(bfs_data.l_next_frontier_size.read()).set_symbol("get_next_size")->*[&](cudaStream_t s, auto next_frontier_size) {
      CHECK(cudaMemcpy(&next_f_size,  
                      next_frontier_size.data_handle(),
                      sizeof(vertex_t), cudaMemcpyDeviceToHost));
    };

    ctx.host_launch(bfs_data.l_next_frontier_size.rw(),
                    bfs_data.l_frontier.rw(), bfs_data.l_next_frontier.rw(),
                    bfs_data.l_frontier_size.rw()).set_symbol("check_done")->*[&](auto next_frontier_size, 
                    auto frontier, auto next_frontier,auto frontier_size) 
    {
      vertex_t global_next_frontier_size = 0;
      MPI_CHECK(MPI_Allreduce(
          &next_f_size, &global_next_frontier_size, 1,
          MPI_UINT32_T, MPI_SUM, MPI_COMM_WORLD));

      done = (global_next_frontier_size == 0);

      next_f_size = 0;
      total_send = 0;
      total_recv = 0;

      if (!done) {
        level++;
      }
    };

    ctx.task(bfs_data.l_frontier.rw(), bfs_data.l_next_frontier.rw(),
        bfs_data.l_frontier_size.rw(), bfs_data.l_next_frontier_size.rw())
      .set_symbol("swap_frontiers")->*[&](cudaStream_t s, auto frontier, auto next_frontier,
            auto frontier_size, auto next_frontier_size) {
      swap_frontiers_kernel<<<256, 256, 0, s>>>(
          next_frontier, frontier, next_frontier_size, frontier_size);
    };

    cudaStreamSynchronize(ctx.task_fence());

    //* TIMING
    perf.level_times.push_back(level_timer.elapsed());
  };

  //* TIMING
  perf.bfs_time = bfs_timer.elapsed();
  perf.total_time = total_timer.elapsed();

  cudaStreamSynchronize(ctx.task_fence());

  // -------- BFS Statistics --------
  ctx.host_launch(bfs_data.l_visited.read(), bfs_data.l_distances.read())
          .set_symbol("gather_stats")->*
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

        MPI_CHECK(MPI_Reduce(&local_visited, &global_visited, 1, MPI_INT,
                             MPI_SUM, 0, MPI_COMM_WORLD));
        MPI_CHECK(MPI_Reduce(&max_distance, &global_max_distance, 1, MPI_INT,
                             MPI_MAX, 0, MPI_COMM_WORLD));
        MPI_CHECK(MPI_Reduce(&sum_distances, &global_sum_distances, 1,
                             MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD));

        // Collect per-level statistics
        std::vector<int> local_level_counts(level + 1, 0);
        for (int i = 0; i < num_vertices; i++) {
          if (visited(i) > 0 && distances(i) >= 0 && distances(i) <= level) {
            local_level_counts[distances(i)]++;
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
      };

  //! End of BFS Loop

  ctx.finalize();

  // Print performance metrics
  double max_total_time = 0;
  double max_init_time = 0;
  double max_bfs_time = 0;
  double sum_comm_time = 0;

  MPI_CHECK(MPI_Reduce(&perf.total_time, &max_total_time, 1, MPI_DOUBLE,
                       MPI_MAX, 0, MPI_COMM_WORLD));
  MPI_CHECK(MPI_Reduce(&perf.init_time, &max_init_time, 1, MPI_DOUBLE, MPI_MAX,
                       0, MPI_COMM_WORLD));
  MPI_CHECK(MPI_Reduce(&perf.bfs_time, &max_bfs_time, 1, MPI_DOUBLE, MPI_MAX, 0,
                       MPI_COMM_WORLD));
  MPI_CHECK(MPI_Reduce(&perf.comm_time, &sum_comm_time, 1, MPI_DOUBLE, MPI_SUM,
                       0, MPI_COMM_WORLD));

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