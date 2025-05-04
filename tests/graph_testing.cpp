#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <cstdlib>
#include <unordered_set>

using vertex_t = uint32_t;
using edge_t = uint32_t;

// Simple CSR graph structure
struct Graph_CSR {
    std::vector<vertex_t> row_offsets;
    std::vector<vertex_t> col_indices;
    vertex_t num_vertices;
    edge_t num_edges;
};

// Load graph from file
Graph_CSR load_graph(const std::string& fname) {
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
    
    // Use sets to store unique neighbors for each vertex
    std::vector<std::unordered_set<vertex_t>> unique_neighbors(graph.num_vertices);
    
    // Add unique edges in both directions (for undirected graph)
    for (const auto& edge : edges) {
        unique_neighbors[edge.first].insert(edge.second);  // Forward edge
        unique_neighbors[edge.second].insert(edge.first);  // Reverse edge
    }
    
    // Count total unique edges
    edge_t total_edges = 0;
    for (const auto& neighbor_set : unique_neighbors) {
        total_edges += neighbor_set.size();
    }
    graph.num_edges = total_edges;
    
    // Initialize row_offsets with zeros
    graph.row_offsets.resize(graph.num_vertices + 1, 0);

    // Set row_offsets based on unique neighbor counts
    for (vertex_t i = 0; i < graph.num_vertices; i++) {
        graph.row_offsets[i + 1] = unique_neighbors[i].size();
    }

    // Cumulative sum to get row offsets
    for (vertex_t i = 1; i <= graph.num_vertices; i++) {
        graph.row_offsets[i] += graph.row_offsets[i-1];
    }

    // Allocate space for column indices
    graph.col_indices.resize(graph.num_edges);

    // Fill column indices with unique neighbors
    for (vertex_t i = 0; i < graph.num_vertices; i++) {
        vertex_t position = graph.row_offsets[i];
        for (vertex_t neighbor : unique_neighbors[i]) {
            graph.col_indices[position++] = neighbor;
        }
    }

    return graph;
}

// Get neighbors of a vertex
std::vector<vertex_t> get_neighbors(const Graph_CSR& graph, vertex_t vertex) {
    std::vector<vertex_t> neighbors;
    
    if (vertex >= graph.num_vertices) {
        std::cerr << "Vertex " << vertex << " is out of range (max: " 
                  << graph.num_vertices - 1 << ")" << std::endl;
        return neighbors;
    }
    
    vertex_t start = graph.row_offsets[vertex];
    vertex_t end = graph.row_offsets[vertex + 1];
    
    neighbors.reserve(end - start);
    for (vertex_t i = start; i < end; i++) {
        neighbors.push_back(graph.col_indices[i]);
    }
    
    return neighbors;
}

// Get neighbors of a vertex and their associated ranks
std::vector<std::pair<vertex_t, int>> get_neighbors_with_ranks(const Graph_CSR& graph, vertex_t vertex, int world_size) {
    std::vector<std::pair<vertex_t, int>> neighbors_with_ranks;
    
    if (vertex >= graph.num_vertices) {
        std::cerr << "Vertex " << vertex << " is out of range (max: " 
                  << graph.num_vertices - 1 << ")" << std::endl;
        return neighbors_with_ranks;
    }
    
    vertex_t start = graph.row_offsets[vertex];
    vertex_t end = graph.row_offsets[vertex + 1];
    
    neighbors_with_ranks.reserve(end - start);
    for (vertex_t i = start; i < end; i++) {
        vertex_t neighbor = graph.col_indices[i];
        int rank = neighbor % world_size; // Determine which rank this vertex belongs to
        neighbors_with_ranks.emplace_back(neighbor, rank);
    }
    
    return neighbors_with_ranks;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <vertex> <graph_file> [world_size]" << std::endl;
        return 1;
    }
    
    vertex_t vertex = static_cast<vertex_t>(std::stoul(argv[1]));
    std::string graph_file = argv[2];
    
    int world_size = 4; // Default to 4 ranks
    if (argc > 3) {
        world_size = std::stoi(argv[3]);
    }
    
    std::cout << "Loading graph from: " << graph_file << std::endl;
    std::cout << "Using world_size = " << world_size << std::endl;
    Graph_CSR graph = load_graph(graph_file);
    
    std::cout << "Graph loaded with " << graph.num_vertices << " vertices and " 
              << graph.num_edges << " edges" << std::endl;
    
    if (vertex >= graph.num_vertices) {
        std::cerr << "Vertex " << vertex << " is out of range (max: " 
                  << graph.num_vertices - 1 << ")" << std::endl;
        return 1;
    }
    
    // Calculate the rank of the input vertex
    int vertex_rank = vertex % world_size;
    std::cout << "Vertex " << vertex << " belongs to rank " << vertex_rank << std::endl;
    
    auto neighbors_with_ranks = get_neighbors_with_ranks(graph, vertex, world_size);
    
    std::cout << "Vertex " << vertex << " has " << neighbors_with_ranks.size() << " neighbors:" << std::endl;
    std::cout << "Format: neighbor_id (rank)" << std::endl;
    
    // Count neighbors per rank
    std::vector<int> neighbors_per_rank(world_size, 0);
    for (const auto& [neighbor, rank] : neighbors_with_ranks) {
        neighbors_per_rank[rank]++;
    }
    
    // Print summary of neighbors per rank
    std::cout << "\nNeighbors per rank:" << std::endl;
    for (int r = 0; r < world_size; r++) {
        std::cout << "Rank " << r << ": " << neighbors_per_rank[r] << " neighbors" << std::endl;
    }
    
    // Print detailed neighbor list
    std::cout << "\nDetailed neighbor list:" << std::endl;
    for (size_t i = 0; i < neighbors_with_ranks.size(); i++) {
        std::cout << neighbors_with_ranks[i].first << " (rank " << neighbors_with_ranks[i].second << ")";
        if (i < neighbors_with_ranks.size() - 1 && i % 5 != 4) {
            std::cout << ", ";
        }
        if (i % 5 == 4) {
            std::cout << std::endl;
        }
    }
    if (neighbors_with_ranks.size() % 5 != 0) {
        std::cout << std::endl;
    }
    
    return 0;
}
