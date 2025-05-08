#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <cstdlib>
#include <unordered_set>
#include <iomanip>

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

// Get degree of a vertex
vertex_t get_degree(const Graph_CSR& graph, vertex_t vertex) {
    if (vertex >= graph.num_vertices) {
        return 0;
    }
    return graph.row_offsets[vertex + 1] - graph.row_offsets[vertex];
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <graph_file>" << std::endl;
        return 1;
    }
    
    std::string graph_file = argv[1];
    std::cout << "Loading graph from: " << graph_file << std::endl;
    
    Graph_CSR graph = load_graph(graph_file);
    std::cout << "Graph loaded with " << graph.num_vertices << " vertices and " 
              << graph.num_edges << " edges" << std::endl;
    
    // Create vector of vertex IDs with their degrees
    std::vector<std::pair<vertex_t, vertex_t>> vertex_degrees;
    vertex_degrees.reserve(graph.num_vertices);
    for (vertex_t v = 0; v < graph.num_vertices; v++) {
        vertex_t degree = get_degree(graph, v);
        vertex_degrees.emplace_back(v, degree);
    }

    // Number of vertices to display
    const int k = 10;
    const int actual_k = std::min(k, static_cast<int>(vertex_degrees.size()));

    // Find k vertices with highest degrees
    auto highest_degrees = vertex_degrees;
    std::partial_sort(highest_degrees.begin(), highest_degrees.begin() + actual_k,
                     highest_degrees.end(),
                     [](const auto& a, const auto& b) { return a.second > b.second; });

    // Find k vertices with lowest degrees
    auto lowest_degrees = vertex_degrees;
    std::partial_sort(lowest_degrees.begin(), lowest_degrees.begin() + actual_k,
                     lowest_degrees.end(),
                     [](const auto& a, const auto& b) { return a.second < b.second; });

    // Print top k vertices with highest degrees
    std::cout << "\n=== Top " << k << " Vertices with Highest Degrees ===" << std::endl;
    std::cout << std::setw(10) << "Vertex" << std::setw(10) << "Degree" << std::endl;
    std::cout << "-------------------------------" << std::endl;
    for (int i = 0; i < actual_k; i++) {
        std::cout << std::setw(10) << highest_degrees[i].first 
                  << std::setw(10) << highest_degrees[i].second << std::endl;
    }

    // Print k vertices with lowest degrees
    std::cout << "\n=== " << k << " Vertices with Lowest Degrees ===" << std::endl;
    std::cout << std::setw(10) << "Vertex" << std::setw(10) << "Degree" << std::endl;
    std::cout << "-------------------------------" << std::endl;
    for (int i = 0; i < actual_k; i++) {
        std::cout << std::setw(10) << lowest_degrees[i].first 
                  << std::setw(10) << lowest_degrees[i].second << std::endl;
    }
    
    return 0;
}
