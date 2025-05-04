#include <iostream>
#include <fstream>
#include <vector>
#include <queue>
#include <unordered_set>
#include <algorithm>
#include <string>
#include <chrono>
#include <iomanip>

using vertex_t = uint32_t;

// Find connected components in a graph file
std::vector<std::vector<vertex_t>> findConnectedComponents(const std::string& filename) {
    // First pass: Collect all vertices
    std::unordered_set<vertex_t> allVertices;
    std::ifstream file1(filename);
    
    if (!file1.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }
    
    vertex_t src, dst;
    while (file1 >> src >> dst) {
        allVertices.insert(src);
        allVertices.insert(dst);
    }
    
    std::cout << "Graph contains " << allVertices.size() << " unique vertices" << std::endl;
    
    // Build adjacency list
    std::vector<std::vector<vertex_t>> adjList(allVertices.size() * 2);  // Over-allocate to avoid rehashing
    std::ifstream file2(filename);
    
    if (!file2.is_open()) {
        std::cerr << "Error reopening file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }
    
    size_t edgeCount = 0;
    while (file2 >> src >> dst) {
        adjList[src].push_back(dst);
        adjList[dst].push_back(src);  // Undirected graph
        edgeCount++;
    }
    
    std::cout << "Graph contains " << edgeCount << " edges" << std::endl;
    
    // Find connected components using BFS
    std::vector<std::vector<vertex_t>> components;
    std::unordered_set<vertex_t> visited;
    
    for (vertex_t start : allVertices) {
        if (visited.find(start) != visited.end()) {
            continue;  // Skip if already visited
        }
        
        std::vector<vertex_t> component;
        std::queue<vertex_t> queue;
        
        queue.push(start);
        visited.insert(start);
        
        while (!queue.empty()) {
            vertex_t current = queue.front();
            queue.pop();
            
            component.push_back(current);
            
            if (current < adjList.size()) {
                for (vertex_t neighbor : adjList[current]) {
                    if (visited.find(neighbor) == visited.end()) {
                        visited.insert(neighbor);
                        queue.push(neighbor);
                    }
                }
            }
        }
        
        components.push_back(component);
    }
    
    return components;
}

int main(int argc, char* argv[]) {
    std::string graphFile;
    
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <graph_file>" << std::endl;
        std::cout << "Using default graph file: ../data/graph500-scale21-ef16_adj.edges" << std::endl;
        graphFile = "../data/graph500-scale21-ef16_adj.edges";
    } else {
        graphFile = argv[1];
        std::cout << "Using graph file: " << graphFile << std::endl;
    }
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // Find connected components
    std::cout << "Analyzing graph and finding connected components..." << std::endl;
    auto components = findConnectedComponents(graphFile);
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    
    // Calculate total vertices
    size_t totalVertices = 0;
    std::vector<size_t> componentSizes;
    
    for (const auto& component : components) {
        componentSizes.push_back(component.size());
        totalVertices += component.size();
    }
    
    // Sort by size (largest first)
    std::sort(componentSizes.begin(), componentSizes.end(), std::greater<size_t>());
    
    // Print results
    std::cout << "Analysis completed in " << duration.count() / 1000.0 << " seconds." << std::endl;
    std::cout << "Total connected components: " << components.size() << std::endl;
    
    std::cout << "\nComponent size statistics:" << std::endl;
    std::cout << "Rank | Size | Percentage of Graph" << std::endl;
    std::cout << "-----------------------------------" << std::endl;
    
    for (size_t i = 0; i < std::min(componentSizes.size(), size_t(20)); ++i) {
        double percentage = 100.0 * componentSizes[i] / totalVertices;
        std::cout << std::setw(4) << i+1 << " | " 
                  << std::setw(10) << componentSizes[i] << " | " 
                  << std::fixed << std::setprecision(6) << percentage << "%" << std::endl;
    }
    
    if (componentSizes.size() > 20) {
        std::cout << "... and " << components.size() - 20 << " more components" << std::endl;
    }
    
    // Additional statistics for component distribution
    if (!componentSizes.empty()) {
        size_t singleVertexComponents = 0;
        size_t smallComponents = 0;  // 2-10 vertices
        size_t mediumComponents = 0;  // 11-100 vertices
        size_t largeComponents = 0;  // >100 vertices
        
        for (const auto& size : componentSizes) {
            if (size == 1) singleVertexComponents++;
            else if (size <= 10) smallComponents++;
            else if (size <= 100) mediumComponents++;
            else largeComponents++;
        }
        
        std::cout << "\nComponent distribution:" << std::endl;
        std::cout << "Single vertex components: " << singleVertexComponents 
                  << " (" << 100.0 * singleVertexComponents / components.size() << "%)" << std::endl;
        std::cout << "Small components (2-10): " << smallComponents 
                  << " (" << 100.0 * smallComponents / components.size() << "%)" << std::endl;
        std::cout << "Medium components (11-100): " << mediumComponents 
                  << " (" << 100.0 * mediumComponents / components.size() << "%)" << std::endl;
        std::cout << "Large components (>100): " << largeComponents 
                  << " (" << 100.0 * largeComponents / components.size() << "%)" << std::endl;
    }
    
    return 0;
}