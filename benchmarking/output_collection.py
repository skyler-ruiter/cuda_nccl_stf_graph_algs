import os
import re
import csv
import glob
from pathlib import Path

def extract_dataset_name(filename):
    """Extract dataset name from the filename."""
    base = os.path.basename(filename)
    # Remove the _cpu.out, _gpu.out, or _gpu_stf.out suffix
    match = re.match(r'(.+?)_(cpu|gpu|gpu_stf)\.out', base)
    if match:
        return match.group(1)
    return base

def parse_vertices_edges(content):
    """Parse and sum vertices and edges from all ranks."""
    total_vertices = 0
    total_edges = 0
    
    # Find all lines with "Loaded X vertices and Y edges"
    pattern = r'Rank \d+: Loaded (\d+) vertices and (\d+) edges'
    matches = re.findall(pattern, content)
    
    # Sum unique vertices (only count once per rank) and all edges
    seen_vertices = set()
    for vertices, edges in matches:
        vertices, edges = int(vertices), int(edges)
        if vertices not in seen_vertices:
            total_vertices += vertices
            seen_vertices.add(vertices)
        total_edges += edges
        
    return total_vertices, total_edges

def extract_bfs_statistics(content):
    """Extract BFS statistics from the output content."""
    stats = {}
    
    # Extract source vertex
    source_match = re.search(r'Source vertex: (\d+)', content)
    if source_match:
        stats['source_vertex'] = int(source_match.group(1))
    
    # Extract visited vertices
    visited_match = re.search(r'Visited vertices: (\d+) \(([0-9.]+)% of graph\)', content)
    if visited_match:
        stats['visited_vertices'] = int(visited_match.group(1))
        stats['visited_percentage'] = float(visited_match.group(2))
    
    # Extract maximum distance
    max_dist_match = re.search(r'Maximum distance from source: (\d+)', content)
    if max_dist_match:
        stats['max_distance'] = int(max_dist_match.group(1))
    
    # Extract average distance
    avg_dist_match = re.search(r'Average distance from source: ([0-9.]+)', content)
    if avg_dist_match:
        stats['avg_distance'] = float(avg_dist_match.group(1))
    
    return stats

def extract_bfs_performance(content):
    """Extract BFS performance metrics from the output content."""
    perf = {}
    
    # Total execution time
    total_time_match = re.search(r'Total execution time: ([0-9.]+) s', content)
    if total_time_match:
        perf['total_time'] = float(total_time_match.group(1))
    
    # Graph loading/init time
    loading_match = re.search(r'Graph loading/init time: ([0-9.]+) s \(([0-9.]+)%\)', content)
    if loading_match:
        perf['loading_time'] = float(loading_match.group(1))
        perf['loading_percentage'] = float(loading_match.group(2))
    
    # BFS traversal time
    traversal_match = re.search(r'BFS traversal time: ([0-9.]+) s \(([0-9.]+)%\)', content)
    if traversal_match:
        perf['traversal_time'] = float(traversal_match.group(1))
        perf['traversal_percentage'] = float(traversal_match.group(2))
    
    # Communication time
    comm_match = re.search(r'Communication time: ([0-9.]+) s \(([0-9.]+)% of BFS time\)', content)
    if comm_match:
        perf['communication_time'] = float(comm_match.group(1))
        perf['communication_percentage'] = float(comm_match.group(2))
    
    # Computation time
    comp_match = re.search(r'Computation time: ([0-9.]+) s \(([0-9.]+)% of BFS time\)', content)
    if comp_match:
        perf['computation_time'] = float(comp_match.group(1))
        perf['computation_percentage'] = float(comp_match.group(2))
    
    # Number of BFS levels
    levels_match = re.search(r'Number of BFS levels: (\d+)', content)
    if levels_match:
        perf['bfs_levels'] = int(levels_match.group(1))
    
    # Average time per level
    avg_time_match = re.search(r'Average time per level: ([0-9.]+) s', content)
    if avg_time_match:
        perf['avg_time_per_level'] = float(avg_time_match.group(1))
    
    return perf

def process_output_file(filepath):
    """Process a single output file and return extracted data."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Determine if it's CPU, GPU, or GPU_STF
    if "_cpu.out" in filepath:
        implementation = "CPU"
    elif "_gpu_stf.out" in filepath:
        implementation = "GPU_STF"
    else:
        implementation = "GPU"
    
    # Extract dataset name
    dataset = extract_dataset_name(filepath)
    
    # Parse vertices and edges
    total_vertices, total_edges = parse_vertices_edges(content)
    
    # Extract BFS statistics and performance
    stats = extract_bfs_statistics(content)
    perf = extract_bfs_performance(content)
    
    # Combine all data
    result = {
        'dataset': dataset,
        'implementation': implementation,
        'total_vertices': total_vertices,
        'total_edges': total_edges,
        **stats,
        **perf
    }
    
    return result

def main():
    # Find all output files
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    output_files = (
        glob.glob(os.path.join(output_dir, '*_cpu.out')) + 
        glob.glob(os.path.join(output_dir, '*_gpu.out')) + 
        glob.glob(os.path.join(output_dir, '*_gpu_stf.out'))
    )
    
    if not output_files:
        print(f"No output files found in {output_dir}")
        return
    
    results = []
    for filepath in output_files:
        try:
            data = process_output_file(filepath)
            results.append(data)
            print(f"Processed: {os.path.basename(filepath)}")
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
    
    if not results:
        print("No data was successfully extracted.")
        return
    
    # Write results to CSV
    csv_path = os.path.join(output_dir, 'bfs_benchmark_results.csv')
    
    # Get all possible headers from all results
    headers = set()
    for result in results:
        headers.update(result.keys())
    headers = sorted(list(headers))
    
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Results written to: {csv_path}")

if __name__ == "__main__":
    main()
