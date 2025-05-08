import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Set output directory for plots
PLOT_DIR = os.path.join(os.path.dirname(__file__), 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)

# Read the benchmark data
csv_path = os.path.join(os.path.dirname(__file__), 'output', 'bfs_benchmark_results.csv')
df = pd.read_csv(csv_path)

# Extract unique datasets and implementations
datasets = df['dataset'].unique()
implementations = df['implementation'].unique()

# --------------- Scaling: Execution Time vs Graph Size (Edges) ---------------
plt.figure(figsize=(12, 6))

# Extract graph size (using total_edges as the measure)
df['graph_size'] = df['total_edges']

# Create a marker and color mapping for each implementation
impl_markers = {
    'CPU': 'o',    # circle
    'GPU': '^',    # triangle up
    'GPU_STF': 's' # square
}

impl_colors = {
    'CPU': 'blue',
    'GPU': 'red', 
    'GPU_STF': 'green'
}

# Plot for each implementation with consistent colors
for impl in implementations:
    impl_df = df[df['implementation'] == impl].sort_values('graph_size')
    plt.plot(impl_df['graph_size'], impl_df['traversal_time'], 
             marker=impl_markers.get(impl, 'o'),
             color=impl_colors.get(impl, 'blue'),
             label=impl)

plt.xlabel('Graph Size (Number of Edges)')
plt.ylabel('BFS Traversal Time (seconds)')
plt.title('Scaling: BFS Traversal Time vs Number of Edges')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xscale('log')  # Log scale for better visualization of range
plt.yscale('log')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'scaling_edges.png'))
plt.close()

# --------------- Scaling: Execution Time vs Graph Size (Vertices) ---------------
plt.figure(figsize=(12, 6))

# Plot for each implementation
for impl in implementations:
    impl_df = df[df['implementation'] == impl].sort_values('total_vertices')
    plt.plot(impl_df['total_vertices'], impl_df['traversal_time'], 
             marker=impl_markers.get(impl, 'o'),
             color=impl_colors.get(impl, 'blue'),
             label=impl)

plt.xlabel('Graph Size (Number of Vertices)')
plt.ylabel('BFS Traversal Time (seconds)')
plt.title('Scaling: BFS Traversal Time vs Number of Vertices')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xscale('log')  # Log scale for better visualization of range
plt.yscale('log')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'scaling_vertices.png'))
plt.close()

# --------------- Average Time per Level vs Graph Size (Edges) ---------------
plt.figure(figsize=(12, 6))

# Plot for each implementation
for impl in implementations:
    impl_df = df[df['implementation'] == impl].sort_values('total_edges')
    plt.plot(impl_df['total_edges'], impl_df['avg_time_per_level'], 
             marker=impl_markers.get(impl, 'o'),
             color=impl_colors.get(impl, 'blue'),
             label=impl)

plt.xlabel('Graph Size (Number of Edges)')
plt.ylabel('Average Time per Level (seconds)')
plt.title('Average Time per BFS Level vs Number of Edges')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xscale('log')  # Log scale for better visualization of range
plt.yscale('log')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'avg_time_per_level_edges.png'))
plt.close()

# --------------- Average Time per Level vs Graph Size (Vertices) ---------------
plt.figure(figsize=(12, 6))

# Plot for each implementation
for impl in implementations:
    impl_df = df[df['implementation'] == impl].sort_values('total_vertices')
    plt.plot(impl_df['total_vertices'], impl_df['avg_time_per_level'], 
             marker=impl_markers.get(impl, 'o'),
             color=impl_colors.get(impl, 'blue'),
             label=impl)

plt.xlabel('Graph Size (Number of Vertices)')
plt.ylabel('Average Time per Level (seconds)')
plt.title('Average Time per BFS Level vs Number of Vertices')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xscale('log')  # Log scale for better visualization of range
plt.yscale('log')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'avg_time_per_level_vertices.png'))
plt.close()

# --------------- Computation vs Communication Times ---------------
fig, ax = plt.subplots(figsize=(12, 9))

# Store point coordinates for better annotation placement
point_coords = []

# Function to simplify dataset names for better readability
def simplify_dataset_name(name):
    name = name.strip()
    if 'g500-scale' in name:
        # Extract scale number and format as 'S##'
        scale_num = name.split('g500-scale')[1]
        return f'S{scale_num}'
    return name  # Return original name for non-g500 datasets like web-uk

# Create a scatter plot of computation vs communication times
for impl in implementations:
    impl_df = df[df['implementation'] == impl]
    
    # Plot each dataset as a point
    points = ax.scatter(
        impl_df['computation_time'], 
        impl_df['communication_time'],
        s=100,  # marker size
        label=impl,
        alpha=0.7,
        marker=impl_markers.get(impl, 'o'),
        color=impl_colors.get(impl, 'blue')
    )
    
    # Store points and dataset info for later annotation with simplified names
    for _, row in impl_df.iterrows():
        point_coords.append({
            'x': row['computation_time'],
            'y': row['communication_time'],
            'dataset': simplify_dataset_name(row['dataset'])
        })

# Find max value for equal axis scaling and diagonal line
max_val = max(df['computation_time'].max(), df['communication_time'].max()) * 1.1
ax.plot([0, max_val], [0, max_val], 'r--', label='Communication = Computation')

# Set up main plot
ax.set_xlabel('Computation Time (seconds)')
ax.set_ylabel('Communication Time (seconds)')
ax.set_title('Computation vs Communication Times by Implementation')
ax.grid(True, alpha=0.3)
ax.legend()

# Fix the spacing issue between y-axis and x=0
ax.axis('scaled')
ax.set_xlim(0, max_val)
ax.set_ylim(0, max_val)

# Add an inset axes for the zoomed region
# First, determine the cluster region bounds
small_points = [p for p in point_coords if p['x'] < 0.5 and p['y'] < 0.5]
if small_points:
    x_min = max(0, min(p['x'] for p in small_points) - 0.05)
    x_max = max(p['x'] for p in small_points) + 0.05
    y_min = max(0, min(p['y'] for p in small_points) - 0.05)
    y_max = max(p['y'] for p in small_points) + 0.05
    
    # Create inset axes in the top left corner instead of bottom right
    axins = ax.inset_axes([0.07, 0.55, 0.4, 0.4])
    
    # Plot the same data in the inset
    for impl in implementations:
        impl_df = df[df['implementation'] == impl]
        axins.scatter(
            impl_df['computation_time'], 
            impl_df['communication_time'],
            s=80,
            label=impl,
            alpha=0.7,
            marker=impl_markers.get(impl, 'o'),
            color=impl_colors.get(impl, 'blue')
        )
    
    # Add annotations in the inset with better spacing
    for i, point in enumerate(small_points):
        # Calculate text offsets to avoid overlap
        offset_x = 0.01 * (i % 3)  # Vary x offset based on point index
        offset_y = 0.01 * (i % 5)  # Vary y offset based on point index
        
        axins.annotate(
            point['dataset'],
            (point['x'], point['y']),
            xytext=(5 + offset_x*30, 5 + offset_y*30),
            textcoords='offset points',
            fontsize=8,
            arrowprops=dict(arrowstyle='->', lw=0.5)
        )
    
    # Set the limits for the inset axes
    axins.set_xlim(x_min, x_max)
    axins.set_ylim(y_min, y_max)
    axins.grid(True, alpha=0.3)
    
    # Draw lines connecting the zoomed region with the inset
    ax.indicate_inset_zoom(axins, edgecolor="black")
    
    # Add diagonal line in the inset
    axins.plot([x_min, x_max], [x_min, x_max], 'r--')

# Add annotations to the main plot for non-clustered points
for point in point_coords:
    # Only annotate points outside the zoomed region
    if small_points and (point['x'] > x_max or point['y'] > y_max):
        ax.annotate(
            point['dataset'],
            (point['x'], point['y']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8
        )

plt.tight_layout(pad=1.1)
plt.savefig(os.path.join(PLOT_DIR, 'computation_vs_communication.png'))
plt.close()

# --------------- Communication to Computation Ratio ---------------
# Create a dynamic vertical layout with as many rows as implementations
fig, axes = plt.subplots(len(implementations), 1, figsize=(10, 4 * len(implementations)))

# For each implementation
for i, impl in enumerate(implementations):
    ax = axes[i]
    
    # Get data for this implementation
    impl_df = df[df['implementation'] == impl].copy()
    
    # Calculate ratios
    impl_df['ratio'] = impl_df.apply(
        lambda row: row['communication_time'] / row['computation_time'] 
        if row['computation_time'] > 0 else 0, 
        axis=1
    )
    
    # Sort by ratio
    impl_df = impl_df.sort_values('ratio')
    
    # Create a colormap based on the number of vertices
    norm = plt.Normalize(df['total_vertices'].min(), df['total_vertices'].max())
    colors = plt.cm.viridis(norm(impl_df['total_vertices']))
    
    # Plot bars with coloring by vertices
    x_pos = np.arange(len(impl_df))
    bars = ax.bar(x_pos, impl_df['ratio'], color=colors)
    
    # Add labels and styling - ensure clear dataset names
    ax.set_xticks(x_pos)
    ax.set_xticklabels(impl_df['dataset'].values, rotation=45, ha='right', fontsize=9)
    ax.set_title(f'{impl} Implementation', pad=10)
    ax.axhline(y=1, color='r', linestyle='--', label='Equal ratio')
    ax.grid(True, alpha=0.3)
    
    # Add y-axis label to each subplot
    ax.set_ylabel('Communication/Computation Ratio')

# Use a title for the whole figure
fig.suptitle('Communication to Computation Ratio (Sorted by Ratio)', fontsize=14, y=0.99)

# Single, unified adjustment for layout with proper spacing - more space below title
plt.subplots_adjust(left=0.15, right=0.9, bottom=0.15, top=0.92, hspace=0.5)

# Add a common colorbar
norm = plt.Normalize(df['total_vertices'].min(), df['total_vertices'].max())
sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
sm.set_array([])

# Position the colorbar properly
cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02])  # [left, bottom, width, height]
cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
cbar.set_label('Number of Vertices')

plt.savefig(os.path.join(PLOT_DIR, 'comm_comp_ratio.png'))
plt.close()

print(f"Plots saved to {PLOT_DIR}")

# --------------- Distances from Source Comparison (Summary Graph) ---------------
plt.figure(figsize=(14, 7))

# First, standardize dataset names to ensure consistency
df['std_dataset'] = df['dataset'].apply(lambda x: x.strip().lower())
std_datasets = sorted(df['std_dataset'].unique())

# For visual clarity, create a grouped bar chart
n_datasets = len(std_datasets)
n_implementations = len(implementations)
bar_width = 0.7 / n_implementations
x = np.arange(n_datasets)

# Create a mapping for prettier dataset display names
display_names = []
for std_dataset in std_datasets:
    # Get any instance of the dataset to extract the original name
    orig_name = df[df['std_dataset'] == std_dataset]['dataset'].iloc[0]
    display_names.append(simplify_dataset_name(orig_name))

# Plot one bar per implementation for each dataset
for i, impl in enumerate(implementations):
    avg_distances = []
    for std_dataset in std_datasets:
        # Find this implementation for this standardized dataset
        impl_df = df[(df['implementation'] == impl) & (df['std_dataset'] == std_dataset)]
        
        # Use avg_distance if available, otherwise fallback to 0
        if 'avg_distance' in df.columns and len(impl_df) > 0 and not pd.isna(impl_df['avg_distance'].iloc[0]):
            avg_distances.append(impl_df['avg_distance'].iloc[0])
        else:
            # Check if we need to use max_distance instead
            if 'max_distance' in df.columns and len(impl_df) > 0 and not pd.isna(impl_df['max_distance'].iloc[0]):
                avg_distances.append(impl_df['max_distance'].iloc[0])
            else:
                avg_distances.append(0)  # Placeholder if no distance data
    
    # Position bars side by side within each dataset group
    plt.bar(x + (i - n_implementations/2 + 0.5) * bar_width, 
            avg_distances, 
            width=bar_width, 
            label=impl,
            color=impl_colors.get(impl, 'blue'),
            alpha=0.8)

# Add labels, title and customize the plot
plt.xlabel('Dataset', fontsize=12)
plt.ylabel('Average Distance from Source', fontsize=12)
plt.title('Average BFS Distances from Source Across Implementations', fontsize=14)
plt.xticks(x, display_names, rotation=45, ha='right')
plt.legend(title="Implementation")
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'avg_distances_summary.png'))
plt.close()
