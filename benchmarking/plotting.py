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

# Plot for each implementation
for impl in implementations:
    impl_df = df[df['implementation'] == impl].sort_values('graph_size')
    plt.plot(impl_df['graph_size'], impl_df['traversal_time'], marker='o', label=impl)

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
    plt.plot(impl_df['total_vertices'], impl_df['traversal_time'], marker='o', label=impl)

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

# --------------- Computation vs Communication Times ---------------
plt.figure(figsize=(10, 8))

# Create a scatter plot of computation vs communication times
for impl in implementations:
    impl_df = df[df['implementation'] == impl]
    
    # Plot each dataset as a point
    plt.scatter(
        impl_df['computation_time'], 
        impl_df['communication_time'],
        s=100,  # marker size
        label=impl,
        alpha=0.7,
        marker='o' if impl == 'CPU' else '^'  # different markers for CPU vs GPU
    )
    
    # Add dataset labels to each point
    for _, row in impl_df.iterrows():
        dataset_name = row['dataset'].strip()  # Strip any whitespace
        plt.annotate(
            dataset_name,
            (row['computation_time'], row['communication_time']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8
        )

# Find max value for equal axis scaling and diagonal line
max_val = max(df['computation_time'].max(), df['communication_time'].max()) * 1.1
plt.plot([0, max_val], [0, max_val], 'r--', label='Communication = Computation')

plt.xlabel('Computation Time (seconds)')
plt.ylabel('Communication Time (seconds)')
plt.title('Computation vs Communication Times by Implementation')
plt.grid(True, alpha=0.3)
plt.legend()

# Fix the spacing issue between y-axis and x=0
plt.axis('scaled')  # Use 'scaled' instead of 'equal'
plt.xlim(0, max_val)
plt.ylim(0, max_val)

# Add a tight_layout with appropriate padding
plt.tight_layout(pad=1.1)
plt.savefig(os.path.join(PLOT_DIR, 'computation_vs_communication.png'))
plt.close()

# --------------- Communication to Computation Ratio ---------------
# Change to a vertical layout with 2 rows, 1 column and make figure taller than wide
fig, axes = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

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
    
    # Add labels and styling - fix the FixedFormatter warning by properly setting ticks
    ax.set_xticks(x_pos)
    ax.set_xticklabels(impl_df['dataset'].values, rotation=45, ha='right')
    ax.set_title(f'{impl} Implementation')
    ax.axhline(y=1, color='r', linestyle='--', label='Equal ratio')
    ax.grid(True, alpha=0.3)
    
    # Add y-axis label to each subplot
    ax.set_ylabel('Communication/Computation Ratio')

# Use a subtitle instead of y-axis label
fig.suptitle('Communication to Computation Ratio (Sorted by Ratio)', fontsize=14)

# Create space for the colorbar by adjusting the figure
plt.subplots_adjust(left=0.15, right=0.85, bottom=0.22, top=0.9, hspace=0.1)

# Add x-axis label to bottom subplot only, with proper positioning
fig.text(0.5, 0.1, 'Dataset', ha='center', fontsize=12)

# Add a common colorbar
norm = plt.Normalize(df['total_vertices'].min(), df['total_vertices'].max())
sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
sm.set_array([])

# Position the colorbar properly
cbar_ax = fig.add_axes([0.15, 0.04, 0.7, 0.02])  # [left, bottom, width, height]
cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
cbar.set_label('Number of Vertices')

plt.savefig(os.path.join(PLOT_DIR, 'comm_comp_ratio.png'))
plt.close()

print(f"Plots saved to {PLOT_DIR}")
