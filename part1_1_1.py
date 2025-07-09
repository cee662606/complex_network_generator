import igraph as ig
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
n = 900  # number of nodes
p_values = [0.002, 0.006, 0.012, 0.045, 0.1]  # edge probabilities

# Generate Erdős–Rényi networks for each probability value
networks = {}

print("Generating Erdős–Rényi networks...")
print(f"Number of nodes: {n}")
print("-" * 50)

for p in p_values:
    # Generate undirected random network using Erdős–Rényi model
    # In igraph, erdos_renyi_game generates G(n,p) model
    G = ig.Graph.Erdos_Renyi(n=n, p=p, directed=False)

    networks[p] = G

    # Basic network statistics
    num_edges = G.ecount()
    expected_edges = n * (n - 1) * p / 2  # Expected number of edges in G(n,p)

# Create subplots for degree distributions
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

# Store statistics for summary
statistics = {}

for i, p in enumerate(p_values):
    G = networks[p]

    # Get degree sequence
    degrees = G.degree()

    # Calculate statistics
    mean_degree = np.mean(degrees)
    var_degree = np.var(degrees, ddof=1)  # Sample variance
    std_degree = np.std(degrees, ddof=1)

    # Store statistics
    statistics[p] = {
        'mean': mean_degree,
        'variance': var_degree,
        'std': std_degree,
        'min': min(degrees),
        'max': max(degrees)
    }

    # Create degree distribution
    degree_counts = Counter(degrees)
    degree_values = sorted(degree_counts.keys())
    frequencies = [degree_counts[d] for d in degree_values]

    # Plot degree distribution
    ax = axes[i]
    ax.bar(degree_values, frequencies, alpha=0.7, color=f'C{i}')
    ax.set_title(f'Degree Distribution (p = {p})', fontsize=12, fontweight='bold')
    ax.set_xlabel('Degree')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3)

    # Add statistics text to plot
    stats_text = f'Mean: {mean_degree:.3f}\nVar: {var_degree:.3f}\nStd: {std_degree:.3f}'
    ax.text(0.65, 0.85, stats_text, transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            verticalalignment='top', fontsize=10)

# Remove the extra subplot
axes[-1].remove()

plt.tight_layout()
plt.show()

# Print detailed statistics
print("\nDETAILED STATISTICS:")
print("-" * 60)
print(f"{'p':<8} {'Mean':<10} {'Variance':<12} {'Std Dev':<10} {'Min':<6} {'Max':<6}")
print("-" * 60)

for p in p_values:
    stats = statistics[p]
    print(f"{p:<8} {stats['mean']:<10.3f} {stats['variance']:<12.3f} "
          f"{stats['std']:<10.3f} {stats['min']:<6} {stats['max']:<6}")

# Theoretical comparison
print("\nTHEORETICAL vs OBSERVED:")
print("-" * 60)
print(f"{'p':<8} {'Theoretical Mean':<17} {'Observed Mean':<15} {'Theoretical Var':<16} {'Observed Var':<13}")
print("-" * 60)

for p in p_values:
    theoretical_mean = (n - 1) * p
    theoretical_var = (n - 1) * p * (1 - p)  # Binomial variance
    observed_mean = statistics[p]['mean']
    observed_var = statistics[p]['variance']

    print(f"{p:<8} {theoretical_mean:<17.3f} {observed_mean:<15.3f} "
          f"{theoretical_var:<16.3f} {observed_var:<13.3f}")
