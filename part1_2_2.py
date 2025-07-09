#!/usr/bin/env python3
"""
Network Science Assignment: Task 2(d) & 2(e)
Degree distribution and neighbor-degree distribution analysis
"""

import igraph as ig
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import random


def get_degree_distribution(degrees):
    """Get degree distribution as (degree, frequency) pairs."""
    unique_degrees, counts = np.unique(degrees, return_counts=True)
    return unique_degrees, counts


def plot_log_distribution(degrees, frequencies, ax, title, color='blue'):
    """Plot degree distribution on log-log scale and return slope."""
    # Filter out zeros for log scale
    mask = (degrees > 0) & (frequencies > 0)
    log_degrees = np.log10(degrees[mask])
    log_frequencies = np.log10(frequencies[mask])

    # Linear regression on log-log data
    slope, intercept, r_value, p_value, std_err = linregress(log_degrees, log_frequencies)

    # Plot on given axis
    ax.loglog(degrees[mask], frequencies[mask], 'o', color=color, alpha=0.7, markersize=4)

    # Add fitted line
    x_fit = degrees[mask]
    y_fit = 10 ** (intercept) * x_fit ** slope
    ax.loglog(x_fit, y_fit, 'r--', linewidth=2,
              label=f'Slope = {slope:.3f}, R² = {r_value ** 2:.3f}')

    ax.set_xlabel('log(degree)')
    ax.set_ylabel('log(frequency)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return slope


def sample_neighbor_degrees(G, num_samples=10000):
    """Sample neighbor degrees through random node-neighbor selection."""
    neighbor_degrees = []

    for _ in range(num_samples):
        # Randomly pick a node
        node_i = random.randint(0, G.vcount() - 1)

        # Get its neighbors
        neighbors = G.neighbors(node_i)

        # If node has neighbors, randomly select one
        if neighbors:
            neighbor_j = random.choice(neighbors)
            neighbor_degrees.append(G.degree(neighbor_j))

    return neighbor_degrees


def analyze_network(n, m, network_name, axes):
    """Analyze both degree and neighbor-degree distributions."""
    print(f"\n{network_name} (n={n}, m={m}):")

    # Generate BA network
    G = ig.Graph.Barabasi(n=n, m=m, directed=False)

    # 1. Node degree distribution
    degrees = G.degree()
    deg_unique, deg_counts = get_degree_distribution(degrees)

    slope_deg = plot_log_distribution(
        deg_unique, deg_counts, axes[0],
        f'{network_name}: Node Degree Distribution',
        'blue'
    )

    # 2. Neighbor degree distribution
    neighbor_degrees = sample_neighbor_degrees(G, 10000)
    neighbor_unique, neighbor_counts = get_degree_distribution(neighbor_degrees)

    slope_neighbor = plot_log_distribution(
        neighbor_unique, neighbor_counts, axes[1],
        f'{network_name}: Neighbor Degree Distribution',
        'red'
    )

    print(f"  Node degree slope: {slope_deg:.3f}")
    print(f"  Neighbor degree slope: {slope_neighbor:.3f}")
    print(f"  Slope difference: {slope_neighbor - slope_deg:+.3f}")

    return slope_deg, slope_neighbor


def main():

    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Create subplot layout: 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('BA Network Degree Distribution Analysis', fontsize=16)

    # Analyze small network (top row)
    slope_small_deg, slope_small_neighbor = analyze_network(1050, 1, "Small Network", axes[0])

    # Analyze large network (bottom row)
    slope_large_deg, slope_large_neighbor = analyze_network(10500, 1, "Large Network", axes[1])

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('all_degree_distributions.png', dpi=150, bbox_inches='tight')
    plt.show()  # This will display in PyCharm

    # Summary comparison
    print(f"\nSummary:")
    print(f"  Small network - Node: {slope_small_deg:.3f}, Neighbor: {slope_small_neighbor:.3f}")
    print(f"  Large network - Node: {slope_large_deg:.3f}, Neighbor: {slope_large_neighbor:.3f}")

    print(f"\nKey Observations:")
    print(f"  • BA networks show power-law degree distribution (negative slopes)")
    print(f"  • Neighbor distributions are biased toward higher degrees")
    print(f"  • Neighbor slopes are less negative (flatter) due to preferential sampling")
    print(f"  • This reflects the 'friendship paradox' - your neighbors have more connections")

    print(f"\nFile generated: all_degree_distributions.png")


if __name__ == "__main__":
    main()