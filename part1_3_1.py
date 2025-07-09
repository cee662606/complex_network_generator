import igraph as ig
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import random


def create_modified_preferential_attachment_network():

    # Network parameters
    N = 1050
    m = 1
    alpha = 1
    beta = -1
    a = c = d = 1
    b = 0

    # Initialize graph with one node
    g = ig.Graph()
    g.add_vertices(1)

    # Track node ages (time since addition)
    node_ages = [0]  # First node has age 0

    # Add nodes one by one
    for t in range(1, N):
        # Add new node
        g.add_vertices(1)

        # Update ages of existing nodes
        node_ages = [age + 1 for age in node_ages]
        node_ages.append(0)  # New node has age 0

        # Calculate connection probabilities for existing nodes
        probabilities = []
        for i in range(t):  # For each existing node
            k_i = g.degree(i)  # Degree of node i
            l_i = node_ages[i]  # Age of node i

            # Calculate probability: P[i] ~ (k_i^α + a)(l_i^β + b)
            prob = (k_i ** alpha + a) * (l_i ** beta + b)
            probabilities.append(prob)

        # Normalize probabilities
        total_prob = sum(probabilities)
        if total_prob > 0:
            probabilities = [p / total_prob for p in probabilities]
        else:
            # If all probabilities are 0, use uniform distribution
            probabilities = [1 / t for _ in range(t)]

        # Select m nodes to connect to (m=1 in this case)
        selected_nodes = np.random.choice(range(t), size=m, replace=False, p=probabilities)

        # Add edges from new node to selected nodes
        for node in selected_nodes:
            g.add_edge(t, node)

        # Progress indicator
        if (t + 1) % 100 == 0:
            print(f"  Added {t + 1} nodes...")

    print(f"Network generation complete! Final network has {g.vcount()} nodes and {g.ecount()} edges.")
    return g


def analyze_degree_distribution(g):

    print("\nAnalyzing degree distribution...")

    # Get degree sequence
    degrees = g.degree()
    degree_counts = Counter(degrees)

    # Prepare data for log-log plot
    unique_degrees = sorted(degree_counts.keys())
    counts = [degree_counts[d] for d in unique_degrees]

    # Remove zeros for log-log plot
    nonzero_mask = np.array(counts) > 0
    log_degrees = np.log10(np.array(unique_degrees)[nonzero_mask])
    log_counts = np.log10(np.array(counts)[nonzero_mask])

    # Estimate power-law exponent using linear regression in log-log space
    # P(k) ~ k^(-γ), so log(P(k)) ~ -γ * log(k)
    if len(log_degrees) > 1:
        coeffs = np.polyfit(log_degrees, log_counts, 1)
        power_law_exponent = -coeffs[0]  # Negative because we want the exponent of k^(-γ)
    else:
        power_law_exponent = np.nan

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot degree distribution
    plt.subplot(1, 2, 1)
    plt.hist(degrees, bins=max(30, len(unique_degrees) // 3), alpha=0.7, edgecolor='black')
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.title('Degree Distribution')
    plt.grid(True, alpha=0.3)

    # Plot log-log degree distribution
    plt.subplot(1, 2, 2)
    plt.scatter(log_degrees, log_counts, alpha=0.7, s=50)

    # Add fitted line
    if not np.isnan(power_law_exponent):
        fit_line = coeffs[0] * log_degrees + coeffs[1]
        plt.plot(log_degrees, fit_line, 'r--', alpha=0.8,
                 label=f'Power-law fit: γ ≈ {power_law_exponent:.2f}')
        plt.legend()

    plt.xlabel('log₁₀(Degree)')
    plt.ylabel('log₁₀(Count)')
    plt.title('Log-Log Degree Distribution')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"Degree distribution statistics:")
    print(f"  Mean degree: {np.mean(degrees):.2f}")
    print(f"  Max degree: {max(degrees)}")
    print(f"  Min degree: {min(degrees)}")
    print(f"  Estimated power-law exponent: {power_law_exponent:.2f}")

    return power_law_exponent


def detect_communities(g):
    """
    Detect communities using the Fast Greedy method and return modularity.
    """
    print("\nDetecting communities using Fast Greedy method...")

    # Apply Fast Greedy community detection
    communities = g.community_fastgreedy()

    # Convert to clustering (optimal number of communities)
    optimal_clustering = communities.as_clustering()

    # Calculate modularity
    modularity = optimal_clustering.modularity

    print(f"Community detection results:")
    print(f"  Number of communities: {len(optimal_clustering)}")
    print(f"  Modularity: {modularity:.4f}")

    # Show community size distribution
    community_sizes = [len(community) for community in optimal_clustering]
    print(f"  Community sizes: {sorted(community_sizes, reverse=True)}")
    print(f"  Largest community size: {max(community_sizes)}")
    print(f"  Smallest community size: {min(community_sizes)}")

    return modularity, optimal_clustering


def main():

    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)

    # Step 1: Generate the network
    g = create_modified_preferential_attachment_network()

    # Step 2: Analyze degree distribution
    power_law_exponent = analyze_degree_distribution(g)

    # Step 3: Detect communities
    modularity, communities = detect_communities(g)

    print(f"Network size: {g.vcount()} nodes, {g.ecount()} edges")
    print(f"Average degree: {2 * g.ecount() / g.vcount():.2f}")
    print(f"Power-law exponent estimate: {power_law_exponent:.2f}")
    print(f"Number of communities: {len(communities)}")
    print(f"Modularity: {modularity:.4f}")

    # Optional: Use igraph's samplepage if requested
    # This demonstrates the usage of igraph's sample_page function
    print(f"\nNetwork can also be analyzed using igraph's built-in functions.")
    print(f"For example, clustering coefficient: {g.transitivity_undirected():.4f}")

    return g, power_law_exponent, modularity, communities


# Execute the analysis
if __name__ == "__main__":
    graph, exponent, mod, comms = main()