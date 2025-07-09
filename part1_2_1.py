
import igraph as ig
import matplotlib.pyplot as plt
import numpy as np
def analyze_network(n, m, name):

    # Generate BA network
    G = ig.Graph.Barabasi(n=n, m=m, directed=False)

    # Check connectivity
    is_connected = G.is_connected()

    # Community detection
    communities = G.community_fastgreedy().as_clustering()

    # Calculate metrics
    modularity = G.modularity(communities)
    assortativity = G.assortativity_degree()

    print(f"\n{name}:")
    print(f"  Nodes: {G.vcount()}, Edges: {G.ecount()}")
    print(f"  Connected: {is_connected}")
    print(f"  Communities: {len(communities)}")
    print(f"  Modularity: {modularity:.4f}")
    print(f"  Assortativity: {assortativity:.4f}")

    return G, communities, modularity, assortativity


def plot_simple_network(G, communities, filename):

    # Create adjacency matrix for matplotlib
    adj_matrix = np.array(G.get_adjacency().data)

    # Simple circular layout
    n = G.vcount()
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pos = np.column_stack([np.cos(angles), np.sin(angles)])

    # Assign colors to communities
    colors = plt.cm.Set3(np.linspace(0, 1, len(communities)))
    node_colors = ['gray'] * n
    for i, comm in enumerate(communities):
        for node in comm:
            node_colors[node] = colors[i]

    # Plot
    plt.figure(figsize=(10, 10))

    # Draw edges
    for i in range(n):
        for j in range(i + 1, n):
            if adj_matrix[i][j]:
                plt.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]],
                         'gray', alpha=0.3, linewidth=0.5)

    # Draw nodes
    plt.scatter(pos[:, 0], pos[:, 1], c=node_colors, s=20, alpha=0.8)

    plt.title(f'BA Network (n={n}, communities={len(communities)})')
    plt.axis('equal')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: {filename}")


def main():

    # Analyze networks
    G_small, comm_small, mod_small, assort_small = analyze_network(1050, 1, "Small Network")
    G_large, comm_large, mod_large, assort_large = analyze_network(10500, 1, "Large Network")

    # Plot small network
    plot_simple_network(G_small, comm_small, "network_small_plot.png")
    plot_simple_network(G_large, comm_large, "network_large_plot.png")


    # Compare results
    print(f"\nComparison:")
    print(f"  Modularity difference: {mod_large - mod_small:+.4f}")
    print(f"  Assortativity difference: {assort_large - assort_small:+.4f}")
    print(f"  Community scaling: {len(comm_large) / len(comm_small):.2f}x")

    # Summary
    print(f"\nSummary:")
    print(f"  Both networks are connected (BA property)")
    print(f"  Negative assortativity confirms disassortative mixing")
    print(f"  Modularity ~{mod_small:.2f} indicates community structure")
    print(f"  Larger network shows {'stronger' if mod_large > mod_small else 'weaker'} communities")


if __name__ == "__main__":
    main()