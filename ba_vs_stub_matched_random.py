
import igraph as ig
import numpy as np
import matplotlib.pyplot as plt


def analyze_network(G, network_name):
    print(f"\n{network_name}:")
    print(f"  Nodes: {G.vcount()}")
    print(f"  Edges: {G.ecount()}")

    communities = G.community_fastgreedy().as_clustering()
    num_communities = len(communities)
    print(f"  Communities: {num_communities}")

    modularity = G.modularity(communities)
    print(f"  Modularity: {modularity:.4f}")

    assortativity = G.assortativity_degree()
    print(f"  Assortativity: {assortativity:.4f}")

    clustering = G.transitivity_undirected()
    print(f"  Clustering: {clustering:.4f}")

    return {
        'num_communities': num_communities,
        'modularity': modularity,
        'assortativity': assortativity,
        'clustering': clustering
    }


def plot_comparison(metrics_ba, metrics_random):
    fig, ax = plt.subplots(figsize=(10, 6))

    metrics = ['Modularity', 'Assortativity', 'Clustering', 'Communities']
    ba_values = [metrics_ba['modularity'], metrics_ba['assortativity'],
                 metrics_ba['clustering'], metrics_ba['num_communities']]
    random_values = [metrics_random['modularity'], metrics_random['assortativity'],
                     metrics_random['clustering'], metrics_random['num_communities']]

    x = np.arange(len(metrics))
    width = 0.35

    ax.bar(x - width / 2, ba_values, width, label='BA Network', alpha=0.7)
    ax.bar(x + width / 2, random_values, width, label='Random Network', alpha=0.7)

    ax.set_xlabel('Metrics')
    ax.set_ylabel('Values')
    ax.set_title('BA vs Stub-Matched Random Network Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ba_vs_random_metrics.png', dpi=150, bbox_inches='tight')
    plt.show()


def main():
    print("BA vs Stub-Matched Random Network Comparison")
    G_ba = ig.Graph.Barabasi(n=1050, m=1, directed=False)

    degree_sequence = G_ba.degree()

    try:
        G_random = ig.Graph.Degree_Sequence(degree_sequence, method="vl")
    except:
        G_random = ig.Graph.Degree_Sequence(degree_sequence, method="simple")

    metrics_ba = analyze_network(G_ba, "Barabási–Albert Network")
    metrics_random = analyze_network(G_random, "Stub-Matched Random Network")

    print("COMPARISON")

    mod_diff = metrics_ba['modularity'] - metrics_random['modularity']
    assort_diff = metrics_ba['assortativity'] - metrics_random['assortativity']
    clust_diff = metrics_ba['clustering'] - metrics_random['clustering']
    comm_diff = metrics_ba['num_communities'] - metrics_random['num_communities']

    print(f"  Modularity difference: {mod_diff:+.4f}")
    print(f"  Assortativity difference: {assort_diff:+.4f}")
    print(f"  Clustering difference: {clust_diff:+.4f}")
    print(f"  Community difference: {comm_diff:+d}")

    plot_comparison(metrics_ba, metrics_random)


if __name__ == "__main__":
    main()
