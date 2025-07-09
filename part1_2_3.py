#!/usr/bin/env python3
"""
Task 2(f): Node Age vs Final Degree — log-log fit
Expected: power-law decay ~ 1/sqrt(i), slope ≈ -0.5
"""

import igraph as ig
import numpy as np
import matplotlib.pyplot as plt

def build_ba_with_join_order(n, m):
    """Manually build BA graph and track node join order."""
    G = ig.Graph()
    G.add_vertices(m + 1)
    for i in range(m + 1):
        for j in range(i + 1, m + 1):
            G.add_edge(i, j)
    join_order = list(range(m + 1))  # initial m+1 nodes

    for t in range(m + 1, n):
        degrees = G.degree()
        probs = np.array(degrees) / np.sum(degrees)
        targets = np.random.choice(G.vcount(), size=m, replace=False, p=probs)
        G.add_vertex()
        new_id = G.vcount() - 1
        for target in targets:
            G.add_edge(new_id, target)
        join_order.append(t)  # t 就是第 new_id 个节点的加入时间

    return G, join_order

def plot_loglog_fit(join_times, degrees):
    """Plot log-log scatter and fit."""
    join_times = np.array(join_times)
    degrees = np.array(degrees)

    # Remove zero degrees to avoid log(0)
    mask = (degrees > 0) & (np.array(join_times) > 0)
    join_times = np.array(join_times)[mask]
    degrees = np.array(degrees)[mask]

    log_i = np.log10(join_times)
    log_deg = np.log10(degrees)

    # Linear fit in log-log space
    z = np.polyfit(log_i, log_deg, 1)
    slope = z[0]
    p = np.poly1d(z)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(log_i, log_deg, s=10, alpha=0.5, color='blue')
    plt.plot(log_i, p(log_i), 'r--', label=f"Slope = {slope:.4f}")
    plt.xlabel("log(Node Join Time)")
    plt.ylabel("log(Final Degree)")
    plt.title("Log-Log: Node Age vs Final Degree in BA Network")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(f"\nLog-log linear regression result:")
    print(f"  Slope ≈ {slope:.4f} (expected ≈ -0.5)")
    return slope

def main():
    print("Analyzing BA Network: Node Age vs Final Degree (log-log)")
    G, join_times = build_ba_with_join_order(n=1050, m=1)
    degrees = G.degree()
    slope = plot_loglog_fit(join_times, degrees)

    if slope < -0.4:
        print("  → Strong negative slope confirms early nodes gain more degree (preferential attachment).")
    elif slope < -0.2:
        print("  → Moderate trend: age-degree correlation is present.")
    else:
        print("  → Weak slope: unexpected, recheck data or generation.")

if __name__ == "__main__":
    main()

