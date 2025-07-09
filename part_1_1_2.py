import igraph as ig
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(90)

# Parameters
n = 900  # number of nodes
p_values = [0.002, 0.006, 0.012, 0.045, 0.1]  # edge probabilities

# Generate Erdős–Rényi networks for each probability value
networks = {}

print(f"Number of nodes: {n}")


for p in p_values:
    G = ig.Graph.Erdos_Renyi(n=n, p=p, directed=False)
    networks[p] = G

    num_edges = G.ecount()
    expected_edges = n * (n - 1) * p / 2

    print(f"p = {p}:")
    print(f"  Actual edges: {num_edges}")
    print(f"  Expected edges: {expected_edges:.1f}")
    print(f"  Average degree: {2 * num_edges / n:.3f}")
    print()


connectivity_results = {}
num_trials = 100  # Number of networks to generate for empirical probability

for p in p_values:
    # Check if original network is connected
    original_connected = networks[p].is_connected()

    # Generate multiple networks to estimate empirical probability
    connected_count = 0
    for trial in range(num_trials):
        G_trial = ig.Graph.Erdos_Renyi(n=n, p=p, directed=False)
        if G_trial.is_connected():
            connected_count += 1

    empirical_prob = connected_count / num_trials

    connectivity_results[p] = {
        'original_connected': original_connected,
        'empirical_probability': empirical_prob,
        'connected_trials': connected_count,
        'total_trials': num_trials
    }

    print(f"p = {p}:")
    print(f"  Original network connected: {original_connected}")
    print(f"  Connected networks: {connected_count}/{num_trials}")
    print(f"  Empirical probability: {empirical_prob:.4f}")
    print()

gcc_results = {}

for p in p_values:
    G = networks[p]

    # Get connected components
    components = G.connected_components()

    # Find giant connected component (largest component)
    gcc_indices = max(components, key=len)
    gcc = G.induced_subgraph(gcc_indices)

    # Compute diameter of GCC
    gcc_diameter = gcc.diameter()

    gcc_results[p] = {
        'num_components': len(components),
        'gcc_size': len(gcc_indices),
        'gcc_diameter': gcc_diameter,
        'gcc_edges': gcc.ecount(),
        'component_sizes': sorted([len(comp) for comp in components], reverse=True)
    }

    print(f"p = {p}:")
    print(f"  Number of components: {len(components)}")
    print(f"  GCC size: {len(gcc_indices)} nodes ({len(gcc_indices) / n * 100:.1f}% of total)")
    print(f"  GCC edges: {gcc.ecount()}")
    print(f"  GCC diameter: {gcc_diameter}")
    print(f"  Component sizes: {gcc_results[p]['component_sizes'][:10]}")  # Show top 10
    print()

# Create comprehensive visualization
fig = plt.figure(figsize=(20, 16))

# Plot 1: Connectivity probability vs p
plt.subplot(3, 3, 1)
p_vals = list(connectivity_results.keys())
emp_probs = [connectivity_results[p]['empirical_probability'] for p in p_vals]
plt.plot(p_vals, emp_probs, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Edge probability (p)')
plt.ylabel('Empirical connectivity probability')
plt.title('Connectivity Probability vs p', fontweight='bold')
plt.grid(True, alpha=0.3)
plt.ylim(-0.05, 1.05)

# Plot 2: Number of components vs p
plt.subplot(3, 3, 2)
num_components = [gcc_results[p]['num_components'] for p in p_vals]
plt.semilogy(p_vals, num_components, 'ro-', linewidth=2, markersize=8)
plt.xlabel('Edge probability (p)')
plt.ylabel('Number of components (log scale)')
plt.title('Number of Components vs p', fontweight='bold')
plt.grid(True, alpha=0.3)

# Plot 3: GCC size vs p
plt.subplot(3, 3, 3)
gcc_sizes = [gcc_results[p]['gcc_size'] for p in p_vals]
gcc_fractions = [size / n for size in gcc_sizes]
plt.plot(p_vals, gcc_fractions, 'go-', linewidth=2, markersize=8)
plt.xlabel('Edge probability (p)')
plt.ylabel('GCC size (fraction of n)')
plt.title('Giant Component Size vs p', fontweight='bold')
plt.grid(True, alpha=0.3)
plt.ylim(0, 1.05)

# Plot 4: GCC diameter vs p
plt.subplot(3, 3, 4)
gcc_diameters = [gcc_results[p]['gcc_diameter'] for p in p_vals]
plt.plot(p_vals, gcc_diameters, 'mo-', linewidth=2, markersize=8)
plt.xlabel('Edge probability (p)')
plt.ylabel('GCC diameter')
plt.title('GCC Diameter vs p', fontweight='bold')
plt.grid(True, alpha=0.3)

# Plots 5-9: Component size distributions for each p
for i, p in enumerate(p_vals):
    plt.subplot(3, 3, 5 + i)
    comp_sizes = gcc_results[p]['component_sizes']

    # Show top 20 components or all if fewer
    top_components = comp_sizes[:min(20, len(comp_sizes))]
    x_pos = range(1, len(top_components) + 1)

    plt.bar(x_pos, top_components, alpha=0.7, color=f'C{i}')
    plt.xlabel('Component rank')
    plt.ylabel('Component size')
    plt.title(f'Component Sizes (p = {p})', fontweight='bold')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Summary table
print(f"{'p':<8} {'Connected':<10} {'Emp.Prob':<10} {'#Comp':<8} {'GCC Size':<10} {'GCC %':<8} {'Diameter':<10}")


for p in p_values:
    conn_res = connectivity_results[p]
    gcc_res = gcc_results[p]

    connected = "Yes" if conn_res['original_connected'] else "No"
    emp_prob = conn_res['empirical_probability']
    num_comp = gcc_res['num_components']
    gcc_size = gcc_res['gcc_size']
    gcc_percent = gcc_size / n * 100
    diameter = gcc_res['gcc_diameter']

    print(f"{p:<8} {connected:<10} {emp_prob:<10.4f} {num_comp:<8} {gcc_size:<10} "
          f"{gcc_percent:<8.1f} {diameter:<10}")
