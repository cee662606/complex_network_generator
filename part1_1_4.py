import igraph as ig
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Set random seed for reproducibility
np.random.seed(123)

# Parameters
c_values = [0.5, 1.0, 1.15, 1.25, 1.35]  # Five values of c
num_trials = 50  # Number of trials per (n, c) combination

# Create n values on logarithmic scale from 100 to 10000
n_values = np.logspace(np.log10(100), np.log10(10000), 30).astype(int)
# Remove duplicates and sort
n_values = sorted(list(set(n_values)))


print(f"n range: {n_values[0]} to {n_values[-1]} ({len(n_values)} values)")
print(f"c values: {c_values}")
print(f"Number of trials per (n, c): {num_trials}")
print(f"Total networks to generate: {len(n_values) * len(c_values) * num_trials}")
print()

# Store results for each c value
results = {c: {'n_values': [], 'avg_gcc_sizes': [], 'std_gcc_sizes': []} for c in c_values}


# Track progress
total_combinations = len(n_values) * len(c_values)
current_combination = 0

for c in c_values:
    print(f"\nProcessing c = {c}...")

    n_list = []
    avg_gcc_list = []
    std_gcc_list = []

    for n in n_values:
        # Calculate edge probability p = c/n
        p = c / n

        gcc_sizes = []

        # Generate multiple networks for this (n, c) combination
        for trial in range(num_trials):
            # Generate Erdős–Rényi network
            G = ig.Graph.Erdos_Renyi(n=n, p=p, directed=False)

            # Find connected components
            components = G.connected_components()

            # Get size of giant connected component
            if len(components) > 0:
                gcc_size = max(len(comp) for comp in components)
            else:
                gcc_size = 0

            gcc_sizes.append(gcc_size)

        # Compute statistics
        avg_gcc_size = np.mean(gcc_sizes)
        std_gcc_size = np.std(gcc_sizes)

        n_list.append(n)
        avg_gcc_list.append(avg_gcc_size)
        std_gcc_list.append(std_gcc_size)

        current_combination += 1

        # Progress update
        if current_combination % 5 == 0:
            progress = current_combination / total_combinations * 100
            print(f"  Progress: {progress:.1f}% - n = {n}, avg GCC = {avg_gcc_size:.1f}")

    # Store results for this c value
    results[c]['n_values'] = n_list
    results[c]['avg_gcc_sizes'] = avg_gcc_list
    results[c]['std_gcc_sizes'] = std_gcc_list

# Create comprehensive visualization
plt.figure(figsize=(16, 12))

# Colors for different c values
colors = ['blue', 'red', 'green', 'orange', 'purple']
markers = ['o', 's', '^', 'D', 'v']

# Main plot: Expected GCC size vs n for all c values
plt.subplot(2, 2, 1)
for i, c in enumerate(c_values):
    n_vals = results[c]['n_values']
    gcc_vals = results[c]['avg_gcc_sizes']

    plt.plot(n_vals, gcc_vals, color=colors[i], marker=markers[i],
             linewidth=2, markersize=6, label=f'c = {c}', alpha=0.8)

plt.xlabel('Number of nodes (n)')
plt.ylabel('Expected GCC size')
plt.title('Expected Giant Component Size vs Network Size', fontweight='bold', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.xscale('log')

# Log-log plot
plt.subplot(2, 2, 2)
for i, c in enumerate(c_values):
    n_vals = results[c]['n_values']
    gcc_vals = results[c]['avg_gcc_sizes']

    # Only plot non-zero values for log scale
    non_zero_mask = np.array(gcc_vals) > 0
    n_nonzero = np.array(n_vals)[non_zero_mask]
    gcc_nonzero = np.array(gcc_vals)[non_zero_mask]

    if len(gcc_nonzero) > 0:
        plt.loglog(n_nonzero, gcc_nonzero, color=colors[i], marker=markers[i],
                   linewidth=2, markersize=6, label=f'c = {c}', alpha=0.8)

plt.xlabel('Number of nodes (n)')
plt.ylabel('Expected GCC size')
plt.title('Log-Log Plot: GCC Size vs Network Size', fontweight='bold', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

# Normalized GCC size (fraction of total nodes)
plt.subplot(2, 2, 3)
for i, c in enumerate(c_values):
    n_vals = results[c]['n_values']
    gcc_vals = results[c]['avg_gcc_sizes']
    gcc_fractions = [gcc / n for gcc, n in zip(gcc_vals, n_vals)]

    plt.plot(n_vals, gcc_fractions, color=colors[i], marker=markers[i],
             linewidth=2, markersize=6, label=f'c = {c}', alpha=0.8)

plt.xlabel('Number of nodes (n)')
plt.ylabel('GCC size / n (fraction)')
plt.title('Normalized Giant Component Size vs Network Size', fontweight='bold', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.xscale('log')
plt.ylim(0, 1.1)

# Growth rate analysis
plt.subplot(2, 2, 4)
for i, c in enumerate(c_values):
    n_vals = np.array(results[c]['n_values'])
    gcc_vals = np.array(results[c]['avg_gcc_sizes'])

    # Compute growth rate (derivative approximation)
    if len(n_vals) > 1:
        growth_rates = np.diff(gcc_vals) / np.diff(n_vals)
        n_mid = (n_vals[:-1] + n_vals[1:]) / 2

        plt.plot(n_mid, growth_rates, color=colors[i], marker=markers[i],
                 linewidth=2, markersize=4, label=f'c = {c}', alpha=0.8)

plt.xlabel('Number of nodes (n)')
plt.ylabel('Growth rate (dGCC/dn)')
plt.title('Growth Rate of Giant Component', fontweight='bold', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.xscale('log')

plt.tight_layout()
plt.show()

# Print summary statistics

print(f"{'c':<6} {'Max GCC':<10} {'At n=':<8} {'Final GCC':<12} {'Final n=':<10} {'Final %':<10}")

for c in c_values:
    n_vals = results[c]['n_values']
    gcc_vals = results[c]['avg_gcc_sizes']

    max_gcc = max(gcc_vals)
    max_gcc_idx = gcc_vals.index(max_gcc)
    max_gcc_n = n_vals[max_gcc_idx]

    final_gcc = gcc_vals[-1]
    final_n = n_vals[-1]
    final_percent = (final_gcc / final_n) * 100

    print(f"{c:<6} {max_gcc:<10.1f} {max_gcc_n:<8} {final_gcc:<12.1f} {final_n:<10} {final_percent:<10.1f}%")

# Theoretical insights

print("• c < 1: Subcritical regime - GCC size O(log n)")
print("• c = 1: Critical point - GCC size O(n^(2/3))")
print("• c > 1: Supercritical regime - GCC size O(n)")
print()
print("Critical values analysis:")
for c in c_values:
    if c < 1:
        regime = "Subcritical"
    elif c == 1:
        regime = "Critical"
    else:
        regime = "Supercritical"
    print(f"  c = {c}: {regime} regime")


# Sample data export
print("\nSample data for c = 1.0:")
c_sample = 1.0
sample_n = results[c_sample]['n_values'][:5]
sample_gcc = results[c_sample]['avg_gcc_sizes'][:5]
for n, gcc in zip(sample_n, sample_gcc):
    print(f"  n = {n:4d}, GCC = {gcc:6.1f} ({gcc / n * 100:5.1f}%)")