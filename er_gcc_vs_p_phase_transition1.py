import igraph as ig
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(123)

# Parameters
n = 900  # number of nodes
num_trials = 100  # number of networks per p value

# Create p values from 0 to 0.02
# Use more dense sampling around the critical region
p_values = np.linspace(0.001, 0.02, 40)  # 40 points from 0.001 to 0.02

print(f"Number of nodes: {n}")
print(f"Number of trials per p: {num_trials}")
print(f"p range: {p_values[0]:.4f} to {p_values[-1]:.4f}")
print(f"Number of p values: {len(p_values)}")
print()

# Store results
gcc_results = []
gcc_dict = {}

# Progress tracking
total_networks = len(p_values) * num_trials
networks_processed = 0

for i, p in enumerate(p_values):
    gcc_fractions = []

    # Generate 100 networks for this p value
    for trial in range(num_trials):
        # Generate Erdős–Rényi network
        G = ig.Graph.Erdos_Renyi(n=n, p=p, directed=False)

        # Find connected components
        components = G.connected_components()

        # Get size of giant connected component (largest component)
        if len(components) > 0:
            gcc_size = max(len(comp) for comp in components)
        else:
            gcc_size = 0

        # Normalize by total number of nodes
        gcc_fraction = gcc_size / n
        gcc_fractions.append(gcc_fraction)

        networks_processed += 1

    # Compute average normalized GCC size for this p
    avg_gcc_fraction = np.mean(gcc_fractions)
    std_gcc_fraction = np.std(gcc_fractions)

    # Store results
    gcc_results.append((p, avg_gcc_fraction))
    gcc_dict[p] = {
        'avg_gcc_fraction': avg_gcc_fraction,
        'std_gcc_fraction': std_gcc_fraction,
        'all_fractions': gcc_fractions
    }

    # Progress update every 5 p values
    if (i + 1) % 5 == 0:
        progress = (i + 1) / len(p_values) * 100
        print(f"Progress: {progress:.1f}% - p = {p:.4f}, avg GCC = {avg_gcc_fraction:.4f}")

# Find critical region (where GCC fraction rapidly increases)
p_vals = [result[0] for result in gcc_results]
gcc_fractions = [result[1] for result in gcc_results]

# Find approximate critical point (steepest increase)
differences = np.diff(gcc_fractions)
critical_idx = np.argmax(differences)
critical_p = p_vals[critical_idx]

print(f"Approximate critical p (steepest increase): {critical_p:.4f}")
print(f"GCC fraction at critical p: {gcc_fractions[critical_idx]:.4f}")

# Print sample of results

print(f"{'p':<8} {'Avg GCC':<10} {'Std Dev':<10}")

for i in range(0, len(gcc_results), 5):  # Show every 5th result
    p, avg_gcc = gcc_results[i]
    std_gcc = gcc_dict[p]['std_gcc_fraction']
    print(f"{p:<8.4f} {avg_gcc:<10.4f} {std_gcc:<10.4f}")

# Create visualization
plt.figure(figsize=(12, 8))

# Main plot: Average GCC fraction vs p
plt.subplot(2, 2, 1)
plt.plot(p_vals, gcc_fractions, 'b-', linewidth=2, marker='o', markersize=4)
plt.axvline(x=critical_p, color='r', linestyle='--', alpha=0.7, label=f'Critical p ≈ {critical_p:.4f}')
plt.xlabel('Edge probability (p)')
plt.ylabel('Average normalized GCC size')
plt.title('Giant Connected Component vs p', fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend()

# Plot with error bars
plt.subplot(2, 2, 2)
stds = [gcc_dict[p]['std_gcc_fraction'] for p in p_vals]
plt.errorbar(p_vals, gcc_fractions, yerr=stds, fmt='o-', capsize=3, capthick=1, alpha=0.7)
plt.axvline(x=critical_p, color='r', linestyle='--', alpha=0.7)
plt.xlabel('Edge probability (p)')
plt.ylabel('Average normalized GCC size')
plt.title('GCC Size with Standard Deviation', fontweight='bold')
plt.grid(True, alpha=0.3)

# Phase transition region (zoomed)
plt.subplot(2, 2, 3)
# Focus on critical region
critical_start = max(0, critical_idx - 10)
critical_end = min(len(p_vals), critical_idx + 10)
p_critical = p_vals[critical_start:critical_end]
gcc_critical = gcc_fractions[critical_start:critical_end]

plt.plot(p_critical, gcc_critical, 'ro-', linewidth=2, markersize=6)
plt.axvline(x=critical_p, color='g', linestyle='--', alpha=0.7, label=f'Critical p ≈ {critical_p:.4f}')
plt.xlabel('Edge probability (p)')
plt.ylabel('Average normalized GCC size')
plt.title('Phase Transition Region (Zoomed)', fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend()

# Derivative plot (rate of change)
plt.subplot(2, 2, 4)
derivatives = np.diff(gcc_fractions) / np.diff(p_vals)
p_deriv = p_vals[:-1]
plt.plot(p_deriv, derivatives, 'g-', linewidth=2)
plt.axvline(x=critical_p, color='r', linestyle='--', alpha=0.7)
plt.xlabel('Edge probability (p)')
plt.ylabel('Rate of change (dGCC/dp)')
plt.title('Rate of GCC Growth', fontweight='bold')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
