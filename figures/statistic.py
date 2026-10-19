import matplotlib.pyplot as plt
import numpy as np

solvers = [
    'CVX-Mosek', 'CVX-Gurobi', 'Mosek', 'Gurobi',
    'SGD Primal', 'GD Primal', 'FGD Primal', 'ProxGD Primal',
    'FProxGD Primal', 'ALM Dual', 'ADMM Dual', 'ADMM Primal',
    'PPA Dual', 'BCD Primal'
]

optvals = np.array([
    0.670575, 0.670575, 0.670575, 0.670575,
    0.670575, 0.671027, 0.670672, 0.670575,
    0.670575, 0.670576, 0.670575, 0.670575,
    0.670575, 0.670575
])

sparsities = np.array([
    0.103, 0.106, 0.103, 0.104,
    0.100, 0.100, 0.100, 0.103,
    0.103, 0.100, 0.100, 0.100,
    0.100, 0.103
])

cpus = np.array([
    0.46, 0.94, 0.18, 0.34,
    3.29, 3.40, 1.62, 2.17,
    2.18, 0.52, 1.29, 1.75,
    5.40, 24.71
])

f_star = optvals[0]
rel_obj_err = np.maximum(np.abs(optvals - f_star) / f_star, 1e-8)

y_pos = np.arange(len(solvers))

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 6), sharey=True)

# 1. CPU Time (Log scale)
ax1.barh(y_pos, cpus, color='#2b5c8f', alpha=0.85, edgecolor='black', linewidth=0.6)
ax1.set_xscale('log')
ax1.set_xlabel('CPU Time (s) [log scale]')
ax1.set_title('(a) Computational Time')
ax1.set_yticks(y_pos)
ax1.set_yticklabels(solvers)
ax1.invert_yaxis()
ax1.grid(True, which='both', linestyle='--', alpha=0.4, axis='x')

# 2. Sparsity (聚焦区间 [0.095, 0.110])
ax2.barh(y_pos, sparsities, color='#3c8d76', alpha=0.85, edgecolor='black', linewidth=0.6)
ax2.axvline(0.100, color='#d95f02', linestyle='--', linewidth=1.5, label='True Ratio (0.100)')
ax2.set_xlabel('Sparsity (Nonzero Ratio)')
ax2.set_title('(b) Solution Sparsity')
ax2.set_xlim(0.095, 0.110)
ax2.legend(loc='lower right')
ax2.grid(True, linestyle='--', alpha=0.4, axis='x')

# 3. Relative Objective Error
ax3.barh(y_pos, rel_obj_err, color='#d95f02', alpha=0.85, edgecolor='black', linewidth=0.6)
ax3.set_xscale('log')
ax3.set_xlabel(r'$|f - f^*| / f^*$ [log scale]')
ax3.set_title('(c) Relative Objective Gap')
ax3.grid(True, which='both', linestyle='--', alpha=0.4, axis='x')

plt.tight_layout()
plt.savefig('solver_comparison.png', dpi=300)
plt.close(fig)