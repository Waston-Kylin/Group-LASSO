import os
os.environ["MOSEKLM_LICENSE_FILE"] = os.path.join(os.path.dirname(__file__), "mosek.lic")
os.environ["GRB_LICENSE_FILE"] = os.path.join(os.path.dirname(__file__), "gurobi.lic")
import time
from typing import Any
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np

rng = np.random.RandomState(97006855)

n = 512
m = 256
k = int(round(n * 0.1))
l = 2
A = rng.randn(m, n)
p = rng.permutation(n)[:k]
u = np.zeros((n, l))
u[p, :] = rng.randn(k, l)
b = A @ u
mu = 1e-2

# 初始点
x0 = rng.randn(n, l)

# 相对误差
def errfun(x1, x2):
    return np.linalg.norm(x1 - x2, 'fro') / (1.0 + np.linalg.norm(x1, 'fro'))

def errfun_exact(x):
    return np.linalg.norm(x - u, 'fro') / (1.0 + np.linalg.norm(u, 'fro'))

def sparsity(x):
    return np.sum(np.abs(x) > 1e-6 * np.max(np.abs(x))) / (n * l)

def run_solver(solver_fn, opts=None):
    if opts is None:
        opts = {}
    t0 = time.perf_counter()
    x_sol, iter_cnt, out = solver_fn(x0, A, b, mu, opts)
    t_elapsed = time.perf_counter() - t0
    return x_sol, t_elapsed, iter_cnt, out

def print_stat(name, x_sol, t_elapsed, iter_cnt, out, x_ref_mosek, x_ref_gurobi):
    sparsity_val = sparsity(x_sol)

    err_exact = errfun_exact(x_sol)
    err_mosek = errfun(x_ref_mosek, x_sol)
    err_gurobi = errfun(x_ref_gurobi, x_sol)

    print(f"{name:>15}: cpu: {t_elapsed:5.2f}s, iter: {iter_cnt:5d}, optval: {out['fval']:6.5e}, "
          f"sparsity: {sparsity_val:4.3f}, err-to-exact: {err_exact:3.2e}, "
          f"err-to-cvx-mosek: {err_mosek:3.2e}, err-to-cvx-gurobi: {err_gurobi:3.2e}")

def plot_results(x_sol, name, save_path, u, x_ref_mosek=None, x_ref_gurobi=None):
    n, l = u.shape
    idx = np.arange(1, n + 1)
    
    dir_name = os.path.dirname(save_path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)
        
    fig, axes = plt.subplots(l, 1, figsize=(10, 4 * l), sharex=True, squeeze=False)
    
    # 告知 Pylance 每个元素都是 matplotlib 的 Axes 对象
    ax_list: list[matplotlib.axes.Axes] = list(axes.flatten())
        
    for j in range(l):
        ax: Any = ax_list[j]
        ax.plot(idx, u[:, j], 'k-', label='Exact $u$', alpha=0.4, linewidth=1.5)
        ax.plot(idx, x_sol[:, j], 'r.', label=f'{name} Solution', markersize=6)
        ax.set_xlim([1, n])
        ax.set_ylabel(f'Column {j+1}')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='upper right')
        
    ax_list[0].set_title(f'Recovery Result: {name}')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

from gl_cvx_mosek import gl_cvx_mosek
from gl_cvx_gurobi import gl_cvx_gurobi
from gl_mosek import gl_mosek
from gl_gurobi import gl_gurobi
from gl_SGD_primal import gl_SGD_primal
from gl_GD_primal import gl_GD_primal
from gl_FGD_primal import gl_FGD_primal
from gl_ProxGD_primal import gl_ProxGD_primal
from gl_FProxGD_primal import gl_FProxGD_primal
from gl_ALM_dual import gl_ALM_dual
from gl_ADMM_dual import gl_ADMM_dual
from gl_ADMM_primal import gl_ADMM_primal
from gl_PPA_dual import gl_PPA_dual
from gl_BCD_primal import gl_BCD_primal

if __name__ == "__main__":
    x1, t1, iter1, out1 = run_solver(gl_cvx_mosek)
    x2, t2, iter2, out2 = run_solver(gl_cvx_gurobi)
    x3, t3, iter3, out3 = run_solver(gl_mosek)
    x4, t4, iter4, out4 = run_solver(gl_gurobi)
    x5, t5, iter5, out5 = run_solver(gl_SGD_primal)
    x6, t6, iter6, out6 = run_solver(gl_GD_primal)
    x7, t7, iter7, out7 = run_solver(gl_FGD_primal)
    x8, t8, iter8, out8 = run_solver(gl_ProxGD_primal)
    x9, t9, iter9, out9 = run_solver(gl_FProxGD_primal)
    x10, t10, iter10, out10 = run_solver(gl_ALM_dual)
    x11, t11, iter11, out11 = run_solver(gl_ADMM_dual)
    x12, t12, iter12, out12 = run_solver(gl_ADMM_primal)
    x13, t13, iter13, out13 = run_solver(gl_PPA_dual)
    x14, t14, iter14, out14 = run_solver(gl_BCD_primal)

    print_stat("CVX-Mosek", x1, t1, iter1, out1, x1, x2)
    print_stat("CVX-Gurobi", x2, t2, iter2, out2, x1, x2)
    print_stat("Mosek", x3, t3, iter3, out3, x1, x2)
    print_stat("Gurobi", x4, t4, iter4, out4, x1, x2)
    print_stat("SGD Primal", x5, t5, iter5, out5, x1, x2)
    print_stat("GD Primal", x6, t6, iter6, out6, x1, x2)
    print_stat("FGD Primal", x7, t7, iter7, out7, x1, x2)
    print_stat("ProxGD Primal", x8, t8, iter8, out8, x1, x2)
    print_stat("FProxGD Primal", x9, t9, iter9, out9, x1, x2)
    print_stat("ALM Dual", x10, t10, iter10, out10, x1, x2)
    print_stat("ADMM Dual", x11, t11, iter11, out11, x1, x2)
    print_stat("ADMM Primal", x12, t12, iter12, out12, x1, x2)
    print_stat("PPA Dual", x13, t13, iter13, out13, x1, x2)
    print_stat("BCD Primal", x14, t14, iter14, out14, x1, x2)

    plot_results(u, 'Exact', '../figures/gl_exact.png', u, x1, x2)
    plot_results(x1, 'CVX-Mosek', '../figures/gl_cvx_mosek.png', u, x1, x2)
    plot_results(x2, 'CVX-Gurobi', '../figures/gl_cvx_gurobi.png', u, x1, x2)
    plot_results(x3, 'Mosek', '../figures/gl_mosek.png', u, x1, x2)
    plot_results(x4, 'Gurobi', '../figures/gl_gurobi.png', u, x1, x2)
    plot_results(x5, 'SGD Primal', '../figures/gl_SGD_Primal.png', u, x1, x2)
    plot_results(x6, 'GD Primal', '../figures/gl_GD_primal.png', u, x1, x2)
    plot_results(x7, 'FGD Primal', '../figures/gl_FGD_primal.png', u, x1, x2)
    plot_results(x8, 'ProxGD Primal', '../figures/gl_ProxGD_primal.png', u, x1, x2)
    plot_results(x9, 'FProxGD Primal', '../figures/gl_FProxGD_primal.png', u, x1, x2)
    plot_results(x10, 'ALM Dual', '../figures/gl_ALM_dual.png', u, x1, x2)
    plot_results(x11, 'ADMM Dual', '../figures/gl_ADMM_dual.png', u, x1, x2)
    plot_results(x12, 'ADMM Primal', '../figures/gl_ADMM_primal.png', u, x1, x2)
    plot_results(x13, 'PPA Dual', '../figures/gl_PPA_dual.png', u, x1, x2)
    plot_results(x14, 'BCD Primal', '../figures/gl_BCD_primal.png', u, x1, x2)