import time
import numpy as np

def gl_ProxGD_primal(x0, A, b, mu, opts=None):
    if opts is None:
        opts = {}

    max_iter = opts.get('max_iter', 7500)
    mu0_factor = opts.get('mu0_factor', 500.0) # μ_0 = 500 * μ
    gamma = opts.get('gamma', 0.999) # 每步衰减率

    n, l = x0.shape
    X = np.array(x0, dtype=float).copy()

    L = np.linalg.norm(A.T @ A, 2)
    t_k = 1.0 / L

    mu_t = mu0_factor * mu

    f_best = np.inf
    X_best = X.copy()
    iter_cnt = max_iter

    t0 = time.perf_counter()

    for k in range(1, max_iter + 1):
        # mu 连续化
        if mu_t > mu:
            mu_t = max(mu, mu_t * gamma)

        # 梯度
        grad_g = A.T @ (A @ X - b)
        Z = X - t_k * grad_g

        # 组软阈值（μ_t，随连续化变化）
        row_norms = np.linalg.norm(Z, axis=1, keepdims=True)
        scaling = np.maximum(1.0 - (t_k * mu_t) / (row_norms + 1e-12), 0.0)
        X = Z * scaling

        # f_best
        res = A @ X - b
        f_cur = 0.5 * np.sum(res**2) + mu * np.sum(np.linalg.norm(X, axis=1))
        if f_cur < f_best:
            f_best = f_cur
            X_best = X.copy()

    t_elapsed = time.perf_counter() - t0

    out = {'fval': f_best, 'cputime': t_elapsed}
    return X_best, iter_cnt, out