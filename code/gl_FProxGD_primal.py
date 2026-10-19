import time
import numpy as np

def gl_FProxGD_primal(x0, A, b, mu, opts=None):
    if opts is None:
        opts = {}

    max_iter = opts.get('max_iter', 7500)
    mu0_factor = opts.get('mu0_factor', 500.0)
    gamma = opts.get('gamma', 0.999)
    use_restart = opts.get('use_restart', True)

    n, l = x0.shape
    X_prev = np.array(x0, dtype=float).copy()
    X = X_prev.copy()

    L = np.linalg.norm(A.T @ A, 2)
    t_k = 1.0 / L

    mu_t = mu0_factor * mu

    f_best = np.inf
    X_best = X.copy()
    iter_cnt = max_iter
    f_prev_iter = np.inf

    t0 = time.perf_counter()

    for k in range(1, max_iter + 1):
        # mu 连续化
        if mu_t > mu:
            mu_t = max(mu, mu_t * gamma)

        # FISTA 动量外推
        momentum = (k - 2) / (k + 1) if k >= 2 else 0.0
        Y = X + momentum * (X - X_prev)

        # 梯度
        grad_g = A.T @ (A @ Y - b)
        V = Y - t_k * grad_g

        # 组软阈值
        row_norms_V = np.linalg.norm(V, axis=1, keepdims=True)
        scaling = np.maximum(1.0 - (t_k * mu_t) / (row_norms_V + 1e-12), 0.0)
        X_new = V * scaling

        # 更新后 X 范数评估 f
        res = A @ X_new - b
        f_cur = 0.5 * np.sum(res**2) + mu * np.sum(np.linalg.norm(X_new, axis=1))

        if f_cur < f_best:
            f_best = f_cur
            X_best = X_new.copy()

        # 目标反弹则重置
        if use_restart and k >= 2 and f_cur > f_prev_iter:
            X_prev = X.copy()
            momentum = 0.0
            # 下次迭代 Y = X
        else:
            X_prev = X.copy()

        X = X_new
        f_prev_iter = f_cur

    t_elapsed = time.perf_counter() - t0

    out = {'fval': f_best, 'cputime': t_elapsed}
    return X_best, iter_cnt, out