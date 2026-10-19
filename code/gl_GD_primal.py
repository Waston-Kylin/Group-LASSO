import time
import numpy as np

def gl_GD_primal(x0, A, b, mu, opts=None):
    if opts is None:
        opts = {}

    max_iter = opts.get('max_iter', 10000)
    mu0_factor = opts.get('mu0_factor', 500.0)
    gamma = opts.get('gamma', 0.999)
    mu_s_ratio = opts.get('mu_s_ratio', 0.02)
    n_keep_pct = opts.get('n_keep_pct', 0.1)

    n, l = x0.shape
    X = np.array(x0, dtype=float).copy()

    L1 = np.linalg.norm(A, 2) ** 2
    mu_t = mu0_factor * mu

    f_best = np.inf
    X_best = X.copy()

    # Zhang-Hager 非单调线搜索状态变量
    eta = 0.85
    Q_k = 1.0
    C_k = None

    X_prev = None
    G_prev = None

    t0 = time.perf_counter()

    for k in range(1, max_iter + 1):
        # mu 逐轮平滑指数连续化
        if mu_t > mu:
            mu_t = max(mu, mu_t * gamma)
        mu_s = max(1e-5, mu_s_ratio * mu_t)
        L = L1 + mu_t / mu_s
        t_fixed = 1.0 / L

        # Huber 光滑化目标函数与梯度
        residual = A @ X - b
        grad_smooth = A.T @ residual

        rn = np.linalg.norm(X, axis=1, keepdims=True)
        V = np.zeros_like(X)
        mask_small = (rn[:, 0] <= mu_s)
        V[mask_small, :] = X[mask_small, :] / mu_s
        mask_large = (rn[:, 0] > mu_s)
        V[mask_large, :] = X[mask_large, :] / rn[mask_large, :]

        G = grad_smooth + mu_t * V

        # 当前光滑函数值
        phi_cur = np.where(rn[:, 0] <= mu_s, (rn[:, 0]**2) / (2.0 * mu_s), rn[:, 0] - 0.5 * mu_s)
        f_smooth_cur = 0.5 * np.sum(residual**2) + mu_t * np.sum(phi_cur)

        if C_k is None:
            C_k = f_smooth_cur
        else:
            Q_next = eta * Q_k + 1.0
            C_k = (eta * Q_k * C_k + f_smooth_cur) / Q_next
            Q_k = Q_next

        # 原问题目标函数最优值
        f_true = 0.5 * np.sum(residual**2) + mu * np.sum(rn)
        if f_true < f_best:
            f_best = f_true
            X_best = X.copy()

        # Barzilai-Borwein 步长
        if X_prev is not None and G_prev is not None:
            S = X - X_prev
            Y_grad = G - G_prev
            sy = np.sum(S * Y_grad)
            if sy > 1e-14:
                if k % 2 == 1:
                    t_bb = np.sum(S * S) / sy # BB1
                else:
                    t_bb = sy / np.sum(Y_grad * Y_grad) # BB2
                t_k = np.clip(t_bb, t_fixed, 10.0 / L1)
            else:
                t_k = t_fixed
        else:
            t_k = t_fixed

        # 非单调 Armijo 回溯线搜索保护
        X_prev = X.copy()
        G_prev = G.copy()

        g_norm_sq = np.sum(G**2)
        for _ in range(5):
            X_trial = X - t_k * G
            res_trial = A @ X_trial - b
            rn_trial = np.linalg.norm(X_trial, axis=1)
            phi_trial = np.where(rn_trial <= mu_s, (rn_trial**2) / (2.0 * mu_s), rn_trial - 0.5 * mu_s)
            f_smooth_trial = 0.5 * np.sum(res_trial**2) + mu_t * np.sum(phi_trial)

            # Zhang-Hager 非单调下降条件: f(x + alpha*d) <= C_k - sigma * alpha * ||g||^2
            if f_smooth_trial <= C_k - 1e-4 * t_k * g_norm_sq:
                break
            t_k *= 0.5
        else:
            X_trial = X - t_fixed * G

        X = X_trial

    t_elapsed = time.perf_counter() - t0

    # 分位数硬阈值
    rn = np.linalg.norm(X_best, axis=1)
    k_est = max(1, int(n_keep_pct * n))
    if k_est < n:
        thresh = np.partition(rn, n - k_est)[n - k_est]
        X_best[rn < thresh, :] = 0.0

    residual = A @ X_best - b
    f_final = 0.5 * np.sum(residual**2) + mu * np.sum(np.linalg.norm(X_best, axis=1))

    out = {'fval': f_final, 'cputime': t_elapsed}
    return X_best, max_iter, out