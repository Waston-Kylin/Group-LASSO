import time
import numpy as np

def gl_FGD_primal(x0, A, b, mu, opts=None):
    if opts is None:
        opts = {}

    max_iter = opts.get('max_iter', 5000)
    mu_s_list = opts.get('mu_s_list',
                          [1e-2, 3e-3, 1e-3, 3e-4, 1e-4]) # 1e-2 降到 1e-4
    n_keep_pct = opts.get('n_keep_pct', 0.1)
    use_restart = opts.get('use_restart', True)

    n, l = x0.shape
    X = np.array(x0, dtype=float).copy()
    L1 = np.linalg.norm(A.T @ A, 2)

    f_best = np.inf
    X_best = X.copy()
    iter_cnt = 0

    t0 = time.perf_counter()

    num_stages = len(mu_s_list)
    stage_iter = max(1, max_iter // num_stages)

    for stage, mu_s in enumerate(mu_s_list):
        L = L1 + mu / mu_s
        t_k = 1.0 / L

        # 每阶段重启
        X_prev = X.copy()
        f_prev = np.inf

        for k in range(1, stage_iter + 1):
            iter_cnt += 1

            # FISTA 动量外推
            momentum = (k - 2) / (k + 1) if k >= 2 else 0.0
            Y = X + momentum * (X - X_prev)

            # Huber 光滑梯度
            residual = A @ Y - b
            grad_smooth = A.T @ residual

            rn = np.linalg.norm(Y, axis=1, keepdims=True)
            V = np.zeros_like(Y)
            mask_small = rn[:, 0] <= mu_s
            V[mask_small, :] = Y[mask_small, :] / mu_s
            mask_large = rn[:, 0] > mu_s
            V[mask_large, :] = Y[mask_large, :] / rn[mask_large, :]

            G = grad_smooth + mu * V

            # 更新
            X_prev = X.copy()
            X = Y - t_k * G

            # 非光滑目标评估 f_best
            res_true = A @ X - b
            f_cur = 0.5 * np.sum(res_true**2) + mu * np.sum(np.linalg.norm(X, axis=1))
            if f_cur < f_best:
                f_best = f_cur
                X_best = X.copy()

            # 目标反弹则重置
            if use_restart and k >= 2 and f_cur > f_prev:
                X_prev = X.copy()
                # 下次迭代归零 momentum

            f_prev = f_cur

    t_elapsed = time.perf_counter() - t0

    # 分位数硬阈值 - 稀疏
    rn = np.linalg.norm(X_best, axis=1)
    k_est = max(1, int(n_keep_pct * n))
    if k_est < n:
        thresh = np.partition(rn, n - k_est)[n - k_est]
        X_best[rn < thresh, :] = 0.0

    # 非光滑目标评估
    residual = A @ X_best - b
    f_final = 0.5 * np.sum(residual**2) + mu * np.sum(np.linalg.norm(X_best, axis=1))

    out = {'fval': f_final, 'cputime': t_elapsed}
    return X_best, iter_cnt, out