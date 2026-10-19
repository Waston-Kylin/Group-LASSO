import time
import numpy as np

def gl_BCD_primal(x0, A, b, mu, opts=None):
    if opts is None: opts = {}
    max_iter = opts.get('max_iter', 10000)
    mu0_factor = opts.get('mu0_factor', 20.0) # 前期 mu 大加速清零
    gamma = opts.get('gamma', 0.999)
    tol = opts.get('tol', 1e-12)
    use_zero_init = opts.get('use_zero_init', True)

    m, n = A.shape
    l = b.shape[1]

    X = np.zeros((n, l)) if use_zero_init else np.array(x0, dtype=float).copy()
    R = A @ X - b

    A_col_norms_sq = np.sum(A**2, axis=0)

    f_best = np.inf
    X_best = X.copy()

    mu_t = mu0_factor * mu
    t0 = time.perf_counter()

    for k in range(max_iter):
        # 连续化
        if mu_t > mu:
            mu_t = max(mu, mu_t * gamma)
        thresholds = mu_t / np.maximum(A_col_norms_sq, 1e-12)

        for i in range(n):
            norm_ai_sq = A_col_norms_sq[i]
            if norm_ai_sq < 1e-12:
                continue

            a_i = A[:, i]
            x_i_old = X[i, :].copy()

            w_i = x_i_old - (a_i.T @ R) / norm_ai_sq
            norm_wi = np.linalg.norm(w_i)
            thr = thresholds[i]

            if norm_wi <= thr:
                x_i_new = np.zeros(l)
            else:
                x_i_new = (1.0 - thr / norm_wi) * w_i

            X[i, :] = x_i_new
            R += np.outer(a_i, x_i_new - x_i_old)

        # mu 评估
        f_cur = 0.5 * np.sum(R**2) + mu * np.sum(np.linalg.norm(X, axis=1))
        if f_cur < f_best:
            f_best = f_cur
            X_best = X.copy()
        elif k > 100 and abs(f_cur - f_best) / (1.0 + abs(f_best)) < tol:
            break

    t_elapsed = time.perf_counter() - t0
    out = {'fval': f_best, 'cputime': t_elapsed}
    return X_best, k + 1, out