import time
import numpy as np

def gl_PPA_dual(x0, A, b, mu, opts=None):
    if opts is None:
        opts = {}
    max_iter_outer = opts.get('max_iter', 500)
    max_iter_inner = opts.get('inner_iter', 50)
    t = opts.get('t', 1.0)

    m, n = A.shape
    l = b.shape[1]

    X = np.array(x0, dtype=float).copy()
    Lambda = np.zeros((m, l))

    f_best = np.inf
    X_best = X.copy()

    # FISTA 常数
    c = t / (1.0 + t)
    L_inner = c * np.linalg.norm(A.T @ A, 2)
    step_inner = 1.0 / L_inner

    t0 = time.perf_counter()

    for k in range(max_iter_outer):
        # 内层 X: X^{k+1} = argmin_X 0.5*c*||AX - d||^2 + mu*||X||_{1,2}
        # 其中 c = t/(1+t), d = b - Λ^k/t
        d = b - Lambda / t
        X_prev = X.copy()
        X_inner = X.copy()

        for inner_k in range(1, max_iter_inner + 1):
            momentum = (inner_k - 2) / (inner_k + 1) if inner_k >= 2 else 0.0
            Y_extrap = X_inner + momentum * (X_inner - X_prev)

            grad = c * A.T @ (A @ Y_extrap - d)
            V = Y_extrap - step_inner * grad

            X_prev = X_inner.copy()
            row_norms = np.linalg.norm(V, axis=1, keepdims=True)
            scaling = np.maximum(1.0 - (step_inner * mu) / (row_norms + 1e-12), 0.0)
            X_inner = V * scaling

        X = X_inner

        # Y 更新: Y = (t*(b - AX) - Λ) / (1+t)
        Y = (t * (b - A @ X) - Lambda) / (1.0 + t)

        # Λ 更新: Λ^{k+1} = (Λ^k + t*(AX^{k+1} - b)) / (1+t)
        Lambda = (Lambda + t * (A @ X - b)) / (1.0 + t)

        # f_best
        res_true = A @ X - b
        f_cur = 0.5 * np.sum(res_true**2) + mu * np.sum(np.linalg.norm(X, axis=1))
        if f_cur < f_best:
            f_best = f_cur
            X_best = X.copy()

    t_elapsed = time.perf_counter() - t0
    X_best[np.abs(X_best) < 1e-4] = 0.0

    out = {'fval': f_best, 'cputime': t_elapsed}
    return X_best, max_iter_outer, out