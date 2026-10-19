import time
import numpy as np
from scipy.linalg import cho_factor, cho_solve

def gl_ADMM_dual(x0, A, b, mu, opts=None):
    if opts is None:
        opts = {}
    
    max_iter = opts.get('max_iter', 2000)
    t = opts.get('t', 1.0)
    
    m, n = A.shape
    l = b.shape[1]
    
    X = np.array(x0, dtype=float).copy()
    Lambda = np.zeros((m, l))
    Z = np.zeros((n, l))
    
    f_best = np.inf
    X_best = X.copy()
    
    # 外层 Cholesky 矩阵分解，循环内仅回代
    matrix_sys = np.eye(m) + t * (A @ A.T)
    c, lower = cho_factor(matrix_sys)
    
    t0 = time.perf_counter()
    
    for k in range(max_iter):
        # Lambda 更新
        rhs = b + t * (A @ Z) - A @ X
        Lambda = cho_solve((c, lower), rhs)
        
        # Z 更新
        V = A.T @ Lambda + X / t
        row_norms = np.linalg.norm(V, axis=1, keepdims=True)
        scaling = np.minimum(1.0, mu / (row_norms + 1e-12))
        Z = V * scaling
        
        # X 更新
        X = X + t * (A.T @ Lambda - Z)
        
        res_true = A @ X - b
        f_current = 0.5 * np.sum(res_true**2) + mu * np.sum(np.linalg.norm(X, axis=1))
        if f_current < f_best:
            f_best = f_current
            X_best = X.copy()

    t_elapsed = time.perf_counter() - t0
    
    rn = np.linalg.norm(X_best, axis=1)
    k_est = max(1, int(0.1 * n))
    thresh = np.partition(rn, n - k_est)[n - k_est]
    X_best[rn < thresh, :] = 0.0
    
    out = {'fval': f_best, 'cputime': t_elapsed}
    return X_best, max_iter, out