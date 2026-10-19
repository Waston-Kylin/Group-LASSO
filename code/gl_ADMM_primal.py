import time
import numpy as np
from scipy.linalg import cho_factor, cho_solve

def gl_ADMM_primal(x0, A, b, mu, opts=None):
    if opts is None:
        opts = {}

    max_iter = opts.get('max_iter', 2000)
    beta = opts.get('beta', 1.0)

    m, n = A.shape
    l = b.shape[1]

    # 预分解 (A^T A + beta I)
    M = A.T @ A + beta * np.eye(n)
    c, lower = cho_factor(M)
    Atb = A.T @ b

    # 初始化
    X = np.array(x0, dtype=float).copy()
    Z = X.copy()
    U = np.zeros((n, l))

    f_best = np.inf
    Z_best = Z.copy()

    t0 = time.perf_counter()

    # 增广拉格朗日: L = 0.5||AX-b||^2 + mu||Z||_{1,2} + (beta/2)||X - Z + U||^2
    for k in range(max_iter):
        # X 步：(A^T A + beta I) X = A^T b + beta (Z - U)
        rhs = Atb + beta * (Z - U)
        X = cho_solve((c, lower), rhs)

        # Z 步：组软阈值 Z = soft_threshold_{mu/beta}(X + U)
        V = X + U
        rn = np.linalg.norm(V, axis=1, keepdims=True)
        Z = V * np.maximum(0.0, 1.0 - (mu / beta) / (rn + 1e-12))

        # U 步：对偶乘子更新 U = U + (X - Z)
        U = U + (X - Z)

        # Z 上评估
        res = A @ Z - b
        f_cur = 0.5 * np.sum(res**2) + mu * np.sum(np.linalg.norm(Z, axis=1))
        if f_cur < f_best:
            f_best = f_cur
            Z_best = Z.copy()

    t_elapsed = time.perf_counter() - t0

    # 清除残余浮点误差
    Z_best[np.abs(Z_best) < 1e-4] = 0.0

    out = {'fval': f_best, 'cputime': t_elapsed}
    return Z_best, max_iter, out