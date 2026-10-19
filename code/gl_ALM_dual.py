import time
import numpy as np
from scipy.optimize import minimize

def gl_ALM_dual(x0, A, b, mu, opts=None):
    if opts is None:
        opts = {}

    max_iter = opts.get('max_iter', 500)
    t = opts.get('t', 1.0) # 增广罚参数
    
    m, n = A.shape
    l = b.shape[1]
    
    X = np.array(x0, dtype=float).copy()
    Lambda = np.zeros((m, l))
    
    f_best = np.inf
    X_best = X.copy()
    
    t0 = time.perf_counter()
    
    for k in range(max_iter):
        # ALM 内层子问题目标函数 & 梯度
        def obj_grad(lambda_flat):
            Lbd = lambda_flat.reshape((m, l))
            V = A.T @ Lbd + X / t
            
            # P_C(V) 行投影软截断
            row_norms = np.linalg.norm(V, axis=1, keepdims=True)
            scaling = np.minimum(1.0, mu / (row_norms + 1e-12))
            Z_proj = V * scaling
            
            diff = V - Z_proj
            # 目标函数
            f_val = 0.5 * np.sum(Lbd**2) - np.sum(b * Lbd) + 0.5 * t * np.sum(diff**2)
            grad_Lbd = Lbd - b + t * (A @ diff)
            
            return f_val, grad_Lbd.flatten()
            
        # L-BFGS-B 内层平滑
        res = minimize(obj_grad, Lambda.flatten(), jac=True, method='L-BFGS-B', 
                       options={'maxiter': 10, 'ftol': 1e-4})
        Lambda = res.x.reshape((m, l))
        
        # 恢复投影变量 Z
        V = A.T @ Lambda + X / t
        row_norms = np.linalg.norm(V, axis=1, keepdims=True)
        scaling = np.minimum(1.0, mu / (row_norms + 1e-12))
        Z = V * scaling
        
        # ALM 乘子更新（原问题 X 更新）
        X = X + t * (A.T @ Lambda - Z)
        
        res_true = A @ X - b
        f_current = 0.5 * np.sum(res_true**2) + mu * np.sum(np.linalg.norm(X, axis=1))
        if f_current < f_best:
            f_best = f_current
            X_best = X.copy()

    t_elapsed = time.perf_counter() - t0

    # 截断极小浮点误差
    rn = np.linalg.norm(X_best, axis=1)
    k_est = max(1, int(0.1 * n))
    thresh = np.partition(rn, n - k_est)[n - k_est]
    X_best[rn < thresh, :] = 0.0
    
    out = {'fval': f_best, 'cputime': t_elapsed}
    return X_best, max_iter, out