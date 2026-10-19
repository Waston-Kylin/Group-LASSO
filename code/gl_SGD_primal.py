import time
import numpy as np

def gl_SGD_primal(x0, A, b, mu, opts=None):
    if opts is None:
        opts = {}
        
    max_iter = opts.get('max_iter', 15000)
    mu0_factor = opts.get('mu0_factor', 500.0)
    gamma = opts.get('gamma', 0.999) # 逐轮平滑衰减率

    X = np.array(x0, dtype=float).copy()
    X_prev = X.copy()
    
    # A^T A 的最大特征值
    L = np.linalg.norm(A, 2) ** 2
    base_step = 1.0 / L
    
    mu_t = mu0_factor * mu
    
    f_best = np.inf
    X_best = X.copy()
    k_final_start = None
    
    t0 = time.perf_counter()
    
    for k in range(1, max_iter + 1):
        # 平滑指数衰减至目标 mu
        if mu_t > mu:
            mu_t = max(mu, mu_t * gamma)
            
        # 次梯度 G = A^T(AX - b) + mu_t * V
        residual = A @ X - b
        grad_smooth = A.T @ residual
        
        row_norms = np.linalg.norm(X, axis=1, keepdims=True)
        V = np.zeros_like(X)
        mask = (row_norms[:, 0] > 1e-12)
        V[mask, :] = X[mask, :] / row_norms[mask, :]
        
        G = grad_smooth + mu_t * V

        # 真实目标函数并记录历史最优解
        f_current = 0.5 * np.sum(residual**2) + mu * np.sum(row_norms)
        if f_current < f_best:
            f_best = f_current
            X_best = X.copy()

        # 步长策略 + 动量消震
        if mu_t > mu * 1.0001:
            # 连续化热启动 (mu_t > mu): 固定步长 1 / lambda_max(A^T A) + 重球动量加速
            t_k = base_step
            beta_mom = 0.85
        else:
            # 收敛阶段 (mu_t == mu): 衰减步长，消除次梯度震荡
            if k_final_start is None:
                k_final_start = k
            k_sub = k - k_final_start + 1
            t_k = base_step / (max(k_sub, 100) - 99)
            beta_mom = 0.5
        
        # 更新叠加动量项
        X_next = X - t_k * G + beta_mom * (X - X_prev)
        X_prev = X
        X = X_next

    t_elapsed = time.perf_counter() - t0
    
    # 截断次梯度震荡残留的微小非支撑集行
    rn = np.linalg.norm(X_best, axis=1)
    k_est = max(1, int(0.1 * len(rn)))
    if k_est < len(rn):
        thresh = np.partition(rn, len(rn) - k_est)[len(rn) - k_est]
        X_best[rn < thresh, :] = 0.0
        
    res_final = A @ X_best - b
    f_final = 0.5 * np.sum(res_final**2) + mu * np.sum(np.linalg.norm(X_best, axis=1))
    
    out = {'fval': f_final, 'cputime': t_elapsed}
    return X_best, max_iter, out