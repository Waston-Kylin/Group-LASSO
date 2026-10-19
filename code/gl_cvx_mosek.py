import cvxpy as cp

def gl_cvx_mosek(x0, A, b, mu, opts=None):
    n, l = x0.shape
    m = A.shape[0]
    
    # 优化变量
    x = cp.Variable((n, l))
    
    # 目标函数: 0.5 * ||Ax - b||_F^2 + mu * sum_i ||x[i, :]||_2
    residual = cp.norm(A @ x - b, 'fro') # Frobenius 范数
    reg = cp.sum(cp.norm(x, 2, axis=1)) # 计算每行 2 范数
    objective = cp.Minimize(0.5 * cp.square(residual) + mu * reg)
    
    prob = cp.Problem(objective)
    
    # 调用 MOSEK
    prob.solve(solver=cp.MOSEK, verbose=False)
    
    x_sol = x.value

    try:
        iter_cnt = prob.solver_stats.num_iters
    except Exception:
        iter_cnt = -1
    
    out = {'fval': float(prob.value)}
    
    return x_sol, iter_cnt, out