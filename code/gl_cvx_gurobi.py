import cvxpy as cp
import gurobipy as gp

def gl_cvx_gurobi(x0, A, b, mu, opts=None):
    n, l = x0.shape
    x = cp.Variable((n, l))
    
    # 目标函数: Frobenius范数平方 + L1,2 范数正则化项
    residual = cp.norm(A @ x - b, 'fro')
    reg = cp.sum(cp.norm(x, 2, axis=1))

    objective = cp.Minimize(0.5 * cp.square(residual) + mu * reg)
    prob = cp.Problem(objective)

    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.start()

        # 调用 GUROBI
        prob.solve(solver=cp.GUROBI, env=env, verbose=False)
    
    x_sol = x.value

    try:
        iter_cnt = prob.solver_stats.num_iters
    except Exception:
        iter_cnt = -1
    
    out = {'fval': float(prob.value)}
    
    return x_sol, iter_cnt, out