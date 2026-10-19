import time
import gurobipy as gp
from gurobipy import GRB

def gl_gurobi(x0, A, b, mu, opts=None):
    m_rows, n_cols = A.shape
    l = b.shape[1]
    
    # Gurobi 模型
    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.start()
        model = gp.Model("Group_LASSO_SOCP", env=env)
        model.setParam("FeasibilityTol", 1e-8)
        model.setParam("OptimalityTol", 1e-8)
        model.setParam("BarConvTol", 1e-8)
        
        # 矩阵变量 MVar
        x = model.addMVar((n_cols, l), lb=-GRB.INFINITY, name="x")
        t = model.addMVar(n_cols, lb=0.0, name="t")
        R = model.addMVar((m_rows, l), lb=-GRB.INFINITY, name="R")
        t0 = model.addMVar(1, lb=0.0, name="t0")
        
        # 仿射约束 R = Ax - b
        model.addConstr(R == A @ x - b, name="residual")
        
        # 旋转二阶锥约束 t0 >= 0.5 * sum(R^2)
        model.addConstr(t0[0] >= 0.5 * (R * R).sum(), name="rot_cone")
        
        # 标准二阶锥约束 sum(x[i]^2) <= t[i]^2
        # for i in range(n_cols):
        #     model.addConstr((x[i, :] * x[i, :]).sum() <= t[i] * t[i], name=f"soc_{i}")
        model.addConstr((x * x).sum(axis=1) <= t * t, name="soc_all")
            
        # 线性目标函数: min t0 + mu * sum(t)
        model.setObjective(t0[0] + mu * t.sum(), GRB.MINIMIZE)
        
        # 求解
        t_start = time.perf_counter()
        model.optimize()
        t_elapsed = time.perf_counter() - t_start

        if model.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Gurobi failed: status={model.Status}")

        x_sol = x.X

        try:
            iter_cnt = int(model.BarIterCount)
        except Exception:
            iter_cnt = -1
        
        out = {'fval': model.ObjVal, 'cputime': t_elapsed}
        model.dispose()
    
    return x_sol, iter_cnt, out