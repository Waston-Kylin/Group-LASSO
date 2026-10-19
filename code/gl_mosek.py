import time
import numpy as np
import mosek.fusion as mf

def gl_mosek(x0, A, b, mu, opts=None):
    m_rows, n_cols = A.shape
    l = b.shape[1]
    
    # Mosek Fusion 模型
    M = mf.Model("Group_LASSO_SOCP")
    M.setSolverParam("intpntCoTolRelGap", 1e-8)
    
    # 决策变量
    x  = M.variable("x", [n_cols, l], mf.Domain.unbounded())
    t  = M.variable("t", n_cols, mf.Domain.greaterThan(0.0))
    R  = M.variable("R", [m_rows, l], mf.Domain.unbounded())
    t0 = M.variable("t0", 1, mf.Domain.greaterThan(0.0))
    
    # 仿射约束 R = Ax - b  =>  Ax - R = b
    M.constraint("residual", 
                 mf.Expr.sub(mf.Expr.mul(A, x), R), 
                 mf.Domain.equalsTo(b))
                 
    # 旋转二阶锥约束 2 * t0 * 1 >= ||R||_F^2
    M.constraint("rot_cone",
                 mf.Expr.vstack(t0, 1.0, mf.Expr.flatten(R)),
                 mf.Domain.inRotatedQCone())
                 
    # 标准二阶锥约束 ||x(i,:)||_2 <= t_i
    # for i in range(n_cols):
    #     M.constraint(f"soc_{i}",
    #                  mf.Expr.vstack(t.index(i), mf.Expr.flatten(x.slice([i, 0], [i+1, l]))),
    #                  mf.Domain.inQCone())
    Y = mf.Expr.hstack([t, x])
    M.constraint("soc_all", Y, mf.Domain.inQCone())
                     
    # 线性目标函数 min t0 + mu * sum(t)
    obj = mf.Expr.add(t0, mf.Expr.mul(mu, mf.Expr.sum(t)))
    M.objective(mf.ObjectiveSense.Minimize, obj)
    
    # 求解
    t_start = time.perf_counter()
    M.solve()
    t_elapsed = time.perf_counter() - t_start

    status = M.getProblemStatus()
    if status != mf.ProblemStatus.PrimalAndDualFeasible:
        raise RuntimeError(f"Mosek status: {status}")

    x_sol = np.array(x.level()).reshape(n_cols, l) 
    try:
        iter_cnt = int(M.getSolverIntInfo("intpntIter"))
    except Exception:
        iter_cnt = -1

    out = {'fval': M.primalObjValue(), 'cputime': t_elapsed}
    M.dispose()
    return x_sol, iter_cnt, out