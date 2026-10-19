# Group LASSO Optimization
## Mathematical Model
本项目求解如下无约束复合凸优化问题：
$$\min_{X\in\mathbb{R}^{n\times l}}F(X)=\frac{1}{2}\Vert{}AX-b\Vert{}_F^2+\mu\Vert{}X\Vert{}_{1,2}$$
其中：
* $A\in\mathbb{R}^{m\times n}$ 为给定的测量矩阵，$b\in\mathbb{R}^{m\times l}$ 为观测矩阵，$\mu>0$ 为正则化惩罚系数。
* $\Vert{}\cdot\Vert{}_F$ 表示 Frobenius 范数。
* $\Vert{}X\Vert{}_{1,2} = \sum_{i=1}^n \Vert{}X_{i,:}\Vert{}_2$ 为诱导行稀疏（Group Sparsity）的 $L_{1,2}$ 混合范数。

基准测试（`Test_group_lasso.py`）中，维度设定为 $n=512, m=256, l=2$，真实稀疏度比例为 $k/n =0.1$，正则化系数 $\mu=10^{-2}$。

## Dependencies

需 Python 3.8+ 环境及以下科学计算与优化库：

| 依赖包 | 用途说明 |
| :--- | :--- |
| `numpy` | 矩阵运算与向量化数组操作 |
| `scipy` | Cholesky 分解 (`scipy.linalg`) 与内层拟牛顿优化 (`L-BFGS-B`) |
| `matplotlib` | 解的恢复对比可视化与性能指标绘图 |
| `cvxpy` | 凸优化高层建模框架（用于构建基准对照组） |
| `Mosek` | 商业级二阶锥规划（SOCP）内点法求解器及 Fusion API |
| `gurobipy` | Gurobi 商业数学规划求解器 Python 原生矩阵接口 |

安装命令：
```bash
pip install numpy scipy matplotlib cvxpy Mosek gurobipy
```

## License Setup
前 4 个基准算法调用了 MOSEK 和 Gurobi 商业求解器，运行前须配置合法的许可证。其中 MOSEK 可在 [MOSEK 官网](https://www.mosek.com/) 免费注册，Gurobi 许可证需国外学生验证（没有此 license 代码无法正常运行），可到 [Gurobi 官网](https://www.gurobi.com/) 获取。

**我在文件中将两个 license 放在了 `code` 目录下，烦请不要传播使用。**

主测试脚本 `Test_group_lasso.py` 在顶部通过环境变量自动挂载当前目录下的许可证文件：
```python
os.environ["MOSEKLM_LICENSE_FILE"] = os.path.join(os.path.dirname(__file__), "mosek.lic")
os.environ["GRB_LICENSE_FILE"] = os.path.join(os.path.dirname(__file__), "gurobi.lic")
```
配置方式：
- 项目目录：将 `mosek.lic` 和 `gurobi.lic` 放在与 `code/Test_group_lasso.py` 代码目录下。
- 系统默认路径：若许可证在默认路径（如 `~/mosek/mosek.lic` 或 `/Library/gurobi/`），将 `Test_group_lasso.py` 顶部两行
```python
os.environ["MOSEKLM_LICENSE_FILE"] = os.path.join(os.path.dirname(__file__), "mosek.lic")
os.environ["GRB_LICENSE_FILE"] = os.path.join(os.path.dirname(__file__), "gurobi.lic")
```
注释掉，求解器会检索系统默认位置。


## Running
进入 `code` 目录，运行主测试脚本：
```bash
python Test_group_lasso.py
```
终端输出示例：
```bash
    Mosek: cpu:  0.18s, iter:    10, optval: 6.70575e-01, sparsity: 0.103, err-to-exact: 3.18e-05, err-to-cvx-mosek: 1.70e-07, err-to-cvx-gurobi: 1.28e-06
   Gurobi: cpu:  0.34s, iter:     9, optval: 6.70575e-01, sparsity: 0.104, err-to-exact: 3.20e-05, err-to-cvx-mosek: 2.26e-07, err-to-cvx-gurobi: 1.02e-06
```
所有的解恢复对比图保存至 `../figures/` 目录下（目录不存在自动创建），若不想生成图片请注释掉 `Test_group_lasso.py` 主函数最后 14 行。

## Solver Interface & `opt` parameter
求解器遵循统一的函数签名：
```python
x_sol, iter_cnt, out = solver_fn(x0, A, b, mu, opts=None)
```
### 输入与输出参数：
- 输入参数：
    - `x0 (np.ndarray, shape (n, l))`：迭代初始点（内点法求解器忽略）。
    - `A (np.ndarray, shape (m, n))`：测量矩阵。
    - `b (np.ndarray, shape (m, l))`：观测矩阵。
    - `mu (float)`：组稀疏正则化参数 $\mu$。
    - `opts` (dict, 可选)：算法超参数字典，默认值为 `None`（启用函数预设的默认配置）。
- 返回值：
    - `x_sol (np.ndarray, shape (n, l))`：求解器恢复的稀疏解矩阵。
    - `iter_cnt (int)`：实际执行迭代步数（若底层求解器未返回则为 -1）。
    - `out (dict)`：包含 `fval`（最终目标函数值 float）与 `cputime`（内部核心求解耗时 float）。

### `opt` 参数设置：
若要显示设置请在 `Test_group_lasso.py` 主函数调用中添加 `opts` 参数（如 `opts={"max_iter": 100}`）。所有 `opt` 参数：
- 次梯度法 `gl_SGD_primal`：
    - `max_iter`：最大迭代步数，默认 15000。
    - `mu0_factor`：连续化初始正则化放大倍数（即 $\mu_0 = \mu_0\cdot\mu$），默认 500.0。
    - `gamma`：连续化每步指数衰减率，默认 0.999。
- 光滑化梯度下降法 `gl_GD_primal`：
    - `max_iter`：最大迭代步数，默认 10000。
    - `mu0_factor`：连续化初始放大倍数，默认 500.0。
    - `gamma`：连续化每步衰减率，默认 0.999。
    - `mu_s_ratio`：Huber 光滑参数与当前 $\mu_t$ 的联动比例（$\mu_s = \max(10^{-5},\mu_{s,ratio}\cdot\mu_t)$），默认 0.02。
    - `n_keep_pct`：尾部硬阈值截断保留的非零行分位数比例，默认 0.1。
- 光滑化快速梯度法 `gl_FGD_primal`：
    - `max_iter`：总迭代步数（均分至各平滑阶段），默认 5000。
    - `mu_s_list`：多阶段 Huber 光滑参数递减列表，默认 `[1e-2, 3e-3, 1e-3, 3e-4, 1e-4]`。
    - `use_restart`：是否启用目标函数反弹动量重置（FISTA Restart），默认 True。
    - `n_keep_pct`：硬阈值后处理保留比例，默认 0.1。
- 近端梯度法 `gl_ProxGD_primal`：
    - `max_iter`：最大迭代步数，默认 7500。
    - `mu0_factor`：连续化初始放大倍数，默认 500.0。
    - `gamma`：连续化每步衰减率，默认 0.999。
- 快速近端梯度法 / FISTA `gl_FProxGD_primal`：
    - `max_iter`：最大迭代步数，默认 7500。
    - `mu0_factor`：连续化初始放大倍数，默认 500.0。
    - `gamma`：连续化每步衰减率，默认 0.999。
    - `use_restart`：是否启用非单调反弹重启机制，默认 True。
- 对偶增广拉格朗日乘子法 `gl_ALM_dual`：
    - `max_iter`：外层乘子更新次数，默认 500。
    - `t`：增广拉格朗日惩罚参数，默认 1.0。
- 对偶交替方向乘子法 `gl_ADMM_dual`：
    - `max_iter`：最大交替迭代步数，默认 2000。
    - `t`：二次罚参数，默认 1.0。
- 主问题交替方向乘子法 `gl_ADMM_primal`：
    - `max_iter`：最大交替迭代步数，默认 2000。
    - `beta`：一致性约束 $X - Z = 0$ 的增广罚参数 $\beta$，默认 1.0。
- 对偶近似点算法 `gl_PPA_dual`：
    - `max_iter`：外层 PPA 步数，默认 500。
    - `inner_iter`：内层子问题 FISTA 求解步数，默认 50。
    - `t`：PPA 近端步长参数，默认 1.0。
- 块坐标下降法 `gl_BCD_primal`：
    - `max_iter`：外层全行扫描（Epoch）最大次数，默认 5000。
    - `mu0_factor`：连续化初始放大倍数，默认 20.0。
    - `gamma`：每轮衰减率，默认 0.999。
    - `tol`：相对目标函数值提前终止容差，默认 1e-12。
    - `use_zero_init`：是否强制从全零矩阵初始化（极大加速稀疏支撑集识别），默认 True。

## Notes
- 随机种子：`Test_group_lasso.py` 固定了随机数生成器 `rng = np.random.RandomState(97006855)`，可按需修改。
- 评价指标：
    - 相对误差 (`err-to-exact` / `err-to-cvx`)：定义为 $\frac{\Vert{}X_1 - X_2\Vert{}_F}{1 + \Vert{}X_1\Vert{}_F}$。
    - 稀疏度：定义为满足 $\vert{}X_{ij}\vert{} > 10^{-6} \max_{k,l}\vert{}X_{kl}\vert{}$ 的元素占总元素数 $n \times l$ 的比例。
