import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import root
from types import SimpleNamespace

p = SimpleNamespace()

p.beta = 0.95
p.alpha = np.array([0.35, 0.25, 0.3])
p.delta = 0.08
p.tol = 1e-6
p.min_c = 1e-10
p.maxiter = 100000
p.T = 150
p.z_old = 0.9
p.z_new = 1.1
p.k_count = len(p.alpha)
p.z_path = np.full(p.T, p.z_new)


def compute_y(k, z, params):
    """计算 Cobb-Douglas 产出。

    参数：
        k：各类资本存量向量。
        z：技术水平。
        params：模型参数命名空间。

    返回值：
        float，产出。
    """

    return z * np.prod(k ** params.alpha)


def compute_mp(k, z, params):
    """计算各类资本的边际产出。

    参数：
        k：各类资本存量向量。
        z：技术水平。
        params：模型参数命名空间。

    返回值：
        ndarray，各类资本的边际产出。
    """

    y = compute_y(k, z, params)
    return params.alpha * y / k


def compute_ss(z, params):
    """计算给定技术水平下的稳态。

    参数：
        z：技术水平。
        params：模型参数命名空间。

    返回值：
        SimpleNamespace，包含稳态消费、产出和资本向量。
    """

    r = 1 / params.beta - 1 + params.delta
    alpha = params.alpha
    alpha_sum = np.sum(alpha)

    y = (z * np.prod((alpha / r) ** alpha)) ** (1 / (1 - alpha_sum))
    k = alpha * y / r
    c = y - params.delta * np.sum(k)

    return SimpleNamespace(c=c, y=y, k=k)


def paths_from_x(x, k0, kT1, params):
    """由未知资本向量还原完整路径。

    参数：
        x：未知量，对应 MATLAB 中的 k(:, 2:T)。
        k0：初始资本。
        kT1：终端资本。
        params：模型参数命名空间。

    返回值：
        tuple，依次为资本路径、消费路径、产出路径。
    """

    k_count = params.k_count
    T = params.T

    # MATLAB reshape 是列优先；这里必须指定 order="F" 保持一致。
    k_mid = x.reshape((k_count, T - 1), order="F")

    k_path = np.zeros((k_count, T + 1))
    k_path[:, 0] = k0
    k_path[:, 1:T] = k_mid
    k_path[:, T] = kT1

    c_path = np.zeros(T)
    y_path = np.zeros(T)

    for t in range(T):
        k_today = k_path[:, t]
        k_next = k_path[:, t + 1]
        y_path[t] = compute_y(k_today, params.z_path[t], params)
        c_path[t] = y_path[t] + (1 - params.delta) * np.sum(k_today) - np.sum(k_next)

    return k_path, c_path, y_path


def residual_x(x, k0, kT1, params):
    """计算 Euler 方程残差。

    参数：
        x：未知资本向量。
        k0：初始资本。
        kT1：终端资本。
        params：模型参数命名空间。

    返回值：
        ndarray，按 MATLAB 列优先顺序拉直的残差。
    """

    k_path, c_path, _ = paths_from_x(x, k0, kT1, params)
    k_count = params.k_count
    T = params.T

    if np.min(c_path) <= params.min_c or np.min(k_path) <= 0:
        return 1e8 * np.ones(k_count * (T - 1))

    F = np.zeros((k_count, T - 1))

    for t in range(T - 1):
        k_next = k_path[:, t + 1]
        mp_next = compute_mp(k_next, params.z_path[t + 1], params)
        lhs = c_path[t + 1] / c_path[t]
        rhs = params.beta * (mp_next + 1 - params.delta)
        F[:, t] = np.log(lhs) - np.log(rhs)

    return F.reshape(-1, order="F")


def initial_guess(old_ss, new_ss, params):
    """构造线性过渡初始猜测。

    参数：
        old_ss：旧稳态。
        new_ss：新稳态。
        params：模型参数命名空间。

    返回值：
        ndarray，按 MATLAB 列优先顺序拉直的初始猜测。
    """

    T = params.T
    k_count = params.k_count
    k_path0 = np.zeros((k_count, T + 1))

    for t in range(T + 1):
        w = t / T
        k_path0[:, t] = (1 - w) * old_ss.k + w * new_ss.k

    return k_path0[:, 1:T].reshape(-1, order="F")


def solve_model(params):
    """求解转移路径。

    参数：
        params：模型参数命名空间。

    返回值：
        tuple，依次为旧稳态、新稳态、资本路径、消费路径、产出路径、最终残差。
    """

    old_ss = compute_ss(params.z_old, params)
    new_ss = compute_ss(params.z_new, params)
    x0 = initial_guess(old_ss, new_ss, params)

    def Ffun(x):
        """返回给定未知资本向量对应的 Euler 残差。"""

        return residual_x(x, old_ss.k, new_ss.k, params)

    # root 的 fatol 控制最大分量残差；这里换算后约束整体二范数残差。
    root_tol = params.tol / np.sqrt(x0.size)
    result = root(
        Ffun,
        x0,
        method="krylov",
        options={"fatol": root_tol, "maxiter": params.maxiter, "disp": False},
    )

    if not result.success:
        print("警告：求解器未报告收敛。")
        print(result.message)

    k_path, c_path, y_path = paths_from_x(result.x, old_ss.k, new_ss.k, params)
    F_final = Ffun(result.x)

    return old_ss, new_ss, k_path, c_path, y_path, F_final


def plot_results(k_path, c_path, y_path, F_final, params):
    """绘制主要结果。

    参数：
        k_path：资本路径。
        c_path：消费路径。
        y_path：产出路径。
        F_final：最终 Euler 残差。
        params：模型参数命名空间。

    返回值：
        无。
    """

    T = params.T
    k_count = params.k_count
    time_k = np.arange(T + 1)
    time_c = np.arange(1, T + 1)
    residual_by_time = np.linalg.norm(F_final.reshape((k_count, T - 1), order="F"), axis=0)

    plt.figure(figsize=(10, 8))

    plt.subplot(2, 2, 1)
    plt.plot(time_k, k_path.T, linewidth=1.5)
    plt.xlabel("t")
    plt.ylabel("k_i")
    plt.title("Capital paths")
    plt.legend(["k_1", "k_2", "k_3"], loc="best")
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(time_c, c_path, linewidth=1.5)
    plt.xlabel("t")
    plt.ylabel("c")
    plt.title("Consumption path")
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(time_c, y_path, linewidth=1.5)
    plt.xlabel("t")
    plt.ylabel("y")
    plt.title("Output path")
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(time_c[:-1], residual_by_time, linewidth=1.5)
    plt.xlabel("t")
    plt.ylabel("Euler residual norm")
    plt.title("Euler residual by time")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


old_ss, new_ss, k_path, c_path, y_path, F_final = solve_model(p)

print("old steady state:")
print(old_ss)
print("new steady state:")
print(new_ss)
print(f"\nFinal residual norm = {np.linalg.norm(F_final):.4e}")
print(f"Minimum consumption = {np.min(c_path):.4e}")

plot_results(k_path, c_path, y_path, F_final, p)
