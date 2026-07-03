"""
复现与画图
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.io as sio
from scipy.stats import gaussian_kde


@dataclass(frozen=True)
class ProjectPaths:
    """保存项目路径。

    参数：
        self_dir: 当前脚本所在目录。

    返回值：
        dataclass 实例本身不返回额外对象；属性用于后续读取数据和写入输出。
    """

    self_dir: Path

    @property
    def project_root(self) -> Path:
        """返回 JIE2019 复刻项目根目录。"""
        return self.self_dir.parent

    @property
    def original_dir(self) -> Path:
        """返回作者原始复刻代码目录。"""
        return self.project_root / "原始复刻代码"

    @property
    def csv_dir(self) -> Path:
        """返回自我复刻 CSV 输出目录。"""
        return self.self_dir / "CSV_数据处理与校准输出"

    @property
    def output_dir(self) -> Path:
        """返回 Python 图形和校验输出目录。"""
        return self.self_dir / "Python_复刻输出"


def load_mat(path: Path) -> dict[str, object]:
    """读取 MATLAB v5 .mat 文件。

    参数：
        path: .mat 文件路径。

    返回值：
        去掉 MATLAB 内部元数据键后的字典。
    """
    raw = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    return {key: value for key, value in raw.items() if not key.startswith("__")}


def as_str_list(values: Iterable[object]) -> list[str]:
    """把 MATLAB 读入的字符串数组转为普通 Python 字符串列表。

    参数：
        values: MATLAB 字符串数组或可迭代对象。

    返回值：
        去除空白后的字符串列表。
    """
    return [str(value).strip() for value in values]


def max_relative_error(left: np.ndarray, right: np.ndarray) -> float:
    """计算最大相对误差。

    参数：
        left: 待比较数组。
        right: 基准数组。

    返回值：
        最大相对误差；分母用一个很小的正数截断，避免除零。
    """
    denominator = np.maximum(1e-12, np.abs(right))
    return float(np.nanmax(np.abs(left - right) / denominator))


def compare_vector(name: str, python_values: np.ndarray, matlab_values: np.ndarray) -> dict[str, float | str]:
    """比较一个 Python 向量和一个 MATLAB 向量。

    参数：
        name: 变量名称。
        python_values: Python 结果。
        matlab_values: MATLAB 结果。

    返回值：
        包含最大绝对误差、最大相对误差、中位比例和相关系数的字典。
    """
    left = np.asarray(python_values, dtype=float).reshape(-1)
    right = np.asarray(matlab_values, dtype=float).reshape(-1)
    mask = np.isfinite(left) & np.isfinite(right)
    ratio_denominator = np.where(np.abs(right[mask]) < 1e-12, np.nan, right[mask])
    ratio_values = left[mask] / ratio_denominator
    median_ratio = float(np.nanmedian(ratio_values)) if np.isfinite(ratio_values).any() else np.nan
    return {
        "variable": name,
        "max_abs_error": float(np.nanmax(np.abs(left[mask] - right[mask]))),
        "max_relative_error": max_relative_error(left[mask], right[mask]),
        "median_ratio": median_ratio,
        "correlation": float(np.corrcoef(left[mask], right[mask])[0, 1]) if mask.sum() > 1 else np.nan,
    }


def write_calibration_check(paths: ProjectPaths, parameters_mat: dict[str, object]) -> Path:
    """生成 Python CSV 与 MATLAB 参数文件的校验表。

    参数：
        paths: 项目路径。
        parameters_mat: 作者 ParametersSimple.mat 读取结果。

    返回值：
        校验表 CSV 路径。
    """
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    country = pd.read_csv(paths.csv_dir / "01_baseline_country_table.csv")
    parameter = pd.read_csv(paths.csv_dir / "03_baseline_parameter_table.csv")
    trade = pd.read_csv(paths.csv_dir / "02_baseline_trade_table.csv")
    isocode = as_str_list(parameters_mat["isocode"])

    country = country.set_index("country").loc[isocode].reset_index()
    parameter = parameter.set_index("country").loc[isocode].reset_index()

    checks: list[dict[str, float | str]] = []
    column_pairs = [
        ("GDP", "dat_GDP", country),
        ("Con", "dat_Con", country),
        ("GFCF", "dat_GFCF", country),
        ("GO_c", "dat_GO_c", country),
        ("GO_m", "dat_GO_m", country),
        ("GO_x", "dat_GO_x", country),
        ("VA_c", "dat_VA_c", country),
        ("VA_m", "dat_VA_m", country),
        ("VA_x", "dat_VA_x", country),
        ("ABS", "dat_ABS", country),
        ("export", "dat_EXP", country),
        ("import", "dat_IMP", country),
        ("emp", "dat_emp", country),
        ("pl_c", "dat_pl_c", country),
        ("pl_i", "dat_pl_i", country),
        ("P_m", "dat_P_m", country),
        ("T_m0", "T_m0", parameter),
        ("A_c0", "A_c0", parameter),
        ("A_x0", "A_x0", parameter),
        ("nu_m0", "nu_m0", parameter),
        ("nu_x0", "nu_x0", parameter),
        ("nu_c0", "nu_c0", parameter),
        ("L0", "L0", parameter),
        ("K0", "dat_K", parameter),
        ("F", "dat_F", parameter),
    ]
    for python_column, matlab_key, frame in column_pairs:
        checks.append(compare_vector(python_column, frame[python_column].to_numpy(), parameters_mat[matlab_key]))

    trade_share = (
        trade.pivot(index="destination", columns="origin", values="trade_share")
        .reindex(index=isocode, columns=isocode)
        .to_numpy(dtype=float)
    )
    trade_cost = (
        trade.pivot(index="destination", columns="origin", values="trade_cost")
        .reindex(index=isocode, columns=isocode)
        .to_numpy(dtype=float)
    )
    checks.append(compare_vector("trade_share", trade_share.ravel(), np.asarray(parameters_mat["dat_bts"]).ravel()))
    checks.append(compare_vector("trade_cost", trade_cost.ravel(), np.asarray(parameters_mat["d_m0"]).ravel()))

    output_path = paths.output_dir / "calibration_check.csv"
    pd.DataFrame(checks).to_csv(output_path, index=False, encoding="utf-8")
    return output_path


def execute_steady_state_notebook(paths: ProjectPaths) -> dict[str, object]:
    """执行稳态求解笔记本中定义和求解稳态的代码单元。

    参数：
        paths: 项目路径。

    返回值：
        笔记本执行后的命名空间，其中包含 ss、cp、p 等对象。
    """
    notebook_path = paths.self_dir / "稳态求解.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    namespace: dict[str, object] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for index, cell in enumerate(notebook["cells"]):
            if cell.get("cell_type") != "code" or index > 13:
                continue
            source = "".join(cell.get("source", []))
            if source.strip():
                exec(compile(source, f"{notebook_path.name}:cell{index}", "exec"), namespace)
    return namespace


def write_steady_state_alignment(paths: ProjectPaths, baseline_mat: dict[str, object]) -> Path:
    """生成 Python 稳态结果与 MATLAB 初始稳态结果的对齐表。

    参数：
        paths: 项目路径。
        baseline_mat: 作者 MergedBaselineResults.mat 读取结果。

    返回值：
        稳态对齐表 CSV 路径。
    """
    namespace = execute_steady_state_notebook(paths)
    steady_state = namespace["ss"]
    country_params = namespace["cp"]
    quantity = steady_state["quantity"]
    lib = 2
    alpha = np.full_like(np.asarray(country_params.L, dtype=float), float(namespace["p"].alpha), dtype=float)
    income = steady_state["wage"] / ((1 - alpha) * steady_state["P_c"])
    tfp = income / ((quantity.K / country_params.L) ** alpha)
    rel_price = steady_state["P_x"] / steady_state["P_c"]

    pairs = [
        ("wage", steady_state["wage"], get_column(baseline_mat, "w1_CfUniformLib", lib)),
        ("rent", steady_state["r"], get_column(baseline_mat, "r1_CfUniformLib", lib)),
        ("P_x", steady_state["P_x"], get_column(baseline_mat, "P_x1_CfUniformLib", lib)),
        ("P_m", steady_state["P_m"], get_column(baseline_mat, "P_m1_CfUniformLib", lib)),
        ("P_c", steady_state["P_c"], get_column(baseline_mat, "P_c1_CfUniformLib", lib)),
        ("K", quantity.K, get_column(baseline_mat, "K1_CfUniformLib", lib)),
        ("C", quantity.C, get_column(baseline_mat, "C1_CfUniformLib", lib)),
        ("X", quantity.X, get_column(baseline_mat, "X1_CfUniformLib", lib)),
        ("income_y", income, get_column(baseline_mat, "y1_CfUniformLib", lib)),
        ("TFP", tfp, get_column(baseline_mat, "TFP1_CfUniformLib", lib)),
        ("relative_price", rel_price, get_column(baseline_mat, "relP1_CfUniformLib", lib)),
        ("trade_share", steady_state["trade_share"].ravel(), np.asarray(baseline_mat["pi_m1_CfUniformLib"], dtype=float)[:, :, matlab_column(lib)].ravel()),
    ]
    output_path = paths.output_dir / "steady_state_alignment.csv"
    pd.DataFrame([compare_vector(name, left, right) for name, left, right in pairs]).to_csv(output_path, index=False, encoding="utf-8")
    return output_path


def matlab_column(lib: int) -> int:
    """把 MATLAB 的 lib 下标转为 Python 下标。

    参数：
        lib: MATLAB 代码中的 lib，例如 2 表示 20% 贸易成本下降。

    返回值：
        Python 零基列下标。
    """
    return lib - 1


def get_column(mat: dict[str, object], key: str, lib: int) -> np.ndarray:
    """读取 MATLAB 汇总结果中某个二维变量的指定 lib 列。

    参数：
        mat: MATLAB 汇总结果字典。
        key: 变量名。
        lib: MATLAB lib 下标。

    返回值：
        一维数组。
    """
    values = np.asarray(mat[key], dtype=float)
    return values[:, matlab_column(lib)]


def get_path(mat: dict[str, object], key: str, lib: int) -> np.ndarray:
    """读取 MATLAB 汇总结果中某个三维路径变量的指定 lib 切片。

    参数：
        mat: MATLAB 汇总结果字典。
        key: 变量名。
        lib: MATLAB lib 下标。

    返回值：
        形状为“国家数乘时期数”的二维数组。
    """
    values = np.asarray(mat[key], dtype=float)
    return values[:, :, matlab_column(lib)]


def save_current_figure(paths: ProjectPaths, name: str, records: list[dict[str, str]]) -> None:
    """保存当前图形并记录清单。

    参数：
        paths: 项目路径。
        name: 不含扩展名的图名。
        records: 图形清单列表，函数会向其中追加记录。

    返回值：
        无。
    """
    path = paths.output_dir / "figures" / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    records.append({"figure": name, "path": str(path)})


def kde_line(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """计算核密度曲线。

    参数：
        values: 一维数据。

    返回值：
        横轴网格和密度值。
    """
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    grid = np.linspace(clean.min(), clean.max(), 300)
    density = gaussian_kde(clean)(grid)
    return grid, density


def add_country_labels(x_values: np.ndarray, y_values: np.ndarray, labels: list[str], highlight: set[str] | None = None) -> None:
    """在散点图中添加国家标签。

    参数：
        x_values: 横轴数值。
        y_values: 纵轴数值。
        labels: 国家代码。
        highlight: 需要突出显示的国家代码集合。

    返回值：
        无。
    """
    highlight = highlight or set()
    for x_value, y_value, label in zip(x_values, y_values, labels):
        color = "tab:red" if label in highlight else "black"
        plt.text(x_value, y_value, label, ha="center", va="center", fontsize=8, color=color)


def plot_transition(
    paths: ProjectPaths,
    name: str,
    data: np.ndarray,
    baseline: np.ndarray,
    cty_indices: list[int],
    labels: list[str],
    records: list[dict[str, str]],
    x_max: int = 50,
) -> None:
    """绘制四个国家的过渡路径图。

    参数：
        paths: 项目路径。
        name: 图形文件名。
        data: 形状为“国家数乘时期数”的路径数组。
        baseline: 初始稳态一维数组。
        cty_indices: 要绘制的国家下标。
        labels: 图例标签。
        records: 图形清单。
        x_max: 横轴显示最大年份。

    返回值：
        无。
    """
    years = np.arange(data.shape[1])
    plt.figure(figsize=(7, 4.2))
    for cty_index, label in zip(cty_indices, labels):
        series = data[cty_index, :] / baseline[cty_index]
        sns.lineplot(x=years, y=series, label=label)
    plt.xlim(0, x_max)
    plt.xlabel("Year")
    plt.ylabel("Ratio to initial steady state")
    save_current_figure(paths, name, records)


@dataclass
class PlotContext:
    """保存绘图所需的共享上下文。

    参数：
        paths: 项目路径。
        baseline_mat: 基准 MATLAB 汇总结果。
        static_mat: 静态模型 MATLAB 汇总结果。
        records: 图形清单。
        isocode: 国家代码。
        highlight: 重点标注国家。
        cty_indices: 重点国家下标。
        lib: MATLAB lib 下标，2 对应 20% 贸易成本下降。
        real_income: 实际收入指标。

    返回值：
        dataclass 实例本身不返回额外对象。
    """

    paths: ProjectPaths
    baseline_mat: dict[str, object]
    static_mat: dict[str, object]
    records: list[dict[str, str]]
    isocode: list[str]
    highlight: set[str]
    cty_indices: list[int]
    lib: int
    real_income: np.ndarray


def build_plot_context(paths: ProjectPaths, baseline_mat: dict[str, object], static_mat: dict[str, object]) -> PlotContext:
    """构造绘图共享上下文。

    参数：
        paths: 项目路径。
        baseline_mat: 基准 MATLAB 汇总结果。
        static_mat: 静态模型 MATLAB 汇总结果。

    返回值：
        PlotContext 实例。
    """
    isocode = as_str_list(baseline_mat["isocode"])
    real_income = (
        (1 - np.asarray(baseline_mat["alpha1"], dtype=float))
        * np.asarray(baseline_mat["dat_GDP"], dtype=float)
        / np.asarray(baseline_mat["dat_pl_c"], dtype=float)
        * 1e-6
    )
    return PlotContext(
        paths=paths,
        baseline_mat=baseline_mat,
        static_mat=static_mat,
        records=[],
        isocode=isocode,
        highlight={"BGR", "PRT", "FRA", "USA"},
        cty_indices=[isocode.index(code) for code in ["BGR", "PRT", "FRA", "USA"]],
        lib=2,
        real_income=real_income,
    )


def plot_distribution_and_scatter(ctx: PlotContext) -> None:
    """绘制份额分布和核心散点图。

    参数：
        ctx: 绘图上下文。

    返回值：
        无。
    """
    mat = ctx.baseline_mat
    plt.figure(figsize=(7, 4.2))
    for values, label in [
        (1 - np.asarray(mat["nu_c1"], dtype=float), "1-nu_c"),
        (1 - np.asarray(mat["nu_x1"], dtype=float), "1-nu_x"),
        (1 - np.asarray(mat["nu_m1"], dtype=float), "1-nu_m"),
    ]:
        grid, density = kde_line(values)
        sns.lineplot(x=grid, y=density / len(values) * 100, label=label)
    plt.xlabel("Tradables intensity")
    plt.ylabel("Frequency (percent)")
    save_current_figure(ctx.paths, "VAshares_density_cty44", ctx.records)

    plt.figure(figsize=(7, 4.2))
    add_country_labels(ctx.real_income / 1000, get_column(mat, "LamDyn_CfUniformLib", ctx.lib), ctx.isocode, ctx.highlight)
    plt.xscale("log")
    plt.xlabel("Total real GDP, billions U.S. dollars")
    plt.ylabel("Percent")
    save_current_figure(ctx.paths, "DynGain20pct_vs_GDP", ctx.records)


def current_account_first_period(ctx: PlotContext) -> np.ndarray:
    """计算第一期经常账户占 GDP 比例。

    参数：
        ctx: 绘图上下文。

    返回值：
        第一时期经常账户占 GDP 比例。
    """
    mat = ctx.baseline_mat
    return get_path(mat, "B_CfUniformLib", ctx.lib)[:, 0] / (
        get_path(mat, "w_CfUniformLib", ctx.lib)[:, 0]
        * np.asarray(mat["L1"], dtype=float)
        / (1 - np.asarray(mat["alpha1"], dtype=float))
    )


def plot_transition_figures(ctx: PlotContext) -> np.ndarray:
    """绘制过渡路径图。

    参数：
        ctx: 绘图上下文。

    返回值：
        第一时期经常账户占 GDP 比例，供后续图形复用。
    """
    mat = ctx.baseline_mat
    labels = ["BGR", "PRT", "FRA", "USA"]
    plot_transition(ctx.paths, "TransC_Uniform20pct_4cty", get_path(mat, "C_CfUniformLib", ctx.lib), get_column(mat, "C1_CfUniformLib", ctx.lib), ctx.cty_indices, labels, ctx.records)

    ca_path = get_path(mat, "B_CfUniformLib", ctx.lib) / (
        get_path(mat, "w_CfUniformLib", ctx.lib)
        * np.asarray(mat["L"], dtype=float)
        / (1 - np.asarray(mat["alpha"], dtype=float))
    )
    plt.figure(figsize=(7, 4.2))
    years = np.arange(ca_path.shape[1])
    for cty_index, label in zip(ctx.cty_indices, labels):
        sns.lineplot(x=years, y=ca_path[cty_index, :], label=label)
    plt.xlim(0, 50)
    plt.xlabel("Year")
    plt.ylabel("Current account to GDP")
    save_current_figure(ctx.paths, "TransCAoverGDP_Uniform20pct_4cty", ctx.records)

    ca_first = current_account_first_period(ctx)
    plt.figure(figsize=(7, 4.2))
    add_country_labels(ctx.real_income / 1000, ca_first, ctx.isocode, ctx.highlight)
    plt.axhline(0, color="black", linewidth=1, linestyle=":")
    plt.xscale("log")
    plt.xlabel("Total real GDP, billions U.S. dollars")
    plt.ylabel("Current account to GDP, period 1")
    save_current_figure(ctx.paths, "CAoverGDPper1_vs_GDP", ctx.records)

    plot_transition(ctx.paths, "TransTFP_Uniform20pct_4cty", get_path(mat, "TFP_CfUniformLib", ctx.lib), get_column(mat, "TFP1_CfUniformLib", ctx.lib), ctx.cty_indices, labels, ctx.records)
    plot_transition(ctx.paths, "TransRelP_Uniform20pct_4cty", get_path(mat, "relP_CfUniformLib", ctx.lib), get_column(mat, "relP1_CfUniformLib", ctx.lib), ctx.cty_indices, labels, ctx.records)
    plot_transition(ctx.paths, "Transy_Uniform20pct_4cty", get_path(mat, "y_CfUniformLib", ctx.lib), get_column(mat, "y1_CfUniformLib", ctx.lib), ctx.cty_indices, labels, ctx.records)
    plot_transition(ctx.paths, "TransK_Uniform20pct_4cty", get_path(mat, "K_CfUniformLib", ctx.lib)[:, :150], get_column(mat, "K1_CfUniformLib", ctx.lib), ctx.cty_indices, labels, ctx.records, x_max=50)
    return ca_first


def plot_gain_mechanism_figures(ctx: PlotContext, ca_first: np.ndarray) -> None:
    """绘制动态收益机制相关图。

    参数：
        ctx: 绘图上下文。
        ca_first: 第一时期经常账户占 GDP 比例。

    返回值：
        无。
    """
    mat = ctx.baseline_mat
    plt.figure(figsize=(7, 4.2))
    add_country_labels(ca_first, get_column(mat, "halflife_CfUniformLib", ctx.lib), ctx.isocode, ctx.highlight)
    plt.axvline(0, color="black", linewidth=1, linestyle=":")
    plt.xlabel("Ratio of current account to GDP")
    plt.ylabel("Half-life")
    save_current_figure(ctx.paths, "HalfLifeK20pct_vs_GDP", ctx.records)

    plt.figure(figsize=(7, 4.2))
    for lib_value in [2, 4, 6, 8]:
        values = -get_column(mat, "elast_CfUniformLib", lib_value)
        for cty_index in ctx.cty_indices:
            plt.text(lib_value * 10, values[cty_index], ctx.isocode[cty_index], fontsize=8, ha="center")
    plt.xlim(0, 100)
    plt.xlabel("Percent reduction in trade costs")
    plt.ylabel("Elasticity of dynamic gains")
    save_current_figure(ctx.paths, "ElastDynGain4cty_vs_lib20406080100", ctx.records)

    plt.figure(figsize=(7, 4.2))
    add_country_labels(get_column(mat, "LamDyn_CfUniformLib", ctx.lib), np.asarray(mat["LamDyn_UnilatMulticty"], dtype=float), ctx.isocode)
    plt.plot([-3, 40], [-3, 40], color="black")
    plt.xlabel("Baseline gains (percent)")
    plt.ylabel("Unilateral gains (percent)")
    save_current_figure(ctx.paths, "Gains_Uniform_vs_Unilat", ctx.records)

    initial_nfa = get_column(mat, "A1_CfUniformLib", ctx.lib) / (
        get_column(mat, "w1_CfUniformLib", ctx.lib)
        * np.asarray(mat["L1"], dtype=float)
        / (1 - np.asarray(mat["alpha1"], dtype=float))
    )
    immediate_gain = get_path(mat, "y_CfUniformLib", ctx.lib)[:, 0] / get_column(mat, "y1_CfUniformLib", ctx.lib) - 1
    dynamic_relative = get_column(mat, "LamDyn_CfUniformLib", ctx.lib) / immediate_gain / 100
    plt.figure(figsize=(7, 4.2))
    add_country_labels(initial_nfa, dynamic_relative, ctx.isocode, ctx.highlight)
    plt.axvline(0, color="black", linewidth=1, linestyle=":")
    plt.xlabel("Ratio of initial net foreign assets to GDP")
    plt.ylabel("Dynamic gains relative to immediate gains")
    save_current_figure(ctx.paths, "Gains_DynRelToInsty_vs_NFA", ctx.records)


def plot_nu_sensitivity_figures(ctx: PlotContext) -> None:
    """绘制 nu_c 与 nu_x 替换实验图。

    参数：
        ctx: 绘图上下文。

    返回值：
        无。
    """
    mat = ctx.baseline_mat
    for country_code, suffix in [("BGR", "BGR"), ("USA", "USA")]:
        cty_index = ctx.isocode.index(country_code)
        years = np.arange(get_path(mat, "TFP_CfUniformLib", ctx.lib).shape[1])
        plt.figure(figsize=(7, 4.2))
        sns.lineplot(x=years, y=get_path(mat, "TFP_CfUniformLib", ctx.lib)[cty_index, :] / get_column(mat, "TFP1_CfUniformLib", ctx.lib)[cty_index], label="Baseline")
        sns.lineplot(x=years, y=np.asarray(mat["TFP_CfUniformLibNuCeqNuX"], dtype=float)[cty_index, :] / np.asarray(mat["TFP1_CfUniformLibNuCeqNuX"], dtype=float)[cty_index], label="Fix nu_x")
        sns.lineplot(x=years, y=np.asarray(mat["TFP_CfUniformLibNuXeqNuC"], dtype=float)[cty_index, :] / np.asarray(mat["TFP1_CfUniformLibNuXeqNuC"], dtype=float)[cty_index], label="Fix nu_c")
        plt.xlim(0, 60)
        plt.xlabel("Year")
        plt.ylabel("TFP ratio")
        save_current_figure(ctx.paths, f"TFP_SameNuXandNuC_{suffix}", ctx.records)

        years_k = np.arange(np.asarray(mat["K_CfUniformLibNuCeqNuX"], dtype=float).shape[1])
        plt.figure(figsize=(7, 4.2))
        sns.lineplot(x=years_k, y=get_path(mat, "K_CfUniformLib", ctx.lib)[cty_index, :] / get_column(mat, "K1_CfUniformLib", ctx.lib)[cty_index], label="Baseline")
        sns.lineplot(x=years_k, y=np.asarray(mat["K_CfUniformLibNuCeqNuX"], dtype=float)[cty_index, :] / np.asarray(mat["K1_CfUniformLibNuCeqNuX"], dtype=float)[cty_index], label="Fix nu_x")
        sns.lineplot(x=years_k, y=np.asarray(mat["K_CfUniformLibNuXeqNuC"], dtype=float)[cty_index, :] / np.asarray(mat["K1_CfUniformLibNuXeqNuC"], dtype=float)[cty_index], label="Fix nu_c")
        plt.xlim(0, 120)
        plt.xlabel("Year")
        plt.ylabel("Capital ratio")
        save_current_figure(ctx.paths, f"K_SameNuXandNuC_{suffix}", ctx.records)


def plot_appendix_figures(ctx: PlotContext) -> None:
    """绘制附录和补充图。

    参数：
        ctx: 绘图上下文。

    返回值：
        无。
    """
    mat = ctx.baseline_mat
    plt.figure(figsize=(7, 4.2))
    sns.regplot(x=ctx.real_income / 1000, y=np.asarray(mat["nu_c1"], dtype=float), scatter=True, label="Baseline model", truncate=False)
    sns.regplot(x=ctx.real_income / 1000, y=np.asarray(ctx.static_mat["nu_c1"], dtype=float), scatter=True, label="Static model", truncate=False)
    plt.xscale("log")
    plt.xlabel("Total real GDP, billions U.S. dollars")
    plt.ylabel("nu_c")
    plt.legend()
    save_current_figure(ctx.paths, "nu_c_dynamicAndStatic_cty44", ctx.records)

    plt.figure(figsize=(7, 4.2))
    nominal_investment_rate = 100 * np.asarray(mat["dat_GFCF"], dtype=float) / np.asarray(mat["dat_GDP"], dtype=float)
    grid, density = kde_line(nominal_investment_rate)
    sns.lineplot(x=grid, y=density / len(nominal_investment_rate) * 100, label="Data")
    plt.xlabel("Nominal investment rate (percent)")
    plt.ylabel("Frequency (percent)")
    save_current_figure(ctx.paths, "NomInvRate_density_cty44", ctx.records)

    plt.figure(figsize=(7, 4.2))
    openness = 1 / np.diag(np.asarray(mat["dat_bts"], dtype=float))
    add_country_labels(ctx.real_income / 1000, openness, ctx.isocode)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Total real GDP, billions U.S. dollars")
    plt.ylabel("Openness: 1/pi_ii")
    save_current_figure(ctx.paths, "Openness_vs_GDP_cty44", ctx.records)


def plot_figures(paths: ProjectPaths, baseline_mat: dict[str, object], static_mat: dict[str, object]) -> Path:
    """用 seaborn 复现 MATLAB PlotResults.m 的主要图形。

    参数：
        paths: 项目路径。
        baseline_mat: MergedBaselineResults.mat 读取结果。
        static_mat: MergedStaticXinCResults.mat 读取结果。

    返回值：
        图形清单 CSV 路径。
    """
    sns.set_theme(style="whitegrid")
    ctx = build_plot_context(paths, baseline_mat, static_mat)
    plot_distribution_and_scatter(ctx)
    ca_first = plot_transition_figures(ctx)
    plot_gain_mechanism_figures(ctx, ca_first)
    plot_nu_sensitivity_figures(ctx)
    plot_appendix_figures(ctx)
    manifest_path = paths.output_dir / "python_figure_manifest.csv"
    pd.DataFrame(ctx.records).to_csv(manifest_path, index=False, encoding="utf-8")
    return manifest_path


def main() -> None:
    """执行校验和绘图。

    参数：
        无。

    返回值：
        无。结果写入 Python_复刻输出。
    """
    paths = ProjectPaths(Path(__file__).resolve().parent)
    parameters_mat = load_mat(paths.original_dir / "Data" / "Calibration" / "ParametersSimple.mat")
    baseline_mat = load_mat(paths.original_dir / "MatlabProgramsSimple" / "BaselineModel" / "Results" / "MergedBaselineResults.mat")
    static_mat = load_mat(paths.original_dir / "MatlabProgramsSimple" / "ModelInvestmentInC" / "Results" / "MergedStaticXinCResults.mat")

    calibration_path = write_calibration_check(paths, parameters_mat)
    steady_state_path = write_steady_state_alignment(paths, baseline_mat)
    manifest_path = plot_figures(paths, baseline_mat, static_mat)
    print(f"校验表已写入：{calibration_path}")
    print(f"稳态对齐表已写入：{steady_state_path}")
    print(f"图形清单已写入：{manifest_path}")


if __name__ == "__main__":
    main()
