"""
预测脚本：加载训练好的 DeepONet，对指定工况和时刻进行 PSD 预测。

用法示例:
    python predict.py                          # 使用默认参数
    python predict.py --sheet CR_2_50 --times 1800 3600 5400 7200
    python predict.py --sheet CR_1_00 --times 900 2700 5400 8100 10800

自定义输入（不依赖 sheet）的用法示例:
    # 1) 预测一个点 (L,t)，并给出加权聚合（只有一个点时即该点本身）
    python predict.py --custom_json inputs/custom_case.json --points 50,3600 --point_weights 1

    # 2) 预测多个点 (L,t)，并按权重输出加权和/加权平均
    python predict.py --custom_json inputs/custom_case.json ^
        --points 10,900 30,3600 60,7200 ^
        --point_weights 0.2 0.3 0.5

支持的功能:
    1. 指定任意工况（sheet 名称）进行预测
    2. 指定任意时刻（不必是训练快照时刻）
    3. 自动与仿真真值对比（若该时刻有快照数据）
    4. 输出完整 PSD 曲线图 + 误差统计
    5. 导出预测结果为 CSV 文件

    6. 自定义输入：给定初始粒度分布、温度分布、初始浓度，
       对任意一个或多个 (L,t) 坐标预测，并按指定权重输出聚合结果
"""

import os
import argparse
import json
from typing import Dict, Any, List, Tuple, Optional

from deeponet_pbe.gpu_config import setup_gpu
setup_gpu()

import numpy as np
import pandas as pd
import tensorflow as tf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from deeponet_pbe.model import DeepONet


# ======================================================================
# 工具函数
# ======================================================================

def normalize_weights_prefix(weights_path: str) -> str:
    """把 ``--weights`` 转成 TensorFlow 期望的权重「前缀」路径。"""
    p = os.path.normpath(weights_path.strip())
    if os.path.isdir(p):
        p = os.path.join(p, "deeponet")
    if p.lower().endswith(".index"):
        p = p[: -len(".index")]
    return p


def load_model_and_params(results_dir, weights_path=None):
    """加载模型权重和归一化参数。

    Parameters
    ----------
    results_dir : str
        训练结果目录（含 norm_params.npz）。
    weights_path : str | None
        自定义权重路径。None 则使用 results_dir/model/deeponet。
        示例: results/model/ckpt_epoch_0150/deeponet
    """
    params = np.load(
        os.path.join(results_dir, "norm_params.npz"), allow_pickle=True
    )

    branch_dim = int(params["branch_dim"])
    branch_hiddens = list(params["branch_hiddens"])
    trunk_hiddens = list(params["trunk_hiddens"])
    latent_dim = int(params["latent_dim"])

    model = DeepONet(
        branch_input_dim=branch_dim,
        trunk_input_dim=2,
        branch_hiddens=branch_hiddens,
        trunk_hiddens=trunk_hiddens,
        latent_dim=latent_dim,
    )
    dummy_b = tf.zeros((1, branch_dim))
    dummy_t = tf.zeros((1, 2))
    _ = model([dummy_b, dummy_t])

    if weights_path:
        ckpt_path = normalize_weights_prefix(weights_path)
    else:
        # 兼容历史目录结构：优先 results_dir/weights/best/deeponet（train.py 默认）
        cand1 = os.path.join(results_dir, "weights", "best", "deeponet")
        cand2 = os.path.join(results_dir, "model", "deeponet")
        ckpt_path = cand1 if os.path.exists(cand1 + ".index") else cand2

    if not os.path.exists(ckpt_path + ".index"):
        raise FileNotFoundError(
            f"找不到权重: {ckpt_path}.index\n"
            f"可用写法: --weights ...\\weights\\best\\deeponet   或   --weights ...\\weights\\best   或   --weights ...\\deeponet.index"
        )
    model.load_weights(ckpt_path)
    print(f"Model loaded from {ckpt_path}")

    return model, params


def load_experiment(data_path, sheet_name):
    """从 .mat 或 .xlsx 加载单个工况的原始数据。"""
    ext = os.path.splitext(data_path)[1].lower()

    if ext == ".mat":
        import h5py
        f = h5py.File(data_path, "r")
        root = f["Dataset"] if "Dataset" in f else f
        grp = root[sheet_name]
        record = {}
        record["Time_s"] = np.asarray(grp["Time_s"]).flatten()
        record["Temp_K"] = np.asarray(grp["Temp_K"]).flatten()
        record["Conc"] = np.asarray(grp["Conc"]).flatten()
        record["L_mid_um"] = np.asarray(grp["L_mid_um"]).flatten()
        psd_raw = np.asarray(grp["psd"], dtype=np.float64)
        if psd_raw.shape[0] == len(record["L_mid_um"]):
            psd_raw = psd_raw.T
        record["psd"] = psd_raw
        record["snapshot_times"] = np.asarray(grp["snapshot_times"]).flatten()
        record["C0"] = float(np.asarray(grp["C0"]).flat[0])
        record["n_L0"] = np.asarray(grp["n_L0"]).flatten()
        f.close()
    else:
        df = pd.read_excel(data_path, sheet_name=sheet_name)
        record = {}
        record["Time_s"] = df["Time_s"].values
        record["Temp_K"] = df["Temp_K"].values
        record["Conc"] = df["Conc"].values
        L_mid = df["L_mid_um"].dropna().values
        record["L_mid_um"] = L_mid
        n_L = len(L_mid)
        psd_cols = [
            c for c in df.columns
            if c.startswith("Time_") and c != "Time_s"
        ]
        snapshot_times, psd_snapshots = [], []
        for col in psd_cols:
            t_val = float(col.replace("Time_", "").replace("s", ""))
            snapshot_times.append(t_val)
            psd_snapshots.append(df[col].iloc[:n_L].values)
        record["snapshot_times"] = np.array(snapshot_times)
        record["psd"] = np.array(psd_snapshots)
        record["C0"] = record["Conc"][0]
        record["n_L0"] = record["psd"][0]

    return record


def build_branch_vector(record, params):
    """构建 Branch 输入向量（与训练时一致）。"""
    T_min = float(params["T_min"])
    T_max = float(params["T_max"])
    C_min = float(params["C_min"])
    C_max = float(params["C_max"])
    n_scale = float(params["n_scale"])
    snapshot_times = params["snapshot_times"]
    L_sensor_idx = params["L_sensor_idx"].astype(int)

    dt = record["Time_s"][1] - record["Time_s"][0]
    T_at_snapshots = []
    for t in snapshot_times:
        idx = min(int(t / dt), len(record["Temp_K"]) - 1)
        T_at_snapshots.append(record["Temp_K"][idx])
    T_sensors = (np.array(T_at_snapshots) - T_min) / (T_max - T_min + 1e-12)

    n_L0_sampled = record["n_L0"][L_sensor_idx]
    n_L0_norm = n_L0_sampled / n_scale

    C0_norm = (record["C0"] - C_min) / (C_max - C_min + 1e-12)

    return np.concatenate([T_sensors, n_L0_norm, [C0_norm]]).astype(np.float32)


def predict_psd(model, branch_vec, L_eval, t_query, params):
    """预测单个时刻的 PSD。

    Parameters
    ----------
    model : 训练好的 DeepONet
    branch_vec : (branch_dim,) Branch 输入
    L_eval : (n_L,) 评估粒径点 (μm)
    t_query : float 查询时刻 (s)
    params : 归一化参数

    Returns
    -------
    psd_pred : (n_L,) 预测的 n(L,t) (真实尺度)
    """
    L_max = float(params["L_max"])
    t_max = float(params["t_max"])
    n_scale = float(params["n_scale"])

    n_L = len(L_eval)
    L_norm = L_eval / L_max
    t_norm = t_query / t_max

    trunk_batch = np.stack(
        [L_norm, np.full(n_L, t_norm)], axis=-1
    ).astype(np.float32)
    branch_batch = np.tile(branch_vec, (n_L, 1))

    pred_norm = model([branch_batch, trunk_batch], training=False).numpy().flatten()
    return pred_norm * n_scale


def predict_points(
    model,
    branch_vec: np.ndarray,
    points_Lt: np.ndarray,
    params,
) -> np.ndarray:
    """预测一个或多个 (L,t) 点的 n(L,t)（真实尺度）。

    Parameters
    ----------
    points_Lt : (N,2) 其中 [:,0]=L(um), [:,1]=t(s)
    """
    L_max = float(params["L_max"])
    t_max = float(params["t_max"])
    n_scale = float(params["n_scale"])

    pts = np.asarray(points_Lt, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points_Lt 必须是形状 (N,2) 的数组，列为 [L_um, t_s]")

    L_norm = (pts[:, 0] / L_max).astype(np.float32)
    t_norm = (pts[:, 1] / t_max).astype(np.float32)
    trunk_batch = np.stack([L_norm, t_norm], axis=-1).astype(np.float32)
    branch_batch = np.tile(branch_vec.astype(np.float32), (len(pts), 1))

    pred_norm = model([branch_batch, trunk_batch], training=False).numpy().reshape(-1)
    return pred_norm * n_scale


def get_true_psd(record, L_eval_idx, t_query):
    """获取仿真真值 PSD（仅当 t_query 恰好是快照时刻时可用）。

    Returns
    -------
    true_psd : ndarray | None
    """
    snap_times = record["snapshot_times"]
    tol = 1.0
    match = np.where(np.abs(snap_times - t_query) < tol)[0]
    if len(match) > 0:
        idx = match[0]
        return record["psd"][idx][L_eval_idx]
    return None


def _parse_points(points_tokens: List[str]) -> np.ndarray:
    """解析 CLI 里的 `L,t` 列表为 (N,2) 数组。"""
    pts: List[Tuple[float, float]] = []
    for tok in points_tokens:
        s = (tok or "").strip()
        if not s:
            continue
        if "," not in s:
            raise ValueError(f"points 参数格式应为 'L,t'，但收到: {tok}")
        a, b = s.split(",", 1)
        pts.append((float(a), float(b)))
    if not pts:
        raise ValueError("至少需要提供一个 --points L,t")
    return np.array(pts, dtype=np.float64)


def _parse_weights(weights: Optional[List[float]], n: int) -> np.ndarray:
    if weights is None or len(weights) == 0:
        return np.ones((n,), dtype=np.float64)
    if len(weights) != n:
        raise ValueError(f"--weights 数量必须与 --points 数量一致：points={n}, weights={len(weights)}")
    w = np.asarray(weights, dtype=np.float64)
    return w


def _weighted_aggregate(y: np.ndarray, w: np.ndarray) -> Dict[str, float]:
    """返回加权和与加权平均（w 全零时平均返回 NaN）。"""
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    s = float(np.sum(w * y))
    ws = float(np.sum(w))
    mean = float(s / ws) if abs(ws) > 1e-15 else float("nan")
    return {"weighted_sum": s, "weighted_mean": mean, "weight_sum": ws}


def _load_custom_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError("custom_json 顶层必须是 JSON 对象")
    return obj


def build_branch_vector_custom(custom: Dict[str, Any], params) -> np.ndarray:
    """从自定义输入构建 Branch 向量。

    custom_json 支持字段（二选一/可选）：
    - **T_at_snapshots**: 长度 = len(snapshot_times) 的温度数组 (K)
      或
    - **Time_s** + **Temp_K**: 原始时间网格与温度曲线（用训练 snapshot_times 采样）

    - **n_L0_sensors**: 长度 = n_L_sensors 的初始 PSD（已对齐训练的 L_sensor_idx 下采样）
      （若你手头是“全 L 网格”的 PSD，请在外部先下采样成 sensors 维度，以免因缺少训练 L_full 导致对齐歧义）

    - **C0**: 初始浓度标量
    """
    T_min = float(params["T_min"])
    T_max = float(params["T_max"])
    C_min = float(params["C_min"])
    C_max = float(params["C_max"])
    n_scale = float(params["n_scale"])
    snapshot_times = np.array(params["snapshot_times"], dtype=np.float64).reshape(-1)

    # ---- 温度：优先用 T_at_snapshots ----
    if "T_at_snapshots" in custom and custom["T_at_snapshots"] is not None:
        T_at_snapshots = np.asarray(custom["T_at_snapshots"], dtype=np.float64).reshape(-1)
        if len(T_at_snapshots) != len(snapshot_times):
            raise ValueError(
                f"T_at_snapshots 长度必须等于 snapshot_times 长度：{len(snapshot_times)}，但收到 {len(T_at_snapshots)}"
            )
    else:
        # Time_s + Temp_K
        if "Time_s" not in custom or "Temp_K" not in custom:
            raise ValueError("custom_json 必须提供 T_at_snapshots 或 (Time_s + Temp_K)")
        time_s = np.asarray(custom["Time_s"], dtype=np.float64).reshape(-1)
        temp_k = np.asarray(custom["Temp_K"], dtype=np.float64).reshape(-1)
        if len(time_s) != len(temp_k) or len(time_s) < 2:
            raise ValueError("Time_s 与 Temp_K 必须等长且至少 2 个点")

        # 用“最接近的时间索引”在 snapshot_times 上采样（兼容非等间隔/非严格 dt）
        T_at_snapshots = []
        for t in snapshot_times:
            idx = int(np.argmin(np.abs(time_s - t)))
            T_at_snapshots.append(float(temp_k[idx]))
        T_at_snapshots = np.asarray(T_at_snapshots, dtype=np.float64)

    T_sensors = (T_at_snapshots - T_min) / (T_max - T_min + 1e-12)

    # ---- 初始 PSD（要求已是 sensors 下采样）----
    if "n_L0_sensors" not in custom:
        raise ValueError("custom_json 必须提供 n_L0_sensors（长度 = 训练用 n_L_sensors）")
    n_L0_sensors = np.asarray(custom["n_L0_sensors"], dtype=np.float64).reshape(-1)
    n_L_sensors_expected = int(params["branch_dim"]) - len(snapshot_times) - 1
    if len(n_L0_sensors) != n_L_sensors_expected:
        raise ValueError(
            f"n_L0_sensors 长度应为 {n_L_sensors_expected}，但收到 {len(n_L0_sensors)}"
        )
    n_L0_norm = n_L0_sensors / (n_scale + 1e-30)

    # ---- 初始浓度 ----
    if "C0" not in custom:
        raise ValueError("custom_json 必须提供 C0")
    C0 = float(custom["C0"])
    C0_norm = (C0 - C_min) / (C_max - C_min + 1e-12)

    return np.concatenate([T_sensors, n_L0_norm, [C0_norm]]).astype(np.float32)


# ======================================================================
# 可视化
# ======================================================================

def plot_single_prediction(L_eval, pred, true, t_query, sheet_name, save_path):
    """绘制单时刻 PSD 预测 vs 真值。"""
    plt.figure(figsize=(9, 5))
    if true is not None:
        plt.plot(L_eval, true, "b-", linewidth=1.5, label="Simulation (FVM)")
    plt.plot(L_eval, pred, "r--", linewidth=1.5, label="DeepONet Prediction")
    plt.xlabel("$L$ (μm)")
    plt.ylabel("$n(L,t)$")
    plt.title(f"PSD Prediction at t = {t_query:.0f}s  [{sheet_name}]")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Plot saved: {save_path}")


def plot_multi_time_prediction(L_eval, predictions, times, sheet_name, save_path):
    """绘制多时刻 PSD 预测叠加图。"""
    plt.figure(figsize=(10, 6))
    cmap = plt.cm.viridis
    n = len(times)
    for i, (t, pred) in enumerate(zip(times, predictions)):
        color = cmap(i / max(n - 1, 1))
        plt.plot(L_eval, pred, color=color, linewidth=1.2,
                 label=f"t={t:.0f}s")
    plt.xlabel("$L$ (μm)")
    plt.ylabel("$n(L,t)$")
    plt.title(f"Predicted PSD Evolution [{sheet_name}]")
    plt.legend(fontsize=8, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Evolution plot saved: {save_path}")


# ======================================================================
# 主流程
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="DeepONet PBE Prediction")
    parser.add_argument(
        "--data", type=str, default=None,
        help="数据文件路径（.mat 或 .xlsx）。不填则默认当前目录下 Simulation_Data_DeepONet.mat",
    )
    parser.add_argument(
        "--sheet", type=str, default="CR_2_50",
        help="工况名称（Excel sheet 名），默认 CR_2_50",
    )
    parser.add_argument(
        "--times", type=float, nargs="+",
        default=[900, 1800, 3600, 5400, 7200, 9000, 10800],
        help="预测时刻列表 (秒)，默认 [900,1800,3600,5400,7200,9000,10800]",
    )
    parser.add_argument(
        "--results_dir", type=str, default=None,
        help="训练结果目录 (含 model/ 和 norm_params.npz)，默认 results/",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="预测输出目录，默认 predictions/",
    )
    parser.add_argument(
        "--weights", type=str, default=None,
        help="模型权重路径：目录（.../weights/best）或前缀（.../deeponet），也可带 .index 后缀",
    )

    # ---------- 自定义输入模式 ----------
    parser.add_argument(
        "--custom_json", type=str, default=None,
        help="自定义输入 JSON（提供初始 PSD、温度曲线、初始浓度），启用后忽略 --sheet/--times 的 PSD 曲线预测流程",
    )
    parser.add_argument(
        "--points", type=str, nargs="+", default=None,
        help="一个或多个查询点，格式 'L,t'（单位: um, s），例如 --points 50,3600 80,7200",
    )
    parser.add_argument(
        "--point_weights", type=float, nargs="*", default=None,
        help="与 --points 对应的权重列表（不填则默认全 1）。将输出加权和与加权平均。",
    )
    args = parser.parse_args()

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    RESULTS_DIR = args.results_dir or os.path.join(SCRIPT_DIR, "results")
    OUTPUT_DIR = args.output_dir or os.path.join(SCRIPT_DIR, "predictions")
    DATA_PATH = args.data or os.path.join(SCRIPT_DIR, "Simulation_Data_DeepONet.mat")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. 加载模型
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Loading trained model...")
    model, params = load_model_and_params(RESULTS_DIR, weights_path=args.weights)

    L_eval = params["L_eval"]
    L_eval_idx = params["L_eval_idx"].astype(int)
    n_L = len(L_eval)

    # ==================================================================
    # A) 自定义输入：预测任意 (L,t) 点（一个或多个）+ 权重聚合
    # ==================================================================
    if args.custom_json:
        if not args.points:
            raise ValueError("使用 --custom_json 时必须提供 --points")

        custom = _load_custom_json(args.custom_json)
        branch_vec = build_branch_vector_custom(custom, params)

        pts = _parse_points(args.points)
        w = _parse_weights(args.point_weights, len(pts))
        y = predict_points(model, branch_vec, pts, params)
        agg = _weighted_aggregate(y, w)

        out_rows = []
        for (L_um, t_s), wi, yi in zip(pts, w, y):
            out_rows.append(
                {"L_um": float(L_um), "t_s": float(t_s), "weight": float(wi), "pred_n": float(yi)}
            )

        out_df = pd.DataFrame(out_rows)
        out_path = os.path.join(OUTPUT_DIR, "custom_points_prediction.csv")
        out_df.to_csv(out_path, index=False, float_format="%.10e")

        summary_path = os.path.join(OUTPUT_DIR, "custom_points_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "n_points": int(len(pts)),
                    "weighted_sum": agg["weighted_sum"],
                    "weighted_mean": agg["weighted_mean"],
                    "weight_sum": agg["weight_sum"],
                    "csv": os.path.abspath(out_path),
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        print("\n" + "-" * 60)
        print(f"[Custom] points predicted: {len(pts)}")
        print(f"[Custom] weighted_sum  = {agg['weighted_sum']:.10e}")
        print(f"[Custom] weighted_mean = {agg['weighted_mean']:.10e}  (sum_w={agg['weight_sum']:.6g})")
        print(f"[Custom] CSV saved:  {out_path}")
        print(f"[Custom] JSON saved: {summary_path}")
        print("\n" + "=" * 60)
        print("Prediction complete!")
        print(f"All outputs saved to: {OUTPUT_DIR}")
        return

    # ------------------------------------------------------------------
    # 2. 加载指定工况数据
    # ------------------------------------------------------------------
    sheet_name = args.sheet
    print(f"\nLoading experiment: {sheet_name}")
    record = load_experiment(DATA_PATH, sheet_name)

    T_start = record["Temp_K"][0]
    T_end = record["Temp_K"][-1]
    cooling_rate = (T_end - T_start) / record["Time_s"][-1] * 3600
    print(f"  Temperature: {T_start:.2f} → {T_end:.2f} K "
          f"(cooling rate: {cooling_rate:.2f} K/h)")
    print(f"  Initial concentration C(0): {record['C0']:.6f}")

    branch_vec = build_branch_vector(record, params)

    # ==================================================================
    # B) 使用数据集初始条件，但只预测任意 (L,t) 点（一个或多个）+ 权重聚合
    #    这用于“测试阶段”：先复用 .mat 的初始条件，后续再切换到 --custom_json
    # ==================================================================
    if args.points:
        pts = _parse_points(args.points)
        w = _parse_weights(args.point_weights, len(pts))
        y = predict_points(model, branch_vec, pts, params)
        agg = _weighted_aggregate(y, w)

        out_rows = []
        for (L_um, t_s), wi, yi in zip(pts, w, y):
            out_rows.append(
                {"L_um": float(L_um), "t_s": float(t_s), "weight": float(wi), "pred_n": float(yi)}
            )

        out_df = pd.DataFrame(out_rows)
        out_path = os.path.join(OUTPUT_DIR, f"sheet_points_prediction_{sheet_name}.csv")
        out_df.to_csv(out_path, index=False, float_format="%.10e")

        summary_path = os.path.join(OUTPUT_DIR, f"sheet_points_summary_{sheet_name}.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "sheet": sheet_name,
                    "data_path": os.path.abspath(DATA_PATH),
                    "n_points": int(len(pts)),
                    "weighted_sum": agg["weighted_sum"],
                    "weighted_mean": agg["weighted_mean"],
                    "weight_sum": agg["weight_sum"],
                    "csv": os.path.abspath(out_path),
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        print("\n" + "-" * 60)
        print(f"[Sheet] points predicted: {len(pts)}  (sheet={sheet_name})")
        print(f"[Sheet] weighted_sum  = {agg['weighted_sum']:.10e}")
        print(f"[Sheet] weighted_mean = {agg['weighted_mean']:.10e}  (sum_w={agg['weight_sum']:.6g})")
        print(f"[Sheet] CSV saved:  {out_path}")
        print(f"[Sheet] JSON saved: {summary_path}")
        print("\n" + "=" * 60)
        print("Prediction complete!")
        print(f"All outputs saved to: {OUTPUT_DIR}")
        return

    # ------------------------------------------------------------------
    # 3. 逐时刻预测
    # ------------------------------------------------------------------
    query_times = sorted(args.times)
    print(f"\nPredicting PSD at times: {query_times}")
    print("-" * 60)

    all_predictions = []
    csv_data = {"L_um": L_eval}

    for t_query in query_times:
        pred = predict_psd(model, branch_vec, L_eval, t_query, params)
        all_predictions.append(pred)
        csv_data[f"pred_t{int(t_query)}s"] = pred

        true = get_true_psd(record, L_eval_idx, t_query)
        if true is not None:
            csv_data[f"true_t{int(t_query)}s"] = true

        # 误差统计
        if true is not None:
            mse = np.mean((pred - true) ** 2)
            mask = true > true.max() * 0.01
            rel_err = (
                np.linalg.norm(pred[mask] - true[mask])
                / (np.linalg.norm(true[mask]) + 1e-15)
            )
            print(f"  t={t_query:>8.0f}s | MSE={mse:.4e} | "
                  f"Rel.L2(active)={rel_err:.4f} | max_pred={pred.max():.4e}")
        else:
            print(f"  t={t_query:>8.0f}s | (no ground truth) | "
                  f"max_pred={pred.max():.4e}")

        # 单时刻对比图
        plot_single_prediction(
            L_eval, pred, true, t_query, sheet_name,
            save_path=os.path.join(
                OUTPUT_DIR, f"pred_{sheet_name}_t{int(t_query)}.png"
            ),
        )

    # ------------------------------------------------------------------
    # 4. 多时刻演化总图
    # ------------------------------------------------------------------
    plot_multi_time_prediction(
        L_eval, all_predictions, query_times, sheet_name,
        save_path=os.path.join(OUTPUT_DIR, f"evolution_{sheet_name}.png"),
    )

    # ------------------------------------------------------------------
    # 5. 导出 CSV
    # ------------------------------------------------------------------
    csv_path = os.path.join(OUTPUT_DIR, f"prediction_{sheet_name}.csv")
    df_out = pd.DataFrame(csv_data)
    df_out.to_csv(csv_path, index=False, float_format="%.8e")
    print(f"\nResults exported to {csv_path}")

    print("\n" + "=" * 60)
    print("Prediction complete!")
    print(f"All outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
