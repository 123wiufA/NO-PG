"""
PBE 结晶过程可视化工具。
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_loss(train_loss, val_loss=None, save_path=None):
    """绘制训练/验证损失曲线。"""
    plt.figure(figsize=(8, 5))
    plt.semilogy(train_loss, label="Train Loss")
    if val_loss:
        plt.semilogy(val_loss, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss (normalized)")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Loss curve saved to {save_path}")
    plt.close()


def plot_psd_comparison(
    L_eval,
    true_psd,
    pred_psd,
    time_val,
    sheet_name="",
    save_path=None,
):
    """对比单个时刻的真实 PSD 与预测 PSD。

    Parameters
    ----------
    L_eval : (N_L,) 粒径坐标 (μm)
    true_psd : (N_L,) 真实 n(L,t)
    pred_psd : (N_L,) 预测 n(L,t)
    time_val : float 时刻 (s)
    sheet_name : str 工况名称
    save_path : str 保存路径
    """
    plt.figure(figsize=(9, 5))
    plt.plot(L_eval, true_psd, "b-", linewidth=1.5, label="True $n(L,t)$")
    plt.plot(L_eval, pred_psd, "r--", linewidth=1.5, label="DeepONet Pred")
    plt.xlabel("$L$ (μm)")
    plt.ylabel("$n(L,t)$")
    plt.title(f"PSD at t={time_val:.0f}s [{sheet_name}]")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"PSD plot saved to {save_path}")
    plt.close()


def plot_psd_evolution(
    L_eval,
    psd_snapshots,
    snapshot_times,
    title="",
    save_path=None,
    n_curves=8,
):
    """绘制 PSD 随时间的演化。

    Parameters
    ----------
    L_eval : (N_L,) 粒径坐标
    psd_snapshots : (n_times, N_L) 各时刻 PSD
    snapshot_times : (n_times,) 对应时刻
    n_curves : int 绘制的曲线数量
    """
    plt.figure(figsize=(10, 6))
    indices = np.linspace(0, len(snapshot_times) - 1, n_curves, dtype=int)
    cmap = plt.cm.viridis

    for i, idx in enumerate(indices):
        color = cmap(i / (n_curves - 1))
        plt.plot(L_eval, psd_snapshots[idx], color=color,
                 label=f"t={snapshot_times[idx]:.0f}s")

    plt.xlabel("$L$ (μm)")
    plt.ylabel("$n(L,t)$")
    plt.title(title or "PSD Evolution")
    plt.legend(fontsize=8, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Evolution plot saved to {save_path}")
    plt.close()


def plot_concentration_comparison(
    times,
    C_true,
    C_pred,
    sheet_name="",
    save_path=None,
):
    """对比浓度时间演化: 仿真真值 vs 网络预测。

    Parameters
    ----------
    times : (N,) 时刻 (s)
    C_true : (N,) 真实浓度
    C_pred : (N,) 预测浓度
    sheet_name : str 工况名称
    save_path : str 保存路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(times, C_true, "b-o", markersize=3, linewidth=1.5,
                 label="Simulation")
    axes[0].plot(times, C_pred, "r--s", markersize=3, linewidth=1.5,
                 label="DeepONet Pred")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Concentration $C(t)$")
    axes[0].set_title(f"Concentration Evolution [{sheet_name}]")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    rel_err = np.abs(C_pred - C_true) / (np.abs(C_true) + 1e-15) * 100
    axes[1].plot(times, rel_err, "k-^", markersize=3, linewidth=1.2)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Relative Error (%)")
    axes[1].set_title(f"Concentration Relative Error [{sheet_name}]")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Concentration plot saved to {save_path}")
    plt.close()


def plot_temperature_profiles(dataset, save_path=None):
    """绘制所有工况的温度曲线。"""
    plt.figure(figsize=(10, 5))
    for name in dataset._sheet_names:
        record = dataset._raw[name]
        t = record["Time_s"]
        T = record["Temp_K"]
        step = max(1, len(t) // 500)
        plt.plot(t[::step], T[::step], label=name, linewidth=0.8)

    plt.xlabel("Time (s)")
    plt.ylabel("Temperature (K)")
    plt.title("Temperature Profiles (All Experiments)")
    plt.legend(fontsize=6, ncol=5, loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Temperature profiles saved to {save_path}")
    plt.close()
