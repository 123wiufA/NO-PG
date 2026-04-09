"""
纯数据驱动 DeepONet 训练脚本：仅 MSE 监督，无物理约束。

每次训练自动创建以启动时间命名的输出文件夹（如 ``results/20260408_153045/``），
内含定期 checkpoint、best 权重、loss 曲线和归一化参数，方便人为早停后调用任意阶段权重。

用法（命令行）::

    # 从头训练
    python -u train.py

    # 从已有权重继续训练
    python -u train.py --weights results/20260408_153045/weights/best/deeponet

PyCharm Edit Configurations → Parameters 栏直接粘贴::

    从头训练:        （留空）
    继续训练(best):  --weights results/20260408_153045/weights/best/deeponet
    继续训练(目录):  --weights results/20260408_153045/weights/best
    继续训练(.index): --weights results/20260408_153045/weights/best/deeponet.index
    继续训练(ckpt):  --weights results/20260408_153045/weights/ckpt_epoch_0200/deeponet
    指定数据+轮数:   --data D:\Data\my_experiment.mat --epochs 400
    完整示例:        --weights results/20260408_153045/weights/best/deeponet --epochs 300 --lr 1e-4 --batch 2048
"""

import os
import json
import argparse
from datetime import datetime

from deeponet_pbe.gpu_config import setup_gpu
setup_gpu()

import numpy as np
import tensorflow as tf

from deeponet_pbe.data import PBEDataset
from deeponet_pbe.model import DeepONet
from deeponet_pbe.trainer import Trainer
from deeponet_pbe.utils import (
    plot_loss, plot_psd_comparison, plot_psd_evolution,
)

def normalize_weights_prefix(weights_path: str) -> str:
    """把 ``--weights`` 转成 TensorFlow 期望的权重「前缀」路径。

    支持三种写法::
        .../best/deeponet              （推荐）
        .../best/deeponet.index        （PyCharm 里选了文件时常见）
        .../best 或 ...\\best          （权重目录，自动补全 …/best/deeponet）
    """
    p = os.path.normpath(weights_path.strip())
    if os.path.isdir(p):
        p = os.path.join(p, "deeponet")
    if p.lower().endswith(".index"):
        p = p[: -len(".index")]
    return p


def parse_args():
    parser = argparse.ArgumentParser(description="DeepONet 数据驱动训练")
    parser.add_argument(
        "--weights", type=str, default=None,
        help="预训练权重：目录（.../weights/best）或前缀（.../deeponet），"
             "也可带 .index 后缀。不指定则从头训练。",
    )
    parser.add_argument(
        "--data", type=str, default=None,
        help="数据文件路径（.mat 或 .xlsx），默认当前目录下 Simulation_Data_DeepONet.mat",
    )
    parser.add_argument("--epochs", type=int, default=None, help="训练轮数")
    parser.add_argument("--lr", type=float, default=None, help="初始学习率")
    parser.add_argument("--batch", type=int, default=None, help="批大小")
    return parser.parse_args()


def main():
    args = parse_args()

    # ==================================================================
    # 路径与超参数
    # ==================================================================

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = args.data or os.path.join(SCRIPT_DIR, "Simulation_Data_DeepONet.mat")

    RUN_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, "results", RUN_TAG)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    N_L_SENSORS = 100
    N_L_EVAL = 200
    TEST_SHEETS = ["CR_1_00", "CR_2_00", "CR_3_00", "CR_4_00"]

    BRANCH_HIDDENS = [256, 256, 256]
    TRUNK_HIDDENS = [128, 128, 128]
    LATENT_DIM = 128

    LEARNING_RATE = args.lr or 5e-4
    DECAY_STEPS = 3000
    DECAY_RATE = 0.9
    EPOCHS = args.epochs or 600
    BATCH_SIZE = args.batch or 4096
    PRINT_EVERY = 10
    SAVE_EVERY = 50

    RESUME_WEIGHTS = (
        normalize_weights_prefix(args.weights) if args.weights else None
    )

    # 记录本次运行的全部超参数，便于回顾
    hparams = dict(
        data_path=DATA_PATH, run_tag=RUN_TAG,
        resume_weights=RESUME_WEIGHTS,
        resume_weights_raw=args.weights,
        n_L_sensors=N_L_SENSORS, n_L_eval=N_L_EVAL,
        test_sheets=TEST_SHEETS,
        branch_hiddens=BRANCH_HIDDENS, trunk_hiddens=TRUNK_HIDDENS,
        latent_dim=LATENT_DIM, learning_rate=LEARNING_RATE,
        decay_steps=DECAY_STEPS, decay_rate=DECAY_RATE,
        epochs=EPOCHS, batch_size=BATCH_SIZE, save_every=SAVE_EVERY,
    )
    hparams_path = os.path.join(OUTPUT_DIR, "hparams.json")
    with open(hparams_path, "w", encoding="utf-8") as f:
        json.dump(hparams, f, indent=2, ensure_ascii=False)
    print(f"[Run] {RUN_TAG}  →  {OUTPUT_DIR}")
    if RESUME_WEIGHTS:
        raw_w = (args.weights or "").strip()
        if raw_w and os.path.normpath(raw_w) != os.path.normpath(RESUME_WEIGHTS):
            print(f"[Run] Resume input:  {raw_w}")
        print(f"[Run] Resume prefix: {RESUME_WEIGHTS}")
    else:
        print("[Run] Training from scratch")
    print(f"[Run] Hyperparams saved to {hparams_path}")

    # ==================================================================
    # 数据加载
    # ==================================================================
    print("=" * 60)
    print(f"[DeepONet] Loading data from {DATA_PATH} ...")
    dataset = PBEDataset(
        data_path=DATA_PATH,
        n_L_sensors=N_L_SENSORS,
        n_L_eval=N_L_EVAL,
        test_sheets=TEST_SHEETS,
        skip_t0=False,
    )
    dataset.summary()
    # 重要：PBEDataset 会对缺失的 sheet 名做自动替换/剔除，这里同步更新
    TEST_SHEETS = dataset.test_sheets

    print("\nBuilding training data...")
    train_data = dataset.get_train_data()
    print(f"  Branch: {train_data[0].shape}")
    print(f"  Trunk:  {train_data[1].shape}")
    print(f"  Labels: {train_data[2].shape}")

    print("Building test data...")
    test_data = dataset.get_test_data()
    print(f"  Test samples: {test_data[0].shape[0]}")

    # ==================================================================
    # 构建模型
    # ==================================================================
    print("=" * 60)
    print("Building DeepONet model...")
    model = DeepONet(
        branch_input_dim=dataset.branch_dim,
        trunk_input_dim=dataset.trunk_dim,
        branch_hiddens=BRANCH_HIDDENS,
        trunk_hiddens=TRUNK_HIDDENS,
        latent_dim=LATENT_DIM,
    )
    dummy_b = tf.zeros((1, dataset.branch_dim))
    dummy_t = tf.zeros((1, dataset.trunk_dim))
    _ = model([dummy_b, dummy_t])
    model.summary()

    # ==================================================================
    # 加载预训练权重（可选）
    # ==================================================================
    if RESUME_WEIGHTS:
        idx_file = RESUME_WEIGHTS + ".index"
        if not os.path.exists(idx_file):
            raise FileNotFoundError(
                f"找不到权重: {idx_file}\n"
                f"已解析的前缀: {RESUME_WEIGHTS}\n"
                f"可填写: …\\weights\\best\\deeponet   或   …\\weights\\best   或   …\\deeponet.index"
            )

        model.load_weights(RESUME_WEIGHTS)
        print(f"[Resume] Loaded weights from {RESUME_WEIGHTS}")

        val_pred = model([test_data[0], test_data[1]], training=False).numpy()
        resume_val_loss = float(np.mean((val_pred - test_data[2]) ** 2))
        print(f"[Resume] Initial val_loss = {resume_val_loss:.6e}")
    else:
        print("[Init] Training from scratch (random initialization)")

    # ==================================================================
    # 训练
    # ==================================================================

    trainer = Trainer(
        model=model,
        learning_rate=LEARNING_RATE,
        decay_steps=DECAY_STEPS,
        decay_rate=DECAY_RATE,
    )

    # 归一化参数在训练开始前即保存，中断训练后也能直接用于预测
    norm_params_path = os.path.join(OUTPUT_DIR, "norm_params.npz")
    np.savez(
        norm_params_path,
        branch_dim=dataset.branch_dim,
        branch_hiddens=np.array(BRANCH_HIDDENS),
        trunk_hiddens=np.array(TRUNK_HIDDENS),
        latent_dim=LATENT_DIM,
        T_min=dataset.T_min, T_max=dataset.T_max,
        C_min=dataset.C_min, C_max=dataset.C_max,
        L_max=dataset.L_max, t_max=dataset.t_max,
        n_scale=dataset.n_scale,
        snapshot_times=dataset.snapshot_times,
        L_sensor_idx=dataset._L_sensor_idx,
        L_eval=dataset.L_eval,
        L_eval_idx=dataset._L_eval_idx,
    )
    print(f"[Pre-train] norm_params.npz saved to {norm_params_path}")

    ckpt_dir = os.path.join(OUTPUT_DIR, "weights")
    os.makedirs(ckpt_dir, exist_ok=True)

    print("=" * 60)
    print("Start DeepONet training (data-driven, MSE only)...")
    print(f"  Epochs: {EPOCHS}, Batch: {BATCH_SIZE}, LR: {LEARNING_RATE}")
    print(f"  Checkpoints  → {ckpt_dir}/ckpt_epoch_XXXX/")
    print(f"  Best weights → {ckpt_dir}/best/")

    trainer.fit(
        train_data=train_data,
        val_data=test_data,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        print_every=PRINT_EVERY,
        save_dir=ckpt_dir,
        save_every=SAVE_EVERY,
    )

    # ==================================================================
    # 可视化
    # ==================================================================

    print("=" * 60)
    print("Plotting results...")

    plot_loss(
        trainer.train_loss_history,
        trainer.val_loss_history,
        save_path=os.path.join(OUTPUT_DIR, "loss_curve.png"),
    )

    L_eval = dataset.L_eval
    n_L = len(L_eval)
    snapshot_times = dataset.snapshot_times
    start_idx = 1 if dataset.skip_t0 else 0
    eval_times = snapshot_times[start_idx:]
    n_times = len(eval_times)
    compare_indices = [n_times // 4, n_times // 2, 3 * n_times // 4, n_times - 1]

    for sheet_name in TEST_SHEETS:
        record = dataset._raw[sheet_name]
        branch_vec = dataset._build_branch_vector(record)

        pred_all = []
        for t_idx in range(n_times):
            t_val = eval_times[t_idx]
            t_norm = dataset._normalize_t(np.array([t_val]))[0]
            L_norm = dataset._normalize_L(L_eval)

            trunk_batch = np.stack(
                [L_norm, np.full(n_L, t_norm)], axis=-1
            ).astype(np.float32)
            branch_batch = np.tile(branch_vec, (n_L, 1))

            pred_norm = model([branch_batch, trunk_batch], training=False)
            pred_real = dataset.inverse_normalize_n(pred_norm.numpy().flatten())

            psd_idx = start_idx + t_idx
            true_real = record["psd"][psd_idx][dataset._L_eval_idx]

            pred_all.append(pred_real)

            if t_idx in compare_indices:
                plot_psd_comparison(
                    L_eval, true_real, pred_real,
                    time_val=t_val, sheet_name=sheet_name,
                    save_path=os.path.join(
                        OUTPUT_DIR, f"psd_{sheet_name}_t{int(t_val)}.png"
                    ),
                )

        plot_psd_evolution(
            L_eval, np.array(pred_all), eval_times,
            title=f"DeepONet Predicted PSD [{sheet_name}]",
            save_path=os.path.join(OUTPUT_DIR, f"evolution_{sheet_name}.png"),
        )

    # ==================================================================
    # 测试集整体误差
    # ==================================================================

    all_pred = model([test_data[0], test_data[1]], training=False).numpy()
    mse_norm = np.mean((all_pred - test_data[2]) ** 2)
    rel_l2 = np.linalg.norm(all_pred - test_data[2]) / (
        np.linalg.norm(test_data[2]) + 1e-12
    )
    pred_real = dataset.inverse_normalize_n(all_pred)
    true_real = dataset.inverse_normalize_n(test_data[2])
    mse_real = np.mean((pred_real - true_real) ** 2)

    print(f"\n[Test] MSE (normalized):  {mse_norm:.6e}")
    print(f"[Test] Relative L2 Error: {rel_l2:.6e}")
    print(f"[Test] MSE (real scale):  {mse_real:.6e}")

    # ==================================================================
    # 保存最终权重（final）
    # ==================================================================
    final_path = os.path.join(ckpt_dir, "final", "deeponet")
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    model.save_weights(final_path)

    print(f"\n[Final] weights saved to {final_path}")

    # 保存 loss 历史到 CSV，便于后续分析
    loss_csv = os.path.join(OUTPUT_DIR, "loss_history.csv")
    with open(loss_csv, "w", encoding="utf-8") as f:
        has_val = len(trainer.val_loss_history) > 0
        header = "epoch,train_loss" + (",val_loss" if has_val else "")
        f.write(header + "\n")
        for i, tl in enumerate(trainer.train_loss_history):
            row = f"{i + 1},{tl:.8e}"
            if has_val:
                row += f",{trainer.val_loss_history[i]:.8e}"
            f.write(row + "\n")

    print(f"[Final] loss history saved to {loss_csv}")

    # ==================================================================
    # 训练总结
    # ==================================================================

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"  Run tag:       {RUN_TAG}")
    print(f"  Output dir:    {OUTPUT_DIR}")
    print(f"  Best epoch:    {trainer.best_epoch}  "
          f"(loss={trainer.best_val_loss:.6e})")
    print(f"  Final epoch:   {EPOCHS}")
    print()
    print("Directory structure:")
    print(f"  {OUTPUT_DIR}/")
    print(f"    hparams.json          ← 超参数记录")
    print(f"    norm_params.npz       ← 归一化参数（predict 需要）")
    print(f"    loss_history.csv      ← epoch-by-epoch loss")
    print(f"    loss_curve.png        ← loss 图")
    print(f"    weights/")
    print(f"      best/deeponet.*     ← 最优权重 (epoch {trainer.best_epoch})")
    print(f"      final/deeponet.*    ← 最终权重 (epoch {EPOCHS})")
    print(f"      ckpt_epoch_XXXX/    ← 每 {SAVE_EVERY} epoch 存档")


if __name__ == "__main__":
    main()
