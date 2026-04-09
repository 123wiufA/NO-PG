"""
训练器模块：封装训练循环、验证、学习率调度。
"""

import os
import time
from typing import Optional, Tuple, List

import numpy as np
import tensorflow as tf
from .model import DeepONet


class Trainer:
    """DeepONet 训练器。

    Parameters
    ----------
    model : DeepONet
        待训练模型。
    learning_rate : float
        初始学习率。
    decay_steps : int | None
        指数衰减步数，None 则不使用衰减。
    decay_rate : float
        衰减率。
    """

    def __init__(
        self,
        model: DeepONet,
        learning_rate: float = 1e-3,
        decay_steps: Optional[int] = 5000,
        decay_rate: float = 0.95,
    ):
        self.model = model

        if decay_steps is not None:
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=learning_rate,
                decay_steps=decay_steps,
                decay_rate=decay_rate,
                staircase=False,
            )
        else:
            lr_schedule = learning_rate

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        self.loss_fn = tf.keras.losses.MeanSquaredError()

        self.train_loss_history: List[float] = []
        self.val_loss_history: List[float] = []

    # ------------------------------------------------------------------
    # 单步
    # ------------------------------------------------------------------

    @tf.function
    def _train_step(self, branch_in, trunk_in, labels):
        with tf.GradientTape() as tape:
            preds = self.model([branch_in, trunk_in], training=True)
            loss = self.loss_fn(labels, preds)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    @tf.function
    def _val_step(self, branch_in, trunk_in, labels):
        preds = self.model([branch_in, trunk_in], training=False)
        return self.loss_fn(labels, preds)

    # ------------------------------------------------------------------
    # 数据管道
    # ------------------------------------------------------------------

    @staticmethod
    def _build_dataset(branch, trunk, labels, batch_size, shuffle=True):
        ds = tf.data.Dataset.from_tensor_slices((branch, trunk, labels))
        if shuffle:
            ds = ds.shuffle(buffer_size=min(len(branch), 100000))
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    # ------------------------------------------------------------------
    # 训练主循环
    # ------------------------------------------------------------------

    def fit(
        self,
        train_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
        val_data: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
        epochs: int = 200,
        batch_size: int = 4096,
        print_every: int = 10,
        save_dir: Optional[str] = None,
        save_every: int = 50,
    ):
        """训练模型。

        Parameters
        ----------
        train_data : (branch, trunk, labels)
        val_data   : 同上，可选。
        epochs     : 训练轮数。
        batch_size : 批大小。
        print_every: 每隔多少轮打印日志。
        save_dir   : 权重保存根目录，None 则不保存中间 checkpoint。
                     同时会在此目录下维护 ``best/deeponet`` 权重。
        save_every : 每隔多少轮保存一次权重。
        """
        train_ds = self._build_dataset(*train_data, batch_size)
        val_ds = (
            self._build_dataset(*val_data, batch_size, shuffle=False)
            if val_data is not None
            else None
        )

        self.best_val_loss = float("inf")
        self.best_epoch = 0

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            epoch_losses = []
            for b_in, t_in, lbl in train_ds:
                loss = self._train_step(b_in, t_in, lbl)
                epoch_losses.append(float(loss))
            train_loss = float(np.mean(epoch_losses))
            self.train_loss_history.append(train_loss)

            val_loss = None
            if val_ds is not None:
                val_losses = []
                for b_in, t_in, lbl in val_ds:
                    vl = self._val_step(b_in, t_in, lbl)
                    val_losses.append(float(vl))
                val_loss = float(np.mean(val_losses))
                self.val_loss_history.append(val_loss)

            # ---------- best 权重更新 ----------
            monitor = val_loss if val_loss is not None else train_loss
            if save_dir and monitor < self.best_val_loss:
                self.best_val_loss = monitor
                self.best_epoch = epoch
                best_path = os.path.join(save_dir, "best", "deeponet")
                os.makedirs(os.path.dirname(best_path), exist_ok=True)
                self.model.save_weights(best_path)
                if epoch % print_every == 0 or epoch == 1:
                    tag = "val" if val_loss is not None else "train"
                    print(f"  ★ Best {tag}_loss={monitor:.6e} @ epoch {epoch} → {best_path}")

            # ---------- 日志打印 ----------
            if epoch % print_every == 0 or epoch == 1:
                elapsed = time.time() - t0
                msg = f"Epoch {epoch:>4d}/{epochs} | train_loss={train_loss:.6e}"
                if val_loss is not None:
                    msg += f" | val_loss={val_loss:.6e}"
                msg += f" | best@{self.best_epoch} ({self.best_val_loss:.6e})"
                msg += f" | {elapsed:.2f}s"
                print(msg)

            # ---------- 定期 checkpoint ----------
            if save_dir and epoch % save_every == 0:
                ckpt_path = os.path.join(save_dir, f"ckpt_epoch_{epoch:04d}", "deeponet")
                os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                self.model.save_weights(ckpt_path)
                print(f"  [Checkpoint] saved: {ckpt_path}")
