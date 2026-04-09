"""
DeepONet 模型模块（PBE 结晶过程版本）。

架构：
    Branch Net — 输入: [T(t)传感器值, n(L,0)下采样, C(0)]  → (batch, p)
    Trunk Net  — 输入: [L, t]                              → (batch, p)
    输出       — Branch ⊙ Trunk 求和 + bias                → (batch, 1) ≈ n(L,t)
"""

from typing import List, Optional

import tensorflow as tf
from tensorflow import keras


class BranchNet(keras.Model):
    """Branch 网络：编码工况条件（温度曲线 + 初始 PSD + 初始浓度）。"""

    def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int):
        super().__init__()
        layers = []
        dim = input_dim
        for h in hidden_dims:
            layers.append(keras.layers.Dense(h, activation="tanh",
                                             input_shape=(dim,)))
            dim = h
        layers.append(keras.layers.Dense(latent_dim))
        self.net = keras.Sequential(layers)

    def call(self, x):
        return self.net(x)


class TrunkNet(keras.Model):
    """Trunk 网络：编码查询坐标 (L, t)。

    最后一层使用 tanh 激活，输出可解释为基函数。
    """

    def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int):
        super().__init__()
        layers = []
        dim = input_dim
        for h in hidden_dims:
            layers.append(keras.layers.Dense(h, activation="tanh",
                                             input_shape=(dim,)))
            dim = h
        layers.append(keras.layers.Dense(latent_dim, activation="tanh"))
        self.net = keras.Sequential(layers)

    def call(self, y):
        return self.net(y)


class DeepONet(keras.Model):
    """Deep Operator Network（PBE 结晶过程版本）。

    Parameters
    ----------
    branch_input_dim : int
        Branch 输入维度 = n_T_sensors + n_L_sensors + 1。
    trunk_input_dim : int
        Trunk 输入维度 = 2 (L, t)。
    branch_hiddens : list[int]
        Branch 网络各隐藏层宽度。
    trunk_hiddens : list[int]
        Trunk 网络各隐藏层宽度。
    latent_dim : int
        Branch 与 Trunk 输出的共同维度 p。
    """

    def __init__(
        self,
        branch_input_dim: int,
        trunk_input_dim: int = 2,
        branch_hiddens: Optional[List[int]] = None,
        trunk_hiddens: Optional[List[int]] = None,
        latent_dim: int = 128,
    ):
        super().__init__()
        if branch_hiddens is None:
            branch_hiddens = [256, 256, 256]
        if trunk_hiddens is None:
            trunk_hiddens = [128, 128, 128]

        self.branch = BranchNet(branch_input_dim, branch_hiddens, latent_dim)
        self.trunk = TrunkNet(trunk_input_dim, trunk_hiddens, latent_dim)
        self.bias = self.add_weight(
            name="output_bias", shape=(1,), initializer="zeros", trainable=True
        )

    def call(self, inputs):
        """
        Parameters
        ----------
        inputs : (branch_input, trunk_input)
            branch_input : (batch, branch_dim)
            trunk_input  : (batch, 2)   ← [L, t]

        Returns
        -------
        output : (batch, 1) ≈ n(L,t) (归一化值)
        """
        branch_input, trunk_input = inputs
        b = self.branch(branch_input)   # (batch, p)
        t = self.trunk(trunk_input)     # (batch, p)
        output = tf.reduce_sum(b * t, axis=-1, keepdims=True) + self.bias
        return output
