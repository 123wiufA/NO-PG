# 数据格式说明：.mat / .xlsx → Pickle 缓存

## 一、总体流程

```
数据文件 (.mat 或 .xlsx)             Pickle 缓存 (.pkl)
┌──────────────────┐     首次运行      ┌──────────────────────┐
│ 25 个工况        │  ──自动转换──→   │ Python dict          │
│ 每个工况含完整   │                  │ {case_name: record}  │
│ 时间序列 + PSD   │                  │ 加载 < 1 秒          │
└──────────────────┘                  └──────────────────────┘
        ↑                                      ↑
  原始仿真数据                          程序实际使用的格式
  (人工准备)                          (自动生成, 无需手动维护)
```

**你只需准备数据文件，程序会自动在同目录生成缓存。**
**如果修改了数据文件，程序会检测到时间戳变化并重新生成缓存。**

---

## 二、支持的数据格式

### 格式 A：MATLAB v7.3 `.mat`（HDF5）— **推荐**

文件名: 任意 `.mat`，当前为 `Simulation_Data_DeepONet.mat`（放在 `PBE_DeepONet/` 根目录）。

用 `h5py` 读取。文件结构：

```
Dataset/                            (顶层 Group，名为 "Dataset")
├── CR_1_00/                        (每个工况一个子 Group)
│   ├── Time_s          (1, N_time)    仿真时间序列 (秒)
│   ├── Temp_K          (1, N_time)    温度随时间变化 (K)
│   ├── Conc            (1, N_time)    浓度随时间变化
│   ├── L_mid_um        (1, N_L)       粒径网格中心点 (μm)
│   ├── Growth_Rate_G   (1, N_time)    生长速率 G(t) (μm/s)
│   ├── Nuc_Rate_B      (1, N_time)    成核速率 B₀(t)
│   ├── psd             (N_L, N_snap)  PSD 快照矩阵 → 程序转置为 (N_snap, N_L)
│   ├── snapshot_times  (N_snap, 1)    快照时刻 (秒) → 程序 flatten 为 (N_snap,)
│   ├── C0              (1, 1)         初始浓度
│   └── n_L0            (1, N_L)       初始 PSD
├── CR_1_13/
│   └── ...
└── ...
```

> **注意维度**：`.mat` 中 `psd` 为 `(N_L, N_snap)`；程序会自动检测并转置为
> 内部统一格式 `(N_snap, N_L)`。`snapshot_times`、`C0` 等也会自动 flatten。

### 格式 B：Excel `.xlsx`（保留向后兼容）

每个 Sheet = 一个工况，命名如 `CR_1_00`, `CR_2_50` 等。列结构：

```
列名                  含义                        长度
─────────────────────────────────────────────────────────
Time_s               仿真时间序列 (秒)           N_time
Temp_K               温度随时间变化 (K)          N_time
Conc                 浓度随时间变化              N_time
L_mid_um             粒径网格中心点 (μm)         N_L
Growth_Rate_G        生长速率 G(t) (μm/s)        N_time
Nuc_Rate_B           成核速率 B₀(t)              N_time
Time_0s              PSD 快照: t=0s              N_L
Time_130s            PSD 快照: t=130s            N_L
...                  (每隔 Δt 一个快照)           N_L
Time_7800s           PSD 快照: t=7800s           N_L
```

PSD 快照列的命名规则：`Time_{秒数}s`。`Time_s`（时间序列列）和
`Time_0s`（PSD 快照列）是两个不同的列。

---

## 三、内部数据的统一格式（record 字典）

无论 `.mat` 还是 `.xlsx`，程序均转为统一结构：

```
record 字段              shape / type        说明
─────────────────────────────────────────────────────────
record["Time_s"]         ndarray (N_time,)   仿真时间序列
record["Temp_K"]         ndarray (N_time,)   温度曲线
record["Conc"]           ndarray (N_time,)   浓度曲线
record["L_mid_um"]       ndarray (N_L,)      粒径网格
record["Growth_Rate_G"]  ndarray (N_time,)   生长速率（PI-DeepONet 用）
record["Nuc_Rate_B"]     ndarray (N_time,)   成核速率（PI-DeepONet 用）
record["snapshot_times"] ndarray (N_snap,)   快照时刻
record["psd"]            ndarray (N_snap, N_L) PSD 快照矩阵
record["C0"]             float               初始浓度
record["n_L0"]           ndarray (N_L,)      初始 PSD
```

---

## 四、Pickle 缓存文件

**文件名:** `{数据文件名去掉扩展名}_cache.pkl`

例如: `Simulation_Data_DeepONet.mat` → `Simulation_Data_DeepONet_cache.pkl`

---

## 五、准备新数据集的检查清单

1. **所有工况的 L_mid_um 必须完全相同**（相同 bin 数和值）
2. **所有工况的快照时刻必须完全相同**
3. **Time_s, Temp_K, Conc 长度必须一致**（同一 dt）
4. **Growth_Rate_G 和 Nuc_Rate_B 可选**:
   - Data-Driven (`train.py`) 不需要
   - PI-DeepONet (`train_pi.py`) 需要
5. **将数据文件放到 `PBE_DeepONet/` 根目录**，修改 `train_pi.py` 中的 `DATA_PATH`
6. **删除旧缓存** `*_cache.pkl`，程序会自动重建

---

## 六、数据在 DeepONet 中的使用方式

```
               Branch 输入 (162 维)                    Trunk 输入 (2 维)
┌──────────────────────────────────────────┐    ┌──────────────────┐
│ T(t₀) T(t₁) ... T(t₆₀)  │ n(L,0)下采样 │ C₀ │    │  L_norm   t_norm  │
│      61 个温度值          │   100 个值   │ 1  │    │  L/L_max  t/t_max │
└──────────────────────────────────────────┘    └──────────────────┘
         ↓                                              ↓
    Branch Net (MLP)                              Trunk Net (MLP)
         ↓                                              ↓
     (batch, 128)                                  (batch, 128)
         └─────────── 逐元素乘 + 求和 + bias ──────────┘
                              ↓
                     n(L, t) 预测值 (归一化)
```

### 归一化规则

| 量 | 归一化方式 | 说明 |
|----|-----------|------|
| T(t) | (T - T_min) / (T_max - T_min) | 全局 min-max |
| C(0) | (C - C_min) / (C_max - C_min) | 全局 min-max |
| L | L / L_max | 除以最大粒径 |
| t | t / t_max | 除以最大时间 |
| n(L,t) | n / n_scale | n_scale = 所有正值 PSD 的第 99 百分位 |
