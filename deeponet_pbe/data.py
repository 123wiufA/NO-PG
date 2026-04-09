"""
PBE 结晶过程数据集模块。

从仿真数据（.mat 或 .xlsx）加载并构建 DeepONet 所需的 (branch, trunk, label) 三元组。

支持的数据格式:
    1. MATLAB v7.3 (.mat / HDF5) — 推荐，用 h5py 读取
    2. Excel (.xlsx) — 原始格式，用 pandas 读取（有 pkl 缓存）

算子映射: (n(L,0), C(0), T(t)) → n(L,t)
  - Branch 输入: [T(t) 在传感器时刻的值, n(L,0) 下采样, C(0)]
  - Trunk  输入: [L, t] (归一化后的粒径和时间坐标)
  - 标签:        n(L,t)
"""

import os
import pickle
from typing import Optional, Tuple, List, Dict
import numpy as np


class PBEDataset:
    """结晶过程 PBE 数据集。

    Parameters
    ----------
    data_path : str
        仿真数据文件路径。支持 ``.mat`` (MATLAB v7.3 HDF5) 和 ``.xlsx``。
        旧参数名 ``excel_path`` 仍兼容。
    n_L_sensors : int
        Branch 网络中初始 PSD n(L,0) 的下采样 bin 数。
    n_L_eval : int
        Trunk 网络评估时对 L 轴的下采样 bin 数。
    test_sheets : list[str] | None
        用作测试集的 sheet 名称列表。None 则自动选取。
    skip_t0 : bool
        是否跳过 t=0 的快照（初始 PSD 几乎为零）。
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        n_L_sensors: int = 100,
        n_L_eval: int = 200,
        test_sheets: Optional[List[str]] = None,
        skip_t0: bool = True,
        *,
        excel_path: Optional[str] = None,
    ):
        # 兼容旧参数名 excel_path
        self.data_path = data_path or excel_path
        if self.data_path is None:
            raise ValueError("必须提供 data_path 或 excel_path")
        # 保留旧属性以免外部引用报错
        self.excel_path = self.data_path
        self.n_L_sensors = n_L_sensors
        self.n_L_eval = n_L_eval
        self.skip_t0 = skip_t0

        # 加载并解析所有数据
        self._raw = self._load_all_cases()
        self._sheet_names = list(self._raw.keys())

        # 训练 / 测试划分
        if test_sheets is None:
            test_sheets = self._sheet_names[::5]  # 每 5 个取 1 个做测试
        self.test_sheets = self._resolve_test_sheets(test_sheets)
        self.train_sheets = [s for s in self._sheet_names if s not in self.test_sheets]

        # 全局 L 网格与下采样索引
        self._L_full = self._raw[self._sheet_names[0]]["L_mid_um"]
        self._L_sensor_idx = np.linspace(
            0, len(self._L_full) - 1, n_L_sensors, dtype=int
        )
        self._L_eval_idx = np.linspace(
            0, len(self._L_full) - 1, n_L_eval, dtype=int
        )
        self.L_sensors = self._L_full[self._L_sensor_idx]
        self.L_eval = self._L_full[self._L_eval_idx]

        # PSD 快照时刻
        self.snapshot_times = self._raw[self._sheet_names[0]]["snapshot_times"]

        # 计算归一化参数
        self._compute_normalization()

    # ------------------------------------------------------------------
    # sheet 名解析 / 容错
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_case_index(sheet_name: str) -> Optional[int]:
        """从类似 'CR_2_04' 的名称解析数字索引 4；失败返回 None。"""
        try:
            tail = sheet_name.split("_")[-1]
            return int(tail)
        except Exception:
            return None

    def _best_match_sheet(self, requested: str) -> Optional[str]:
        """为缺失的 sheet 名在现有 keys 中找一个“最接近”的替代。

        规则（按优先级）：
        - 同名（忽略首尾空白）存在则直接返回
        - 若符合 'CR_{i}_{xx}'：在同一 CR_{i}_ 前缀下选 index 最小的 case
        - 否则返回 None
        """
        req = (requested or "").strip()
        if not req:
            return None
        if req in self._raw:
            return req

        # 忽略首尾空白匹配
        if requested in self._raw:
            return requested

        parts = req.split("_")
        if len(parts) >= 2 and parts[0] == "CR":
            prefix = "_".join(parts[:2]) + "_"
            candidates = [k for k in self._sheet_names if k.startswith(prefix)]
            if not candidates:
                return None
            # 按尾部数字排序，取最小
            scored = []
            for c in candidates:
                idx = self._parse_case_index(c)
                scored.append((idx if idx is not None else 10**9, c))
            scored.sort(key=lambda x: (x[0], x[1]))
            return scored[0][1]
        return None

    def _resolve_test_sheets(self, requested: List[str]) -> List[str]:
        """把用户指定的 test_sheets 解析成存在于数据中的列表（带容错）。"""
        resolved: List[str] = []
        missing: List[str] = []
        replaced: List[tuple] = []

        for s in requested:
            s0 = (s or "").strip()
            if not s0:
                continue
            if s0 in self._raw:
                resolved.append(s0)
                continue
            alt = self._best_match_sheet(s0)
            if alt is None:
                missing.append(s0)
                continue
            if alt not in resolved:
                resolved.append(alt)
            replaced.append((s0, alt))

        # 去重但保持顺序
        seen = set()
        resolved_unique = []
        for s in resolved:
            if s not in seen:
                resolved_unique.append(s)
                seen.add(s)

        if replaced:
            msg = ", ".join([f"{a}→{b}" for a, b in replaced])
            print(f"[Data] Warning: test_sheets 部分不存在，已自动替换: {msg}")
        if missing:
            print(f"[Data] Warning: test_sheets 不存在且无法替换，已忽略: {missing}")

        if not resolved_unique:
            # 兜底：保证至少有一个测试 case
            resolved_unique = [self._sheet_names[0]]
            print(f"[Data] Warning: 未能解析任何 test_sheets，已兜底为: {resolved_unique}")

        return resolved_unique

    # ------------------------------------------------------------------
    # 数据加载（自动识别 .mat / .xlsx）
    # ------------------------------------------------------------------

    def _load_all_cases(self) -> Dict:
        """根据文件扩展名选择加载方式。"""
        ext = os.path.splitext(self.data_path)[1].lower()
        if ext == ".mat":
            return self._load_from_mat(self.data_path)
        elif ext in (".xlsx", ".xls"):
            return self._load_from_excel(self.data_path)
        else:
            raise ValueError(
                f"不支持的数据格式 '{ext}'。请使用 .mat 或 .xlsx 文件。"
            )

    # ---- .mat (MATLAB v7.3 / HDF5) ----

    @staticmethod
    def _load_from_mat(mat_path: str) -> Dict:
        """从 MATLAB v7.3 HDF5 .mat 文件加载全部工况。

        期望的 .mat 结构::

            Dataset/
              CR_1_00/
                Time_s          (1, N_time)
                Temp_K          (1, N_time)
                Conc            (1, N_time)
                L_mid_um        (1, N_L)
                Growth_Rate_G   (1, N_time)
                Nuc_Rate_B      (1, N_time)
                psd             (N_L, N_snap)   -- 转置后存为 (N_snap, N_L)
                snapshot_times  (N_snap, 1)
                C0              (1, 1)
                n_L0            (1, N_L)
              CR_1_13/
                ...
        """
        import h5py

        cache_path = os.path.splitext(mat_path)[0] + "_cache.pkl"
        if os.path.exists(cache_path):
            if os.path.getmtime(cache_path) >= os.path.getmtime(mat_path):
                with open(cache_path, "rb") as f:
                    data = pickle.load(f)
                print(f"[Cache] Loaded from {cache_path} (skip .mat)")
                return data

        print(f"[Data] Loading .mat: {mat_path}")
        f = h5py.File(mat_path, "r")

        if "Dataset" in f:
            root = f["Dataset"]
        else:
            root = f

        data = {}
        for name in sorted(root.keys()):
            grp = root[name]
            if not hasattr(grp, "keys"):
                continue
            record = {}

            record["Time_s"] = np.asarray(grp["Time_s"]).flatten().astype(np.float64)
            record["Temp_K"] = np.asarray(grp["Temp_K"]).flatten().astype(np.float64)
            record["Conc"] = np.asarray(grp["Conc"]).flatten().astype(np.float64)
            record["L_mid_um"] = np.asarray(grp["L_mid_um"]).flatten().astype(np.float64)

            psd_raw = np.asarray(grp["psd"], dtype=np.float64)
            snap_raw = np.asarray(grp["snapshot_times"], dtype=np.float64).flatten()

            if psd_raw.shape[0] == len(record["L_mid_um"]):
                psd_raw = psd_raw.T
            record["psd"] = psd_raw
            record["snapshot_times"] = snap_raw

            if "Growth_Rate_G" in grp:
                record["Growth_Rate_G"] = np.asarray(grp["Growth_Rate_G"]).flatten().astype(np.float64)
            if "Nuc_Rate_B" in grp:
                record["Nuc_Rate_B"] = np.asarray(grp["Nuc_Rate_B"]).flatten().astype(np.float64)

            record["C0"] = float(np.asarray(grp["C0"]).flat[0])
            record["n_L0"] = np.asarray(grp["n_L0"]).flatten().astype(np.float64)

            data[name] = record

        f.close()

        with open(cache_path, "wb") as fp:
            pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[Cache] Saved to {cache_path}")

        return data

    # ---- .xlsx (Excel，保留向后兼容) ----

    @staticmethod
    def _load_from_excel(excel_path: str) -> Dict:
        """从 Excel 文件加载全部工况（有 pkl 缓存）。"""
        import pandas as pd

        cache_path = os.path.splitext(excel_path)[0] + "_cache.pkl"

        if os.path.exists(cache_path) and os.path.exists(excel_path):
            if os.path.getmtime(cache_path) >= os.path.getmtime(excel_path):
                with open(cache_path, "rb") as f:
                    data = pickle.load(f)
                print(f"[Cache] Loaded from {cache_path} (skip Excel)")
                return data

        print("[Cache] No cache found, loading from Excel (first time)...")
        xls = pd.ExcelFile(excel_path)
        data = {}

        for name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=name)
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
            snapshot_times = []
            psd_snapshots = []
            for col in psd_cols:
                t_val = float(col.replace("Time_", "").replace("s", ""))
                snapshot_times.append(t_val)
                psd_snapshots.append(df[col].iloc[:n_L].values)

            record["snapshot_times"] = np.array(snapshot_times)
            record["psd"] = np.array(psd_snapshots)

            if "Growth_Rate_G" in df.columns:
                record["Growth_Rate_G"] = df["Growth_Rate_G"].values
            if "Nuc_Rate_B" in df.columns:
                record["Nuc_Rate_B"] = df["Nuc_Rate_B"].values

            record["C0"] = record["Conc"][0]
            record["n_L0"] = record["psd"][0]

            data[name] = record

        xls.close()

        with open(cache_path, "wb") as fp:
            pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[Cache] Saved to {cache_path}")

        return data

    # ------------------------------------------------------------------
    # 归一化
    # ------------------------------------------------------------------

    def _compute_normalization(self):
        """计算全局归一化参数（min-max）。"""
        all_T = np.concatenate([r["Temp_K"] for r in self._raw.values()])
        all_C = np.concatenate([r["Conc"] for r in self._raw.values()])
        all_psd = np.concatenate([r["psd"].ravel() for r in self._raw.values()])

        self.T_min, self.T_max = all_T.min(), all_T.max()
        self.C_min, self.C_max = all_C.min(), all_C.max()
        self.L_max = self._L_full.max()
        self.t_max = self.snapshot_times.max()
        self.n_scale = np.percentile(all_psd[all_psd > 0], 99) if np.any(all_psd > 0) else 1.0

    def _normalize_T(self, T: np.ndarray) -> np.ndarray:
        return (T - self.T_min) / (self.T_max - self.T_min + 1e-12)

    def _normalize_C(self, C: float) -> float:
        return (C - self.C_min) / (self.C_max - self.C_min + 1e-12)

    def _normalize_L(self, L: np.ndarray) -> np.ndarray:
        return L / self.L_max

    def _normalize_t(self, t: np.ndarray) -> np.ndarray:
        return t / self.t_max

    def _normalize_n(self, n: np.ndarray) -> np.ndarray:
        return n / self.n_scale

    # ------------------------------------------------------------------
    # Branch 输入构建
    # ------------------------------------------------------------------

    def _build_branch_vector(self, record: Dict) -> np.ndarray:
        """为单个实验构建 Branch 输入向量。

        结构: [ T(t_0), T(t_1), ..., T(t_K) | n(L_1,0), ..., n(L_M,0) | C(0) ]
        """
        # 温度曲线在 PSD 快照时刻采样
        dt = record["Time_s"][1] - record["Time_s"][0]
        T_at_snapshots = []
        for t in self.snapshot_times:
            idx = min(int(t / dt), len(record["Temp_K"]) - 1)
            T_at_snapshots.append(record["Temp_K"][idx])
        T_sensors = self._normalize_T(np.array(T_at_snapshots))

        # 初始 PSD 下采样
        n_L0_sampled = record["n_L0"][self._L_sensor_idx]
        n_L0_norm = self._normalize_n(n_L0_sampled)

        # 初始浓度
        C0_norm = self._normalize_C(record["C0"])

        return np.concatenate([T_sensors, n_L0_norm, [C0_norm]]).astype(np.float32)

    # ------------------------------------------------------------------
    # 数据集构建
    # ------------------------------------------------------------------

    def _build_for_sheets(self, sheet_names: List[str]):
        """为指定的 sheets 构建 (branch, trunk, labels) 数据。"""
        branch_list = []
        trunk_list = []
        label_list = []

        for name in sheet_names:
            record = self._raw[name]
            branch_vec = self._build_branch_vector(record)

            snapshots = self.snapshot_times
            start_idx = 1 if self.skip_t0 else 0

            for t_idx in range(start_idx, len(snapshots)):
                t_val = snapshots[t_idx]
                t_norm = self._normalize_t(np.array([t_val]))[0]

                psd_at_t = record["psd"][t_idx]  # (n_L_full,)
                psd_eval = psd_at_t[self._L_eval_idx]  # 下采样

                for l_idx, (L_val, n_val) in enumerate(
                    zip(self.L_eval, psd_eval)
                ):
                    L_norm = self._normalize_L(np.array([L_val]))[0]
                    branch_list.append(branch_vec)
                    trunk_list.append([L_norm, t_norm])
                    label_list.append([self._normalize_n(n_val)])

        branch = np.array(branch_list, dtype=np.float32)
        trunk = np.array(trunk_list, dtype=np.float32)
        labels = np.array(label_list, dtype=np.float32)

        return branch, trunk, labels

    def get_train_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """返回训练集 (branch, trunk, labels)。"""
        return self._build_for_sheets(self.train_sheets)

    def get_test_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """返回测试集 (branch, trunk, labels)。"""
        return self._build_for_sheets(self.test_sheets)

    # ------------------------------------------------------------------
    # 浓度数据构建
    # ------------------------------------------------------------------

    def _build_concentration_for_sheets(
        self, sheet_names: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """为指定 sheets 构建浓度监督数据 (branch, tau, C_norm_label)。

        对每个 sheet 的每个 PSD 快照时刻，生成一条浓度样本。
        """
        branch_list = []
        tau_list = []
        C_list = []

        start_idx = 1 if self.skip_t0 else 0

        for name in sheet_names:
            record = self._raw[name]
            branch_vec = self._build_branch_vector(record)

            for t_idx in range(start_idx, len(self.snapshot_times)):
                t_val = self.snapshot_times[t_idx]
                t_norm = self._normalize_t(np.array([t_val]))[0]

                dt = record["Time_s"][1] - record["Time_s"][0]
                c_idx = min(int(t_val / dt), len(record["Conc"]) - 1)
                C_val = record["Conc"][c_idx]
                C_norm = self._normalize_C(C_val)

                branch_list.append(branch_vec)
                tau_list.append([t_norm])
                C_list.append([C_norm])

        return (
            np.array(branch_list, dtype=np.float32),
            np.array(tau_list, dtype=np.float32),
            np.array(C_list, dtype=np.float32),
        )

    def get_concentration_train_data(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """返回浓度训练集 (branch, tau, C_norm_label)。"""
        return self._build_concentration_for_sheets(self.train_sheets)

    def get_concentration_test_data(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """返回浓度测试集 (branch, tau, C_norm_label)。"""
        return self._build_concentration_for_sheets(self.test_sheets)

    # ------------------------------------------------------------------
    # 辅助信息
    # ------------------------------------------------------------------

    @property
    def branch_dim(self) -> int:
        """Branch 输入维度 = n_T_sensors + n_L_sensors + 1。"""
        return len(self.snapshot_times) + self.n_L_sensors + 1

    @property
    def trunk_dim(self) -> int:
        """Trunk 输入维度 = 2 (L, t)。"""
        return 2

    def inverse_normalize_n(self, n_norm: np.ndarray) -> np.ndarray:
        """将归一化的 PSD 值还原为原始尺度。"""
        return n_norm * self.n_scale

    def summary(self):
        """打印数据集摘要。"""
        print(f"Data: {self.data_path}")
        print(f"Total experiments: {len(self._sheet_names)}")
        print(f"  Train: {len(self.train_sheets)} sheets {self.train_sheets}")
        print(f"  Test:  {len(self.test_sheets)} sheets {self.test_sheets}")
        print(f"L range: {self._L_full[0]:.1f} ~ {self._L_full[-1]:.1f} um"
              f" ({len(self._L_full)} bins)")
        print(f"  L sensors (branch): {self.n_L_sensors} bins")
        print(f"  L eval (trunk):     {self.n_L_eval} bins")
        print(f"Snapshot times: {len(self.snapshot_times)} "
              f"(0 ~ {self.snapshot_times[-1]:.0f} s)")
        print(f"Branch dim: {self.branch_dim}")
        print(f"Trunk dim:  {self.trunk_dim}")
        print(f"Normalization: T=[{self.T_min:.2f},{self.T_max:.2f}], "
              f"n_scale={self.n_scale:.6e}")
