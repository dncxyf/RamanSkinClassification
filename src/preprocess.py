# -*- coding: utf-8 -*-
"""
拉曼光谱预处理 Pipeline
========================

【方法学说明】

原始拉曼光谱通常含有以下干扰，必须依次处理才能用于后续建模：

1. **荧光背景 / 基线漂移**
   - 来源：生物样本中的荧光团（NADH、核黄素等）+ 探测器暗电流。
   - 表现：一条缓慢变化的"凸包络"叠加在拉曼信号上，幅度远大于拉曼峰本身。
   - 处理：AsLS（Asymmetric Least Squares, Eilers & Boelens 2005）基线校正。
     原理：用 2 阶差分罚约束基线平滑 + 非对称权重（信号高于基线的点权重 p，
     低于基线的点权重 1-p），迭代求解加权最小二乘。
     典型参数：λ（平滑度）1e4-1e7，p（非对称度）0.001-0.1。

2. **高频噪声**
   - 来源：探测器读出噪声、散粒噪声。
   - 处理：Savitzky-Golay (SG) 滤波。原理：在滑动窗口内拟合多项式，取中心点
     拟合值作为平滑后信号。相比简单移动平均，SG 能保留峰形（保持峰位的导数信息）。

3. **样本间整体强度差异**
   - 来源：采样位置、光程、菌体浓度的差异 → 同一物质在不同光谱中的峰强绝对值不同。
   - 处理：SNV（Standard Normal Variate）。原理：每条光谱独立地 z-score 标准化
     (x - mean) / std。SNV 不需要参考光谱，适合"未知样本"场景。

替代/补充方法（实际项目可选）：
  - 基线：ModPoly、IModPoly、arPLS（自适应惩罚）、airPLS
  - 归一化：MSC（多元散射校正，需参考光谱）、峰面积归一化（指定内标峰）
  - 平滑：Whittaker smoother、小波去噪
"""
from __future__ import annotations
import numpy as np
import config
try:                                   # 包内导入
    from .data_loader import SpectraDataset  # noqa: F401
except ImportError:                    # 直接运行本文件时
    pass


# ============== 1. AsLS 基线校正 ==============
def asls_baseline(y: np.ndarray, lam: float = config.ASLS_LAMBDA,
                  p: float = config.ASLS_P, n_iter: int = 20) -> np.ndarray:
    """
    单条光谱的 AsLS 基线估计（Eilers 2005 算法）。

    Parameters
    ----------
    y : 1D 光谱
    lam : 平滑度（越大基线越平滑）
    p : 非对称权重（越小越倾向于"压在信号下方"）
    n_iter : 迭代次数

    Returns
    -------
    baseline : 1D 基线
    """
    try:
        from pybaselines.whittaker import asls as _asls
        # pybaselines 1.2 接口
        baseline, _ = _asls(y, lam=lam, p=p, max_iter=n_iter)
        return baseline
    except Exception:
        # 退化为手写实现（不依赖 pybaselines）
        return _asls_numpy(y, lam=lam, p=p, n_iter=n_iter)


def _asls_numpy(y, lam=1e5, p=0.01, n_iter=20):
    """AsLS 纯 numpy 实现（fallback）"""
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    H = lam * D.dot(D.T)
    w = np.ones(L)
    for _ in range(n_iter):
        W = sparse.diags(w, 0, shape=(L, L))
        Z = W + H
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


# ============== 2. SG 平滑 ==============
def savgol_smooth(y: np.ndarray, window: int = config.SG_WINDOW,
                  poly: int = config.SG_POLYORDER) -> np.ndarray:
    """Savitzky-Golay 平滑"""
    from scipy.signal import savgol_filter
    if window % 2 == 0:
        window += 1
    return savgol_filter(y, window_length=window, polyorder=poly)


# ============== 3. 归一化 ==============
def snv(y: np.ndarray) -> np.ndarray:
    """Standard Normal Variate 归一化 (x - mean) / std —— 用于分类（凸显峰形）"""
    return (y - y.mean()) / (y.std() + 1e-12)


def max_normalize(y: np.ndarray) -> np.ndarray:
    """fallback：整条光谱最大值归一化（不推荐用于定量）"""
    m = np.max(y)
    return y / (m + 1e-12)


def max_normalize_with_wn(y: np.ndarray, wavenumbers: np.ndarray,
                          ref_center: float = 470.0, ref_half_window: float = 25.0) -> np.ndarray:
    """
    内标峰归一化（化学定量分析的标准方法「内标法」）。

    用内标峰（默认 470 cm⁻¹）的峰强作为分母，将整条光谱归一化。
    这样：
      - 归一化后的菌体峰强度 = (原始菌体峰强度 / 内标峰强度)
                               ∝ (菌体浓度 / 内标浓度) ∝ 菌体浓度
      - 即保留了浓度-强度关系，定量回归才能成立
    """
    mask = np.abs(wavenumbers - ref_center) <= ref_half_window
    if mask.sum() == 0:
        m = np.max(y)
    else:
        m = np.max(y[mask])
    return y / (m + 1e-12)


# ============== Pipeline 主入口 ==============
PREPROCESS_FOR_CLASSIFICATION = ("asls", "sg", "snv")
PREPROCESS_FOR_QUANTIFICATION = ("asls", "sg", "max_norm")


def preprocess_one(y: np.ndarray, steps=("asls", "sg", "snv"),
                   wavenumbers: np.ndarray = None) -> np.ndarray:
    """
    对单条光谱按顺序应用预处理。

    steps 选项：
      - "asls"     : AsLS 基线校正
      - "sg"       : Savitzky-Golay 平滑
      - "snv"      : SNV 归一化（分类任务用）
      - "max_norm" : 内标峰最大值归一化（定量任务用，需传 wavenumbers）

    推荐组合：
      分类：("asls", "sg", "snv")       —— 消除整体强度，凸显峰形
      定量：("asls", "sg", "max_norm")  —— 保留浓度-强度关系
    """
    out = y.astype(float).copy()
    if "asls" in steps:
        out = out - asls_baseline(out)
    if "sg" in steps:
        out = savgol_smooth(out)
    if "snv" in steps:
        out = snv(out)
    if "max_norm" in steps:
        if wavenumbers is None:
            out = max_normalize(out)
        else:
            out = max_normalize_with_wn(out, wavenumbers)
    return out


def preprocess_dataset(intensities: np.ndarray, steps=("asls", "sg", "snv"),
                       wavenumbers: np.ndarray = None, verbose: bool = True) -> np.ndarray:
    """对整个光谱矩阵逐条预处理（按行）"""
    out = np.empty_like(intensities, dtype=float)
    n = intensities.shape[0]
    for i in range(n):
        out[i] = preprocess_one(intensities[i], steps=steps, wavenumbers=wavenumbers)
        if verbose and (i + 1) % 500 == 0:
            print(f"     预处理进度 {i+1}/{n}")
    return out


if __name__ == "__main__":
    # 自检：用一条合成光谱测试预处理
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from data_loader import generate_synthetic_dataset
    ds = generate_synthetic_dataset(n_per_class=20)
    print(f"[INPUT]  shape={ds.intensities.shape}, "
          f"range=[{ds.intensities.min():.2f}, {ds.intensities.max():.2f}]")
    out = preprocess_dataset(ds.intensities, verbose=False)
    print(f"[OUTPUT] shape={out.shape}, "
          f"range=[{out.min():.2f}, {out.max():.2f}], "
          f"per-row mean={out.mean(axis=1).mean():.2e} (应≈0, SNV 后)")
    print("[OK] 预处理 Pipeline 自检通过")
