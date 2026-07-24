# -*- coding: utf-8 -*-
"""
特征工程
========

【方法学说明】
拉曼光谱分类/定量建模有两类典型特征：

A. **全光谱特征 (Full Spectrum)**
   - 直接用所有 701 个波数点作为特征向量
   - 优点：不丢失任何信息；缺点：维度高 + 共线性严重（相邻通道几乎完全相关）
   - 通常配合 PCA 降维使用

B. **物理特征 (Peak-based Features)**
   - 从光谱中提取峰位、峰强、峰面积、峰宽
   - 优点：可解释、维度低、对预处理不严格；缺点：依赖峰检测算法可靠性
   - 适合"物质定性识别"

C. **PCA 降维 (Principal Component Analysis)**
   - 将 701 维压缩为 ~10-20 维主成分（Loading 谱解释生物意义）
   - 是化学计量学的"标准动作"
   - 同时降维消除共线性，对线性模型（PLS-DA、SVM 线性核）尤其有效

本模块同时提供三类特征，供下游模型选择。
"""
from __future__ import annotations
import numpy as np
from typing import Tuple
from sklearn.decomposition import PCA

import config


# ============== A. 全光谱特征 ==============
def full_spectrum(X: np.ndarray) -> np.ndarray:
    """直接返回原始（已预处理）光谱矩阵"""
    return X.copy()


# ============== B. 峰特征 ==============
def extract_peak_features(
    X: np.ndarray,
    wavenumbers: np.ndarray,
    target_peaks: list = None,
    half_window_cm: float = 10.0,
) -> Tuple[np.ndarray, list]:
    """
    在指定目标峰位附近提取峰强（最大值）与峰面积（积分），构造低维特征。

    Parameters
    ----------
    X : (N, W) 预处理后的光谱矩阵
    wavenumbers : (W,) 波数轴
    target_peaks : 目标峰位列表 (cm⁻¹)；None 则用生物拉曼主峰
    half_window_cm : 每个峰位附近取积分的半窗宽 (cm⁻¹)

    Returns
    -------
    feats : (N, 2*K)  每个 peak 两列：[峰强, 峰面积]
    feat_names : 特征名（用于可解释性）
    """
    if target_peaks is None:
        target_peaks = [854, 938, 1003, 1063, 1098, 1126, 1157,
                        1250, 1305, 1335, 1350, 1448, 1528, 1550,
                        1580, 1607, 1660]

    dw = wavenumbers[1] - wavenumbers[0]
    half_idx = int(round(half_window_cm / dw))
    feats, names = [], []
    for pk in target_peaks:
        idx_c = int(np.argmin(np.abs(wavenumbers - pk)))
        lo, hi = max(idx_c - half_idx, 0), min(idx_c + half_idx + 1, len(wavenumbers))
        window = X[:, lo:hi]
        wavenumbers_window = wavenumbers[lo:hi]
        # 峰强（窗口内最大值）
        intensity = window.max(axis=1)
        # 峰面积（梯形积分）
        area = np.trapz(window, x=wavenumbers_window, axis=1)
        feats.append(intensity)
        names.append(f"intensity_{pk}")
        feats.append(area)
        names.append(f"area_{pk}")
    feats = np.array(feats).T  # (N, 2K)
    return feats, names


# ============== C. PCA 降维 ==============
class PCAReducer:
    """封装 PCA：拟合 + 报告解释方差"""
    def __init__(self, n_components: int = config.PCA_N_COMPONENTS, random_state=config.RANDOM_SEED):
        self.n_components = n_components
        self.random_state = random_state
        self.pca = PCA(n_components=n_components, random_state=random_state)
        self.cumvar_ = None

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        scores = self.pca.fit_transform(X)
        self.cumvar_ = np.cumsum(self.pca.explained_variance_ratio_)
        return scores

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.pca.transform(X)

    def n_components_for_variance(self, threshold: float = 0.99) -> int:
        """达到指定累计解释方差所需的主成分数"""
        return int(np.searchsorted(self.cumvar_, threshold) + 1)


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from data_loader import generate_synthetic_dataset
    from preprocess import preprocess_dataset
    ds = generate_synthetic_dataset(n_per_class=30)
    Xp = preprocess_dataset(ds.intensities, verbose=False)

    feats, names = extract_peak_features(Xp, ds.wavenumbers)
    print(f"[PEAKS] 形状 {feats.shape}, 前 5 个特征名: {names[:5]}")

    reducer = PCAReducer(n_components=20)
    scores = reducer.fit_transform(Xp)
    print(f"[PCA]   scores 形状 {scores.shape}")
    print(f"       前 5 个 PC 解释方差: {reducer.pca.explained_variance_ratio_[:5].round(3)}")
    print(f"       达 95% / 99% 方差所需 PC 数: "
          f"{reducer.n_components_for_variance(0.95)} / {reducer.n_components_for_variance(0.99)}")
    print("[OK] 特征工程自检通过")
