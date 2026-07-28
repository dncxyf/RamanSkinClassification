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

    # ==================== 新增：Kaiser准则自动选择主成分数 ====================
    def select_n_components_kaiser(self, X: np.ndarray, max_components: int = None) -> int:
        """
        使用Kaiser准则选择主成分数（保留特征值 > 1的主成分）

        Parameters
        ----------
        X : np.ndarray
            输入数据 (n_samples, n_features)
        max_components : int
            最大候选主成分数，None则使用 min(n_samples, n_features)

        Returns
        -------
        optimal_n : int
            推荐的主成分数量
        """
        n_samples, n_features = X.shape
        max_components = max_components or min(n_samples, n_features)

        # 拟合完整PCA
        pca_full = PCA(n_components=max_components, random_state=self.random_state)
        pca_full.fit(X)

        # Kaiser准则：特征值 > 1
        # 对于标准化数据，特征值 = 解释方差 * n_features
        eigen_values = pca_full.explained_variance_ratio_ * n_features
        optimal_n = np.sum(eigen_values > 1.0)

        # 至少保留1个主成分
        if optimal_n == 0:
            optimal_n = 1

        print(f"   [Kaiser准则] 特征值>1的主成分有 {optimal_n} 个")
        print(
            f"   [Kaiser准则] 累计解释方差: {np.cumsum(pca_full.explained_variance_ratio_)[optimal_n - 1] * 100:.1f}%")

        # 更新类属性
        self.n_components = optimal_n
        return optimal_n

    # ==================== 新增：肘部法则自动选择主成分数 ====================
    def select_n_components_elbow(self, X: np.ndarray, max_components: int = None) -> int:
        """
        使用肘部法则选择主成分数（找方差下降的拐点）

        Parameters
        ----------
        X : np.ndarray
            输入数据 (n_samples, n_features)
        max_components : int
            最大候选主成分数，None则使用 min(n_samples, n_features, 50)

        Returns
        -------
        optimal_n : int
            推荐的主成分数量
        """
        n_samples, n_features = X.shape
        max_components = max_components or min(n_samples, n_features, 50)

        # 拟合完整PCA
        pca_full = PCA(n_components=max_components, random_state=self.random_state)
        pca_full.fit(X)

        explained_variance = pca_full.explained_variance_ratio_

        # 计算每个点的"肘部得分"：方差下降的加速度
        # 拐点就是方差下降从"快"变"慢"的那个点
        diffs = np.diff(explained_variance)  # 一阶差分：下降速度
        second_diffs = np.diff(diffs)  # 二阶差分：加速度变化

        # 找加速度变化最大的点（即方差下降速度变化最剧烈的点）
        # 取绝对值，因为我们要找变化最剧烈的点
        elbow_score = np.abs(second_diffs)

        # 加一点平滑，避免局部抖动（取前5个点的平均）
        if len(elbow_score) > 5:
            from scipy.signal import savgol_filter
            elbow_score_smooth = savgol_filter(elbow_score, 5, 2)
            optimal_n = np.argmax(elbow_score_smooth) + 2
        else:
            optimal_n = np.argmax(elbow_score) + 2

        # 确保在合理范围内
        optimal_n = max(2, min(optimal_n, max_components))

        cumsum_var = np.cumsum(explained_variance)
        print(f"   [肘部法则] 拐点位置: {optimal_n} 个主成分")
        print(f"   [肘部法则] 累计解释方差: {cumsum_var[optimal_n - 1] * 100:.1f}%")

        # 更新类属性
        self.n_components = optimal_n
        return optimal_n

    # ==================== 新增：BIC贝叶斯信息准则 ====================
    def select_n_components_bic(self, X: np.ndarray, max_components: int = None) -> int:
        """
        使用BIC（贝叶斯信息准则）选择最优主成分数

        原理：在拟合优度和模型复杂度之间找平衡
        BIC值越小，模型越好

        Parameters
        ----------
        X : np.ndarray
            输入数据 (n_samples, n_features)
        max_components : int
            最大候选主成分数，None则使用 min(n_samples, n_features, 50)

        Returns
        -------
        optimal_n : int
            推荐的主成分数量
        """
        from sklearn.decomposition import PCA

        n_samples, n_features = X.shape
        max_components = max_components or min(n_samples, n_features, 50)

        # 标准化数据（BIC假设数据已标准化）
        from sklearn.preprocessing import StandardScaler
        X_scaled = StandardScaler().fit_transform(X)

        # 计算不同n_component的BIC
        bics = []
        candidates = list(range(1, max_components + 1))

        print(f"   [BIC法] 正在计算 {len(candidates)} 个候选主成分数...")

        for n in candidates:
            pca_temp = PCA(n_components=n, random_state=self.random_state)
            pca_temp.fit(X_scaled)

            # 重构数据
            X_reconstructed = pca_temp.inverse_transform(pca_temp.transform(X_scaled))

            # 计算均方误差
            mse = np.mean((X_scaled - X_reconstructed) ** 2)

            # BIC计算公式：BIC = n * ln(MSE) + k * ln(n)
            # 其中 k 是参数数量
            # 对于PCA，参数数量 ≈ n * (n_features - n) + n
            # 但简化版本为：k = n * n_features - n * (n + 1) / 2
            k = n * n_features - n * (n + 1) / 2
            bic = n_samples * np.log(mse) + k * np.log(n_samples)
            bics.append(bic)

            # 显示进度（仅当候选数不多时）
            if len(candidates) <= 20:
                print(f"      n={n}: BIC={bic:.2f}")

        # 找到BIC最小的点
        bics = np.array(bics)
        optimal_n = np.argmin(bics) + 1

        # 同时计算拐点（BIC下降变慢的位置）
        if len(bics) > 3:
            diffs = np.diff(bics)
            second_diffs = np.diff(diffs)
            # 如果存在明显的拐点，使用拐点
            if len(second_diffs) > 2:
                elbow_idx = np.argmax(np.abs(second_diffs)) + 2
                # 如果拐点比BIC最小值更合理（更靠前），使用拐点
                if elbow_idx < optimal_n and bics[elbow_idx - 1] - bics[optimal_n - 1] < 0.1 * np.max(bics):
                    optimal_n = elbow_idx

        optimal_n = max(1, min(optimal_n, max_components))

        # 计算该主成分数下的累计方差
        pca_full = PCA(n_components=max_components, random_state=self.random_state)
        pca_full.fit(X_scaled)
        cumsum_var = np.cumsum(pca_full.explained_variance_ratio_)

        print(f"   [BIC法] 最优主成分数: {optimal_n}")
        print(f"   [BIC法] 累计解释方差: {cumsum_var[optimal_n - 1] * 100:.1f}%")
        print(f"   [BIC法] BIC最小值: {bics[optimal_n - 1]:.2f}")

        # 更新类属性
        self.n_components = optimal_n
        return optimal_n

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

