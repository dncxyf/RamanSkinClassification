# -*- coding: utf-8 -*-
"""
可视化模块
==========
所有图统一样式：配色用 config.CLASS_COLORS，中文用微软雅黑，
输出同时保存 PNG（看效果）和 PDF（矢量，入报告）。
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 非交互后端
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.metrics import confusion_matrix

import config

# ============== 字体与样式 ==============
rcParams["font.family"] = config.MATPLOTLIB_FONT["family"]
rcParams["font.size"] = config.MATPLOTLIB_FONT["size"]
rcParams["axes.unicode_minus"] = False
rcParams["axes.spines.top"] = False
rcParams["axes.spines.right"] = False
rcParams["axes.grid"] = True
rcParams["grid.alpha"] = 0.3
rcParams["grid.linestyle"] = "--"


def _save(fig, name: str) -> list:
    """保存 PNG + PDF，返回路径列表"""
    paths = []
    for fmt in config.FIG_FMT:
        path = config.FIG_DIR / f"{name}.{fmt}"
        fig.savefig(path, dpi=config.FIG_DPI, bbox_inches="tight")
        paths.append(str(path))
    plt.close(fig)
    return paths


# ============== 图 1：原始光谱概览（每类均值 + 置信带）==============
def plot_raw_spectra_overview(ds, max_per_class=50) -> list:
    fig, ax = plt.subplots(figsize=(11, 5))
    for sp in ds.species_list:
        mask = ds.labels == sp
        idx = np.where(mask)[0][:max_per_class]
        X = ds.intensities[idx]
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        color = config.CLASS_COLORS[sp]
        ax.plot(ds.wavenumbers, mean, color=color, lw=1.5,
                label=f"{sp} ({config.CLASS_LABEL_ZH[sp]})")
        ax.fill_between(ds.wavenumbers, mean - std, mean + std,
                        color=color, alpha=0.15)
    ax.set_xlabel("波数 Raman Shift (cm$^{-1}$)")
    ax.set_ylabel("强度 (a.u.)")
    ax.set_title("原始拉曼光谱概览（每类均值 ± 标准差）", fontweight="bold")
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    return _save(fig, "01_raw_spectra_overview")


# ============== 图 2：预处理前后对比 ==============
def plot_preprocessing_comparison(ds, idx=0) -> list:
    from preprocess import asls_baseline, savgol_smooth, snv
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    wn = ds.wavenumbers
    raw = ds.intensities[idx]
    baseline = asls_baseline(raw)
    after_asls = raw - baseline
    after_sg = savgol_smooth(after_asls)
    after_snv = snv(after_sg)
    label_zh = config.CLASS_LABEL_ZH[ds.labels[idx]]

    axes[0, 0].plot(wn, raw, color="#2C3E50", lw=1)
    axes[0, 0].plot(wn, baseline, color="#E64B35", lw=1.5, ls="--", label="AsLS 基线")
    axes[0, 0].set_title(f"(a) 原始光谱 + AsLS 基线（{label_zh}）", fontsize=10)
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(wn, after_asls, color="#2980B9", lw=1)
    axes[0, 1].set_title("(b) 基线校正后", fontsize=10)

    axes[1, 0].plot(wn, after_sg, color="#00A087", lw=1)
    axes[1, 0].set_title("(c) Savitzky-Golay 平滑后", fontsize=10)
    axes[1, 0].set_xlabel("波数 (cm$^{-1}$)")

    axes[1, 1].plot(wn, after_snv, color="#F39B7F", lw=1)
    axes[1, 1].set_title("(d) SNV 归一化后", fontsize=10)
    axes[1, 1].set_xlabel("波数 (cm$^{-1}$)")

    fig.suptitle("预处理 Pipeline 各步骤效果（同一条样本）", fontweight="bold", y=1.0)
    fig.tight_layout()
    return _save(fig, "02_preprocessing_steps")


# ============== 图 3：预处理后 5 类平均光谱 ==============
def plot_processed_mean_spectra(ds, Xp) -> list:
    fig, ax = plt.subplots(figsize=(11, 5))
    for sp in ds.species_list:
        mask = ds.labels == sp
        mean = Xp[mask].mean(axis=0)
        ax.plot(ds.wavenumbers, mean, color=config.CLASS_COLORS[sp], lw=1.4,
                label=f"{sp} ({config.CLASS_LABEL_ZH[sp]})")
    # 标注主要生物分子峰
    annotations = {1003: "Phe", 1098: "PO4", 1448: "CH2", 1660: "Amide I"}
    for wn, txt in annotations.items():
        ax.axvline(wn, color="gray", ls=":", lw=0.7, alpha=0.6)
        ax.text(wn, ax.get_ylim()[1] * 0.95, txt, fontsize=8, ha="center", color="dimgray")
    ax.set_xlabel("波数 (cm$^{-1}$)")
    ax.set_ylabel("SNV 强度")
    ax.set_title("预处理后 5 类菌种平均光谱对比", fontweight="bold")
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    return _save(fig, "03_processed_mean_spectra")


# ============== 图 4：PCA 解释方差 ==============
def plot_pca_variance(reducer) -> list:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    n = len(reducer.pca.explained_variance_ratio_)
    xs = np.arange(1, n + 1)
    ax.bar(xs, reducer.pca.explained_variance_ratio_, color="#2980B9", alpha=0.7, label="单 PC")
    ax.plot(xs, reducer.cumvar_, "o-", color="#E64B35", label="累计")
    ax.axhline(0.95, ls="--", color="gray", lw=1)
    ax.text(n, 0.96, "95%", fontsize=8, ha="right", color="gray")
    ax.axhline(0.99, ls="--", color="gray", lw=1)
    ax.text(n, 1.0, "99%", fontsize=8, ha="right", color="gray")
    ax.set_xlabel("主成分序号")
    ax.set_ylabel("解释方差比例")
    ax.set_title(f"PCA 解释方差（前 {n} 个主成分累计 {reducer.cumvar_[-1] * 100:.1f}%）",
                 fontweight="bold")
    ax.legend(fontsize=9)
    return _save(fig, "04_pca_variance")


# ============== 图 5：PCA 2D 投影（按类着色）==============
def plot_pca_2d(scores, labels, species_list, explained) -> list:
    fig, ax = plt.subplots(figsize=(8, 6))
    for sp in species_list:
        mask = labels == sp
        ax.scatter(scores[mask, 0], scores[mask, 1], s=14, alpha=0.6,
                   color=config.CLASS_COLORS[sp],
                   label=f"{sp} ({config.CLASS_LABEL_ZH[sp]})")
    ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained[1]*100:.1f}%)")
    ax.set_title("PCA 2D 投影（前两个主成分）", fontweight="bold")
    ax.legend(fontsize=8, loc="best")
    return _save(fig, "05_pca_2d_projection")


# ============== 图 6：模型对比（Accuracy / Macro-F1 / 耗时 + 加权评分）==============
def plot_model_comparison(score_df: pd.DataFrame) -> list:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    df_plot = score_df.copy()
    # 左：Acc / F1 柱状图
    x = np.arange(len(df_plot))
    w = 0.35
    axes[0].bar(x - w/2, df_plot["accuracy"], w, label="Accuracy", color="#2980B9")
    axes[0].bar(x + w/2, df_plot["macro_f1"], w, label="Macro-F1", color="#E64B35")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df_plot.index, rotation=15)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("指标值")
    axes[0].set_title("分类模型：Accuracy & Macro-F1", fontweight="bold", fontsize=11)
    axes[0].legend(fontsize=9)
    for i, (a, f) in enumerate(zip(df_plot["accuracy"], df_plot["macro_f1"])):
        axes[0].text(i - w/2, a + 0.01, f"{a:.3f}", ha="center", fontsize=8)
        axes[0].text(i + w/2, f + 0.01, f"{f:.3f}", ha="center", fontsize=8)

    # 右：综合加权评分
    bars = axes[1].barh(df_plot.index, df_plot["weighted_score"], color="#00A087")
    axes[1].set_xlim(0, 1.05)
    axes[1].set_xlabel("综合加权评分（归一化）")
    axes[1].set_title("三维度加权综合评分（0.4·Acc + 0.4·F1 + 0.2·速度）",
                      fontweight="bold", fontsize=11)
    for i, v in enumerate(df_plot["weighted_score"]):
        axes[1].text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)
    fig.tight_layout()
    return _save(fig, "06_model_comparison")


# ============== 图 7：最优模型混淆矩阵 ==============
def plot_confusion_matrix(y_true, y_pred, species_list, model_name="PLS-DA") -> list:
    cm = confusion_matrix(y_true, y_pred, labels=species_list)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(species_list)))
    ax.set_yticks(range(len(species_list)))
    short_labels = [s[:3] + ". " + s.split()[1][:4] + "." for s in species_list]
    ax.set_xticklabels(short_labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(short_labels, fontsize=9)
    ax.set_xlabel("预测菌种")
    ax.set_ylabel("真实菌种")
    ax.set_title(f"{model_name} 混淆矩阵（按行归一化 %）", fontweight="bold")
    for i in range(len(species_list)):
        for j in range(len(species_list)):
            color = "white" if cm_pct[i, j] > 50 else "black"
            ax.text(j, i, f"{cm_pct[i, j]:.0f}", ha="center", va="center",
                    color=color, fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.8, label="百分比 (%)")
    fig.tight_layout()
    return _save(fig, "07_confusion_matrix_best")


# ============== 图 8：PLS 定量拟合 ==============
def plot_pls_regression(reg_result) -> list:
    y_true = reg_result["y_true"]
    y_pred = reg_result["y_pred"]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, s=12, alpha=0.4, color="#3C5488", edgecolor="none")
    # y=x 参考线
    lo, hi = y_true.min(), y_true.max()
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="y = x (理想)")
    # 线性拟合线
    z = np.polyfit(y_true, y_pred, 1)
    slope, intercept = float(z[0]), float(z[1])
    xs = np.linspace(lo, hi, 50)
    ax.plot(xs, np.polyval(z, xs), "-", color="#E64B35", lw=1.5,
            label=f"拟合: y={slope:.2f}x+{intercept:.2f}")
    ax.set_xlabel("真实菌体浓度（合成标签）")
    ax.set_ylabel("PLS 预测浓度")
    ax.set_title(f"PLS 定量回归（5 折 CV）  R²={reg_result['r2_cv']:.3f}  "
                 f"RMSE={reg_result['rmse_cv']:.3f}",
                 fontweight="bold")
    ax.legend(fontsize=9)
    return _save(fig, "08_pls_regression")


# ============== 图 9：聚类结果（KMeans 在 PCA 空间）==============
def plot_clustering(scores, labels_true, labels_cluster, species_list, clu_metrics) -> list:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    # 左：真实标签
    for sp in species_list:
        mask = labels_true == sp
        axes[0].scatter(scores[mask, 0], scores[mask, 1], s=12, alpha=0.5,
                        color=config.CLASS_COLORS[sp],
                        label=f"{sp[:3]}. {sp.split()[1][:4]}.")
    axes[0].set_title("真实菌种标签", fontweight="bold")
    axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")
    axes[0].legend(fontsize=7, loc="best")
    # 右：KMeans 聚类
    n_clusters = len(np.unique(labels_cluster))
    cmap = plt.cm.get_cmap("Set2", n_clusters)
    for k in range(n_clusters):
        mask = labels_cluster == k
        axes[1].scatter(scores[mask, 0], scores[mask, 1], s=12, alpha=0.5,
                        color=cmap(k), label=f"Cluster {k}")
    axes[1].set_title(f"GMM 聚类 (k={n_clusters})  "
                      f"ARI={clu_metrics['ARI']:.3f}  NMI={clu_metrics['NMI']:.3f}",
                      fontweight="bold")
    axes[1].set_xlabel("PC1"); axes[1].set_ylabel("PC2")
    axes[1].legend(fontsize=8, loc="best")
    fig.tight_layout()
    return _save(fig, "09_gmm_clustering")
