# -*- coding: utf-8 -*-
"""
建模模块
========

【方法学说明】

A. **分类模型对比**
   本项目对比 4 类经典化学计量学/机器学习模型：

   1. **PCA + SVM (RBF)**：先 PCA 降维去共线性，再用径向基核 SVM。
      SVM 是小样本高维数据的金标准，拉曼分类文献中应用最广。
   2. **PLS-DA (Partial Least Squares Discriminant Analysis)**：化学计量学
      最经典方法。PLS 同时考虑 X 的方差和 X-Y 协方差，对共线性的光谱数据
      特别有效。判别分析将类别编码为 one-hot 后做 PLS 回归，argmax 分类。
   3. **Random Forest**：树模型天然抗共线性、抗过拟合，可输出特征重要性。
   4. **KNN**：作为简单基线（baseline），距离度量下的最近邻分类。

B. **评估指标**
   - Accuracy：整体准确率
   - Macro-F1：类别均衡的 F1（不受类别不平衡影响）
   - 推理耗时：单样本预测延迟（实际部署关心）
   - 三维度加权评分：参考过往项目「肥胖分类算法对比研究」的成熟框架
     Score = 0.4·norm(Acc) + 0.4·norm(MacroF1) + 0.2·norm(-Time)

C. **定量分析**
   - **PLS 回归**：用光谱预测"菌体浓度"（连续变量）
   - 评估：R²、RMSE、5 折交叉验证

D. **聚类**
   - KMeans 聚类数=类别数，用 ARI/NMI 评估"无监督发现类别结构"的能力
"""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
from typing import Tuple, Dict

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_predict, KFold
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             classification_report,
                             r2_score, mean_squared_error)
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

import config


# ============== 数据划分 ==============
def stratified_split(X, y, test_size=config.TEST_SIZE, seed=config.RANDOM_SEED):
    """分层划分，保证训练/测试集中每类比例一致"""
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


# ============== PLS-DA ==============
class PLSDA:
    """PLS-DA: PLS 回归 + one-hot + argmax"""
    def __init__(self, n_components=10):
        self.n_components = n_components
        self.pls = PLSRegression(n_components=n_components, scale=False)
        self.label_encoder = LabelEncoder()

    def fit(self, X, y):
        y_int = self.label_encoder.fit_transform(y)
        n_classes = len(self.label_encoder.classes_)
        Y = np.eye(n_classes)[y_int]
        self.pls.fit(X, Y)
        return self

    def predict(self, X):
        pred = self.pls.predict(X)
        idx = np.argmax(pred, axis=1)
        return self.label_encoder.inverse_transform(idx)


# ============== 分类模型库 ==============
def build_classifiers():
    """构建候选分类器字典"""
    return {
        "SVM_RBF": SVC(kernel="rbf", C=10.0, gamma="scale", probability=False),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=None, random_state=config.RANDOM_SEED, n_jobs=-1),
        "KNN": KNeighborsClassifier(n_neighbors=7, metric="euclidean"),
        "PLS_DA": PLSDA(n_components=15),
    }


def evaluate_classifier(model, X_train, y_train, X_test, y_test) -> Dict:
    """训练 + 测试 + 计时，返回指标字典"""
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    y_pred = model.predict(X_test)
    pred_time_total = time.perf_counter() - t1
    pred_time_per_sample = pred_time_total / max(len(y_test), 1)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "macro_f1": f1_score(y_test, y_pred, average="macro"),
        "fit_time_s": fit_time,
        "pred_time_per_sample_ms": pred_time_per_sample * 1000,
        "y_pred": y_pred,
    }


def weighted_score(metrics_dict: Dict, weight_acc=0.4, weight_f1=0.4, weight_time=0.2) -> pd.DataFrame:
    """
    三维度加权评分，参考过往项目「肥胖分类算法对比研究」的成熟框架。
    对每个指标做 min-max 归一化后线性组合；时间用负值（越快越好）。
    """
    # 只取数值列（剔除 y_pred 等非标量）
    num_cols = ["accuracy", "macro_f1", "fit_time_s", "pred_time_per_sample_ms"]
    df = pd.DataFrame({k: {c: v[c] for c in num_cols} for k, v in metrics_dict.items()}).T
    norm_acc = (df["accuracy"] - df["accuracy"].min()) / (df["accuracy"].max() - df["accuracy"].min() + 1e-12)
    norm_f1 = (df["macro_f1"] - df["macro_f1"].min()) / (df["macro_f1"].max() - df["macro_f1"].min() + 1e-12)
    norm_time = -(df["pred_time_per_sample_ms"] - df["pred_time_per_sample_ms"].min()) / (
            df["pred_time_per_sample_ms"].max() - df["pred_time_per_sample_ms"].min() + 1e-12)
    df["weighted_score"] = weight_acc * norm_acc + weight_f1 * norm_f1 + weight_time * norm_time
    df["weighted_score"] = (df["weighted_score"] - df["weighted_score"].min()) / (
            df["weighted_score"].max() - df["weighted_score"].min() + 1e-12)
    return df


# ============== PLS 定量回归 ==============
def pls_quantitative_regression(X, concentrations,
                                n_components=10, kfold=config.CV_FOLDS, seed=config.RANDOM_SEED):
    """
    PLS 回归预测"菌体浓度"，5 折交叉验证。

    Returns
    -------
    result : dict
        含 R2_cv, RMSE_cv, y_true_all, y_pred_all
    """
    kf = KFold(n_splits=kfold, shuffle=True, random_state=seed)
    pls = PLSRegression(n_components=n_components, scale=True)
    y_pred_all = cross_val_predict(pls, X, concentrations, cv=kf)
    r2 = r2_score(concentrations, y_pred_all)
    rmse = np.sqrt(mean_squared_error(concentrations, y_pred_all))
    return {
        "r2_cv": r2,
        "rmse_cv": rmse,
        "y_true": concentrations,
        "y_pred": y_pred_all,
        "n_components": n_components,
        "kfold": kfold,
    }


# ============== 聚类分析 ==============
def kmeans_clustering(X, y_true, n_clusters=None, seed=config.RANDOM_SEED):
    """KMeans 聚类 + 用 ARI/NMI 评估与真实标签的一致性"""
    if n_clusters is None:
        n_clusters = len(np.unique(y_true))
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    y_cluster = km.fit_predict(X)
    return {
        "labels": y_cluster,
        "ARI": adjusted_rand_score(y_true, y_cluster),
        "NMI": normalized_mutual_info_score(y_true, y_cluster),
        "n_clusters": n_clusters,
        "method": "KMeans",
    }


def gmm_clustering(X, y_true, n_clusters=None, seed=config.RANDOM_SEED):
    """高斯混合聚类（允许椭圆形簇，更适合 PCA 空间的延伸分布）"""
    if n_clusters is None:
        n_clusters = len(np.unique(y_true))
    gmm = GaussianMixture(n_components=n_clusters, covariance_type="full",
                          random_state=seed, n_init=3)
    y_cluster = gmm.fit_predict(X)
    return {
        "labels": y_cluster,
        "ARI": adjusted_rand_score(y_true, y_cluster),
        "NMI": normalized_mutual_info_score(y_true, y_cluster),
        "n_clusters": n_clusters,
        "method": "GMM",
    }


def compare_clustering(X, y_true, seed=config.RANDOM_SEED):
    """对比 KMeans 与 GMM 聚类（化学计量学结论：监督 > 无监督）"""
    return {
        "KMeans": kmeans_clustering(X, y_true, seed=seed),
        "GMM": gmm_clustering(X, y_true, seed=seed),
    }


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from data_loader import generate_synthetic_dataset
    from preprocess import preprocess_dataset
    from feature import PCAReducer, extract_peak_features

    ds = generate_synthetic_dataset(n_per_class=100)
    Xp = preprocess_dataset(ds.intensities, verbose=False)
    reducer = PCAReducer(n_components=20)
    scores = reducer.fit_transform(Xp)
    print(f"[FEAT] PCA scores shape={scores.shape}, "
          f"95% 方差所需 PC 数 = {reducer.n_components_for_variance(0.95)}")

    # 分类对比
    Xtr, Xte, ytr, yte = stratified_split(scores, ds.labels)
    models = build_classifiers()
    results = {}
    for name, m in models.items():
        print(f"  训练 {name} ...", end=" ")
        results[name] = evaluate_classifier(m, Xtr, ytr, Xte, yte)
        print(f"Acc={results[name]['accuracy']:.3f}, "
              f"F1={results[name]['macro_f1']:.3f}, "
              f"t/sample={results[name]['pred_time_per_sample_ms']:.3f} ms")
    score_df = weighted_score(results)
    print(score_df[["accuracy", "macro_f1", "pred_time_per_sample_ms", "weighted_score"]].round(3))

    # 定量
    reg = pls_quantitative_regression(scores, ds.concentrations)
    print(f"[REG]  PLS 定量 R²={reg['r2_cv']:.3f}, RMSE={reg['rmse_cv']:.3f}")

    # 聚类
    clu = kmeans_clustering(scores, ds.labels)
    print(f"[CLU]  KMeans ARI={clu['ARI']:.3f}, NMI={clu['NMI']:.3f}")
    print("[OK] 建模自检通过")
