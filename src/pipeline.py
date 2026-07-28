# -*- coding: utf-8 -*-
"""
主入口：一键运行完整 Pipeline
================================

用法:
    python pipeline.py            # 完整流程（生成数据 + 预处理 + 建模 + 可视化 + 报告）
    python pipeline.py --quick    # 快速版（每类 100 样本，用于调试）

输出:
    - data/processed/  预处理后数据（CSV）
    - results/figures/ 全部图（PNG + PDF）
    - results/metrics.csv  模型对比表
    - results/summary.json  最终结果汇总（供报告生成使用）
"""
from __future__ import annotations
import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from data_loader import generate_synthetic_dataset, save_dataset_csv
from preprocess import (preprocess_dataset,
                        PREPROCESS_FOR_CLASSIFICATION,
                        PREPROCESS_FOR_QUANTIFICATION)
from feature import PCAReducer, extract_peak_features
from models import (stratified_split, build_classifiers, evaluate_classifier,
                    weighted_score, pls_quantitative_regression, compare_clustering)
import visualize as viz
from sklearn.preprocessing import StandardScaler # 用于峰特征和pca特征融合

def run(n_per_class=config.N_SAMPLES_PER_CLASS):
    t_start = time.perf_counter()
    print("=" * 70)
    print("  皮肤拉曼光谱分类与定量分析 Pipeline")
    print("=" * 70)

    # ---------- Step 0: 数据生成 ----------
    print("\n[0/7] 生成合成拉曼光谱数据 ...")
    ds = generate_synthetic_dataset(n_per_class=n_per_class)
    print(f"      样本数 = {ds.n_samples}, 波数点 = {ds.n_wavenumbers}")
    print(f"      类别 = {len(ds.species_list)} 种皮肤相关微生物")

    # ---------- Step 1: EDA 可视化 ----------
    print("\n[1/7] EDA：原始光谱概览图 ...")
    viz.plot_raw_spectra_overview(ds)

    # ---------- Step 2: 预处理（分类路径 + 定量路径）----------
    print("\n[2/7] 预处理 Pipeline ...")
    print("      分类路径: AsLS 基线 + SG 平滑 + SNV 归一化")
    Xp_clf = preprocess_dataset(ds.intensities, steps=PREPROCESS_FOR_CLASSIFICATION,
                                verbose=True)
    print("      定量路径: AsLS 基线 + SG 平滑 + 内标峰归一化 (470 cm⁻¹)")
    Xp_reg = preprocess_dataset(ds.intensities, steps=PREPROCESS_FOR_QUANTIFICATION,
                                wavenumbers=ds.wavenumbers, verbose=True)
    viz.plot_preprocessing_comparison(ds, idx=0)
    viz.plot_processed_mean_spectra(ds, Xp_clf)

    # 保存预处理后数据
    save_dataset_csv(ds, config.DATA_PROCESSED / "synthetic_raw.csv")
    np.save(config.DATA_PROCESSED / "Xp_clf.npy", Xp_clf)
    np.save(config.DATA_PROCESSED / "Xp_reg.npy", Xp_reg)

    # ---------- Step 3: 特征工程（PCA）----------
    print("\n[3/7] 特征工程：PCA 降维 ...")
    reducer_clf = PCAReducer(n_components=config.PCA_N_COMPONENTS)
    scores_clf = reducer_clf.fit_transform(Xp_clf)
    n95 = reducer_clf.n_components_for_variance(0.95)
    print(f"      PCA 后维度: {Xp_clf.shape[1]} -> {scores_clf.shape[1]}")
    print(f"      达 95% 方差需 {n95} 个主成分（前 5 个 PC 解释 "
          f"{reducer_clf.cumvar_[4]*100:.1f}%）")

    # optimal_n = reducer_clf.select_n_components_bic(Xp_clf)

    reducer_reg = PCAReducer(n_components=config.PCA_N_COMPONENTS)
    scores_reg = reducer_reg.fit_transform(Xp_reg)

    viz.plot_pca_variance(reducer_clf)
    viz.plot_pca_2d(scores_clf, ds.labels, ds.species_list,
                    reducer_clf.pca.explained_variance_ratio_)

    # ---------- Step 3.1: 特征工程（峰数据+合并）----------
    # 1.提取峰数据
    peak_features, peak_names = extract_peak_features(
        Xp_clf,ds.wavenumbers)

    print(f"      峰特征维度: {peak_features.shape} (17个峰 × 2特征)")
    # 2.标准化，回归不需要标准化
    scaler_pca = StandardScaler()
    scaler_peak = StandardScaler()

    scores_clf = scaler_pca.fit_transform(scores_clf)
    peak_std = scaler_peak.fit_transform(peak_features)

    # 3.拼接特征
    scores_clf = np.hstack([scores_clf, peak_std])
    scores_reg = np.hstack([scores_reg, peak_features])

    # ---------- Step 4: 分类模型对比 ----------
    print("\n[4/7] 分类模型对比 (SVM / RF / KNN / PLS-DA) ...")
    Xtr, Xte, ytr, yte = stratified_split(scores_clf, ds.labels)
    print(f"      训练集 {len(ytr)} 条, 测试集 {len(yte)} 条")
    results = {}
    for name, model in build_classifiers().items():
        print(f"      训练 {name:14s} ...", end=" ", flush=True)
        r = evaluate_classifier(model, Xtr, ytr, Xte, yte)
        results[name] = r
        print(f"Acc={r['accuracy']:.3f}  F1={r['macro_f1']:.3f}  "
              f"t/sample={r['pred_time_per_sample_ms']:.3f} ms")

    score_df = weighted_score(results)
    best_name = score_df["weighted_score"].idxmax()
    print(f"\n      >>> 综合最优: {best_name} "
          f"(加权评分 {score_df.loc[best_name, 'weighted_score']:.3f})")

    # 模型对比图
    viz.plot_model_comparison(score_df)

    # 保存混淆矩阵（最优模型）
    best_results = results[best_name]
    viz.plot_confusion_matrix(yte, best_results["y_pred"], ds.species_list,
                              model_name=best_name)

    # 输出 metrics.csv
    metrics_out = score_df[["accuracy", "macro_f1", "fit_time_s",
                            "pred_time_per_sample_ms", "weighted_score"]].copy()
    metrics_out.to_csv(config.RESULTS / "metrics.csv", encoding="utf-8-sig")
    print(f"      模型对比表已保存: results/metrics.csv")

    # ---------- Step 5: 定量分析 ----------
    print("\n[5/7] 定量分析：PLS 回归预测菌体浓度（5 折 CV）...")
    reg = pls_quantitative_regression(scores_reg, ds.concentrations,
                                      n_components=10, kfold=config.CV_FOLDS)
    print(f"      R²_cv = {reg['r2_cv']:.3f},  RMSE_cv = {reg['rmse_cv']:.3f}")
    viz.plot_pls_regression(reg)

    # ---------- Step 6: 聚类（无监督）----------
    print("\n[6/7] 聚类分析：KMeans vs GMM (k=5) 在 PCA 空间 ...")
    clu_results = compare_clustering(scores_clf, ds.labels)
    for method, r in clu_results.items():
        print(f"      {method:8s}  ARI={r['ARI']:.3f}  NMI={r['NMI']:.3f}")
    # 用 ARI 较高者作为"展示"聚类
    best_clu = max(clu_results.values(), key=lambda x: x["ARI"])
    viz.plot_clustering(scores_clf, ds.labels, best_clu["labels"], ds.species_list, best_clu)

    # ---------- Step 7: 汇总输出 ----------
    print("\n[7/7] 汇总结果 ...")
    summary = {
        "data": {
            "n_samples": int(ds.n_samples),
            "n_wavenumbers": int(ds.n_wavenumbers),
            "wavenumber_range": [float(ds.wavenumbers.min()), float(ds.wavenumbers.max())],
            "n_classes": len(ds.species_list),
            "species": ds.species_list,
        },
        "classification": {
            name: {
                "accuracy": float(r["accuracy"]),
                "macro_f1": float(r["macro_f1"]),
                "pred_time_per_sample_ms": float(r["pred_time_per_sample_ms"]),
                "weighted_score": float(score_df.loc[name, "weighted_score"]),
            } for name, r in results.items()
        },
        "best_model": best_name,
        "quantification": {
            "r2_cv": float(reg["r2_cv"]),
            "rmse_cv": float(reg["rmse_cv"]),
            "n_components": reg["n_components"],
            "kfold": reg["kfold"],
        },
        "clustering": {
            method: {"ARI": float(r["ARI"]), "NMI": float(r["NMI"])}
            for method, r in clu_results.items()
        },
        "pca": {
            "n_components_95pct": int(n95),
            "cumvar_top5_pct": float(reducer_clf.cumvar_[4] * 100),
        },
        "runtime_s": round(time.perf_counter() - t_start, 1),
    }
    with open(config.RESULTS / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"      汇总已保存: results/summary.json")

    print("\n" + "=" * 70)
    print(f"  完成！总用时 {summary['runtime_s']} 秒")
    print(f"  最优分类模型: {best_name}  Acc={results[best_name]['accuracy']:.3f}")
    print(f"  PLS 定量 R²={reg['r2_cv']:.3f}  RMSE={reg['rmse_cv']:.3f}")
    clu_summary = "; ".join(f"{m} ARI={r['ARI']:.3f}" for m, r in clu_results.items())
    print(f"  聚类: {clu_summary}")
    print("=" * 70)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="快速版：每类 100 样本")
    args = parser.parse_args()
    n = 100 if args.quick else config.N_SAMPLES_PER_CLASS
    run(n_per_class=n)
