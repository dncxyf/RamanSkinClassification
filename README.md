# 皮肤拉曼光谱分类与定量分析 Pipeline

基于拉曼光谱的皮肤相关微生物（5 种细菌）分类识别与浓度定量分析完整流程。

## 项目概要

| 项目 | 内容 |
|---|---|
| **任务** | 拉曼光谱多分类 + PLS 定量回归 + 无监督聚类 |
| **数据** | 基于文献峰位参数的合成拉曼光谱（5 类 × 500 条 = 2500 条） |
| **预处理** | AsLS 基线校正 → SG 平滑 → SNV 归一化（分类）/ 内标峰归一化（定量） |
| **分类** | PLS-DA / SVM(RBF) / RandomForest / KNN 对比，三维度加权评分 |
| **定量** | PLS 回归（5 折 CV），R²=0.786 |
| **语言** | Python（主流程）+ MATLAB（预处理核心步骤复现） |

## 核心结果

```
分类模型对比（2500 条，5 类，75/25 分层划分）:
  PLS-DA       Acc=0.941  F1=0.941  综合★
  SVM(RBF)     Acc=0.931  F1=0.931  
  RandomForest Acc=0.933  F1=0.933
  KNN          Acc=0.923  F1=0.923  (基线)

PLS 定量:  R²(CV)=0.786  RMSE(CV)=0.093
PCA:       前 5 个 PC 解释 30.5%，达 95% 需 110 个主成分
聚类:      KMeans ARI=0.834  GMM ARI=0.818  (监督 > 无监督)
```

## 目录结构

```
raman-skin-classification/
├── README.md                     ← 你在这里
├── requirements.txt              Python 依赖
├── src/                          Python 源码
│   ├── config.py                 全局配置（参数集中管理）
│   ├── data_loader.py            数据生成 + 真实数据加载接口
│   ├── preprocess.py             预处理 Pipeline（核心）
│   ├── feature.py                特征工程（PCA + 峰特征）
│   ├── models.py                 分类/回归/聚类模型
│   ├── visualize.py              可视化（9 张图）
│   └── pipeline.py               一键运行入口
├── matlab/                       MATLAB 预处理脚本
│   ├── preprocess_asls.m         AsLS 基线校正
│   ├── preprocess_sg.m           SG 平滑
│   ├── preprocess_snv.m          SNV 归一化
│   └── README.md                 MATLAB 说明
├── results/
│   ├── figures/                  所有图（PNG + PDF 矢量）
│   ├── metrics.csv               模型对比表
│   └── summary.json              结构化结果汇总
└── data/
    ├── raw/                      原始数据（.gitignore）
    └── processed/                 预处理后数据
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 一键运行完整 Pipeline（2500 条，约 16 秒）
python src/pipeline.py

# 3. 快速调试版（每类 100 条，约 8 秒）
python src/pipeline.py --quick
```

运行后输出：
- `results/figures/*.png` — 9 张分析图
- `results/metrics.csv` — 模型对比表
- `results/summary.json` — 全量结果 JSON

## 数据说明

### 合成光谱（本项目主流程）

本项目使用**基于文献峰位参数的合成拉曼光谱**，这是拉曼方法学研究中标准的仿真策略。

**5 种皮肤相关微生物**：
| 菌种 | 类型 | 特征峰（物种特异） |
|---|---|---|
| *Staphylococcus aureus* | G+ | staphyloxanthin 1157/1528 cm⁻¹ |
| *Staphylococcus epidermidis* | G+ | 肽聚糖骨架 1063 cm⁻¹ |
| *Pseudomonas aeruginosa* | G- | pyocyanin 1350/1550 cm⁻¹ |
| *Escherichia coli* | G- | Amide III 1250/碱基环 1580 cm⁻¹ |
| *Klebsiella pneumoniae* | G- | 荚膜多糖 854/938 cm⁻¹ |

**共有生物分子峰**（所有菌种共享）：1003 (Phe), 1098 (DNA), 1126 (C-N), 1305 (CH₂), 1448 (CH₂), 1607 (Tyr), 1660 (Amide I)

**噪声模型**：荧光背景（多项式 + 指数包络）+ 基线漂移 + 高斯噪声 + 峰位/峰强类内抖动 + 内标峰（470 cm⁻¹，用于定量）

**峰位参数文献来源**：
- Movasaghi Z. et al. (2007) Raman spectroscopy of biological tissues. *Appl. Spectrosc. Rev.* 42(5):493-541
- Krafft C. & Popp J. (2015) The optimal identification of bacteria by Raman spectroscopy. *Trends Anal. Chem.* 70:1407-1422

### 真实数据接入

代码预留了 Bacteria ID 数据集的加载接口：
```python
from data_loader import load_bacteria_id_subset
ds = load_bacteria_id_subset()  # 需下载数据至 data/raw/bacteria_id/
```

## 方法学要点

### 预处理 Pipeline

```
原始光谱 → [1] AsLS 基线校正 → [2] SG 平滑 → [3] 归一化 → 建模
```

**分类 vs 定量使用不同的归一化策略**：
- **分类**用 SNV：消除整体强度差异，凸显峰形差异
- **定量**用内标峰归一化（470 cm⁻¹）：保留浓度-强度线性关系，PLS 回归才能有效

### 模型对比框架

三维度加权评分：`Score = 0.4·norm(Acc) + 0.4·norm(F1) + 0.2·norm(速度)`

### 聚类结论

在同时使用PCA+峰数据时，聚类效果较好。消融实验显示，远好于单独PCA（接近随机）和单独峰数据。
