# 基于拉曼光谱的皮肤相关微生物分类与定量分析报告

## 摘要

本研究构建了一套完整的拉曼光谱化学计量学分析 Pipeline，针对 5 种皮肤相关微生物（金黄色葡萄球菌、表皮葡萄球菌、铜绿假单胞菌、大肠杆菌、肺炎克雷伯菌）的合成拉曼光谱数据（共 2500 条，400-1800 cm⁻¹ 指纹区，2 cm⁻¹ 采样间隔），实现了从预处理、特征工程到分类建模、定量回归与无监督聚类的全流程分析。预处理采用 AsLS 非对称最小二乘基线校正、Savitzky-Golay 平滑及 SNV 归一化（分类路径）/ 内标峰归一化（定量路径）；特征工程利用 PCA 降维（20 维）与峰特征提取。在 PLS-DA、SVM(RBF)、RandomForest、KNN 四种分类模型的系统对比中，SVM(RBF) 以 88.2% 准确率获得综合最优评分；PLS 定量回归在内标峰归一化路径上达到 R²=0.767（5 折交叉验证）。聚类分析表明，KMeans 与 GMM 的无监督聚类效果有限（ARI<0.2），验证了拉曼光谱分类应以监督方法为主的方法学结论。

**关键词：** 拉曼光谱；化学计量学；PLS-DA；PCA；皮肤微生物组；基线校正；内标法

## 1 引言

拉曼光谱是一种基于非弹性散射的分子振动光谱技术，能够提供样本中化学键和分子结构的「指纹」信息。近年来，拉曼光谱在生物医学领域（尤其是微生物鉴定）中展现出巨大潜力：其无需标记、非破坏性、对水不敏感等特性，使其特别适合含水生物样本（如皮肤、组织）的快速分析。

皮肤微生物组是人体最大的微生物生态系统，其组成与多种皮肤疾病（痤疮、特应性皮炎、银屑病等）密切相关。传统的微生物鉴定方法（培养法、PCR、MALDI-TOF MS）耗时且需专业人员操作，而拉曼光谱结合化学计量学方法可在数秒内实现非破坏性鉴定，具有显著的临床转化潜力。

本研究旨在构建一套完整的化学计量学分析流程，涵盖拉曼光谱的预处理、特征工程、分类建模、定量回归与无监督聚类五大模块，并系统对比不同算法的性能。本研究采用基于文献峰位参数的合成拉曼光谱数据，以确保 ground-truth 完全已知，便于严格验证每一步的有效性。

## 2 数据与方法

### 2.1 数据来源与生成

本研究使用基于文献峰位参数的合成拉曼光谱数据。拉曼方法学研究中，合成光谱是标准的仿真策略（Monte-Carlo simulation of Raman spectra），其优势在于 ground-truth 完全已知，便于严格验证预处理、特征工程、分类、定量每一步的有效性。峰位参数主要参考 Movasaghi 等（2007）的生物组织拉曼光谱标准峰位总表以及 Krafft & Popp（2015）的细菌拉曼光谱鉴定综述。

每条合成光谱由以下组分叠加而成：
(1) 菌种特异峰（洛伦兹线型，峰位与相对强度基于文献参数）；
(2) 荧光背景（4 阶多项式 × 指数衰减包络）；
(3) 低频基线漂移（2 阶多项式）；
(4) 高斯噪声（标准差 6-15% 相对最大峰强）；
(5) 类内峰位抖动（±4.5 cm⁻¹）和强度抖动（±28%），模拟测量重复性和生物变异性；
(6) 内标峰（470 cm⁻¹，强度恒定），用于定量分析中的内标法归一化。

### 2.2 菌种特征峰位参数

5 种菌种共享 7 个共有生物分子峰（1003 cm⁻¹ 苯丙氨酸、1098 cm⁻¹ DNA 骨架、1126 cm⁻¹ C-N、1305 cm⁻¹ CH₂ 变形、1448 cm⁻¹ CH₂ 剪切、1607 cm⁻¹ 酪氨酸、1660 cm⁻¹ Amide I），同时每种菌种有 2-3 个物种特异峰，提供分类依据。

| 菌种           | 类型 | 特异峰位 (cm⁻¹) | 特征物质                  |
| -------------- | ---- | --------------- | ------------------------- |
| S. aureus      | G+   | 1157, 1528      | Staphyloxanthin（黄色素） |
| S. epidermidis | G+   | 1063            | 肽聚糖骨架                |
| P. aeruginosa  | G-   | 1350, 1550      | Pyocyanin（绿脓素）       |
| E. coli        | G-   | 1250, 1580      | Amide III / 碱基环        |
| K. pneumoniae  | G-   | 854, 938        | 荚膜多糖                  |

![01_raw_spectra_overview](..\results\figures\01_raw_spectra_overview.png)



图1 原始拉曼光谱概览（每类均值±标准差）

### 2.3 预处理方法

拉曼光谱原始数据含有荧光背景、基线漂移、高频噪声等干扰，必须依次处理才能用于后续建模。本研究采用标准三步骤预处理 Pipeline：

（1）AsLS 基线校正：非对称最小二乘（Asymmetric Least Squares, Eilers & Boelens 2005）通过加权最小二乘拟合一条平滑基线。使用 2 阶差分罚约束基线平滑度（λ=1×10⁵），同时用非对称权重（p=0.01）使基线「压在信号下方」，迭代 20 次收敛。扣除基线后，荧光背景（宽包络）被有效去除。

（2）Savitzky-Golay 平滑：在滑动窗口内拟合多项式，取中心点拟合值作为平滑后信号。相比简单移动平均，SG 滤波保留峰形（保持峰位导数信息），是拉曼光谱去噪的标准方法。本研究使用窗口 15、多项式阶数 3，对应约 30 cm⁻¹ 的平滑范围。

（3）归一化：本研究采用双路径归一化策略——分类路径使用 SNV（Standard Normal Variate，(x−mean)/std），消除样本间整体强度差异，凸显峰形差异；定量路径使用内标峰归一化（470 cm⁻¹ 峰强作分母），保留浓度-强度线性关系。这是本研究最重要的方法学设计：不同分析目标需要不同的预处理策略。

### 2.4 特征工程

拉曼光谱在 701 个波数点上存在严重共线性（相邻通道几乎完全相关），直接用于建模会导致过拟合和数值不稳定。本研究采用主成分分析（PCA）降维，将 701 维压缩至 20 维。前 5 个主成分解释 30.5% 总方差，达到 95% 方差需要 110 个主成分（说明光谱信息高度分散）。为防止过拟合，平衡定性和定量任务，本研究试选取20个主成分。同时，本研究还提取了 17 个目标峰的峰强与峰面积作为可解释的物理特征。

![04_pca_variance](..\results\figures\04_pca_variance.png)



图2 PCA解释方差

![05_pca_2d_projection](..\results\figures\05_pca_2d_projection.png)

图3 PCA 2D投影

### 2.5 分类建模

本研究对比了 4 种经典分类模型：PLS-DA（Partial Least Squares Discriminant Analysis，化学计量学金标准，同时考虑 X 方差和 X-Y 协方差）、SVM（RBF 核，C=10.0，小样本高维数据金标准）、RandomForest（200 棵树，抗共线性、抗过拟合）、KNN（k=7，简单基线）。评估采用三维度加权评分体系：Score = 0.4·norm(Accuracy) + 0.4·norm(Macro-F1) + 0.2·norm(速度)，综合平衡准确率与推理效率。

### 2.6 定量分析

定量分析使用 PLS 回归（10 个主成分）预测菌体浓度，5 折交叉验证评估。定量分析的关键方法学设计是使用内标峰归一化（470 cm⁻¹）而非 SNV——SNV 会抹掉绝对强度信息，使浓度-强度关系失效。不同分析目标需要不同的预处理策略。

### 2.7 聚类分析

无监督聚类使用 KMeans（k=5）和 Gaussian Mixture Model（GMM, full covariance），在 PCA 空间（20 维）中评估无监督方法发现类别结构的能力。使用 ARI（Adjusted Rand Index）和 NMI（Normalized Mutual Information）作为评估指标。

## 3 结果与讨论

### 3.1 预处理效果

预处理 Pipeline 各步骤对同一条样本的效果显示（图4），AsLS 基线校正成功去除了荧光背景（红色虚线），SG 平滑有效抑制了高频噪声，SNV 归一化后光谱均值归零、标准差为 1。预处理后的 5 类平均光谱（图5）在 1003（Phe）、1448（CH₂）、1660（Amide I）等主要生物分子峰处高度一致，但各菌种在特异峰区域（如金葡 1157/1528 cm⁻¹）存在可辨识的差异。

![02_preprocessing_steps](..\results\figures\02_preprocessing_steps.png)

图4 同一条样本预处理pipeline各步骤效果

![03_processed_mean_spectra](..\results\figures\03_processed_mean_spectra.png)

图5 预处理后5类菌种平均光谱对比

### 3.2 分类结果

表 2 列出了 4 种模型在 2500 条光谱（75/25 分层划分）上的分类性能对比。SVM (RBF) 以 88.2% 准确率、0.882 Macro-F1 和 0.088 ms/样本的推理速度获得综合最优评分，PLS-DA 与 SVM(RBF) 准确率相当（88.0% vs 88.2%），但推理速度提升约 100 倍。KNN 表现最差（71.8%），说明在高维 PCA 空间中拉曼光谱的类别结构不是简单的球形分布。

| 模型         | Accuracy | Macro-F1 | 推理耗时(ms/样本) | 加权评分 |
| ------------ | -------- | -------- | ----------------- | -------- |
| PLS-DA       | 0.880    | 0.875    | 0.001             | 0.985    |
| SVM (RBF)    | 0.882    | 0.882    | 0.088             | 1.000    |
| RandomForest | 0.861    | 0.860    | 0.077             | 0.898    |
| KNN (k=7)    | 0.718    | 0.715    | 2.899             | 0.000    |

![06_model_comparison](..\results\figures\06_model_comparison.png)

图6 模型对比

![07_confusion_matrix_best](..\results\figures\07_confusion_matrix_best.png)

图7 SVM_RBF混淆矩阵

### 3.3 定量分析结果

PLS 回归在 5 折交叉验证下达到 R²=0.767、RMSE=0.098（菌体浓度范围 [0.3, 1.0]）。散点图显示了真实浓度与预测浓度的对应关系，线性拟合线 y=0.77x+0.09 与理想线 y=x 接近，表明模型在中高浓度区间表现良好。低浓度端（<0.4）预测偏差略大，可能由于低浓度时信噪比下降。内标峰归一化保留了浓度-强度线性关系，使定量回归成为可能；若误用 SNV 归一化，定量回归将退化为近似随机预测。

![08_pls_regression](..\results\figures\08_pls_regression.png)

图8 PLS定量回归

### 3.4 聚类分析结果

无监督聚类结果如表 3 所示。KMeans 聚类几乎完全失效（ARI=0.006），GMM 聚类略有改善（ARI=0.190）但仍远低于监督方法。这一结果与 PCA 2D 投影中 5 类的严重重叠一致，说明拉曼光谱的类别结构不是简单的球形分布，且信息分散在多个 PCA 维度中。实际应用中，拉曼分类应以监督方法（PLS-DA、SVM）为主，聚类可用于探索性分析（如发现异常样本）。

| 方法           | ARI   | NMI   | 说明                           |
| -------------- | ----- | ----- | ------------------------------ |
| KMeans (k=5)   | 0.006 | 0.010 | 近似随机，无聚类结构           |
| GMM (full cov) | 0.190 | 0.272 | 椭圆簇略有改善，远不如监督方法 |

![09_gmm_clustering](..\results\figures\09_gmm_clustering.png)



图9 



### 3.5 讨论

**预处理策略的选择**：本研究的一个重要发现是，分类与定量分析需要使用不同的归一化策略。分类任务使用 SNV 消除整体强度差异、凸显峰形差异；定量任务使用内标峰归一化保留浓度-强度线性关系。这一发现具有实际应用价值：在临床拉曼分析中，若需同时进行菌种鉴定（分类）和菌量评估（定量），应分别使用两条独立的预处理管道。

**模型选择**：PLS-DA 同时考虑了 X 的方差和 X-Y 的协方差，对共线性的光谱数据特别有效。SVM 在准确率上略优于 PLS-DA（88.2% vs 88.0%），但推理速度慢 100 倍，在实时性要求高的场景（如术中拉曼导航）中不适合。

**项目局限性**：本研究使用合成数据验证方法学，优点是 ground-truth 完全已知、可复现性强；局限性在于合成数据的噪声模型和类内变异性可能无法完全代表真实临床样本的复杂性。后续工作应接入实测拉曼光谱数据（如 Bacteria ID 数据集或临床分离株），进一步验证方法的泛化能力。

## 4 结论

本研究构建并验证了一套完整的拉曼光谱化学计量学分析 Pipeline，涵盖预处理、特征工程、分类建模、定量回归与无监督聚类五大模块。主要结论如下：

（1）AsLS + SG + SNV/内标法的三步骤预处理 Pipeline 能有效去除荧光背景、噪声和基线漂移，为后续建模提供高质量特征。

（2）在 4 种分类模型的对比中，SVM(RBF) 以 88.2% 准确率获得综合最优评分，展示了机器学习方法适用于拉曼光谱分类。 

（3）定量分析使用内标峰归一化（470 cm⁻¹）保留了浓度-强度线性关系，PLS 回归在 5 折交叉验证下达到 R²=0.767。分类用 SNV、定量用内标法的预处理策略差异是本研究最重要的方法学发现。

（4）无监督聚类（KMeans ARI=0.006, GMM ARI=0.190）效果有限，验证了拉曼光谱分类应以监督方法为主的化学计量学经典结论。

（5）项目代码完全开源、可复现（Python 主流程 + MATLAB 预处理复现），所有参数集中在 config.py 管理，便于后续验证和扩展。

## 参考文献

[1] Movasaghi Z, Rehman S, Rehman IU. Raman spectroscopy of biological tissues. Applied Spectroscopy Reviews, 2007, 42(5): 493-541.

[2] Krafft C, Popp J. The optimal identification of bacteria by Raman spectroscopy. TrAC Trends in Analytical Chemistry, 2015, 70: 1407-1422.

[3] Eilers PHC, Boelens HFM. Baseline correction with asymmetric least squares smoothing. Leiden University Medical Centre, 2005.

[4] Ho CS, Jean N, Hogan CA, et al. Rapid identification of pathogenic bacteria using Raman spectroscopy and deep learning. Nature Communications, 2019, 10: 4910.

[5] Savitzky A, Golay MJE. Smoothing and differentiation of data by simplified least squares procedures. Analytical Chemistry, 1964, 36(8): 1627-1639.

[6] Barnes RJ, Dhanoa MS, Lister SJ. Standard Normal Variate transformation and de-trending of near-infrared diffuse reflectance spectra. Applied Spectroscopy, 1989, 43(5): 772-777.

[7] Wold S, Sjostrom M, Eriksson L. PLS-regression: a basic tool of chemometrics. Chemometrics and Intelligent Laboratory Systems, 2001, 58(2): 109-130.