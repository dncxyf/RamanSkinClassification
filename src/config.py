# -*- coding: utf-8 -*-
"""
项目全局配置
所有可调参数集中在此，便于复现与超参数实验。
"""
from pathlib import Path

# ============== 路径 ==============
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
RESULTS = PROJECT_ROOT / "results"
FIG_DIR = RESULTS / "figures"
MATLAB_DIR = PROJECT_ROOT / "matlab"

for _p in (DATA_RAW, DATA_PROCESSED, RESULTS, FIG_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# ============== 光谱设置 ==============
WAVENUMBER_MIN = 400      # cm^-1
WAVENUMBER_MAX = 1800     # cm^-1  —— 指纹区（生物拉曼常用区间）
WAVENUMBER_STEP = 2.0     # cm^-1 采样间隔
# 真实光谱仪通常 1-4 cm^-1 分辨率；这里 2 cm^-1 -> 701 个通道

# ============== 数据生成 ==============
RANDOM_SEED = 42
N_SAMPLES_PER_CLASS = 500          # 每类样本数
N_CLASSES = 5                       # 5 种皮肤相关微生物
ADD_FLUORESCENCE = True             # 加荧光背景（模拟真实拉曼）
ADD_BASELINE_DRIFT = True           # 加基线漂移
ADD_NOISE = True                    # 加高斯噪声
NOISE_STD_RANGE = (0.06, 0.15)      # 噪声标准差范围（相对最大峰强）—— 平衡难度
PEAK_JITTER_STD = 4.5              # 类内峰位抖动 (cm⁻¹)
AMP_JITTER_STD = 0.28              # 类内强度抖动 std（相对值）
CONCENTRATION_RANGE = (0.3, 1.0)    # 模拟"细菌浓度"因子（用于 PLS 定量）

# ============== 预处理参数 ==============
ASLS_LAMBDA = 1e5                   # AsLS 平滑度参数（典型 1e4-1e7）
ASLS_P = 0.01                       # AsLS 非对称权重（典型 0.001-0.1）
SG_WINDOW = 15                      # Savitzky-Golay 窗口（奇数，对应约 30 cm^-1）
SG_POLYORDER = 3                    # SG 多项式阶数

# ============== 建模参数 ==============
PCA_N_COMPONENTS = 20              # PCA 主成分数（后续看解释方差再裁剪）
TEST_SIZE = 0.25
CV_FOLDS = 5

# ============== 可视化 ==============
FIG_DPI = 150
FIG_FMT = ["png", "pdf"]            # 同时输出位图与矢量图

# 5 种皮肤相关微生物配色（含色盲友好色）
CLASS_COLORS = {
    "Staphylococcus aureus":    "#E64B35",   # 金黄色葡萄球菌（红）
    "Staphylococcus epidermidis": "#4DBBD5", # 表皮葡萄球菌（蓝）
    "Pseudomonas aeruginosa":   "#00A087",   # 铜绿假单胞菌（绿）
    "Escherichia coli":         "#3C5488",   # 大肠杆菌（深蓝）
    "Klebsiella pneumoniae":    "#F39B7F",   # 肺炎克雷伯菌（橙）
}

CLASS_LABEL_ZH = {
    "Staphylococcus aureus":    "金黄色葡萄球菌",
    "Staphylococcus epidermidis": "表皮葡萄球菌",
    "Pseudomonas aeruginosa":   "铜绿假单胞菌",
    "Escherichia coli":         "大肠杆菌",
    "Klebsiella pneumoniae":    "肺炎克雷伯菌",
}

# matplotlib 中文字体配置（Windows 微软雅黑）
MATPLOTLIB_FONT = {
    "family": "Microsoft YaHei",
    "size": 10,
}
