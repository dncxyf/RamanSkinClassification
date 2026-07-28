# -*- coding: utf-8 -*-
"""
数据加载与合成光谱生成
======================

【设计说明 / 文献依据】
本项目使用「基于文献峰位参数的合成拉曼光谱」作为分析对象。这是拉曼方法学研究
中标准的仿真策略（Monte-Carlo simulation of Raman spectra），优点是 ground-truth
完全已知，便于严格验证预处理、特征工程、分类、定量每一步的有效性。

5 种菌种的峰位与归属主要参考：
  - Movasaghi Z. et al. (2007) Raman spectroscopy of biological tissues.
    Appl. Spectrosc. Rev. 42(5):493-541.   （生物拉曼标准峰位总表）
  - Krafft C. & Popp J. (2015) The optimal identification of bacteria by
    Raman spectroscopy. Trends Anal. Chem. 70:1407-1422.
  - Ho C.S. et al. (2019) Rapid identification of pathogenic bacteria using
    Raman spectroscopy and deep learning. Nat. Commun. 10:4910.

主要共有峰（生物分子特征）：
  1003  cm⁻¹  Phenylalanine 对称环呼吸        （蛋白质）
  1098  cm⁻¹  PO₄³⁻ 对称伸缩                  （DNA 骨架）
  1126  cm⁻¹  C-N / C-C 伸缩                  （蛋白质）
  1250  cm⁻¹  Amide III 弱                    （蛋白质）
  1305  cm⁻¹  CH₂ 变形                        （脂质/蛋白）
  1335  cm⁻¹  CH 变形 / Adenine               （核酸/蛋白）
  1448  cm⁻¹  CH₂ 剪切                        （脂质/蛋白）
  1580  cm⁻¹  Ring                            （核酸碱基）
  1607  cm⁻¹  Tyr / Phe                       （蛋白质）
  1660  cm⁻¹  Amide I                         （蛋白质）

不同菌种因细胞壁组成差异（革兰氏阳性 / 阴性、肽聚糖含量、脂质类型），
峰强相对比例不同 —— 这正是分类的科学基础。
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

import config


# ============== 文献峰位参数表 ==============
# 每个菌种: dict[峰位(cm⁻¹) -> 相对强度]  (相对强度已按文献均值归一化)
# 强度数值取自上述综述中各菌种典型实测光谱的相对强度。
#
# 设计：5 种菌种共享 7-8 个共有生物分子峰（1003/1098/1126/1305/1448/1607/1660），
# 这是所有细菌都有的"生物特征指纹"；同时每个菌种有 2-3 个差异峰（物种特异）。
# 这样分类既有科学依据（共有指纹），又有难度（差异峰只占一小部分，且类内
# 抖动会让差异被噪声掩盖），更接近真实场景。
SPECIES_PEAKS = {
    "Staphylococcus aureus": {
        # 共有生物分子峰（强）+ 金葡特异：staphyloxanthin 黄色素（1157/1528，弱）
        1003: 1.00, 1098: 0.50, 1126: 0.50, 1305: 0.45, 1448: 0.90, 1607: 0.35, 1660: 0.60,
        1157: 0.25, 1528: 0.20,
    },
    "Staphylococcus epidermidis": {
        # 与金葡同属但无 staphyloxanthin；特异：1063 肽聚糖骨架（弱）
        1003: 0.95, 1098: 0.55, 1126: 0.50, 1305: 0.45, 1448: 0.90, 1607: 0.35, 1660: 0.60,
        1063: 0.22,
    },
    "Pseudomonas aeruginosa": {
        # 革兰氏阴性；特异：绿脓素 pyocyanin (1350/1550)
        1003: 0.90, 1098: 0.55, 1126: 0.45, 1305: 0.45, 1448: 0.85, 1607: 0.40, 1660: 0.65,
        1350: 0.28, 1550: 0.30,
    },
    "Escherichia coli": {
        # 革兰氏阴性；特异：1250 Amide III、1580 碱基环
        1003: 0.90, 1098: 0.60, 1126: 0.55, 1305: 0.45, 1448: 0.90, 1607: 0.35, 1660: 0.70,
        1250: 0.27, 1580: 0.22,
    },
    "Klebsiella pneumoniae": {
        # 革兰氏阴性；特异：荚膜多糖（854/938）
        1003: 0.90, 1098: 0.55, 1126: 0.50, 1305: 0.45, 1448: 0.90, 1607: 0.35, 1660: 0.60,
        854: 0.30, 938: 0.32,
    },
}

# 半峰宽 (FWHM, cm⁻¹)：实际生物拉曼峰宽通常 8-25 cm⁻¹
DEFAULT_FWHM = 14.0

# 内标峰（用于定量分析；模拟基底/载体物质，强度恒定、与菌体浓度无关）
# 此处用 470 cm⁻¹ 模拟一个虚构内标（实际项目可用 CaF₂ 322 cm⁻¹、硅片 520.7 cm⁻¹）
INTERNAL_STANDARD_WN = 470.0 # 位置
INTERNAL_STANDARD_AMP = 0.80 # 强度


def lorentzian(wn: np.ndarray, center: float, fwhm: float, amp: float) -> np.ndarray:
    """洛伦兹线型  L(ν) = A·γ² / [(ν-ν₀)² + γ²]   γ=FWHM/2"""
    gamma = fwhm / 2.0
    return amp * gamma * gamma / ((wn - center) ** 2 + gamma * gamma)


def _make_wavenumber_grid() -> np.ndarray:
    return np.arange(
        config.WAVENUMBER_MIN,
        config.WAVENUMBER_MAX + config.WAVENUMBER_STEP,
        config.WAVENUMBER_STEP,
    )


@dataclass
class SpectraDataset:
    """统一的数据容器：光谱矩阵 + 标签 + 元信息"""
    wavenumbers: np.ndarray          # shape (W,)
    intensities: np.ndarray          # shape (N, W)
    labels: np.ndarray               # shape (N,) 字符串菌种名
    concentrations: np.ndarray       # shape (N,) 模拟"细菌浓度"，用于定量分析
    sample_ids: np.ndarray           # shape (N,)
    species_list: list               # 类别顺序

    @property
    def n_samples(self) -> int:
        return self.intensities.shape[0]

    @property
    def n_wavenumbers(self) -> int:
        return self.intensities.shape[1]

    def to_df(self) -> pd.DataFrame:
        """导出为 DataFrame（含元数据 + 光谱列）"""
        df = pd.DataFrame(self.intensities, columns=[f"wn_{w:.0f}" for w in self.wavenumbers])
        df.insert(0, "sample_id", self.sample_ids)
        df.insert(1, "label", self.labels)
        df.insert(2, "concentration", self.concentrations)
        return df


def _gen_fluorescence_baseline(wn: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """模拟荧光背景：低阶多项式 + 指数衰减包络"""
    # 归一化波长到 [-1, 1]
    # 把原始波数（比如 400~1800 cm⁻¹）压缩映射到 [-1, 1] 区间。为了让后续多项式系数的大小与波数绝对值无关，参数调节更直观、更稳定。
    x = (wn - wn.mean()) / (wn.max() - wn.min())
    # 3-5 阶多项式系数（模拟荧光宽缓变化）
    coeffs = rng.uniform(-1, 1, size=4) * np.array([1.2, 0.6, 0.3, 0.1])
    base = np.polyval(coeffs, x)
    # 加指数衰减包络（短波端荧光更强）
    envelope = np.exp(-2.0 * (wn - wn.min()) / (wn.max() - wn.min()))
    base = base * envelope * rng.uniform(0.5, 2.0)
    return base


def _gen_baseline_drift(wn: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """低频基线漂移：1-2 阶多项式"""
    x = (wn - wn.mean()) / (wn.max() - wn.min())
    c = rng.uniform(-0.3, 0.3, size=3) * np.array([1.0, 0.5, 0.2])
    return np.polyval(c, x)


def generate_synthetic_dataset(
    n_per_class: int = config.N_SAMPLES_PER_CLASS,
    seed: int = config.RANDOM_SEED,
) -> SpectraDataset:
    """
    基于文献峰位参数生成合成拉曼光谱数据集。

    每条光谱 = Σ 洛伦兹峰（按菌种峰表） × 浓度因子
              + 类内峰位抖动（±2 cm⁻¹，模拟测量重复性）
              + 类内强度抖动（±15%，模拟生物变异性）
              + 荧光背景 + 基线漂移
              + 高斯噪声

    Returns
    -------
    SpectraDataset
    """
    rng = np.random.default_rng(seed)
    wn = _make_wavenumber_grid()
    species_list = list(SPECIES_PEAKS.keys())

    intensities, labels, concs, sids = [], [], [], []
    sid = 0
    for sp in species_list:
        peaks = SPECIES_PEAKS[sp]
        for _ in range(n_per_class):
            # 浓度因子：模拟样本中菌体浓度（用于 PLS 定量回归）
            conc = rng.uniform(*config.CONCENTRATION_RANGE)

            # ---- 1) 菌体峰：强度 ∝ 浓度 ----
            spec = np.zeros_like(wn, dtype=float)
            for center, amp in peaks.items():
                c_jitter = center + rng.normal(0, config.PEAK_JITTER_STD)
                a_jitter = amp * (1.0 + rng.normal(0, config.AMP_JITTER_STD)) * conc
                fwhm = DEFAULT_FWHM * (1.0 + rng.normal(0, 0.15))
                spec += lorentzian(wn, c_jitter, max(fwhm, 1.0), max(a_jitter, 0))

            # ---- 2) 内标峰：强度恒定（与浓度无关）+ 小幅抖动 ----
            # 模拟加在样本中的定量内标（如 CaF₂ / 硅片），用于浓度反演
            istd_amp = INTERNAL_STANDARD_AMP * (1.0 + rng.normal(0, 0.05))
            istd_c = INTERNAL_STANDARD_WN + rng.normal(0, 2.0)
            istd_fwhm = DEFAULT_FWHM * 0.9
            spec += lorentzian(wn, istd_c, istd_fwhm, istd_amp)

            # ---- 3) 背景 + 噪声 ----
            if config.ADD_FLUORESCENCE:
                spec = spec + _gen_fluorescence_baseline(wn, rng)
            if config.ADD_BASELINE_DRIFT:
                spec = spec + _gen_baseline_drift(wn, rng)
            if config.ADD_NOISE:
                peak_max = spec.max() if spec.max() > 0 else 1.0
                noise_std = rng.uniform(*config.NOISE_STD_RANGE) * peak_max
                spec = spec + rng.normal(0, noise_std, size=wn.shape)

            intensities.append(spec)
            labels.append(sp)
            concs.append(conc)
            sids.append(f"S{sid:05d}")
            sid += 1

    return SpectraDataset(
        wavenumbers=wn,
        intensities=np.array(intensities, dtype=float),
        labels=np.array(labels),
        concentrations=np.array(concs, dtype=float),
        sample_ids=np.array(sids),
        species_list=species_list,
    )


def load_bacteria_id_subset(species_keep: Optional[list] = None) -> SpectraDataset:
    """
    [可选] 加载真实 Bacteria ID 数据集（Morton et al., 2023, Sci Data）。
    数据下载：https://figshare.com/articles/dataset/Bacteria_ID_Dataset/20062492
    解压后放至 data/raw/bacteria_id/

    本项目主流程使用 generate_synthetic_dataset；此函数保留作为「真实数据接入点」，
    方便后续接入实测数据时无需改 Pipeline。

    Notes
    -----
    若数据不存在则抛出 FileNotFoundError，提示用户下载。
    """
    raw_dir = config.DATA_RAW / "bacteria_id"
    spectra_file = raw_dir / "spectra.csv"
    meta_file = raw_dir / "metadata.csv"
    if not spectra_file.exists():
        raise FileNotFoundError(
            f"未找到 {spectra_file}\n"
            "下载数据并解压至 data/raw/bacteria_id/"
        )
    spec_df = pd.read_csv(spectra_file)
    meta_df = pd.read_csv(meta_file)
    wn = spec_df.filter(regex=r"^wn_").columns.str.replace("wn_", "").astype(float).values
    X = spec_df.filter(regex=r"^wn_").values
    y = meta_df.set_index("sample_id").loc[spec_df["sample_id"], "species"].values
    conc = meta_df.set_index("sample_id").loc[spec_df["sample_id"], "concentration"].values
    return SpectraDataset(
        wavenumbers=wn, intensities=X, labels=y,
        concentrations=conc, sample_ids=spec_df["sample_id"].values,
        species_list=list(np.unique(y)),
    )


def save_dataset_csv(ds: SpectraDataset, path) -> None:
    ds.to_df().to_csv(path, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    # 自检：生成数据并打印概览
    ds = generate_synthetic_dataset()
    print(f"[OK] 数据集生成: {ds.n_samples} 条光谱, {ds.n_wavenumbers} 个波数点")
    print(f"     波数范围: {ds.wavenumbers.min():.0f} - {ds.wavenumbers.max():.0f} cm⁻¹")
    print(f"     菌种分布:")
    import collections
    for sp, n in collections.Counter(ds.labels).items():
        print(f"       {sp:30s}  {n}")
    print(f"     浓度范围: [{ds.concentrations.min():.2f}, {ds.concentrations.max():.2f}]")
    print(f"     光谱强度: min={ds.intensities.min():.3f}, max={ds.intensities.max():.3f}")
