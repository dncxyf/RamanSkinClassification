# MATLAB 预处理脚本

本目录包含拉曼光谱预处理 Pipeline 三个核心步骤的 MATLAB 实现，
与 Python 版本 (`src/preprocess.py`) 语义等价。

## 文件说明

| 文件 | 功能 | 对应 Python |
|---|---|---|
| `preprocess_asls.m` | AsLS 基线校正 | `preprocess.py → asls_baseline()` |
| `preprocess_sg.m` | Savitzky-Golay 平滑 | `preprocess.py → savgol_smooth()` |
| `preprocess_snv.m` | SNV 归一化 | `preprocess.py → snv()` |

## 运行环境

- MATLAB R2018b 及以上（需要 `spdiags`、`sgolayfilt`，均为内置函数）
- 或 GNU Octave 6.0+（`sgolayfilt` 需 `signal` 包）

## 快速验证

在 MATLAB 命令窗口中运行：

```matlab
% 生成一条模拟光谱 + 荧光背景
wn = 400:2:1800;
y = 0.8 * exp(-((wn - 1003).^2) / (2 * 14^2)) + ...  % Phenylalanine 峰
    0.5 * exp(-((wn - 1448).^2) / (2 * 14^2)) + ...  % CH2 剪切
    polyval([0.02 -0.5 1.2], (wn - 1100) / 1400);    % 荧光背景

% Step 1: AsLS 基线校正
[baseline, y_corrected] = preprocess_asls(y, 1e5, 0.01, 20);

% Step 2: SG 平滑
y_smooth = preprocess_sg(y_corrected, 15, 3);

% Step 3: SNV 归一化
y_snv = preprocess_snv(y_smooth);

% 可视化对比
figure('Position', [100 100 900 500]);
subplot(2,2,1); plot(wn, y, 'b', wn, baseline, 'r--');
title('(a) 原始光谱 + AsLS 基线');
subplot(2,2,2); plot(wn, y_corrected, 'b');
title('(b) 基线校正后');
subplot(2,2,3); plot(wn, y_smooth, 'b');
title('(c) SG 平滑后');
subplot(2,2,4); plot(wn, y_snv, 'b');
title('(d) SNV 归一化后');
sgtitle('MATLAB 预处理 Pipeline 演示');
```

## 与 Python 版本的一致性验证

Python 版 Pipeline 在 2500 条合成光谱上已验证通过。MATLAB 版由于
使用相同的算法（AsLS 迭代加权最小二乘、SG 多项式拟合、SNV z-score），
在相同参数下应得到数值等价的结果（浮点精度 ~1e-10 内一致）。

若需严格数值对比：
1. 在 Python 中保存一条测试光谱：`np.save("test_spectrum.npy", spectrum)`
2. 在 MATLAB 中读取：`y = double(py.numpy.load('test_spectrum.npy'));`
3. 分别运行 Python/MATLAB Pipeline，对比输出
