%% preprocess_snv.m
%  SNV (Standard Normal Variate) 归一化
%  ====================================================
%  方法：标准化学计量学归一化
%  原理：每条光谱独立做 z-score 标准化 (x - mean) / std。
%        消除样本间整体强度差异（由采样位置、光程、浓度引起），
%        使后续分类/聚类聚焦于"峰形差异"而非"绝对强度差异"。
%
%  输入：
%    y : N×M 光谱矩阵（N 条光谱，M 个波数点）；也接受 1×M 单条光谱
%
%  输出：
%    ysnv : 同尺寸的 SNV 归一化后光谱矩阵
%
%  用法示例：
%    X = randn(100, 701) + 5;     % 100 条模拟光谱
%    Xsnv = preprocess_snv(X);
%    fprintf('每行均值应 ≈ 0: %.2e\n', mean(mean(Xsnv)));
%
%  对应 Python 实现：src/preprocess.py → snv()
%  ====================================================

function ysnv = preprocess_snv(y)
    if isrow(y)
        ysnv = (y - mean(y)) / (std(y) + eps);
    else
        % 矩阵模式：逐行 SNV
        ysnv = (y - mean(y, 2)) ./ (std(y, 0, 2) + eps);
    end
end
