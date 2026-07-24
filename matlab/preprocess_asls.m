%% preprocess_asls.m
%  AsLS (Asymmetric Least Squares) 基线校正
%  ====================================================
%  方法：Eilers & Boelens (2005)
%  原理：通过加权最小二乘拟合一条平滑基线（包络在信号下方），
%        迭代更新权重 —— 高于基线的点权重 p，低于基线的点权重 (1-p)。
%
%  输入：
%    y       : 1×N 或 N×1 原始光谱（行/列向量均可）
%    lambda  : 平滑度参数（越大基线越平滑，典型 1e4 ~ 1e7）
%    p       : 非对称权重（越小越"压在信号下方"，典型 0.001 ~ 0.1）
%    nIter   : 迭代次数
%
%  输出：
%    baseline : 与 y 同尺寸的基线估计
%    corrected: y - baseline（基线校正后的光谱）
%
%  用法示例：
%    wn = 400:2:1800;
%    y = randn(size(wn)) .* (1 + sin(wn/200));
%    [bl, yc] = preprocess_asls(y, 1e5, 0.01, 20);
%    plot(wn, y, 'b', wn, bl, 'r--', wn, yc, 'k');
%
%  对应 Python 实现：src/preprocess.py → asls_baseline()
%  ====================================================

function [baseline, corrected] = preprocess_asls(y, lambda, p, nIter)
    % 参数默认值
    if nargin < 4 || isempty(nIter); nIter = 20; end
    if nargin < 3 || isempty(p);      p      = 0.01; end
    if nargin < 2 || isempty(lambda); lambda = 1e5; end

    y = y(:);                      % 强制列向量
    L = length(y);

    % 构建 2 阶差分罚矩阵 D'D
    % D'D 是三对角矩阵，主对角 = [1, 5, 6, ..., 6, 5, 1]
    %               次对角 = [-2, -4, ..., -4, -2]
    %           次次对角 = [1, 1, ..., 1]
    e = ones(L, 1);
    D = spdiags([e, -2*e, e], 0:2, L-2, L);
    H = lambda * D' * D;

    w = ones(L, 1);                % 初始权重
    baseline = y;

    for iter = 1:nIter
        W = spdiags(w, 0, L, L);  % 对角权重矩阵
        Z = W + H;
        z = Z \ (w .* y);          % 解线性系统
        w = p * (y > z) + (1-p) * (y <= z);  % 更新权重
        baseline = z;
    end

    corrected = y - baseline;
end
