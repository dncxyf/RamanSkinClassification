%% preprocess_sg.m
%  Savitzky-Golay 平滑滤波
%  ====================================================
%  方法：Savitzky & Golay (1964)
%  原理：在滑动窗口内拟合多项式，取中心点拟合值作为平滑后信号。
%        相比简单移动平均，SG 滤波能保留峰形（保持峰位的导数信息），
%        是拉曼光谱去噪的标准方法。
%
%  输入：
%    y        : 1×N 或 N×1 光谱
%    windowLen : 窗口长度（奇数，默认 15，对应约 30 cm⁻¹ @ 2cm⁻¹步长）
%    polyorder : 多项式阶数（默认 3，需 < windowLen）
%
%  输出：
%    smoothed : 平滑后的光谱
%
%  用法示例：
%    wn = 400:2:1800;
%    y = sin(wn/100) + randn(size(wn))*0.1;
%    ys = preprocess_sg(y, 15, 3);
%    plot(wn, y, 'Color', [0.8 0.8 0.8], wn, ys, 'b', 'LineWidth', 1.5);
%
%  MATLAB 内置：sgolayfilt() —— 本函数是语义等价的封装（含参数校验）
%  对应 Python 实现：src/preprocess.py → savgol_smooth()
%  ====================================================

function smoothed = preprocess_sg(y, windowLen, polyorder)
    if nargin < 3 || isempty(polyorder); polyorder = 3;  end
    if nargin < 2 || isempty(windowLen); windowLen = 15; end

    % 参数校验
    if mod(windowLen, 2) == 0
        warning('windowLen 必须为奇数，已自动 +1');
        windowLen = windowLen + 1;
    end
    if polyorder >= windowLen
        error('polyorder 必须 < windowLen');
    end

    y = y(:);
    smoothed = sgolayfilt(y, polyorder, windowLen);
end
