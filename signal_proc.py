"""signal_proc.py — rPPG 訊號處理核心函式庫

本模組封裝 rPPG 系統的核心訊號處理演算法，設計為獨立可複用的函式庫，
不依賴任何相機、GUI 或硬體模組，可單獨 import 使用。

核心演算法：
1. POS (Plane Orthogonal to Skin) — 從 RGB 時序中分離脈搏訊號
   - 原理：光照變化會同時影響 R/G/B 三通道（共模干擾），
     而脈搏引起的血液容積變化主要影響特定色彩比例。
     POS 利用正交投影將光照共模分量投影掉，保留脈搏分量。
   - 來源：Wang et al. (2017) 論文

2. Butterworth 帶通濾波器 — 限制訊號在人體心跳生理頻率範圍內
   - 通帶 0.7-3.0 Hz = 42-180 BPM
   - 消除低頻光照漂移與高頻相機雜訊
"""

import numpy as np                       # 矩陣運算與數學函式
from scipy.signal import butter, filtfilt  # Butterworth 濾波器設計與零相位濾波


def bandpass_filter(signal, fs, low=0.7, high=3.0, order=4):
    """Butterworth 帶通濾波器（零相位 filtfilt）。

    只保留指定頻率範圍內的訊號分量，濾除帶外雜訊。
    使用 filtfilt（前向+反向濾波）確保零相位延遲，不會讓波峰位置偏移。

    Args:
        signal: 1D 時序訊號陣列
        fs: 取樣率 (Hz)，如 25.0 或 30.0
        low: 低截止頻率 (Hz)，預設 0.7（對應最慢心率 42 BPM）
        high: 高截止頻率 (Hz)，預設 3.0（對應最快心率 180 BPM）
        order: 濾波器階數，越高衰減越陡但可能引入振盪

    Returns:
        濾波後的 1D 訊號陣列
    """
    nyq = fs / 2.0  # Nyquist 頻率 = 取樣率的一半（數位濾波器設計的基準）
    # butter() 設計 Butterworth 濾波器，回傳分子 (b) 與分母 (a) 係數
    # Wn 參數需正規化到 [0, 1] 範圍（相對於 Nyquist 頻率）
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    # filtfilt = forward-backward filtering，零相位濾波
    # 比 lfilter（單向濾波）多一次反向運算，消除群延遲
    return filtfilt(b, a, signal)


def pos_process(rgb_matrix, fs, window_sec=2.5):
    """POS 演算法：從 RGB 時序萃取 rPPG 脈搏波。

    POS (Plane Orthogonal to Skin) 是一種色彩空間投影方法，
    利用皮膚光學模型將 RGB 三通道訊號投影到與膚色平面正交的方向，
    藉此分離出微弱的脈搏訊號（血液容積變化）並消除光照干擾。

    演算法三步驟：
    1. Mean Normalization — 除以均值，消除靜態光強基線
    2. POS Projection — 用固定投影矩陣將 RGB 映射到兩個正交方向
    3. Dynamic Alpha Weighting — 根據兩通道的標準差動態加權融合

    Args:
        rgb_matrix: (N, 3) ndarray，每列為一幀的 [R均值, G均值, B均值]
        fs: 取樣率 (Hz)
        window_sec: 滑動視窗長度（秒），預設 2.5s（涵蓋 2-3 個心跳週期）
                    太短會不穩定，太長反應慢且混入狀態變化

    Returns:
        rppg_signal: (N,) 帶通濾波後的 rPPG 脈搏波訊號
    """
    n = rgb_matrix.shape[0]            # 總幀數
    win_len = int(window_sec * fs)     # 視窗長度（幀數）
    rppg = np.zeros(n)                 # 初始化輸出訊號（全零）

    # 滑動視窗逐段處理（視窗每次向前移動 1 幀，重疊部分會累加取平均）
    for start in range(0, n - win_len + 1):
        end = start + win_len
        window = rgb_matrix[start:end, :]  # 取出當前視窗的 RGB 資料 (win_len, 3)

        # ── Step 1: Mean Normalization（均值正規化）──────────
        # 將每個通道除以其視窗內的均值，消除「整體亮度」的影響
        # 例如：R 通道從 180 變成 ~1.0，保留的是相對變化比例
        means = window.mean(axis=0)          # 各通道的視窗均值 [R_mean, G_mean, B_mean]
        if np.any(means < 1e-6):             # 防止除以零（全黑畫面）
            continue
        cn = window / means                  # 正規化後的 RGB，各通道均值接近 1.0

        # ── Step 2: POS Projection（正交投影）────────────────
        # 投影矩陣 [[0, 1, -1], [-2, 1, 1]] 來自 Wang et al. (2017)
        # S0 = G_n - B_n         → 捕捉綠藍色差（對脈搏敏感）
        # S1 = -2*R_n + G_n + B_n → 捕捉色調變化（與 S0 正交）
        # 這兩個方向都正交於膚色平面，能有效消除鏡面反射與光照共模干擾
        s0 = cn[:, 1] - cn[:, 2]                       # G - B
        s1 = -2 * cn[:, 0] + cn[:, 1] + cn[:, 2]       # -2R + G + B

        # ── Step 3: Dynamic Alpha Weighting（動態加權融合）───
        # 兩個投影方向的訊號振幅可能不同，用標準差比值動態調整權重
        # alpha = std(S0) / std(S1)，讓兩通道對最終輸出的貢獻平衡
        # H = S0 + alpha * S1 → 融合後的單軌脈搏波
        std_s0 = np.std(s0)
        std_s1 = np.std(s1)
        if std_s1 < 1e-10:      # 防止除以零
            continue
        alpha = std_s0 / std_s1
        h = s0 + alpha * s1     # 融合後的 rPPG 脈搏波

        # 去均值後累加到輸出（重疊視窗的部分會被多次累加，等效於平均）
        rppg[start:end] += (h - h.mean())

    # ── 最終帶通濾波 ──────────────────────────────────────
    # 只在資料長度足夠時才濾波（至少 2 秒，否則濾波器不穩定）
    # 通帶 0.7-3.0 Hz 對應心率 42-180 BPM
    if n > int(fs * 2):
        rppg = bandpass_filter(rppg, fs)

    return rppg
