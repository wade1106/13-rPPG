"""hrv_features.py — HRV 特徵萃取模組

從 RR intervals（相鄰心跳波峰的時間間隔）萃取三大類 HRV 特徵，
供 XGBoost 風險分類器作為輸入。

HRV (Heart Rate Variability) 是心跳間隔的變異程度，
反映自律神經系統（交感/副交感）的調控能力：
- 健康人的 HRV 較高（心跳間隔靈活變化）
- HRV 降低代表自律神經失調，與心血管疾病、壓力、疲勞相關

三大類特徵：
1. 時域 (Time Domain) — 直接從 RR interval 的統計量計算
2. 頻域 (Frequency Domain) — 對 RR interval 做頻譜分析
3. 非線性 (Nonlinear) — Poincaré 散點圖的幾何分析
"""

import numpy as np
from scipy.signal import welch
from scipy.interpolate import interp1d


# ── 時域特徵 ──────────────────────────────────────────────

def compute_time_domain(rr_intervals):
    """計算時域 HRV 特徵。

    Args:
        rr_intervals: 1D ndarray，RR interval 序列（單位：毫秒）

    Returns:
        dict，包含以下特徵：
        - mean_rr: 平均 RR interval (ms)
        - sdnn: RR interval 的標準差 (ms)
            — 反映整體 HRV，越大越健康
        - rmssd: 相鄰 RR interval 差值的均方根 (ms)
            — 反映短期（副交感神經）HRV
        - pnn50: 相鄰 RR interval 差值 > 50ms 的比例 (%)
            — 副交感神經活性的簡易指標
        - mean_hr: 平均心率 (BPM)
    """
    if len(rr_intervals) < 2:
        return None

    rr = np.array(rr_intervals, dtype=np.float64)

    mean_rr = np.mean(rr)
    sdnn = np.std(rr, ddof=1)  # 樣本標準差（ddof=1 為無偏估計）

    # 相鄰 RR interval 的差值
    diff_rr = np.diff(rr)
    rmssd = np.sqrt(np.mean(diff_rr ** 2))

    # 差值 > 50ms 的比例
    pnn50 = np.sum(np.abs(diff_rr) > 50) / len(diff_rr) * 100.0

    # 平均心率
    mean_hr = 60000.0 / mean_rr  # ms → BPM

    return {
        'mean_rr': mean_rr,
        'sdnn': sdnn,
        'rmssd': rmssd,
        'pnn50': pnn50,
        'mean_hr': mean_hr,
    }


# ── 頻域特徵 ──────────────────────────────────────────────

def compute_frequency_domain(rr_intervals, fs_interp=4.0):
    """計算頻域 HRV 特徵。

    原理：
    RR interval 序列本身是不等距取樣的（因為心跳間隔不等），
    需要先用 interp1d 內插成等距序列，再用 Welch 方法估算功率頻譜密度 (PSD)。

    頻帶定義（依 ESC/NASPE 標準）：
    - LF (Low Frequency): 0.04 - 0.15 Hz
      → 反映交感 + 副交感神經的混合調控
    - HF (High Frequency): 0.15 - 0.40 Hz
      → 主要反映副交感神經（迷走神經）活性
      → 與呼吸性竇性心律不整 (RSA) 密切相關
    - LF/HF ratio: 交感/副交感平衡指標
      → 比值升高 = 交感神經佔優勢（壓力/緊張）

    Args:
        rr_intervals: 1D ndarray，RR interval 序列（單位：毫秒）
        fs_interp: 內插後的目標取樣率 (Hz)，預設 4 Hz

    Returns:
        dict，包含 lf_power, hf_power, lf_hf_ratio, total_power
    """
    if len(rr_intervals) < 10:
        return None

    rr = np.array(rr_intervals, dtype=np.float64)

    # 建立不等距時間軸（累加 RR interval）
    t_rr = np.cumsum(rr) / 1000.0  # ms → 秒
    t_rr = t_rr - t_rr[0]          # 起始點歸零

    # 去均值（去除直流分量，只保留變異部分）
    rr_detrend = rr - np.mean(rr)

    # 內插成等距取樣（Welch PSD 需要等距資料）
    t_uniform = np.arange(0, t_rr[-1], 1.0 / fs_interp)
    if len(t_uniform) < 16:
        return None
    f_interp = interp1d(t_rr, rr_detrend, kind='cubic', fill_value='extrapolate')
    rr_uniform = f_interp(t_uniform)

    # Welch 功率頻譜密度估算
    # nperseg = 取樣率 × 視窗秒數，這裡用全長或 256 取較小值
    nperseg = min(len(rr_uniform), 256)
    freqs, psd = welch(rr_uniform, fs=fs_interp, nperseg=nperseg)

    # 計算各頻帶的功率（頻譜下的面積 = 梯形積分）
    lf_mask = (freqs >= 0.04) & (freqs <= 0.15)
    hf_mask = (freqs >= 0.15) & (freqs <= 0.40)

    lf_power = np.trapezoid(psd[lf_mask], freqs[lf_mask]) if np.any(lf_mask) else 0.0
    hf_power = np.trapezoid(psd[hf_mask], freqs[hf_mask]) if np.any(hf_mask) else 0.0
    total_power = lf_power + hf_power

    # LF/HF ratio（防止除以零）
    lf_hf_ratio = lf_power / hf_power if hf_power > 1e-10 else 10.0

    return {
        'lf_power': lf_power,
        'hf_power': hf_power,
        'lf_hf_ratio': lf_hf_ratio,
        'total_power': total_power,
    }


# ── 非線性特徵 ────────────────────────────────────────────

def compute_poincare(rr_intervals):
    """計算 Poincaré 散點圖的幾何特徵。

    Poincaré Plot 原理：
    將 RR(n) vs RR(n+1) 繪成散點圖，觀察連續心跳間隔的關係。
    點群的形狀反映 HRV 的短期與長期特性：

    - SD1: 垂直於對角線方向的標準差（短軸）
      → 反映短期 HRV（逐拍變異），與 RMSSD 高度相關
      → 代表副交感神經的即時調控能力
    - SD2: 平行於對角線方向的標準差（長軸）
      → 反映長期 HRV（整體趨勢），與 SDNN 相關
      → 代表交感 + 副交感的綜合調控

    健康人的 Poincaré Plot 像橢圓（SD1, SD2 都大），
    自律神經失調者的點群會收縮成小圓點。

    Args:
        rr_intervals: 1D ndarray，RR interval 序列（單位：毫秒）

    Returns:
        dict，包含 sd1, sd2, sd_ratio
    """
    if len(rr_intervals) < 3:
        return None

    rr = np.array(rr_intervals, dtype=np.float64)

    # RR(n) 和 RR(n+1) 配對
    rr_n = rr[:-1]
    rr_n1 = rr[1:]

    # SD1 = 垂直於 y=x 對角線的分散度
    # 數學上等於 diff(RR) 的標準差 / sqrt(2)
    diff = rr_n1 - rr_n
    sd1 = np.std(diff, ddof=1) / np.sqrt(2)

    # SD2 = 平行於 y=x 對角線的分散度
    # 數學上等於 sum(RR) 相鄰對的標準差 / sqrt(2)
    sum_rr = rr_n1 + rr_n
    sd2 = np.std(sum_rr, ddof=1) / np.sqrt(2)

    # SD1/SD2 比值（短期/長期 HRV 平衡）
    sd_ratio = sd1 / sd2 if sd2 > 1e-10 else 0.0

    return {
        'sd1': sd1,
        'sd2': sd2,
        'sd_ratio': sd_ratio,
    }


# ── 整合函式 ──────────────────────────────────────────────

# 特徵名稱清單（訓練和推論時保持一致的欄位順序）
FEATURE_NAMES = [
    'mean_rr', 'sdnn', 'rmssd', 'pnn50', 'mean_hr',
    'lf_power', 'hf_power', 'lf_hf_ratio', 'total_power',
    'sd1', 'sd2', 'sd_ratio',
]


def extract_hrv_features(rr_intervals):
    """從 RR intervals 萃取完整的 HRV 特徵向量。

    整合時域、頻域、非線性三類特徵，回傳固定長度的特徵向量。
    任一類計算失敗時填入 0（確保向量長度一致）。

    Args:
        rr_intervals: 1D ndarray，RR interval 序列（單位：毫秒）

    Returns:
        numpy 1D array，長度 12 的特徵向量；若資料不足回傳 None
    """
    if len(rr_intervals) < 10:
        return None

    # 三類特徵分別計算
    td = compute_time_domain(rr_intervals)
    fd = compute_frequency_domain(rr_intervals)
    pc = compute_poincare(rr_intervals)

    # 合併成字典（失敗的類別填 0）
    features = {}
    for src in [td, fd, pc]:
        if src is not None:
            features.update(src)

    # 按固定順序輸出為 numpy array
    vec = np.array([features.get(name, 0.0) for name in FEATURE_NAMES], dtype=np.float64)
    return vec
