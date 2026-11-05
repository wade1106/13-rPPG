"""Stage 4: POS 訊號處理合成測試（零硬體依賴的單元測試）

本程式不需要相機，用數學方式生成「假的」皮膚 RGB 時序資料，
內嵌已知心率 (75 BPM)，再通過 POS + 帶通濾波處理，
最後用 FFT 頻譜分析驗證能否正確還原出 75 BPM。

目的：在接上真實相機之前，先確認訊號處理演算法本身是正確的。
如果連合成訊號都無法正確還原，接真實相機只會更糟。

驗收標準：FFT 主頻與真值的誤差 < 2.0 BPM
"""

import numpy as np                     # 數值運算與陣列處理
import matplotlib.pyplot as plt         # 繪製診斷圖表
from signal_proc import pos_process     # 匯入我們自己寫的 POS 演算法

# ── 參數設定 ──────────────────────────────────────────────

FS = 30.0           # 模擬取樣率 (Hz)，模擬 WebCam 30 FPS
DURATION = 20.0     # 模擬訊號長度 (秒)
TRUE_HR_BPM = 75.0  # 設定的真實心率 (BPM = Beats Per Minute)
TRUE_HR_HZ = TRUE_HR_BPM / 60.0  # 轉為頻率 = 1.25 Hz
BREATH_HZ = 0.25    # 模擬呼吸漂移的頻率 (0.25 Hz = 15 次/分鐘)
TOLERANCE = 2.0     # 允許的最大誤差 (BPM)


def generate_synthetic_rgb():
    """生成包含已知心率的模擬皮膚色彩 RGB 時序資料。

    模擬內容：
    1. DC offset — 基礎膚色（R=180, G=150, B=120），模擬皮膚的靜態反射光
    2. 脈搏分量 — 75 BPM 的正弦波，主要影響 Green 通道
       （因為血紅素對綠光的吸收變化最明顯）
    3. 呼吸漂移 — 0.25 Hz 的慢速低頻波動，模擬胸廓運動帶來的光照變化
    4. 高斯雜訊 — 隨機雜訊，模擬相機量化誤差與環境干擾

    Returns:
        rgb: (N, 3) ndarray，模擬的 RGB 時序矩陣
        t: (N,) ndarray，時間軸（秒）
    """
    n = int(DURATION * FS)   # 總樣本數 = 20s × 30Hz = 600 點
    t = np.arange(n) / FS    # 時間軸：[0, 0.033, 0.067, ..., 19.967]

    # 基礎膚色 DC offset（直流分量）：模擬皮膚在穩定光照下的平均反射值
    dc = np.array([180.0, 150.0, 120.0])

    # 脈搏分量：75 BPM = 1.25 Hz 的正弦波
    # Green 通道受脈搏影響最大（係數 1.0），R 和 B 較小（0.3, 0.5）
    # 這符合皮膚光學模型：血紅素主要吸收綠光
    pulse = np.sin(2 * np.pi * TRUE_HR_HZ * t)
    pulse_rgb = np.outer(pulse, [0.3, 1.0, 0.5])

    # 呼吸漂移：0.25 Hz 的慢速低頻波動，振幅 2.0
    # 模擬呼吸時胸廓運動導致的整體亮度緩慢變化（影響所有通道）
    breath = 2.0 * np.sin(2 * np.pi * BREATH_HZ * t)
    breath_rgb = np.outer(breath, [1.0, 1.0, 1.0])

    # 高斯雜訊：標準差 0.5，模擬相機感測器的隨機量化誤差
    noise = np.random.randn(n, 3) * 0.5

    # 合成最終 RGB 訊號 = 靜態膚色 + 脈搏 + 呼吸漂移 + 雜訊
    rgb = dc + pulse_rgb + breath_rgb + noise

    return rgb, t


def main():
    # ── 1. 生成合成訊號 ──────────────────────────────────────
    print(f"[INFO] 生成合成訊號：HR={TRUE_HR_BPM} BPM, fs={FS} Hz, duration={DURATION}s")
    rgb, t = generate_synthetic_rgb()
    n = len(t)

    # ── 2. POS + 帶通濾波處理 ────────────────────────────────
    print("[INFO] 執行 POS + 帶通濾波...")
    rppg = pos_process(rgb, FS)  # 呼叫 signal_proc.py 的核心函式

    # ── 3. FFT 頻譜分析 ──────────────────────────────────────
    # FFT (Fast Fourier Transform) 將時域訊號轉換為頻域
    # rfftfreq 計算每個 FFT bin 對應的頻率值
    freqs = np.fft.rfftfreq(n, d=1.0 / FS)  # 頻率軸 (Hz)
    fft_mag = np.abs(np.fft.rfft(rppg))       # 各頻率的振幅（取絕對值）

    # 只在生理合理的心率範圍內搜尋主頻：0.7-3.0 Hz = 42-180 BPM
    mask = (freqs >= 0.7) & (freqs <= 3.0)
    peak_idx = np.argmax(fft_mag[mask])        # 找到最大振幅的索引
    peak_freq = freqs[mask][peak_idx]          # 對應的頻率 (Hz)
    estimated_bpm = peak_freq * 60.0           # 轉為 BPM（Hz × 60 = 次/分鐘）

    # ── 4. 驗證結果 ──────────────────────────────────────────
    error = abs(estimated_bpm - TRUE_HR_BPM)
    passed = error < TOLERANCE

    print(f"[RESULT] 真值: {TRUE_HR_BPM:.1f} BPM")
    print(f"[RESULT] 估計: {estimated_bpm:.1f} BPM")
    print(f"[RESULT] 誤差: {error:.2f} BPM")
    print(f"[{'PASS' if passed else 'FAIL'}] 誤差 {'<' if passed else '>='} {TOLERANCE} BPM")

    # ── 5. 診斷圖表（三張子圖上下排列）─────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))

    # 圖 1：原始合成 RGB 時序（可看到三通道各自的波動）
    axes[0].plot(t, rgb[:, 0], 'r-', alpha=0.7, label='R')
    axes[0].plot(t, rgb[:, 1], 'g-', alpha=0.7, label='G')
    axes[0].plot(t, rgb[:, 2], 'b-', alpha=0.7, label='B')
    axes[0].set_title('Raw Synthetic RGB Signal')
    axes[0].set_xlabel('Time (s)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 圖 2：POS + 帶通濾波後的 rPPG 波形（應該看到乾淨的脈搏波）
    axes[1].plot(t, rppg, 'k-', linewidth=0.8)
    axes[1].set_title('POS + Bandpass Filtered rPPG Signal')
    axes[1].set_xlabel('Time (s)')
    axes[1].grid(True, alpha=0.3)

    # 圖 3：FFT 頻譜（主峰應該精準對準 1.25 Hz = 75 BPM）
    axes[2].plot(freqs[mask], fft_mag[mask], 'b-')
    axes[2].axvline(TRUE_HR_HZ, color='r', linestyle='--',
                    label=f'True HR: {TRUE_HR_BPM} BPM')      # 紅色虛線：真值位置
    axes[2].axvline(peak_freq, color='g', linestyle='--',
                    label=f'Estimated: {estimated_bpm:.1f} BPM')  # 綠色虛線：估計位置
    axes[2].set_title('FFT Spectrum (0.7 - 3.0 Hz)')
    axes[2].set_xlabel('Frequency (Hz)')
    axes[2].set_ylabel('Magnitude')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('stage4_diagnostic.png', dpi=150)
    print("[INFO] 診斷圖表已存為 stage4_diagnostic.png")
    plt.show()


if __name__ == "__main__":
    main()
