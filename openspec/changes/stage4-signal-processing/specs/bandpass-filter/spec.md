## ADDED Requirements

### Requirement: Butterworth bandpass filter design
系統 SHALL 設計 4 階 Butterworth 帶通濾波器，通帶為 0.7-3.0 Hz，對應心率 42-180 BPM。

#### Scenario: Design filter with given sampling rate
- **WHEN** 指定取樣率 fs（如 30 Hz）
- **THEN** 系統產生 Butterworth 帶通濾波器係數 (b, a)，通帶 0.7-3.0 Hz

### Requirement: Zero-phase filtering
系統 SHALL 使用 `scipy.signal.filtfilt` 執行零相位濾波，避免相位延遲。

#### Scenario: Apply bandpass filter to POS signal
- **WHEN** 輸入 POS 運算後的 rPPG 訊號
- **THEN** 輸出僅保留 0.7-3.0 Hz 頻帶的濾波訊號，無相位偏移

### Requirement: Reject out-of-band noise
系統 SHALL 有效抑制低於 0.7 Hz 的慢速漂移（呼吸、光照變化）與高於 3.0 Hz 的高頻雜訊（相機量化、快速動作）。

#### Scenario: Remove DC drift and high-frequency noise
- **WHEN** 輸入包含直流偏移和高頻雜訊的訊號
- **THEN** 濾波後訊號的直流分量趨近於零，高頻雜訊大幅衰減
