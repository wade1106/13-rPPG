## ADDED Requirements

### Requirement: Synthetic signal generation
測試腳本 SHALL 生成包含已知真實心率（如 75 BPM / 1.25 Hz）的模擬皮膚色彩時序資料。

#### Scenario: Generate synthetic RGB with known HR
- **WHEN** 指定目標心率 75 BPM 與取樣率 30 Hz
- **THEN** 產生 RGB 三通道時序矩陣，Green 通道內嵌 1.25 Hz 正弦脈搏分量

### Requirement: Physiological noise injection
測試腳本 SHALL 在合成訊號中疊加直流背景底色、慢速呼吸漂移 (0.25 Hz) 與高斯白雜訊。

#### Scenario: Add realistic noise to synthetic signal
- **WHEN** 生成基礎合成訊號後
- **THEN** 疊加 DC offset、0.25 Hz 呼吸漂移與 SNR 可控的高斯雜訊

### Requirement: FFT frequency validation
測試腳本 SHALL 對 POS + 帶通濾波後的訊號執行 FFT，驗證主頻心率與真值誤差小於 2.0 BPM。

#### Scenario: Validate recovered HR within tolerance
- **WHEN** 合成訊號經 POS + bandpass 處理後
- **THEN** FFT 主頻對應的心率與真值 (75 BPM) 的絕對誤差 < 2.0 BPM

### Requirement: Diagnostic chart output
測試腳本 SHALL 以 Matplotlib 繪製時域訊號波形與頻譜對照分析圖。

#### Scenario: Generate diagnostic plot
- **WHEN** 測試執行完成
- **THEN** 輸出包含原始訊號、濾波後訊號、FFT 頻譜與標註主頻的診斷圖表
