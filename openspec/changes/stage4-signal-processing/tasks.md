## 1. Signal Processing Library (signal_proc.py)

- [ ] 1.1 建立 signal_proc.py 模組檔案，定義公開函式介面
- [ ] 1.2 實作 Mean Normalization：各通道除以滑動視窗均值
- [ ] 1.3 實作 POS Projection：投影矩陣 [[0,1,-1],[-2,1,1]] 產出 S0, S1
- [ ] 1.4 實作 Dynamic Alpha Weighting：alpha = std(S0)/std(S1), H = S0 + alpha*S1
- [ ] 1.5 實作滑動視窗處理邏輯（預設 2.5 秒窗口）

## 2. Bandpass Filter

- [ ] 2.1 實作 Butterworth 4 階帶通濾波器設計 (0.7-3.0 Hz)
- [ ] 2.2 使用 scipy.signal.filtfilt 執行零相位濾波
- [ ] 2.3 封裝為可指定 fs 與頻率範圍的通用函式

## 3. Synthetic Test Script (stage4_signal_processing.py)

- [ ] 3.1 生成 75 BPM 合成皮膚色彩時序（含 DC offset + 呼吸漂移 + 高斯雜訊）
- [ ] 3.2 呼叫 signal_proc.py 進行 POS + 帶通濾波處理
- [ ] 3.3 實作 FFT 頻譜分析，計算主頻心率
- [ ] 3.4 驗證 FFT 主頻與真值 75 BPM 誤差 < 2.0 BPM
- [ ] 3.5 以 Matplotlib 輸出時域 + 頻域診斷對照圖

## 4. Verification

- [ ] 4.1 執行 stage4_signal_processing.py 確認測試通過
- [ ] 4.2 檢查診斷圖表中濾波後訊號波形清晰
- [ ] 4.3 確認 signal_proc.py 可被外部模組獨立 import 使用
