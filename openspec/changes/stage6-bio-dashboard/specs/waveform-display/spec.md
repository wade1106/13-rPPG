## ADDED Requirements

### Requirement: Real-time pulse waveform oscilloscope
系統 SHALL 使用 pyqtgraph 在右側下方繪製即時脈搏波形，X 軸為物理時間（秒），Y 軸為 rPPG 訊號振幅。

#### Scenario: Waveform scrolling display
- **WHEN** 量測運行中且有足夠資料
- **THEN** 波形圖以 30 秒物理時間軸滾動顯示，新資料從右側進入

### Requirement: Fixed 30-second time window
系統 SHALL 將示波器 X 軸鎖定為 30 秒視窗 (`setXRange(0, 30)`)，以物理時間戳裁切資料，不以幀數計算。

#### Scenario: Consistent time axis regardless of FPS
- **WHEN** WebCam 實際 FPS 波動（如 16-30 FPS）
- **THEN** 示波器時間軸始終為 30 秒，不膨脹或壓縮

### Requirement: 1 Hz numeric refresh rate
系統 SHALL 以獨立的 QTimer (1000 ms) 驅動數值面板更新，與工作線程運算頻率解耦。

#### Scenario: Stable numeric display
- **WHEN** 工作線程以 5 Hz 發送運算結果
- **THEN** HR 和 RR 數值每秒僅更新一次，避免視覺疲勞與字型微跳

### Requirement: Startup buffer indicator
系統 SHALL 在資料累積期間（開機前 15 秒）於數值面板顯示 "--" 並提示緩衝狀態。

#### Scenario: Display buffering status
- **WHEN** 量測啟動後不足 15 秒
- **THEN** HR 和 RR 顯示 "--"，並顯示「緩衝中...」提示文字
