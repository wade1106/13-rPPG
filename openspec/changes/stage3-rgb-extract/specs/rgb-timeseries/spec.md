## ADDED Requirements

### Requirement: ROI pixel mean calculation
系統 SHALL 對每幀 Skin ROI 區域內的所有像素計算 R、G、B 三通道的空間均值。

#### Scenario: Compute mean RGB from ROI
- **WHEN** 收到一幀影像與有效的 ROI 座標
- **THEN** 系統回傳 (mean_R, mean_G, mean_B) 三個浮點數值

#### Scenario: ROI area too small
- **WHEN** ROI 面積小於最小閾值（如 100 像素）
- **THEN** 系統跳過該幀，不將無效資料加入緩衝區

### Requirement: Time-series ring buffer
系統 SHALL 維護固定長度的時序緩衝區，儲存歷史 RGB 均值與對應的時間戳。

#### Scenario: Buffer accumulation
- **WHEN** 新的一幀 RGB 均值計算完成
- **THEN** 系統將 (timestamp, mean_R, mean_G, mean_B) 追加至緩衝區

#### Scenario: Buffer overflow
- **WHEN** 緩衝區已滿（達到最大長度）
- **THEN** 自動丟棄最舊的資料點，保持固定長度滑動窗口

### Requirement: Physical timestamp recording
系統 SHALL 為每幀記錄高精度物理時間戳 (`time.time() - start_time`)，不依賴幀編號作為時間軸。

#### Scenario: Timestamp for each frame
- **WHEN** 擷取到新的一幀影像
- **THEN** 記錄該幀的真實物理時間（秒），精度至毫秒級

### Requirement: Real-time RGB waveform visualization
系統 SHALL 即時繪製 R、G、B 三通道的時序波形圖。

#### Scenario: Display RGB curves
- **WHEN** 緩衝區有足夠資料（至少 2 個資料點）
- **THEN** 以紅、綠、藍三色曲線即時顯示各通道的時間序列波形

#### Scenario: Observable pulse-like oscillation
- **WHEN** 使用者安靜面對鏡頭數秒
- **THEN** Green 通道波形可觀察到微小、規律的脈搏光學波動
