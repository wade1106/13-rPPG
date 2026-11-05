## ADDED Requirements

### Requirement: RSA bandpass filtering
系統 SHALL 對 RR interval 序列套用帶通濾波 (0.15-0.4 Hz)，萃取呼吸性竇性心律不整 (RSA) 分量。

#### Scenario: Extract RSA from RR intervals
- **WHEN** 輸入足夠長度的 RR interval 序列
- **THEN** 輸出僅包含 0.15-0.4 Hz（9-24 RPM）呼吸頻帶的 RSA 訊號

### Requirement: Autocorrelation breathing rate extraction
系統 SHALL 對 RSA 訊號計算自相關函數，在生理延遲窗口內尋找最大峰值以推估呼吸週期。

#### Scenario: Detect breathing period via autocorrelation
- **WHEN** RSA 訊號具有規律的呼吸調變
- **THEN** 自相關函數在延遲窗口 [90:130]（對應 ~3-6 秒呼吸週期）內出現明顯峰值

#### Scenario: Convert autocorrelation peak to RPM
- **WHEN** 找到自相關峰值位置 br_r
- **THEN** 計算 `BR = (fs / br_r) * 60` 得到呼吸率 (RPM)

### Requirement: Breathing rate startup protection
系統 SHALL 在資料累積不足 15 秒時回傳安全預設值，不強行計算呼吸率。

#### Scenario: Startup buffer period
- **WHEN** 系統啟動後不足 15 秒
- **THEN** 呼吸率顯示 "--"，狀態提示「呼吸緩衝中」

### Requirement: Breathing rate smoothing
系統 SHALL 對呼吸率套用變化率限幅 (slew rate +/- 1.0 RPM) 與指數移動平均 (EMA)。

#### Scenario: Smooth breathing rate output
- **WHEN** 連續計算的呼吸率波動超過 1.0 RPM
- **THEN** 輸出值被限幅並以 EMA 平滑，呈現穩定收斂的趨勢
