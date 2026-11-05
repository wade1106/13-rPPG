## ADDED Requirements

### Requirement: Heart rate BPM calculation
系統 SHALL 從 RR interval 均值計算心率：`HR = 60 / mean(RR_intervals)`。

#### Scenario: Calculate HR from RR intervals
- **WHEN** 有至少 3 個有效的 RR interval
- **THEN** 輸出心率值 (BPM)，範圍在 42-180 BPM 內

#### Scenario: Insufficient data for HR
- **WHEN** 資料累積不足 15 秒
- **THEN** 系統回傳安全預設值 "--"，不強行計算

### Requirement: Dual median filter smoothing
系統 SHALL 對 RR interval 序列套用雙重中值濾波 (medfilt kernel=47 與 kernel=97)，消除脈衝離群值。

#### Scenario: Remove outlier RR interval
- **WHEN** 說話或眨眼導致單一 RR interval 暴增至 2.0 秒
- **THEN** 中值濾波後該異常值被替換為鄰近中位數，HR 數值不劇烈跳動

### Requirement: Slew rate limiting
系統 SHALL 限制 HR 數值每次更新的最大變化量，防止瞬間跳動。

#### Scenario: Limit HR change rate
- **WHEN** 連續兩次計算的 HR 差值超過閾值
- **THEN** 輸出值被 clamp 至前次值 +/- 最大允許變化量
