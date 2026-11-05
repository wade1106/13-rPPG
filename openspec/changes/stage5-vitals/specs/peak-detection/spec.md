## ADDED Requirements

### Requirement: Systolic peak detection with physiological constraints
系統 SHALL 使用 `scipy.signal.find_peaks` 偵測 rPPG 脈搏波的收縮峰值，並設定生理不反應期 (minimum distance >= 0.35s * fs) 與 prominence 閾值。

#### Scenario: Detect peaks in clean pulse signal
- **WHEN** 輸入帶通濾波後的 rPPG 脈搏波
- **THEN** 偵測到的相鄰波峰間隔均 >= 0.35 秒（對應最大 171 BPM）

#### Scenario: Reject noise spikes
- **WHEN** 訊號中存在小幅度雜訊波動
- **THEN** prominence 閾值將其排除，僅保留真正的收縮峰值

### Requirement: RR interval calculation
系統 SHALL 計算相鄰波峰的時間間隔 (RR interval)，單位為秒。

#### Scenario: Compute RR intervals from peaks
- **WHEN** 偵測到 N 個波峰位置
- **THEN** 輸出 N-1 個 RR interval 值，每個值為相鄰峰間的時間差
