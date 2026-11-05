## Context

POS + 帶通濾波後的訊號已是乾淨的脈搏波，但需要將波形轉為臨床可用的數值指標。Peak-to-peak 間隔即為一次心跳週期，而心跳間隔的慢速調變 (Respiratory Sinus Arrhythmia, RSA) 攜帶呼吸節律資訊。

## Goals / Non-Goals

**Goals:**
- 從 rPPG 脈搏波中偵測收縮峰值 (systolic peaks)
- 計算 RR interval 並換算為 HR (BPM)
- 從 RR interval 序列萃取呼吸率 (Breathing Rate, RPM)
- 使用雙重中值濾波消除脈衝離群值，輸出穩定數值
- 整合 Stage 1-4 模組形成完整即時管線

**Non-Goals:**
- 不做 GUI 儀表板（Stage 6）
- 不做 HRV 分析（進階功能）
- 不做 Signal Quality Index（進階挑戰）

## Decisions

### Decision 1: scipy.signal.find_peaks 搭配生理約束
- **選擇**: `find_peaks(signal, distance=min_distance, prominence=threshold)`
- **理由**: 生理不反應期（心跳間隔最少 0.35s = distance >= 0.35*fs）排除假峰；prominence 排除雜訊小波動
- **替代方案**: 自製過零偵測（不穩定）、模板匹配（計算量大）

### Decision 2: 自相關函數萃取呼吸率
- **選擇**: `np.correlate(rsa_signal, rsa_signal, mode='same')`，搜尋延遲窗口 [90:130] 的最大峰值
- **理由**: 呼吸頻率極低 (0.15-0.4 Hz)，直接 FFT 容易被 1/f 雜訊淹沒；自相關的「自我重複偵測」對微弱週期訊號穿透力更強
- **替代方案**: FFT（低頻解析度差）、Welch PSD（需更長資料窗）

### Decision 3: 雙重中值濾波 medfilt(47) + medfilt(97)
- **選擇**: 兩級串聯中值濾波，47 點 (~6s) 消除快速脈衝，97 點 (~12s) 追蹤長週期穩定趨勢
- **理由**: 中值濾波崩潰點 50%，即使一半資料為離群值仍能正確輸出；比線性移動平均抗脈衝干擾強得多
- **替代方案**: 移動平均（被單一極端值污染）、Kalman filter（實作複雜度高）

### Decision 4: 30 秒滑動視窗
- **選擇**: vitals 計算使用 30 秒資料窗口
- **理由**: 涵蓋 6-10 次完整呼吸週期，自相關共振峰尖銳明確；12 秒窗口僅 2-3 個週期，自相關邊緣被截斷

## Risks / Trade-offs

- [開機前 15 秒資料不足] → 安全回傳 "--"，不強行計算污染 EMA
- [說話/眨眼導致 RR interval 突變] → 雙重中值濾波 + 變化率限幅 (slew rate +-1.0 RPM)
- [低心率或高心率超出帶通範圍] → 0.7-3.0 Hz 已涵蓋 42-180 BPM 的常見生理範圍
