## Why

Stage 4 產出了乾淨的 rPPG 脈搏波，但有波形不代表已經知道 BPM。需要從濾波後的脈搏波中偵測波峰 (Peak Detection)，計算 RR interval 換算心率 (HR)，並利用 RR interval 的慢速調變 (RSA) 透過自相關函數萃取呼吸率 (RR)。還需要雙重中值濾波消除說話、眨眼等脈衝離群值，讓數值穩定如醫療監視器。

## What Changes

- 新增 `stage5_vitals.py`：生命徵象運算核心，即時輸出 HR (BPM) 與 RR (RPM)
- Peak Detection：使用 `scipy.signal.find_peaks` 搭配生理不反應期 (minimum distance >= 0.35s) 與 prominence 約束
- HR 計算：`HR = 60 / mean(RR_intervals)`
- RR 計算：對 RR interval 序列帶通濾波 (0.15-0.4 Hz) 後執行自相關函數，鎖定呼吸週期
- 雙重中值濾波 (medfilt 47/97)：消除脈衝離群值
- 整合 Stage 1-4 模組，從相機串流到數值輸出的完整管線

## Capabilities

### New Capabilities
- `peak-detection`: 脈搏波峰偵測，生理不反應期約束，prominence 閾值，RR interval 計算
- `heart-rate`: HR 從 RR interval 均值換算 BPM，雙重中值濾波平滑
- `breathing-rate`: RSA 帶通濾波、自相關函數萃取呼吸週期、延遲窗口鎖定呼吸率

### Modified Capabilities

## Impact

- 依賴：`scipy`（find_peaks, medfilt, correlate）、`numpy`
- 產出檔案：`stage5_vitals.py`
- 此模組產出的 HR/RR 數值為 Stage 6 儀表板的核心資料來源
