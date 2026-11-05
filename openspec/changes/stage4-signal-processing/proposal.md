## Why

Stage 3 產出的原始 RGB 時序包含大量環境光照干擾、頭部晃動雜訊與相機量化噪點，直接用 Green 通道做分析會追蹤到「光照變化」而非「脈搏」。需要 POS (Plane Orthogonal to Skin) 演算法進行色彩空間正交投影，將鏡面反射與光照變異投影掉，保留純粹的脈搏訊號，再以帶通濾波限制在人體心跳生理頻率範圍內。此模組需獨立於硬體，可用合成訊號進行單元測試。

## What Changes

- 新增 `signal_proc.py`：獨立可複用的 POS 訊號處理函式庫
- 新增 `stage4_signal_processing.py`：離線合成訊號單元測試腳本
- POS 演算法三步驟：Mean Normalization → Projection Matrix → Dynamic Alpha Weighting
- Butterworth 帶通濾波器 (0.7-3.0 Hz) 消除高低頻雜訊
- 滑動視窗 (2.5 秒) 局部時間窗估計
- 合成訊號驗證（已知 75 BPM → FFT 誤差 < 2.0 BPM）

## Capabilities

### New Capabilities
- `pos-algorithm`: POS 色彩正規化、雙通道正交投影、動態 alpha 加權融合、滑動視窗處理
- `bandpass-filter`: Butterworth 帶通濾波器設計與套用，生理頻率範圍 0.7-3.0 Hz
- `synthetic-test`: 合成已知心率訊號、加入生理雜訊、FFT 頻譜驗證、Matplotlib 診斷圖表

### Modified Capabilities

## Impact

- 新增依賴：`scipy`（Butterworth filter、interp1d、FFT）、`numpy`、`matplotlib`
- 產出檔案：`signal_proc.py`、`stage4_signal_processing.py`
- 此模組為 Stage 5 生命徵象計算的核心訊號輸入
