## Why

有了 Skin ROI 後，需要將每幀影像的 ROI 像素「降維」為一組 RGB 均值，形成時間序列。這是從「影像」跨入「訊號處理」的關鍵轉換。沒有穩定的 RGB 時序資料，後續 POS 演算法與帶通濾波無法運作。此階段還需即時視覺化 RGB 波形，讓使用者觀察到微弱的脈搏光學變化。

## What Changes

- 新增 `stage3_rgb_extract.py`：ROI 像素空間降維與 RGB 時序緩衝區
- 每幀計算 ROI 內所有像素的 R、G、B 三通道均值
- 建立固定長度的時序緩衝區（Ring Buffer）儲存歷史 RGB 均值
- 使用 Matplotlib 或 OpenCV 即時繪製 R、G、B 三條波形曲線
- 延續 Stage 1-2 架構（相機串流 + YOLO 人臉 + ROI）

## Capabilities

### New Capabilities
- `rgb-timeseries`: ROI 像素 RGB 均值計算、時序緩衝區管理、即時 RGB 波形可視化

### Modified Capabilities

## Impact

- 可能新增依賴：`matplotlib`（若用於即時繪圖）或使用 OpenCV 內建繪圖
- 產出檔案：`stage3_rgb_extract.py`
- 繼承 Stage 1-2 模組，為 Stage 4 POS 訊號處理提供 RGB 時序輸入
