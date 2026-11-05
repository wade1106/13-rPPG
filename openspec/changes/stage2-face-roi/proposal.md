## Why

有了穩定的視訊串流後，需要從整張畫面中精準定位人臉並萃取皮膚感興趣區域 (ROI)。ROI 的品質直接決定後續 rPPG 訊號的訊雜比——ROI 定義錯了，後面 POS、Filter、FFT 再漂亮都救不了。需要 YOLO 做即時人臉偵測，再從人臉框中提取額頭/臉頰等穩定皮膚區域。

## What Changes

- 新增 `stage2_face_roi.py`：整合 YOLO 人臉偵測與 Skin ROI 萃取
- 引入 YOLOv8-face 模型進行即時人臉偵測
- 從人臉 bounding box 中計算臉頰與額頭 ROI 座標
- ROI 隨人臉移動即時跟隨
- 邊界防呆：ROI 座標 clamp 至畫面範圍內，避免 IndexError
- 延續 Stage 1 相機串流架構（DirectShow、FPS、鏡像、Q 退出）

## Capabilities

### New Capabilities
- `face-detection`: YOLO 人臉偵測模型載入與即時推論，bounding box 繪製與信心分數顯示
- `skin-roi`: 從人臉框計算額頭/臉頰 ROI 座標，邊界 clamp 防呆，ROI 框即時跟隨人臉移動

### Modified Capabilities

## Impact

- 新增依賴：`ultralytics` (YOLOv8)
- 需下載 YOLO face 預訓練權重
- 產出檔案：`stage2_face_roi.py`
- 繼承 Stage 1 相機模組，為 Stage 3 RGB 萃取提供 ROI 輸入
