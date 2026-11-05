## Context

Stage 1 已建立穩定的 WebCam 串流。現在需要在每幀影像中偵測人臉並提取適合 rPPG 量測的皮膚區域。YOLO 負責「人在哪裡」（Detection），ROI 負責「量哪裡」（Measurement）——這是兩個不同的問題。

## Goals / Non-Goals

**Goals:**
- 使用 YOLOv8 即時偵測人臉，取最大/最主要的人臉
- 從人臉框中計算額頭與臉頰 ROI（避開眼睛、嘴唇、頭髮等干擾）
- ROI 座標隨人臉移動即時更新
- 邊界防呆：所有座標 clamp 至 `[0, width)` / `[0, height)` 範圍
- 在畫面上繪製人臉框與 ROI 框供視覺驗證

**Non-Goals:**
- 不做多人臉追蹤（僅取最大臉）
- 不做 YCbCr 膚色分割（進階穩定機制，可後續加入）
- 不做 RGB 均值萃取或訊號處理（Stage 3）

## Decisions

### Decision 1: YOLOv8-face 作為人臉偵測模型
- **選擇**: `ultralytics` 的 YOLOv8n-face 或 YOLOv8s-face
- **理由**: 單次前向傳播即輸出所有偵測框，速度可達 30+ FPS；比 Haar Cascade 準確，比 dlib 快
- **替代方案**: MediaPipe Face Detection（輕量但框不穩定）、dlib HOG（慢）

### Decision 2: ROI 選擇額頭優先
- **選擇**: 主 ROI 為額頭區域（face box 上方 1/3），備選臉頰
- **理由**: 額頭皮膚平坦、微血管豐富且不受說話/咀嚼影響；臉頰在說話時肌肉運動會產生雜訊

### Decision 3: 座標 clamp 防呆
- **選擇**: `x1 = max(0, x1)`, `x2 = min(width, x2)` 等
- **理由**: 人臉轉頭或靠近畫面邊緣時，ROI 座標可能超出影像範圍導致 IndexError

## Risks / Trade-offs

- [YOLO 推論速度拖慢 FPS] → 使用 nano 版本 (YOLOv8n)；必要時降低推論頻率（每 N 幀偵測一次）
- [人臉消失時程式崩潰] → 偵測不到人臉時保持上一幀的 ROI 或顯示提示文字
- [模型權重檔案大] → 首次自動下載，後續使用快取
