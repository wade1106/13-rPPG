## 1. Dependencies Setup

- [ ] 1.1 安裝 ultralytics 套件 (`pip install ultralytics`)
- [ ] 1.2 下載或確認 YOLOv8 face 預訓練權重可用

## 2. Face Detection Module

- [ ] 2.1 載入 YOLO face 模型，封裝為可重用函式
- [ ] 2.2 實作人臉偵測：對每幀執行推論，回傳 bounding box 列表
- [ ] 2.3 實作最大臉選取邏輯（面積最大 + 信心分數最高）
- [ ] 2.4 處理無人臉偵測情境（回傳空結果，不崩潰）

## 3. Skin ROI Extraction

- [ ] 3.1 實作額頭 ROI 座標計算（人臉框上方 1/3 區域）
- [ ] 3.2 實作臉頰 ROI 座標計算（備選方案）
- [ ] 3.3 實作邊界 clamp 防呆 (`max(0, x1)`, `min(width, x2)` 等)
- [ ] 3.4 ROI 隨人臉位置即時更新

## 4. Visualization

- [ ] 4.1 繪製人臉 bounding box（矩形框 + 信心分數文字）
- [ ] 4.2 繪製額頭/臉頰 ROI 綠色矩形框
- [ ] 4.3 延續 Stage 1 的 FPS 顯示與鏡像翻轉

## 5. Integration & Verification

- [ ] 5.1 整合 Stage 1 相機串流 + YOLO 偵測 + ROI 為 stage2_face_roi.py
- [ ] 5.2 驗證移動頭部時 ROI 緊密跟隨
- [ ] 5.3 驗證人臉靠近邊緣或消失時程式不崩潰
- [ ] 5.4 驗證開口說話時額頭 ROI 不受影響
