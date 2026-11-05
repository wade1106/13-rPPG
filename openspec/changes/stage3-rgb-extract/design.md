## Context

Stage 2 產出的 Skin ROI 是一個影像區域（2D 像素矩陣），需要將其壓縮為每幀一筆 (R, G, B) 數據點，累積成時間序列後才能進入訊號處理管線。每幀只做一件事：把 ROI 的像素壓縮成一組代表值。

## Goals / Non-Goals

**Goals:**
- 計算每幀 ROI 內像素的 R、G、B 通道均值
- 建立固定長度的時序緩衝區（如 300 幀 = ~10 秒 @ 30 FPS）
- 即時繪製 RGB 三通道波形圖，可觀察到微小規律的脈搏光學波動
- 記錄每幀的高精度時間戳 (`time.time()`)，為 Stage 4 時間戳內插做準備

**Non-Goals:**
- 不做 POS 投影或帶通濾波（Stage 4）
- 不做 Peak Detection 或 HR 計算（Stage 5）
- 不做 YCbCr 膚色分割（進階穩定機制）

## Decisions

### Decision 1: NumPy 陣列作為 Ring Buffer
- **選擇**: 使用 `collections.deque(maxlen=N)` 或固定大小 NumPy 陣列
- **理由**: deque 自動丟棄最舊資料，簡單高效；NumPy 陣列便於後續矩陣運算
- **替代方案**: Python list（動態擴展，需手動裁切）

### Decision 2: OpenCV 繪圖作為即時視覺化
- **選擇**: 使用 OpenCV 在獨立視窗或畫面下方繪製 RGB 曲線
- **理由**: 不需額外引入 matplotlib 的即時繪圖複雜度，與現有 OpenCV 視窗整合自然
- **替代方案**: matplotlib animation（功能強但即時效能差）、pyqtgraph（Stage 6 再引入）

### Decision 3: 記錄物理時間戳
- **選擇**: 每幀記錄 `time.time() - start_time` 作為真實時間軸
- **理由**: WebCam FPS 不穩定，不能用 frame index 當時間軸；此數據為 Stage 4 interp1d 校正基礎

## Risks / Trade-offs

- [ROI 面積太小導致均值不穩定] → 設定最小 ROI 面積閾值，低於閾值跳過該幀
- [即時繪圖拖慢主迴圈] → 降低繪圖更新頻率（每 N 幀更新一次圖表）
- [緩衝區太短看不到趨勢] → 預設 10 秒，可調整
