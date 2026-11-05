## 1. RGB Mean Extraction

- [ ] 1.1 實作 ROI 像素 RGB 三通道空間均值計算函式
- [ ] 1.2 加入 ROI 最小面積閾值檢查，面積過小時跳過

## 2. Time-series Buffer

- [ ] 2.1 建立固定長度的 Ring Buffer（deque 或 NumPy 陣列，預設 300 幀）
- [ ] 2.2 每幀將 (timestamp, mean_R, mean_G, mean_B) 追加至緩衝區
- [ ] 2.3 記錄每幀高精度物理時間戳 `time.time() - start_time`

## 3. Real-time Visualization

- [ ] 3.1 實作即時 RGB 波形繪圖（使用 OpenCV 或 matplotlib）
- [ ] 3.2 以紅、綠、藍三色分別繪製 R、G、B 通道曲線
- [ ] 3.3 控制繪圖更新頻率，避免拖慢主迴圈

## 4. Integration & Verification

- [ ] 4.1 整合 Stage 1-2 模組，建立 stage3_rgb_extract.py
- [ ] 4.2 驗證 RGB 波形即時顯示且連續滾動
- [ ] 4.3 驗證安靜面對鏡頭時 Green 通道可觀察到微弱規律波動
- [ ] 4.4 驗證時間戳記錄正確（非幀編號）
