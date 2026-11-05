## 1. Environment Setup

- [ ] 1.1 確認 conda 環境 rPPG (Python 3.11) 已建立，安裝 opencv-python
- [ ] 1.2 驗證 `import cv2` 可正常執行

## 2. Camera Initialization

- [ ] 2.1 實作 `cv2.VideoCapture(0, cv2.CAP_DSHOW)` 開啟相機
- [ ] 2.2 實作 fallback 機制：index 0 失敗時自動嘗試 index 1
- [ ] 2.3 加入 `isOpened()` 檢查與診斷訊息輸出

## 3. Video Streaming Loop

- [ ] 3.1 實作主迴圈：讀取每幀影像並以 `cv2.imshow` 顯示
- [ ] 3.2 實作水平鏡像翻轉 `cv2.flip(frame, 1)`
- [ ] 3.3 實作即時 FPS 計算（使用 `time.time()` 計算幀間隔）
- [ ] 3.4 以 `cv2.putText` 將 FPS 數值疊加顯示於畫面左上角

## 4. Safe Exit

- [ ] 4.1 實作按 Q 鍵偵測 (`cv2.waitKey(1) & 0xFF == ord('q')`)
- [ ] 4.2 退出時呼叫 `cap.release()` 與 `cv2.destroyAllWindows()`
- [ ] 4.3 處理 `cap.read()` 回傳 `ret=False` 的異常情境

## 5. Verification

- [ ] 5.1 執行 stage1_camera_test.py 確認畫面即時顯示
- [ ] 5.2 確認 FPS 數值穩定顯示
- [ ] 5.3 確認鏡像翻轉正確（揮右手畫面右側動）
- [ ] 5.4 確認按 Q 正常退出無殘留視窗
