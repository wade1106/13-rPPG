## Context

rPPG 系統需要穩定的即時視訊輸入作為整條管線的起點。Windows 平台上 OpenCV 的預設後端常有相容性問題（黑屏、卡住），需要明確指定 DirectShow 後端。WebCam FPS 會受 USB 頻寬、CPU 排程影響，需即時監測。

## Goals / Non-Goals

**Goals:**
- 建立可靠的 WebCam 視訊串流，確保畫面即時、不卡頓
- 即時計算並顯示 FPS，作為後續取樣率校準的基礎
- 鏡像翻轉讓使用者操作直覺
- 按 Q 安全退出並正確釋放所有硬體資源
- 相機開啟失敗時自動嘗試其他裝置索引

**Non-Goals:**
- 不處理人臉偵測（Stage 2）
- 不處理 ROI 萃取或訊號處理
- 不建立 GUI 框架（Stage 6）
- 不處理多相機同時串流

## Decisions

### Decision 1: 使用 DirectShow 後端
- **選擇**: `cv2.VideoCapture(index, cv2.CAP_DSHOW)`
- **理由**: Windows 上 MSMF 後端常導致黑屏或延遲，DirectShow 在大多數 USB WebCam 上最穩定
- **替代方案**: MSMF（預設但不穩定）、GStreamer（需額外安裝）

### Decision 2: FPS 使用滑動平均計算
- **選擇**: 以 `time.time()` 記錄每幀時間差，計算即時 FPS
- **理由**: 比 `cv2.CAP_PROP_FPS` 讀取的標稱值更能反映真實效能，且為 Stage 4 時間戳校正打基礎

### Decision 3: Fallback 相機探測
- **選擇**: index 0 失敗時自動嘗試 index 1，最多探測 2 個裝置
- **理由**: 筆電常有虛擬相機佔用 index 0，實體相機在 index 1

## Risks / Trade-offs

- [DirectShow 不支援某些相機] → 輸出明確錯誤訊息，提示使用者檢查驅動
- [FPS 波動大] → 此階段僅顯示，不影響功能；Stage 4 再處理時間戳內插校正
- [相機被其他程式佔用] → 檢查 `isOpened()` 並輸出診斷訊息
