## Context

Stage 1-5 的模組（camera、face、signal、vitals）已各自驗證通過。現在需要組裝為一個專業的桌面監測應用。核心挑戰在於：運算密集的訊號處理不能阻塞 UI 線程，且數值刷新頻率需與運算頻率解耦，才能呈現如醫療監視器般的穩定體驗。

## Goals / Non-Goals

**Goals:**
- PyQt5 桌面應用，左右分欄佈局
- QThread 工作線程執行相機擷取 + YOLO + POS + vitals 運算
- Signal/Slot 機制將運算結果傳回 UI 主線程
- 數值以 QTimer 1 Hz 定拍刷新（非跟隨運算頻率）
- pyqtgraph 示波器以物理時間戳為 X 軸，鎖定 30 秒滾動視窗
- Start/Stop 按鈕控制量測生命週期
- 重用 Stage 1-5 既有模組，不重寫核心邏輯

**Non-Goals:**
- 不做資料匯出 CSV（延伸功能）
- 不做 Signal Quality Index 顯示（進階挑戰）
- 不做多相機或遠端串流支援
- 不做跨平台打包

## Decisions

### Decision 1: PyQt5 + pyqtgraph 技術棧
- **選擇**: PyQt5 做 UI 框架，pyqtgraph 做即時波形繪圖
- **理由**: pyqtgraph 基於 Qt，與 PyQt5 原生整合，即時繪圖效能遠超 matplotlib；支持 30+ FPS 波形更新
- **替代方案**: Tkinter（功能弱）、matplotlib animation（效能差）、Dear ImGui（學習曲線高）

### Decision 2: QThread + Signal/Slot 架構
- **選擇**: 獨立 QThread 工作線程，透過 pyqtSignal 發送結果
- **理由**: 保持 UI 主線程不凍結；Signal/Slot 是 Qt 的線程安全通訊機制
- **替代方案**: QRunnable/ThreadPool（生命週期管理複雜）、asyncio（與 Qt 事件迴圈整合困難）

### Decision 3: 運算 5 Hz vs 顯示 1 Hz 雙頻架構
- **選擇**: SignalWorker 每 5 幀發送一次運算結果（~5 Hz），但數值 QLabel 由獨立 QTimer 1 秒刷新一次
- **理由**: 工作線程保持高頻保證示波器即時性；數值低頻刷新避免視覺疲勞與字型寬度微跳
- **替代方案**: 統一頻率（數值跳動太快或波形太慢）

### Decision 4: 示波器 30 秒物理時間軸
- **選擇**: X 軸 setXRange(0, 30)，資料以 `timestamp - cutoff` 裁切，不以幀數計算
- **理由**: WebCam FPS 不穩定，以幀數計算會導致時間軸膨脹/壓縮；物理時間軸保證 30 秒永遠是 30 秒

### Decision 5: 模組化重用
- **選擇**: import camera / face / signal_proc 模組，不在 dashboard 中重寫邏輯
- **理由**: 維持單一職責，各模組可獨立測試與修改

## Risks / Trade-offs

- [PyQt5 與 OpenCV imshow 衝突] → dashboard 中不使用 cv2.imshow，改用 QLabel + QImage 顯示相機畫面
- [工作線程崩潰拖垮主程式] → try/except 包裹工作線程主迴圈，錯誤透過 signal 回報
- [相機被佔用] → 啟動前檢查相機可用性，提供錯誤提示
- [非等寬字型數值微跳] → 使用 monospace 字型 + 固定寬度 QLabel + AlignCenter
