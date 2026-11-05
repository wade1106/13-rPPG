## Why

Stage 1-5 已建立完整的 rPPG 訊號處理管線，但各模組散落於獨立腳本中，缺乏統一的使用者介面。需要一個 PyQt5 桌面儀表板將所有積木組裝起來：左側顯示相機畫面與 ROI，右側顯示即時心率/呼吸率數值與脈搏波形圖，並以 QThread 多執行緒架構確保 UI 不凍結。運算頻率與顯示刷新率需解耦，實現如 ICU 監視器般的穩定易讀體驗。

## What Changes

- 新增 `stage6_bio_dashboard.py`：PyQt5 整合型桌面監測儀表板
- 重用 Stage 1-5 模組：camera (相機串流)、face (YOLO/ROI)、signal (POS/filter)、vitals (HR/RR)
- QThread 工作線程：5 Hz 高頻運算，與 UI 主線程分離
- 左側：相機畫面 + ROI 疊加
- 右側上方：HR (BPM) 與 RR (RPM) 大字數值顯示
- 右側下方：即時脈搏波形示波器（pyqtgraph，30 秒滾動視窗）
- Start/Stop 按鈕控制量測流程
- QTimer 1 Hz 定拍刷新數值，避免高頻跳動
- 示波器以物理時間戳為 X 軸，鎖定 30 秒視窗

## Capabilities

### New Capabilities
- `qt-dashboard`: PyQt5 主視窗佈局、相機畫面顯示、數值面板、Start/Stop 控制
- `worker-thread`: QThread 信號工作線程、信號槽 (signal/slot) 通訊、運算與 UI 解耦
- `waveform-display`: pyqtgraph 即時脈搏波示波器、30 秒物理時間軸滾動視窗、1 Hz 數值定拍刷新

### Modified Capabilities

## Impact

- 新增依賴：`PyQt5`、`pyqtgraph`
- 產出檔案：`stage6_bio_dashboard.py`
- 重用模組：camera.py / face.py / signal_proc.py / vitals 邏輯
- 這是最終整合產出，需要 Stage 1-5 全部模組就緒
