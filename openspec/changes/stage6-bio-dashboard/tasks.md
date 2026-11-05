## 1. Dependencies & Module Refactoring

- [ ] 1.1 安裝 PyQt5 與 pyqtgraph
- [ ] 1.2 將 Stage 1-5 核心邏輯重構為可 import 的模組（camera.py, face.py, signal_proc.py, vitals.py）
- [ ] 1.3 確認各模組可獨立 import 且無 side effect

## 2. PyQt5 Main Window Layout

- [ ] 2.1 建立 QMainWindow 主視窗，左右 QSplitter/QHBoxLayout 分欄
- [ ] 2.2 左側：QLabel 作為相機畫面顯示區域
- [ ] 2.3 右側上方：HR (BPM) 與 RR (RPM) 大字 QLabel，monospace 字型，居中對齊
- [ ] 2.4 右側下方：pyqtgraph PlotWidget 波形示波器
- [ ] 2.5 底部：Start / Stop 按鈕

## 3. QThread Worker

- [ ] 3.1 建立 SignalWorker(QThread)，封裝完整管線：camera → face → signal → vitals
- [ ] 3.2 定義 pyqtSignal：frame_ready(QImage)、vitals_ready(dict)、error(str)
- [ ] 3.3 主迴圈以 try/except 包裹，異常發射 error signal
- [ ] 3.4 實作停止機制：running flag + 退出時釋放相機

## 4. UI Update & Display

- [ ] 4.1 連接 frame_ready signal 至左側 QLabel 更新 slot
- [ ] 4.2 OpenCV BGR → RGB → QImage 轉換
- [ ] 4.3 vitals_ready signal 緩存最新數值
- [ ] 4.4 QTimer 1000ms 定拍讀取緩存數值，更新 HR/RR QLabel
- [ ] 4.5 pyqtgraph 波形以物理時間戳為 X 軸，setXRange(0, 30) 鎖定 30 秒視窗

## 5. Start/Stop Lifecycle

- [ ] 5.1 Start 按鈕：啟動 SignalWorker 線程
- [ ] 5.2 Stop 按鈕：設定 running=False，等待線程結束，數值歸零
- [ ] 5.3 關閉視窗事件 (closeEvent)：確保線程安全退出
- [ ] 5.4 開機前 15 秒顯示 "--" 與「緩衝中...」提示

## 6. Integration & Verification

- [ ] 6.1 執行 stage6_bio_dashboard.py 驗證完整 UI 顯示
- [ ] 6.2 驗證相機畫面即時且 ROI 標記可見
- [ ] 6.3 驗證 HR/RR 數值穩定（1 秒更新一次，無劇烈跳動）
- [ ] 6.4 驗證波形示波器 30 秒平穩滾動
- [ ] 6.5 驗證 Start/Stop 正常運作，Stop 後無殘留線程
- [ ] 6.6 驗證關閉視窗後程式完全退出
