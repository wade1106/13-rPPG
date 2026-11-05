## ADDED Requirements

### Requirement: QThread signal worker
系統 SHALL 建立獨立的 QThread 工作線程，執行相機擷取、YOLO 偵測、ROI 萃取、POS 處理與 vitals 計算。

#### Scenario: Worker thread runs independently
- **WHEN** Start 按鈕被點擊
- **THEN** 工作線程在背景執行完整管線，UI 主線程不凍結

### Requirement: Signal/Slot communication
工作線程 SHALL 透過 pyqtSignal 將運算結果（影像幀、HR、RR、波形資料）傳送至 UI 主線程的對應 Slot。

#### Scenario: Emit vitals data to UI
- **WHEN** 工作線程完成一輪運算
- **THEN** 發射 signal 包含最新影像幀、HR 值、RR 值與波形資料

### Requirement: Thread-safe lifecycle management
系統 SHALL 在 Stop 時安全終止工作線程，釋放所有硬體資源，不產生殘留線程或記憶體洩漏。

#### Scenario: Clean thread shutdown
- **WHEN** 使用者點擊 Stop 或關閉主視窗
- **THEN** 工作線程收到停止訊號，完成當前迭代後退出，相機被正確釋放

### Requirement: Worker thread error handling
工作線程 SHALL 以 try/except 包裹主迴圈，異常時透過 signal 回報錯誤訊息，不拖垮主程式。

#### Scenario: Camera disconnected during operation
- **WHEN** 相機在運行中被拔除
- **THEN** 工作線程捕獲異常，發射錯誤 signal，UI 顯示錯誤提示
