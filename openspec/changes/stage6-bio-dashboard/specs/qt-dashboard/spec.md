## ADDED Requirements

### Requirement: Main window layout
系統 SHALL 建立 PyQt5 主視窗，左右分欄佈局：左側為相機畫面，右側為數值面板與波形圖。

#### Scenario: Application window startup
- **WHEN** 使用者執行 stage6_bio_dashboard.py
- **THEN** 顯示主視窗，左側為相機畫面區域，右側上方為 HR/RR 數值，右側下方為波形圖

### Requirement: Camera feed display via QLabel
系統 SHALL 將 OpenCV 影像幀轉換為 QImage 並顯示於左側 QLabel，不使用 cv2.imshow。

#### Scenario: Display camera with ROI overlay
- **WHEN** 工作線程產出帶有 ROI 標記的影像幀
- **THEN** 左側 QLabel 即時更新顯示相機畫面與 ROI 疊加框

### Requirement: Vitals numeric display
系統 SHALL 在右側面板以大字體顯示 HR (BPM) 和 RR (RPM) 數值，使用 monospace 字型與居中對齊。

#### Scenario: Display heart rate and breathing rate
- **WHEN** vitals 計算結果就緒
- **THEN** 右側面板顯示 HR 與 RR 數值，格式如 "72 BPM" 和 "12 RPM"

### Requirement: Start/Stop control
系統 SHALL 提供 Start 和 Stop 按鈕控制量測流程的啟動與停止。

#### Scenario: Start measurement
- **WHEN** 使用者點擊 Start 按鈕
- **THEN** 啟動工作線程，開始相機擷取與訊號處理

#### Scenario: Stop measurement
- **WHEN** 使用者點擊 Stop 按鈕
- **THEN** 停止工作線程，釋放相機資源，數值歸零或顯示 "--"
