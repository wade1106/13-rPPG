## ADDED Requirements

### Requirement: YOLO face detection model loading
系統 SHALL 載入 YOLOv8 人臉偵測預訓練模型，並在載入失敗時輸出明確錯誤訊息。

#### Scenario: Model loaded successfully
- **WHEN** 程式啟動時
- **THEN** YOLO 模型成功載入且可進行推論

### Requirement: Real-time face detection
系統 SHALL 對每幀影像執行 YOLO 人臉偵測，取信心分數最高且面積最大的人臉作為主要偵測目標。

#### Scenario: Single face detected
- **WHEN** 畫面中有一張人臉
- **THEN** 系統回傳該人臉的 bounding box 座標 (x1, y1, x2, y2)

#### Scenario: Multiple faces detected
- **WHEN** 畫面中有多張人臉
- **THEN** 系統選取面積最大的人臉作為主要目標

#### Scenario: No face detected
- **WHEN** 畫面中無人臉
- **THEN** 系統不拋出例外，回傳空結果或保持上一幀的偵測結果

### Requirement: Face bounding box visualization
系統 SHALL 在畫面上繪製人臉 bounding box 矩形框與信心分數文字。

#### Scenario: Draw face box on frame
- **WHEN** 偵測到人臉
- **THEN** 畫面上以矩形框標示人臉位置，並顯示信心分數
