## ADDED Requirements

### Requirement: Camera initialization with DirectShow backend
系統 SHALL 使用 OpenCV `cv2.VideoCapture` 搭配 `CAP_DSHOW` 後端開啟 WebCam，並在開啟後驗證 `isOpened()` 回傳 True。

#### Scenario: Successfully open default camera
- **WHEN** 程式啟動且 index 0 的相機可用
- **THEN** 系統開啟相機並開始擷取影像幀

#### Scenario: Default camera unavailable, fallback to index 1
- **WHEN** index 0 的相機無法開啟 (`isOpened()` 回傳 False)
- **THEN** 系統自動嘗試 index 1，若成功則繼續串流；若皆失敗則輸出診斷訊息並安全退出

### Requirement: Real-time FPS display
系統 SHALL 計算即時幀率並以文字疊加方式顯示於畫面左上角。

#### Scenario: FPS overlay on video frame
- **WHEN** 視訊串流運行中
- **THEN** 每幀畫面左上角顯示 `FPS: XX.X` 格式的即時幀率數值

### Requirement: Mirror flip
系統 SHALL 對每一幀影像進行水平翻轉 (`cv2.flip(frame, 1)`)，使畫面呈現鏡像效果。

#### Scenario: User waves right hand
- **WHEN** 使用者揮動右手
- **THEN** 畫面上顯示的手在使用者視角的右側（鏡像行為）

### Requirement: Safe exit and resource release
系統 SHALL 支援按 Q 鍵退出，退出時正確釋放相機資源並關閉所有 OpenCV 視窗。

#### Scenario: User presses Q to quit
- **WHEN** 使用者按下 Q 鍵
- **THEN** 系統呼叫 `cap.release()` 和 `cv2.destroyAllWindows()` 後正常結束程式

#### Scenario: Frame read failure during streaming
- **WHEN** `cap.read()` 回傳 `ret=False`
- **THEN** 系統輸出警告訊息並安全退出，不拋出未處理的例外
