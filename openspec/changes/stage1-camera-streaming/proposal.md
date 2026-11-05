## Why

rPPG 系統的第一步是建立穩定的 WebCam 視訊串流基礎。沒有可靠的影像輸入，後續所有人臉偵測、ROI 萃取、訊號處理都無法進行。需要處理 Windows DirectShow 後端相容性、相機裝置探測與容錯機制。

## What Changes

- 新增 `stage1_camera_test.py`：建立 WebCam 即時視訊串流程式
- 使用 OpenCV `cv2.VideoCapture` 搭配 DirectShow 後端 (`CAP_DSHOW`) 開啟相機
- 即時顯示 FPS 於畫面左上角
- 支援鏡像翻轉 (水平 flip)，讓畫面符合直覺操作
- 按 Q 鍵安全退出，正確釋放相機資源
- 相機裝置索引 fallback 機制 (index 0 → 1 自動探測)

## Capabilities

### New Capabilities
- `camera-streaming`: WebCam 視訊串流擷取、DirectShow 後端初始化、FPS 計算與顯示、鏡像翻轉、安全退出與資源釋放

### Modified Capabilities

## Impact

- 新增依賴：`opencv-python`
- 需要 conda 環境 `rPPG` (Python 3.11)
- 產出檔案：`stage1_camera_test.py`
- 此模組為後續所有 Stage 的基礎輸入層
