# rPPG 非接觸式生理監測系統

基於遠端光體積變化描記法（Remote Photoplethysmography, rPPG），透過普通 WebCam 擷取人臉影像，從皮膚微弱的色彩變化中萃取心跳脈搏訊號，實現**非接觸式**的即時生理監測。

## 功能

1. **攝影機擷取人臉影像** — WebCam 即時串流 + YOLO11-pose 人臉偵測
2. **rPPG 估測 HR** — POS 演算法從 RGB 時序萃取脈搏波，FFT 頻譜法估算心率
3. **呼吸訊號估測 RR** — 自相關函數從 RR interval 低頻調變萃取呼吸率
4. **HRV 特徵分析** — 從 RR intervals 計算時域、頻域、非線性共 12 項 HRV 特徵
5. **XGBoost 生理異常風險分類** — 基於 HRV 特徵進行三級風險分類（Normal / Warning / Abnormal）
6. **PyQt6 即時監控 Dashboard** — 整合相機畫面、生命徵象數值、rPPG 波形示波器、風險等級顯示
7. **Docker 部署**（規劃中）

## 系統架構

```
WebCam
  │
  ▼
┌──────────────────────────────────────────────────────┐
│  Stage 1: 相機串流                                     │
│  cv2.VideoCapture + DirectShow                        │
└──────────┬───────────────────────────────────────────┘
           ▼
┌──────────────────────────────────────────────────────┐
│  Stage 2: 人臉 ROI 偵測                               │
│  YOLO11-pose → 關鍵點 → 額頭/臉頰 ROI                   │
│  + YCbCr 膚色分割                                      │
└──────────┬───────────────────────────────────────────┘
           ▼
┌──────────────────────────────────────────────────────┐
│  Stage 3: RGB 時序萃取                                 │
│  多 ROI 膚色像素加權 RGB 均值 → Ring Buffer              │
└──────────┬───────────────────────────────────────────┘
           ▼
┌──────────────────────────────────────────────────────┐
│  Stage 4: 訊號處理                                     │
│  interp1d 等距內插 → POS 演算法 → Butterworth 帶通濾波   │
└──────────┬───────────────────────────────────────────┘
           ▼
┌──────────────────────────────────────────────────────┐
│  Stage 5: 生命徵象估算                                  │
│  FFT → HR (BPM)  /  自相關函數 → RR (RPM)              │
└──────────┬───────────────────────────────────────────┘
           ▼
┌──────────────────────────────────────────────────────┐
│  HRV + XGBoost 風險分類                                │
│  RR intervals → 12 項 HRV 特徵 → XGBoost → 風險等級     │
└──────────┬───────────────────────────────────────────┘
           ▼
┌──────────────────────────────────────────────────────┐
│  Stage 6: PyQt6 即時 Dashboard                        │
│  相機畫面 │ HR / RR / Risk Level │ rPPG 波形示波器       │
└──────────────────────────────────────────────────────┘
```

## 檔案結構

```
13-rPPG/
├── stage1_camera_test.py      # Stage 1: 相機串流測試
├── stage2_face_roi.py         # Stage 2: YOLO 人臉偵測 + ROI 框定
├── stage3_rgb_extract.py      # Stage 3: 多 ROI 膚色 RGB 均值萃取
├── stage4_signal_processing.py # Stage 4: POS 演算法 + 帶通濾波驗證
├── stage5_vitals.py           # Stage 5: 即時 HR + RR（OpenCV 版）
├── stage6_bio_dashboard.py    # Stage 6: PyQt6 整合 Dashboard（主程式）
├── signal_proc.py             # 核心訊號處理函式庫（POS、Butterworth）
├── hrv_features.py            # HRV 特徵萃取模組（時域/頻域/非線性）
├── xgboost_risk.py            # XGBoost 風險分類推論模組
├── train_xgboost.py           # XGBoost 模型訓練腳本（執行一次）
├── models/
│   └── xgboost_risk.json      # 訓練好的 XGBoost 模型
├── pyproject.toml             # 專案設定與依賴管理（uv）
└── yolo11n-pose.pt            # YOLO11 Pose 模型權重
```

## 安裝

### 環境需求

- Python >= 3.11
- WebCam（USB 或筆電內建）

### 安裝步驟

```bash
# 1. 安裝 uv（如果還沒有）
pip install uv

# 2. 安裝專案依賴
uv sync

# 3. 訓練 XGBoost 模型（首次使用需執行一次）
uv run python train_xgboost.py
```

## 執行

### 啟動 Dashboard（主程式）

```bash
uv run python stage6_bio_dashboard.py
```

啟動後點擊 **Start** 開始量測，約 8 秒後顯示 HR / RR / Risk Level。

### 執行各 Stage 獨立測試

```bash
uv run python stage1_camera_test.py      # 測試相機
uv run python stage2_face_roi.py         # 測試人臉偵測
uv run python stage3_rgb_extract.py      # 測試 RGB 萃取
uv run python stage4_signal_processing.py # 測試訊號處理
uv run python stage5_vitals.py           # 測試 HR/RR（OpenCV 版）
```

## 核心演算法

### POS 演算法（Plane Orthogonal to Skin）

將 RGB 三通道訊號投影到與膚色平面正交的方向，分離出微弱的脈搏訊號並消除光照干擾。來源：Wang et al. (2017)。

### HRV 特徵（12 維）

| 類別 | 特徵 | 說明 |
|------|------|------|
| 時域 | mean_rr, sdnn, rmssd, pnn50, mean_hr | 心跳間隔的統計量 |
| 頻域 | lf_power, hf_power, lf_hf_ratio, total_power | 自律神經頻帶能量 |
| 非線性 | sd1, sd2, sd_ratio | Poincare 散點圖幾何特徵 |

### XGBoost 風險分類

基於 HRV 特徵進行三級分類：

| 等級 | 顏色 | 意義 |
|------|------|------|
| Normal | 綠色 | 自律神經調控正常 |
| Warning | 橙色 | 輕度失調（壓力/疲勞） |
| Abnormal | 紅色 | 嚴重失調（建議就醫） |

## 技術棧

- **影像處理**: OpenCV, YOLO11-pose (Ultralytics)
- **訊號處理**: SciPy (Butterworth, Welch PSD, find_peaks)
- **機器學習**: XGBoost, scikit-learn
- **GUI**: PyQt6, pyqtgraph
- **套件管理**: uv
