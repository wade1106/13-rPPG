## Context

原始 RGB 時序中，環境光照變化、頭部運動產生的亮度波動可達脈搏訊號的 20-50 倍。單純看 Green 通道無法有效分離脈搏與光照。POS 演算法利用 RGB 三通道的色彩比例關係，通過正交投影消除與光照共線的分量，保留與血液容積變化正交的分量。

## Goals / Non-Goals

**Goals:**
- 封裝獨立的 POS 演算法函式庫 (`signal_proc.py`)，輸入 RGB 矩陣與取樣率，輸出濾波後的 rPPG 脈搏波
- 實作 POS 三步驟：Mean Normalization → Projection (S0, S1) → Alpha Weighting (H = S0 + alpha * S1)
- 實作 Butterworth 帶通濾波器 (0.7-3.0 Hz)
- 建立零硬體依賴的合成訊號單元測試，驗證 FFT 主頻誤差 < 2.0 BPM
- 生成 Matplotlib 診斷圖表（時域 + 頻域對照）

**Non-Goals:**
- 不做 Peak Detection 或 HR/RR 數值計算（Stage 5）
- 不做即時相機串流整合（此階段純離線驗證）
- 不做 YCbCr 膚色分割

## Decisions

### Decision 1: POS 投影矩陣
- **選擇**: 固定投影矩陣 `[[0, 1, -1], [-2, 1, 1]]`
- **理由**: 來自 Wang et al. (2017) 論文，S0 通道捕捉 G-B 差異，S1 通道捕捉色調變化，二者正交於膚色平面
- **替代方案**: CHROM（類似但權重不同）、ICA（計算量大）

### Decision 2: 滑動視窗 2.5 秒
- **選擇**: 每次 POS 運算使用 2.5 秒的資料窗口
- **理由**: 太短（<1.5s）不穩定，太長（>5s）反應慢且容易混入狀態變化；2.5s 涵蓋約 2-3 個心跳週期

### Decision 3: Butterworth 4 階帶通
- **選擇**: `scipy.signal.butter(order=4, Wn=[0.7, 3.0], btype='band', fs=fs)`
- **理由**: 0.7 Hz = 42 BPM（最慢合理心率），3.0 Hz = 180 BPM（最快合理心率）；4 階平衡衰減斜率與相位失真

### Decision 4: 合成測試驗證策略
- **選擇**: 生成 75 BPM 正弦波 + 直流偏移 + 呼吸漂移 + 高斯雜訊，通過 POS + filter 後用 FFT 驗證主頻
- **理由**: 零硬體依賴，可在 CI/CD 中自動執行，閉環驗證演算法正確性

## Risks / Trade-offs

- [POS 投影矩陣假設膚色光學模型] → 對不同膚色個體可能有偏差，但對多數情境已夠用
- [帶通濾波引入邊界效應] → 使用 `filtfilt`（零相位濾波）消除相位延遲
- [滑動視窗重疊計算量] → 每次只處理一個窗口，計算量可控
