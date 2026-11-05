## ADDED Requirements

### Requirement: Mean normalization
系統 SHALL 對 RGB 時序資料執行均值正規化：`Cn = C / mean(C)`，將各通道除以其滑動視窗內的均值，消除靜態環境光強基線。

#### Scenario: Normalize RGB channels
- **WHEN** 輸入一段 RGB 時序矩陣 (N x 3) 與視窗長度
- **THEN** 輸出正規化後的矩陣，各通道均值接近 1.0

### Requirement: POS projection
系統 SHALL 使用投影矩陣 `[[0, 1, -1], [-2, 1, 1]]` 將正規化後的 RGB 資料投影為兩個正交訊號通道 S0 和 S1。

#### Scenario: Project normalized RGB to S0, S1
- **WHEN** 輸入正規化後的 RGB 矩陣
- **THEN** 輸出 S0 = G_n - B_n 和 S1 = -2*R_n + G_n + B_n 兩條時序訊號

### Requirement: Dynamic alpha weighting
系統 SHALL 計算 `alpha = std(S0) / std(S1)` 並融合為 `H = S0 + alpha * S1`，產出單軌 rPPG 脈搏波。

#### Scenario: Fuse S0 and S1 into pulse signal
- **WHEN** 輸入 S0 和 S1 兩條訊號
- **THEN** 輸出動態加權融合後的單軌訊號 H，脈搏特徵比單獨任一通道更穩定

### Requirement: Sliding window processing
系統 SHALL 以滑動視窗方式處理時序資料，視窗長度預設 2.5 秒。

#### Scenario: Process with 2.5s window
- **WHEN** 緩衝區資料長度超過 2.5 秒
- **THEN** 系統對最近 2.5 秒的資料執行 POS 運算

### Requirement: Modular function interface
signal_proc.py SHALL 封裝為獨立函式庫，函式介面接受 RGB 時序矩陣與取樣率 (fs) 作為輸入，回傳處理後的 rPPG 訊號。

#### Scenario: Call POS function independently
- **WHEN** 外部模組呼叫 `pos_process(rgb_matrix, fs)` 函式
- **THEN** 回傳濾波後的 rPPG 訊號陣列，不依賴任何相機或 GUI 模組
