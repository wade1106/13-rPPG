## ADDED Requirements

### Requirement: Forehead ROI extraction
系統 SHALL 從人臉 bounding box 中計算額頭 ROI 座標，區域為人臉框上方約 1/3 高度、水平方向內縮避開邊緣。

#### Scenario: Calculate forehead ROI from face box
- **WHEN** 偵測到人臉 bounding box (x1, y1, x2, y2)
- **THEN** 系統計算額頭 ROI 座標，覆蓋人臉框上方 30%-40% 的區域

#### Scenario: Head movement tracking
- **WHEN** 使用者移動頭部
- **THEN** 額頭 ROI 框即時跟隨人臉位置更新

### Requirement: Cheek ROI extraction (alternative)
系統 SHALL 支援臉頰 ROI 作為備選量測區域，位於人臉框中下方兩側。

#### Scenario: Calculate cheek ROI
- **WHEN** 偵測到人臉 bounding box
- **THEN** 系統可計算左右臉頰 ROI 座標，避開眼睛與嘴唇區域

### Requirement: ROI boundary clamping
系統 SHALL 對所有 ROI 座標進行邊界 clamp，確保不超出影像範圍。

#### Scenario: Face near frame edge
- **WHEN** 人臉靠近畫面邊緣，導致 ROI 座標計算結果超出影像尺寸
- **THEN** 系統以 `max(0, coord)` 和 `min(dimension, coord)` clamp 座標，不拋出 IndexError

#### Scenario: Face partially out of frame
- **WHEN** 人臉部分超出畫面（如轉頭）
- **THEN** ROI 自動裁切至可見範圍內，程式不崩潰

### Requirement: ROI visualization
系統 SHALL 在畫面上以綠色矩形框標示 Skin ROI 位置。

#### Scenario: Draw ROI overlay
- **WHEN** ROI 計算完成
- **THEN** 畫面上以綠色矩形框顯示 ROI 範圍，與人臉框顏色區分
