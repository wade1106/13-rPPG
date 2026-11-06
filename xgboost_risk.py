"""xgboost_risk.py — XGBoost 風險分類推論模組

提供輕量級的推論介面，供 Dashboard 即時呼叫。
啟動時載入預訓練模型（models/xgboost_risk.json），
推論一次 < 1ms，不影響即時效能。

使用方式：
    from xgboost_risk import RiskClassifier

    clf = RiskClassifier()  # 載入模型
    result = clf.predict(hrv_feature_vector)
    # result = {'label': 'Normal', 'confidence': 0.92,
    #           'probabilities': {'Normal': 0.92, 'Warning': 0.06, 'Abnormal': 0.02}}
"""

import os
import numpy as np
import xgboost as xgb

# 風險等級定義
RISK_LABELS = {0: 'Normal', 1: 'Warning', 2: 'Abnormal'}

# 風險等級對應的顏色（用於 Dashboard 顯示）
RISK_COLORS = {
    'Normal': '#00FF00',    # 綠色
    'Warning': '#FFAA00',   # 橙色
    'Abnormal': '#FF0000',  # 紅色
}

# 模型檔案路徑（相對於本檔案所在目錄）
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'xgboost_risk.json')


class RiskClassifier:
    """XGBoost 生理異常風險分類器。

    載入預訓練模型，對 HRV 特徵向量進行風險分類。
    """

    def __init__(self, model_path=None):
        """初始化並載入模型。

        Args:
            model_path: 模型檔案路徑，預設為 models/xgboost_risk.json
        """
        path = model_path or MODEL_PATH
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"找不到模型檔案: {path}\n"
                f"請先執行 python train_xgboost.py 訓練模型"
            )
        self.model = xgb.XGBClassifier()
        self.model.load_model(path)

    def predict(self, features):
        """對單筆 HRV 特徵向量進行風險分類。

        Args:
            features: 1D ndarray，長度 12 的 HRV 特徵向量
                     （順序須與 hrv_features.FEATURE_NAMES 一致）

        Returns:
            dict:
                - label: 風險等級名稱 ('Normal' / 'Warning' / 'Abnormal')
                - label_id: 風險等級代碼 (0 / 1 / 2)
                - confidence: 預測信心度 (0.0 ~ 1.0)
                - color: 對應的顯示顏色 (hex)
                - probabilities: 各類別的預測機率 dict
        """
        if features is None:
            return self._default_result()

        # 確保輸入為 2D (1, n_features)
        X = np.array(features, dtype=np.float64).reshape(1, -1)

        # 預測類別與機率
        label_id = int(self.model.predict(X)[0])
        probs = self.model.predict_proba(X)[0]

        label = RISK_LABELS[label_id]
        confidence = float(probs[label_id])

        return {
            'label': label,
            'label_id': label_id,
            'confidence': confidence,
            'color': RISK_COLORS[label],
            'probabilities': {RISK_LABELS[i]: float(p) for i, p in enumerate(probs)},
        }

    @staticmethod
    def _default_result():
        """資料不足時的預設結果。"""
        return {
            'label': '--',
            'label_id': -1,
            'confidence': 0.0,
            'color': '#888888',
            'probabilities': {},
        }
