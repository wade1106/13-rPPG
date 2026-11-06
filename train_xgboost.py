"""train_xgboost.py — 合成 HRV 資料生成 + XGBoost 風險分類器訓練

本腳本執行一次即可，產出 models/xgboost_risk.json 供即時推論使用。

流程：
1. 根據醫學文獻的 HRV 參考範圍，為三種生理狀態生成合成資料
2. 訓練 XGBoost 多類別分類器（3 類：Normal / Warning / Abnormal）
3. 評估模型準確率並輸出混淆矩陣
4. 儲存模型為 JSON 格式

三類生理狀態的 HRV 特徵參考範圍（來自 ESC/NASPE 標準與臨床文獻）：

| 特徵        | Normal          | Warning          | Abnormal         |
|------------|-----------------|------------------|------------------|
| SDNN (ms)  | 100-200         | 50-100           | 10-50            |
| RMSSD (ms) | 30-80           | 15-30            | 5-15             |
| pNN50 (%)  | 10-40           | 3-10             | 0-3              |
| LF/HF      | 0.5-2.0         | 2.0-5.0          | 5.0-15.0         |
| SD1 (ms)   | 20-60           | 10-20            | 3-10             |
| SD2 (ms)   | 50-150          | 25-50            | 8-25             |

注意：合成資料無法完全代替真實臨床資料，但足以建立一個合理的基線模型。
未來可用 PhysioNet 等公開資料集替換合成資料，重新訓練即可提升準確度。
"""

import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from hrv_features import FEATURE_NAMES

# 隨機種子（確保可重現）
np.random.seed(42)

# 類別定義
LABELS = {0: 'Normal', 1: 'Warning', 2: 'Abnormal'}
N_SAMPLES_PER_CLASS = 2000  # 每類 2000 筆 = 總計 6000 筆


def generate_samples(n, label):
    """為指定類別生成合成 HRV 特徵。

    每個特徵用常態分布取樣，mean 和 std 根據醫學文獻設定。
    加入隨機噪聲模擬個體差異。

    Args:
        n: 生成筆數
        label: 0=Normal, 1=Warning, 2=Abnormal

    Returns:
        (N, 12) ndarray，每列對應 FEATURE_NAMES 的順序
    """
    # 參考範圍定義：(mean, std) for each label
    # 順序：mean_rr, sdnn, rmssd, pnn50, mean_hr,
    #       lf_power, hf_power, lf_hf_ratio, total_power,
    #       sd1, sd2, sd_ratio
    params = {
        0: {  # Normal — 健康成人靜息狀態
            'mean_rr': (850, 80),      # ~70 BPM
            'sdnn': (140, 30),
            'rmssd': (50, 15),
            'pnn50': (25, 10),
            'mean_hr': (72, 8),
            'lf_power': (500, 200),
            'hf_power': (400, 150),
            'lf_hf_ratio': (1.2, 0.4),
            'total_power': (900, 300),
            'sd1': (38, 10),
            'sd2': (90, 25),
            'sd_ratio': (0.42, 0.1),
        },
        1: {  # Warning — 輕度自律神經失調（壓力/疲勞/輕度心律不整）
            'mean_rr': (750, 70),      # ~80 BPM
            'sdnn': (70, 15),
            'rmssd': (22, 6),
            'pnn50': (6, 3),
            'mean_hr': (82, 10),
            'lf_power': (350, 150),
            'hf_power': (120, 60),
            'lf_hf_ratio': (3.2, 1.0),
            'total_power': (470, 180),
            'sd1': (15, 4),
            'sd2': (38, 10),
            'sd_ratio': (0.40, 0.08),
        },
        2: {  # Abnormal — 嚴重自律神經失調（高風險）
            'mean_rr': (650, 100),     # ~95 BPM
            'sdnn': (28, 10),
            'rmssd': (9, 3),
            'pnn50': (1.2, 1.0),
            'mean_hr': (95, 15),
            'lf_power': (150, 80),
            'hf_power': (30, 20),
            'lf_hf_ratio': (8.0, 3.0),
            'total_power': (180, 90),
            'sd1': (6, 2),
            'sd2': (15, 5),
            'sd_ratio': (0.38, 0.1),
        },
    }

    p = params[label]
    samples = np.zeros((n, len(FEATURE_NAMES)))
    for i, name in enumerate(FEATURE_NAMES):
        mean, std = p[name]
        samples[:, i] = np.random.normal(mean, std, n)

    # 確保所有值非負（生理特徵不可能為負）
    samples = np.clip(samples, 0, None)

    return samples


def main():
    print("=" * 60)
    print("XGBoost 風險分類器訓練")
    print("=" * 60)

    # ── 1. 生成合成資料 ──────────────────────────────────────
    print(f"\n[1/4] 生成合成 HRV 資料（每類 {N_SAMPLES_PER_CLASS} 筆）...")

    X_list, y_list = [], []
    for label in range(3):
        samples = generate_samples(N_SAMPLES_PER_CLASS, label)
        X_list.append(samples)
        y_list.append(np.full(N_SAMPLES_PER_CLASS, label))
        print(f"  - {LABELS[label]}: {samples.shape}")

    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    # 打亂順序
    shuffle_idx = np.random.permutation(len(y))
    X, y = X[shuffle_idx], y[shuffle_idx]

    print(f"  總計: {X.shape[0]} 筆, {X.shape[1]} 個特徵")

    # ── 2. 切分訓練/測試集 ───────────────────────────────────
    print("\n[2/4] 切分訓練集 (80%) / 測試集 (20%)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  訓練集: {X_train.shape[0]} 筆")
    print(f"  測試集: {X_test.shape[0]} 筆")

    # ── 3. 訓練 XGBoost ─────────────────────────────────────
    print("\n[3/4] 訓練 XGBoost 模型...")

    model = xgb.XGBClassifier(
        n_estimators=100,           # 100 棵樹（合成資料不需要太多）
        max_depth=4,                # 最大深度 4（防止過擬合）
        learning_rate=0.1,          # 學習率
        objective='multi:softprob', # 多類別分類，輸出機率
        num_class=3,
        eval_metric='mlogloss',     # 多類別交叉熵損失
        random_state=42,
        use_label_encoder=False,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # ── 4. 評估模型 ─────────────────────────────────────────
    print("\n[4/4] 模型評估結果:")
    y_pred = model.predict(X_test)

    print("\n分類報告:")
    print(classification_report(
        y_test, y_pred,
        target_names=[LABELS[i] for i in range(3)]
    ))

    print("混淆矩陣:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  {'':>10} {'Normal':>8} {'Warning':>8} {'Abnormal':>8}")
    for i, row in enumerate(cm):
        print(f"  {LABELS[i]:>10} {row[0]:>8} {row[1]:>8} {row[2]:>8}")

    # 特徵重要性排行
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    print("\n特徵重要性排行:")
    for rank, idx in enumerate(sorted_idx, 1):
        print(f"  {rank}. {FEATURE_NAMES[idx]:>15}: {importances[idx]:.4f}")

    # ── 5. 儲存模型 ─────────────────────────────────────────
    model_path = "models/xgboost_risk.json"
    model.save_model(model_path)
    print(f"\n模型已儲存至: {model_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
