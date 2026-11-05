"""Stage 2: YOLO11 人臉偵測 + Skin ROI 萃取

本程式為 rPPG 系統的第二階段，負責從視訊串流中偵測人臉，
並精準定位適合量測脈搏的皮膚區域 (ROI = Region of Interest)。

核心概念：
- 「偵測」(Detection) 和「量測」(Measurement) 是兩個不同問題：
  - YOLO 負責「人在哪裡？」→ 輸出人體 bounding box + 臉部關鍵點
  - ROI 負責「量哪裡？」→ 從關鍵點計算額頭/臉頰的皮膚區域
- ROI 品質直接決定後續訊號的訊雜比——ROI 錯了，後面演算法再好都救不了

使用 YOLO11n-pose 模型：
- 不使用 face 專用模型（無官方版本），而是用 pose 模型的臉部關鍵點
- COCO 17 個關鍵點中，前 5 個是臉部：鼻子、左右眼、左右耳
- 從這些關鍵點直接計算 ROI，比從估算的臉框推算更精準
"""

import cv2                      # OpenCV：影像處理與顯示
import time                     # 計時與 FPS 計算
import numpy as np              # 數值運算
from ultralytics import YOLO    # Ultralytics：YOLO11 模型推論框架


def open_camera(max_index=2):
    """嘗試開啟相機，支援 fallback 探測多個裝置索引。
    詳細說明見 stage1_camera_test.py。"""
    for index in range(max_index):
        print(f"[INFO] 嘗試開啟相機 index={index} (DirectShow)...")
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            print(f"[OK] 相機 index={index} 開啟成功")
            return cap
        cap.release()
        print(f"[WARN] 相機 index={index} 開啟失敗")

    print("[ERROR] 無法開啟任何相機")
    return None


def clamp_roi(x1, y1, x2, y2, width, height):
    """將 ROI 座標 clamp（裁切）至影像範圍內。

    為什麼需要 clamp？
    當人臉靠近畫面邊緣或轉頭時，計算出的 ROI 座標可能超出影像尺寸，
    直接用 frame[y1:y2, x1:x2] 會導致 IndexError 或取到錯誤的像素。
    clamp 確保所有座標都在 [0, width) / [0, height) 範圍內。
    """
    x1 = max(0, min(int(x1), width))
    y1 = max(0, min(int(y1), height))
    x2 = max(0, min(int(x2), width))
    y2 = max(0, min(int(y2), height))
    return x1, y1, x2, y2


def get_face_rois(keypoints, img_w, img_h):
    """從 YOLO pose 臉部關鍵點直接計算臉框與皮膚 ROI。

    COCO 關鍵點定義：
    0=nose（鼻子）, 1=left_eye（左眼）, 2=right_eye（右眼）,
    3=left_ear（左耳）, 4=right_ear（右耳）

    計算策略：以「眼距」(eye_dist) 作為臉部尺度的基準單位，
    所有 ROI 的大小和位置都是眼距的倍數，這樣不管人離鏡頭多遠，
    ROI 都能自動縮放到正確比例。

    Args:
        keypoints: (17, 3) ndarray，每個關鍵點的 [x, y, confidence]
        img_w, img_h: 影像寬高（用於 clamp）

    Returns:
        (face_box, forehead_roi, left_cheek, right_cheek) 四個 ROI 的座標，
        或 None（關鍵點不可見時）
    """
    # 取臉部前 5 個關鍵點
    kps = keypoints[:5]
    nose = kps[0]
    left_eye = kps[1]
    right_eye = kps[2]

    # 檢查關鍵點信心度：至少鼻子和雙眼必須可見（confidence > 0.5）
    # 如果人臉被遮擋或轉頭導致關鍵點消失，直接回傳 None
    if nose[2] < 0.5 or left_eye[2] < 0.5 or right_eye[2] < 0.5:
        return None

    # 提取座標
    nx, ny = nose[0], nose[1]        # 鼻子座標
    lex, ley = left_eye[0], left_eye[1]   # 左眼座標
    rex, rey = right_eye[0], right_eye[1]  # 右眼座標

    # 眼距 = 左右眼的水平距離，作為臉部尺度的基準
    # 最小值 20px 防止關鍵點重疊時除以零
    eye_dist = max(abs(lex - rex), 20)

    # ── 臉框：以鼻子為中心，用眼距估算臉部大小 ──
    face_half_w = eye_dist * 1.0           # 臉寬 ≈ 2 倍眼距
    face_top = ley - eye_dist * 0.7        # 上邊界：眉毛上方
    face_bot = ny + eye_dist * 0.9         # 下邊界：下巴
    face_box = clamp_roi(
        nx - face_half_w, face_top,
        nx + face_half_w, face_bot,
        img_w, img_h
    )

    # ── 額頭 ROI：雙眼上方的皮膚區域（避開頭髮和眉毛）──
    # 額頭是 rPPG 最佳量測區域：皮膚平坦、微血管豐富、不受說話/咀嚼影響
    eye_cy = (ley + rey) / 2  # 雙眼的中心 Y 座標
    forehead_roi = clamp_roi(
        rex + eye_dist * 0.1,         # 左邊界：右眼稍微內縮
        eye_cy - eye_dist * 0.7,      # 上邊界：眉毛上方（皮膚區）
        lex - eye_dist * 0.1,         # 右邊界：左眼稍微內縮
        eye_cy - eye_dist * 0.15,     # 下邊界：眉毛下緣
        img_w, img_h
    )

    # ── 臉頰 ROI：鼻子兩側的皮膚區域 ──
    # 臉頰是備選量測區域，避開眼睛與嘴唇
    cheek_top = ny - eye_dist * 0.2   # 上邊界：鼻子稍上方
    cheek_bot = ny + eye_dist * 0.4   # 下邊界：鼻子下方
    cheek_w = eye_dist * 0.35         # 臉頰寬度

    # 左臉頰（畫面上在左眼附近）
    left_cheek = clamp_roi(
        lex - cheek_w * 0.3, cheek_top,
        lex + cheek_w, cheek_bot,
        img_w, img_h
    )
    # 右臉頰（畫面上在右眼附近）
    right_cheek = clamp_roi(
        rex - cheek_w, cheek_top,
        rex + cheek_w * 0.3, cheek_bot,
        img_w, img_h
    )

    return face_box, forehead_roi, left_cheek, right_cheek


def get_best_person(results, img_w, img_h):
    """從 YOLO pose 結果中取面積最大的人，回傳 ROI 資訊與信心分數。

    為什麼取「面積最大」？
    畫面中可能有多人（如路人經過），面積最大的通常是正對鏡頭的受測者。
    """
    best_rois = None
    best_area = 0
    best_conf = 0

    for r in results:
        if r.boxes is None or r.keypoints is None:
            continue
        for i, box in enumerate(r.boxes):
            conf = float(box.conf[0])                    # 偵測信心分數
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # 人體 bounding box
            area = (x2 - x1) * (y2 - y1)                # 計算面積
            if area > best_area:
                # 取出該人的 17 個關鍵點 (17, 3)：[x, y, confidence]
                kps = r.keypoints[i].data[0].cpu().numpy()
                rois = get_face_rois(kps, img_w, img_h)
                if rois is not None:
                    best_area = area
                    best_rois = rois
                    best_conf = conf

    return best_rois, best_conf


def main():
    # ── A. 載入 YOLO11 pose 模型 ────────────────────────────
    # yolo11n-pose.pt = YOLO11 nano 版 pose 模型（最輕量，推論速度快）
    # 首次執行會自動從 GitHub 下載權重檔（~6MB）
    print("[INFO] 載入 YOLO11-pose 模型...")
    try:
        model = YOLO("yolo11n-pose.pt")
        print("[OK] 模型載入成功")
    except Exception as e:
        print(f"[ERROR] 模型載入失敗: {e}")
        return

    # ── B. 開啟相機 ─────────────────────────────────────────
    cap = open_camera()
    if cap is None:
        return

    prev_time = time.time()
    last_rois = None   # 上一幀的 ROI（人臉消失時保持顯示）
    last_conf = 0

    print("[INFO] 串流啟動，按 Q 退出")

    # ── C. 主迴圈 ───────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] 無法讀取影像幀，結束串流")
            break

        frame = cv2.flip(frame, 1)  # 鏡像翻轉
        h, w = frame.shape[:2]      # 取得影像尺寸

        # YOLO pose 推論：輸入一幀影像，輸出所有人的 bounding box + 關鍵點
        # verbose=False 關閉推論過程的日誌輸出
        results = model(frame, verbose=False)
        rois, conf = get_best_person(results, w, h)

        # 如果本幀偵測到人臉，更新 ROI；否則沿用上一幀的結果
        # 這樣人臉短暫消失（如快速轉頭）時不會閃爍
        if rois is not None:
            last_rois = rois
            last_conf = conf
        elif last_rois is not None:
            rois = last_rois
            conf = last_conf

        # 繪製偵測結果
        if rois is not None:
            face_box, forehead, left_cheek, right_cheek = rois

            # 藍色：人臉 bounding box + 信心分數
            cv2.rectangle(frame, (face_box[0], face_box[1]), (face_box[2], face_box[3]), (255, 0, 0), 2)
            cv2.putText(frame, f"Face {conf:.2f}", (face_box[0], face_box[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # 綠色：額頭 ROI（主要量測區域）
            cv2.rectangle(frame, (forehead[0], forehead[1]), (forehead[2], forehead[3]), (0, 255, 0), 2)
            cv2.putText(frame, "Forehead", (forehead[0], forehead[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # 黃色：左右臉頰 ROI（備選量測區域）
            cv2.rectangle(frame, (left_cheek[0], left_cheek[1]), (left_cheek[2], left_cheek[3]), (0, 255, 255), 2)
            cv2.rectangle(frame, (right_cheek[0], right_cheek[1]), (right_cheek[2], right_cheek[3]), (0, 255, 255), 2)

        # FPS 計算與顯示
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        cv2.imshow("rPPG - Stage 2 Face ROI", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] 使用者按下 Q，退出")
            break

    # ── D. 釋放資源 ─────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] 資源已釋放，程式結束")


if __name__ == "__main__":
    main()
