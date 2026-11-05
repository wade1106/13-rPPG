"""Stage 2: YOLO11 人臉偵測 + Skin ROI 萃取
- YOLO11-pose 透過臉部關鍵點直接定位皮膚 ROI
- 額頭 ROI (主要) + 臉頰 ROI (備選)
- ROI 邊界 clamp 防呆
- 按 Q 安全退出
"""

import cv2
import time
import numpy as np
from ultralytics import YOLO


def open_camera(max_index=2):
    """嘗試開啟相機，支援 fallback 探測多個裝置索引。"""
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
    """將 ROI 座標 clamp 至影像範圍內。"""
    x1 = max(0, min(int(x1), width))
    y1 = max(0, min(int(y1), height))
    x2 = max(0, min(int(x2), width))
    y2 = max(0, min(int(y2), height))
    return x1, y1, x2, y2


def get_face_rois(keypoints, img_w, img_h):
    """從 YOLO pose 臉部關鍵點直接計算臉框、額頭 ROI、臉頰 ROI。

    COCO keypoints: 0=nose, 1=left_eye, 2=right_eye, 3=left_ear, 4=right_ear
    回傳 (face_box, forehead_roi, left_cheek, right_cheek) 或 None
    """
    # 取臉部 5 個關鍵點
    kps = keypoints[:5]
    nose = kps[0]
    left_eye = kps[1]
    right_eye = kps[2]

    # 至少需要鼻子和雙眼可見
    if nose[2] < 0.5 or left_eye[2] < 0.5 or right_eye[2] < 0.5:
        return None

    nx, ny = nose[0], nose[1]
    lex, ley = left_eye[0], left_eye[1]
    rex, rey = right_eye[0], right_eye[1]

    # 眼距作為臉部尺度基準
    eye_dist = max(abs(lex - rex), 20)

    # --- 臉框：以鼻子為中心，用眼距估算 ---
    face_half_w = eye_dist * 1.0
    face_top = ley - eye_dist * 0.7   # 眉毛上方一點
    face_bot = ny + eye_dist * 0.9    # 下巴
    face_box = clamp_roi(
        nx - face_half_w, face_top,
        nx + face_half_w, face_bot,
        img_w, img_h
    )

    # --- 額頭 ROI：雙眼上方到眉毛區域 ---
    eye_cy = (ley + rey) / 2
    forehead_roi = clamp_roi(
        rex + eye_dist * 0.1,         # 內縮一點
        eye_cy - eye_dist * 0.7,      # 眉毛上方
        lex - eye_dist * 0.1,
        eye_cy - eye_dist * 0.15,     # 眉毛下緣
        img_w, img_h
    )

    # --- 臉頰 ROI：鼻子兩側 ---
    cheek_top = ny - eye_dist * 0.2
    cheek_bot = ny + eye_dist * 0.4
    cheek_w = eye_dist * 0.35

    left_cheek = clamp_roi(
        lex - cheek_w * 0.3, cheek_top,
        lex + cheek_w, cheek_bot,
        img_w, img_h
    )
    right_cheek = clamp_roi(
        rex - cheek_w, cheek_top,
        rex + cheek_w * 0.3, cheek_bot,
        img_w, img_h
    )

    return face_box, forehead_roi, left_cheek, right_cheek


def get_best_person(results, img_w, img_h):
    """從 YOLO pose 結果中取面積最大的人，回傳 ROI 資訊與信心分數。"""
    best_rois = None
    best_area = 0
    best_conf = 0

    for r in results:
        if r.boxes is None or r.keypoints is None:
            continue
        for i, box in enumerate(r.boxes):
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                kps = r.keypoints[i].data[0].cpu().numpy()  # (17, 3)
                rois = get_face_rois(kps, img_w, img_h)
                if rois is not None:
                    best_area = area
                    best_rois = rois
                    best_conf = conf

    return best_rois, best_conf


def main():
    print("[INFO] 載入 YOLO11-pose 模型...")
    try:
        model = YOLO("yolo11n-pose.pt")
        print("[OK] 模型載入成功")
    except Exception as e:
        print(f"[ERROR] 模型載入失敗: {e}")
        return

    cap = open_camera()
    if cap is None:
        return

    prev_time = time.time()
    last_rois = None
    last_conf = 0

    print("[INFO] 串流啟動，按 Q 退出")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] 無法讀取影像幀，結束串流")
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # YOLO pose 偵測
        results = model(frame, verbose=False)
        rois, conf = get_best_person(results, w, h)

        if rois is not None:
            last_rois = rois
            last_conf = conf
        elif last_rois is not None:
            rois = last_rois
            conf = last_conf

        if rois is not None:
            face_box, forehead, left_cheek, right_cheek = rois

            # 人臉框 (藍色)
            cv2.rectangle(frame, (face_box[0], face_box[1]), (face_box[2], face_box[3]), (255, 0, 0), 2)
            cv2.putText(frame, f"Face {conf:.2f}", (face_box[0], face_box[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # 額頭 ROI (綠色)
            cv2.rectangle(frame, (forehead[0], forehead[1]), (forehead[2], forehead[3]), (0, 255, 0), 2)
            cv2.putText(frame, "Forehead", (forehead[0], forehead[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # 臉頰 ROI (黃色)
            cv2.rectangle(frame, (left_cheek[0], left_cheek[1]), (left_cheek[2], left_cheek[3]), (0, 255, 255), 2)
            cv2.rectangle(frame, (right_cheek[0], right_cheek[1]), (right_cheek[2], right_cheek[3]), (0, 255, 255), 2)

        # FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        cv2.imshow("rPPG - Stage 2 Face ROI", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] 使用者按下 Q，退出")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] 資源已釋放，程式結束")


if __name__ == "__main__":
    main()
