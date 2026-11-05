"""Stage 3: ROI 像素 RGB 均值萃取與即時波形可視化

本程式為 rPPG 系統的第三階段，負責將 Stage 2 的 Skin ROI（影像區域）
「降維」為每幀一筆 (R, G, B) 數據點，累積成時間序列。

這是從「影像處理」跨入「訊號處理」的關鍵轉換：
- 每幀只做一件事：把 ROI 內的所有像素壓縮成一組均值 (mean R, mean G, mean B)
- 累積到 Ring Buffer 後就形成 RGB(t) 時間序列，可以開始做訊號分析
- 同時記錄物理時間戳，為 Stage 4 的時間戳內插校正打基礎

即時波形可視化：
- 用 OpenCV 在畫面下方繪製 R/G/B 三通道波形
- 安靜面對鏡頭時，Green 通道應可觀察到微弱的脈搏光學波動
"""

import cv2                      # OpenCV：影像處理、繪圖、顯示
import time                     # 物理時間戳記錄
import numpy as np              # 數值運算
from collections import deque   # deque：固定長度的雙端佇列，自動丟棄舊資料
from ultralytics import YOLO    # YOLO11 模型


# ── Stage 1 & 2 共用函式（相機、ROI 計算）─────────────────
# 這些函式與 stage2_face_roi.py 相同，詳細註解請參考該檔案

def open_camera(max_index=2):
    """開啟相機，支援 fallback。"""
    for index in range(max_index):
        print(f"[INFO] 嘗試開啟相機 index={index} (DirectShow)...")
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            print(f"[OK] 相機 index={index} 開啟成功")
            return cap
        cap.release()
    print("[ERROR] 無法開啟任何相機")
    return None


def clamp_roi(x1, y1, x2, y2, width, height):
    """將 ROI 座標裁切至影像範圍內，防止 IndexError。"""
    return (max(0, min(int(x1), width)), max(0, min(int(y1), height)),
            max(0, min(int(x2), width)), max(0, min(int(y2), height)))


def get_face_rois(keypoints, img_w, img_h):
    """從 YOLO pose 關鍵點計算臉框、額頭 ROI、臉頰 ROI。"""
    kps = keypoints[:5]
    nose, left_eye, right_eye = kps[0], kps[1], kps[2]
    if nose[2] < 0.5 or left_eye[2] < 0.5 or right_eye[2] < 0.5:
        return None

    nx, ny = nose[0], nose[1]
    lex, ley = left_eye[0], left_eye[1]
    rex, rey = right_eye[0], right_eye[1]
    eye_dist = max(abs(lex - rex), 20)

    face_half_w = eye_dist * 1.0
    face_box = clamp_roi(nx - face_half_w, ley - eye_dist * 0.7,
                         nx + face_half_w, ny + eye_dist * 0.9, img_w, img_h)

    eye_cy = (ley + rey) / 2
    forehead_roi = clamp_roi(rex + eye_dist * 0.1, eye_cy - eye_dist * 0.7,
                             lex - eye_dist * 0.1, eye_cy - eye_dist * 0.15,
                             img_w, img_h)

    cheek_top = ny - eye_dist * 0.2
    cheek_bot = ny + eye_dist * 0.4
    cheek_w = eye_dist * 0.35
    left_cheek = clamp_roi(lex - cheek_w * 0.3, cheek_top,
                           lex + cheek_w, cheek_bot, img_w, img_h)
    right_cheek = clamp_roi(rex - cheek_w, cheek_top,
                            rex + cheek_w * 0.3, cheek_bot, img_w, img_h)

    return face_box, forehead_roi, left_cheek, right_cheek


def get_best_person(results, img_w, img_h):
    """取面積最大的人，回傳 ROI 資訊與信心分數。"""
    best_rois, best_area, best_conf = None, 0, 0
    for r in results:
        if r.boxes is None or r.keypoints is None:
            continue
        for i, box in enumerate(r.boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                kps = r.keypoints[i].data[0].cpu().numpy()
                rois = get_face_rois(kps, img_w, img_h)
                if rois is not None:
                    best_area = area
                    best_rois = rois
                    best_conf = float(box.conf[0])
    return best_rois, best_conf


# ── Stage 3 新增功能 ──────────────────────────────────────

MIN_ROI_AREA = 100  # ROI 最小面積閾值（像素），低於此值跳過（避免均值不穩定）
BUFFER_LEN = 300    # Ring Buffer 最大長度，約 10 秒 @ 30 FPS


def compute_multi_roi_mean(frame, rois):
    """合併多個 ROI 區域，以面積加權計算 RGB 均值。

    為什麼合併多個 ROI？
    額頭 + 左右臉頰合計的像素量比單一額頭大 2-3 倍，
    更多像素 → 均值更穩定 → 訊雜比 (SNR) 更高。

    面積加權的原因：大 ROI 的像素更多，對均值的影響也應更大。

    Args:
        frame: BGR 影像
        rois: ROI 座標列表 [(x1,y1,x2,y2), ...]

    Returns:
        (mean_R, mean_G, mean_B) 或 None（所有 ROI 都太小時）
    """
    total_pixels = 0
    sum_r, sum_g, sum_b = 0.0, 0.0, 0.0

    for roi in rois:
        x1, y1, x2, y2 = roi
        area = (x2 - x1) * (y2 - y1)
        if area < MIN_ROI_AREA:   # 面積太小，跳過
            continue
        patch = frame[y1:y2, x1:x2]       # 擷取 ROI 區域的像素
        b, g, r = cv2.mean(patch)[:3]      # OpenCV 預設是 BGR 順序
        sum_r += r * area                  # 用面積加權累加
        sum_g += g * area
        sum_b += b * area
        total_pixels += area

    if total_pixels == 0:
        return None
    # 除以總面積 → 面積加權平均
    return sum_r / total_pixels, sum_g / total_pixels, sum_b / total_pixels


def draw_rgb_plot(buf_r, buf_g, buf_b, plot_w=640, plot_h=200):
    """用 OpenCV 繪製 RGB 三通道即時波形圖。

    在黑色畫布上以紅/綠/藍三色分別繪製 R/G/B 通道的時間序列曲線。
    三條曲線共用 Y 軸範圍，方便比較相對變化。

    Args:
        buf_r, buf_g, buf_b: R/G/B 通道的 deque 緩衝區
        plot_w, plot_h: 畫布寬高（像素）

    Returns:
        (plot_h, plot_w, 3) 的 BGR 影像（可直接拼接到相機畫面下方）
    """
    canvas = np.zeros((plot_h, plot_w, 3), dtype=np.uint8)  # 黑色背景
    n = len(buf_r)
    if n < 2:   # 資料太少無法畫線
        return canvas

    def to_points(data, color_range=None):
        """將數值資料轉換為畫布上的 (x, y) 座標點序列。"""
        arr = np.array(data)
        if color_range is None:
            vmin, vmax = arr.min(), arr.max()
        else:
            vmin, vmax = color_range
        if vmax - vmin < 1e-6:   # 防止除以零（所有值相同時）
            vmax = vmin + 1
        margin = 10   # 上下留白
        # 正規化到 [0, 1]，再映射到 [margin, plot_h - margin]
        normalized = (arr - vmin) / (vmax - vmin)
        ys = (plot_h - margin) - normalized * (plot_h - 2 * margin)  # Y 軸反轉（畫布 Y 向下）
        xs = np.linspace(0, plot_w - 1, n)   # 等距分配 X 座標
        pts = np.stack([xs, ys], axis=1).astype(np.int32)
        return pts

    # 計算三通道合併的 Y 軸範圍，讓三條曲線在同一尺度
    all_vals = np.concatenate([np.array(buf_r), np.array(buf_g), np.array(buf_b)])
    vmin, vmax = all_vals.min(), all_vals.max()
    color_range = (vmin, vmax)

    # 繪製三條曲線（OpenCV 顏色是 BGR 順序）
    pts_r = to_points(buf_r, color_range)
    pts_g = to_points(buf_g, color_range)
    pts_b = to_points(buf_b, color_range)

    cv2.polylines(canvas, [pts_r], False, (0, 0, 255), 1)   # 紅色 = R 通道
    cv2.polylines(canvas, [pts_g], False, (0, 255, 0), 1)   # 綠色 = G 通道
    cv2.polylines(canvas, [pts_b], False, (255, 0, 0), 1)   # 藍色 = B 通道

    # 左上角圖例
    cv2.putText(canvas, "R", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(canvas, "G", (25, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(canvas, "B", (45, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    return canvas


def main():
    # ── A. 載入模型與相機 ────────────────────────────────────
    print("[INFO] 載入 YOLO11-pose 模型...")
    model = YOLO("yolo11n-pose.pt")
    print("[OK] 模型載入成功")

    cap = open_camera()
    if cap is None:
        return

    # ── B. 初始化 Ring Buffer ────────────────────────────────
    # deque(maxlen=N) 是固定長度的佇列，滿了自動丟棄最舊的資料
    # 分別儲存：時間戳、R 均值、G 均值、B 均值
    buf_time = deque(maxlen=BUFFER_LEN)
    buf_r = deque(maxlen=BUFFER_LEN)
    buf_g = deque(maxlen=BUFFER_LEN)
    buf_b = deque(maxlen=BUFFER_LEN)

    start_time = time.time()   # 記錄程式啟動的絕對時間（用於計算物理時間戳）
    prev_time = start_time
    last_rois = None
    last_conf = 0

    print("[INFO] 串流啟動，按 Q 退出")

    # ── C. 主迴圈 ───────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] 無法讀取影像幀，結束串流")
            break

        frame = cv2.flip(frame, 1)      # 鏡像翻轉
        h, w = frame.shape[:2]
        # 物理時間戳 = 當前時間 - 程式啟動時間（秒）
        # 不用幀編號當時間軸，因為 WebCam FPS 不穩定
        timestamp = time.time() - start_time

        # YOLO pose 偵測 → 取得 ROI
        results = model(frame, verbose=False)
        rois, conf = get_best_person(results, w, h)

        # 人臉消失時沿用上一幀的 ROI
        if rois is not None:
            last_rois = rois
            last_conf = conf
        elif last_rois is not None:
            rois = last_rois
            conf = last_conf

        if rois is not None:
            face_box, forehead, left_cheek, right_cheek = rois

            # ── 核心：計算 ROI 的 RGB 均值並存入緩衝區 ──
            # 合併額頭 + 左右臉頰，以面積加權計算均值
            rgb = compute_multi_roi_mean(frame, [forehead, left_cheek, right_cheek])
            if rgb is not None:
                buf_time.append(timestamp)   # 記錄物理時間戳
                buf_r.append(rgb[0])         # R 通道均值
                buf_g.append(rgb[1])         # G 通道均值
                buf_b.append(rgb[2])         # B 通道均值

            # 繪製 ROI 框（視覺驗證用）
            cv2.rectangle(frame, (face_box[0], face_box[1]),
                          (face_box[2], face_box[3]), (255, 0, 0), 2)       # 藍色臉框
            cv2.rectangle(frame, (forehead[0], forehead[1]),
                          (forehead[2], forehead[3]), (0, 255, 0), 2)       # 綠色額頭
            cv2.putText(frame, "Forehead", (forehead[0], forehead[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.rectangle(frame, (left_cheek[0], left_cheek[1]),
                          (left_cheek[2], left_cheek[3]), (0, 255, 255), 2)  # 黃色左臉頰
            cv2.rectangle(frame, (right_cheek[0], right_cheek[1]),
                          (right_cheek[2], right_cheek[3]), (0, 255, 255), 2) # 黃色右臉頰

        # FPS 計算與顯示
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        # 繪製 RGB 即時波形圖（寬度與相機畫面相同）
        plot = draw_rgb_plot(buf_r, buf_g, buf_b, plot_w=w, plot_h=200)

        # 上下拼接：相機畫面（上）+ 波形圖（下）
        combined = np.vstack([frame, plot])
        cv2.imshow("rPPG - Stage 3 RGB Extract", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] 使用者按下 Q，退出")
            break

    # ── D. 釋放資源 ─────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    # 輸出收集統計
    print(f"[INFO] 共收集 {len(buf_time)} 筆資料，"
          f"時間範圍 {buf_time[0]:.1f}s ~ {buf_time[-1]:.1f}s" if buf_time else "")
    print("[INFO] 資源已釋放，程式結束")


if __name__ == "__main__":
    main()
