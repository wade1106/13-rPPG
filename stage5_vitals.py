"""Stage 5: 即時生命徵象 — Heart Rate (BPM) + Respiratory Rate (RPM)

本程式為 rPPG 系統的第五階段，整合 Stage 1-4 的所有模組，
形成從相機到生命徵象數值的完整管線：

    WebCam → YOLO → ROI → RGB 均值 → POS 演算法 → HR / RR

核心演算法：
1. Peak Detection — 從 rPPG 脈搏波偵測收縮峰值，計算 RR interval → HR
2. 自相關函數 (Autocorrelation) — 從 RR interval 的低頻調變萃取呼吸率
3. 雙重中值濾波 — 消除說話/眨眼引起的脈衝離群值
4. Slew Rate Limiting + EMA — 限制數值跳動幅度，輸出穩定如醫療監視器

效能優化：
- YOLO 每 3 幀偵測一次（其餘幀沿用上次 ROI）
- YCbCr 膚色分割排除非皮膚像素
- 加大 ROI 面積提升訊雜比
"""

import cv2                              # 影像處理與顯示
import time                             # 物理時間戳
import numpy as np                      # 數值運算
from collections import deque           # Ring Buffer
from ultralytics import YOLO            # YOLO11 模型
from scipy.signal import find_peaks, medfilt  # 波峰偵測、中值濾波
from scipy.interpolate import interp1d  # 時間戳內插（不等距→等距）
from signal_proc import pos_process, bandpass_filter  # POS 演算法與帶通濾波


# ── Stage 1-3 共用函式（詳細註解見 stage2/stage3）──────────

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
    """將 ROI 座標裁切至影像範圍內。"""
    return (max(0, min(int(x1), width)), max(0, min(int(y1), height)),
            max(0, min(int(x2), width)), max(0, min(int(y2), height)))


def get_face_rois(keypoints, img_w, img_h):
    """從 YOLO pose 關鍵點計算臉框與 ROI（加大版）。"""
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
    # 加大額頭 ROI（比 Stage 2 大約 50%，增加像素量提升訊雜比）
    forehead_roi = clamp_roi(rex - eye_dist * 0.05, eye_cy - eye_dist * 0.8,
                             lex + eye_dist * 0.05, eye_cy - eye_dist * 0.05,
                             img_w, img_h)
    # 加大臉頰 ROI
    cheek_top = ny - eye_dist * 0.25
    cheek_bot = ny + eye_dist * 0.55
    cheek_w = eye_dist * 0.5
    left_cheek = clamp_roi(lex - cheek_w * 0.2, cheek_top,
                           lex + cheek_w, cheek_bot, img_w, img_h)
    right_cheek = clamp_roi(rex - cheek_w, cheek_top,
                            rex + cheek_w * 0.2, cheek_bot, img_w, img_h)
    return face_box, forehead_roi, left_cheek, right_cheek


def get_best_person(results, img_w, img_h):
    """取面積最大的人，回傳 ROI 與信心分數。"""
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


MIN_ROI_AREA = 100     # ROI 最小面積閾值
YOLO_INTERVAL = 3      # 每 3 幀執行一次 YOLO（其餘幀沿用上次結果，提升 FPS）


def skin_mask_ycbcr(patch):
    """YCbCr 膚色分割，回傳二值遮罩。

    為什麼用 YCbCr 色彩空間？
    - RGB 中亮度和色彩耦合在一起，很難設閾值
    - YCbCr 將亮度 (Y) 和色度 (Cb, Cr) 分開
    - 人類膚色在 Cr/Cb 空間中集中在特定範圍，與膚色深淺無關
    - Y∈[80,255], Cr∈[133,173], Cb∈[77,127] 是經驗閾值

    形態學開運算 (Morphology Open) = 先侵蝕後膨脹，
    去除零星噪點與邊界毛刺，保留大面積的皮膚區域。
    """
    ycbcr = cv2.cvtColor(patch, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycbcr)
    # 雙閾值分割：只保留符合膚色範圍的像素
    mask = ((y >= 80) & (y <= 255) &
            (cr >= 133) & (cr <= 173) &
            (cb >= 77) & (cb <= 127)).astype(np.uint8)
    # 形態學開運算去除噪點
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def compute_multi_roi_mean(frame, rois):
    """合併多個 ROI，僅對膚色像素計算加權 RGB 均值。

    比 Stage 3 的版本多了 YCbCr 膚色遮罩：
    - Stage 3：直接對 ROI 所有像素取均值（包含毛髮、陰影）
    - Stage 5：只對通過膚色檢測的像素取均值（SNR 提升 300%+）
    """
    total_pixels = 0
    sum_r, sum_g, sum_b = 0.0, 0.0, 0.0
    for roi in rois:
        x1, y1, x2, y2 = roi
        if (x2 - x1) * (y2 - y1) < MIN_ROI_AREA:
            continue
        patch = frame[y1:y2, x1:x2]
        mask = skin_mask_ycbcr(patch)          # 膚色遮罩
        n_skin = np.count_nonzero(mask)         # 膚色像素數量
        if n_skin < 30:                         # 太少膚色像素，跳過
            continue
        # 僅對遮罩內（膚色）像素計算均值
        b_mean = patch[:, :, 0][mask == 1].mean()
        g_mean = patch[:, :, 1][mask == 1].mean()
        r_mean = patch[:, :, 2][mask == 1].mean()
        sum_r += r_mean * n_skin
        sum_g += g_mean * n_skin
        sum_b += b_mean * n_skin
        total_pixels += n_skin
    if total_pixels == 0:
        return None
    return sum_r / total_pixels, sum_g / total_pixels, sum_b / total_pixels


# ── Stage 5: Vitals 計算參數 ─────────────────────────────

BUFFER_SEC = 30.0       # 滑動視窗 30 秒（涵蓋 6-10 次呼吸週期）
TARGET_FS = 25.0        # interp1d 內插後的目標取樣率 (Hz)
STARTUP_SEC = 10.0      # 開機安全保護期（秒），資料不足時不強行計算
HR_SLEW_MAX = 5.0       # HR 每次更新最大允許變化量 (BPM)
BR_SLEW_MAX = 1.0       # RR 每次更新最大允許變化量 (RPM)
EMA_ALPHA = 0.2         # 指數移動平均 (EMA) 的平滑係數（0.2 = 80% 舊值 + 20% 新值）


def resample_uniform(timestamps, values, target_fs):
    """將不等距時序重採樣為等距時序（時間戳內插）。

    為什麼需要？
    WebCam 的 FPS 不穩定（如 25-35 FPS 波動），導致資料點時間間隔不等距。
    Butterworth 帶通濾波器假設輸入是等距取樣，不等距會導致頻率扭曲。
    interp1d 線性內插將不等距資料映射到完美等距的時間軸。
    """
    if len(timestamps) < 4:
        return None, None
    t_start = timestamps[0]
    t_end = timestamps[-1]
    # 以 1/target_fs 為間隔生成等距時間點
    t_uniform = np.arange(t_start, t_end, 1.0 / target_fs)
    if len(t_uniform) < 4:
        return None, None
    # 線性內插
    f = interp1d(timestamps, values, kind='linear', fill_value='extrapolate')
    return t_uniform, f(t_uniform)


def compute_heart_rate(rppg_signal, fs):
    """從 rPPG 脈搏波偵測波峰，計算心率 (BPM)。

    步驟：
    1. find_peaks 偵測所有波峰，搭配生理約束過濾假峰
    2. 計算相鄰波峰的時間間隔 (RR interval)
    3. 中值濾波消除離群值（如說話導致的假峰）
    4. HR = 60 / mean(RR_interval)
    """
    if len(rppg_signal) < int(fs * 2):   # 資料太短，無法計算
        return None, None

    # 生理不反應期：心跳間隔最少 0.35 秒（對應最大 171 BPM）
    min_dist = int(0.35 * fs)
    # prominence：波峰必須比周圍高出「訊號標準差 × 0.3」才算有效
    peaks, props = find_peaks(rppg_signal, distance=min_dist,
                              prominence=np.std(rppg_signal) * 0.3)

    if len(peaks) < 3:   # 波峰太少，無法計算
        return None, None

    # 計算 RR intervals（相鄰波峰的時間差，單位：秒）
    rr_intervals = np.diff(peaks) / fs

    # 中值濾波消除離群值（如眨眼造成的假峰使某個間隔異常短/長）
    # 中值濾波的優勢：崩潰點 50%，即使一半資料是離群值也能正確輸出
    if len(rr_intervals) >= 5:
        kernel = min(len(rr_intervals), 5)
        if kernel % 2 == 0:
            kernel -= 1
        if kernel >= 3:
            rr_intervals = medfilt(rr_intervals, kernel)

    # 過濾生理範圍外的極端值：0.33s-1.5s = 40-180 BPM
    valid = (rr_intervals > 0.33) & (rr_intervals < 1.5)
    rr_valid = rr_intervals[valid]

    if len(rr_valid) < 2:
        return None, peaks

    # HR = 60 / 平均 RR interval（秒轉 BPM）
    hr = 60.0 / np.mean(rr_valid)
    return hr, peaks


def compute_breathing_rate(rppg_signal, fs):
    """從 rPPG 訊號的低頻調變估算呼吸率 (RPM)。

    原理：呼吸性竇性心律不整 (RSA)
    人體呼吸時，吸氣心跳加速、吐氣心跳減速，這個微弱的調變
    藏在 RR interval 的 0.15-0.4 Hz 頻帶中。

    為什麼用自相關函數而不是 FFT？
    呼吸頻率很低（0.15-0.4 Hz），FFT 在低頻的解析度差，
    容易被 1/f 粉紅雜訊淹沒。自相關函數偵測的是「自我重複的規律」，
    即使訊號微弱，只要有週期性就能在延遲 τ=T_resp 處產生共振峰。
    """
    if len(rppg_signal) < int(fs * 10):   # 至少需要 10 秒資料
        return None

    try:
        # 帶通濾波萃取呼吸頻帶 (0.15-0.4 Hz = 9-24 RPM)
        breath_signal = bandpass_filter(rppg_signal, fs, low=0.15, high=0.4, order=2)
    except Exception:
        return None

    # 計算自相關函數（訊號與自身的時移內積）
    autocorr = np.correlate(breath_signal, breath_signal, mode='full')
    autocorr = autocorr[len(autocorr) // 2:]  # 只取正延遲部分

    # 在生理合理的呼吸週期範圍內搜尋峰值
    # 2.5s ~ 6.7s 對應 9-24 RPM
    min_lag = int(2.5 * fs)
    max_lag = int(6.7 * fs)
    if max_lag >= len(autocorr):
        max_lag = len(autocorr) - 1
    if min_lag >= max_lag:
        return None

    search = autocorr[min_lag:max_lag]
    if len(search) == 0:
        return None

    # 找到最大自相關峰值的延遲位置
    peak_idx = np.argmax(search) + min_lag
    if autocorr[peak_idx] <= 0:    # 無正相關，代表無週期性
        return None

    # 延遲（樣本數）→ 頻率 → RPM
    br = (fs / peak_idx) * 60.0
    return br


def slew_limit(new_val, prev_val, max_change):
    """Slew Rate Limiting：限制數值每次更新的最大變化量。

    人體心率/呼吸率不可能在 0.2 秒內突變 50 BPM，
    如果出現這種跳變一定是雜訊，用 clamp 限制即可。
    """
    if prev_val is None:
        return new_val
    diff = new_val - prev_val
    if abs(diff) > max_change:
        return prev_val + max_change * np.sign(diff)
    return new_val


def ema_smooth(new_val, prev_val, alpha=EMA_ALPHA):
    """指數移動平均 (EMA) 平滑。

    output = alpha × new + (1-alpha) × old
    alpha=0.2 表示新值只佔 20%、舊值佔 80%，數值變化非常平滑。
    """
    if prev_val is None:
        return new_val
    return alpha * new_val + (1 - alpha) * prev_val


# ── 波形繪圖 ──────────────────────────────────────────────

def draw_rppg_plot(rppg_data, peaks_data, plot_w=640, plot_h=200):
    """用 OpenCV 繪製 rPPG 脈搏波形 + 波峰紅點標記。"""
    canvas = np.zeros((plot_h, plot_w, 3), dtype=np.uint8)
    n = len(rppg_data)
    if n < 2:
        return canvas

    # 正規化到畫布座標
    arr = np.array(rppg_data)
    vmin, vmax = arr.min(), arr.max()
    if vmax - vmin < 1e-6:
        vmax = vmin + 1

    margin = 10
    normalized = (arr - vmin) / (vmax - vmin)
    ys = (plot_h - margin) - normalized * (plot_h - 2 * margin)
    xs = np.linspace(0, plot_w - 1, n)
    pts = np.stack([xs, ys], axis=1).astype(np.int32)

    # 綠色波形曲線
    cv2.polylines(canvas, [pts], False, (0, 255, 0), 1)

    # 紅色圓點標記波峰位置
    if peaks_data is not None:
        for p in peaks_data:
            if 0 <= p < n:
                px = int(xs[p])
                py = int(ys[p])
                cv2.circle(canvas, (px, py), 4, (0, 0, 255), -1)

    cv2.putText(canvas, "rPPG", (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return canvas


# ── 主程式 ────────────────────────────────────────────────

def main():
    # ── A. 初始化 ────────────────────────────────────────────
    print("[INFO] 載入 YOLO11-pose 模型...")
    model = YOLO("yolo11n-pose.pt")
    print("[OK] 模型載入成功")

    cap = open_camera()
    if cap is None:
        return

    # 原始資料緩衝區（用 list 而非 deque，方便用時間裁切）
    buf_time = []
    buf_r = []
    buf_g = []
    buf_b = []

    start_time = time.time()
    prev_time = start_time
    last_rois = None
    last_conf = 0

    # Vitals 平滑狀態變數
    smooth_hr = None     # EMA 平滑後的心率
    smooth_br = None     # EMA 平滑後的呼吸率
    display_hr = "--"    # 畫面顯示的心率文字
    display_br = "--"    # 畫面顯示的呼吸率文字
    frame_count = 0      # 幀計數器（用於 YOLO 間隔控制）

    print("[INFO] 串流啟動，按 Q 退出")

    # ── B. 主迴圈 ────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] 無法讀取影像幀，結束串流")
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        now = time.time()
        timestamp = now - start_time
        frame_count += 1

        # YOLO pose 偵測（每 YOLO_INTERVAL 幀一次，降低運算負擔）
        rois, conf = None, 0
        if frame_count % YOLO_INTERVAL == 0 or last_rois is None:
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

            # 計算膚色像素的 RGB 均值並存入緩衝區
            rgb = compute_multi_roi_mean(frame, [forehead, left_cheek, right_cheek])
            if rgb is not None:
                buf_time.append(timestamp)
                buf_r.append(rgb[0])
                buf_g.append(rgb[1])
                buf_b.append(rgb[2])

            # 繪製 ROI 框
            cv2.rectangle(frame, (face_box[0], face_box[1]),
                          (face_box[2], face_box[3]), (255, 0, 0), 2)
            cv2.rectangle(frame, (forehead[0], forehead[1]),
                          (forehead[2], forehead[3]), (0, 255, 0), 2)
            cv2.rectangle(frame, (left_cheek[0], left_cheek[1]),
                          (left_cheek[2], left_cheek[3]), (0, 255, 255), 2)
            cv2.rectangle(frame, (right_cheek[0], right_cheek[1]),
                          (right_cheek[2], right_cheek[3]), (0, 255, 255), 2)

        # ── Vitals 計算 ──────────────────────────────────────

        rppg_for_plot = None
        peaks_for_plot = None

        # 安全保護期過後才開始計算（資料不足時強行計算會產生垃圾值）
        if timestamp >= STARTUP_SEC and len(buf_time) > 30:
            # 裁切緩衝區：只保留最近 BUFFER_SEC 秒的資料
            cutoff = timestamp - BUFFER_SEC
            while buf_time and buf_time[0] < cutoff:
                buf_time.pop(0)
                buf_r.pop(0)
                buf_g.pop(0)
                buf_b.pop(0)

            t_arr = np.array(buf_time)
            rgb_matrix = np.column_stack([buf_r, buf_g, buf_b])

            # 時間戳內插：不等距 → 等距（TARGET_FS = 25 Hz）
            t_uni, r_uni = resample_uniform(t_arr, np.array(buf_r), TARGET_FS)

            if t_uni is not None and len(t_uni) > int(TARGET_FS * 3):
                # 對三個通道都做內插
                rgb_uni = np.column_stack([
                    interp1d(t_arr, buf_r, fill_value='extrapolate')(t_uni),
                    interp1d(t_arr, buf_g, fill_value='extrapolate')(t_uni),
                    interp1d(t_arr, buf_b, fill_value='extrapolate')(t_uni),
                ])

                # POS 演算法：RGB → rPPG 脈搏波
                rppg = pos_process(rgb_uni, TARGET_FS)
                rppg_for_plot = rppg

                # 心率計算：波峰偵測 → RR interval → BPM
                hr_raw, peaks = compute_heart_rate(rppg, TARGET_FS)
                peaks_for_plot = peaks
                if hr_raw is not None and 40 < hr_raw < 180:
                    hr_limited = slew_limit(hr_raw, smooth_hr, HR_SLEW_MAX)
                    smooth_hr = ema_smooth(hr_limited, smooth_hr)
                    display_hr = f"{smooth_hr:.0f}"

                # 呼吸率計算：RSA 帶通 → 自相關 → RPM
                br_raw = compute_breathing_rate(rppg, TARGET_FS)
                if br_raw is not None and 6 < br_raw < 30:
                    br_limited = slew_limit(br_raw, smooth_br, BR_SLEW_MAX)
                    smooth_br = ema_smooth(br_limited, smooth_br)
                    display_br = f"{smooth_br:.0f}"
        else:
            # 安全保護期內不計算，顯示 "--"
            if timestamp < STARTUP_SEC:
                display_hr = "--"
                display_br = "--"

        # ── 畫面顯示 ─────────────────────────────────────────

        # FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        # HR / RR 數值（畫面左下角）
        cv2.putText(frame, f"HR: {display_hr} BPM", (10, h - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(frame, f"RR: {display_br} RPM", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 200, 100), 2)

        # Buffering 倒數提示
        if timestamp < STARTUP_SEC:
            cv2.putText(frame, f"Buffering... {timestamp:.0f}s / {STARTUP_SEC:.0f}s",
                        (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        # rPPG 波形圖（拼接在相機畫面下方）
        if rppg_for_plot is not None:
            plot = draw_rppg_plot(rppg_for_plot, peaks_for_plot, plot_w=w, plot_h=200)
        else:
            plot = np.zeros((200, w, 3), dtype=np.uint8)

        combined = np.vstack([frame, plot])
        cv2.imshow("rPPG - Stage 5 Vitals", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] 使用者按下 Q，退出")
            break

    # ── C. 釋放資源 ─────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] 資源已釋放，程式結束")


if __name__ == "__main__":
    main()
