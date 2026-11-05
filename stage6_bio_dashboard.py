"""Stage 6: PyQt6 即時生醫監測儀表板

本程式為 rPPG 系統的最終整合階段，將 Stage 1-5 的所有模組
組裝成一個專業的桌面監測應用程式。

架構設計：
- 使用 PyQt6 建立 GUI（不使用 cv2.imshow，避免衝突）
- QThread 工作線程執行運算密集的管線（相機→YOLO→POS→HR/RR），UI 不凍結
- Signal/Slot 機制在工作線程與 UI 主線程之間安全通訊
- 運算與顯示頻率解耦：波形即時更新，數值 1 Hz 定拍刷新

介面配置：
- 左側：相機畫面 + ROI 框疊加
- 右上：FPS、Heart Rate (BPM)、Respiratory Rate (RPM)、執行時間
- 右下：pyqtgraph 即時脈搏波示波器（30 秒滾動視窗）
- 底部：Start / Stop 按鈕

效能優化（繼承自 Stage 5）：
- YOLO 每 3 幀偵測一次
- Vitals 每 5 幀計算一次
- YCbCr 膚色分割 + 加大 ROI
- HR 使用 FFT 頻譜法（比 peak detection 更抗噪）
"""

import sys                              # 系統退出碼
import time                             # 物理時間戳
import numpy as np                      # 數值運算
from collections import deque           # Ring Buffer（此檔案未直接使用，但保留 import）

import cv2                              # OpenCV：影像處理
from ultralytics import YOLO            # YOLO11 模型
from scipy.interpolate import interp1d  # 時間戳內插
from scipy.signal import find_peaks     # 波峰偵測（保留備用）

# PyQt6：GUI 框架
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer  # 核心：線程、信號、計時器
from PyQt6.QtGui import QImage, QPixmap, QFont             # 影像轉換、字型
from PyQt6.QtWidgets import (                               # UI 元件
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QHBoxLayout, QVBoxLayout,
)
import pyqtgraph as pg                  # pyqtgraph：高效能即時波形繪圖

from signal_proc import pos_process, bandpass_filter  # 自製 POS 演算法與帶通濾波


# ══════════════════════════════════════════════════════════
# 共用函式（來自 Stage 2-5，詳細註解見各 stage 檔案）
# ══════════════════════════════════════════════════════════

def open_camera(max_index=2):
    """開啟相機，支援 DirectShow fallback。"""
    for index in range(max_index):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            return cap
        cap.release()
    return None


def clamp_roi(x1, y1, x2, y2, w, h):
    """將 ROI 座標裁切至影像範圍內，防止 IndexError。"""
    return (max(0, min(int(x1), w)), max(0, min(int(y1), h)),
            max(0, min(int(x2), w)), max(0, min(int(y2), h)))


def get_face_rois(kps, img_w, img_h):
    """從 YOLO pose 關鍵點計算臉框 + 額頭 ROI + 臉頰 ROI（加大版）。"""
    nose, leye, reye = kps[0], kps[1], kps[2]
    if nose[2] < 0.5 or leye[2] < 0.5 or reye[2] < 0.5:
        return None
    nx, ny = nose[0], nose[1]
    lx, ly = leye[0], leye[1]
    rx, ry = reye[0], reye[1]
    ed = max(abs(lx - rx), 20)  # 眼距 = 臉部尺度基準

    face = clamp_roi(nx - ed, ly - ed * 0.7, nx + ed, ny + ed * 0.9, img_w, img_h)
    ec = (ly + ry) / 2  # 雙眼中心 Y 座標
    forehead = clamp_roi(rx - ed * 0.05, ec - ed * 0.8,
                         lx + ed * 0.05, ec - ed * 0.05, img_w, img_h)
    ct, cb = ny - ed * 0.25, ny + ed * 0.55
    cw = ed * 0.5
    left_c = clamp_roi(lx - cw * 0.2, ct, lx + cw, cb, img_w, img_h)
    right_c = clamp_roi(rx - cw, ct, rx + cw * 0.2, cb, img_w, img_h)
    return face, forehead, left_c, right_c


def get_best_person(results, img_w, img_h):
    """取面積最大的人，回傳 ROI 資訊。"""
    best, area, conf = None, 0, 0
    for r in results:
        if r.boxes is None or r.keypoints is None:
            continue
        for i, box in enumerate(r.boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            a = (x2 - x1) * (y2 - y1)
            if a > area:
                rois = get_face_rois(r.keypoints[i].data[0].cpu().numpy(), img_w, img_h)
                if rois:
                    best, area, conf = rois, a, float(box.conf[0])
    return best, conf


def skin_mask_ycbcr(patch):
    """YCbCr 膚色分割：只保留皮膚像素，排除毛髮/眼睛/背景。"""
    ycbcr = cv2.cvtColor(patch, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycbcr)
    mask = ((y >= 80) & (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # 形態學開運算去噪


def compute_multi_roi_mean(frame, rois):
    """合併多個 ROI，僅對膚色像素計算加權 RGB 均值。"""
    total, sr, sg, sb = 0, 0.0, 0.0, 0.0
    for x1, y1, x2, y2 in rois:
        if (x2 - x1) * (y2 - y1) < 100:  # 面積太小跳過
            continue
        p = frame[y1:y2, x1:x2]
        m = skin_mask_ycbcr(p)
        n = np.count_nonzero(m)
        if n < 30:  # 膚色像素太少跳過
            continue
        sr += p[:, :, 2][m == 1].mean() * n  # R (OpenCV 的 channel 2)
        sg += p[:, :, 1][m == 1].mean() * n  # G (channel 1)
        sb += p[:, :, 0][m == 1].mean() * n  # B (channel 0)
        total += n
    return (sr / total, sg / total, sb / total) if total > 0 else None


def compute_heart_rate(rppg, fs):
    """FFT 頻譜法估算心率（比 peak detection 更抗噪）。

    原理：對 rPPG 訊號做 FFT，在 0.7-3.0 Hz（42-180 BPM）範圍內
    找到能量最大的頻率，乘以 60 轉為 BPM。
    """
    if len(rppg) < int(fs * 3):  # 至少 3 秒資料
        return None
    freqs = np.fft.rfftfreq(len(rppg), d=1.0 / fs)  # 頻率軸
    fft_mag = np.abs(np.fft.rfft(rppg))               # FFT 振幅譜
    # 只在生理範圍內搜尋
    mask = (freqs >= 0.7) & (freqs <= 3.0)
    if not np.any(mask):
        return None
    peak_freq = freqs[mask][np.argmax(fft_mag[mask])]  # 最大能量的頻率
    return peak_freq * 60.0  # Hz → BPM


def compute_breathing_rate(rppg, fs):
    """自相關函數法估算呼吸率。詳細原理見 stage5_vitals.py。"""
    if len(rppg) < int(fs * 10):
        return None
    try:
        bs = bandpass_filter(rppg, fs, low=0.15, high=0.4, order=2)
    except Exception:
        return None
    ac = np.correlate(bs, bs, mode='full')
    ac = ac[len(ac) // 2:]  # 只取正延遲
    lo, hi = int(2.5 * fs), min(int(6.7 * fs), len(ac) - 1)
    if lo >= hi:
        return None
    idx = np.argmax(ac[lo:hi]) + lo
    return (fs / idx) * 60.0 if ac[idx] > 0 else None


# ══════════════════════════════════════════════════════════
# QThread 工作線程：在背景執行完整的 rPPG 管線
# ══════════════════════════════════════════════════════════

YOLO_INTERVAL = 3    # 每 3 幀執行一次 YOLO 偵測（提升 FPS）
TARGET_FS = 25.0     # interp1d 內插後的目標取樣率 (Hz)
BUFFER_SEC = 30.0    # 滑動視窗 30 秒
STARTUP_SEC = 10.0   # 開機安全保護期 10 秒


class SignalWorker(QThread):
    """背景工作線程：相機擷取 → YOLO → ROI → POS → HR/RR。

    使用 pyqtSignal 將結果傳回 UI 主線程：
    - frame_ready: 帶有 ROI 標記的相機影像（QImage）
    - vitals_ready: HR/RR/FPS/波形資料（dict）
    - error_signal: 錯誤訊息（str）
    """
    frame_ready = pyqtSignal(QImage)    # 影像幀就緒
    vitals_ready = pyqtSignal(dict)     # 生命徵象數據就緒
    error_signal = pyqtSignal(str)      # 錯誤通知

    def __init__(self):
        super().__init__()
        self.running = False  # 控制迴圈的旗標

    def stop(self):
        """設定停止旗標，讓 run() 迴圈在下一次迭代結束。"""
        self.running = False

    def run(self):
        """工作線程主體（在獨立線程中執行）。"""
        self.running = True

        # 載入 YOLO 模型
        try:
            model = YOLO("yolo11n-pose.pt")
        except Exception as e:
            self.error_signal.emit(f"模型載入失敗: {e}")
            return

        # 開啟相機
        cap = open_camera()
        if cap is None:
            self.error_signal.emit("無法開啟相機")
            return

        # 初始化緩衝區與狀態
        buf_t, buf_r, buf_g, buf_b = [], [], [], []  # RGB 時序緩衝
        start = time.time()      # 啟動時間
        prev_time = start        # FPS 計算用
        last_rois = None         # 上一幀的 ROI（YOLO 間隔幀沿用）
        frame_count = 0          # 幀計數器
        smooth_hr, smooth_br = None, None  # EMA 平滑狀態

        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)  # 鏡像翻轉
                h, w = frame.shape[:2]
                ts = time.time() - start    # 物理時間戳（秒）
                frame_count += 1

                # ── YOLO 偵測（每 N 幀一次，提升 FPS）──
                rois = None
                if frame_count % YOLO_INTERVAL == 0 or last_rois is None:
                    results = model(frame, verbose=False)
                    rois, _ = get_best_person(results, w, h)

                # 人臉消失時沿用上一幀
                if rois:
                    last_rois = rois
                elif last_rois:
                    rois = last_rois

                # ── RGB 均值萃取 + ROI 繪製 ──
                if rois:
                    face, fh, lc, rc = rois
                    rgb = compute_multi_roi_mean(frame, [fh, lc, rc])
                    if rgb:
                        buf_t.append(ts)
                        buf_r.append(rgb[0])
                        buf_g.append(rgb[1])
                        buf_b.append(rgb[2])

                    # 在影像上繪製 ROI 框（藍=臉框、綠=額頭、黃=臉頰）
                    cv2.rectangle(frame, (face[0], face[1]), (face[2], face[3]), (255, 0, 0), 2)
                    cv2.rectangle(frame, (fh[0], fh[1]), (fh[2], fh[3]), (0, 255, 0), 2)
                    cv2.rectangle(frame, (lc[0], lc[1]), (lc[2], lc[3]), (0, 255, 255), 2)
                    cv2.rectangle(frame, (rc[0], rc[1]), (rc[2], rc[3]), (0, 255, 255), 2)

                # ── Vitals 計算（每 5 幀一次，降低運算負擔）──
                hr_val, br_val = None, None
                rppg_data = None

                if frame_count % 5 == 0 and ts >= STARTUP_SEC and len(buf_t) > 30:
                    # 裁切到最近 30 秒
                    cutoff = ts - BUFFER_SEC
                    while buf_t and buf_t[0] < cutoff:
                        buf_t.pop(0); buf_r.pop(0); buf_g.pop(0); buf_b.pop(0)

                    t_arr = np.array(buf_t)
                    # 時間戳內插：不等距 → 等距 25 Hz
                    t_uni = np.arange(t_arr[0], t_arr[-1], 1.0 / TARGET_FS)

                    if len(t_uni) > int(TARGET_FS * 3):
                        # 對 R/G/B 三通道都做線性內插
                        rgb_uni = np.column_stack([
                            interp1d(t_arr, buf_r, fill_value='extrapolate')(t_uni),
                            interp1d(t_arr, buf_g, fill_value='extrapolate')(t_uni),
                            interp1d(t_arr, buf_b, fill_value='extrapolate')(t_uni),
                        ])
                        # POS 演算法：RGB → rPPG 脈搏波
                        rppg = pos_process(rgb_uni, TARGET_FS)
                        rppg_data = rppg.tolist()

                        # HR（FFT 頻譜法）+ slew rate + EMA 平滑
                        hr_raw = compute_heart_rate(rppg, TARGET_FS)
                        if hr_raw and 40 < hr_raw < 180:
                            if smooth_hr is None:
                                smooth_hr = hr_raw
                            else:
                                d = np.clip(hr_raw - smooth_hr, -5, 5)  # slew rate ±5 BPM
                                smooth_hr = 0.8 * smooth_hr + 0.2 * (smooth_hr + d)  # EMA
                            hr_val = smooth_hr

                        # RR（自相關函數法）+ slew rate + EMA 平滑
                        br_raw = compute_breathing_rate(rppg, TARGET_FS)
                        if br_raw and 6 < br_raw < 30:
                            if smooth_br is None:
                                smooth_br = br_raw
                            else:
                                d = np.clip(br_raw - smooth_br, -1, 1)  # slew rate ±1 RPM
                                smooth_br = 0.8 * smooth_br + 0.2 * (smooth_br + d)  # EMA
                            br_val = smooth_br

                # ── FPS 計算 ──
                now = time.time()
                fps = 1.0 / (now - prev_time) if (now - prev_time) > 0 else 0
                prev_time = now

                # ── 透過 Signal 發送結果給 UI 主線程 ──
                self.vitals_ready.emit({
                    'hr': hr_val, 'br': br_val,
                    'rppg': rppg_data,
                    'elapsed': ts, 'fps': fps,
                })

                # 將 OpenCV BGR 影像轉為 Qt 的 QImage (RGB) 發送
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                qimg = QImage(rgb_frame.data, w, h, w * 3, QImage.Format.Format_RGB888)
                self.frame_ready.emit(qimg.copy())  # .copy() 確保資料在線程間安全

        except Exception as e:
            self.error_signal.emit(str(e))
        finally:
            cap.release()  # 無論如何都要釋放相機


# ══════════════════════════════════════════════════════════
# PyQt6 主視窗
# ══════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    """rPPG 生醫監測儀表板主視窗。

    左右分欄佈局：
    - 左側 (stretch=3)：相機畫面 QLabel
    - 右側 (stretch=2)：FPS + HR + RR + 計時器 + 波形 + 按鈕
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("rPPG Bio Dashboard")
        self.resize(1200, 700)

        self.worker = None           # 工作線程實例
        self.cached_hr = None        # 緩存的最新 HR（供 1Hz 定拍讀取）
        self.cached_br = None        # 緩存的最新 RR
        self.cached_rppg = None      # 緩存的最新波形資料
        self.cached_elapsed = 0      # 緩存的執行時間

        self._build_ui()

        # 1 Hz 定拍刷新數值（獨立於工作線程的運算頻率）
        # 目的：避免數值每秒跳動 5 次（工作線程每 5 幀發送一次），
        #       改為每秒只更新一次，如 ICU 監視器般穩定易讀
        self.vitals_timer = QTimer()
        self.vitals_timer.timeout.connect(self._refresh_vitals_display)
        self.vitals_timer.start(1000)  # 1000 ms = 1 Hz

    def _build_ui(self):
        """建立 UI 元件與佈局。"""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # ── 左側：相機畫面 ──────────────────────────────────
        self.camera_label = QLabel("Press Start")
        self.camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_label.setMinimumSize(640, 480)
        self.camera_label.setStyleSheet("background: #111; color: #888; font-size: 24px;")
        main_layout.addWidget(self.camera_label, stretch=3)

        # ── 右側面板 ───────────────────────────────────────
        right = QVBoxLayout()
        main_layout.addLayout(right, stretch=2)

        # FPS 顯示（綠色小字）
        self.fps_label = QLabel("FPS: --")
        self.fps_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.fps_label.setFont(QFont("Consolas", 14))
        self.fps_label.setStyleSheet("color: #00FF00;")
        right.addWidget(self.fps_label)

        # Heart Rate 區塊（青色大字 + BPM 單位在右側）
        hr_title = QLabel("Heart Rate")
        hr_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hr_title.setFont(QFont("Consolas", 18))
        hr_title.setStyleSheet("color: #00AAAA;")
        right.addWidget(hr_title)

        hr_row = QHBoxLayout()
        hr_row.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.hr_label = QLabel("--")
        self.hr_label.setFont(QFont("Consolas", 64, QFont.Weight.Bold))
        self.hr_label.setStyleSheet("color: #00FFFF;")
        hr_row.addWidget(self.hr_label)
        hr_unit = QLabel("BPM")
        hr_unit.setFont(QFont("Consolas", 20))
        hr_unit.setStyleSheet("color: #00AAAA;")
        hr_unit.setAlignment(Qt.AlignmentFlag.AlignBottom)
        hr_row.addWidget(hr_unit)
        right.addLayout(hr_row)

        # Respiratory Rate 區塊（橙色大字 + RPM 單位在右側）
        rr_title = QLabel("Respiratory Rate")
        rr_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        rr_title.setFont(QFont("Consolas", 18))
        rr_title.setStyleSheet("color: #AA8040;")
        right.addWidget(rr_title)

        rr_row = QHBoxLayout()
        rr_row.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.rr_label = QLabel("--")
        self.rr_label.setFont(QFont("Consolas", 48, QFont.Weight.Bold))
        self.rr_label.setStyleSheet("color: #FFD080;")
        rr_row.addWidget(self.rr_label)
        rr_unit = QLabel("RPM")
        rr_unit.setFont(QFont("Consolas", 18))
        rr_unit.setStyleSheet("color: #AA8040;")
        rr_unit.setAlignment(Qt.AlignmentFlag.AlignBottom)
        rr_row.addWidget(rr_unit)
        right.addLayout(rr_row)

        # 執行時間計時器（灰色，MM:SS 格式）
        self.elapsed_label = QLabel("Elapsed: 00:00")
        self.elapsed_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.elapsed_label.setFont(QFont("Consolas", 16))
        self.elapsed_label.setStyleSheet("color: #AAAAAA;")
        right.addWidget(self.elapsed_label)

        # 狀態提示（Buffering 倒數 / 錯誤訊息）
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setFont(QFont("Consolas", 14))
        self.status_label.setStyleSheet("color: #FF8800;")
        right.addWidget(self.status_label)

        # pyqtgraph 波形示波器
        pg.setConfigOptions(antialias=True)  # 抗鋸齒
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('#111')                    # 深色背景
        self.plot_widget.setLabel('bottom', 'Time', units='s')    # X 軸標籤
        self.plot_widget.setLabel('left', 'rPPG')                 # Y 軸標籤
        self.plot_widget.setXRange(0, 30)                         # 鎖定 30 秒視窗
        self.plot_widget.setMouseEnabled(x=False, y=False)        # 禁止滑鼠拖曳
        self.plot_curve = self.plot_widget.plot(pen=pg.mkPen('#00FF00', width=1.5))  # 綠色曲線
        right.addWidget(self.plot_widget, stretch=2)

        # Start / Stop 按鈕
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start")
        self.start_btn.setFont(QFont("Consolas", 16))
        self.start_btn.setStyleSheet("background: #228B22; color: white; padding: 10px;")
        self.start_btn.clicked.connect(self._on_start)  # 點擊 → 啟動工作線程
        btn_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setFont(QFont("Consolas", 16))
        self.stop_btn.setStyleSheet("background: #B22222; color: white; padding: 10px;")
        self.stop_btn.setEnabled(False)  # 初始時禁用
        self.stop_btn.clicked.connect(self._on_stop)    # 點擊 → 停止工作線程
        btn_layout.addWidget(self.stop_btn)
        right.addLayout(btn_layout)

    # ── Start / Stop 生命週期管理 ────────────────────────────

    def _on_start(self):
        """啟動工作線程，開始量測。"""
        if self.worker and self.worker.isRunning():
            return  # 已在運行中
        self.worker = SignalWorker()
        # 連接 Signal → Slot
        self.worker.frame_ready.connect(self._on_frame)      # 影像 → 更新左側畫面
        self.worker.vitals_ready.connect(self._on_vitals)     # 數據 → 緩存 + 更新波形
        self.worker.error_signal.connect(self._on_error)      # 錯誤 → 顯示提示
        self.worker.start()  # 啟動線程（呼叫 run()）
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("Buffering...")

    def _on_stop(self):
        """停止工作線程，釋放資源，重置 UI。"""
        if self.worker:
            self.worker.stop()      # 設定 running=False
            self.worker.wait(5000)  # 最多等 5 秒讓線程結束
            self.worker = None
        # 重置所有 UI 元素
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.hr_label.setText("--")
        self.rr_label.setText("--")
        self.fps_label.setText("FPS: --")
        self.elapsed_label.setText("Elapsed: 00:00")
        self.status_label.setText("")
        self.cached_elapsed = 0
        self.plot_curve.setData([], [])              # 清空波形
        self.camera_label.setPixmap(QPixmap())        # 清空畫面
        self.camera_label.setText("Press Start")

    # ── Signal Slot 處理函式 ─────────────────────────────────

    def _on_frame(self, qimg: QImage):
        """收到工作線程的影像幀，更新左側相機畫面。"""
        pixmap = QPixmap.fromImage(qimg)
        # 等比例縮放到 QLabel 的大小
        scaled = pixmap.scaled(self.camera_label.size(),
                               Qt.AspectRatioMode.KeepAspectRatio,
                               Qt.TransformationMode.SmoothTransformation)
        self.camera_label.setPixmap(scaled)

    def _on_vitals(self, data: dict):
        """收到工作線程的生命徵象數據，緩存並即時更新波形。

        數值（HR/RR）不在此處更新，而是由 1Hz QTimer 定拍讀取緩存值，
        避免數值跳動太頻繁。波形則即時更新以保持流暢。
        """
        # 緩存最新數據（供 _refresh_vitals_display 讀取）
        self.cached_hr = data.get('hr')
        self.cached_br = data.get('br')
        self.cached_rppg = data.get('rppg')
        self.cached_elapsed = data.get('elapsed', 0)

        # FPS 即時更新（不需要定拍）
        fps = data.get('fps')
        if fps is not None:
            self.fps_label.setText(f"FPS: {fps:.1f}")

        # 波形即時更新（pyqtgraph 高效能，可每幀更新不卡頓）
        if self.cached_rppg and len(self.cached_rppg) > 2:
            n = len(self.cached_rppg)
            t = np.linspace(0, n / TARGET_FS, n)  # 時間軸
            self.plot_curve.setData(t, self.cached_rppg)

            # 鎖定 30 秒滾動視窗（X 軸始終顯示最近 30 秒）
            duration = n / TARGET_FS
            if duration > 30:
                self.plot_widget.setXRange(duration - 30, duration)
            else:
                self.plot_widget.setXRange(0, max(30, duration))

    def _refresh_vitals_display(self):
        """1 Hz 定拍刷新：HR/RR 數值 + Buffering 倒數 + 執行時間。

        為什麼不直接在 _on_vitals 中更新？
        因為工作線程的發送頻率較高（每 5 幀一次 ≈ 5 Hz），
        如果數值每秒跳動 5 次，使用者會感到視覺疲勞。
        用獨立的 1Hz QTimer 讓數值每秒只更新一次，如醫療監視器般穩定。
        """
        # Buffering 倒數：0~10 秒顯示進度，>=10 秒立即消失
        if self.cached_elapsed > 0 and self.cached_elapsed < STARTUP_SEC:
            count = min(int(self.cached_elapsed) + 1, int(STARTUP_SEC))
            self.status_label.setText(f"Buffering... {count}/{int(STARTUP_SEC)}s")
        else:
            self.status_label.setText("")

        # 更新 HR / RR 數值
        if self.cached_hr is not None:
            self.hr_label.setText(f"{self.cached_hr:.0f}")
        if self.cached_br is not None:
            self.rr_label.setText(f"{self.cached_br:.0f}")

        # 更新執行時間（MM:SS 格式）
        if self.cached_elapsed > 0:
            mins = int(self.cached_elapsed) // 60
            secs = int(self.cached_elapsed) % 60
            self.elapsed_label.setText(f"Elapsed: {mins:02d}:{secs:02d}")

    def _on_error(self, msg: str):
        """收到工作線程的錯誤訊息，顯示提示並停止。"""
        self.status_label.setText(f"Error: {msg}")
        self._on_stop()

    def closeEvent(self, event):
        """視窗關閉事件：確保工作線程安全退出，不留殘留。"""
        self._on_stop()
        event.accept()


# ── 程式進入點 ────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")       # 使用 Fusion 風格（跨平台一致的現代外觀）
    win = MainWindow()
    win.show()
    sys.exit(app.exec())         # 進入 Qt 事件迴圈（阻塞直到視窗關閉）


if __name__ == "__main__":
    main()
