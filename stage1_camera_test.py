"""Stage 1: WebCam 視訊串流測試

本程式為 rPPG 系統的第一階段，負責建立穩定的 WebCam 即時視訊串流。
這是整條管線的基礎——沒有可靠的影像輸入，後續所有處理都無法進行。

功能：
- 使用 DirectShow 後端開啟 WebCam（Windows 上最穩定的後端）
- 即時計算並顯示 FPS（為後續取樣率校準打基礎）
- 水平鏡像翻轉（讓使用者操作直覺，揮右手畫面右側動）
- 按 Q 安全退出，正確釋放所有硬體資源
- 相機裝置 fallback 機制（index 0 失敗自動嘗試 index 1）
"""

import cv2    # OpenCV：電腦視覺核心函式庫，負責相機控制與影像顯示
import time   # time：用於計算每幀的真實物理時間戳與 FPS


def open_camera(max_index=2):
    """嘗試開啟相機，支援 fallback 探測多個裝置索引。

    Windows 上筆電常有虛擬相機佔用 index 0（如 OBS Virtual Camera），
    實體 WebCam 可能在 index 1，因此需要依序嘗試多個索引。

    Args:
        max_index: 最多嘗試的裝置索引數量（預設探測 index 0 和 1）

    Returns:
        成功開啟的 cv2.VideoCapture 物件，或 None（全部失敗時）
    """
    for index in range(max_index):
        print(f"[INFO] 嘗試開啟相機 index={index} (DirectShow)...")
        # cv2.CAP_DSHOW = DirectShow 後端
        # Windows 上 MSMF（預設後端）常導致黑屏或延遲，DirectShow 最穩定
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():  # isOpened() 確認相機是否真正可用
            print(f"[OK] 相機 index={index} 開啟成功")
            return cap
        cap.release()  # 開啟失敗時必須釋放資源，避免佔用裝置
        print(f"[WARN] 相機 index={index} 開啟失敗")

    print("[ERROR] 無法開啟任何相機，請確認相機已連接且未被其他程式佔用")
    return None


def main():
    # ── A. 開啟相機 ─────────────────────────────────────────
    cap = open_camera()
    if cap is None:
        return  # 相機開啟失敗，直接結束

    # prev_time 用於計算每幀的時間差，進而得出即時 FPS
    prev_time = time.time()

    print("[INFO] 串流啟動，按 Q 退出")

    # ── B. 主迴圈：逐幀擷取、處理、顯示 ─────────────────────
    while True:
        # cap.read() 回傳 (ret, frame)
        # ret=True 表示成功讀取一幀，ret=False 表示相機斷線或錯誤
        ret, frame = cap.read()
        if not ret:
            print("[WARN] 無法讀取影像幀，結束串流")
            break

        # 水平鏡像翻轉（flipCode=1）
        # 原因：WebCam 拍攝的影像是左右相反的（如照鏡子），
        # 翻轉後使用者揮右手，畫面上右側會動，操作更直覺
        frame = cv2.flip(frame, 1)

        # 計算即時 FPS（Frames Per Second，每秒幀數）
        # 方法：用當前時間減去上一幀時間，取倒數
        # 這比 cv2.CAP_PROP_FPS 的標稱值更能反映真實效能
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time  # 更新為下一幀的基準時間

        # 將 FPS 數值以綠色文字疊加在畫面左上角 (10, 30) 位置
        # 參數：影像, 文字, 座標, 字型, 大小, 顏色(BGR), 線寬
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                     cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        # 顯示影像視窗
        cv2.imshow("rPPG - Stage 1 Camera Test", frame)

        # waitKey(1) 等待 1 毫秒的鍵盤輸入
        # & 0xFF 是位元遮罩，確保跨平台相容（某些系統回傳值超過 8 bit）
        # ord('q') 將字元 'q' 轉為 ASCII 碼進行比對
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] 使用者按下 Q，退出")
            break

    # ── C. 釋放資源 ─────────────────────────────────────────
    cap.release()            # 釋放相機裝置，讓其他程式可使用
    cv2.destroyAllWindows()  # 關閉所有 OpenCV 建立的視窗
    print("[INFO] 資源已釋放，程式結束")


# Python 慣用寫法：只有直接執行此檔案時才呼叫 main()，
# 被其他模組 import 時不會自動執行
if __name__ == "__main__":
    main()
