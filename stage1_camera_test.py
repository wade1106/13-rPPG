"""Stage 1: WebCam 視訊串流測試
- DirectShow 後端
- FPS 即時顯示
- 鏡像翻轉
- 按 Q 安全退出
"""

import cv2
import time


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

    print("[ERROR] 無法開啟任何相機，請確認相機已連接且未被其他程式佔用")
    return None


def main():
    cap = open_camera()
    if cap is None:
        return

    prev_time = time.time()

    print("[INFO] 串流啟動，按 Q 退出")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] 無法讀取影像幀，結束串流")
            break

        # 鏡像翻轉
        frame = cv2.flip(frame, 1)

        # 計算 FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        # 顯示 FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                     cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        cv2.imshow("rPPG - Stage 1 Camera Test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] 使用者按下 Q，退出")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] 資源已釋放，程式結束")


if __name__ == "__main__":
    main()
