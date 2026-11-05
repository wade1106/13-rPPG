## 1. Peak Detection

- [ ] 1.1 實作 find_peaks 搭配 distance (>= 0.35s * fs) 與 prominence 約束
- [ ] 1.2 計算相鄰波峰的 RR interval 序列
- [ ] 1.3 處理波峰數量不足的情境（回傳安全預設值）

## 2. Heart Rate Calculation

- [ ] 2.1 實作 HR = 60 / mean(RR_intervals) 計算
- [ ] 2.2 實作雙重中值濾波 medfilt(47) + medfilt(97)
- [ ] 2.3 實作 slew rate limiting 限制 HR 每次更新最大變化量
- [ ] 2.4 實作開機 15 秒安全保護期（回傳 "--"）

## 3. Breathing Rate Calculation

- [ ] 3.1 對 RR interval 序列套用帶通濾波 (0.15-0.4 Hz) 萃取 RSA
- [ ] 3.2 實作自相關函數計算 np.correlate(rsa, rsa, 'same')
- [ ] 3.3 在延遲窗口 [90:130] 內搜尋最大峰值，換算 BR = (fs/br_r)*60
- [ ] 3.4 實作 slew rate +/- 1.0 RPM 限幅與 EMA 平滑
- [ ] 3.5 實作開機 15 秒防呆保護

## 4. Integration & Verification

- [ ] 4.1 整合 Stage 1-4 模組，建立 stage5_vitals.py 完整管線
- [ ] 4.2 驗證安靜靜息狀態下 HR 數值穩定（如 60-80 BPM）
- [ ] 4.3 驗證 HR 與 RR 數值同時顯示於畫面
- [ ] 4.4 驗證說話/眨眼後數值快速回穩而非劇烈跳動
