# mmWave 生命體徵 + 異常動作偵測 專案（GUI 版）

這版是依照你最新需求重整的整合專案，重點不是只有「載入模型」，而是把 **MOTION 分數 + 模型輸出** 一起拿來做動作判斷。

## 這版的判斷邏輯

1. 先做 range FFT 與 target bin 選擇
2. 從 target bin 取 phase，估測呼吸 / 心跳
3. 同時計算 motion level
4. **如果 MOTION 低於門檻** → 直接判成 `normal`
5. **如果 MOTION 高於門檻** → 啟用已訓練模型做動作分類
6. 模型分數不足時，GUI 顯示 `疑似動作`

也就是說，這份不是單純 `model.predict()`，而是：

**Action = Motion Gate + Model Inference**

---

## GUI 版面

已改成接近你草圖、並再做優化：

### 上半部
- 左側：即時數值卡片
  - BIN
  - SCORE
  - MOTION
  - RPM
  - BPM
  - FPS / Frame
- 右側：目前動作顯示
  - 文字動作標籤
  - 可放動作圖片
  - 顯示 model score / event

### 下半部波形區
- Range
- Phase
- Respiration
- Heart
- Motion

其中 Motion 放成底部橫向大圖，方便看異常動作能量變化。

---

## 專案結構

- `main.py`：主程式
- `config.py`：集中設定
- `state.py`：GUI / pipeline 共用狀態
- `dsp.py`：FFT / unwrap / 頻域工具
- `bin_tracker.py`：自動選 target bin
- `motion_analyzer.py`：motion 分數與穩定判定
- `vitals.py`：呼吸 / 心跳估測
- `model_runner.py`：載入 sklearn / joblib 模型並推論
- `pipeline.py`：生命體徵 + 動作判斷整合流程
- `device_kkt.py`：KKT_Module / FRM 裝置接收
- `gui.py`：新版 GUI
- `dataset_tools.py`：資料集與特徵建構
- `train_model.py`：訓練模型
- `predict_model.py`：離線模型測試

---

## 動作圖片

若你要在 GUI 右上角顯示動作圖片，把圖片放在：

`assets/actions/`

檔名對應 label，例如：
- `normal.png`
- `cough.png`
- `fall.png`
- `walk.png`

沒有圖片時，GUI 仍可正常跑，會直接顯示文字。

---

## 使用方式

### 1. 修改雷達設定路徑

改 `config.py`：

```python
DeviceConfig.setting_dir
```

### 2. 指定模型

在 `main.py` 裡設定：

```python
cfg.model.model_path = r"trained_models/mmwave_action_rf.joblib"
```

### 3. 調整動作門檻

在 `config.py`：

```python
ModelConfig.motion_gate_threshold
```

### 4. 執行

```bash
python main.py
```

---

## 模型格式

目前支援：
- `joblib`
- `pickle`
- sklearn 類別模型

模型需至少支援：
- `predict()`

若有 `predict_proba()`，GUI 會一起顯示 score。

---

## 安裝

```bash
pip install numpy pyqtgraph PySide2 scikit-learn joblib
```

並放在有 `KKT_Module` 的電腦上執行。

---

## 你接下來通常還會想補的

1. 把異常動作事件自動存成 segment
2. 做事件列表與事件時間軸
3. 加入模型類別分數條圖
4. 加入錄影 / 回放模式



## v3 說明
- 未載入模型時，GUI 不再顯示 motion_detected，而是顯示「偵測到動作」。
- Frame 若為負值，GUI 會顯示 --，避免誤判為正常 frame 編號。


## 並行版說明

此版本已改為「動作偵測與生命體徵並行」：
- 不再因為 `motion_flag=True` 就停止 RPM / BPM 計算。
- 有動作時仍持續更新 `phase buffer` 與生命體徵波形。
- GUI 中若 RPM / BPM 後面帶有 `*`，代表該數值是在動作干擾期間估得，可信度較低。
- 狀態列中的 `VITALS` 會顯示 `stable`、`motion_interference` 或 `warming_up`。
