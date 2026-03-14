# mmWave 生命體徵 + 異常動作偵測 專案（4 類動作 + 生理警示）

這版已改成兩條邏輯同時跑：
- **動作辨識**：normal / fall / cough / agitation
- **生命體徵警示**：呼吸異常 / 心率異常 / 呼吸&心率異常

## 新增內容

### 1. 動作分類固定為 4 類
`config.py > ModelConfig.label_map`
- 0: `normal`
- 1: `fall`
- 2: `cough`
- 3: `agitation`

如果 `motion_level < motion_gate_threshold`，系統直接顯示 `normal`。
如果 `motion_level >= motion_gate_threshold`，才進模型推論。

### 2. 生理警示邏輯
新增 `AlertConfig`：
- `resp_low_rpm`
- `resp_high_rpm`
- `heart_low_bpm`
- `heart_high_bpm`
- `sustain_sec`

只有在數值**持續**超出範圍一段時間後，才會觸發警示圖片，避免瞬間雜訊誤報。

預設：
- 呼吸異常：RPM < 8 或 RPM > 24
- 心率異常：BPM < 50 或 BPM > 120
- 持續時間：8 秒

### 3. GUI 新增警示圖片區
右側動作面板下方新增：
- 呼吸 / 心率警示文字
- 警示圖片框

請把圖片放在：
- `assets/actions/normal.png`
- `assets/actions/fall.png`
- `assets/actions/cough.png`
- `assets/actions/agitation.png`
- `assets/alerts/normal.png`
- `assets/alerts/watch.png`
- `assets/alerts/resp_alert.png`
- `assets/alerts/heart_alert.png`
- `assets/alerts/resp_heart_alert.png`

沒有圖片也可正常執行，GUI 會顯示文字。

## 主要檔案
- `pipeline.py`：主流程，整合動作與生理警示
- `alert_evaluator.py`：持續異常判斷
- `state.py`：新增警示狀態欄位
- `gui.py`：新增警示圖片與文字
- `dataset_tools.py`：資料標籤改成 4 類
- `config.py`：新增 `AlertConfig`

## 你需要同步修改的地方

### 訓練資料
訓練資料標籤請改成 4 類：
- `normal`
- `fall`
- `cough`
- `agitation`

### 模型
你的訓練模型也必須對應這 4 類輸出。

## 執行
```bash
python main.py
```

## 常用調整
```python
cfg.model.model_path = r"trained_models/mmwave_action_rf.joblib"
cfg.model.motion_gate_threshold = 3.0
cfg.alert.resp_low_rpm = 8.0
cfg.alert.resp_high_rpm = 24.0
cfg.alert.heart_low_bpm = 50.0
cfg.alert.heart_high_bpm = 120.0
cfg.alert.sustain_sec = 8.0
```
