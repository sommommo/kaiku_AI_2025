# radar-gesture-recognition

以毫米波雷達進行**手勢辨識**。目前功能聚焦在：

- 資料**前處理**（`.h5` → `.npz`）
- **訓練** 3D CNN
- **評估**（混淆矩陣）
- **線上即時推論（GUI）**

> 本專案使用 **開酷科技 K60168A Dongle**；其中的 **Library** 與 **KKT_Module** 皆為開酷科技提供之 API，請依原廠指引安裝設定。  
> 官方網站：[@開酷科技 KaikuTek](https://www.kaikutek.com/zh-tw)

---

## Kaggle 資料集

- Kaggle dataset：https://www.kaggle.com/datasets/shengkai1020/handgesture-mmwave-v1-0?select=HandGesture-mmWave-v1.0

### 資料格式

每個 `.h5` 檔至少包含：

- `DS1`: 雷達特徵序列  
  - 形狀： `(2, 32, 32, T)`  
- `LABEL`: 長度 `T` 的 0/1（按住手勢=1、放開=0），`uint8`

資料夾結構（依類別分資料夾；順序需與模型輸出一致：`["Background","PatPat","Wave","Come"]`：
- `data/`
  - `train/`
    - `background/` — `*.h5`
    - `patpat/` — `*.h5`
    - `wave/` — `*.h5`
    - `come/` — `*.h5`
  - `val/`
    - `background/` — `*.h5`
    - `patpat/` — `*.h5`
    - `wave/` — `*.h5`
    - `come/` — `*.h5`
    - 
---

## Python 版本與環境

- **建議使用 Python 3.8.9**（本專案以此版本測試最穩定）。
- Windows 安裝步驟：
  1. 下載並安裝 **Python 3.8.9**（安裝時勾選 *Add Python to PATH*）：<https://www.python.org/downloads/release/python-389/>
  2. 確認版本：
     ```powershell
     python --version
     ```
  3. 建立並啟用虛擬環境：
     ```powershell
     py -3.8 -m venv venv
     .\venv\Scripts\activate
     ```
     > 若 `python` 已指向 3.8.9，也可：
     > ```powershell
     > python -m venv venv
     > .\venv\Scripts\activate
     > ```

---

## 安裝環境

```powershell
.\setup.bat
```
# torch 請依你的 CUDA/CPU 平台安裝（範例）
# pip install torch --index-url https://download.pytorch.org/whl/cu121

### 執行流程
### A. 前處理
將 .h5 轉為訓練用 .npz（輸出到 data/processed_data/）：
```
python .\src\data_preprocessing.py
```
### B. 檢視前處理結果（可選）
互動檢視指定索引的 labels 與 ground_truths：
```
python .\src\read.py
```
### C. 訓練 3D CNN
```
python .\src\training.py
```
### D. 評估（混淆矩陣）
```
python .\src\confusion.py
```
### E. 線上即時推論（GUI）
```
python .\src\online_inference_gui.py
```
