import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# ---------------------------------------------------
# Parameters
# ---------------------------------------------------
# TEST_DATA_FILE = r"C:\kai\Radar_Gesture_HandOff\data\processed_data\0702_val_10_test.npz"
#TEST_DATA_FILE = r"C:\kai\full_handoff\data\processed_data\test_dataset.npz"
TEST_DATA_FILE = os.path.join("data", "processed_data", "test_dataset.npz")

MODEL_PATH = r"C:\Users\USER\Desktop\radar-gesture-recognition-chore-update-20250815\radar-gesture-recognition-chore-update-20250815\src\output\models\20251117_111428\epoch_44_valLoss_0.0001.pth"

WINDOW_SIZE    = 30
HIGH_TH        = 0.5   # 进入手势阈值
LOW_TH         = 0.1   # 退出手势阈值

# Clip‐level classes: 0=Background, 1=Come, 2=Forward, 3=Wave, 4=Multi-Gesture, 5=Incomplete
row_names = ['Background','Down','Left','Right','Tap','Up']
col_names = ['Background','Down','Left','Right','Tap','Up','Multi-Gesture','Incomplete']
true_labels = [0, 1, 2, 3, 4, 5]
pred_labels = [0, 1, 2, 3, 4, 5, 6, 7]  # 6, 7 分別是 Multi-Gesture 和 Incomplete


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------------------------------------------
# 1. Model definition
# ---------------------------------------------------
class Gesture3DCNN(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(2, 32, 3), nn.ReLU(), nn.MaxPool3d(2), nn.BatchNorm3d(32),
            nn.Conv3d(32, 64, 3), nn.ReLU(), nn.MaxPool3d(2), nn.BatchNorm3d(64),
            nn.Conv3d(64,128, 3), nn.ReLU(), nn.MaxPool3d(2), nn.BatchNorm3d(128),
        )
        self.pool = nn.AdaptiveAvgPool3d((1,1,1))
        self.classifier = nn.Sequential(
            nn.Linear(128,128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return torch.softmax(self.classifier(x), dim=1)

# ---------------------------------------------------
# 2. Load data & model
# ---------------------------------------------------
data = np.load(TEST_DATA_FILE, allow_pickle=True)
features      = data['features']      # object array: (N_clips,)
ground_truths = data['ground_truths']

model = Gesture3DCNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()

N_clips = len(features)
print(f"Found {N_clips} clips.")

# ---------------------------------------------------
# Helpers
# ---------------------------------------------------
def extract_window(clip_feat, center_frame, window_size=WINDOW_SIZE):
    total = clip_feat.shape[-1]; half = window_size//2
    start = max(0, min(center_frame-half, total-window_size))
    end   = start + window_size
    win   = clip_feat[..., start:end]              # (2,32,32,window)
    win   = np.transpose(win, (0,3,1,2))           # (2,window,32,32)
    return np.expand_dims(win, 0)                  # (1,2,window,32,32)

def find_gesture_events(gt_clip, eps=1e-7):
    events=[]; in_g=False
    for i in range(len(gt_clip)):
        if np.max(gt_clip[i,1:])>eps:
            if not in_g:
                in_g=True; start=i
        else:
            if in_g:
                events.append((start,i-1)); in_g=False
    if in_g: events.append((start,len(gt_clip)-1))
    return events

# ---------------------------------------------------
# 3. Clip‐level inference & labeling
# ---------------------------------------------------
true_clip_labels=[]; pred_clip_labels=[]

with torch.no_grad():
    for idx in range(N_clips):
        clip_feat = features[idx]     # ndarray (2,32,32,frames)
        gt_clip   = ground_truths[idx]
        frames    = clip_feat.shape[-1]
        probs     = np.zeros((frames, 6))
        print(f'\r{idx}/{N_clips}', end='')

        # frame‐wise prediction
        for t in range(frames):
            win = extract_window(clip_feat, t)
            inp = torch.from_numpy(win).float().to(device)
            out = model(inp).cpu().numpy().squeeze()
            probs[t] = out

        # dual‐threshold state sequence
        pred_seq = np.zeros(frames, dtype=int); current=0
        for t in range(frames):
            non_bg = probs[t,1:]; i_max=np.argmax(non_bg)+1; p_max=non_bg[i_max-1]
            if current==0:
                if p_max>=HIGH_TH: current=i_max
            else:
                if p_max<LOW_TH: current=0
            pred_seq[t]=current

        # extract event labels
        events=[]; prev=0
        for lbl in pred_seq:
            if lbl!=0 and prev==0: events.append(lbl)
            prev=lbl

        # determine clip‐level label
        if len(events)==0:
            pred_lbl=0
        elif len(events)>1:
            pred_lbl=4   # Multi‐Gesture
        else:
            # single event—but check incomplete if never dropped to 0 at end
            if pred_seq[-1]!=0:
                pred_lbl=5  # Incomplete
            else:
                pred_lbl=events[0]
        pred_clip_labels.append(pred_lbl)

        # true label
        gt_events = find_gesture_events(gt_clip)
        if len(gt_events)==0:
            true_lbl=0
        elif len(gt_events)>1:
            true_lbl=4
        else:
            s,e=gt_events[0]; mid=(s+e)//2
            true_lbl=int(np.argmax(gt_clip[mid,1:])+1)
        true_clip_labels.append(true_lbl)

# ---------------------------------------------------
# 4. Confusion matrix & report (6 true × 6 pred)
# ---------------------------------------------------
cm_full = confusion_matrix(true_clip_labels, pred_clip_labels, labels=pred_labels)
cm = cm_full[:6, :]  # only first 6 true‐label rows

print("Confusion Matrix (4 true × 6 pred):")
print(cm)

report = classification_report(
    true_clip_labels, pred_clip_labels,
    labels=true_labels, target_names=row_names
)
print("\nClassification Report (no true ‘Multi’/‘Incomplete’):")
print(report)

# 绘制横向标签的混淆矩阵
plt.figure(figsize=(8, 6))
ax = sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=col_names, yticklabels=row_names,
    annot_kws={"size": 14}
)

# 把 x/y 轴的文字都横写
ax.set_xticklabels(col_names, rotation=0, ha='center', fontsize=12)
ax.set_yticklabels(row_names, rotation=0, va='center', fontsize=12)

plt.xlabel("Predicted", fontsize=14)
plt.ylabel("True", fontsize=14)
plt.title("Clip-level Confusion Matrix", fontsize=16)
plt.tight_layout()
plt.show()
