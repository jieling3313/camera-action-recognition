# MediaPipe 33 點 One-Shot Action Recognition 模型架構

## 一、概述

本模型實作論文「One-Shot Action Recognition via Multi-Scale Spatial-Temporal Skeleton Matching」(2307.07286v2)，適配為 MediaPipe 33 關節格式，用於即時骨架動作辨識。

### 主要特點
- **輸入格式**: MediaPipe 33 點骨架序列 (T, 33, 3) 或 (T, 33, 4)
- **多尺度處理**: 3 個空間尺度 + 3 個時間尺度
- **匹配策略**: Earth Mover's Distance (EMD) 最佳傳輸匹配
- **應用場景**:
  1. NTU RGB+D 60 類動作分類
  2. One-Shot Learning 新動作辨識

---

## 二、MediaPipe 33 關節點定義

```
索引  名稱              描述
────────────────────────────────────────
 0    nose             鼻子
 1    left_eye_inner   左眼內側
 2    left_eye         左眼
 3    left_eye_outer   左眼外側
 4    right_eye_inner  右眼內側
 5    right_eye        右眼
 6    right_eye_outer  右眼外側
 7    left_ear         左耳
 8    right_ear        右耳
 9    mouth_left       嘴巴左側
10    mouth_right      嘴巴右側
11    left_shoulder    左肩
12    right_shoulder   右肩
13    left_elbow       左肘
14    right_elbow      右肘
15    left_wrist       左腕
16    right_wrist      右腕
17    left_pinky       左小指
18    right_pinky      右小指
19    left_index       左食指
20    right_index      右食指
21    left_thumb       左拇指
22    right_thumb      右拇指
23    left_hip         左髖
24    right_hip        右髖
25    left_knee        左膝
26    right_knee       右膝
27    left_ankle       左踝
28    right_ankle      右踝
29    left_heel        左腳跟
30    right_heel       右腳跟
31    left_foot_index  左腳尖
32    right_foot_index 右腳尖
```

---

## 三、資料集格式

### 3.1 已處理的 MediaPipe Dataset

**位置**: `/media/jieling/Expansion/NTU RGB+D/nturgb+d_rgb_mediapipe/mediapipe_skeletons/`

**檔案格式**:
- `.npy`: NumPy 陣列，形狀 `(T, 33, 4)` 其中:
  - `T`: 幀數（可變長度）
  - `33`: MediaPipe 關節數
  - `4`: 座標 (x, y, z, visibility)
- `.json`: JSON 格式，包含完整的幀和關節資訊

**資料集統計**:
- 總檔案數: 56,880 (完整 NTU RGB+D 60)
- 動作類別: 60 類
- 命名規則: `SsssCcccPpppRrrrAaaa.npy`
  - S: setup number (1-17)
  - C: camera ID (1-3)
  - P: performer ID (1-40)
  - R: replication number (1-2)
  - A: action class (1-60)

### 3.2 資料格式範例

```python
import numpy as np
skeleton = np.load('S001C001P001R001A001.npy')
# Shape: (103, 33, 4)  # 103 幀, 33 關節, 4 通道 (x, y, z, visibility)

# 座標範圍 (正規化至 [0, 1]):
# x: 0.51 ~ 0.58
# y: 0.29 ~ 0.68
# z: -0.29 ~ 0.19
# visibility: 0.50 ~ 1.00
```

---

## 四、模型架構

### 4.1 整體流程

```
輸入骨架序列 (N, T, 33, 3)
         ↓
   Data BatchNorm
         ↓
   共享 AGC 區塊 (1-6)
    ├─ Block 1: 3 → 64
    ├─ Block 2-3: 64 → 64
    ├─ Block 4: 64 → 128 (stride=2)
    └─ Block 5-6: 128 → 128
         ↓
   多空間尺度處理
    ├─ Scale 1 (33 關節): 128 → 256
    ├─ Scale 2 (12 部位): 128 → 256
    └─ Scale 3 (6 肢體):  128 → 256
         ↓
   全域平均池化
         ↓
   特徵向量: (N, 256)
         ↓
┌────────────────┬────────────────┐
│  分類模式      │  One-Shot 模式  │
├────────────────┼────────────────┤
│ Classifier     │ EMD Matcher    │
│ 256 → 60      │ 特徵比較        │
└────────────────┴────────────────┘
         ↓
   動作類別預測
```

### 4.2 多尺度空間池化群組

**尺度 2 (12 部位)**:
```
群組  名稱      包含關節
────────────────────────────────────
 0    頭部      0-10 (臉部所有點)
 1    軀幹      11, 12, 23, 24
 2    左上臂    11, 13
 3    左前臂    13, 15
 4    左手      15, 17, 19, 21
 5    右上臂    12, 14
 6    右前臂    14, 16
 7    右手      16, 18, 20, 22
 8    左大腿    23, 25
 9    左小腿+腳  25, 27, 29, 31
10    右大腿    24, 26
11    右小腿+腳  26, 28, 30, 32
```

**尺度 3 (6 超級部位)**:
```
群組  名稱      包含關節
────────────────────────────────────
 0    頭部      0-10
 1    軀幹      11, 12, 23, 24
 2    左臂      11, 13, 15, 17, 19, 21
 3    右臂      12, 14, 16, 18, 20, 22
 4    左腿      23, 25, 27, 29, 31
 5    右腿      24, 26, 28, 30, 32
```

### 4.3 核心元件

#### AGCBlock (自適應圖卷積區塊)
```python
class AGCBlock(nn.Module):
    def forward(self, x):
        # 步驟 1: 空間圖卷積 (GraphConv)
        x = self.gcn(x)

        # 步驟 2: 時間卷積 (TemporalConv)
        x = self.tcn(x)

        # 步驟 3: 殘差連接
        return x + residual
```

#### EMD Matcher (Earth Mover's Distance)
```python
class EMDMatcher:
    def compute_emd(self, X, Y):
        # 1. 計算成對距離矩陣
        D = self.compute_distance_matrix(X, Y)

        # 2. 計算權重 (交叉參考機制)
        r, c = self.compute_weights(X, Y)

        # 3. 求解最佳傳輸問題
        pi = ot.emd(r, c, D)

        # 4. 計算語義相關性分數
        score = (similarity * pi).sum()
        return emd, score
```

---

## 五、訓練策略

### 5.1 兩階段訓練

**階段 1: 預訓練 (Pre-training)**
- 使用 NTU RGB+D 60 類進行監督學習
- 損失函數: Cross-Entropy Loss
- 目標: 學習通用的骨架特徵表示

**階段 2: 元學習 (Meta-training)**
- 使用 5-way 1-shot 任務
- 損失函數: Softmax Cross-Entropy
- 目標: 學習 One-Shot 匹配能力

### 5.2 資料分割 (Cross-Subject)

**訓練集 Performers**: 1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38

**驗證集 Performers**: 其餘 performers

### 5.3 建議的訓練參數

```bash
python train_mediapipe33.py \
    --data_path "/media/jieling/Expansion/NTU RGB+D/nturgb+d_rgb_mediapipe/mediapipe_skeletons" \
    --epochs 200 \
    --batch_size 32 \
    --num_classes 60 \
    --benchmark xsub \
    --lr 0.001 \
    --device cuda \
    --num_workers 4 \
    --max_frames 64 \
    --save_dir checkpoints_mediapipe33
```

---

## 六、One-Shot Learning 功能

### 6.1 支援集 (Support Set) 設定

One-Shot Learning 允許用少量樣本（1-5 個）定義新動作：

```python
# 新增自訂動作
support_set = {
    "wave_hand": [feature_1, feature_2, feature_3],  # 3 個示範
    "thumbs_up": [feature_1],                         # 1 個示範
}
```

### 6.2 辨識流程

1. **錄製動作**: 使用 GUI Page 1 錄製 2-3 秒的影片
2. **提取特徵**: 使用模型提取 256 維特徵向量
3. **儲存支援集**: 將特徵儲存至 ROS Parameter Server
4. **即時匹配**: 使用 EMD 或餘弦相似度比較

### 6.3 建議的最小樣本數

| 動作複雜度 | 建議樣本數 | 影片長度 |
|-----------|-----------|---------|
| 簡單動作   | 1-2 個    | 2 秒    |
| 中等動作   | 3-5 個    | 3 秒    |
| 複雜動作   | 5-10 個   | 3-5 秒  |

---

## 七、與 D435i 相機整合

### 7.1 即時骨架提取流程

```
D435i RGB 影像 (30fps @ 640x480)
         ↓
   MediaPipe Pose
         ↓
   33 關鍵點 (x, y, z, visibility)
         ↓
   骨架序列緩衝區 (64 幀)
         ↓
   MediaPipe33 模型
         ↓
   動作辨識結果
```

### 7.2 FPS 選擇建議

| 相機 FPS | 緩衝區幀數 | 動作涵蓋時間 | 適用場景 |
|---------|-----------|-------------|---------|
| 30 fps  | 64 幀     | 2.13 秒     | 快速動作 |
| 15 fps  | 64 幀     | 4.27 秒     | 緩慢動作 |

---

## 八、待完成項目

### 8.1 必要項目

- [ ] **建立訓練腳本** `train_mediapipe33.py`
  - 支援 MediaPipe 33 點格式
  - 支援 NTU RGB+D 60 類預訓練
  - 支援 One-Shot 元學習

- [ ] **修改 ROS 辨識節點**
  - 從 COCO 17 點改為 MediaPipe 33 點
  - 載入 MediaPipe33 模型

- [ ] **建立資料集載入器**
  - 讀取 `/media/jieling/Expansion/NTU RGB+D/nturgb+d_rgb_mediapipe/mediapipe_skeletons/`
  - 支援 Cross-Subject 和 Cross-View 分割

### 8.2 可選改進

- [ ] 加入資料增強（時間裁剪、空間抖動）
- [ ] 支援多人骨架
- [ ] 加入骨架視覺化工具

---

## 九、檔案結構

```
models/mediapipe33/
├── __init__.py                      # 模組初始化
├── skeleton_model_mediapipe33.py    # 核心模型定義
├── extract_mediapipe_from_ntu.py    # NTU 影片提取腳本
├── train_mediapipe33.py             # [待建立] 訓練腳本
├── mediapipe33_model.md             # 本文件
└── .venv/                           # Python 虛擬環境
```

---

## 十、參考資料

1. **論文**: "One-Shot Action Recognition via Multi-Scale Spatial-Temporal Skeleton Matching" (arXiv:2307.07286v2)
2. **NTU RGB+D Dataset**: https://rose1.ntu.edu.sg/dataset/actionRecognition/
3. **MediaPipe Pose**: https://google.github.io/mediapipe/solutions/pose.html
4. **POT (Python Optimal Transport)**: https://pythonot.github.io/

---

*最後更新: 2026-03-30*
*作者: Claude AI Assistant*
