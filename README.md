# One-Shot Action Recognition 技術指南

## 目錄

1. [環境建置](#環境建置)
2. [模型架構](#模型架構)
3. [骨架動作辨識原理](#骨架動作辨識原理)
4. [訓練流程](#訓練流程)
5. [測試流程](#測試流程)
6. [完整程式碼範例](#完整程式碼範例)

---

## 環境建置

### GPU 環境建置（RTX 5080）

#### 步驟 1：安裝 NVIDIA Container Toolkit

```bash
cd ros-yolo-opencv-project3/.devcontainer
./setup_gpu.sh
```

**腳本功能**：
- 添加 NVIDIA Container Toolkit GPG 金鑰
- 添加套件庫
- 安裝 nvidia-container-toolkit
- 配置 Docker Runtime
- 重啟 Docker 服務

#### 步驟 2：配置 Docker Compose

**檔案**：`ros-yolo-opencv-project3/.devcontainer/docker-compose.yml`

```yaml
services:
  ros-dev:
    build: .
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=all

    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

#### 步驟 3：配置 Dockerfile

**關鍵配置**：
```dockerfile
FROM cogrobot/robospection-ros-noetic:torch29cu128

# 使用 Python 3.10 安裝依賴
RUN python3.10 -m pip install --no-cache-dir --upgrade pip && \
    python3.10 -m pip install --no-cache-dir numpy scipy

# 安裝 PyTorch 2.9.1 Stable with CUDA 12.8
RUN python3.10 -m pip install --no-cache-dir --default-timeout=1000 \
    torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 安裝其他依賴
RUN python3.10 -m pip install --no-cache-dir \
    "psutil>=5.9.0" "tqdm>=4.64.0" "pyyaml>=5.3.1" "requests>=2.23.0"

RUN python3.10 -m pip install --default-timeout=1000 --no-cache-dir \
    "opencv-python>=4.6.0" "pandas>=1.1.4" \
    "matplotlib>=3.4.0" "seaborn>=0.11.0"

RUN python3.10 -m pip install --default-timeout=1000 --no-cache-dir ultralytics
RUN python3.10 -m pip install --no-cache-dir "pot>=0.9.6"
```

#### 步驟 4：建置並驗證

```bash
# 建置容器
cd ros-yolo-opencv-project3/.devcontainer
docker compose build
docker compose up -d

# 驗證 GPU
docker compose exec ros-dev python3.10 -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA:', torch.version.cuda)
print('GPU Available:', torch.cuda.is_available())
print('GPU Name:', torch.cuda.get_device_name(0))
print('GPU Capability:', torch.cuda.get_device_capability(0))
"
```

**預期輸出**：
```
PyTorch: 2.9.1+cu128
CUDA: 12.8
GPU Available: True
GPU Name: NVIDIA GeForce RTX 5080 Laptop GPU
GPU Capability: (12, 0)
```

---

### CPU 環境建置（備用方案）

#### 步驟 1：使用標準 Docker Compose

**檔案**：`ros-yolo-opencv-project3/.devcontainer/docker-compose.yml`

```yaml
services:
  ros-dev:
    build: .
    # 不需要 GPU 配置
```

#### 步驟 2：建置容器

```bash
cd ros-yolo-opencv-project3/.devcontainer
docker compose build
docker compose up -d
```

#### CPU vs GPU 對比

| 特性 | CPU | GPU (RTX 5080) |
|------|-----|----------------|
| 訓練速度 (50 epochs) | 2-3 天 | 4-5 小時 |
| Batch Size | 8-16 | 32-64 |
| 適用場景 | 小規模測試 | 完整訓練 |
| 記憶體需求 | 8GB RAM | 8GB VRAM |

---

## 模型架構

### 整體架構圖

```
輸入骨架序列 (T, 17, 3)
         ↓
   SkeletonEmbedding (特徵提取)
    ├─ AGCBlock (空間圖卷積)
    ├─ TemporalConv (時間卷積)
    └─ Multi-scale Pooling
         ↓
   特徵向量 (256-d)
         ↓
   EMDMatcher (最優傳輸距離匹配)
         ↓
   動作類別預測
```

### 核心組件

#### 1. COCOGraph - COCO 17 關鍵點圖結構

**定義 17 個關鍵點之間的連接關係**：

```python
class COCOGraph:
    num_nodes = 17  # COCO 17 keypoints

    # 定義骨架連接 (邊)
    edges = [
        (0, 1), (0, 2),     # 鼻子-眼睛
        (1, 3), (2, 4),     # 眼睛-耳朵
        (0, 5), (0, 6),     # 鼻子-肩膀
        (5, 7), (7, 9),     # 左臂
        (6, 8), (8, 10),    # 右臂
        (5, 11), (6, 12),   # 肩膀-臀部
        (11, 13), (13, 15), # 左腿
        (12, 14), (14, 16)  # 右腿
    ]
```

**關鍵點編號**：
```
0:  鼻子
1-2: 眼睛
3-4: 耳朵
5-6: 肩膀
7-8: 手肘
9-10: 手腕
11-12: 臀部
13-14: 膝蓋
15-16: 腳踝
```

#### 2. GraphConv - 圖卷積層

**功能**：在骨架圖上進行卷積運算，捕捉關節之間的空間關係。

```python
class GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, A):
        # A: 鄰接矩陣 (17x17)
        # 定義關節之間的連接強度

    def forward(self, x):
        # x: (N, C, T, V) = (batch, channels, time, vertices)
        # 對每個時間步驟執行圖卷積
        return graph_features
```

**原理**：
- 每個關節的特徵會受到相鄰關節的影響
- 例如：手肘的特徵會聚合來自肩膀和手腕的信息

#### 3. TemporalConv - 時間卷積層

**功能**：捕捉動作的時間動態變化。

```python
class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9):
        # kernel_size: 時間窗口大小
        # 例如 9 表示觀察前後 4 個時間步

    def forward(self, x):
        # 沿時間軸進行 1D 卷積
        return temporal_features
```

**原理**：
- 捕捉動作的時序模式
- 例如：揮手動作中手臂位置的連續變化

#### 4. AGCBlock - 自適應圖卷積區塊

**功能**：結合空間圖卷積和時間卷積。

```python
class AGCBlock(nn.Module):
    def forward(self, x):
        # 步驟 1: 空間圖卷積
        x = self.graph_conv(x)

        # 步驟 2: 時間卷積
        x = self.temporal_conv(x)

        # 步驟 3: 殘差連接
        return x + residual
```

#### 5. SkeletonEmbedding - 骨架特徵嵌入網路

**功能**：將骨架序列轉換為固定維度的特徵向量。

```python
class SkeletonEmbedding(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        # 3 通道: x, y, confidence
        # base_channels: 基礎特徵維度

        # 多層 AGCBlock
        self.blocks = nn.Sequential(
            AGCBlock(3, 64),    # 輸入層
            AGCBlock(64, 128),  # 中間層
            AGCBlock(128, 256)  # 輸出層
        )

    def forward(self, x):
        # x: (N, T, V, C) = (batch, time, vertices, channels)
        features = self.blocks(x)

        # 多尺度池化
        global_feat = global_pool(features)  # 全局平均
        max_feat = max_pool(features)        # 最大池化

        return concat(global_feat, max_feat)  # (N, 256)
```

#### 6. EMDMatcher - Earth Mover's Distance 匹配器

**功能**：計算兩個骨架序列之間的最優傳輸距離。

```python
class EMDMatcher(nn.Module):
    def forward(self, support_features, query_features):
        # support_features: 支援集特徵 (K, 256)
        # query_features: 查詢特徵 (1, 256)

        # 計算成本矩陣
        cost_matrix = pairwise_distance(support, query)

        # 使用最優傳輸求解最小匹配成本
        emd_distance = ot.emd2(uniform_dist, uniform_dist, cost_matrix)

        return similarity_score
```

**原理**：
- EMD 衡量兩個分佈之間的最小移動成本
- 相似的動作會有較小的 EMD 距離

#### 7. OneShotActionRecognition - 完整模型

**架構**：
```python
class OneShotActionRecognition(nn.Module):
    def __init__(self):
        self.embedding = SkeletonEmbedding()
        self.matcher = MultiScaleMatcher()

    def forward(self, support_set, query):
        # 步驟 1: 提取支援集特徵
        support_features = []
        for action_samples in support_set:
            features = self.embedding(action_samples)
            support_features.append(features.mean(0))

        # 步驟 2: 提取查詢特徵
        query_feature = self.embedding(query)

        # 步驟 3: 計算相似度
        similarities = self.matcher(support_features, query_feature)

        # 步驟 4: 預測類別
        predicted_class = argmax(similarities)
        return predicted_class
```

---

## 骨架動作辨識原理

### 1. 骨架表示

**輸入格式**：
```
骨架序列: (T, V, C)
- T: 時間步數 (例如 64 幀)
- V: 關節點數 (COCO 17 個關鍵點)
- C: 座標維度 (x, y, confidence)
```

**範例**：
```python
# 一個 64 幀的骨架序列
skeleton = np.array([
    # 幀 0
    [[x0, y0, conf0], [x1, y1, conf1], ..., [x16, y16, conf16]],
    # 幀 1
    [[x0, y0, conf0], [x1, y1, conf1], ..., [x16, y16, conf16]],
    ...
    # 幀 63
    [[x0, y0, conf0], [x1, y1, conf1], ..., [x16, y16, conf16]]
])  # shape: (64, 17, 3)
```

### 2. 特徵提取流程

```
步驟 1: 空間特徵提取
├─ 圖卷積捕捉關節之間的關係
├─ 例如：手臂揮動時，肩-肘-腕的協同運動
└─ 輸出: 每個時間步的空間特徵

步驟 2: 時間特徵提取
├─ 時間卷積捕捉動作的時序變化
├─ 例如：揮手動作中手臂的週期性運動
└─ 輸出: 時空特徵

步驟 3: 多尺度聚合
├─ 全局平均池化：捕捉整體動作模式
├─ 最大池化：捕捉關鍵動作瞬間
└─ 輸出: 固定長度特徵向量 (256-d)
```

### 3. One-Shot Learning 原理

**傳統方法 vs One-Shot**：

| 特性 | 傳統監督學習 | One-Shot Learning |
|------|-------------|-------------------|
| 訓練數據 | 每類需要數百個樣本 | 每類只需 1-5 個樣本 |
| 新類別 | 需要重新訓練 | 直接新增樣本即可 |
| 應用場景 | 固定類別識別 | 動態類別識別 |

**One-Shot 工作流程**：

```
步驟 1: 建立支援集 (Support Set)
├─ 每個動作類別錄製 1-5 個示範樣本
├─ 例如: 「揮手」動作錄製 3 次
└─ 儲存: support_sets/wave_hand/

步驟 2: 特徵嵌入
├─ 使用 SkeletonEmbedding 將所有樣本映射到特徵空間
├─ 支援集: [feat_1, feat_2, feat_3]
└─ 查詢樣本: feat_query

步驟 3: 相似度匹配
├─ 計算查詢樣本與每個類別支援集的相似度
├─ 使用 EMD 距離衡量相似度
└─ 選擇最相似的類別作為預測結果

步驟 4: 預測
├─ argmax(similarities)
└─ 輸出: 預測類別
```

### 4. 為什麼需要預訓練？

**未預訓練的問題**：
- 特徵提取器是隨機初始化的
- 不知道如何提取有意義的骨架特徵
- One-Shot 匹配效果差

**使用 NTU RGB+D 預訓練的好處**：
```
好處 1: 學習通用骨架特徵
├─ 在 60 種動作上訓練
├─ 學習到人體運動的基本模式
└─ 特徵提取器變得更強大

好處 2: 提升 One-Shot 效果
├─ 預訓練的特徵更具區分性
├─ 相似動作在特徵空間中距離近
└─ 不同動作在特徵空間中距離遠

好處 3: 遷移學習
├─ 即使新動作不在 NTU 60 類中
├─ 仍能提取有用的特徵
└─ One-Shot 識別準確度提升
```

---

## 訓練流程

### NTU RGB+D Dataset 預訓練

#### 數據集準備

**數據集位置**：
```
/root/catkin_ws/src/yolo_ros/nturgbd_skeletons_s001_to_s017/nturgb+d_skeletons/
```

**檢查數據集**：
```bash
docker compose exec ros-dev bash
cd /root/catkin_ws/src/yolo_ros/nturgbd_skeletons_s001_to_s017/nturgb+d_skeletons/
ls *.skeleton | wc -l  # 應該顯示 56880
```

#### 快速測試訓練（5 epochs）

**目的**：驗證訓練流程是否正常

```bash
docker compose exec ros-dev bash
cd /root/catkin_ws/src/yolo_ros/scripts

python3.10 train_ntu_rgbd.py \
    --data_path /root/catkin_ws/src/yolo_ros/nturgbd_skeletons_s001_to_s017/nturgb+d_skeletons \
    --epochs 5 \
    --batch_size 16 \
    --num_classes 60 \
    --benchmark xsub \
    --device cuda
```

**預期輸出**：
```
Loading NTU RGB+D dataset...
Cross-Subject protocol
Training samples: 40320
Validation samples: 16560

Epoch 1/5: 100%|██████████| 2520/2520 [05:30<00:00, 7.62it/s, loss=3.245]
Validation Accuracy: 15.3%
✓ Saved best model: checkpoints/best.pth

Epoch 2/5: 100%|██████████| 2520/2520 [05:28<00:00, 7.67it/s, loss=2.891]
Validation Accuracy: 22.7%
✓ Saved best model: checkpoints/best.pth

...

Training completed!
Best validation accuracy: 35.2%
Model saved to: checkpoints/best.pth
```

**訓練時間**：約 30-40 分鐘（RTX 5080）

#### 完整訓練（50 epochs）

**GPU 訓練**：
```bash
python3.10 train_ntu_rgbd.py \
    --data_path /root/catkin_ws/src/yolo_ros/nturgbd_skeletons_s001_to_s017/nturgb+d_skeletons \
    --epochs 50 \
    --batch_size 32 \
    --num_classes 60 \
    --benchmark xsub \
    --lr 0.001 \
    --device cuda \
    --num_workers 4
```

**CPU 訓練**（不推薦，僅供測試）：
```bash
python3.10 train_ntu_rgbd.py \
    --data_path /root/catkin_ws/src/yolo_ros/nturgbd_skeletons_s001_to_s017/nturgb+d_skeletons \
    --epochs 5 \
    --batch_size 8 \
    --num_classes 60 \
    --benchmark xsub \
    --device cpu \
    --num_workers 2
```

#### 背景執行訓練

```bash
cd /root/catkin_ws/src/yolo_ros/scripts

# 背景執行
nohup python3.10 train_ntu_rgbd.py \
    --data_path /root/catkin_ws/src/yolo_ros/nturgbd_skeletons_s001_to_s017/nturgb+d_skeletons \
    --epochs 50 \
    --batch_size 32 \
    --device cuda > training.log 2>&1 &

# 查看 PID
echo $!

# 查看訓練進度
tail -f training.log

# 查看最後 50 行
tail -50 training.log

# 檢查是否有錯誤
grep -i "error\|warning" training.log

# 檢查當前準確度
grep "Validation Accuracy" training.log | tail -5
```

#### Checkpoint 管理

**Checkpoint 位置**：
```
/root/catkin_ws/src/yolo_ros/scripts/checkpoints/
├── best.pth       # 最佳驗證準確度
├── latest.pth     # 最新模型
├── epoch_10.pth   # 每 10 個 epoch
├── epoch_20.pth
└── ...
```

**載入 Checkpoint**：
```python
import torch
from skeleton_model import SkeletonEmbedding

# 載入最佳模型
model = SkeletonEmbedding(in_channels=3, base_channels=64)
checkpoint = torch.load('checkpoints/best.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# 查看訓練資訊
print(f"Epoch: {checkpoint['epoch']}")
print(f"Best Accuracy: {checkpoint['best_acc']:.2f}%")
print(f"Loss: {checkpoint['loss']:.4f}")
```

---

## 測試流程

### 測試 1：骨架提取測試

**目的**：驗證 YOLOv8-Pose 能否從圖片提取骨架

**測試腳本**：`test_skeleton_from_images.py`

```bash
docker compose exec ros-dev bash
cd /root/catkin_ws/src/yolo_ros/scripts

# 測試單張圖片
python3.10 test_skeleton_from_images.py /path/to/image.jpg

# 批次測試整個目錄
python3.10 test_skeleton_from_images.py /root/catkin_ws/src/yolo_ros/test_picture/
```

**輸出**：
```
Processing: image.jpg
✓ Detected 1 person
✓ Extracted 17 keypoints
✓ Skeleton visualization saved: skeleton_output/image_skeleton.jpg
✓ Skeleton data saved: skeleton_output/image_skeleton.npy

Keypoint 0 (Nose): (245.3, 123.7), confidence: 0.95
Keypoint 1 (Left Eye): (238.1, 115.2), confidence: 0.92
...
```

### 測試 2：Dataset 載入測試

**目的**：驗證訓練腳本能否正確讀取 NTU RGB+D

**測試腳本**：`test_dataset_loading.py`

```bash
cd /root/catkin_ws/src/yolo_ros/scripts
python3.10 test_dataset_loading.py
```

**預期輸出**：
```
Testing NTU RGB+D Dataset Loading...

✓ Training set loaded: 40320 samples
✓ Validation set loaded: 16560 samples

✓ Sample loaded successfully
  - Skeleton shape: torch.Size([64, 17, 3])
  - 64 frames, 17 keypoints, 3 coordinates

✓ DataLoader works
  - Batch skeleton shape: torch.Size([4, 64, 17, 3])
  - Batch labels shape: torch.Size([4])

✓ Model inference successful
  - Input shape: torch.Size([4, 64, 17, 3])
  - Output shape: torch.Size([4, 60])

All tests passed! ✓
```

### 測試 3：One-Shot 動作辨識測試

**目的**：測試完整的 One-Shot 辨識流程

#### 步驟 1：錄製支援集

```bash
docker compose exec ros-dev bash
cd /root/catkin_ws/src/yolo_ros/scripts

# 錄製「揮手」動作示範
python3.10 record_support_set.py --action wave_hand --samples 3
```

**互動流程**：
```
Recording support set for action: wave_hand
Number of samples: 3

準備錄製樣本 1/3
按 Enter 開始錄製...
[錄製中] 請執行「揮手」動作...
✓ 樣本 1 已儲存

準備錄製樣本 2/3
按 Enter 開始錄製...
[錄製中] 請執行「揮手」動作...
✓ 樣本 2 已儲存

準備錄製樣本 3/3
按 Enter 開始錄製...
[錄製中] 請執行「揮手」動作...
✓ 樣本 3 已儲存

✓ 支援集已儲存至: support_sets/wave_hand/
```

#### 步驟 2：啟動 ROS 節點

```bash
# 啟動 RealSense 相機
roslaunch realsense2_camera rs_camera.launch

# 啟動 One-Shot 辨識節點
roslaunch yolo_ros action_recognition.launch
```

#### 步驟 3：測試辨識

**呼叫服務**：
```bash
rosservice call /recognize_action "query_duration: 3.0"
```

**預期輸出**：
```
action: "wave_hand"
confidence: 0.87
```

---

## 完整程式碼範例

### 範例 1：簡單訓練（無 Dataset）

**使用場景**：快速驗證模型架構

```python
import torch
import torch.nn as nn
from skeleton_model import SkeletonEmbedding

# 創建隨機數據
batch_size = 4
num_frames = 64
num_keypoints = 17
num_coords = 3

# 隨機骨架數據
skeletons = torch.randn(batch_size, num_frames, num_keypoints, num_coords)
labels = torch.randint(0, 60, (batch_size,))

# 創建模型
model = SkeletonEmbedding(in_channels=3, base_channels=64)
classifier = nn.Linear(256, 60)  # 60 個類別

# 訓練循環
optimizer = torch.optim.Adam(
    list(model.parameters()) + list(classifier.parameters()),
    lr=0.001
)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    # 前向傳播
    features = model(skeletons)
    logits = classifier(features)
    loss = criterion(logits, labels)

    # 反向傳播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 計算準確度
    predictions = logits.argmax(dim=1)
    accuracy = (predictions == labels).float().mean()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Acc: {accuracy.item():.2%}")

# 儲存模型
torch.save({
    'model_state_dict': model.state_dict(),
    'classifier_state_dict': classifier.state_dict()
}, 'simple_model.pth')
```

### 範例 2：使用 NTU RGB+D Dataset 訓練

**完整訓練腳本**：

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from skeleton_model import SkeletonEmbedding
from ntu_dataset import NTURGBDDataset
from tqdm import tqdm
import os

# 配置
config = {
    'data_path': '/root/catkin_ws/src/yolo_ros/nturgbd_skeletons_s001_to_s017/nturgb+d_skeletons',
    'batch_size': 32,
    'epochs': 50,
    'lr': 0.001,
    'num_classes': 60,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# 載入數據集
train_dataset = NTURGBDDataset(
    root_path=config['data_path'],
    split='train',
    benchmark='xsub'
)
val_dataset = NTURGBDDataset(
    root_path=config['data_path'],
    split='val',
    benchmark='xsub'
)

train_loader = DataLoader(
    train_dataset,
    batch_size=config['batch_size'],
    shuffle=True,
    num_workers=4
)
val_loader = DataLoader(
    val_dataset,
    batch_size=config['batch_size'],
    shuffle=False,
    num_workers=4
)

# 創建模型
model = SkeletonEmbedding(in_channels=3, base_channels=64)
classifier = nn.Linear(256, config['num_classes'])

model = model.to(config['device'])
classifier = classifier.to(config['device'])

# 優化器和損失函數
optimizer = torch.optim.Adam(
    list(model.parameters()) + list(classifier.parameters()),
    lr=config['lr']
)
criterion = nn.CrossEntropyLoss()

# 訓練循環
best_acc = 0.0
for epoch in range(config['epochs']):
    # 訓練階段
    model.train()
    classifier.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}")
    for skeletons, labels in pbar:
        skeletons = skeletons.to(config['device'])
        labels = labels.to(config['device'])

        # 前向傳播
        features = model(skeletons)
        logits = classifier(features)
        loss = criterion(logits, labels)

        # 反向傳播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 統計
        train_loss += loss.item()
        predictions = logits.argmax(dim=1)
        train_correct += (predictions == labels).sum().item()
        train_total += labels.size(0)

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    # 驗證階段
    model.eval()
    classifier.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for skeletons, labels in val_loader:
            skeletons = skeletons.to(config['device'])
            labels = labels.to(config['device'])

            features = model(skeletons)
            logits = classifier(features)
            predictions = logits.argmax(dim=1)

            val_correct += (predictions == labels).sum().item()
            val_total += labels.size(0)

    # 計算準確度
    train_acc = 100.0 * train_correct / train_total
    val_acc = 100.0 * val_correct / val_total

    print(f"Epoch {epoch+1}/{config['epochs']}")
    print(f"  Train Loss: {train_loss/len(train_loader):.4f}, Acc: {train_acc:.2f}%")
    print(f"  Val Acc: {val_acc:.2f}%")

    # 儲存最佳模型
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'classifier_state_dict': classifier.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc,
            'loss': train_loss / len(train_loader)
        }, 'checkpoints/best.pth')
        print(f"  ✓ Saved best model (acc: {best_acc:.2f}%)")

print(f"\nTraining completed!")
print(f"Best validation accuracy: {best_acc:.2f}%")
```

### 範例 3：使用預訓練模型進行 One-Shot 辨識

```python
import torch
import numpy as np
from skeleton_model import OneShotActionRecognition

# 載入預訓練模型
model = OneShotActionRecognition(in_channels=3, base_channels=64)
checkpoint = torch.load('checkpoints/best.pth')
model.embedding.load_state_dict(checkpoint['model_state_dict'], strict=False)
model.eval()
model = model.cuda()

# 載入支援集（每個動作 3 個樣本）
support_set = {
    'wave_hand': [
        np.load('support_sets/wave_hand/sample_0.npy'),
        np.load('support_sets/wave_hand/sample_1.npy'),
        np.load('support_sets/wave_hand/sample_2.npy')
    ],
    'clap': [
        np.load('support_sets/clap/sample_0.npy'),
        np.load('support_sets/clap/sample_1.npy'),
        np.load('support_sets/clap/sample_2.npy')
    ]
}

# 準備支援集張量
support_tensors = []
support_labels = []
for action_name, samples in support_set.items():
    for sample in samples:
        support_tensors.append(torch.FloatTensor(sample))
        support_labels.append(action_name)

support_tensors = torch.stack(support_tensors).cuda()

# 查詢樣本（即時錄製或從檔案載入）
query_skeleton = np.load('query_sample.npy')  # shape: (64, 17, 3)
query_tensor = torch.FloatTensor(query_skeleton).unsqueeze(0).cuda()

# 進行辨識
with torch.no_grad():
    # 提取特徵
    support_features = model.embedding(support_tensors)  # (N, 256)
    query_feature = model.embedding(query_tensor)        # (1, 256)

    # 計算相似度
    similarities = model.matcher(support_features, query_feature)

    # 預測
    predicted_idx = similarities.argmax().item()
    predicted_action = support_labels[predicted_idx]
    confidence = similarities.max().item()

    print(f"Predicted Action: {predicted_action}")
    print(f"Confidence: {confidence:.2%}")
```

### 範例 4：從攝像頭即時辨識

```python
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from collections import deque

# 載入 YOLOv8-Pose
pose_model = YOLO('yolov8n-pose.pt')

# 載入 One-Shot 模型
action_model = OneShotActionRecognition(in_channels=3, base_channels=64)
checkpoint = torch.load('checkpoints/best.pth')
action_model.embedding.load_state_dict(checkpoint['model_state_dict'], strict=False)
action_model.eval()
action_model = action_model.cuda()

# 載入支援集（省略，同範例 3）
# ...

# 骨架緩衝區（儲存最近 64 幀）
skeleton_buffer = deque(maxlen=64)

# 開啟攝像頭
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv8-Pose 骨架提取
    results = pose_model(frame, verbose=False)

    if len(results[0].keypoints) > 0:
        # 取第一個人的骨架
        keypoints = results[0].keypoints.xy[0].cpu().numpy()  # (17, 2)
        confidences = results[0].keypoints.conf[0].cpu().numpy()  # (17,)

        # 組合成 (17, 3)
        skeleton_frame = np.concatenate([
            keypoints,
            confidences.reshape(-1, 1)
        ], axis=1)

        skeleton_buffer.append(skeleton_frame)

        # 當收集到 64 幀時進行辨識
        if len(skeleton_buffer) == 64:
            skeleton_sequence = np.array(list(skeleton_buffer))  # (64, 17, 3)
            query_tensor = torch.FloatTensor(skeleton_sequence).unsqueeze(0).cuda()

            with torch.no_grad():
                query_feature = action_model.embedding(query_tensor)
                similarities = action_model.matcher(support_features, query_feature)
                predicted_idx = similarities.argmax().item()
                predicted_action = support_labels[predicted_idx]
                confidence = similarities.max().item()

            # 顯示結果
            cv2.putText(frame, f"Action: {predicted_action}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Conf: {confidence:.2%}",
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 繪製骨架
        results[0].plot()

    cv2.imshow('One-Shot Action Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## 常見問題

### Q1: 訓練時出現 "ValueError: numpy.dtype size changed"

**原因**：POT 版本與 NumPy 不相容

**解決方案**：
```bash
docker compose exec ros-dev python3.10 -m pip install --force-reinstall pot>=0.9.6
```

### Q2: GPU 訓練時記憶體不足 (OOM)

**解決方案**：
```bash
# 減少 batch size
python3.10 train_ntu_rgbd.py --batch_size 16 --device cuda

# 或減少 base_channels
python3.10 train_ntu_rgbd.py --base_channels 32 --device cuda
```

### Q3: 找不到 CUDA

**檢查步驟**：
```bash
# 1. 檢查 Host GPU
nvidia-smi

# 2. 檢查容器內 CUDA
docker compose exec ros-dev python3.10 -c "import torch; print(torch.cuda.is_available())"

# 3. 檢查 Docker Compose 配置
grep -A 5 "deploy:" ros-yolo-opencv-project3/.devcontainer/docker-compose.yml
```

### Q4: One-Shot 辨識準確度低

**改進方法**：
1. 使用 NTU RGB+D 預訓練模型
2. 錄製更多支援集樣本（5-10 個）
3. 確保動作執行清晰、幅度大
4. 增加訓練 epochs（50-100）

---

## 參考資源

1. **PyTorch 官方文檔**: https://pytorch.org/docs/stable/index.html
2. **Ultralytics YOLOv8**: https://docs.ultralytics.com/
3. **NTU RGB+D Dataset**: https://rose1.ntu.edu.sg/dataset/actionRecognition/
4. **Python Optimal Transport (POT)**: https://pythonot.github.io/
5. **ROS Noetic**: http://wiki.ros.org/noetic
