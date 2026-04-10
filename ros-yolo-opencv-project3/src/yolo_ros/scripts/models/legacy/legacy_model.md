# One-Shot Action Recognition
#### [第四季報_第三年部分_1128.pptx](https://docs.google.com/presentation/d/1QrL2CUkElyvwMfcFuTXpB5fSdpDqCqLI/edit?usp=drive_link&ouid=104306470118628608106&rtpof=true&sd=true)
#### [完整程式碼](https://github.com/jieling3313/camera-action-recognition)
> github [jieling3313](https://github.com/jieling3313)

## 一、目前進度
- [x] 整理docker 環境內容
- [x] 完善hackmd 使用教學
- [x] 上傳完整程式碼至github
* 嘗試完成: 攝影機yolo人體追蹤
    - [x] 軟體 將攝影機機構(yolo追蹤人體->馬達控制)程式碼整合至目前docker環境上
    - [ ] 硬體 SolidWorks繪製360度辨識攝影機3D列印模型
* 人體姿態: 模型辨識
    - [ ] 提高epoch至500or更高

## 二、整體架構
```
┌─────────────────────────────────────────────────────────┐
│                    QT GUI (3 Pages)                     │
│  ┌───────────┐  ┌──────────────┐  ┌─────────────────┐   │
│  │ Page 1    │  │   Page 2     │  │    Page 3       │   │
│  │ Data      │→ │   Model      │→ │  Recognition    │   │
│  │ Collection│  │  Management  │  │  (Live)         │   │
│  └───────────┘  └──────────────┘  └─────────────────┘   │
└─────────────────────────────────────────────────────────┘
         ↓                ↓                  ↓
    ROS Topics      ROS Params       ROS Topics/Services
         ↓                ↓                  ↓
┌─────────────────────────────────────────────────────────┐
│              ROS Noetic (Docker Container)              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │   Camera     │  │    Model     │  │ Recognition  │   │
│  │   Display    │→ │   Manager    │→ │   Display    │   │
│  │   Node       │  │              │  │   Node v2    │   │
│  └──────────────┘  └──────────────┘  └──────────────┘   │
└─────────────────────────────────────────────────────────┘
         ↓                                      ↓
┌───────────────────┐                  ┌──────────────────┐
│  RealSense D435i  │                  │  PyTorch Model   │
│  Camera           │                  │  (RTX 5080 GPU)  │
│  30fps @ 640x480  │                  │  CUDA 12.8       │
└───────────────────┘                  └──────────────────┘
```
### 檔案結構
```
.
|-- client
|   `-- src
|       `-- yolo_ros  <-- (基本 ROS 節點原始碼)
|           |-- scripts
|           `-- src
|
`-- ros-yolo-opencv-project3  <-- (主要專案核心)
    `-- src
        `-- yolo_ros
            |-- actionset       <-- (動作數據/Dataset相關)
            |
            |-- nturgbd_skeletons_s001_to_s017  <-- (訓練用資料集 Source)
            |   `-- nturgb+d_skeletons
            |
            |-- qt_gui          <-- [GUI 核心程式碼]
            |   |-- pages
            |   |-- utils
            |   `-- widgets
            |
            |-- scripts
            |   |-- checkpoints     <-- [模型訓練] 權重存檔位置 (.pt/.pth)
            |   `-- skeleton_output <-- [模型輸出] 推論結果數據
            |
            |-- support_sets    <-- [模型訓練] 支援集 (Few-shot learning 用)
            `-- motor_control
```
### gui 界面
![image](https://hackmd.io/_uploads/B1ITQBwPZg.png =800x)
![image](https://hackmd.io/_uploads/ry0aQHDPbe.png =800x)
![image](https://hackmd.io/_uploads/rk4AXBvvZg.png =800x)


## 三、環境建置(for RTX50xx)

### GPU 環境建置
#### 1. 安裝 NVIDIA Container Toolkit
#### 2. 建置 Docker 容器
```bash
cd .devcontainer
docker compose build
docker compose up -d
```
#### 3. 驗證 GPU

```bash
nvidia-smi

# 檢查容器內 GPU
docker compose exec ros-dev python3.10 -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA:', torch.version.cuda)
print('GPU Available:', torch.cuda.is_available())
print('GPU Name:', torch.cuda.get_device_name(0))
print('GPU Memory:', torch.cuda.get_device_properties(0).total_memory / 1024**3, 'GB')
"
```

**Output**：
```
PyTorch: 2.9.1+cu128
CUDA: 12.8
GPU Available: True
GPU Name: NVIDIA GeForce RTX 5080 Laptop GPU
GPU Memory: 16.0 GB
```

---

## 四、啟動
### 懶人版
- **查看makefile**
`ros-yolo-opencv-project3/Makefile`
`ros-yolo-opencv-project3/src/yolo_ros/Makefile`
- 進入以下終端位置後輸入 `make help` 查看相關指令
    * docker container 環境:`ros-yolo-opencv-project3/`
    * container 內部環境:`root/catkin_ws/src/yolo_ros/`
### 完整系統啟動

**啟動指令**：
```bash
source /opt/ros/noetic/setup.bash
source /root/catkin_ws/devel/setup.bash
roslaunch yolo_ros qt_gui_full_system.launch
```

**包含的組件**：
- RealSense D435i 相機節點
- MediaPipe 骨架處理節點
- 雙模式辨識節點（One-Shot + NTU RGB+D）
- QT GUI 介面
- 3 個影像顯示視窗
---

### 分開啟動組件
#### 啟動相機
```bash
source /opt/ros/noetic/setup.bash
roslaunch realsense2_camera rs_camera.launch \
    enable_depth:=false \
    enable_infra:=false \
    color_width:=640 \
    color_height:=480 \
    color_fps:=30
```

**預期輸出**：
```
[INFO] color stream is enabled - width: 640, height: 480, fps: 30
[INFO] RealSense Node Is Up!
```

#### 啟動骨架處理節點

```bash
source /opt/ros/noetic/setup.bash
source /root/catkin_ws/devel/setup.bash
rosrun yolo_ros camera_display_node.py
```

#### 啟動辨識節點

```bash
source /opt/ros/noetic/setup.bash
source /root/catkin_ws/devel/setup.bash
rosrun yolo_ros recognition_display_node_v2.py
```

#### 啟動 GUI

```bash
source /opt/ros/noetic/setup.bash
source /root/catkin_ws/devel/setup.bash
rosrun yolo_ros qt_gui_main.py
```

#### 啟動影像視窗

```bash
# Raw Camera
rosrun image_view image_view image:=/camera/color/image_raw

# Skeleton
rosrun image_view image_view image:=/camera/skeleton_image

# Recognition Results
rosrun image_view image_view image:=/recognition_display/output_image
```
---

## 五、訓練流程
### NTU RGB+D Dataset 預訓練
#### 數據集準備
* 到 [NTU RGB+D Dataset ](https://rose1.ntu.edu.sg/dataset/actionRecognition/)下載dataset至下方位置

**檢查數據集**：
```bash
cd /root/catkin_ws/src/yolo_ros/nturgbd_skeletons_s001_to_s017/nturgb+d_skeletons
ls *.skeleton | wc -l  # 應該顯示 56880
```

### GPU訓練（200 epochs）
```bash
cd /root/catkin_ws/src/yolo_ros/scripts

python3.10 train_ntu_rgbd.py \
    --data_path /root/catkin_ws/src/yolo_ros/nturgbd_skeletons_s001_to_s017/nturgb+d_skeletons \
    --epochs 200 \
    --batch_size 32 \
    --num_classes 60 \
    --benchmark xsub \
    --lr 0.001 \
    --device cuda \
    --num_workers 4 \
    --save_dir checkpoints
```

---

### 背景執行訓練

```bash
cd /root/catkin_ws/src/yolo_ros/scripts

# 使用 nohup 在背景執行
nohup python3.10 train_ntu_rgbd.py \
    --data_path /root/catkin_ws/src/yolo_ros/nturgbd_skeletons_s001_to_s017/nturgb+d_skeletons \
    --epochs 200 \
    --batch_size 32 \
    --num_classes 60 \
    --benchmark xsub \
    --lr 0.001 \
    --device cuda \
    --num_workers 4 > training.log 2>&1 &

# 記錄 PID
echo $!  # 顯示進程 ID

# 查看訓練進度
tail -f training.log

# 查看最後 50 行
tail -50 training.log
```

---

### Checkpoint 管理

**Checkpoint 位置**：
```
/root/catkin_ws/src/yolo_ros/scripts/checkpoints/
├── best.pth       # 最佳驗證準確度的模型
├── latest.pth     # 最新的模型
├── epoch_10.pth   # 每 10 個 epoch 保存
├── epoch_20.pth
└── ...
```

**檢查 Checkpoint 資訊**：
```bash
cd /root/catkin_ws/src/yolo_ros/scripts

python3.10 inspect_checkpoints.py
```

**預期輸出**：
```
Checkpoint: checkpoints/best.pth
  Epoch: 127
  Best Accuracy: 82.35%
  Loss: 0.4521
  Model trained with 60 classes (NTU RGB+D)
```

**載入特定 Checkpoint**：
```python
import torch
from skeleton_model import SkeletonEmbedding

model = SkeletonEmbedding(in_channels=3, base_channels=64, num_classes=60)
checkpoint = torch.load('checkpoints/epoch_100.pth')
model.load_state_dict(checkpoint['model_state_dict'])

print(f"Loaded model from Epoch {checkpoint['epoch']}")
print(f"Validation Accuracy: {checkpoint['best_acc']:.2f}%")
```

---

## 六、系統狀態檢查

### 檢查 ROS 節點

```bash
# 列出所有運行中的節點
rosnode list

# 預期輸出
/camera/realsense2_camera
/camera_display_node
/recognition_display_node_v2
/action_recognition_gui
/image_view_raw
/image_view_skeleton
/image_view_recognition
```

**檢查特定節點狀態**：
```bash
rosnode info /recognition_display_node_v2
```

---

### 檢查 ROS Topics

```bash
# 列出所有 topics
rostopic list

# 檢查相機影像頻率
rostopic hz /camera/color/image_raw

# 預期輸出
average rate: 30.xxx
	min: 0.029s max: 0.035s std dev: 0.00214s

# 檢查骨架影像頻率
rostopic hz /camera/skeleton_image

# 檢查辨識結果頻率
rostopic hz /recognition_display/output_image
```

---

### 檢查 ROS Services

```bash
# 列出所有服務
rosservice list

# 檢查 camera_display 服務
rosservice list | grep camera_display

# 預期輸出
/camera_display/capture
/camera_display/start_recording
/camera_display/stop_recording

# 測試拍照服務
rosservice call /camera_display/capture "filename: 'test_capture'"

# 測試載入動作服務
rosservice call /recognition_display/load_actions
```

---

### 查看自訂動作列表
```bash
# 查看自訂動作列表
rosparam get /model_manager/custom_actions

# 預期輸出
- forward
- stop
- left
- right

# 查看動作特徵（部分）
rosparam get /model_manager/support_features

# 查看所有 parameters
rosparam list
```

---

### 相機測試

**使用 ROS image_view**：
```bash
# 測試原始影像
rosrun image_view image_view image:=/camera/color/image_raw

# 測試骨架影像
rosrun image_view image_view image:=/camera/skeleton_image
```

**使用 rqt_image_view（圖形介面）**：
```bash
rqt_image_view
```

---

### GUI 功能測試

#### Page 1：資料收集測試
![image](https://hackmd.io/_uploads/B1ITQBwPZg.png =800x)

**測試拍照功能**：
```bash
# 在 GUI 中：
# 1. 輸入檔名：test_capture_$(date +%Y%m%d_%H%M%S)
# 2. 點擊 "Capture"
# 3. 檢查檔案

ls -la /root/catkin_ws/src/yolo_ros/actionset/test_capture*/
```

**預期檔案**：
```
test_capture_20251128_114530/
├── test_capture_20251128_114530_raw.jpg
├── test_capture_20251128_114530_skeleton.jpg
└── test_capture_20251128_114530_skeleton.npy
```

**測試錄影功能**：
```bash
# 在 GUI 中：
# 1. 輸入檔名：test_video
# 2. 點擊 "Start Recording"（按鈕變綠）
# 3. 執行動作 2-3 秒
# 4. 點擊 "Stop Recording"
# 5. 檢查檔案

ls -la /root/catkin_ws/src/yolo_ros/actionset/test_video/
```

**預期檔案**：
```
test_video/
├── test_video_video.avi                          # 影片
├── test_video_skeleton_sequence.npy              # MediaPipe 33 關鍵點
└── test_video_skeleton_sequence_coco17.npy       # COCO 17 關鍵點（自動轉換）
```

**檢查骨架序列格式**：
```bash
python3.10 -c "
import numpy as np

# MediaPipe 33 格式
mp_data = np.load('/root/catkin_ws/src/yolo_ros/actionset/test_video/test_video_skeleton_sequence.npy')
print(f'MediaPipe shape: {mp_data.shape}')  # (frames, 33, 3)

# COCO 17 格式
coco_data = np.load('/root/catkin_ws/src/yolo_ros/actionset/test_video/test_video_skeleton_sequence_coco17.npy')
print(f'COCO 17 shape: {coco_data.shape}')  # (frames, 17, 3)
print(f'Frames recorded: {coco_data.shape[0]}')
"
```

---

#### Page 2：模型管理測試
![image](https://hackmd.io/_uploads/ry0aQHDPbe.png =800x)

**測試模型載入**：
```bash
# 在 GUI Page 2 中：
# 1. 左側選擇模型：best
# 2. 觀察模型資訊

# 檢查 log
grep "Successfully loaded model" ~/.ros/log/*/action_recognition_gui-*.log
```

**測試動作新增**：
```bash
# 在 GUI Page 2 中：
# 1. 中間選擇 actionset：test_video
# 2. 輸入動作名稱：my_test_action
# 3. 點擊 "Add Action"
# 4. 檢查 ROS parameter

rosparam get /model_manager/custom_actions
# 應該包含 "my_test_action"
```

**測試動作預覽**：
```bash
# 在 GUI Page 2 中：
# 1. 點擊 Current Action List 中的動作
# 2. 右側應該顯示骨架動畫
# 3. 使用 Play/Pause 控制播放
```

---

#### Page 3：辨識測試
![image](https://hackmd.io/_uploads/rk4AXBvvZg.png =800x)
**測試自動載入**：
```bash
# 系統啟動後，切換到 Page 3
# 觀察 log

grep "Auto-loaded" ~/.ros/log/*/recognition_display_node_v2-*.log

# 預期輸出
Auto-loaded 3 custom actions: forward, stop, my_test_action
```

**測試手動載入**：
```bash
# 在 GUI Page 3 中：
# 點擊 "Load Actions"

# 觀察彈出訊息框
# "Loaded 3 custom actions: forward, stop, my_test_action"
```

**測試即時辨識**：
```bash
# 在相機前做出動作
# 觀察 GUI 和 Recognition Results 視窗

# 檢查辨識結果 topic
rostopic echo /recognition_display/result
```

**預期輸出**：
```
data: "forward,75.32"
---
data: "forward,78.15"
---
data: "stop,82.45"
---
```

---


## 七、人體姿態辨識-模型架構

### SkeletonEmbedding 網路結構

```
輸入: (N, T, V, C) = (batch, 64, 17, 3)
  ↓
Data BatchNorm
  ↓
共享 AGC 區塊 (1-6)
  ├─ Block 1: 3 → 64
  ├─ Block 2-3: 64 → 64
  ├─ Block 4: 64 → 128 (stride=2)
  └─ Block 5-6: 128 → 128
  ↓
多尺度處理
  ├─ Scale 1 (關節): 128 → 256
  ├─ Scale 2 (部位): 128 → 256
  └─ Scale 3 (肢體): 128 → 256
  ↓
全域平均池化
  ↓
特徵向量: (N, 256)
  ↓
分類器 (可選): 256 → 60
  ↓
輸出: (N, 60) 或 (N, 256)
```
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

### 1. COCOGraph - COCO 17 關鍵點圖結構

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
![image](https://hackmd.io/_uploads/rJhXqrvvWx.png =800x)

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

### 2. GraphConv - 圖卷積層

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

### 3. TemporalConv - 時間卷積層

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

### 4. AGCBlock - 自適應圖卷積區塊

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

### 5. SkeletonEmbedding - 骨架特徵嵌入網路

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

### 6. EMDMatcher - Earth Mover's Distance 匹配器

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

### 7. OneShotActionRecognition - 完整模型

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

## 八、人體姿態辨識-骨架動作辨識原理

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

| 特性     | 傳統監督學習       | One-Shot Learning   |
|:------- | ------------------ |:------------------- |
| 訓練數據 | 每類需要數百個樣本 | 每類只需 1-5 個樣本 |
| 新類別   | 需要重新訓練       | 直接新增樣本即可    |
| 應用場景 | 固定類別識別       | 動態類別識別        |

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


---

## *指令

### Docker 相關

```bash
# 建置容器
docker compose build

# 啟動容器
docker compose up -d

# 停止容器
docker compose down

# 重啟容器
docker compose restart

# 進入容器
docker compose exec ros-dev bash

# 查看容器 log
docker compose logs -f ros-dev

# 查看容器資源使用
docker stats ros-dev
```

---

### ROS 相關

```bash
# 設定環境
source /opt/ros/noetic/setup.bash
source /root/catkin_ws/devel/setup.bash

# 啟動 roscore
roscore

# 列出節點
rosnode list

# 列出 topics
rostopic list

# 列出服務
rosservice list

# 查看 topic 頻率
rostopic hz <topic_name>

# 查看 topic 內容
rostopic echo <topic_name>

# 呼叫服務
rosservice call <service_name>

# 設定 parameter
rosparam set <param_name> <value>

# 查看 parameter
rosparam get <param_name>
```

---

### 訓練相關

```bash
# 快速測試（5 epochs）
python3.10 train_ntu_rgbd.py \
    --data_path /root/catkin_ws/src/yolo_ros/nturgbd_skeletons_s001_to_s017/nturgb+d_skeletons \
    --epochs 5 \
    --batch_size 16 \
    --device cuda

# 完整訓練（200 epochs）
python3.10 train_ntu_rgbd.py \
    --data_path /root/catkin_ws/src/yolo_ros/nturgbd_skeletons_s001_to_s017/nturgb+d_skeletons \
    --epochs 200 \
    --batch_size 32 \
    --device cuda

# 背景訓練
nohup python3.10 train_ntu_rgbd.py \
    --data_path /root/catkin_ws/src/yolo_ros/nturgbd_skeletons_s001_to_s017/nturgb+d_skeletons \
    --epochs 200 \
    --batch_size 32 \
    --device cuda > training.log 2>&1 &

# 查看訓練進度
tail -f training.log
grep "Validation Accuracy" training.log | tail -20

# 檢查 checkpoint
python3.10 inspect_checkpoints.py
```

---

### 系統測試相關

```bash
# 測試相機
rs-enumerate-devices
rosrun image_view image_view image:=/camera/color/image_raw

# 測試骨架處理
rostopic hz /camera/skeleton_image
rosrun image_view image_view image:=/camera/skeleton_image

# 測試辨識
rostopic echo /recognition_display/result
rosrun image_view image_view image:=/recognition_display/output_image

# 測試服務
rosservice call /camera_display/capture "filename: 'test'"
rosservice call /recognition_display/load_actions

# 檢查系統狀態
rosnode list
rostopic list
rosservice list
rosparam list
```

---

### GPU 相關

```bash
# 查看 GPU 狀態
nvidia-smi

# 持續監控 GPU
watch -n 1 nvidia-smi

# 查看 CUDA 版本
nvcc --version

# 測試 PyTorch GPU
python3.10 -c "import torch; print(torch.cuda.is_available())"
```

---

## *除錯

### Bug 1. 相機無法啟動

**症狀**：
```
[ERROR] No RealSense devices were found!
```

**檢查步驟**：
```bash
# 1. 檢查 USB 連接
lsusb | grep Intel

# 2. 檢查設備權限
ls -la /dev/video*

# 3. 重新插拔相機
# 4. 重啟 Docker 容器
docker compose restart
```

---

### Bug 2. GUI 沒有顯示影像

**檢查 topic 數據流**：
```bash
rostopic hz /camera/color/image_raw
```

**如果顯示 "no new messages"**：
```bash
# 1. 檢查相機節點
rosnode list | grep camera

# 2. 檢查節點狀態
rosnode info /camera/realsense2_camera

# 3. 查看錯誤訊息
rosnode log /camera/realsense2_camera
```

---

### Bug 3. 辨識結果一直顯示相同動作

**檢查 log**：
```bash
grep "ERROR\|WARNING" ~/.ros/log/*/recognition_display_node_v2-*.log
```

**常見錯誤**：
```
[ERROR]: Image callback error: shapes (60,) and (256,) not aligned
```

**解決方案**：參考 `CRITICAL_FIX_2025-11-28.md`

---

### Bug 4. 訓練時 GPU 記憶體不足

**症狀**：
```
RuntimeError: CUDA out of memory
```

**解決方案**：
```bash
# 減少 batch size
python3.10 train_ntu_rgbd.py \
    --batch_size 16 \  # 改為 16 或 8
    --device cuda
```


---
## *參考資源

### 官方文檔

1. **PyTorch**: https://pytorch.org/docs/stable/
2. **ROS Noetic**: http://wiki.ros.org/noetic
3. **RealSense SDK**: https://dev.intelrealsense.com/
4. **MediaPipe**: https://google.github.io/mediapipe/
5. **YOLOv8**: https://docs.ultralytics.com/


### NTU RGB+D Dataset

- **論文**: "NTU RGB+D: A Large Scale Dataset for 3D Human Activity Analysis"
- **下載**: https://rose1.ntu.edu.sg/dataset/actionRecognition/
- **類別數**: 60 個動作類別
- **樣本數**: 56,880 個影片

---
