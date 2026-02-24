# One-Shot Action Recognition System - 操作紀錄

**最後更新**: 2026-01-21
**系統版本**: 2.0
**狀態**: ✅ 系統已修復並穩定運行

---

## 🚀 快速開始 (3 步驟)

```bash
# 1. 查看所有可用指令
make help

# 2. 快速啟動完整系統
make quick-start

# 3. 測試系統狀態
make status
```

---

## 📋 目錄

1. [系統架構](#系統架構)
2. [核心功能](#核心功能)
3. [常用指令](#常用指令)
4. [完整操作流程](#完整操作流程)
5. [已修復的問題](#已修復的問題)
6. [技術規格](#技術規格)
7. [故障排除](#故障排除)
8. [訓練模型](#訓練模型)

---

## 系統架構

### 整體架構
```
┌─────────────────────────────────────────────────────────────┐
│                    QT GUI (3 Pages)                          │
│  ┌──────────┐  ┌──────────────┐  ┌─────────────────┐       │
│  │ Page 1   │  │   Page 2     │  │    Page 3       │       │
│  │ Data     │→ │   Model      │→ │  Recognition    │       │
│  │ Collection│  │  Management  │  │  (Live)         │       │
│  └──────────┘  └──────────────┘  └─────────────────┘       │
└─────────────────────────────────────────────────────────────┘
         ↓                ↓                    ↓
    ROS Topics      ROS Params         ROS Topics/Services
         ↓                ↓                    ↓
┌─────────────────────────────────────────────────────────────┐
│              ROS Noetic (Docker Container)                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Camera     │  │    Model     │  │ Recognition  │      │
│  │   Display    │→ │   Manager    │→ │   Display    │      │
│  │   Node       │  │              │  │   Node v2    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
         ↓                                       ↓
┌─────────────────┐                    ┌─────────────────┐
│  RealSense D435i│                    │  PyTorch Model  │
│  Camera         │                    │  (RTX 5080 GPU) │
│  30fps @ 640x480│                    │  CUDA 12.8      │
└─────────────────┘                    └─────────────────┘
```

### 資料流程

#### 錄製流程 (Page 1)
```
相機影像 → MediaPipe 提取 33 關鍵點 → 儲存兩種格式
                                     ├─ MediaPipe 33 (.npy)
                                     └─ COCO 17 (.npy) ✅ 自動轉換
```

#### 辨識流程 (Page 3)
```
相機影像 → MediaPipe 33 → COCO 17 → 緩衝 64 幀 → 提取特徵向量
                                                   ↓
                                          ┌────────┴────────┐
                                          ↓                 ↓
                                   One-Shot Learning   NTU RGB+D
                                   (自訂動作)          (60 類別)
                                          ↓                 ↓
                                          └────────┬────────┘
                                                   ↓
                                            合併顯示 Top 5
```

---

## 核心功能

### 1. 雙模式辨識系統

#### 模式 1: One-Shot Learning (自訂動作)
- ✅ 無需重新訓練模型
- ✅ 快速新增個人化動作
- ✅ 低延遲、高準確度
- 使用餘弦相似度比對特徵向量

#### 模式 2: NTU RGB+D Classification (60 類別)
- ⚠️ 當前狀態: 已停用 (需要訓練 200 epochs)
- 預訓練類別包括: hand waving, clapping, stop, drink water 等
- 目標準確度: 80%+

### 2. QT GUI 介面

#### Page 1 - 資料收集
- 📹 即時相機畫面顯示
- 🎬 錄製動作影片 (2-5 秒)
- 🦴 MediaPipe 骨架顯示 (33 關鍵點)
- 📸 拍照功能
- ✅ 自動格式轉換 (MediaPipe → COCO)

#### Page 2 - 模型管理
- 📁 Actionset 瀏覽和預覽
- ➕ 新增自訂動作到模型
- 🎬 骨架動畫預覽
- 🔄 自動同步到 Page 3

#### Page 3 - 即時辨識
- 🎯 雙模式同時辨識
- 📊 信心度顯示 (顏色編碼)
- 📈 Top 5 辨識結果
- 🔄 自動載入自訂動作

---

## 常用指令

### 系統啟動

```bash
# 完整系統啟動 (推薦)
make quick-start

# 或分步驟啟動
make setup-x11           # 設定 X11 權限
make container-up        # 啟動容器
make start-full-system   # 啟動完整系統 (4 個視窗)

# 只啟動 GUI
make start-gui

# 只啟動相機
make start-camera        # 30 FPS
make start-camera-60fps  # 60 FPS (需要 USB 3.0)
```

### 測試與監控

```bash
# 測試相機
make test-camera

# 測試完整系統
make test-system

# 測試 GPU
make test-gpu

# 監控相機 FPS
make monitor-camera

# 監控辨識結果
make monitor-recognition

# 查看系統狀態
make status
```

### 開發與除錯

```bash
# 進入容器 shell
make shell

# 查看 log
make logs

# 檢查 ROS parameters
make check-params

# 清理資料
make clean-logs          # 清理 log 檔案
make clean-all           # 清理所有暫存檔案
```

### 模型訓練

```bash
# 訓練 NTU RGB+D 模型 (完整訓練)
make train-ntu           # 200 epochs, 8-12 小時

# 快速測試訓練
make train-ntu-test      # 5 epochs

# 檢查模型
make inspect-model

# 轉換格式
make convert-mediapipe
```

### 停止系統

```bash
# 停止所有服務
make stop-all

# 只停止容器
make container-down

# 重啟系統
make restart-system
```

---

## 完整操作流程

### 工作流程 1: 新增自訂動作

```bash
# Step 1: 啟動系統
make quick-start

# Step 2: 在 Page 1 錄製動作
# - 輸入檔名: my_wave
# - 點擊 "Start Recording"
# - 做出揮手動作 (2-3 秒)
# - 點擊 "Stop Recording"
# ✅ 自動產生 MediaPipe 33 和 COCO 17 格式

# Step 3: 在 Page 2 新增到模型
# - 選擇模型: latest
# - 選擇動作: my_wave
# - 輸入名稱: wave
# - 點擊 "Add Action"
# ✅ 自動儲存到 ROS parameter

# Step 4: 在 Page 3 辨識
# - 切換到 Page 3
# - 系統自動載入 wave 動作
# - 做出揮手動作
# ✅ 觀察辨識結果和信心度

# Step 5: 監控結果 (可選)
make monitor-recognition
```

### 工作流程 2: 訓練 NTU 模型

```bash
# Step 1: 確認 GPU 可用
make test-gpu

# Step 2: 檢查訓練資料
make shell
cd /root/catkin_ws/src/yolo_ros/nturgbd_skeletons_s001_to_s017
ls nturgb+d_skeletons/ | wc -l  # 應該有數萬個檔案

# Step 3: 開始訓練 (需要 8-12 小時)
make train-ntu

# Step 4: 監控訓練進度 (另一個終端)
make shell
tail -f /root/catkin_ws/src/yolo_ros/scripts/training.log

# Step 5: 監控 GPU 使用率 (另一個終端)
make gpu-status

# Step 6: 訓練完成後啟用 NTU 分類
make shell
vim /root/catkin_ws/src/yolo_ros/scripts/recognition_display_node_v2.py
# 修改第 83 行: self.use_ntu_classification = True

# Step 7: 重啟系統測試
make restart-system
```

### 工作流程 3: 除錯問題

```bash
# 問題: 相機無法啟動
make test-camera         # 測試相機連接
lsusb | grep Intel       # 檢查 USB 連接

# 問題: 辨識結果一直是 0%
make check-params        # 檢查是否載入自訂動作
make test-system         # 檢查 recognition node 狀態

# 問題: GPU 無法使用
make test-gpu            # 測試 GPU
make rebuild-gpu         # 重建容器

# 問題: 視窗無法顯示
make setup-x11           # 重新設定 X11 權限

# 查看詳細 log
make logs
```

---

## 已修復的問題

### 關鍵錯誤修復 (2025-11-28)

| # | 問題 | 檔案 | 狀態 |
|---|------|------|------|
| 1 | Vector dimension mismatch (60 vs 256) | `recognition_display_node_v2.py:364-375` | ✅ 已修復 |
| 2 | NTU 分類一直顯示 "cheer up 100%" | `recognition_display_node_v2.py:83` | ✅ 已停用 |
| 3 | 信心度顯示 10000% | `page3_recognition.py:291` | ✅ 已修復 |
| 4 | 顏色閾值錯誤 | `page3_recognition.py:294-299` | ✅ 已修復 |
| 5 | One-Shot 相似度計算錯誤 | `recognition_display_node_v2.py:260-267` | ✅ 已修復 |
| 6 | 相機啟動失敗 | `qt_gui_full_system.launch:20-28` | ✅ 已修復 |

### 系統改進 (2025-11-26)

| 改進項目 | 說明 | 狀態 |
|---------|------|------|
| Actionset 自動更新 | Page 1 → Page 2 通訊機制 | ✅ 完成 |
| 預覽功能 | 支援 COCO 17 格式預覽 | ✅ 完成 |
| 自動載入 | Page 3 自動載入自訂動作 | ✅ 完成 |
| 格式轉換 | 錄製時自動轉換格式 | ✅ 完成 |

---

## 技術規格

### 硬體需求
- **GPU**: NVIDIA RTX 5080 (Blackwell 架構, sm_120)
- **相機**: Intel RealSense D435i
- **USB**: USB 3.0 (60 FPS) / USB 2.0 (30 FPS)
- **記憶體**: 16GB+ RAM 推薦
- **儲存**: 50GB+ 可用空間

### 軟體版本
- **OS**: Ubuntu 22.04 LTS
- **Docker**: 24.0+
- **ROS**: Noetic (Python 3.10)
- **PyTorch**: 2.9.1
- **CUDA**: 12.8
- **MediaPipe**: 0.10+
- **OpenCV**: 4.5+

### 模型配置
- **輸入格式**: COCO 17 關鍵點 (T, 17, 3)
- **特徵維度**: 256
- **緩衝大小**: 64 frames (約 2 秒)
- **辨識間隔**: 10 frames (約 0.33 秒)
- **One-Shot 閾值**: 0.5 餘弦相似度

### ROS Topics & Services
```bash
# Topics
/camera/color/image_raw              # 相機原始影像
/camera/skeleton_image               # MediaPipe 骨架影像
/recognition_display/output_image    # 辨識結果影像
/recognition_display/result          # 辨識結果 JSON
/yolo_ros/recognition_result         # 辨識結果字串
/actionset/updated                   # Actionset 更新通知

# Services
/camera_display/capture              # 拍照服務
/camera_display/start_recording      # 開始錄影
/camera_display/stop_recording       # 停止錄影
/recognition_display/load_actions    # 載入動作列表

# Parameters
/model_manager/custom_actions        # 自訂動作列表
/model_manager/support_features      # 動作特徵向量
```

---

## 故障排除

### 常見問題

#### 1. 相機視窗沒出現
```bash
# 檢查相機是否連接
lsusb | grep Intel

# 檢查 ROS topics
make test-topics | grep camera

# 重新啟動相機
make stop-all
make start-camera
```

#### 2. 信心度一直是 0%
```bash
# 確認已新增自訂動作
make check-params

# 確認模型已載入
make test-system

# 重新載入動作
# 在 Page 2 重新點擊 "Add Action"
```

#### 3. 骨架沒有顯示
- 確保光線充足
- 站在相機前 1-3 公尺
- 確保全身入鏡
- 檢查 MediaPipe 版本

#### 4. GPU 無法使用
```bash
# 測試 GPU
make test-gpu

# 檢查 NVIDIA 驅動
nvidia-smi

# 重建容器
make rebuild-gpu
```

#### 5. 辨識延遲太高
- 降低相機解析度 (640x480 → 320x240)
- 減少辨識間隔 (10 frames → 15 frames)
- 確認使用 GPU (device=cuda)

#### 6. Docker 容器無法啟動
```bash
# 檢查容器狀態
make container-status

# 查看錯誤 log
make logs

# 重啟容器
make container-restart

# 如果仍有問題，重建
make rebuild-gpu
```

### 錯誤訊息解析

| 錯誤訊息 | 原因 | 解決方法 |
|---------|------|---------|
| `shapes (60,) and (256,) not aligned` | 向量維度不匹配 | ✅ 已修復 (v2) |
| `OpenCV matrix error` | 相機啟動失敗 | 檢查 USB 連接和權限 |
| `timeout exceeded while waiting for service` | 服務超時 | 增加超時時間或重啟 node |
| `CUDA out of memory` | GPU 記憶體不足 | 減少 batch_size |
| `No module named 'mediapipe'` | 套件未安裝 | 重建容器 |

---

## 訓練模型

### NTU RGB+D 訓練指南

#### 訓練參數
```bash
# 完整訓練 (推薦)
--epochs 200              # 訓練輪數
--batch_size 32           # 批次大小
--num_classes 60          # 類別數量
--benchmark xsub          # Cross-Subject 基準
--lr 0.001                # 學習率
--device cuda             # 使用 GPU
--num_workers 4           # 資料載入執行緒

# 快速測試
--epochs 5                # 快速驗證
--batch_size 16           # 較小批次
```

#### 預期訓練時間
- **5 epochs**: 20-30 分鐘
- **50 epochs**: 3-4 小時
- **200 epochs**: 8-12 小時

#### 預期準確度
- **Epoch 50**: 70-75%
- **Epoch 100**: 75-82%
- **Epoch 200**: 80%+ (目標)

#### 監控訓練
```bash
# 監控 log (終端 1)
make shell
tail -f /root/catkin_ws/src/yolo_ros/scripts/training.log

# 監控 GPU (終端 2)
make gpu-status

# 查看準確度趨勢
grep "Validation Accuracy" training.log | tail -20
```

#### 訓練完成後
1. 檢查最佳模型: `checkpoints/best.pth`
2. 啟用 NTU 分類: 修改 `recognition_display_node_v2.py:83`
3. 重啟系統測試: `make restart-system`

---

## 資料格式說明

### Actionset 目錄結構
```
actionset/
├── my_action/
│   ├── my_action_video.avi                          # 影片
│   ├── my_action_skeleton_sequence.npy              # MediaPipe 33 (T, 33, 3)
│   └── my_action_skeleton_sequence_coco17.npy       # COCO 17 (T, 17, 3) ✅
├── test1126_stop_v1/
│   ├── test1126_stop_v1_video.avi
│   ├── test1126_stop_v1_skeleton_sequence.npy
│   └── test1126_stop_v1_skeleton_sequence_coco17.npy
└── ...
```

### 關鍵點格式

#### MediaPipe 33 關鍵點
- 用於: 骨架顯示 (Page 1)
- 格式: (T, 33, 3)
- 座標: [x, y, visibility]
- 包含: 臉部、身體、手部關鍵點

#### COCO 17 關鍵點
- 用於: 模型訓練和辨識
- 格式: (T, 17, 3)
- 座標: [x, y, confidence]
- 包含: 身體主要關鍵點

#### 關鍵點對應
```
COCO 17:                     MediaPipe 33:
0  - nose                    0  - nose
1  - left_eye                2  - left_eye_inner
2  - right_eye               5  - right_eye_inner
3  - left_ear                7  - left_ear
4  - right_ear               8  - right_ear
5  - left_shoulder           11 - left_shoulder
6  - right_shoulder          12 - right_shoulder
7  - left_elbow              13 - left_elbow
8  - right_elbow             14 - right_elbow
9  - left_wrist              15 - left_wrist
10 - right_wrist             16 - right_wrist
11 - left_hip                23 - left_hip
12 - right_hip               24 - right_hip
13 - left_knee               25 - left_knee
14 - right_knee              26 - right_knee
15 - left_ankle              27 - left_ankle
16 - right_ankle             28 - right_ankle
```

---

## 效能優化建議

### 相機優化
- 使用 USB 3.0 接口 (藍色)
- 降低解析度: 640x480 → 320x240
- 調整 FPS: 30 fps (標準) / 60 fps (高速)
- 停用 depth stream: `enable_depth:=false`

### 辨識優化
- 減少辨識間隔: 10 → 15 frames
- 減少自訂動作數量 (< 10 個)
- 停用 NTU 分類 (如果不需要)
- 使用 GPU: `device=cuda`

### 訓練優化
- 增加 batch_size (記憶體允許)
- 使用多執行緒: `num_workers=4`
- 使用混合精度訓練 (未實作)
- 使用 tensorboard 監控 (未實作)

---

## 重要檔案位置

### 核心程式碼
```
src/yolo_ros/
├── scripts/
│   ├── skeleton_model.py                      # One-Shot 模型定義
│   ├── ntu_rgbd_classes.py                    # NTU 60 類別定義
│   ├── camera_display_node.py                 # 相機節點
│   ├── recognition_display_node_v2.py         # 辨識節點 v2
│   ├── train_ntu_rgbd.py                      # NTU 訓練腳本
│   ├── convert_mediapipe_to_coco.py           # 格式轉換
│   ├── inspect_checkpoints.py                 # 模型檢查
│   └── checkpoints/
│       ├── best.pth                           # 最佳模型
│       └── latest.pth                         # 最新模型
├── qt_gui/
│   ├── main.py                                # GUI 主程式
│   ├── pages/
│   │   ├── page1_data_collection.py           # Page 1
│   │   ├── page2_model_management.py          # Page 2
│   │   └── page3_recognition.py               # Page 3
│   └── utils/
│       └── model_manager.py                   # 模型管理器
└── launch/
    ├── qt_gui.launch                          # GUI 啟動檔
    └── qt_gui_full_system.launch              # 完整系統啟動檔
```

### 文件
```
/home/jieling/Desktop/workspace/ObjectRecognition/ros-yolo-opencv-project3/
├── Makefile                                   # 🆕 所有指令集合
├── OPERATIONS_RECORD.md                       # 🆕 本檔案
├── TECHNICAL_GUIDE.md                         # 完整技術指南 (1028 行)
├── SYSTEM_STATUS_2025-11-28.md                # 系統狀態和修復記錄
├── README.md                                  # 專案說明
└── .devcontainer/
    ├── docker-compose.yml                     # Docker 配置
    └── Dockerfile                             # Container 定義
```

---

## 下一步建議

### 短期 (完成)
- ✅ 測試所有修復功能
- ✅ 建立 Makefile 指令集
- ✅ 整合文件

### 中期
- 收集更多自訂動作資料 (2-5 秒影片)
- 訓練 NTU RGB+D 模型至 80%+ 準確度
- 優化辨識速度和準確度
- 加入動作統計和日誌功能

### 長期
- 支援多人同時辨識
- 整合機器人控制指令
- 訓練專屬領域的動作分類器
- 支援完整 MediaPipe 33 關鍵點

---

## 參考文件

### 必讀文件 (保留)
1. **OPERATIONS_RECORD.md** (本檔案) - 完整操作指南
2. **TECHNICAL_GUIDE.md** - 詳細技術文件
3. **SYSTEM_STATUS_2025-11-28.md** - 系統狀態和已修復問題
4. **README.md** - 專案概述

### 歷史文件 (可歸檔)
- QT_GUI_DEVELOPMENT.md - GUI 開發歷史
- DUAL_MODE_RECOGNITION_SYSTEM.md - 雙模式系統說明
- CRITICAL_FIX_2025-11-28.md - 特定 bug 修復
- FIXES_2025-11-26.md - 特定版本修復
- MEDIAPIPE_TO_COCO_CONVERSION.md - 格式轉換技術
- SESSION_SUMMARY_*.md - 開發記錄
- 其他 troubleshooting 文件

---

## 聯絡與支援

### 取得幫助
```bash
# 查看所有指令
make help

# 查看操作範例
make ops-full-workflow

# 查看系統狀態
make status

# 進入除錯模式
make shell
```

### 報告問題
1. 執行 `make status` 收集系統資訊
2. 執行 `make logs` 查看錯誤 log
3. 記錄重現步驟
4. 查看 SYSTEM_STATUS_2025-11-28.md 中是否有相關修復

---

**建立時間**: 2026-01-21
**版本**: 2.0
**維護者**: Project Team
**狀態**: ✅ 系統穩定運行

**快速指令**: `make help` | `make quick-start` | `make status`
