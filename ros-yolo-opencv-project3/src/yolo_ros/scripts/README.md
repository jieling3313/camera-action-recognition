# Scripts 目錄結構說明

## 目錄架構

```
scripts/
├── models/                    # 模型架構與訓練
│   ├── legacy/               # 舊版 COCO 17 點模型
│   ├── ntu25/                # NTU 25 點模型 (Azure Kinect DK)
│   ├── mediapipe33/          # MediaPipe 33 點模型 (D435i)
│   └── common/               # 共用模組
├── ros_nodes/                 # ROS 節點
├── data_processing/           # 資料處理工具
├── tools/                     # 測試與檢查工具
├── checkpoints/               # 模型權重檔案
└── qt_gui_main.py            # Qt GUI 啟動器
```

---

## models/ - 模型架構

### legacy/ - 舊版 COCO 17 點
| 檔案 | 說明 |
|------|------|
| `skeleton_model.py` | COCO 17 點 One-Shot 模型 |
| `skeleton_extractor.py` | YOLOv8-Pose 骨架提取器 |
| `train_ntu_rgbd.py` | 訓練腳本 (NTU→COCO 映射，有資訊損失) |

**注意**: 此格式將 NTU 25 點映射到 COCO 17 點，會損失 8 個關節資訊。

### ntu25/ - NTU 25 點 (推薦用於 Azure Kinect DK)
| 檔案 | 說明 |
|------|------|
| `skeleton_model_ntu25.py` | NTU 25 點 One-Shot 模型 |
| `skeleton_extractor_azure_kinect.py` | Azure Kinect DK 骨架提取器 (32→25點) |
| `train_ntu25.py` | 訓練腳本 (待建立) |

**優點**: 直接相容 NTU RGB+D 訓練資料，無資訊損失。

### mediapipe33/ - MediaPipe 33 點 (推薦用於 D435i)
| 檔案 | 說明 |
|------|------|
| `skeleton_model_mediapipe33.py` | MediaPipe 33 點 One-Shot 模型 |
| `extract_mediapipe_from_ntu.py` | 從 NTU RGB 影片提取 MediaPipe 骨架 |
| `train_mediapipe33.py` | 訓練腳本 (待建立) |

**優點**: 包含完整手指和腳部追蹤，適合 D435i + MediaPipe。

### common/ - 共用模組
| 檔案 | 說明 |
|------|------|
| `ntu_rgbd_classes.py` | NTU RGB+D 60 動作類別定義 |
| `mediapipe_compat.py` | MediaPipe 新版 API 兼容層 |
| `convert_mediapipe_to_coco.py` | 格式轉換工具 |

---

## ros_nodes/ - ROS 節點

| 檔案 | 說明 |
|------|------|
| `camera_display_node.py` | D435i 相機顯示 + MediaPipe 骨架 |
| `one_shot_action_node.py` | One-Shot 動作辨識節點 |
| `recognition_display_node.py` | 辨識結果顯示 (v1) |
| `recognition_display_node_v2.py` | 辨識結果顯示 (v2, 雙模式) |
| `yolo_subscriber.py` | YOLO 偵測訂閱者 |
| `yolo_unified_node.py` | YOLO 統一節點 |
| `object_selector_server.py` | 物件選擇服務 |
| `interactive_object_selector.py` | 互動式物件選擇 |
| `get_object_position.py` | 取得 3D 物件位置 |

---

## data_processing/ - 資料處理

| 檔案 | 說明 |
|------|------|
| `process_test_pictures.py` | 處理測試圖片 |
| `organize_skeleton_output.py` | 組織骨架輸出 |
| `record_support_set.py` | 錄製 Support Set |

---

## tools/ - 工具

| 檔案 | 說明 |
|------|------|
| `inspect_checkpoints.py` | 檢查模型 checkpoint |
| `test_dataset_loading.py` | 測試資料集載入 |
| `test_skeleton_from_images.py` | 從圖片測試骨架提取 |

---

## 使用方式

### 匯入模型
```python
# NTU 25 點模型 (Azure Kinect DK)
from models.ntu25 import OneShotActionRecognitionNTU25, AzureKinectSkeletonExtractor

# MediaPipe 33 點模型 (D435i)
from models.mediapipe33 import OneShotActionRecognitionMediaPipe

# 舊版 COCO 17 點模型
from models.legacy import OneShotActionRecognition, SkeletonExtractor

# 共用工具
from models.common import NTU_RGBD_60_CLASSES
```

### 執行 ROS 節點
```bash
rosrun yolo_ros camera_display_node.py
rosrun yolo_ros one_shot_action_node.py
```

---

## 雙軌並行方案

| 路線 | 硬體 | 模型目錄 | 狀態 |
|------|------|----------|------|
| **路線 A** | Azure Kinect DK | `models/ntu25/` | 可直接使用 |
| **路線 B** | D435i + MediaPipe | `models/mediapipe33/` | 需下載 NTU RGB 影片 |

---

*最後更新: 2026-03-23*
