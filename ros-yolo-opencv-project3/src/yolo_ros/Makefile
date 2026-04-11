# ============================================================================
# One-Shot Action Recognition System - ROS Makefile (Container)
# ============================================================================
# This Makefile manages ROS operations INSIDE the Docker container.
# Use this after entering the container with: make shell (from host)
#
# Location: /root/catkin_ws/Makefile or /root/catkin_ws/src/yolo_ros/Makefile
# Quick start: make help
#
# Tab Completion: 在 bash 中輸入 make <TAB><TAB> 即可看到所有可用指令
# ============================================================================

# Force bash shell (needed for 'source' command)
SHELL := /bin/bash

# ROS paths
ROS_WS := /root/catkin_ws
SRC_DIR := $(ROS_WS)/src/yolo_ros
SCRIPTS_DIR := $(SRC_DIR)/scripts
ACTIONSET_DIR := $(SRC_DIR)/actionset
CHECKPOINTS_DIR := $(SCRIPTS_DIR)/checkpoints

# ROS environment
ROS_SETUP := source /opt/ros/noetic/setup.bash && source $(ROS_WS)/devel/setup.bash

# Colors
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
CYAN := \033[0;36m
MAGENTA := \033[0;35m
NC := \033[0m

# ============================================================================
# ALL TARGETS (for tab completion - 按字母排序)
# ============================================================================
.PHONY: help
.PHONY: build check-params check-services clean-actionset clean-build clean-logs
.PHONY: convert-format
.PHONY: guide-body-tracking guide-manage guide-recognize guide-record guide-train
.PHONY: inspect-model
.PHONY: kill-ros
.PHONY: list-actions list-models list-targets
.PHONY: monitor-camera monitor-gpu monitor-recognition monitor-tracking
.PHONY: start-body-tracking start-camera start-gui start-gui-full start-gui-system start-roscore start-tracker-only
.PHONY: test-camera test-gpu test-motor test-nodes test-system test-topics
.PHONY: train-mediapipe33 train-mediapipe33-fast train-mediapipe33-resume train-ntu train-ntu-fast train-ntu-resume
.PHONY: watch-topics

# ============================================================================
# HELP
# ============================================================================

help:
	@echo -e "$(BLUE)======================================================================$(NC)"
	@echo -e "$(GREEN)  One-Shot Action Recognition - ROS Makefile$(NC)"
	@echo -e "$(CYAN)  輸入 make <TAB><TAB> 可列出所有指令$(NC)"
	@echo -e "$(BLUE)======================================================================$(NC)"
	@echo -e ""
	@echo -e "$(YELLOW)  系統啟動$(NC)"
	@echo -e "  $(GREEN)make start-gui-system$(NC)     - 啟動完整 GUI 系統 (4 頁面)"
	@echo -e "  $(GREEN)make start-gui-full$(NC)       - 啟動 GUI + 馬達追蹤 (需 Arduino)"
	@echo -e "  $(GREEN)make start-gui$(NC)            - 只啟動 GUI "
	@echo -e "  $(GREEN)make start-camera$(NC)         - 啟動相機 (60 FPS / USB 3.0)"
	@echo -e "  $(GREEN)make start-roscore$(NC)        - 只啟動 ROS "
	@echo -e ""
	@echo -e "$(YELLOW)  測試與驗證$(NC)"
	@echo -e "  $(GREEN)make test-camera$(NC)          - 測試相機連接和 FPS"
	@echo -e "  $(GREEN)make test-system$(NC)          - 檢查 ROS 狀態"
	@echo -e "  $(GREEN)make test-gpu$(NC)             - 檢查 GPU 和 PyTorch"
	@echo -e "  $(GREEN)make test-topics$(NC)          - 列出所有 ROS topics"
	@echo -e "  $(GREEN)make test-nodes$(NC)           - 列出所有 ROS nodes"
	@echo -e ""
	@echo -e "$(YELLOW)  監控$(NC)"
	@echo -e "  $(GREEN)make monitor-camera$(NC)       - 監控相機 FPS"
	@echo -e "  $(GREEN)make monitor-recognition$(NC)  - 監控 辨識結果"
	@echo -e "  $(GREEN)make monitor-gpu$(NC)          - 監控 GPU 使用率"
	@echo -e "  $(GREEN)make watch-topics$(NC)         - 持續監看所有 topics"
	@echo -e ""
	@echo -e "$(YELLOW)  模型訓練$(NC)"
	@echo -e "  $(GREEN)make train-ntu$(NC)            - 訓練 NTU RGB+D (200 epochs, 8-12hr)"
	@echo -e "  $(GREEN)make train-ntu-fast$(NC)       - 快速測試訓練 (5 epochs)"
	@echo -e "  $(GREEN)make train-ntu-resume$(NC)     - 從中斷處繼續訓練"
	@echo -e "  $(GREEN)make train-mediapipe33$(NC)    - 訓練 MediaPipe 33 點 (500 epochs)"
	@echo -e "  $(GREEN)make train-mediapipe33-fast$(NC) - 快速測試 MediaPipe (10 epochs)"
	@echo -e "  $(GREEN)make train-mediapipe33-resume$(NC) - 從中斷處繼續 MediaPipe 訓練"
	@echo -e "  $(GREEN)make inspect-model$(NC)        - 檢查模型 checkpoint"
	@echo -e "  $(GREEN)make convert-format$(NC)       - 轉換骨架模型 MediaPipe -> COCO 17"
	@echo -e ""
	@echo -e "$(YELLOW)  馬達人體追蹤$(NC)"
	@echo -e "  $(GREEN)make start-body-tracking$(NC)  - 啟動人體追蹤系統 (相機+馬達+GUI)"
	@echo -e "  $(GREEN)make start-tracker-only$(NC)   - 只啟動追蹤節點 (無 GUI)"
	@echo -e "  $(GREEN)make test-motor$(NC)           - 測試馬達連接"
	@echo -e "  $(GREEN)make monitor-tracking$(NC)     - 監控追蹤狀態"
	@echo -e ""
	@echo -e "$(YELLOW)  開發與除錯$(NC)"
	@echo -e "  $(GREEN)make build$(NC)                - 編譯 ROS workspace"
	@echo -e "  $(GREEN)make clean-build$(NC)          - 清理並重新編譯"
	@echo -e "  $(GREEN)make check-params$(NC)         - 顯示 ROS parameters"
	@echo -e "  $(GREEN)make check-services$(NC)       - 列出所有 ROS services"
	@echo -e "  $(GREEN)make kill-ros$(NC)             - 停止所有 ROS 進程"
	@echo -e "  $(GREEN)make list-targets$(NC)         - 列出所有 make 目標"
	@echo -e ""
	@echo -e "$(YELLOW)  資料管理$(NC)"
	@echo -e "  $(GREEN)make list-actions$(NC)         - 列出所有已錄製動作"
	@echo -e "  $(GREEN)make list-models$(NC)          - 列出所有模型 checkpoints"
	@echo -e "  $(GREEN)make clean-logs$(NC)           - 清理 ROS log 檔案"
	@echo -e "  $(GREEN)make clean-actionset$(NC)      - 清理所有動作資料 (警告!)"
	@echo -e ""
	@echo -e "$(YELLOW)  操作流程指引$(NC)"
	@echo -e "  $(GREEN)make guide-record$(NC)         - Page 1: 錄製動作流程"
	@echo -e "  $(GREEN)make guide-manage$(NC)         - Page 2: 管理動作流程"
	@echo -e "  $(GREEN)make guide-recognize$(NC)      - Page 3: 辨識流程"
	@echo -e "  $(GREEN)make guide-body-tracking$(NC)  - Page 4: 馬達追蹤流程"
	@echo -e "  $(GREEN)make guide-train$(NC)          - 訓練模型流程"
	@echo -e ""
	@echo -e "$(BLUE)======================================================================$(NC)"
	@echo -e "  快速開始:"
	@echo -e "    $(YELLOW)make start-gui-system$(NC)     # 無馬達追蹤"
	@echo -e "    $(YELLOW)make start-gui-full$(NC)       # 有馬達追蹤"
	@echo -e ""
	@echo -e "  Tab 補全:"
	@echo -e "    輸入 $(CYAN)make st<TAB>$(NC) 會補全為 start-* 相關指令"
	@echo -e "    輸入 $(CYAN)make test-<TAB>$(NC) 會列出所有測試指令"
	@echo -e ""
	@echo -e "$(BLUE)======================================================================$(NC)"

# 列出所有可用目標（用於檢查）
list-targets:
	@echo -e "$(CYAN)所有可用的 make 目標:$(NC)"
	@echo -e ""
	@echo -e "$(YELLOW)系統啟動:$(NC)"
	@echo "  start-body-tracking start-camera start-gui start-gui-full start-gui-system start-roscore start-tracker-only"
	@echo -e ""
	@echo -e "$(YELLOW)測試:$(NC)"
	@echo "  test-camera test-gpu test-motor test-nodes test-system test-topics"
	@echo -e ""
	@echo -e "$(YELLOW)監控:$(NC)"
	@echo "  monitor-camera monitor-gpu monitor-recognition monitor-tracking watch-topics"
	@echo -e ""
	@echo -e "$(YELLOW)訓練:$(NC)"
	@echo "  train-mediapipe33 train-mediapipe33-fast train-mediapipe33-resume train-ntu train-ntu-fast train-ntu-resume"
	@echo -e ""
	@echo -e "$(YELLOW)開發:$(NC)"
	@echo "  build check-params check-services clean-actionset clean-build clean-logs convert-format inspect-model kill-ros"
	@echo -e ""
	@echo -e "$(YELLOW)資料:$(NC)"
	@echo "  list-actions list-models list-targets"
	@echo -e ""
	@echo -e "$(YELLOW)指引:$(NC)"
	@echo "  guide-body-tracking guide-manage guide-recognize guide-record guide-train"

# ============================================================================
# 系統啟動
# ============================================================================

start-gui-system:
	@echo -e "$(YELLOW)啟動完整 GUI 系統 (4 頁面)...$(NC)"
	@echo -e "$(CYAN)頁面: Data Collection + Model Management + Live Recognition + Body Tracking$(NC)"
	@echo -e "$(CYAN)視窗: QT GUI + Raw Camera + Skeleton + Recognition$(NC)"
	@$(ROS_SETUP) && roslaunch yolo_ros qt_gui_full_system.launch

start-gui-full:
	@echo -e "$(YELLOW)啟動完整 GUI 系統 + 馬達追蹤...$(NC)"
	@echo -e "$(CYAN)包含: 相機 + 骨架辨識 + YOLO追蹤 + Arduino馬達控制$(NC)"
	@echo -e "$(YELLOW)確保 Arduino 已連接到 /dev/ttyACM0$(NC)"
	@$(ROS_SETUP) && roslaunch yolo_ros qt_gui_full_system.launch enable_body_tracking:=true

start-gui:
	@echo -e "$(YELLOW)啟動 QT GUI...$(NC)"
	@$(ROS_SETUP) && cd $(SRC_DIR)/qt_gui && python3.10 main.py

start-camera:
	@echo -e "$(YELLOW)啟動 D435i 相機 (60 FPS)...$(NC)"
	@echo -e "$(YELLOW)  需要 USB 3.0 連接！$(NC)"
	@$(ROS_SETUP) && roslaunch realsense2_camera rs_camera.launch \
		enable_depth:=false \
		enable_color:=true \
		color_width:=640 \
		color_height:=480 \
		color_fps:=60

start-roscore:
	@echo -e "$(YELLOW)啟動 ROS Master...$(NC)"
	@$(ROS_SETUP) && roscore

# ============================================================================
# 測試與驗證
# ============================================================================

test-camera:
	@echo -e "$(YELLOW)測試相機連接...$(NC)"
	@echo -e ""
	@echo -e "$(CYAN)1. 檢查 USB 連接:$(NC)"
	@lsusb | grep -i intel || echo "$(RED)未找到 Intel RealSense$(NC)"
	@echo -e ""
	@echo -e "$(CYAN)2. 檢查相機 topic:$(NC)"
	@$(ROS_SETUP) && rostopic list 2>/dev/null | grep camera || echo "$(RED)未找到相機 topic (需先啟動相機)$(NC)"
	@echo -e ""
	@echo -e "$(CYAN)3. 測試影像頻率 (5 秒):$(NC)"
	@$(ROS_SETUP) && timeout 5 rostopic hz /camera/color/image_raw 2>&1 || echo "$(RED)無法測量 (需先啟動相機)$(NC)"

test-system:
	@echo -e "$(YELLOW)檢查 ROS 系統狀態...$(NC)"
	@echo -e ""
	@echo -e "$(CYAN)1. ROS Nodes:$(NC)"
	@$(ROS_SETUP) && rosnode list 2>/dev/null || echo "$(RED)ROS Master 未運行$(NC)"
	@echo -e ""
	@echo -e "$(CYAN)2. Recognition Node:$(NC)"
	@$(ROS_SETUP) && rosnode list 2>/dev/null | grep recognition || echo "$(YELLOW)Recognition node 未啟動$(NC)"
	@echo -e ""
	@echo -e "$(CYAN)3. ROS Parameters:$(NC)"
	@$(ROS_SETUP) && rosparam list 2>/dev/null | grep model_manager || echo "$(YELLOW)未設置自訂動作$(NC)"

test-gpu:
	@echo -e "$(YELLOW)測試 GPU 和 PyTorch...$(NC)"
	@python3.10 -c "import torch; print('='*60); print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('CUDA Available:', torch.cuda.is_available()); print('GPU Count:', torch.cuda.device_count()); print('GPU Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'); print('='*60)" 2>/dev/null || echo "$(RED)PyTorch 未安裝或 GPU 不可用$(NC)"

test-topics:
	@echo -e "$(CYAN)所有 ROS Topics:$(NC)"
	@$(ROS_SETUP) && rostopic list

test-nodes:
	@echo -e "$(CYAN)所有 ROS Nodes:$(NC)"
	@$(ROS_SETUP) && rosnode list

# ============================================================================
# 監控
# ============================================================================

monitor-camera:
	@echo -e "$(YELLOW)監控相機 FPS ...$(NC)"
	@$(ROS_SETUP) && rostopic hz /camera/color/image_raw

monitor-recognition:
	@echo -e "$(YELLOW)監控辨識結果 ...$(NC)"
	@$(ROS_SETUP) && rostopic echo /yolo_ros/recognition_result

monitor-gpu:
	@echo -e "$(YELLOW)監控 GPU ...$(NC)"
	@watch -n 1 nvidia-smi

watch-topics:
	@echo -e "$(YELLOW)監看所有 topics ...$(NC)"
	@while true; do \
		clear; \
		echo "$(CYAN)=== ROS Topics (更新中) ===$(NC)"; \
		$(ROS_SETUP) && rostopic list; \
		sleep 2; \
	done

# ============================================================================
# 模型訓練
# ============================================================================

# MediaPipe 33 點訓練 (使用外接硬碟資料集)
MEDIAPIPE33_DIR := $(SCRIPTS_DIR)/models/mediapipe33
MEDIAPIPE33_DATA := /media/jieling/Expansion/NTU RGB+D/nturgb+d_rgb_mediapipe/mediapipe_skeletons
MEDIAPIPE33_CHECKPOINTS := $(MEDIAPIPE33_DIR)/checkpoints_mediapipe33

train-ntu:
	@echo -e "$(YELLOW)開始訓練 NTU RGB+D 模型...$(NC)"
	@echo -e "$(MAGENTA)參數: 200 epochs, batch_size=32, lr=0.001$(NC)"
	@echo -e "$(YELLOW)預計時間: 8-12 小時$(NC)"
	@echo -e ""
	@cd $(SCRIPTS_DIR) && python3.10 train_ntu_rgbd.py \
		--data_path /root/catkin_ws/src/yolo_ros/nturgbd_skeletons_s001_to_s017/nturgb+d_skeletons \
		--epochs 200 \
		--batch_size 32 \
		--num_classes 60 \
		--benchmark xsub \
		--lr 0.001 \
		--device cuda \
		--num_workers 4

train-ntu-fast:
	@echo -e "$(YELLOW)快速訓練測試 (5 epochs)...$(NC)"
	@cd $(SCRIPTS_DIR) && python3.10 train_ntu_rgbd.py \
		--data_path /root/catkin_ws/src/yolo_ros/nturgbd_skeletons_s001_to_s017/nturgb+d_skeletons \
		--epochs 5 \
		--batch_size 16 \
		--num_classes 60 \
		--benchmark xsub \
		--device cuda

train-ntu-resume:
	@echo -e "$(YELLOW)從中斷處繼續訓練...$(NC)"
	@echo -e "$(YELLOW)請確認 checkpoints/latest.pth 存在$(NC)"
	@cd $(SCRIPTS_DIR) && python3.10 train_ntu_rgbd.py \
		--data_path /root/catkin_ws/src/yolo_ros/nturgbd_skeletons_s001_to_s017/nturgb+d_skeletons \
		--epochs 200 \
		--batch_size 32 \
		--num_classes 60 \
		--benchmark xsub \
		--lr 0.001 \
		--device cuda \
		--num_workers 4 \
		--resume $(CHECKPOINTS_DIR)/latest.pth

train-mediapipe33:
	@echo -e "$(YELLOW)開始訓練 MediaPipe 33 點模型...$(NC)"
	@echo -e "$(MAGENTA)參數: 500 epochs, batch_size=32, lr=0.001$(NC)"
	@echo -e "$(CYAN)資料集: $(MEDIAPIPE33_DATA)$(NC)"
	@echo -e ""
	@cd $(MEDIAPIPE33_DIR) && python3.10 train_mediapipe33.py \
		--data_path "$(MEDIAPIPE33_DATA)" \
		--epochs 500 \
		--batch_size 32 \
		--num_classes 60 \
		--max_frames 64 \
		--save_dir checkpoints_mediapipe33

train-mediapipe33-fast:
	@echo -e "$(YELLOW)快速測試 MediaPipe 33 點 (10 epochs)...$(NC)"
	@cd $(MEDIAPIPE33_DIR) && python3.10 train_mediapipe33.py \
		--data_path "$(MEDIAPIPE33_DATA)" \
		--epochs 10 \
		--batch_size 32 \
		--num_classes 60 \
		--max_frames 64 \
		--save_dir checkpoints_mediapipe33

train-mediapipe33-resume:
	@echo -e "$(YELLOW)從中斷處繼續訓練 MediaPipe 33...$(NC)"
	@echo -e "$(CYAN)載入: $(MEDIAPIPE33_CHECKPOINTS)/latest.pth$(NC)"
	@cd $(MEDIAPIPE33_DIR) && python3.10 train_mediapipe33.py \
		--data_path "$(MEDIAPIPE33_DATA)" \
		--epochs 500 \
		--batch_size 32 \
		--num_classes 60 \
		--max_frames 64 \
		--save_dir checkpoints_mediapipe33 \
		--resume checkpoints_mediapipe33/latest.pth

inspect-model:
	@echo -e "$(YELLOW)檢查模型 checkpoint...$(NC)"
	@cd $(SCRIPTS_DIR) && python3.10 inspect_checkpoints.py

convert-format:
	@echo -e "$(YELLOW)轉換 MediaPipe 33 → COCO 17...$(NC)"
	@cd $(SCRIPTS_DIR) && python3.10 convert_mediapipe_to_coco.py

# ============================================================================
# 開發與除錯
# ============================================================================

build:
	@echo -e "$(YELLOW)編譯 ROS workspace...$(NC)"
	@cd $(ROS_WS) && $(ROS_SETUP) && catkin_make

clean-build:
	@echo -e "$(YELLOW)清理並重新編譯...$(NC)"
	@cd $(ROS_WS) && rm -rf build/ devel/
	@cd $(ROS_WS) && $(ROS_SETUP) && catkin_make

check-params:
	@echo -e "$(CYAN)ROS Parameters:$(NC)"
	@$(ROS_SETUP) && rosparam list
	@echo -e ""
	@echo -e "$(CYAN)Custom Actions:$(NC)"
	@$(ROS_SETUP) && rosparam get /model_manager/custom_actions 2>/dev/null || echo "$(YELLOW)未設置$(NC)"

check-services:
	@echo -e "$(CYAN)ROS Services:$(NC)"
	@$(ROS_SETUP) && rosservice list

kill-ros:
	@echo -e "$(YELLOW)停止所有 ROS 進程...$(NC)"
	@pkill -9 -f "roscore|roslaunch|rosrun|rosnode" 2>/dev/null || true
	@echo -e "$(GREEN)  已停止$(NC)"

# ============================================================================
# 資料管理
# ============================================================================

list-actions:
	@echo -e "$(CYAN)已錄製的動作:$(NC)"
	@ls -1 $(ACTIONSET_DIR) 2>/dev/null || echo "$(YELLOW)無動作資料$(NC)"

list-models:
	@echo -e "$(CYAN)模型 Checkpoints:$(NC)"
	@ls -lh $(CHECKPOINTS_DIR)/*.pth 2>/dev/null || echo "$(YELLOW)無 checkpoint$(NC)"

clean-logs:
	@echo -e "$(YELLOW)清理 ROS log 檔案...$(NC)"
	@find $(ROS_WS) -name "*.log" -delete 2>/dev/null || true
	@find ~/.ros/log -type f -delete 2>/dev/null || true
	@echo -e "$(GREEN)  log 已清理$(NC)"

clean-actionset:
	@echo -e "$(RED)  Warn: 這將刪除所有已錄製的動作資料！$(NC)"
	@read -p "確定要刪除? (yes/no): " confirm; \
	if [ "$$confirm" = "yes" ]; then \
		rm -rf $(ACTIONSET_DIR)/* && \
		echo "$(GREEN)  Actionset 已清理$(NC)"; \
	else \
		echo "$(YELLOW)已取消$(NC)"; \
	fi

# ============================================================================
# 操作流程指引
# ============================================================================

guide-record:
	@echo -e "$(BLUE)======================================================================$(NC)"
	@echo -e "$(GREEN)  Page 1 - 錄製動作流程$(NC)"
	@echo -e "$(BLUE)======================================================================$(NC)"
	@echo -e ""
	@echo -e "1. 啟動系統:"
	@echo -e "   $(YELLOW)make start-gui-system$(NC)"
	@echo -e ""
	@echo -e "2. 在 Page 1 錄製:"
	@echo -e "   - 輸入檔名 (例如: wave_hand)"
	@echo -e "   - 點擊 'Start Recording'"
	@echo -e "   - 做出動作 2-3 秒"
	@echo -e "   - 點擊 'Stop Recording'"
	@echo -e ""
	@echo -e "3. 自動產生檔案:"
	@echo -e "   - {action}_video.avi"
	@echo -e "   - {action}_skeleton_sequence.npy (MediaPipe 33)"
	@echo -e "   - {action}_skeleton_sequence_coco17.npy (COCO 17)"
	@echo -e ""

guide-manage:
	@echo -e "$(BLUE)======================================================================$(NC)"
	@echo -e "$(GREEN)  Page 2 - 管理動作流程$(NC)"
	@echo -e "$(BLUE)======================================================================$(NC)"
	@echo -e ""
	@echo -e "1. 切換到 Page 2"
	@echo -e "2. 選擇模型: latest 或 best"
	@echo -e "3. 從 actionset 列表選擇動作"
	@echo -e "4. 輸入自訂動作名稱"
	@echo -e "5. 點擊 'Add Action'"
	@echo -e ""
	@echo -e "預期 log:"
	@echo -e "  [INFO]: Loaded custom actions from ROS param: ['your_action']"
	@echo -e ""

guide-recognize:
	@echo -e "$(BLUE)======================================================================$(NC)"
	@echo -e "$(GREEN)  Page 3 - 辨識流程$(NC)"
	@echo -e "$(BLUE)======================================================================$(NC)"
	@echo -e ""
	@echo -e "1. 切換到 Page 3"
	@echo -e "2. 系統自動載入 Page 2 設定的動作"
	@echo -e "3. 在相機前做出動作"
	@echo -e "4. 觀察辨識結果和信心度"
	@echo -e ""
	@echo -e "監控辨識:"
	@echo -e "  $(YELLOW)make monitor-recognition$(NC)"
	@echo -e ""

guide-train:
	@echo -e "$(BLUE)======================================================================$(NC)"
	@echo -e "$(GREEN)  訓練 NTU 模型流程$(NC)"
	@echo -e "$(BLUE)======================================================================$(NC)"
	@echo -e ""
	@echo -e "1. 確認 GPU:"
	@echo -e "   $(YELLOW)make test-gpu$(NC)"
	@echo -e ""
	@echo -e "2. 開始訓練:"
	@echo -e "   $(YELLOW)make train-ntu$(NC)          # 完整訓練 (8-12小時)"
	@echo -e "   $(YELLOW)make train-ntu-fast$(NC)     # 快速測試 (5 epochs)"
	@echo -e ""
	@echo -e "3. 監控訓練 (另一個終端):"
	@echo -e "   $(YELLOW)make monitor-gpu$(NC)"
	@echo -e "   $(YELLOW)tail -f $(SCRIPTS_DIR)/training.log$(NC)"
	@echo -e ""
	@echo -e "4. 訓練完成後:"
	@echo -e "   - 檢查 $(CHECKPOINTS_DIR)/best.pth"
	@echo -e "   - 啟用 NTU 分類 (修改 recognition_display_node_v2.py:83)"
	@echo -e "   - 重啟系統測試"
	@echo -e ""

# ============================================================================
# 馬達人體追蹤
# ============================================================================

# Arduino 設定
ARDUINO_PORT := /dev/ttyACM0
ARDUINO_BAUD := 57600

start-body-tracking:
	@echo -e "$(YELLOW)啟動人體追蹤系統...$(NC)"
	@echo -e "$(CYAN)包含: D435i 相機 + YOLO 追蹤 + 馬達控制 + GUI$(NC)"
	@echo -e ""
	@$(ROS_SETUP) && roslaunch yolo_ros body_tracking_system.launch \
		arduino_port:=$(ARDUINO_PORT) \
		arduino_baud:=$(ARDUINO_BAUD) \
		launch_gui:=true

start-tracker-only:
	@echo -e "$(YELLOW)啟動追蹤節點 (無 GUI)...$(NC)"
	@$(ROS_SETUP) && roslaunch yolo_ros body_tracking_system.launch \
		arduino_port:=$(ARDUINO_PORT) \
		arduino_baud:=$(ARDUINO_BAUD) \
		launch_gui:=false \
		launch_viewer:=true

test-motor:
	@echo -e "$(YELLOW)測試馬達連接...$(NC)"
	@echo -e ""
	@echo -e "$(CYAN)1. 檢查 Arduino 連接:$(NC)"
	@ls -la $(ARDUINO_PORT) 2>/dev/null && echo -e "$(GREEN)  Arduino 已連接$(NC)" || echo -e "$(RED)  未找到 Arduino$(NC)"
	@echo -e ""
	@echo -e "$(CYAN)2. 檢查 rosserial 節點:$(NC)"
	@$(ROS_SETUP) && rosnode list 2>/dev/null | grep -i arduino || echo -e "$(YELLOW)  Arduino 節點未啟動$(NC)"
	@echo -e ""
	@echo -e "$(CYAN)3. 測試馬達 topic:$(NC)"
	@$(ROS_SETUP) && rostopic list 2>/dev/null | grep -i stepper || echo -e "$(YELLOW)  馬達 topic 未找到$(NC)"

monitor-tracking:
	@echo -e "$(YELLOW)監控追蹤狀態...$(NC)"
	@$(ROS_SETUP) && rostopic echo /body_tracking_status

guide-body-tracking:
	@echo -e "$(BLUE)======================================================================$(NC)"
	@echo -e "$(GREEN)  Page 4 - 馬達人體追蹤系統操作指南$(NC)"
	@echo -e "$(BLUE)======================================================================$(NC)"
	@echo -e ""
	@echo -e "$(YELLOW)功能說明:$(NC)"
	@echo -e "  - YOLO 偵測人體並用馬達追蹤"
	@echo -e "  - 支援最大人體追蹤模式"
	@echo -e "  - 支援巡航模式 (36度/秒)"
	@echo -e "  - 同時進行動作辨識"
	@echo -e ""
	@echo -e "$(YELLOW)啟動步驟:$(NC)"
	@echo -e "  1. 確認 Arduino 連接: $(GREEN)make test-motor$(NC)"
	@echo -e "  2. 啟動完整系統: $(GREEN)make start-gui-full$(NC)"
	@echo -e ""
	@echo -e "$(YELLOW)Page 4 功能:$(NC)"
	@echo -e "  - [Start Tracking] 開啟人體追蹤"
	@echo -e "  - [Stop Tracking]  關閉人體追蹤"
	@echo -e "  - [Start Patrol]   啟動巡航模式"
	@echo -e "  - [Track Largest]  追蹤最大人體"
	@echo -e "  - [Home]           馬達歸位"
	@echo -e ""
	@echo -e "$(YELLOW)CPU 優化:$(NC)"
	@echo -e "  - YOLO 每 3 幀偵測一次 (降低 GPU 負載)"
	@echo -e "  - MediaPipe 骨架數據共享 (減少重複處理)"
	@echo -e ""
	@echo -e "$(YELLOW)ROS Topics:$(NC)"
	@echo -e "  - /tracking_annotated_image  追蹤畫面"
	@echo -e "  - /body_tracking_status      追蹤狀態"
	@echo -e "  - /camera/skeleton_keypoints 共享骨架數據"
	@echo -e ""
