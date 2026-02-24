# ============================================================================
# One-Shot Action Recognition System - ROS Makefile (Container)
# ============================================================================
# This Makefile manages ROS operations INSIDE the Docker container.
# Use this after entering the container with: make shell (from host)
#
# Location: /root/catkin_ws/Makefile or /root/catkin_ws/src/yolo_ros/Makefile
# Quick start: make help
# ============================================================================

# Force bash shell (needed for 'source' command)
SHELL := /bin/bash

# ROS paths
ROS_WS := /root/catkin_ws
SRC_DIR := $(ROS_WS)/src/yolo_ros
SCRIPTS_DIR := $(SRC_DIR)/scripts
ACTIONSET_DIR := $(SCRIPTS_DIR)/actionset
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

.PHONY: help

# ============================================================================
# HELP
# ============================================================================

help:
	@echo "$(BLUE)======================================================================$(NC)"
	@echo "$(GREEN)  One-Shot Action Recognition - ROS Makefile$(NC)"
	@echo "$(CYAN)  ROS operations$(NC)"
	@echo "$(BLUE)======================================================================$(NC)"
	@echo ""
	@echo "$(YELLOW)  系統啟動$(NC)"
	@echo "  $(GREEN)make start-gui-system$(NC)     - 啟動完整 GUI 系統 (包括相機)"
	@echo "  $(GREEN)make start-gui$(NC)            - 只啟動 GUI "
	@echo "  $(GREEN)make start-camera$(NC)         - 啟動相機 (60 FPS / USB 3.0)"
	@echo "  $(GREEN)make start-roscore$(NC)        - 只啟動 ROS "
	@echo ""
	@echo "$(YELLOW)  測試與驗證$(NC)"
	@echo "  $(GREEN)make test-camera$(NC)          - 測試相機連接和 FPS"
	@echo "  $(GREEN)make test-system$(NC)          - 檢查 ROS 狀態"
	@echo "  $(GREEN)make test-gpu$(NC)             - 檢查 GPU 和 PyTorch"
	@echo "  $(GREEN)make test-topics$(NC)          - 列出所有 ROS topics"
	@echo "  $(GREEN)make test-nodes$(NC)           - 列出所有 ROS nodes"
	@echo ""
	@echo "$(YELLOW)  監控$(NC)"
	@echo "  $(GREEN)make monitor-camera$(NC)       - 監控相機 FPS"
	@echo "  $(GREEN)make monitor-recognition$(NC)  - 監控 辨識結果"
	@echo "  $(GREEN)make monitor-gpu$(NC)          - 監控 GPU 使用率"
	@echo "  $(GREEN)make watch-topics$(NC)         - 持續監看所有 topics"
	@echo ""
	@echo "$(YELLOW)  模型訓練$(NC)"
	@echo "  $(GREEN)make train-ntu$(NC)            - 訓練 NTU RGB+D (200 epochs, 8-12hr)"
	@echo "  $(GREEN)make train-ntu-fast$(NC)       - 快速測試訓練 (5 epochs)"
	@echo "  $(GREEN)make train-ntu-resume$(NC)     - 從中斷處繼續訓練"
	@echo "  $(GREEN)make inspect-model$(NC)        - 檢查模型 checkpoint"
	@echo "  $(GREEN)make convert-format$(NC)       - 轉換骨架模型 MediaPipe -> COCO 17"
	@echo ""
	@echo "$(YELLOW)  開發與除錯$(NC)"
	@echo "  $(GREEN)make build$(NC)                - 編譯 ROS workspace"
	@echo "  $(GREEN)make clean-build$(NC)          - 清理並重新編譯"
	@echo "  $(GREEN)make check-params$(NC)         - 顯示 ROS parameters"
	@echo "  $(GREEN)make check-services$(NC)       - 列出所有 ROS services"
	@echo "  $(GREEN)make kill-ros$(NC)             - 停止所有 ROS 進程"
	@echo ""
	@echo "$(YELLOW)  資料管理$(NC)"
	@echo "  $(GREEN)make list-actions$(NC)         - 列出所有已錄製動作"
	@echo "  $(GREEN)make list-models$(NC)          - 列出所有模型 checkpoints"
	@echo "  $(GREEN)make clean-logs$(NC)           - 清理 ROS log 檔案"
	@echo "  $(GREEN)make clean-actionset$(NC)      - 清理所有動作資料 (警告!)"
	@echo ""
	@echo "$(YELLOW)  操作流程指引$(NC)"
	@echo "  $(GREEN)make guide-record$(NC)         - Page 1: 錄製動作流程"
	@echo "  $(GREEN)make guide-manage$(NC)         - Page 2: 管理動作流程"
	@echo "  $(GREEN)make guide-recognize$(NC)      - Page 3: 辨識流程"
	@echo "  $(GREEN)make guide-train$(NC)          - 訓練模型流程"
	@echo ""
	@echo "$(BLUE)======================================================================$(NC)"
	@echo "  完整啟動:"
	@echo "    $(YELLOW)make start-gui-system$(NC)"
	@echo ""
	@echo "  測試系統:"
	@echo "    $(YELLOW)make test-camera && make test-system$(NC)"
	@echo ""
	@echo "  監控運行:"
	@echo "    $(YELLOW)make monitor-recognition$(NC)  # 一個終端"
	@echo "    $(YELLOW)make monitor-gpu$(NC)           # 另一個終端"
	@echo ""
	@echo "$(BLUE)======================================================================$(NC)"


# 系統啟動
.PHONY: start-gui-system start-gui start-camera start-camera start-roscore

start-gui-system:
	@echo "$(YELLOW)啟動完整 GUI 系統 (4 個相機視窗)...$(NC)"
	@echo "$(CYAN)視窗: QT GUI + Raw Camera + Skeleton + Recognition$(NC)"
	@$(ROS_SETUP) && roslaunch yolo_ros qt_gui_full_system.launch

start-gui:
	@echo "$(YELLOW)啟動 QT GUI...$(NC)"
	@$(ROS_SETUP) && cd $(SRC_DIR)/qt_gui && python3.10 main.py

start-camera:
	@echo "$(YELLOW)啟動 D435i 相機 (60 FPS)...$(NC)"
	@echo "$(YELLOW)  需要 USB 3.0 連接！$(NC)"
	@$(ROS_SETUP) && roslaunch realsense2_camera rs_camera.launch \
		enable_depth:=false \
		enable_color:=true \
		color_width:=640 \
		color_height:=480 \
		color_fps:=60

start-roscore:
	@echo "$(YELLOW)啟動 ROS Master...$(NC)"
	@$(ROS_SETUP) && roscore

# 測試與驗證
.PHONY: test-camera test-system test-gpu test-topics test-nodes

test-camera:
	@echo "$(YELLOW)測試相機連接...$(NC)"
	@echo ""
	@echo "$(CYAN)1. 檢查 USB 連接:$(NC)"
	@lsusb | grep -i intel || echo "$(RED)未找到 Intel RealSense$(NC)"
	@echo ""
	@echo "$(CYAN)2. 檢查相機 topic:$(NC)"
	@$(ROS_SETUP) && rostopic list 2>/dev/null | grep camera || echo "$(RED)未找到相機 topic (需先啟動相機)$(NC)"
	@echo ""
	@echo "$(CYAN)3. 測試影像頻率 (5 秒):$(NC)"
	@$(ROS_SETUP) && timeout 5 rostopic hz /camera/color/image_raw 2>&1 || echo "$(RED)無法測量 (需先啟動相機)$(NC)"

test-system:
	@echo "$(YELLOW)檢查 ROS 系統狀態...$(NC)"
	@echo ""
	@echo "$(CYAN)1. ROS Nodes:$(NC)"
	@$(ROS_SETUP) && rosnode list 2>/dev/null || echo "$(RED)ROS Master 未運行$(NC)"
	@echo ""
	@echo "$(CYAN)2. Recognition Node:$(NC)"
	@$(ROS_SETUP) && rosnode list 2>/dev/null | grep recognition || echo "$(YELLOW)Recognition node 未啟動$(NC)"
	@echo ""
	@echo "$(CYAN)3. ROS Parameters:$(NC)"
	@$(ROS_SETUP) && rosparam list 2>/dev/null | grep model_manager || echo "$(YELLOW)未設置自訂動作$(NC)"

test-gpu:
	@echo "$(YELLOW)測試 GPU 和 PyTorch...$(NC)"
	@python3.10 -c "import torch; print('='*60); print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('CUDA Available:', torch.cuda.is_available()); print('GPU Count:', torch.cuda.device_count()); print('GPU Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'); print('='*60)" 2>/dev/null || echo "$(RED)PyTorch 未安裝或 GPU 不可用$(NC)"

test-topics:
	@echo "$(CYAN)所有 ROS Topics:$(NC)"
	@$(ROS_SETUP) && rostopic list

test-nodes:
	@echo "$(CYAN)所有 ROS Nodes:$(NC)"
	@$(ROS_SETUP) && rosnode list

# 監控
.PHONY: monitor-camera monitor-recognition monitor-gpu watch-topics

monitor-camera:
	@echo "$(YELLOW)監控相機 FPS ...$(NC)"
	@$(ROS_SETUP) && rostopic hz /camera/color/image_raw

monitor-recognition:
	@echo "$(YELLOW)監控辨識結果 ...$(NC)"
	@$(ROS_SETUP) && rostopic echo /yolo_ros/recognition_result

monitor-gpu:
	@echo "$(YELLOW)監控 GPU ...$(NC)"
	@watch -n 1 nvidia-smi

watch-topics:
	@echo "$(YELLOW)監看所有 topics ...$(NC)"
	@while true; do \
		clear; \
		echo "$(CYAN)=== ROS Topics (更新中) ===$(NC)"; \
		$(ROS_SETUP) && rostopic list; \
		sleep 2; \
	done

# 模型訓練
.PHONY: train-ntu train-ntu-fast train-ntu-resume inspect-model convert-format

train-ntu:
	@echo "$(YELLOW)開始訓練 NTU RGB+D 模型...$(NC)"
	@echo "$(MAGENTA)參數: 200 epochs, batch_size=32, lr=0.001$(NC)"
	@echo "$(YELLOW)預計時間: 8-12 小時$(NC)"
	@echo ""
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
	@echo "$(YELLOW)快速訓練測試 (5 epochs)...$(NC)"
	@cd $(SCRIPTS_DIR) && python3.10 train_ntu_rgbd.py \
		--data_path /root/catkin_ws/src/yolo_ros/nturgbd_skeletons_s001_to_s017/nturgb+d_skeletons \
		--epochs 5 \
		--batch_size 16 \
		--num_classes 60 \
		--benchmark xsub \
		--device cuda

train-ntu-resume:
	@echo "$(YELLOW)從中斷處繼續訓練...$(NC)"
	@echo "$(YELLOW)請確認 checkpoints/latest.pth 存在$(NC)"
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

inspect-model:
	@echo "$(YELLOW)檢查模型 checkpoint...$(NC)"
	@cd $(SCRIPTS_DIR) && python3.10 inspect_checkpoints.py

convert-format:
	@echo "$(YELLOW)轉換 MediaPipe 33 → COCO 17...$(NC)"
	@cd $(SCRIPTS_DIR) && python3.10 convert_mediapipe_to_coco.py

# 開發與除錯
.PHONY: build clean-build check-params check-services kill-ros

build:
	@echo "$(YELLOW)編譯 ROS workspace...$(NC)"
	@cd $(ROS_WS) && $(ROS_SETUP) && catkin_make

clean-build:
	@echo "$(YELLOW)清理並重新編譯...$(NC)"
	@cd $(ROS_WS) && rm -rf build/ devel/
	@cd $(ROS_WS) && $(ROS_SETUP) && catkin_make

check-params:
	@echo "$(CYAN)ROS Parameters:$(NC)"
	@$(ROS_SETUP) && rosparam list
	@echo ""
	@echo "$(CYAN)Custom Actions:$(NC)"
	@$(ROS_SETUP) && rosparam get /model_manager/custom_actions 2>/dev/null || echo "$(YELLOW)未設置$(NC)"

check-services:
	@echo "$(CYAN)ROS Services:$(NC)"
	@$(ROS_SETUP) && rosservice list

kill-ros:
	@echo "$(YELLOW)停止所有 ROS 進程...$(NC)"
	@pkill -9 -f "roscore|roslaunch|rosrun|rosnode" 2>/dev/null || true
	@echo "$(GREEN)  已停止$(NC)"

# 資料管理
.PHONY: list-actions list-models clean-logs clean-actionset

list-actions:
	@echo "$(CYAN)已錄製的動作:$(NC)"
	@ls -1 $(ACTIONSET_DIR) 2>/dev/null || echo "$(YELLOW)無動作資料$(NC)"

list-models:
	@echo "$(CYAN)模型 Checkpoints:$(NC)"
	@ls -lh $(CHECKPOINTS_DIR)/*.pth 2>/dev/null || echo "$(YELLOW)無 checkpoint$(NC)"

clean-logs:
	@echo "$(YELLOW)清理 ROS log 檔案...$(NC)"
	@find $(ROS_WS) -name "*.log" -delete 2>/dev/null || true
	@find ~/.ros/log -type f -delete 2>/dev/null || true
	@echo "$(GREEN)  log 已清理$(NC)"

clean-actionset:
	@echo "$(RED)  Warn: 這將刪除所有已錄製的動作資料！$(NC)"
	@read -p "確定要刪除? (yes/no): " confirm; \
	if [ "$$confirm" = "yes" ]; then \
		rm -rf $(ACTIONSET_DIR)/* && \
		echo "$(GREEN)  Actionset 已清理$(NC)"; \
	else \
		echo "$(YELLOW)已取消$(NC)"; \
	fi

# 操作流程指引
.PHONY: guide-record guide-manage guide-recognize guide-train

guide-record:
	@echo "$(BLUE)======================================================================$(NC)"
	@echo "$(GREEN)  Page 1 - 錄製動作流程$(NC)"
	@echo "$(BLUE)======================================================================$(NC)"
	@echo ""
	@echo "1. 啟動系統:"
	@echo "   $(YELLOW)make start-gui-system$(NC)"
	@echo ""
	@echo "2. 在 Page 1 錄製:"
	@echo "   - 輸入檔名 (例如: wave_hand)"
	@echo "   - 點擊 'Start Recording'"
	@echo "   - 做出動作 2-3 秒"
	@echo "   - 點擊 'Stop Recording'"
	@echo ""
	@echo "3. 自動產生檔案:"
	@echo "   - {action}_video.avi"
	@echo "   - {action}_skeleton_sequence.npy (MediaPipe 33)"
	@echo "   - {action}_skeleton_sequence_coco17.npy (COCO 17)"
	@echo ""

guide-manage:
	@echo "$(BLUE)======================================================================$(NC)"
	@echo "$(GREEN)  Page 2 - 管理動作流程$(NC)"
	@echo "$(BLUE)======================================================================$(NC)"
	@echo ""
	@echo "1. 切換到 Page 2"
	@echo "2. 選擇模型: latest 或 best"
	@echo "3. 從 actionset 列表選擇動作"
	@echo "4. 輸入自訂動作名稱"
	@echo "5. 點擊 'Add Action'"
	@echo ""
	@echo "預期 log:"
	@echo "  [INFO]: Loaded custom actions from ROS param: ['your_action']"
	@echo ""

guide-recognize:
	@echo "$(BLUE)======================================================================$(NC)"
	@echo "$(GREEN)  Page 3 - 辨識流程$(NC)"
	@echo "$(BLUE)======================================================================$(NC)"
	@echo ""
	@echo "1. 切換到 Page 3"
	@echo "2. 系統自動載入 Page 2 設定的動作"
	@echo "3. 在相機前做出動作"
	@echo "4. 觀察辨識結果和信心度"
	@echo ""
	@echo "監控辨識:"
	@echo "  $(YELLOW)make monitor-recognition$(NC)"
	@echo ""

guide-train:
	@echo "$(BLUE)======================================================================$(NC)"
	@echo "$(GREEN)  訓練 NTU 模型流程$(NC)"
	@echo "$(BLUE)======================================================================$(NC)"
	@echo ""
	@echo "1. 確認 GPU:"
	@echo "   $(YELLOW)make test-gpu$(NC)"
	@echo ""
	@echo "2. 開始訓練:"
	@echo "   $(YELLOW)make train-ntu$(NC)          # 完整訓練 (8-12小時)"
	@echo "   $(YELLOW)make train-ntu-fast$(NC)     # 快速測試 (5 epochs)"
	@echo ""
	@echo "3. 監控訓練 (另一個終端):"
	@echo "   $(YELLOW)make monitor-gpu$(NC)"
	@echo "   $(YELLOW)tail -f $(SCRIPTS_DIR)/training.log$(NC)"
	@echo ""
	@echo "4. 訓練完成後:"
	@echo "   - 檢查 $(CHECKPOINTS_DIR)/best.pth"
	@echo "   - 啟用 NTU 分類 (修改 recognition_display_node_v2.py:83)"
	@echo "   - 重啟系統測試"
	@echo ""