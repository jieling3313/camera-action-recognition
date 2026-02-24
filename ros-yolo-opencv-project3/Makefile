# ============================================================================
# One-Shot Action Recognition System - Host Makefile
# ============================================================================
# This Makefile manages Docker environment on the HOST machine.
# For ROS operations inside container, use: make shell,
#  then cd /root/catkin_ws && make <command>
# Quick start: make help
# ============================================================================

# Project directories
PROJECT_DIR := /home/jieling/Desktop/workspace/ObjectRecognition/ros-yolo-opencv-project3
DEVCONTAINER_DIR := $(PROJECT_DIR)/.devcontainer

# Docker compose command
COMPOSE := cd $(DEVCONTAINER_DIR) && docker compose

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
CYAN := \033[0;36m
NC := \033[0m # No Color

.PHONY: help

# ============================================================================
# HELP - Display all available commands
# ============================================================================

help:
	@echo "$(BLUE)======================================================================$(NC)"
	@echo "$(GREEN)  One-Shot Action Recognition - HOST Makefile$(NC)"
	@echo "$(CYAN)  Docker 環境 $(NC)"
	@echo "$(BLUE)======================================================================$(NC)"
	@echo ""
	@echo "$(YELLOW) DOCKER 環境建置$(NC)"
	@echo "  $(GREEN)make up$(NC)               - 啟動 Docker container"
	@echo "  $(GREEN)make down$(NC)             - 停止 Docker container"
	@echo "  $(GREEN)make restart$(NC)          - 重啟 Docker container"
	@echo "  $(GREEN)make rebuild$(NC)          - 重建 Docker image和container"
	@echo "  $(GREEN)make status$(NC)           - 顯示container狀態"
	@echo "  $(GREEN)make logs$(NC)             - 查看container日誌"
	@echo ""
	@echo "$(YELLOW)  系統設置$(NC)"
	@echo "  $(GREEN)make setup-x11$(NC)        - 設置 X11 權限 (GUI 顯示)"
	@echo "  $(GREEN)make setup-gpu$(NC)        - 安裝 NVIDIA Container Toolkit"
	@echo "  $(GREEN)make rebuild-gpu$(NC)      - 重建 GPU container"
	@echo ""
	@echo "$(YELLOW) container操作$(NC)"
	@echo "  $(GREEN)make shell$(NC)            - 進入container shell"
	@echo "  $(GREEN)make root$(NC)             - 以 root 身份進入container"
	@echo "  $(GREEN)make exec CMD='...'$(NC)   - 在container內執行指令"
	@echo ""
	@echo "$(YELLOW) 快速啟動 (整合指令)$(NC)"
	@echo "  $(GREEN)make quick-start$(NC)      - 完整啟動: X11 + 容器 + 進入 shell"
	@echo "  $(GREEN)make start-system$(NC)     - 啟動完整 ROS 系統"
	@echo ""
	@echo "$(BLUE)======================================================================$(NC)"
	@echo "$(CYAN) ROS operations (＠container):$(NC)"
	@echo ""
	@echo "  進入container後使用:"
	@echo "  $(YELLOW)make shell$(NC)           # 進入container"
	@echo "  $(YELLOW)cd /root/catkin_ws$(NC)  # 切換到catkin_ws(工作目錄)"
	@echo "  $(YELLOW)make help$(NC)            # 查看 ROS operations 指令"
	@echo ""
	@echo "  或直接執行:"
	@echo "  $(YELLOW)make exec CMD='cd /root/catkin_ws && make help'$(NC)"

# DOCKER 環境管理
.PHONY: up down restart rebuild status logs

up:
	@echo "$(YELLOW)啟動 Docker container...$(NC)"
	@$(COMPOSE) up -d
	@sleep 2
	@echo "$(GREEN)  container已啟動$(NC)"
	@$(MAKE) status

down:
	@echo "$(YELLOW)停止 Docker container...$(NC)"
	@$(COMPOSE) down
	@echo "$(GREEN)  container已停止$(NC)"

restart: down
	@echo "$(YELLOW)等待 2 秒...$(NC)"
	@sleep 2
	@$(MAKE) up

rebuild:
	@echo "$(YELLOW)重建 Docker image和container...$(NC)"
	@$(COMPOSE) down
	@$(COMPOSE) build --no-cache
	@$(COMPOSE) up -d
	@sleep 3
	@echo "$(GREEN)  重建完成$(NC)"
	@$(MAKE) status

status:
	@echo "$(CYAN)Docker container state:$(NC)"
	@$(COMPOSE) ps

logs:
	@echo "$(CYAN)查看container logs :$(NC)"
	@$(COMPOSE) logs -f

# 系統設置(X11 & gpu)
.PHONY: setup-x11 setup-gpu rebuild-gpu

setup-x11:
	@echo "$(YELLOW)設置 X11 權限...$(NC)"
	@xhost +local:root > /dev/null 2>&1
	@xhost +local:docker > /dev/null 2>&1
	@echo "$(GREEN)  X11 權限已設置$(NC)"

setup-gpu:
	@echo "$(YELLOW)安裝 NVIDIA Container Toolkit...$(NC)"
	@bash $(DEVCONTAINER_DIR)/archive/setup_gpu.sh

rebuild-gpu:
	@echo "$(YELLOW)重建 GPU 支援容器...$(NC)"
	@bash $(DEVCONTAINER_DIR)/archive/rebuild_with_gpu.sh

# container operations
.PHONY: shell root exec

shell:
	@echo "$(CYAN)進入container shell...$(NC)"
	@echo "$(YELLOW)'cd /root/catkin_ws && make help' 查看 ROS 指令$(NC)"
	@echo ""
	@$(COMPOSE) exec ros-dev bash

root:
	@echo "$(CYAN)root啟用中...$(NC)"
	@$(COMPOSE) exec -u root ros-dev bash

exec:
	@if [ -z "$(CMD)" ]; then \
		echo "$(RED)error$(NC)"; \
		echo "用法: make exec CMD='your command'"; \
		exit 1; \
	fi
	@$(COMPOSE) exec ros-dev bash -c "$(CMD)"

# Quick-start
.PHONY: quick-start start-system

quick-start: setup-x11 up
	@echo ""
	@echo "$(GREEN)======================================================================$(NC)"
	@echo "$(GREEN)  系統已準備就緒！$(NC)"
	@echo "$(GREEN)======================================================================$(NC)"
	@echo ""
	@echo "$(CYAN)下一步:$(NC)"
	@echo "  1. 進入容器: $(YELLOW)make shell$(NC)"
	@echo "  2. 切換目錄: $(YELLOW)cd /root/catkin_ws$(NC)"
	@echo "  3. 查看指令: $(YELLOW)make help$(NC)"
	@echo "  4. 啟動系統: $(YELLOW)make start-gui-system$(NC)"
	@echo ""

start-system:
	@echo "$(YELLOW)在container內啟動 ROS 系統...$(NC)"
	@$(COMPOSE) exec ros-dev bash -c "cd /root/catkin_ws && make start-gui-system"

# 別名 (方便使用)
.PHONY: start stop

start: up
stop: down