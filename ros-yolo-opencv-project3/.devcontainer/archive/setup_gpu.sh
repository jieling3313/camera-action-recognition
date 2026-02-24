#!/bin/bash
# GPU 支援設定腳本
# 此腳本會安裝 NVIDIA Container Toolkit 並配置 Docker 以支援 GPU

set -e  # 遇到錯誤立即停止

echo "=========================================="
echo "安裝 NVIDIA Container Toolkit"
echo "=========================================="

# 0. 修復 CD-ROM 套件來源問題
echo "步驟 0/5: 修復 CD-ROM 套件來源..."
if grep -q "^deb.*file:///cdrom" /etc/apt/sources.list 2>/dev/null; then
    sudo sed -i 's/^deb \[check-date=no\] file:\/\/\/cdrom/# deb [check-date=no] file:\/\/\/cdrom/' /etc/apt/sources.list
    echo "  ✓ CD-ROM 套件來源已停用"
else
    echo "  ✓ 無需修復"
fi

# 1. 添加 NVIDIA Container Toolkit GPG 金鑰
echo "步驟 1/5: 添加 GPG 金鑰..."
if [ ! -f /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg ]; then
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
        sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    echo "  ✓ GPG 金鑰已添加"
else
    echo "  ✓ GPG 金鑰已存在"
fi

# 2. 添加套件庫
echo "步驟 2/5: 添加套件庫..."
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null
echo "  ✓ 套件庫已添加"

# 3. 更新套件列表
echo "步驟 3/5: 更新套件列表..."
sudo apt-get update

# 4. 安裝 NVIDIA Container Toolkit
echo "步驟 4/5: 安裝 NVIDIA Container Toolkit..."
sudo apt-get install -y nvidia-container-toolkit

# 5. 配置 Docker
echo "步驟 5/5: 配置 Docker..."
sudo nvidia-ctk runtime configure --runtime=docker

# 重啟 Docker
echo "重啟 Docker 服務..."
sudo systemctl restart docker

echo ""
echo "=========================================="
echo "✓ 安裝完成！"
echo "=========================================="
echo ""
echo "請執行以下指令重啟容器："
echo "  cd /home/jieling/Desktop/workspace/ObjectRecognition/ros-yolo-opencv-project3/.devcontainer"
echo "  docker compose down"
echo "  docker compose up -d"
echo ""
