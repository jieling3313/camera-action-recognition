# --- 使用 cogrobot/robospection-ros-noetic:torch29cu128 作為基礎映像 ---
# 此映像已包含：
#   - Ubuntu 20.04 + ROS Noetic
#   - Python 3.10.16
#   - PyTorch + CUDA 12.x（支援 RTX 50 系列 GPU）
FROM cogrobot/robospection-ros-noetic:torch29cu128

# 設定環境變數
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Taipei
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 更新 ROS 環境設定
RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc
ENV ROS_HOSTNAME=localhost
ENV ROS_MASTER_URI=http://localhost:11311

# 安裝額外的基礎工具（如果基礎映像中缺少）
RUN apt-get update && apt-get install -y --no-install-recommends \
    git cmake build-essential \
    curl wget gnupg2 \
    udev libssl-dev libusb-1.0-0-dev pkg-config libgtk-3-dev \
    python3-rosdep python3-catkin-tools \
    ros-noetic-ddynamic-reconfigure \
    && rm -rf /var/lib/apt/lists/*

# 初始化 rosdep（如果尚未初始化）
RUN rosdep update || (rosdep init && rosdep update)

# 編譯與安裝 librealsense SDK
WORKDIR /tmp
ENV REALSENSE_VERSION v2.50.0
RUN git clone --depth 1 --branch ${REALSENSE_VERSION} https://github.com/IntelRealSense/librealsense.git && \
    cd librealsense && \
    cp config/99-realsense-libusb.rules /etc/udev/rules.d/ && \
    ./scripts/setup_udev_rules.sh && \
    mkdir build && cd build && \
    cmake ../ -DBUILD_EXAMPLES=OFF -DBUILD_GRAPHICAL_EXAMPLES=OFF -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc) && \
    make install && \
    rm -rf /tmp/librealsense

# 安裝 Python 依賴套件（使用 Python 3.10）
RUN python3.10 -m pip install --no-cache-dir --upgrade pip && \
    python3.10 -m pip install --no-cache-dir numpy scipy

# 安裝 PyTorch 2.7.0 (Stable) 以支援 RTX 5080 (sm_120)
# PyTorch 2.7.0 已支援 CUDA 12.8 和 Blackwell 架構 (sm_120)
RUN python3.10 -m pip install --no-cache-dir --default-timeout=1000 \
    torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 強制移除舊版 psutil (distutils 安裝的版本)
RUN rm -rf /usr/lib/python3/dist-packages/psutil* || true

# 分批安裝套件以避免超時,先安裝小型套件
RUN python3.10 -m pip install --no-cache-dir \
    "psutil>=5.9.0" "tqdm>=4.64.0" "pyyaml>=5.3.1" "requests>=2.23.0"

# 安裝大型套件 (opencv, numpy, pandas, matplotlib, seaborn)
RUN python3.10 -m pip install --default-timeout=1000 --no-cache-dir \
    "opencv-python>=4.6.0" "numpy>=1.23.0" "pandas>=1.1.4" \
    "matplotlib>=3.4.0" "seaborn>=0.11.0"

# 安裝 ultralytics（如果 PyTorch 已存在，應該不會重新安裝）
RUN python3.10 -m pip install --default-timeout=1000 --no-cache-dir ultralytics

# 安裝 POT（使用較新版本以支援 NumPy 2.x）
RUN python3.10 -m pip install --no-cache-dir "pot>=0.9.6"

# 安裝 MediaPipe 用於骨骼辨識
RUN python3.10 -m pip install --no-cache-dir mediapipe

# 安裝 QT5 相關套件用於 GUI 開發
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pyqt5 \
    python3-pyqt5.qtmultimedia \
    pyqt5-dev-tools \
    qttools5-dev-tools \
    libqt5multimedia5 \
    libqt5multimediawidgets5 \
    x11-apps \
    && rm -rf /var/lib/apt/lists/*

# 安裝 Python QT5 綁定
RUN python3.10 -m pip install --no-cache-dir PyQt5 PyQt5-sip

# 安裝 rosserial 用於 Arduino 通訊 (支援步進馬達控制)
# 這些套件已針對 ROS Noetic 和 Python 3 優化，無衝突
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-rosserial-arduino \
    ros-noetic-rosserial \
    ros-noetic-rosserial-python \
    python3-serial \
    && rm -rf /var/lib/apt/lists/*

# 建立 Catkin 工作空間並編譯 realsense-ros
WORKDIR /root/catkin_ws/src
ENV REALSENSE_ROS_VERSION 2.3.2
RUN git clone --depth 1 --branch ${REALSENSE_ROS_VERSION} https://github.com/IntelRealSense/realsense-ros.git

# 使用 rosdep 自動安裝所有原始碼套件的相依性
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && \
    cd /root/catkin_ws && \
    rosdep install --from-paths src --ignore-src -r -y"

# 返回工作空間根目錄並編譯
WORKDIR /root/catkin_ws
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && catkin_make"

# 設定容器啟動時的預設指令
CMD ["/bin/bash", "-c", "source /opt/ros/noetic/setup.bash && source /root/catkin_ws/devel/setup.bash && exec /bin/bash"]
