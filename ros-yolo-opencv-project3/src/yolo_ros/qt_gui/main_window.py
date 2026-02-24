#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主視窗類別
包含三個主要頁面和全域控制按鈕

Author: Claude AI Assistant
Date: 2025-11-23
"""

from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                              QPushButton, QStackedWidget, QLabel)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont
import rospy

from pages.page1_data_collection import Page1DataCollection
from pages.page2_model_management import Page2ModelManagement
from pages.page3_recognition import Page3Recognition


class MainWindow(QMainWindow):
    """主視窗類別"""

    # 自定義信號
    stop_signal = pyqtSignal()
    init_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("One-Shot Action Recognition System")
        self.setGeometry(100, 100, 1400, 900)

        # 建立中央元件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 主佈局
        main_layout = QVBoxLayout(central_widget)

        # 標題
        self._create_title(main_layout)

        # 頁面切換按鈕
        self._create_page_buttons(main_layout)

        # 堆疊視窗（三個頁面）
        self.stacked_widget = QStackedWidget()

        # 建立三個頁面
        self.page1 = Page1DataCollection()
        self.page2 = Page2ModelManagement()
        self.page3 = Page3Recognition()

        self.stacked_widget.addWidget(self.page1)
        self.stacked_widget.addWidget(self.page2)
        self.stacked_widget.addWidget(self.page3)

        main_layout.addWidget(self.stacked_widget)

        # 全域控制按鈕（STOP 和 Initial）
        self._create_control_buttons(main_layout)

        # 連接信號
        self._connect_signals()

        rospy.loginfo("Main window initialized")

    def _create_title(self, layout):
        """建立標題"""
        title_label = QLabel("One-Shot Action Recognition System")
        title_font = QFont("Arial", 20, QFont.Bold)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #2c3e50; padding: 10px;")
        layout.addWidget(title_label)

    def _create_page_buttons(self, layout):
        """建立頁面切換按鈕"""
        button_layout = QHBoxLayout()

        self.btn_page1 = QPushButton("Data Collection")
        self.btn_page2 = QPushButton("Model Management")
        self.btn_page3 = QPushButton("Live Recognition")

        # 設定按鈕樣式
        button_style = """
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 10px 20px;
                font-size: 14px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #1c5985;
            }
            QPushButton:checked {
                background-color: #16a085;
            }
        """

        for btn in [self.btn_page1, self.btn_page2, self.btn_page3]:
            btn.setCheckable(True)
            btn.setStyleSheet(button_style)
            button_layout.addWidget(btn)

        # 預設選中第一頁
        self.btn_page1.setChecked(True)

        # 連接切換事件
        self.btn_page1.clicked.connect(lambda: self._switch_page(0))
        self.btn_page2.clicked.connect(lambda: self._switch_page(1))
        self.btn_page3.clicked.connect(lambda: self._switch_page(2))

        layout.addLayout(button_layout)

    def _create_control_buttons(self, layout):
        """建立全域控制按鈕"""
        control_layout = QHBoxLayout()
        control_layout.addStretch()

        # STOP 按鈕
        self.btn_stop = QPushButton("STOP")
        self.btn_stop.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: none;
                padding: 15px 40px;
                font-size: 16px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """)
        self.btn_stop.clicked.connect(self._on_stop)

        # Initial 按鈕
        self.btn_init = QPushButton("Initial")
        self.btn_init.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                border: none;
                padding: 15px 40px;
                font-size: 16px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #229954;
            }
        """)
        self.btn_init.clicked.connect(self._on_initial)

        control_layout.addWidget(self.btn_stop)
        control_layout.addWidget(self.btn_init)
        control_layout.addStretch()

        layout.addLayout(control_layout)

    def _switch_page(self, index):
        """切換頁面"""
        self.stacked_widget.setCurrentIndex(index)

        # 更新按鈕狀態
        self.btn_page1.setChecked(index == 0)
        self.btn_page2.setChecked(index == 1)
        self.btn_page3.setChecked(index == 2)

        rospy.loginfo(f"Switched to page {index + 1}")

    def _connect_signals(self):
        """連接信號"""
        # 將全域信號傳遞給各頁面
        self.stop_signal.connect(self.page1.on_stop)
        self.stop_signal.connect(self.page2.on_stop)
        self.stop_signal.connect(self.page3.on_stop)

        self.init_signal.connect(self.page1.on_initial)
        self.init_signal.connect(self.page2.on_initial)
        self.init_signal.connect(self.page3.on_initial)

    def _on_stop(self):
        """STOP 按鈕點擊事件"""
        rospy.logwarn("STOP button pressed - stopping all operations")
        self.stop_signal.emit()

    def _on_initial(self):
        """Initial 按鈕點擊事件"""
        rospy.loginfo("Initial button pressed - initializing system")
        self.init_signal.emit()

    def closeEvent(self, event):
        """視窗關閉事件"""
        rospy.loginfo("Closing main window...")
        # 停止所有作業
        self.stop_signal.emit()
        event.accept()
