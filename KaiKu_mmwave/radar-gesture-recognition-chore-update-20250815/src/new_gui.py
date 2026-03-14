# gesture_gui_pyside.py
import sys
import random
import os
import sys

from PySide2.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QProgressBar,
    QHBoxLayout, QGridLayout, QFrame, QPushButton,
    QSizePolicy, QSpacerItem, QStackedWidget, QTextEdit
)
from PySide2.QtCore import Qt
from PySide2.QtGui import QFont, QPixmap
from PySide2.QtMultimedia import QSound

# ---------- 資源路徑處理函式 ----------

def resource_path(relative_path: str) -> str:
    """
    讓程式在「原始 Python」和「PyInstaller 打包後」都能正確找到資源檔。
    """
    if hasattr(sys, "_MEIPASS"):
        # exe 被解壓出來的暫存資料夾
        base_path = sys._MEIPASS
    else:
        # 開發階段：new_gui.py 所在資料夾
        base_path = os.path.dirname(__file__)
    return os.path.join(base_path, relative_path)



class GestureGUI(QWidget):
    """
    展示面板風格 + 九宮格 / 病例 / 燈光 / 冷氣 控制 GUI

    模式與手勢對應：

    九宮格畫面 center_stack = 0，內部再分 selection_mode：
      - "room"   : 一般房間模式
          Up/Down/Left/Right：移動九宮格房間
          Tap               ：進入該房間的「病例模式」
          若目前在最左欄(col=0) 且手勢 Left：進入「燈光模式」
      - "light"  : 燈光控制模式
          Up/Down：亮度 ±5%
          Left   ：進入冷氣模式
          Right  ：回到房間模式
      - "ac"     : 冷氣控制模式
          Up/Down：溫度 ±1℃
          Right  ：回到燈光模式

    病例模式 center_stack = 1：
      - Right：下一位病患
      - Left ：上一位病患
      - Up   ：病例向上捲動
      - Down ：病例向下捲動
      - Tap  ：返回九宮格

    鍵盤：
      ↑ ↓ ← → ：對應手勢 Up/Down/Left/Right
      Space    ：Tap
      B        ：Background
      X        ：隨機房號發出緊急呼叫
      G        ：接聽 / 掛斷 緊急呼叫
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("好米波 - Gesture Recognition Room Control")
        self.setMinimumSize(980, 640)

        # --- 房間 / 病患狀態 ---
        self.current_row = 0
        self.current_col = 0
        self.last_row = 0
        self.last_col = 0
        self.last_gesture = "Background"

        self.room_numbers = {
            (0, 0): "301", (0, 1): "302", (0, 2): "303",
            (1, 0): "304", (1, 1): "305", (1, 2): "306",
            (2, 0): "307", (2, 1): "308", (2, 2): "309",
        }
        self.max_patients_per_room = 4          # 每房病患數
        self.current_patient_idx = 0            # 0 ~ max
        self.current_room_id = None             # 病例模式房號

        # 冷氣 / 燈光狀態
        self.ac_current_temp = 26               # 預設 26℃
        self.light_level = 60                   # 預設亮度 60%

        # 九宮格內部模式：room / light / ac
        self.selection_mode = "room"

        # 緊急呼叫狀態
        self.emergency_active = False
        self.emergency_answered = False
        self.emergency_room = None
        self.emergency_room_pos = None
        try:
            self.emergency_sound = QSound(resource_path("xm3669.wav")) #emergency.wav 聲音檔
        except Exception:
            self.emergency_sound = None

        # 整體背景
        self.setStyleSheet("QWidget { background-color: #ECEFF1; }")

        # ========== 主版面：左右兩塊 ==========
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(16)

        # ================= 左側：資訊展示板 =================
        left_root = QFrame()
        left_root.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        left_root.setMinimumWidth(260)
        left_root.setStyleSheet("""
            QFrame {
                background-color: #FAFAFA;
                border-radius: 12px;
                border: 1px solid #CFD8DC;
            }
        """)
        left_layout = QVBoxLayout(left_root)
        left_layout.setContentsMargins(14, 14, 14, 14)
        left_layout.setSpacing(12)

        # 頂部深色條（原本文字 "展示資訊" 已去掉）
        self.header_label = QLabel("")
        self.header_label.setAlignment(Qt.AlignCenter)
        self.header_label.setFont(QFont("Microsoft JhengHei", 12, QFont.Bold))
        self.header_label.setStyleSheet("""
            QLabel {
                background-color: #455A64;
                color: white;
                border-radius: 8px;
                padding: 6px 4px;
            }
        """)
        left_layout.addWidget(self.header_label)

        # --- 合作學校卡片（LOGO 上下排列） ---
        school_card = QFrame()
        school_card.setStyleSheet("""
            QFrame {
                background-color: #FFFFFF;
                border-radius: 10px;
                border: 1px solid #E0E0E0;
            }
        """)
        school_layout = QVBoxLayout(school_card)
        school_layout.setContentsMargins(10, 10, 10, 10)
        school_layout.setSpacing(8)

        school_title = QLabel("合作學校")
        school_title.setFont(QFont("Microsoft JhengHei", 11, QFont.Bold))
        school_title.setAlignment(Qt.AlignCenter)
        school_title.setStyleSheet("color: #37474F;")
        school_layout.addWidget(school_title)

        nfust_logo_label = QLabel()
        nfust_logo = QPixmap(resource_path("nfu.png"))  #nfu.png 虎尾科技大學 Logo
        if not nfust_logo.isNull():
            nfust_logo_label.setPixmap(
                nfust_logo.scaled(110, 110, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
        nfust_logo_label.setAlignment(Qt.AlignCenter)
        nfust_logo_label.setFixedSize(120, 120)
        school_layout.addWidget(nfust_logo_label, alignment=Qt.AlignCenter)

        cgu_logo_label = QLabel()
        cgu_logo   = QPixmap(resource_path("cgu.png")) #cgu.png 長庚大學 Logo
        if not cgu_logo.isNull():
            cgu_logo_label.setPixmap(
                cgu_logo.scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
        cgu_logo_label.setAlignment(Qt.AlignCenter)
        cgu_logo_label.setFixedSize(120, 120)
        school_layout.addWidget(cgu_logo_label, alignment=Qt.AlignCenter)

        left_layout.addWidget(school_card)

        # --- 作品資訊卡片 ---
        work_card = QFrame()
        work_card.setStyleSheet("""
            QFrame {
                background-color: #FFFFFF;
                border-radius: 10px;
                border: 1px solid #E0E0E0;
            }
        """)
        work_layout = QVBoxLayout(work_card)
        work_layout.setContentsMargins(10, 10, 10, 10)
        work_layout.setSpacing(8)

        work_title = QLabel("作品資訊")
        work_title.setFont(QFont("Microsoft JhengHei", 11, QFont.Bold))
        work_title.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        work_title.setStyleSheet("color: #37474F;")
        work_layout.addWidget(work_title)

        work_name = QLabel("作品名稱：好米波")
        work_name.setFont(QFont("Microsoft JhengHei", 10))
        work_name.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        work_layout.addWidget(work_name)

        work_logo_label = QLabel()
        work_logo = QPixmap("logo_work.png") #logo_work.png 作品 Logo
        if not work_logo.isNull():
            work_logo_label.setPixmap(
                work_logo.scaled(120, 120, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
        work_logo_label.setAlignment(Qt.AlignCenter)
        work_layout.addWidget(work_logo_label)

        left_layout.addWidget(work_card)

        # --- 設備控制卡片 (冷氣 + 燈光) ---
        control_card = QFrame()
        self.ac_style_normal = """
            QFrame {
                background-color: #F9FAFB;
                border-radius: 8px;
                border: 1px solid #E0E0E0;
            }
        """
        self.ac_style_active = """
            QFrame {
                background-color: #FFE0B2;
                border-radius: 8px;
                border: 2px solid #FB8C00;
            }
        """
        self.light_style_normal = """
            QFrame {
                background-color: #F9FAFB;
                border-radius: 8px;
                border: 1px solid #E0E0E0;
            }
        """
        self.light_style_active = """
            QFrame {
                background-color: #FFF3E0;
                border-radius: 8px;
                border: 2px solid #FB8C00;
            }
        """

        control_card.setStyleSheet("""
            QFrame {
                background-color: #FFFFFF;
                border-radius: 10px;
                border: 1px solid #E0E0E0;
            }
        """)
        ctrl_layout = QVBoxLayout(control_card)
        ctrl_layout.setContentsMargins(10, 10, 10, 10)
        ctrl_layout.setSpacing(10)

        ctrl_title = QLabel("設備控制（示意）")
        ctrl_title.setFont(QFont("Microsoft JhengHei", 11, QFont.Bold))
        ctrl_title.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        ctrl_title.setStyleSheet("color: #37474F;")
        ctrl_layout.addWidget(ctrl_title)

        ctrl_row = QHBoxLayout()
        ctrl_row.setSpacing(12)

        # 冷氣
        self.ac_frame = QFrame()
        self.ac_frame.setStyleSheet(self.ac_style_normal)
        ac_layout = QVBoxLayout(self.ac_frame)
        ac_layout.setContentsMargins(6, 6, 6, 6)
        ac_layout.setSpacing(4)

        ac_label = QLabel("冷氣")
        ac_label.setFont(QFont("Microsoft JhengHei", 10, QFont.Bold))
        ac_label.setAlignment(Qt.AlignCenter)
        ac_layout.addWidget(ac_label)

        ac_btn_up = QPushButton("▲")
        ac_btn_up.setCursor(Qt.PointingHandCursor)
        ac_btn_up.setStyleSheet("""
            QPushButton {
                border: none;
                font-size: 14px;
                min-height: 22px;
            }
            QPushButton:hover {
                background-color: #E3F2FD;
                border-radius: 4px;
            }
        """)
        ac_layout.addWidget(ac_btn_up, alignment=Qt.AlignCenter)

        self.ac_temp_label = QLabel(f"目前溫度：{self.ac_current_temp}℃")
        self.ac_temp_label.setFont(QFont("Microsoft JhengHei", 9))
        self.ac_temp_label.setAlignment(Qt.AlignCenter)
        self.ac_temp_label.setStyleSheet("color:#37474F;")
        ac_layout.addWidget(self.ac_temp_label, alignment=Qt.AlignCenter)

        ac_btn_down = QPushButton("▼")
        ac_btn_down.setCursor(Qt.PointingHandCursor)
        ac_btn_down.setStyleSheet("""
            QPushButton {
                border: none;
                font-size: 14px;
                min-height: 22px;
            }
            QPushButton:hover {
                background-color: #E3F2FD;
                border-radius: 4px;
            }
        """)
        ac_layout.addWidget(ac_btn_down, alignment=Qt.AlignCenter)

        ac_btn_up.clicked.connect(lambda: self.adjust_ac_temp(+1))
        ac_btn_down.clicked.connect(lambda: self.adjust_ac_temp(-1))

        # 燈光
        self.light_frame = QFrame()
        self.light_frame.setStyleSheet(self.light_style_normal)
        light_layout = QVBoxLayout(self.light_frame)
        light_layout.setContentsMargins(6, 6, 6, 6)
        light_layout.setSpacing(4)

        light_label = QLabel("燈光")
        light_label.setFont(QFont("Microsoft JhengHei", 10, QFont.Bold))
        light_label.setAlignment(Qt.AlignCenter)
        light_layout.addWidget(light_label)

        light_btn_up = QPushButton("▲")
        light_btn_up.setCursor(Qt.PointingHandCursor)
        light_btn_up.setStyleSheet("""
            QPushButton {
                border: none;
                font-size: 14px;
                min-height: 22px;
            }
            QPushButton:hover {
                background-color: #FFF3E0;
                border-radius: 4px;
            }
        """)
        light_layout.addWidget(light_btn_up, alignment=Qt.AlignCenter)

        self.light_level_label = QLabel(f"亮度：{self.light_level}%")
        self.light_level_label.setFont(QFont("Microsoft JhengHei", 9))
        self.light_level_label.setAlignment(Qt.AlignCenter)
        self.light_level_label.setStyleSheet("color:#37474F;")
        light_layout.addWidget(self.light_level_label, alignment=Qt.AlignCenter)

        light_btn_down = QPushButton("▼")
        light_btn_down.setCursor(Qt.PointingHandCursor)
        light_btn_down.setStyleSheet("""
            QPushButton {
                border: none;
                font-size: 14px;
                min-height: 22px;
            }
            QPushButton:hover {
                background-color: #FFF3E0;
                border-radius: 4px;
            }
        """)
        light_layout.addWidget(light_btn_down, alignment=Qt.AlignCenter)

        light_btn_up.clicked.connect(lambda: self.adjust_light_level(+5))
        light_btn_down.clicked.connect(lambda: self.adjust_light_level(-5))

        ctrl_row.addWidget(self.ac_frame)
        ctrl_row.addWidget(self.light_frame)
        ctrl_layout.addLayout(ctrl_row)

        left_layout.addWidget(control_card)
        left_layout.addSpacerItem(
            QSpacerItem(10, 10, QSizePolicy.Minimum, QSizePolicy.Expanding)
        )

        main_layout.addWidget(left_root)

        # ================= 右側：主展示區 =================
        right_panel = QFrame()
        right_panel.setFrameShape(QFrame.NoFrame)
        right_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_layout.addWidget(right_panel)
        main_layout.setStretch(0, 0)
        main_layout.setStretch(1, 1)

        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)

        # 上半：卡片 + StackWidget（九宮格 / 病例）
        main_card = QFrame()
        main_card.setStyleSheet("""
            QFrame {
                background-color: #FFFFFF;
                border-radius: 12px;
                border: 1px solid #CFD8DC;
            }
        """)
        main_card_layout = QVBoxLayout(main_card)
        main_card_layout.setContentsMargins(16, 12, 16, 12)
        main_card_layout.setSpacing(12)

        self.current_gesture_label = QLabel("Current gesture: Background")
        self.current_gesture_label.setAlignment(Qt.AlignCenter)
        self.current_gesture_label.setMinimumHeight(40)
        self.current_gesture_label.setFont(QFont("Microsoft JhengHei", 12, QFont.Bold))
        self.current_gesture_label.setStyleSheet("""
            QLabel {
                background-color: #ECEFF1;
                border-radius: 8px;
                padding: 6px;
            }
        """)
        main_card_layout.addWidget(self.current_gesture_label)

        self.center_stack = QStackedWidget()
        main_card_layout.addWidget(self.center_stack)

        # === Page 0：九宮格 ===
        grid_page = QFrame()
        grid_page_layout = QVBoxLayout(grid_page)
        grid_page_layout.setContentsMargins(0, 0, 0, 0)
        grid_page_layout.setSpacing(0)

        grid_frame = QFrame()
        grid_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        grid_layout = QGridLayout(grid_frame)
        grid_layout.setSpacing(18)
        grid_layout.setContentsMargins(6, 6, 6, 6)

        self.cells = {}
        for row in range(3):
            for col in range(3):
                cell = QLabel()
                cell.setAlignment(Qt.AlignCenter)
                cell.setMinimumSize(130, 80)
                cell.setFont(QFont("Microsoft JhengHei", 18, QFont.Bold))
                room_text = self.room_numbers.get((row, col), "")
                cell.setText(room_text)
                cell.setStyleSheet("""
                    QLabel {
                        background-color: #F5F5F5;
                        border: 2px solid #B0BEC5;
                        border-radius: 10px;
                        color: #37474F;
                    }
                """)
                cell.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                grid_layout.addWidget(cell, row, col)
                self.cells[(row, col)] = cell

        grid_page_layout.addWidget(grid_frame)
        self.center_stack.addWidget(grid_page)  # index 0

        # === Page 1：病例介面 ===
        record_page = QFrame()
        record_page_layout = QVBoxLayout(record_page)
        record_page_layout.setContentsMargins(0, 0, 0, 0)
        record_page_layout.setSpacing(8)

        header_row = QHBoxLayout()
        header_row.setSpacing(8)

        self.record_title_label = QLabel("病房 病例檔案")
        self.record_title_label.setFont(QFont("Microsoft JhengHei", 11, QFont.Bold))
        self.record_title_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        header_row.addWidget(self.record_title_label)
        header_row.addStretch()

        self.btn_back_to_grid = QPushButton("◀ 返回九宮格")
        self.btn_back_to_grid.setCursor(Qt.PointingHandCursor)
        self.btn_back_to_grid.setStyleSheet("""
            QPushButton {
                background-color: #455A64;
                color: white;
                border-radius: 6px;
                padding: 4px 10px;
                border: none;
            }
            QPushButton:hover {
                background-color: #1E88E5;
            }
        """)
        self.btn_back_to_grid.clicked.connect(self.show_grid_view)
        header_row.addWidget(self.btn_back_to_grid)

        record_page_layout.addLayout(header_row)

        body_row = QHBoxLayout()
        body_row.setSpacing(8)

        # 左病例欄（放大）
        left_case_col = QVBoxLayout()
        left_case_col.setSpacing(4)
        self.left_case_title = QLabel("病患1 病例檔案呈現")
        self.left_case_title.setAlignment(Qt.AlignCenter)
        self.left_case_title.setFont(QFont("Microsoft JhengHei", 10, QFont.Bold))
        left_case_col.addWidget(self.left_case_title)

        self.record_left = QTextEdit()
        self.record_left.setReadOnly(True)
        self.record_left.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.record_left.setStyleSheet("""
            QTextEdit {
                border: 1px solid #B0BEC5;
                border-radius: 6px;
                background-color: #FFFFFF;
            }
        """)
        left_case_col.addWidget(self.record_left)
        body_row.addLayout(left_case_col, stretch=1)

        # 右病例欄（放大）
        right_case_col = QVBoxLayout()
        right_case_col.setSpacing(4)
        self.right_case_title = QLabel("病患2 病例檔案呈現")
        self.right_case_title.setAlignment(Qt.AlignCenter)
        self.right_case_title.setFont(QFont("Microsoft JhengHei", 10, QFont.Bold))
        right_case_col.addWidget(self.right_case_title)

        self.record_right = QTextEdit()
        self.record_right.setReadOnly(True)
        self.record_right.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.record_right.setStyleSheet("""
            QTextEdit {
                border: 1px solid #B0BEC5;
                border-radius: 6px;
                background-color: #FFFFFF;
            }
        """)
        right_case_col.addWidget(self.record_right)
        body_row.addLayout(right_case_col, stretch=1)

        record_page_layout.addLayout(body_row)
        self.center_stack.addWidget(record_page)  # index 1

        right_layout.addWidget(main_card)

        # ===== 下半：手勢機率卡片（已隱藏，不加入 GUI） =====
                # ===== 下半：手勢機率卡片（建立但隱藏） =====
        '''
        self.prob_card = QFrame(self)
        self.prob_card.setStyleSheet("""
            QFrame {
                background-color: #FFFFFF;
                border-radius: 12px;
                border: 1px solid #CFD8DC;
            }
        """)
        prob_layout = QVBoxLayout(self.prob_card)
        prob_layout.setContentsMargins(12, 8, 12, 10)
        prob_layout.setSpacing(6)

        bar_title = QLabel("Gesture Probabilities")
        bar_title.setAlignment(Qt.AlignCenter)
        bar_title.setFont(QFont("Arial", 9, QFont.Bold))
        bar_title.setStyleSheet("color: #455A64;")
        prob_layout.addWidget(bar_title)

        bar_row = QHBoxLayout()
        bar_row.setSpacing(12)
        self.gesture_names = ["Background", "Down", "Left", "Right", "Tap", "Up"]
        self.bars = {}

        self.bar_colors = {
            "Background": "#81C784",
            "Down": "#64B5F6",
            "Left": "#FFB74D",
            "Right": "#BA68C8",
            "Tap": "#E57373",
            "Up": "#4DD0E1"
        }
        self.gesture_colors = {
            "Background": "#E0E0E0",
            "Down": "#BBDEFB",
            "Left": "#FFE0B2",
            "Right": "#E1BEE7",
            "Tap": "#FFCDD2",
            "Up": "#B2EBF2"
        }

        for name in self.gesture_names:
            v_layout = QVBoxLayout()
            bar = QProgressBar()
            bar.setOrientation(Qt.Vertical)
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setTextVisible(False)
            bar.setFixedWidth(18)
            bar.setMinimumHeight(80)
            bar.setStyleSheet(f"""
                QProgressBar {{
                    border: 1px solid #B0BEC5;
                    border-radius: 4px;
                    background: #ECEFF1;
                }}
                QProgressBar::chunk {{
                    background-color: {self.bar_colors[name]};
                    margin: 0px;
                }}
            """)
            v_layout.addWidget(bar, alignment=Qt.AlignBottom)

            label = QLabel(name)
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet("font-size: 8.5pt; color: #455A64;")
            v_layout.addWidget(label)

            bar_row.addLayout(v_layout)
            self.bars[name] = bar

        prob_layout.addLayout(bar_row)

        # ✅ 加到 layout 但立刻隱藏，GUI 上看不到，但物件不會被刪
        right_layout.addWidget(self.prob_card)
        self.prob_card.hide()


        # ⚠ 重點：不再顯示在 GUI 上
        # right_layout.addWidget(prob_card)

        # 初始化目前房間高亮 & 控制區樣式
        self.update_grid_display()
        self.update_control_mode_styles()

        '''

    # ====== 冷氣 / 燈光 調整 ======
    def adjust_ac_temp(self, delta):
        self.ac_current_temp = max(16, min(32, self.ac_current_temp + delta))
        self.ac_temp_label.setText(f"目前溫度：{self.ac_current_temp}℃")

    def adjust_light_level(self, delta):
        self.light_level = max(0, min(100, self.light_level + delta))
        self.light_level_label.setText(f"亮度：{self.light_level}%")

    # ====== 病例顯示更新 ======
    def _update_record_texts(self):
        room_id = self.current_room_id or self.room_numbers.get(
            (self.current_row, self.current_col), ""
        )

        cur_idx = self.current_patient_idx
        next_idx = (cur_idx + 1) % self.max_patients_per_room

        self.record_title_label.setText(f"病房 {room_id} 病例檔案")

        self.left_case_title.setText(f"病患{cur_idx + 1} 病例檔案呈現")
        self.right_case_title.setText(f"病患{next_idx + 1} 病例檔案呈現")

        self.record_left.setPlainText(
            f"【病房 {room_id} 病患 {cur_idx + 1}】\n\n"
            f"（此處可放該病患的摘要、生命徵象、重要提醒...）"
        )
        self.record_right.setPlainText(
            f"【病房 {room_id} 病患 {next_idx + 1}】\n\n"
            f"（此處可放下一位病患的相關資訊或預覽內容...）"
        )

        for editor in (self.record_left, self.record_right):
            sb = editor.verticalScrollBar()
            sb.setValue(sb.minimum())

    def show_grid_view(self):
        self.center_stack.setCurrentIndex(0)
        self.selection_mode = "room"
        self.update_control_mode_styles()
        # 回九宮格時把當前房間重新高亮
        self.update_single_cell(self.current_row, self.current_col, True)
        self.last_row = self.current_row
        self.last_col = self.current_col

    def show_record_view(self, room_id: str):
        self.center_stack.setCurrentIndex(1)
        self.current_room_id = room_id
        self.current_patient_idx = 0
        self._update_record_texts()

    def next_patient_in_room(self):
        self.current_patient_idx = (self.current_patient_idx + 1) % self.max_patients_per_room
        self._update_record_texts()

    def prev_patient_in_room(self):
        self.current_patient_idx = (self.current_patient_idx - 1) % self.max_patients_per_room
        self._update_record_texts()

    # ================== 房間格子顯示 ==================
    def update_single_cell(self, row, col, is_current):
        cell = self.cells[(row, col)]
        room_text = self.room_numbers.get((row, col), "")

        # 若這個格子是緊急呼叫房間，優先顯示紅色呼叫樣式
        if self.emergency_active and self.emergency_room_pos == (row, col):
            cell.setText(f"{room_text}\n(呼叫)")
            cell.setStyleSheet("""
                QLabel {
                    background-color: #FFCDD2;
                    border: 3px solid #D32F2F;
                    border-radius: 10px;
                    color: #B71C1C;
                    font-weight: bold;
                }
            """)
            return

        if is_current:
            cell.setText(room_text)
            cell.setStyleSheet("""
                QLabel {
                    background-color: #FFE0B2;
                    border: 3px solid #FB8C00;
                    border-radius: 10px;
                    color: #263238;
                    font-weight: bold;
                }
            """)
        else:
            cell.setText(room_text)
            cell.setStyleSheet("""
                QLabel {
                    background-color: #F5F5F5;
                    border: 2px solid #B0BEC5;
                    border-radius: 10px;
                    color: #37474F;
                }
            """)

    def update_grid_display(self):
        if (self.last_row, self.last_col) != (self.current_row, self.current_col):
            self.update_single_cell(self.last_row, self.last_col, False)
        self.update_single_cell(self.current_row, self.current_col, True)
        self.last_row = self.current_row
        self.last_col = self.current_col

    def clear_grid_highlight(self):
        """把目前選取的九宮格還原成一般顏色"""
        self.update_single_cell(self.current_row, self.current_col, False)
        self.last_row = self.current_row
        self.last_col = self.current_col

    def clear_emergency_highlight(self):
        """把緊急呼叫的房間格子恢復正常樣式"""
        if not self.emergency_room_pos:
            return
        row, col = self.emergency_room_pos

        # 若目前是在九宮格房間模式，且這格是 current，就恢復成高亮；否則恢復一般
        if (self.center_stack.currentIndex() == 0 and
                self.selection_mode == "room" and
                (row, col) == (self.current_row, self.current_col)):
            self.update_single_cell(row, col, True)
        else:
            self.update_single_cell(row, col, False)

        self.emergency_room_pos = None
        self.emergency_room = None

    def update_control_mode_styles(self):
        if self.selection_mode == "light":
            self.light_frame.setStyleSheet(self.light_style_active)
        else:
            self.light_frame.setStyleSheet(self.light_style_normal)

        if self.selection_mode == "ac":
            self.ac_frame.setStyleSheet(self.ac_style_active)
        else:
            self.ac_frame.setStyleSheet(self.ac_style_normal)

    # ================== 緊急呼叫相關 ==================
    def trigger_emergency_call(self):
        """模擬病床按下緊急呼叫鈴：隨機一間房發出呼叫並播放聲音"""
        if self.emergency_active:
            self.clear_emergency_highlight()

        pos = random.choice(list(self.room_numbers.keys()))
        self.emergency_room_pos = pos
        self.emergency_room = self.room_numbers[pos]
        self.emergency_active = True
        self.emergency_answered = False

        row, col = pos
        cell = self.cells[(row, col)]
        cell.setText(f"{self.emergency_room}\n(呼叫)")
        cell.setStyleSheet("""
            QLabel {
                background-color: #FFCDD2;
                border: 3px solid #D32F2F;
                border-radius: 10px;
                color: #B71C1C;
                font-weight: bold;
            }
        """)

        if self.emergency_sound is not None:
            self.emergency_sound.stop()
            self.emergency_sound.play()

    def handle_emergency_answer_or_hangup(self):
        """按下 G：若有緊急呼叫 => 第一次接聽，第二次掛斷"""
        if not self.emergency_active:
            return

        if not self.emergency_answered:
            # 接聽
            self.emergency_answered = True
            if self.emergency_sound is not None:
                self.emergency_sound.stop()
            room = self.emergency_room or ""
            self.header_label.setText(f"通話中 ({room})")
            self.header_label.setStyleSheet("""
                QLabel {
                    background-color: #B71C1C;
                    color: white;
                    border-radius: 8px;
                    padding: 6px 4px;
                    font-weight: bold;
                }
            """)
        else:
            # 掛斷
            self.emergency_answered = False
            self.emergency_active = False
            if self.emergency_sound is not None:
                self.emergency_sound.stop()

            # 左上角恢復原本樣式
            self.header_label.setText("")
            self.header_label.setStyleSheet("""
                QLabel {
                    background-color: #455A64;
                    color: white;
                    border-radius: 8px;
                    padding: 6px 4px;
                }
            """)

            # 九宮格恢復顏色
            self.clear_emergency_highlight()

    # ================== 手勢行為邏輯 ==================
    def move_position(self, gesture):
        idx = self.center_stack.currentIndex()

        # ---- 病例模式 ----
        if idx == 1:
            scroll_step = 60
            if gesture == "Right":
                self.next_patient_in_room()
                return True
            elif gesture == "Left":
                self.prev_patient_in_room()
                return True
            elif gesture == "Up":
                for editor in (self.record_left, self.record_right):
                    sb = editor.verticalScrollBar()
                    sb.setValue(sb.value() - scroll_step)
                return True
            elif gesture == "Down":
                for editor in (self.record_left, self.record_right):
                    sb = editor.verticalScrollBar()
                    sb.setValue(sb.value() + scroll_step)
                return True
            elif gesture == "Tap":
                self.show_grid_view()
                return True
            else:
                return False

        # ---- 九宮格畫面：依 selection_mode 分類 ----
        if idx == 0:
            # 燈光模式
            if self.selection_mode == "light":
                if gesture == "Left":
                    self.selection_mode = "ac"
                    self.update_control_mode_styles()
                    return True
                elif gesture == "Right":
                    # 回房間模式時，恢復當前房間高亮
                    self.selection_mode = "room"
                    self.update_control_mode_styles()
                    self.update_single_cell(self.current_row, self.current_col, True)
                    self.last_row = self.current_row
                    self.last_col = self.current_col
                    return True
                elif gesture == "Up":
                    self.adjust_light_level(+5)
                    return True
                elif gesture == "Down":
                    self.adjust_light_level(-5)
                    return True
                else:
                    return False

            # 冷氣模式
            if self.selection_mode == "ac":
                if gesture == "Right":
                    self.selection_mode = "light"
                    self.update_control_mode_styles()
                    return True
                elif gesture == "Up":
                    self.adjust_ac_temp(+1)
                    return True
                elif gesture == "Down":
                    self.adjust_ac_temp(-1)
                    return True
                else:
                    return False

            # 房間模式
            if self.selection_mode == "room":
                moved = False
                old_row = self.current_row
                old_col = self.current_col

                # 在最左欄 且 向左 => 進入燈光模式，並清掉九宮格高亮
                if gesture == "Left" and self.current_col == 0:
                    self.clear_grid_highlight()
                    self.selection_mode = "light"
                    self.update_control_mode_styles()
                    return True

                if gesture == "Up" and self.current_row > 0:
                    self.current_row -= 1
                    moved = True
                elif gesture == "Down" and self.current_row < 2:
                    self.current_row += 1
                    moved = True
                elif gesture == "Left" and self.current_col > 0:
                    self.current_col -= 1
                    moved = True
                elif gesture == "Right" and self.current_col < 2:
                    self.current_col += 1
                    moved = True
                elif gesture == "Tap":
                    room_id = self.room_numbers.get((self.current_row, self.current_col), "")
                    self.show_record_view(room_id)
                    return True

                if moved:
                    self.update_single_cell(old_row, old_col, False)
                    self.update_single_cell(self.current_row, self.current_col, True)
                    self.last_row = self.current_row
                    self.last_col = self.current_col

                return moved

        return False

    # ================== 更新手勢機率 / 顯示 ==================
    def update_probabilities(self, background_prob, down_prob, left_prob,
                             right_prob, tap_prob, up_prob, current_gesture):

        # 機率條雖然存在，但下方整塊 UI 已經隱藏，不會顯示
        self.bars["Background"].setValue(int(background_prob * 100))
        self.bars["Down"].setValue(int(down_prob * 100))
        self.bars["Left"].setValue(int(left_prob * 100))
        self.bars["Right"].setValue(int(right_prob * 100))
        self.bars["Tap"].setValue(int(tap_prob * 100))
        self.bars["Up"].setValue(int(up_prob * 100))

        self.current_gesture_label.setText(f"Current gesture: {current_gesture}")
        bg_color = self.gesture_colors.get(current_gesture, "#E0E0E0")
        self.current_gesture_label.setStyleSheet(f"""
            QLabel {{
                background-color: {bg_color};
                border-radius: 8px;
                padding: 6px;
                font-size: 12pt;
                font-weight: bold;
            }}
        """)

        # 給實際雷達辨識用的防重觸發邏輯
        if current_gesture != "Background" and current_gesture != self.last_gesture:
            self.move_position(current_gesture)
            self.last_gesture = current_gesture
        elif current_gesture == "Background":
            self.last_gesture = "Background"

    # ================== 鍵盤輸入 → 模擬手勢 & 緊急呼叫 ==================
    def keyPressEvent(self, event):
        key = event.key()

        # 緊急呼叫相關按鍵
        if key == Qt.Key_X:
            self.trigger_emergency_call()
            return

        if key == Qt.Key_G:
            self.handle_emergency_answer_or_hangup()
            return

        # 一般手勢
        gesture = None
        if key == Qt.Key_W:
            gesture = "Up"
        elif key == Qt.Key_S:
            gesture = "Down"
        elif key == Qt.Key_A:
            gesture = "Left"
        elif key == Qt.Key_D:
            gesture = "Right"
        elif key == Qt.Key_F:
            gesture = "Tap"
        elif key == Qt.Key_B:
            gesture = "Background"

        if gesture is None:
            super().keyPressEvent(event)
            return

        # 鍵盤：直接執行 move_position，不受 last_gesture 限制
        self.move_position(gesture)

        # 更新畫面上的手勢顯示與（隱藏中的）機率條
        probs = {g: 0.0 for g in ["Background", "Down", "Left", "Right", "Tap", "Up"]}
        probs[gesture] = 1.0

        self.current_gesture_label.setText(f"Current gesture: {gesture}")
        self.current_gesture_label.setStyleSheet(f"""
            QLabel {{
                background-color: {self.gesture_colors.get(gesture, '#E0E0E0')};
                border-radius: 8px;
                padding: 6px;
                font-size: 12pt;
                font-weight: bold;
            }}
        """)

        for g in self.gesture_names:
            self.bars[g].setValue(int(probs[g] * 100))


# ===== 單獨測試 GUI 用（鍵盤控制） =====
if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = GestureGUI()
    gui.show()
    sys.exit(app.exec_())
