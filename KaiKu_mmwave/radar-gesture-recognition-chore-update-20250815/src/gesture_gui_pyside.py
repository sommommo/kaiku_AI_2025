# gesture_gui_pyside.py
import sys
from PySide2.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QProgressBar,
    QHBoxLayout, QGridLayout, QFrame, QPushButton,
    QSizePolicy, QSpacerItem
)
from PySide2.QtCore import Qt, QTimer
from PySide2.QtGui import QFont, QPixmap


class GestureGUI(QWidget):
    """
    手勢辨識 GUI
    左側：學校 / 作品資訊 + 冷氣 / 燈光控制（含圖示）
    右側：3x3 房間格 (301~309)
    下方：手勢機率條
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("好米波 - Gesture Recognition Room Control")
        self.setMinimumSize(900, 600)

        # 當前位置 (row, col)，初始在 301
        self.current_row = 0
        self.current_col = 0
        self.last_row = 0
        self.last_col = 0

        # 記錄上一次手勢避免重複觸發
        self.last_gesture = "Background"

        # 3x3 房間編號全部填滿
        self.room_numbers = {
            (0, 0): "301",
            (0, 1): "302",
            (0, 2): "303",
            (1, 0): "304",
            (1, 1): "305",
            (1, 2): "306",
            (2, 0): "307",
            (2, 1): "308",
            (2, 2): "309",
        }

        # ========== 整體主版面：左側資訊區 + 右側房間區 ==========
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)  # 收窄外側留白
        main_layout.setSpacing(12)

        # -------- 左側資訊 / 控制區 --------
        left_panel = QFrame()
        left_panel.setFrameShape(QFrame.Box)
        left_panel.setStyleSheet("""
            QFrame {
                background-color: #FAFAFA;
                border-radius: 8px;
                border: 1px solid #BDBDBD;
            }
        """)
        left_panel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        left_panel.setMinimumWidth(220)

        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(14, 14, 14, 14)
        left_layout.setSpacing(16)

        # ====== 兩個 LOGO 區塊 ======
        logo_box = QVBoxLayout()
        logo_box.setSpacing(10)

        # ---- 虎尾科技大學 LOGO ----
        nfust_logo_label = QLabel()
        nfust_logo = QPixmap("logo_nfust.png")
        if not nfust_logo.isNull():
            nfust_logo_label.setPixmap(
                nfust_logo.scaled(120, 120, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
        nfust_logo_label.setAlignment(Qt.AlignCenter)
        logo_box.addWidget(nfust_logo_label)

        # ---- 長庚大學 LOGO ----
        cgu_logo_label = QLabel()
        cgu_logo = QPixmap("logo_cgu.png")
        if not cgu_logo.isNull():
            cgu_logo_label.setPixmap(
                cgu_logo.scaled(120, 120, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
        cgu_logo_label.setAlignment(Qt.AlignCenter)
        logo_box.addWidget(cgu_logo_label)

        left_layout.addLayout(logo_box)

        # ====== 學校名稱 ======
        school_label = QLabel("虎尾科技大學\n長庚大學")
        school_label.setFont(QFont("Microsoft JhengHei", 11))
        school_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(school_label)

        # ====== 作品名稱分離 ======
        work_label = QLabel("作品：好米波")
        work_label.setFont(QFont("Microsoft JhengHei", 12, QFont.Bold))
        work_label.setAlignment(Qt.AlignCenter)
        work_label.setStyleSheet("color: #333333; padding-top: 6px;")
        left_layout.addWidget(work_label)

        left_layout.addSpacerItem(QSpacerItem(10, 10, QSizePolicy.Minimum, QSizePolicy.Expanding))


        # ===== 冷氣控制區 =====
        ac_frame = QFrame()
        ac_frame.setStyleSheet("""
            QFrame {
                background-color: #FFFFFF;
                border-radius: 8px;
                border: 1px solid #E0E0E0;
            }
        """)
        ac_layout = QVBoxLayout(ac_frame)
        ac_layout.setContentsMargins(8, 8, 8, 8)
        ac_layout.setSpacing(6)

        ac_title_layout = QHBoxLayout()
        ac_icon_label = QLabel()
        ac_icon = QPixmap("ac_icon.png")
        if not ac_icon.isNull():
            ac_icon_label.setPixmap(ac_icon.scaled(22, 22, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        ac_icon_label.setFixedSize(24, 24)

        ac_title = QLabel("冷氣")
        ac_title.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        ac_title.setFont(QFont("Microsoft JhengHei", 11, QFont.Bold))
        ac_title_layout.addWidget(ac_icon_label)
        ac_title_layout.addWidget(ac_title)
        ac_title_layout.addStretch()
        ac_layout.addLayout(ac_title_layout)

        ac_btn_up = QPushButton("▲")
        ac_btn_down = QPushButton("▼")
        for btn in (ac_btn_up, ac_btn_down):
            btn.setCursor(Qt.PointingHandCursor)
            btn.setStyleSheet("""
                QPushButton {
                    border: none;
                    font-size: 16px;
                    padding: 2px;
                    min-height: 24px;
                }
                QPushButton:hover {
                    background-color: #E3F2FD;
                    border-radius: 4px;
                }
            """)
        ac_layout.addWidget(ac_btn_up, alignment=Qt.AlignCenter)
        ac_layout.addWidget(ac_btn_down, alignment=Qt.AlignCenter)
        left_layout.addWidget(ac_frame)

        # ===== 燈光控制區 =====
        light_frame = QFrame()
        light_frame.setStyleSheet("""
            QFrame {
                background-color: #FFFFFF;
                border-radius: 8px;
                border: 1px solid #E0E0E0;
            }
        """)
        light_layout = QVBoxLayout(light_frame)
        light_layout.setContentsMargins(8, 8, 8, 8)
        light_layout.setSpacing(6)

        light_title_layout = QHBoxLayout()
        light_icon_label = QLabel()
        light_icon = QPixmap("light_icon.png")
        if not light_icon.isNull():
            light_icon_label.setPixmap(light_icon.scaled(22, 22, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        light_icon_label.setFixedSize(24, 24)

        light_title = QLabel("燈光")
        light_title.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        light_title.setFont(QFont("Microsoft JhengHei", 11, QFont.Bold))
        light_title_layout.addWidget(light_icon_label)
        light_title_layout.addWidget(light_title)
        light_title_layout.addStretch()
        light_layout.addLayout(light_title_layout)

        light_btn_up = QPushButton("▲")
        light_btn_down = QPushButton("▼")
        for btn in (light_btn_up, light_btn_down):
            btn.setCursor(Qt.PointingHandCursor)
            btn.setStyleSheet("""
                QPushButton {
                    border: none;
                    font-size: 16px;
                    padding: 2px;
                    min-height: 24px;
                }
                QPushButton:hover {
                    background-color: #FFF3E0;
                    border-radius: 4px;
                }
            """)
        light_layout.addWidget(light_btn_up, alignment=Qt.AlignCenter)
        light_layout.addWidget(light_btn_down, alignment=Qt.AlignCenter)
        left_layout.addWidget(light_frame)

        main_layout.addWidget(left_panel)

        # -------- 右側房間 / 手勢顯示區 --------
        right_panel = QFrame()
        right_panel.setFrameShape(QFrame.Box)
        right_panel.setStyleSheet("""
            QFrame {
                background-color: #FFFFFF;
                border-radius: 8px;
                border: 1px solid #BDBDBD;
            }
        """)
        right_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(16, 12, 16, 12)
        right_layout.setSpacing(12)

        # 當前手勢標籤
        self.current_gesture_label = QLabel("Current gesture: Background")
        self.current_gesture_label.setAlignment(Qt.AlignCenter)
        self.current_gesture_label.setMinimumHeight(40)
        self.current_gesture_label.setFont(QFont("Microsoft JhengHei", 12, QFont.Bold))
        self.current_gesture_label.setStyleSheet("""
            QLabel {
                background-color: #EEEEEE;
                border-radius: 6px;
                padding: 6px;
            }
        """)
        right_layout.addWidget(self.current_gesture_label)

        # 房間 3x3 格子
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
                cell.setFont(QFont("Microsoft JhengHei", 16))
                room_text = self.room_numbers.get((row, col), "")
                cell.setText(room_text)
                cell.setStyleSheet("""
                    QLabel {
                        background-color: #F9F9F9;
                        border: 2px solid #BDBDBD;
                        border-radius: 10px;
                        color: #424242;
                    }
                """)
                cell.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                grid_layout.addWidget(cell, row, col)
                self.cells[(row, col)] = cell

        right_layout.addWidget(grid_frame)

        # ====== 手勢機率條 ======
        bar_title = QLabel("Gesture Probabilities")
        bar_title.setAlignment(Qt.AlignCenter)
        bar_title.setFont(QFont("Arial", 9, QFont.Bold))
        bar_title.setStyleSheet("color: #616161;")
        right_layout.addWidget(bar_title)

        bar_layout = QHBoxLayout()
        bar_layout.setSpacing(12)
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
            bar.setFixedWidth(16)
            bar.setMinimumHeight(80)
            bar.setStyleSheet(f"""
                QProgressBar {{
                    border: 1px solid #BDBDBD;
                    border-radius: 4px;
                    background: #F5F5F5;
                }}
                QProgressBar::chunk {{
                    background-color: {self.bar_colors[name]};
                    margin: 0px;
                }}
            """)
            v_layout.addWidget(bar, alignment=Qt.AlignBottom)

            label = QLabel(name)
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet("font-size: 8.5pt;")
            v_layout.addWidget(label)

            bar_layout.addLayout(v_layout)
            self.bars[name] = bar

        right_layout.addLayout(bar_layout)

        main_layout.addWidget(right_panel)
        main_layout.setStretch(0, 0)   # 左側固定寬
        main_layout.setStretch(1, 1)   # 右側自動撐滿，減少多餘留白

        # 初始化顯示目前位置
        self.update_grid_display()

    # ================== 房間格子顯示 ==================
    def update_single_cell(self, row, col, is_current):
        cell = self.cells[(row, col)]
        room_text = self.room_numbers.get((row, col), "")

        if is_current:
            cell.setText(room_text)
            cell.setStyleSheet("""
                QLabel {
                    background-color: #FFE0B2;
                    border: 3px solid #FB8C00;
                    border-radius: 10px;
                    color: #212121;
                    font-weight: bold;
                }
            """)
        else:
            cell.setText(room_text)
            cell.setStyleSheet("""
                QLabel {
                    background-color: #F9F9F9;
                    border: 2px solid #BDBDBD;
                    border-radius: 10px;
                    color: #424242;
                }
            """)

    def update_grid_display(self):
        if (self.last_row, self.last_col) != (self.current_row, self.current_col):
            self.update_single_cell(self.last_row, self.last_col, False)
        self.update_single_cell(self.current_row, self.current_col, True)
        self.last_row = self.current_row
        self.last_col = self.current_col

    # ================== 位置移動邏輯 ==================
    def move_position(self, gesture):
        moved = False
        old_row = self.current_row
        old_col = self.current_col

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
            # Tap：目前房間短暫高亮成紅色，模擬「確認」
            current_cell = self.cells[(self.current_row, self.current_col)]
            room_text = self.room_numbers.get((self.current_row, self.current_col), "")
            current_cell.setText(room_text + ("\n✓" if room_text else "✓"))
            current_cell.setStyleSheet("""
                QLabel {
                    background-color: #FFCCBC;
                    border: 3px solid #E64A19;
                    border-radius: 10px;
                    color: #BF360C;
                    font-weight: bold;
                }
            """)
            QTimer.singleShot(
                400, lambda: self.update_single_cell(self.current_row, self.current_col, True)
            )
            return True

        if moved:
            self.update_single_cell(old_row, old_col, False)
            self.update_single_cell(self.current_row, self.current_col, True)
            self.last_row = self.current_row
            self.last_col = self.current_col

        return moved

    # ================== 更新手勢機率 / 顯示 ==================
    def update_probabilities(self, background_prob, down_prob, left_prob,
                             right_prob, tap_prob, up_prob, current_gesture):

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
                border-radius: 6px;
                padding: 6px;
                font-size: 12pt;
                font-weight: bold;
            }}
        """)

        # 防止同一手勢連續觸發
        if current_gesture != "Background" and current_gesture != self.last_gesture:
            self.move_position(current_gesture)
            self.last_gesture = current_gesture
        elif current_gesture == "Background":
            self.last_gesture = "Background"


# 測試用：單獨跑這個檔可以看到 GUI 效果
if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = GestureGUI()
    gui.show()

    # 隨機模擬手勢機率
    import random

    def simulate_data():
        bg = random.uniform(0, 0.7)
        down = random.uniform(0, 1 - bg)
        left = random.uniform(0, 1 - bg - down)
        right = random.uniform(0, 1 - bg - down - left)
        tap = random.uniform(0, 1 - bg - down - left - right)
        up = 1 - (bg + down + left + right + tap)

        probs = {
            "Background": bg,
            "Down": down,
            "Left": left,
            "Right": right,
            "Tap": tap,
            "Up": up,
        }
        gesture = max(probs, key=probs.get)
        gui.update_probabilities(bg, down, left, right, tap, up, gesture)

    timer = QTimer()
    timer.timeout.connect(simulate_data)
    timer.start(2000)

    sys.exit(app.exec_())
