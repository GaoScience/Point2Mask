import sys
import os
import cv2
import json
import copy
import re
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QListWidget, QPushButton,
                             QSlider, QFileDialog, QSplitter, QFrame,
                             QSpinBox, QGroupBox, QComboBox, QCheckBox, QAction,
                             QDialog, QLineEdit, QGridLayout, QMenu, QScrollArea,
                             QScrollBar, QSizePolicy, QMessageBox, QListWidgetItem)
from PyQt5.QtCore import Qt, QPoint, QRect, QRectF, QSize, QThread, pyqtSignal, QObject, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont, QPainter, QPen, QColor, QBrush, QCursor, QPalette

try:
    from .pamg import pamg_grow_mask_fast
except ImportError:
    from pamg import pamg_grow_mask_fast


# ==========================================
# 2. 异步状态检查线程
# ==========================================
class MaskStatusWorker(QThread):
    update_signal = pyqtSignal(int, str)

    def __init__(self, file_list, mask_dir):
        super().__init__()
        self.file_list = file_list
        self.mask_dir = mask_dir
        self.is_running = True

    def run(self):
        for i, fname in enumerate(self.file_list):
            if not self.is_running: break
            p = os.path.join(self.mask_dir, os.path.splitext(fname)[0] + ".png")
            icon = "⬜"
            if os.path.exists(p):
                try:
                    m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                    if m is not None and cv2.countNonZero(m) > 0:
                        icon = "✅"
                    else:
                        icon = "⚠️"
                except:
                    icon = "⚠️"
            self.update_signal.emit(i, icon)

    def stop(self):
        self.is_running = False
        self.wait()


# ==========================================
# 3. 路径配置对话框
# ==========================================
class ConfigDialog(QDialog):
    def __init__(self, parent=None, defaults=None):
        super().__init__(parent)
        self.setWindowTitle("Project Configuration")
        self.resize(600, 220)
        self.paths = defaults or {"img": "", "label": "", "mask": ""}
        self.init_ui()

    def init_ui(self):
        layout = QGridLayout()
        layout.setVerticalSpacing(15)
        layout.setHorizontalSpacing(10)

        layout.addWidget(QLabel("1. Images Dir:"), 0, 0)
        layout.addWidget(QLabel("2. Masks Dir:"), 1, 0)
        layout.addWidget(QLabel("3. Labels Dir (Optional):"), 2, 0)

        self.le_img = QLineEdit(self.paths.get("img", ""))
        self.le_mask = QLineEdit(self.paths.get("mask", ""))
        self.le_label = QLineEdit(self.paths.get("label", ""))
        self.le_label.setPlaceholderText("YOLO .txt format")

        layout.addWidget(self.le_img, 0, 1)
        layout.addWidget(self.le_mask, 1, 1)
        layout.addWidget(self.le_label, 2, 1)

        btn_img = QPushButton("Browse...")
        btn_img.clicked.connect(lambda: self.browse("img"))
        layout.addWidget(btn_img, 0, 2)

        btn_mask = QPushButton("Browse...")
        btn_mask.clicked.connect(lambda: self.browse("mask"))
        layout.addWidget(btn_mask, 1, 2)

        btn_label = QPushButton("Browse...")
        btn_label.clicked.connect(lambda: self.browse("label"))
        layout.addWidget(btn_label, 2, 2)

        btn_ok = QPushButton("Initialize Project")
        btn_ok.setMinimumHeight(40)
        btn_ok.setStyleSheet("background-color: #0078d7; color: white; font-weight: bold; border-radius: 4px;")
        btn_ok.clicked.connect(self.accept)
        layout.addWidget(btn_ok, 4, 0, 1, 3)
        self.setLayout(layout)

    def browse(self, key):
        d = QFileDialog.getExistingDirectory(self, f"Select {key} directory")
        if d:
            if key == "img":
                self.le_img.setText(d)
            elif key == "label":
                self.le_label.setText(d)
            elif key == "mask":
                self.le_mask.setText(d)

    def get_paths(self):
        return {
            "img": self.le_img.text().strip(),
            "label": self.le_label.text().strip(),
            "mask": self.le_mask.text().strip()
        }


# ==========================================
# 4. 图像处理与绘制辅助
# ==========================================
def process_image_display(img_gray, main_window):
    if img_gray is None: return None
    res = img_gray.copy()
    if main_window and main_window.enable_clahe:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        res = clahe.apply(res)
    if main_window and main_window.gamma_val != 1.0:
        invGamma = 1.0 / main_window.gamma_val
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        res = cv2.LUT(res, table)
    if main_window and main_window.colormap_idx >= 0:
        res = cv2.applyColorMap(res, main_window.colormap_idx)
    else:
        res = cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)
    return res


def overlay_mask(bg_img, mask, main_window):
    if main_window is None or not main_window.mask_visible or mask is None:
        return bg_img
    if np.any(mask > 0):
        red_layer = np.zeros_like(bg_img)
        red_layer[mask > 0] = [0, 0, 255]
        alpha = main_window.mask_alpha
        mask_bool = mask > 0
        bg_img[mask_bool] = cv2.addWeighted(bg_img[mask_bool], 1.0 - alpha,
                                            red_layer[mask_bool], alpha, 0)
    return bg_img


def draw_info_overlay(painter, anchor_x, anchor_y, img_pos, val):
    """
    统一的HUD绘制函数
    """
    painter.setRenderHint(QPainter.TextAntialiasing)
    info_text = f"X: {img_pos[0]}  Y: {img_pos[1]}  Val: {val}"
    font = QFont("Consolas", 10, QFont.Bold)
    painter.setFont(font)
    metrics = painter.fontMetrics()
    tw, th = metrics.width(info_text), metrics.height()

    # 绘制背景框 (文字在 anchor 点的上方显示)
    rect_x = anchor_x + 5
    rect_y = anchor_y - th - 8

    painter.setPen(Qt.NoPen)
    painter.setBrush(QColor(30, 30, 30, 200))
    painter.drawRoundedRect(rect_x - 4, rect_y - 2, tw + 8, th + 4, 3, 3)
    painter.setPen(QColor(0, 255, 128))
    painter.drawText(rect_x, rect_y + metrics.ascent(), info_text)


# ==========================================
# 5. 左侧：HUD Overlay Widget (新增)
# ==========================================
class HudOverlay(QWidget):
    """
    一个透明的覆盖层，专门用于绘制左侧的HUD信息。
    它作为OverviewScrollArea的子控件存在，不随图片滚动。
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)  # 鼠标穿透
        self.setAttribute(Qt.WA_NoSystemBackground)  # 无背景
        self.info_data = None  # (img_pos, val)

    def update_info(self, data):
        self.info_data = data
        self.update()

    def paintEvent(self, event):
        if self.info_data:
            img_pos, val = self.info_data
            painter = QPainter(self)
            # 固定绘制在控件左下角
            draw_info_overlay(painter, 0, self.height(), img_pos, val)


# ==========================================
# 6. 左侧：ZoomableLabel
# ==========================================
class ZoomableLabel(QLabel):
    def __init__(self, parent=None, main_window=None, owner=None):
        super().__init__(parent)
        self.main_window = main_window
        self.owner = owner  # Owner is OverviewScrollArea
        self.setMouseTracking(True)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: transparent;")

        self.curr_mouse_pos = None
        self.curr_img_pos = None
        self.synced_crosshair_pos = None

        self.mode = 'idle'
        self.drag_start = None
        self.drag_current = None
        self.active_bbox_idx = -1
        self.resize_edge = None
        self._raw_pixmap = None

    def set_raw_image(self, img, mask):
        if img is None: return
        display_img = process_image_display(img, self.main_window)
        display_img = overlay_mask(display_img, mask, self.main_window)
        h, w = display_img.shape[:2]
        bytesPerLine = 3 * w
        qImg = QImage(display_img.data, w, h, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
        self._raw_pixmap = QPixmap.fromImage(qImg)

    def set_crosshair_pos(self, img_pos):
        self.synced_crosshair_pos = img_pos
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.main_window.full_img is None: return

        painter = QPainter(self)
        orig_h, orig_w = self.main_window.full_img.shape[:2]
        if orig_w == 0: return
        scale = self.width() / orig_w

        # 1. Draw BBoxes
        idx = self.main_window.curr_bbox_idx
        for i, bbox in enumerate(self.main_window.bboxes):
            x1, y1, x2, y2 = bbox
            sx1, sy1 = int(x1 * scale), int(y1 * scale)
            sx2, sy2 = int(x2 * scale), int(y2 * scale)
            w, h = sx2 - sx1, sy2 - sy1

            if i == idx:
                color = QColor(255, 0, 0)
                width = 2
            else:
                color = QColor(0, 255, 0)
                width = 2

            pen = QPen(color, width)
            if i == idx and self.mode in ['move_box', 'resize_box']: pen.setStyle(Qt.DashLine)

            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(sx1, sy1, w, h)

            if i == idx:
                self._draw_handles(painter, sx1, sy1, sx2, sy2)
                painter.setPen(color)
                painter.drawText(sx1, max(0, sy1 - 5), f"#{i + 1}")

        # 2. Creating Box
        if self.mode == 'create_box' and self.drag_start and self.drag_current:
            x1 = min(self.drag_start.x(), self.drag_current.x())
            y1 = min(self.drag_start.y(), self.drag_current.y())
            w_box = abs(self.drag_start.x() - self.drag_current.x())
            h_box = abs(self.drag_start.y() - self.drag_current.y())
            painter.setPen(QPen(QColor(0, 120, 255), 1, Qt.DashLine))
            painter.drawRect(x1, y1, w_box, h_box)

        # 3. Crosshair & HUD Info Calculation
        draw_pos_img = None
        under_mouse = self.underMouse()

        # 优先使用鼠标位置，否则使用同步位置
        if under_mouse and self.curr_img_pos:
            draw_pos_img = self.curr_img_pos
        elif self.synced_crosshair_pos:
            draw_pos_img = self.synced_crosshair_pos

        if draw_pos_img:
            ix, iy = draw_pos_img
            mx, my = int(ix * scale), int(iy * scale)

            # 画十字线
            pen = QPen(QColor(255, 0, 0, 180), 1)
            painter.setPen(pen)
            painter.drawLine(0, my, self.width(), my)
            painter.drawLine(mx, 0, mx, self.height())

            # 计算数值并发送给 Parent (Overlay)
            val = 0
            if 0 <= iy < orig_h and 0 <= ix < orig_w:
                val = self.main_window.full_img[iy, ix]

            # 核心修改：不自己画文字，而是通知 parent 更新覆盖层
            if self.owner:
                self.owner.update_hud(draw_pos_img, val)
        else:
            # 如果没有十字线，通知 parent 清空
            if self.owner:
                self.owner.update_hud(None, 0)

    def _draw_handles(self, painter, x1, y1, x2, y2):
        r = 4
        painter.setBrush(QColor(255, 255, 255))
        painter.setPen(QColor(0, 0, 0))
        points = [(x1, y1), (x2, y1), (x1, y2), (x2, y2),
                  ((x1 + x2) // 2, y1), ((x1 + x2) // 2, y2), (x1, (y1 + y2) // 2), (x2, (y1 + y2) // 2)]
        for px, py in points: painter.drawRect(px - r, py - r, r * 2, r * 2)

    def get_img_coords(self, widget_pos):
        if self.main_window.full_img is None: return None
        orig_h, orig_w = self.main_window.full_img.shape[:2]
        if self.width() == 0: return None
        scale = orig_w / self.width()
        ix, iy = int(widget_pos.x() * scale), int(widget_pos.y() * scale)
        return (max(0, min(orig_w - 1, ix)), max(0, min(orig_h - 1, iy)))

    def get_handle_at(self, pos, bbox, scale):
        x1, y1, x2, y2 = bbox
        sx1, sy1, sx2, sy2 = int(x1 * scale), int(y1 * scale), int(x2 * scale), int(y2 * scale)
        mx, my = pos.x(), pos.y()
        r = 8
        if abs(mx - sx1) < r and abs(my - sy1) < r: return 'tl'
        if abs(mx - sx2) < r and abs(my - sy1) < r: return 'tr'
        if abs(mx - sx1) < r and abs(my - sy2) < r: return 'bl'
        if abs(mx - sx2) < r and abs(my - sy2) < r: return 'br'
        if abs(my - sy1) < r and sx1 < mx < sx2: return 't'
        if abs(my - sy2) < r and sx1 < mx < sx2: return 'b'
        if abs(mx - sx1) < r and sy1 < my < sy2: return 'l'
        if abs(mx - sx2) < r and sy1 < my < sy2: return 'r'
        if sx1 < mx < sx2 and sy1 < my < sy2: return 'inside'
        return None

    def mousePressEvent(self, event):
        if self.main_window.full_img is None: return
        img_coords = self.get_img_coords(event.pos())
        if not img_coords: return
        ix, iy = img_coords
        is_ctrl = (QApplication.keyboardModifiers() & Qt.ControlModifier)

        if is_ctrl and event.button() == Qt.LeftButton:
            orig_w = self.main_window.full_img.shape[1]
            scale = self.width() / orig_w
            hit_idx, handle_type = -1, None
            if self.main_window.curr_bbox_idx != -1:
                idx = self.main_window.curr_bbox_idx
                if 0 <= idx < len(self.main_window.bboxes):
                    ht = self.get_handle_at(event.pos(), self.main_window.bboxes[idx], scale)
                    if ht: hit_idx, handle_type = idx, ht
            if hit_idx == -1:
                for i, bbox in enumerate(self.main_window.bboxes):
                    ht = self.get_handle_at(event.pos(), bbox, scale)
                    if ht: hit_idx, handle_type = i, ht; break
            if hit_idx != -1:
                self.main_window.push_bbox_history()
                self.main_window.curr_bbox_idx = hit_idx
                self.active_bbox_idx = hit_idx
                self.main_window.refresh_all_views()
                if handle_type == 'inside':
                    self.mode = 'move_box';
                    self.setCursor(Qt.SizeAllCursor)
                else:
                    self.mode = 'resize_box';
                    self.resize_edge = handle_type
                    self.setCursor(Qt.SizeVerCursor)  # 简化光标设置
                self.drag_start = (ix, iy)
            else:
                self.mode = 'create_box'
                self.drag_start = event.pos();
                self.drag_current = event.pos()
                self.setCursor(Qt.CrossCursor)
                self.main_window.curr_bbox_idx = -1
                self.main_window.refresh_all_views()
        elif event.button() == Qt.LeftButton:
            orig_w = self.main_window.full_img.shape[1]
            scale = self.width() / orig_w
            clicked_box = False
            for i, bbox in enumerate(self.main_window.bboxes):
                if self.get_handle_at(event.pos(), bbox, scale) == 'inside':
                    self.main_window.curr_bbox_idx = i
                    self.main_window.refresh_all_views()
                    clicked_box = True
                    break
            if not clicked_box:
                self.main_window.curr_bbox_idx = -1
                self.main_window.refresh_all_views()
            self.mode = 'pan'
            self.setCursor(Qt.OpenHandCursor)
            self.owner.is_panning = True
            self.owner.last_pan_pos = event.globalPos()
        elif event.button() == Qt.RightButton:
            orig_w = self.main_window.full_img.shape[1]
            scale = self.width() / orig_w
            for i, bbox in enumerate(self.main_window.bboxes):
                if self.get_handle_at(event.pos(), bbox, scale) == 'inside':
                    self.main_window.curr_bbox_idx = i;
                    self.update()
                    menu = QMenu(self)
                    del_act = menu.addAction('Delete BBox')
                    del_act.triggered.connect(self.main_window.delete_current_bbox)
                    menu.exec_(event.globalPos())
                    break

    def mouseMoveEvent(self, event):
        img_coords = self.get_img_coords(event.pos())
        if img_coords:
            ix, iy = img_coords
            self.curr_mouse_pos = event.pos()
            self.curr_img_pos = (ix, iy)
            if self.main_window:
                self.main_window.sync_cursor(self.curr_img_pos, sender=self)
            self.update()

            if self.mode == 'idle':
                orig_w = self.main_window.full_img.shape[1]
                scale = self.width() / orig_w
                ht = None
                if self.main_window.curr_bbox_idx != -1:
                    ht = self.get_handle_at(event.pos(), self.main_window.bboxes[self.main_window.curr_bbox_idx], scale)
                if ht == 'inside':
                    self.setCursor(Qt.SizeAllCursor)
                elif ht:
                    self.setCursor(Qt.ArrowCursor)
                else:
                    self.setCursor(Qt.ArrowCursor)

        if self.mode == 'pan':
            delta = event.globalPos() - self.owner.last_pan_pos
            self.owner.last_pan_pos = event.globalPos()
            self.owner.horizontalScrollBar().setValue(self.owner.horizontalScrollBar().value() - delta.x())
            self.owner.verticalScrollBar().setValue(self.owner.verticalScrollBar().value() - delta.y())
            self.setCursor(Qt.ClosedHandCursor)
        elif self.mode == 'create_box':
            self.drag_current = event.pos();
            self.update()
        elif self.mode == 'move_box' and img_coords:
            ix, iy = img_coords
            dx, dy = ix - self.drag_start[0], iy - self.drag_start[1]
            x1, y1, x2, y2 = self.main_window.bboxes[self.active_bbox_idx]
            w, h = x2 - x1, y2 - y1
            H, W = self.main_window.full_img.shape[:2]
            nx1, ny1 = max(0, min(W - w, x1 + dx)), max(0, min(H - h, y1 + dy))
            self.main_window.bboxes[self.active_bbox_idx] = (nx1, ny1, nx1 + w, ny1 + h)
            self.drag_start = (ix, iy)
            self.main_window.refresh_all_views()
        elif self.mode == 'resize_box' and img_coords:
            ix, iy = img_coords
            x1, y1, x2, y2 = self.main_window.bboxes[self.active_bbox_idx]
            edge = self.resize_edge
            min_s = 2
            if 'l' in edge: x1 = min(ix, x2 - min_s)
            if 'r' in edge: x2 = max(ix, x1 + min_s)
            if 't' in edge: y1 = min(iy, y2 - min_s)
            if 'b' in edge: y2 = max(iy, y1 + min_s)
            H, W = self.main_window.full_img.shape[:2]
            x1, x2, y1, y2 = max(0, x1), min(W, x2), max(0, y1), min(H, y2)
            self.main_window.bboxes[self.active_bbox_idx] = (x1, y1, x2, y2)
            self.main_window.refresh_all_views()

    def mouseReleaseEvent(self, event):
        if self.mode == 'create_box' and self.drag_start and self.drag_current:
            p1 = self.get_img_coords(self.drag_start)
            p2 = self.get_img_coords(self.drag_current)
            if p1 and p2:
                x1, y1 = min(p1[0], p2[0]), min(p1[1], p2[1])
                x2, y2 = max(p1[0], p2[0]), max(p1[1], p2[1])
                if (x2 - x1) > 2 and (y2 - y1) > 2:
                    self.main_window.push_bbox_history()
                    self.main_window.add_new_bbox((x1, y1, x2, y2))
        self.mode = 'idle'
        self.owner.is_panning = False
        self.setCursor(Qt.ArrowCursor)
        self.update()


# ==========================================
# 7. 左侧：OverviewScrollArea (Container)
# ==========================================
class OverviewScrollArea(QScrollArea):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.setWidgetResizable(False)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: #F0F0F0; border: 1px solid #CCC;")

        self.label = ZoomableLabel(main_window=main_window, owner=self)
        self.setWidget(self.label)

        # 初始化覆盖层 (HUD)
        # 注意：Overlay是ScrollArea的子控件，不是Viewport的子控件
        self.hud_overlay = HudOverlay(self)

        self.scale_factor = 1.0
        self.is_panning = False
        self.last_pan_pos = QPoint()

    def update_hud(self, pos, val):
        if pos is None:
            self.hud_overlay.update_info(None)
        else:
            self.hud_overlay.update_info((pos, val))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # 调整HUD Overlay大小以覆盖整个显示区域
        self.hud_overlay.setGeometry(0, 0, self.width(), self.height())

    def update_content(self):
        if self.main_window.full_img is None: return
        self.label.set_raw_image(self.main_window.full_img, self.main_window.full_mask)
        self.apply_zoom()

    def apply_zoom(self):
        if self.label._raw_pixmap is None: return
        base_w = self.label._raw_pixmap.width()
        base_h = self.label._raw_pixmap.height()
        new_w, new_h = int(base_w * self.scale_factor), int(base_h * self.scale_factor)
        self.label.resize(new_w, new_h)
        scaled_pix = self.label._raw_pixmap.scaled(new_w, new_h, Qt.KeepAspectRatio, Qt.FastTransformation)
        self.label.setPixmap(scaled_pix)
        self.label.update()

    def wheelEvent(self, event):
        if QApplication.keyboardModifiers() & Qt.ControlModifier:
            old_pos = self.label.mapFromGlobal(QCursor.pos())
            delta = 1.2 if event.angleDelta().y() > 0 else 0.8
            new_scale = max(0.1, min(self.scale_factor * delta, 20.0))
            self.scale_factor = new_scale
            self.apply_zoom()
            new_x, new_y = old_pos.x() * delta, old_pos.y() * delta
            vp_pos = self.mapFromGlobal(QCursor.pos())
            self.horizontalScrollBar().setValue(int(new_x - vp_pos.x()))
            self.verticalScrollBar().setValue(int(new_y - vp_pos.y()))
            self.main_window.update_zoom_label(self.scale_factor)
        else:
            super().wheelEvent(event)


# ==========================================
# 8. 右侧：InteractionCanvas (Fixed BG)
# ==========================================
class InteractionCanvas(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background-color: #F0F0F0; border: 1px solid #CCC;")
        self.setMinimumSize(300, 300)
        self.setAlignment(Qt.AlignCenter)
        self.setMouseTracking(True)
        self.main_window = None
        self.display_scale = 10
        self.last_brush_pos = None
        self.curr_mouse_pos = None
        self.curr_img_pos = None
        self.synced_crosshair_pos = None
        self.setCursor(Qt.ArrowCursor)

    def leaveEvent(self, event):
        self.setCursor(Qt.ArrowCursor)
        super().leaveEvent(event)

    def update_display(self):
        if self.main_window is None or self.main_window.full_img is None:
            self.setText("No Image Loaded")
            self.setStyleSheet("background-color: #F0F0F0; color: #888; border: 1px solid #CCC;")
            return

        bbox = self.main_window.get_current_bbox()
        if not bbox:
            self.setText("Select Target in Left View\nto Analyze/Edit")
            self.setStyleSheet("background-color: #F0F0F0; color: #666; border: 1px solid #CCC;")
            return

        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1: return

        self.setStyleSheet("background-color: #F0F0F0; border: 2px solid #0078d7;")

        roi_img = self.main_window.full_img[y1:y2, x1:x2]
        roi_mask = self.main_window.full_mask[y1:y2, x1:x2]

        display_img = process_image_display(roi_img, self.main_window)
        display_img = overlay_mask(display_img, roi_mask, self.main_window)

        h, w = display_img.shape[:2]
        scaled_h, scaled_w = int(h * self.display_scale), int(w * self.display_scale)
        if scaled_h <= 0: return
        display_img_big = cv2.resize(display_img, (scaled_w, scaled_h), interpolation=cv2.INTER_NEAREST)

        h_big, w_big, ch = display_img_big.shape
        bytesPerLine = 3 * w_big
        qImg = QImage(display_img_big.data, w_big, h_big, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
        self.setPixmap(QPixmap.fromImage(qImg))

    def set_crosshair_pos(self, img_pos):
        self.synced_crosshair_pos = img_pos
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)

        draw_pos_img = None
        if self.underMouse() and self.curr_img_pos:
            draw_pos_img = self.curr_img_pos
        elif self.synced_crosshair_pos:
            draw_pos_img = self.synced_crosshair_pos

        if draw_pos_img and self.pixmap() and self.main_window and self.main_window.get_current_bbox():
            gx, gy = draw_pos_img
            x1, y1, x2, y2 = self.main_window.get_current_bbox()

            if x1 <= gx < x2 and y1 <= gy < y2:
                rx, ry = gx - x1, gy - y1
                pm = self.pixmap()
                off_x = (self.width() - pm.width()) // 2
                off_y = (self.height() - pm.height()) // 2

                mx = off_x + int((rx + 0.5) * self.display_scale)
                my = off_y + int((ry + 0.5) * self.display_scale)

                pen = QPen(QColor(255, 0, 0, 150), 1)
                painter.setPen(pen)
                painter.drawLine(0, my, self.width(), my)
                painter.drawLine(mx, 0, mx, self.height())

                modifiers = QApplication.keyboardModifiers()
                is_shift = (modifiers & Qt.ShiftModifier)
                if self.underMouse() and not is_shift:
                    painter.setRenderHint(QPainter.Antialiasing)
                    radius_px = (self.main_window.brush_size * self.display_scale) / 2.0
                    is_right = (QApplication.mouseButtons() & Qt.RightButton)
                    fill = QColor(255, 255, 255, 120) if is_right else QColor(0, 120, 255, 80)
                    border = QColor(0, 0, 0) if is_right else QColor(255, 255, 255)
                    painter.setPen(QPen(border, 1))
                    painter.setBrush(fill)
                    painter.drawEllipse(QPoint(mx, my), int(radius_px), int(radius_px))

                val = 0
                h, w = self.main_window.full_img.shape[:2]
                if 0 <= gy < h and 0 <= gx < w:
                    val = self.main_window.full_img[gy, gx]

                # 右侧HUD绘制：直接调用通用函数，anchor设在widget左下角
                draw_info_overlay(painter, 0, self.height(), (gx, gy), val)

    def get_global_coords(self, pos):
        if self.pixmap() is None: return None
        bbox = self.main_window.get_current_bbox()
        if not bbox: return None
        x1, y1, x2, y2 = bbox
        roi_w, roi_h = x2 - x1, y2 - y1
        pm = self.pixmap()
        off_x, off_y = (self.width() - pm.width()) // 2, (self.height() - pm.height()) // 2
        mx, my = pos.x() - off_x, pos.y() - off_y
        rx, ry = int(mx / self.display_scale), int(my / self.display_scale)
        if 0 <= rx < roi_w and 0 <= ry < roi_h: return (y1 + ry, x1 + rx)
        return None

    def mousePressEvent(self, e):
        coords = self.get_global_coords(e.pos())
        if not coords: return
        gy, gx = coords
        self.main_window.push_mask_history()

        modifiers = QApplication.keyboardModifiers()
        is_shift = (modifiers & Qt.ShiftModifier)

        if is_shift and e.button() == Qt.LeftButton:
            Rs = self.main_window.current_Rs
            m = self.main_window.current_mode
            new_mask = pamg_grow_mask_fast((gy, gx), self.main_window.full_img, Rs=Rs, mode=m)
            self.main_window.full_mask = cv2.bitwise_or(self.main_window.full_mask, new_mask)
            self.main_window.lbl_status.setText(f"Seed Grown at ({gx},{gy})")
        elif e.button() == Qt.LeftButton:
            self.draw_brush(gx, gy, 255);
            self.last_brush_pos = (gx, gy)
        elif e.button() == Qt.RightButton:
            self.draw_brush(gx, gy, 0);
            self.last_brush_pos = (gx, gy)
        self.main_window.refresh_all_views()

    def mouseMoveEvent(self, e):
        self.curr_mouse_pos = e.pos()
        coords = self.get_global_coords(e.pos())
        if coords:
            self.setCursor(Qt.BlankCursor)
        else:
            self.setCursor(Qt.ArrowCursor)
        if coords:
            self.curr_img_pos = (coords[1], coords[0])
            if self.main_window:
                self.main_window.sync_cursor(self.curr_img_pos, sender=self)
        self.update()
        if coords:
            gy, gx = coords
            modifiers = QApplication.keyboardModifiers()
            is_shift = (modifiers & Qt.ShiftModifier)
            if not is_shift and (gx, gy) != self.last_brush_pos:
                if e.buttons() & Qt.LeftButton:
                    self.draw_brush(gx, gy, 255);
                    self.last_brush_pos = (gx, gy);
                    self.update_display()
                elif e.buttons() & Qt.RightButton:
                    self.draw_brush(gx, gy, 0);
                    self.last_brush_pos = (gx, gy);
                    self.update_display()

    def draw_brush(self, x, y, val):
        r = self.main_window.brush_size
        cv2.circle(self.main_window.full_mask, (x, y), r // 2, val, -1)

    def wheelEvent(self, e):
        if QApplication.keyboardModifiers() & Qt.ControlModifier:
            delta = 1 if e.angleDelta().y() > 0 else -1
            self.display_scale = max(1, min(50, self.display_scale + delta))
            self.main_window.spin_zoom.setValue(self.display_scale)
            self.update_display()
        else:
            delta = 1 if e.angleDelta().y() > 0 else -1
            self.main_window.brush_size = max(1, min(50, self.main_window.brush_size + delta))
            self.main_window.lbl_status.setText(f"Brush Size: {self.main_window.brush_size}")
            self.update()


# ==========================================
# 9. 主窗口 (SIRSTLabeler)
# ==========================================
class SIRSTLabeler(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Label-IRST")

        self.setStyleSheet("""
            QWidget { font-family: 'Segoe UI', sans-serif; font-size: 13px; color: #333; }
            QMainWindow { background-color: #F0F0F0; }
            QGroupBox { font-weight: bold; border: 1px solid #ccc; border-radius: 4px; margin-top: 10px; background-color: #fff; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; color: #333; }
            QPushButton { border: 1px solid #aaa; border-radius: 3px; padding: 4px; background-color: #f8f8f8; color: #333; }
            QPushButton:hover { background-color: #e0e0e0; }
            QPushButton:pressed { background-color: #d0d0d0; }
            QListWidget { background-color: #fff; border: 1px solid #ccc; }
            QListWidget::item:selected { background-color: #0078d7; color: white; }
            QLabel { color: #333; }
        """)

        self.config_file = "labeler_config.json"
        self.paths = self.load_config()
        self.current_Rs = 30
        self.current_mode = 1
        self.mask_visible = True
        self.mask_alpha = 0.4
        self.gamma_val = 1.0
        self.colormap_idx = -1
        self.brush_size = 2
        self.enable_clahe = False
        self.bbox_padding = 20

        self.full_img = None
        self.full_mask = None
        self.raw_yolo_data = []
        self.bboxes = []
        self.curr_bbox_idx = -1
        self.file_list = []
        self.curr_file_idx = 0

        self.mask_history = []
        self.bbox_history = []
        self.mask_check_thread = None

        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self.play_next_frame)
        self.is_playing = False

        if not self.configure_paths(): sys.exit(0)
        self.init_ui()
        self.adapt_resolution()
        self.load_first_image()

    def load_config(self):
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {"img": "", "label": "", "mask": ""}

    def save_config(self):
        with open(self.config_file, 'w') as f: json.dump(self.paths, f)

    def configure_paths(self):
        dlg = ConfigDialog(self, self.paths)
        if dlg.exec_() == QDialog.Accepted:
            self.paths = dlg.get_paths()
            self.save_config()
            return True
        return False

    def adapt_resolution(self):
        screen = QApplication.primaryScreen().geometry()
        w, h = screen.width(), screen.height()
        self.resize(int(w * 0.9), int(h * 0.9))
        self.move((w - self.width()) // 2, (h - self.height()) // 2)

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        body_layout = QHBoxLayout()

        sidebar = QVBoxLayout()
        btn_cfg = QPushButton("Workspace Config")
        btn_cfg.clicked.connect(self.reconfigure)
        sidebar.addWidget(btn_cfg)
        self.list_widget = QListWidget()
        self.list_widget.setAlternatingRowColors(True)
        self.list_widget.currentRowChanged.connect(self.on_file_changed)
        sidebar.addWidget(self.list_widget)
        self.lbl_file_count = QLabel("0/0")
        self.lbl_file_count.setAlignment(Qt.AlignCenter)
        sidebar.addWidget(self.lbl_file_count)
        f_side = QFrame()
        f_side.setLayout(sidebar)
        f_side.setFixedWidth(280)
        body_layout.addWidget(f_side)

        self.splitter = QSplitter(Qt.Horizontal)
        self.view_overview = OverviewScrollArea(self)
        self.view_interact = InteractionCanvas()
        self.view_interact.main_window = self

        def wrap(w, t, right_widget=None):
            c = QWidget()
            l = QVBoxLayout(c)
            l.setContentsMargins(0, 0, 0, 0)
            l.setSpacing(2)
            header = QHBoxLayout()
            h_bg = QFrame()
            h_bg.setStyleSheet("background-color: #E5E5E5; border-bottom: 1px solid #CCC;")
            h_layout = QHBoxLayout(h_bg)
            h_layout.setContentsMargins(5, 2, 5, 2)
            lb = QLabel(t)
            lb.setStyleSheet("font-weight: bold; color: #333;")
            h_layout.addWidget(lb)
            h_layout.addStretch()
            if isinstance(w, OverviewScrollArea):
                self.lbl_zoom_info = QLabel("100%")
                self.lbl_zoom_info.setStyleSheet("color: #0055aa; font-weight: bold;")
                h_layout.addWidget(self.lbl_zoom_info)
            if right_widget: h_layout.addWidget(right_widget)
            l.addWidget(h_bg)
            l.addWidget(w, 1)
            return c

        btn_clear = QPushButton("Clear Mask")
        btn_clear.setStyleSheet("color: red; font-weight: bold; font-size: 11px; padding: 2px 6px;")
        btn_clear.setFixedWidth(80)
        btn_clear.clicked.connect(self.clear_mask)

        self.splitter.addWidget(wrap(self.view_overview, "Overview"))
        self.splitter.addWidget(
            wrap(self.view_interact, "Detail View (Seed/Brush) [Hold Ctrl to hide Mask]", btn_clear))
        self.splitter.setSizes([600, 900])
        body_layout.addWidget(self.splitter, 1)
        main_layout.addLayout(body_layout)

        ctrl_panel = QWidget()
        ctrl_panel.setFixedHeight(140)
        gl = QGridLayout(ctrl_panel)
        gl.setContentsMargins(5, 5, 5, 5)
        gl.setHorizontalSpacing(15)

        gb_box = QGroupBox("BBox / Zoom")
        v_box = QVBoxLayout(gb_box)
        l_pad = QHBoxLayout()
        l_pad.addWidget(QLabel("Pad:"))
        self.spin_pad = QSpinBox();
        self.spin_pad.setRange(0, 200);
        self.spin_pad.setValue(20)
        self.spin_pad.valueChanged.connect(self.on_padding_change)
        l_pad.addWidget(self.spin_pad)
        v_box.addLayout(l_pad)
        l_zoom = QHBoxLayout()
        l_zoom.addWidget(QLabel("R-Scale:"))
        self.spin_zoom = QSpinBox();
        self.spin_zoom.setRange(1, 40);
        self.spin_zoom.setValue(10)
        self.spin_zoom.valueChanged.connect(self.on_zoom_change)
        l_zoom.addWidget(self.spin_zoom)
        v_box.addLayout(l_zoom)
        gl.addWidget(gb_box, 0, 0)

        gb_algo = QGroupBox("Seed Grow")
        v_algo = QVBoxLayout(gb_algo)
        l_rad = QHBoxLayout()
        l_rad.addWidget(QLabel("Rs:"))
        self.spin_Rs = QSpinBox();
        self.spin_Rs.setRange(5, 100);
        self.spin_Rs.setValue(30)
        l_rad.addWidget(self.spin_Rs)
        v_algo.addLayout(l_rad)
        l_mode = QHBoxLayout()
        l_mode.addWidget(QLabel("Mode:"))
        self.combo_mode = QComboBox();
        self.combo_mode.addItems(["1:Bright", "-1:Dark", "0:Auto"])
        l_mode.addWidget(self.combo_mode)
        v_algo.addLayout(l_mode)
        btn_apply = QPushButton("Apply Algo")
        btn_apply.clicked.connect(self.apply_settings)
        v_algo.addWidget(btn_apply)
        gl.addWidget(gb_algo, 0, 1)

        gb_vis = QGroupBox("Visualization")
        v_vis = QVBoxLayout(gb_vis)
        self.chk_clahe = QCheckBox("CLAHE Enhance")
        self.chk_clahe.stateChanged.connect(
            lambda s: setattr(self, 'enable_clahe', s == Qt.Checked) or self.refresh_all_views())
        v_vis.addWidget(self.chk_clahe)
        l_cmap = QHBoxLayout()
        l_cmap.addWidget(QLabel("Color:"))
        self.combo_color = QComboBox();
        self.combo_color.addItems(["Gray", "JET", "OCEAN"])
        self.combo_color.currentIndexChanged.connect(self.on_color_change)
        l_cmap.addWidget(self.combo_color)
        v_vis.addLayout(l_cmap)
        gl.addWidget(gb_vis, 0, 2)

        gb_enh = QGroupBox("Mask Overlay")
        v_enh = QVBoxLayout(gb_enh)
        l_g = QHBoxLayout()
        l_g.addWidget(QLabel("Gamma:"))
        self.s_g = QSlider(Qt.Horizontal);
        self.s_g.setRange(5, 40);
        self.s_g.setValue(10)
        self.s_g.valueChanged.connect(lambda v: self.update_vis('gamma', v))
        l_g.addWidget(self.s_g)
        v_enh.addLayout(l_g)
        l_a = QHBoxLayout()
        l_a.addWidget(QLabel("Alpha:"))
        self.s_a = QSlider(Qt.Horizontal);
        self.s_a.setRange(1, 10);
        self.s_a.setValue(4)
        self.s_a.valueChanged.connect(lambda v: self.update_vis('alpha', v))
        l_a.addWidget(self.s_a)
        v_enh.addLayout(l_a)
        btn_reset = QPushButton("Reset Params")
        btn_reset.clicked.connect(self.reset_params)
        v_enh.addWidget(btn_reset)
        gl.addWidget(gb_enh, 0, 3)

        gb_nav = QGroupBox("Navigation")
        v_nav = QVBoxLayout(gb_nav)
        l_play = QHBoxLayout()
        self.btn_play = QPushButton("▶ Play (Space)")
        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_stop = QPushButton("■ Stop")
        self.btn_stop.clicked.connect(self.stop_play)
        self.spin_speed = QSpinBox()
        self.spin_speed.setRange(10, 2000);
        self.spin_speed.setValue(100)
        self.spin_speed.setSuffix(" ms");
        self.spin_speed.setFixedWidth(70)
        self.spin_speed.valueChanged.connect(self.update_speed)
        l_play.addWidget(self.btn_play);
        l_play.addWidget(self.btn_stop)
        l_play.addWidget(QLabel("Spd:"));
        l_play.addWidget(self.spin_speed)
        v_nav.addLayout(l_play)
        l_btns = QHBoxLayout()
        btn_prev = QPushButton("<< Prev (A)");
        btn_prev.clicked.connect(lambda: self.switch_file(-1))
        l_btns.addWidget(btn_prev)
        btn_next = QPushButton("Next (D) >>");
        btn_next.clicked.connect(lambda: self.switch_file(1))
        btn_next.setStyleSheet("background-color: #28a745; color: white; font-weight: bold; border: 1px solid #1e7e34;")
        l_btns.addWidget(btn_next)
        v_nav.addLayout(l_btns)
        gl.addWidget(gb_nav, 0, 4)
        gl.setColumnStretch(0, 1);
        gl.setColumnStretch(1, 1);
        gl.setColumnStretch(2, 1);
        gl.setColumnStretch(3, 1);
        gl.setColumnStretch(4, 2)
        main_layout.addWidget(ctrl_panel)
        self.lbl_status = QLabel("Ready")
        self.lbl_status.setStyleSheet("color: #666; margin-left: 5px;")
        main_layout.addWidget(self.lbl_status)

    def toggle_play(self):
        if self.is_playing:
            self.stop_play()
        else:
            self.start_play()

    def start_play(self):
        self.is_playing = True
        self.btn_play.setText("⏸ Pause (Space)")
        self.play_timer.start(self.spin_speed.value())

    def stop_play(self):
        self.is_playing = False
        self.btn_play.setText("▶ Play (Space)")
        self.play_timer.stop()

    def play_next_frame(self):
        if self.curr_file_idx < len(self.file_list) - 1:
            self.switch_file(1)
        else:
            self.stop_play(); self.lbl_status.setText("Playback Finished.")

    def update_speed(self, val):
        if self.is_playing: self.play_timer.setInterval(val)

    def update_zoom_label(self, scale):
        self.lbl_zoom_info.setText(f"{int(scale * 100)}%")

    def sync_cursor(self, img_pos, sender=None):
        if sender == self.view_overview.label:
            self.view_interact.set_crosshair_pos(img_pos)
        elif sender == self.view_interact:
            self.view_overview.label.set_crosshair_pos(img_pos)

    def on_padding_change(self, val):
        self.bbox_padding = val
        self.recalc_bboxes_from_raw()
        self.refresh_all_views()

    def recalc_bboxes_from_raw(self):
        if self.full_img is None: return
        h, w = self.full_img.shape[:2]
        self.bboxes = []
        for (cx, cy, bw, bh) in self.raw_yolo_data:
            x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
            x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
            pad = self.bbox_padding
            self.bboxes.append((max(0, x1 - pad), max(0, y1 - pad), min(w, x2 + pad), min(h, y2 + pad)))

    def add_new_bbox(self, box):
        self.bboxes.append(box)
        self.curr_bbox_idx = len(self.bboxes) - 1
        self.refresh_all_views()
        self.lbl_status.setText("New BBox Created")

    def delete_current_bbox(self):
        if 0 <= self.curr_bbox_idx < len(self.bboxes):
            self.push_bbox_history()
            self.bboxes.pop(self.curr_bbox_idx)
            if self.curr_bbox_idx >= len(self.bboxes): self.curr_bbox_idx = len(self.bboxes) - 1
            self.refresh_all_views()

    def reconfigure(self):
        if self.configure_paths(): self.load_first_image()

    def natural_sort_key(self, filename):
        match = re.match(r'IR_(\d+)_(\d+)', filename, re.IGNORECASE)
        if match: return (int(match.group(1)), int(match.group(2)))
        return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', filename)]

    def load_first_image(self):
        img_dir = self.paths["img"]
        if os.path.exists(img_dir):
            files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.bmp', '.tif'))]
            self.file_list = sorted(files, key=self.natural_sort_key)
            self.refresh_list_ui()
            if self.file_list: self.list_widget.setCurrentRow(0)

    def refresh_list_ui(self):
        if self.mask_check_thread is not None and self.mask_check_thread.isRunning():
            self.mask_check_thread.stop()
        self.list_widget.blockSignals(True)
        self.list_widget.clear()
        for f in self.file_list: self.list_widget.addItem(f"❓  {f}")
        self.list_widget.blockSignals(False)
        self.lbl_file_count.setText(f"Total: {len(self.file_list)}")
        self.mask_check_thread = MaskStatusWorker(self.file_list, self.paths["mask"])
        self.mask_check_thread.update_signal.connect(self.update_list_item_icon)
        self.mask_check_thread.start()

    def update_list_item_icon(self, idx, icon_str):
        if 0 <= idx < self.list_widget.count():
            item = self.list_widget.item(idx)
            if item: item.setText(f"{icon_str}  {self.file_list[idx]}")

    def on_file_changed(self, row):
        if row < 0 or row >= len(self.file_list): return
        if self.full_mask is not None: self.save_mask()
        self.curr_file_idx = row
        fname = self.file_list[row]
        img_path = os.path.join(self.paths["img"], fname)
        self.full_img = cv2.imread(img_path, 0)
        if self.full_img is None: return
        h, w = self.full_img.shape
        mask_path = os.path.join(self.paths["mask"], os.path.splitext(fname)[0] + ".png")
        if os.path.exists(mask_path):
            self.full_mask = cv2.imread(mask_path, 0)
            if self.full_mask.shape != (h, w): self.full_mask = cv2.resize(self.full_mask, (w, h),
                                                                           interpolation=cv2.INTER_NEAREST)
        else:
            self.full_mask = np.zeros((h, w), dtype=np.uint8)
        self.parse_yolo(os.path.join(self.paths["label"], os.path.splitext(fname)[0] + ".txt"))
        self.mask_history = []
        self.bbox_history = []
        self.curr_bbox_idx = 0 if self.bboxes else -1
        self.refresh_all_views()

    def parse_yolo(self, path):
        self.raw_yolo_data = []
        self.bboxes = []
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    for line in f:
                        p = line.strip().split()
                        if len(p) < 5: continue
                        self.raw_yolo_data.append(tuple(map(float, p[1:5])))
            except:
                pass
        self.recalc_bboxes_from_raw()

    def refresh_all_views(self):
        self.view_overview.update_content()
        self.view_interact.update_display()
        t_info = f"Target {self.curr_bbox_idx + 1}/{len(self.bboxes)}" if self.bboxes else "No Targets"
        status_tail = "[MASK HIDDEN]" if not self.mask_visible else ""
        self.lbl_status.setText(f"{self.file_list[self.curr_file_idx]} | {t_info} {status_tail}")

    def switch_file(self, d):
        r = self.curr_file_idx + d
        if 0 <= r < self.list_widget.count(): self.list_widget.setCurrentRow(r)

    def save_mask(self):
        if self.full_mask is not None and self.file_list:
            n = os.path.splitext(self.file_list[self.curr_file_idx])[0]
            save_p = os.path.join(self.paths["mask"], n + ".png")
            cv2.imwrite(save_p, self.full_mask)
            icon = "✅" if cv2.countNonZero(self.full_mask) > 0 else "⚠️"
            self.list_widget.item(self.curr_file_idx).setText(f"{icon}  {self.file_list[self.curr_file_idx]}")
            self.lbl_status.setText("Saved.")

    def apply_settings(self):
        self.current_Rs = self.spin_Rs.value()
        self.current_mode = int(self.combo_mode.currentText().split(':')[0])
        self.setFocus()
        self.lbl_status.setText(f"Algo Params Applied: Rs={self.current_Rs}, Mode={self.current_mode}")

    def update_vis(self, t, v):
        if t == 'alpha':
            self.mask_alpha = v / 10.0
        elif t == 'gamma':
            self.gamma_val = v / 10.0
        self.refresh_all_views()

    def on_color_change(self, idx):
        maps = {-1: -1, 0: -1, 1: cv2.COLORMAP_JET, 2: cv2.COLORMAP_OCEAN}
        self.colormap_idx = maps.get(idx, -1)
        self.refresh_all_views()

    def on_zoom_change(self, v):
        self.view_interact.display_scale = v
        self.view_interact.update_display()
        self.view_interact.update()

    def push_mask_history(self):
        if self.full_mask is not None:
            self.mask_history.append(self.full_mask.copy())
            if len(self.mask_history) > 10: self.mask_history.pop(0)

    def undo_mask(self):
        if self.mask_history:
            self.full_mask = self.mask_history.pop()
            self.refresh_all_views()
            self.lbl_status.setText("Mask Undo")

    def push_bbox_history(self):
        self.bbox_history.append(copy.deepcopy(self.bboxes))
        if len(self.bbox_history) > 10: self.bbox_history.pop(0)

    def undo_bbox(self):
        if self.bbox_history:
            self.bboxes = self.bbox_history.pop()
            if self.curr_bbox_idx >= len(self.bboxes): self.curr_bbox_idx = len(self.bboxes) - 1
            self.refresh_all_views()
            self.lbl_status.setText("BBox Undo")

    def clear_mask(self):
        if self.full_mask is not None:
            self.push_mask_history()
            self.full_mask.fill(0)
            self.refresh_all_views()
            self.lbl_status.setText("Mask Cleared")

    def reset_params(self):
        self.spin_Rs.blockSignals(True);
        self.combo_mode.blockSignals(True);
        self.chk_clahe.blockSignals(True)
        self.combo_color.blockSignals(True);
        self.s_g.blockSignals(True);
        self.s_a.blockSignals(True)
        self.spin_Rs.setValue(30);
        self.combo_mode.setCurrentIndex(0);
        self.chk_clahe.setChecked(False)
        self.enable_clahe = False;
        self.combo_color.setCurrentIndex(0);
        self.colormap_idx = -1
        self.brush_size = 2;
        self.s_g.setValue(10);
        self.s_a.setValue(4)
        self.gamma_val = 1.0;
        self.mask_alpha = 0.4;
        self.current_Rs = 30;
        self.current_mode = 1
        self.spin_Rs.blockSignals(False);
        self.combo_mode.blockSignals(False);
        self.chk_clahe.blockSignals(False)
        self.combo_color.blockSignals(False)
        self.s_g.blockSignals(False);
        self.s_a.blockSignals(False)
        self.refresh_all_views();
        self.lbl_status.setText("Params Reset")

    def get_current_bbox(self):
        if self.bboxes and 0 <= self.curr_bbox_idx < len(self.bboxes): return self.bboxes[self.curr_bbox_idx]
        return None

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Control and not e.isAutoRepeat():
            self.mask_visible = False;
            self.refresh_all_views();
            return
        if e.key() == Qt.Key_Space: self.toggle_play(); return
        if e.key() == Qt.Key_C and (e.modifiers() & Qt.ControlModifier): self.clear_mask(); return
        if e.key() == Qt.Key_D:
            if self.is_playing: self.stop_play()
            self.switch_file(1)
        elif e.key() == Qt.Key_A:
            if self.is_playing: self.stop_play()
            self.switch_file(-1)
        elif e.key() == Qt.Key_Tab:
            if self.bboxes:
                self.curr_bbox_idx = (self.curr_bbox_idx + 1) % len(self.bboxes)
                self.refresh_all_views()
        if e.key() == Qt.Key_Z and (e.modifiers() & Qt.ControlModifier):
            if e.modifiers() & Qt.ShiftModifier:
                self.undo_bbox()
            else:
                self.undo_mask()
        elif e.key() == Qt.Key_Delete:
            self.delete_current_bbox()
        elif e.key() == Qt.Key_S and (e.modifiers() & Qt.ControlModifier):
            self.save_mask()
        if e.key() == Qt.Key_Shift: self.view_interact.update()

    def keyReleaseEvent(self, e):
        if e.key() == Qt.Key_Control and not e.isAutoRepeat():
            self.mask_visible = True;
            self.refresh_all_views();
            return
        if e.key() == Qt.Key_Shift: self.view_interact.update()

    def closeEvent(self, e):
        self.save_mask()
        if self.mask_check_thread: self.mask_check_thread.stop()
        self.play_timer.stop()
        e.accept()


if __name__ == "__main__":
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 9))
    app.setStyle("Fusion")
    win = SIRSTLabeler()
    win.showMaximized()
    sys.exit(app.exec_())
