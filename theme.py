"""Оформление: палитра, QSS, переключатели, аватары, делегаты списков."""
import math

from PySide6.QtCore import (Qt, QRectF, QPointF, QSize, Signal, QPropertyAnimation, QEasingCurve,
                            Property)
from PySide6.QtGui import (QColor, QFont, QPainter, QPen, QBrush, QLinearGradient, QRadialGradient,
                           QPainterPath, QPixmap, QIcon, QFontMetrics, QImage)
from PySide6.QtWidgets import (QWidget, QStyledItemDelegate, QStyle, QSizePolicy)

# ───────────── палитра ─────────────
C = {
    "bg0": "#04080a", "bg1": "#091210", "bg2": "#0e1b17", "bg3": "#13251f",
    "line": "#17291f", "line2": "#24402f", "line3": "#2f6b50",
    "text": "#e8fff2", "text2": "#a4cdb6", "text3": "#5f8d77",
    "accent": "#39ff88", "accent2": "#00e5ff", "warn": "#ffb000", "danger": "#ff3b5c", "blue": "#4d8dff",
}
UI = "Helvetica Neue"
MONO = "Menlo"


def qc(color, alpha=255):
    c = QColor(color) if not isinstance(color, QColor) else QColor(color)
    c.setAlpha(int(alpha))
    return c


def font(size=13, weight=QFont.Normal, family=UI, spacing=0.0, caps=False):
    f = QFont(family)
    f.setPixelSize(int(size))
    f.setWeight(weight)
    if spacing:
        f.setLetterSpacing(QFont.AbsoluteSpacing, spacing)
    if caps:
        f.setCapitalization(QFont.AllUppercase)
    return f


def _build_qss():
    import os, tempfile
    arrow = os.path.join(tempfile.gettempdir(), "pv_arrow.png")
    try:
        from PySide6.QtGui import QGuiApplication
        pm = QPixmap(24, 24)
        pm.fill(Qt.transparent)
        pp = QPainter(pm)
        pp.setRenderHint(QPainter.Antialiasing)
        pp.setPen(QPen(QColor("#5f8d77"), 2.4, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        pp.drawPolyline([QPointF(6, 9), QPointF(12, 15), QPointF(18, 9)])
        pp.end()
        pm.save(arrow)
    except Exception:
        pass
    return f"""
* {{ font-family: "{UI}", "Helvetica Neue", "Arial"; font-size: 13px; color: {C['text']}; outline: none; }}
QMainWindow, QDialog {{ background: {C['bg0']}; }}
QFrame#panel {{ background: {C['bg1']}; border: 1px solid {C['line']}; border-radius: 16px; }}
QFrame#card {{ background: {C['bg2']}; border: 1px solid {C['line']}; border-radius: 12px; }}
QLabel {{ background: transparent; }}
QLabel#h {{ color: {C['text3']}; font-size: 10px; font-weight: 700; letter-spacing: 1.5px; }}
QLabel#muted {{ color: {C['text3']}; font-size: 12px; }}
QLabel#detail {{ background: {C['bg2']}; border: 1px solid {C['line']}; border-radius: 12px; padding: 12px; color: {C['text2']}; }}
QPushButton {{ background: {C['bg2']}; border: 1px solid {C['line2']}; border-radius: 10px; padding: 8px 14px;
    color: #cfeedd; font-weight: 600; }}
QPushButton:hover {{ background: {C['bg3']}; border-color: {C['line3']}; color: #ffffff; }}
QPushButton:pressed {{ background: {C['bg1']}; }}
QPushButton:checkable:checked {{ background: rgba(57,255,136,0.12); border-color: {C['line3']}; color: {C['accent']}; }}
QPushButton:disabled {{ color: {C['text3']}; border-color: {C['line']}; }}
QPushButton#primary {{ background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #39ff88, stop:1 #00e5b0);
    color: #02140a; border: none; font-weight: 700; }}
QPushButton#primary:hover {{ background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #6bffa6, stop:1 #38f0c4); }}
QPushButton#danger {{ color: {C['danger']}; border-color: #5a2030; }}
QPushButton#danger:hover {{ background: #2a0f17; border-color: {C['danger']}; color: #ff7a92; }}
QPushButton#ghost {{ background: transparent; border: none; color: {C['text2']}; }}
QPushButton#ghost:hover {{ color: #ffffff; background: rgba(255,255,255,0.05); }}
QPushButton[tab="true"] {{ background: transparent; border: none; border-radius: 9px; padding: 7px 4px;
    color: {C['text3']}; font-weight: 700; font-size: 11px; letter-spacing: 1px; }}
QPushButton[tab="true"]:hover {{ color: {C['text2']}; background: rgba(255,255,255,0.04); }}
QPushButton[tab="true"]:checked {{ background: rgba(57,255,136,0.13); color: {C['accent']}; }}
QPushButton[src="true"] {{ background: transparent; border: none; border-radius: 9px; padding: 7px 14px;
    color: {C['text2']}; font-weight: 600; }}
QPushButton[src="true"]:hover {{ background: rgba(255,255,255,0.06); color: #fff; }}
QPushButton[src="true"]:checked {{ background: rgba(57,255,136,0.15); color: {C['accent']}; }}
QLineEdit, QTextEdit, QComboBox, QPlainTextEdit {{ background: #060d0a; border: 1px solid {C['line2']}; border-radius: 10px;
    padding: 8px 10px; selection-background-color: {C['accent']}; selection-color: #02100a; color: {C['text']}; }}
QLineEdit:focus, QTextEdit:focus, QComboBox:focus {{ border: 1px solid {C['accent']}; }}
QComboBox::drop-down {{ border: none; width: 26px; }}
QComboBox::down-arrow {{ image: url({arrow}); width: 12px; height: 12px; }}
QComboBox QAbstractItemView {{ background: {C['bg1']}; border: 1px solid {C['line3']}; border-radius: 8px; padding: 4px;
    selection-background-color: rgba(57,255,136,0.18); selection-color: #ffffff; outline: none; }}
QListWidget {{ background: transparent; border: none; outline: none; }}
QListWidget::item {{ border-radius: 12px; }}
QScrollBar:vertical {{ background: transparent; width: 9px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {C['line2']}; border-radius: 3px; min-height: 28px; }}
QScrollBar::handle:vertical:hover {{ background: {C['line3']}; }}
QScrollBar::add-line, QScrollBar::sub-line, QScrollBar::add-page, QScrollBar::sub-page {{ height: 0; background: none; }}
QScrollBar:horizontal {{ height: 0; }}
QSlider::groove:horizontal {{ height: 4px; background: {C['line2']}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {C['accent']}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: #ffffff; width: 14px; height: 14px; margin: -5px 0; border-radius: 7px; }}
QToolTip {{ background: {C['bg1']}; color: {C['text']}; border: 1px solid {C['line3']}; padding: 5px 8px; border-radius: 6px; }}
QMessageBox {{ background: {C['bg0']}; }}
"""


QSS = None  # собирается в app.py после создания QApplication (build_qss)


def build_qss():
    return _build_qss()


# ───────────── переключатель в стиле iOS ─────────────
class Switch(QWidget):
    toggled = Signal(bool)

    def __init__(self, label, on=True, hint=None, callback=None):
        super().__init__()
        self._on, self._label, self._pos = on, label, 1.0 if on else 0.0
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedHeight(38)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        if hint:
            self.setToolTip(hint)
        self._anim = QPropertyAnimation(self, b"pos_")
        self._anim.setDuration(160)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)
        if callback:
            self.toggled.connect(callback)
            callback(on)

    def get_pos(self):
        return self._pos

    def set_pos(self, v):
        self._pos = v
        self.update()

    pos_ = Property(float, get_pos, set_pos)

    def isChecked(self):
        return self._on

    def setChecked(self, on, emit=False):
        self._on = on
        self._pos = 1.0 if on else 0.0
        self.update()
        if emit:
            self.toggled.emit(on)

    def mouseReleaseEvent(self, e):
        self._on = not self._on
        self._anim.stop()
        self._anim.setStartValue(self._pos)
        self._anim.setEndValue(1.0 if self._on else 0.0)
        self._anim.start()
        self.toggled.emit(self._on)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setFont(font(13, QFont.Medium))
        p.setPen(qc(C["text"] if self._on else C["text2"]))
        p.drawText(QRectF(2, 0, self.width() - 56, self.height()), Qt.AlignVCenter | Qt.AlignLeft, self._label)
        w, h = 38, 22
        x, y = self.width() - w - 2, (self.height() - h) / 2
        track = QRectF(x, y, w, h)
        t = self._pos
        col = QColor(
            int(31 + (57 - 31) * t), int(58 + (255 - 58) * t), int(47 + (136 - 47) * t))
        p.setPen(QPen(qc(C["accent"], 90 * t), 1) if t > 0.05 else Qt.NoPen)
        p.setBrush(col if t > 0.01 else qc(C["line2"]))
        p.drawRoundedRect(track, h / 2, h / 2)
        kx = x + 3 + (w - h) * t
        p.setBrush(QColor("#02140a") if t > 0.5 else QColor("#a4cdb6"))
        p.drawEllipse(QRectF(kx, y + 3, h - 6, h - 6))


# ───────────── логотип ─────────────
class Logo(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(210, 40)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        c = QPointF(20, 20)
        p.setPen(QPen(qc(C["accent"]), 2))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(c, 12, 12)
        p.setPen(QPen(qc(C["accent2"]), 1.6))
        p.drawEllipse(c, 6, 6)
        p.setPen(QPen(qc(C["accent"]), 2))
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            p.drawLine(QPointF(c.x() + dx * 12, c.y() + dy * 12), QPointF(c.x() + dx * 17, c.y() + dy * 17))
        p.setBrush(qc(C["accent"]))
        p.setPen(Qt.NoPen)
        p.drawEllipse(c, 2, 2)
        f = font(17, QFont.Black, spacing=2.4)
        p.setFont(f)
        g = QLinearGradient(46, 0, 200, 0)
        g.setColorAt(0, qc("#ffffff"))
        g.setColorAt(1, qc(C["accent"]))
        p.setPen(QPen(QBrush(g), 1))
        p.drawText(QPointF(46, 21), "PEOPLE")
        p.setFont(font(9, QFont.Bold, spacing=3.2))
        p.setPen(qc(C["text3"]))
        p.drawText(QPointF(47, 34), "VISION  CONSOLE")


def app_icon():
    pm = QPixmap(256, 256)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    path = QPainterPath()
    path.addRoundedRect(QRectF(8, 8, 240, 240), 56, 56)
    g = QLinearGradient(0, 0, 256, 256)
    g.setColorAt(0, qc("#0e1b17"))
    g.setColorAt(1, qc("#030706"))
    p.fillPath(path, g)
    p.setPen(QPen(qc(C["accent"]), 11))
    p.setBrush(Qt.NoBrush)
    p.drawEllipse(QPointF(128, 128), 66, 66)
    p.setPen(QPen(qc(C["accent2"]), 8))
    p.drawEllipse(QPointF(128, 128), 30, 30)
    p.setPen(QPen(qc(C["accent"]), 11, Qt.SolidLine, Qt.RoundCap))
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        p.drawLine(QPointF(128 + dx * 66, 128 + dy * 66), QPointF(128 + dx * 100, 128 + dy * 100))
    p.setBrush(qc(C["accent"]))
    p.setPen(Qt.NoPen)
    p.drawEllipse(QPointF(128, 128), 10, 10)
    p.end()
    return QIcon(pm)


# ───────────── аватары ─────────────
def circle_pixmap(pix, size):
    """Квадратная картинка → круглая миниатюра."""
    out = QPixmap(size, size)
    out.fill(Qt.transparent)
    if pix is None or pix.isNull():
        return out
    pix = pix.scaled(size, size, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
    p = QPainter(out)
    p.setRenderHint(QPainter.Antialiasing)
    path = QPainterPath()
    path.addEllipse(0, 0, size, size)
    p.setClipPath(path)
    p.drawPixmap((size - pix.width()) // 2, (size - pix.height()) // 2, pix)
    p.end()
    return out


def initials_avatar(name, size, color):
    out = QPixmap(size, size)
    out.fill(Qt.transparent)
    p = QPainter(out)
    p.setRenderHint(QPainter.Antialiasing)
    g = QLinearGradient(0, 0, size, size)
    g.setColorAt(0, qc(color, 70))
    g.setColorAt(1, qc(color, 20))
    p.setBrush(g)
    p.setPen(QPen(qc(color, 140), 1.2))
    p.drawEllipse(QRectF(0.6, 0.6, size - 1.2, size - 1.2))
    words = [w for w in name.split() if w[:1].isalpha()]
    txt = "".join(w[0] for w in words[:2]).upper() or "?"
    p.setPen(qc(color))
    p.setFont(font(size * 0.38, QFont.Bold))
    p.drawText(QRectF(0, 0, size, size), Qt.AlignCenter, txt)
    p.end()
    return out


def draw_chip(p, x, y, text, color, size=10, filled=False, dot=True, weight=QFont.Bold, h=None, alpha=1.0):
    """Скруглённая «таблетка»; возвращает её ширину."""
    f = font(size, weight, spacing=0.4)
    fm = QFontMetrics(f)
    tw = fm.horizontalAdvance(text)
    pad = 8
    dw = 10 if dot else 0
    w = tw + pad * 2 + dw
    h = h or (size + 10)
    r = QRectF(x, y, w, h)
    p.setPen(QPen(qc(color, 120 * alpha), 1))
    p.setBrush(qc(color, (200 if filled else 34) * alpha))
    p.drawRoundedRect(r, h / 2, h / 2)
    if dot:
        p.setPen(Qt.NoPen)
        p.setBrush(qc("#02140a" if filled else color, 255 * alpha))
        p.drawEllipse(QPointF(x + pad + 2.5, y + h / 2), 2.5, 2.5)
    p.setFont(f)
    p.setPen(qc("#02140a" if filled else color, 255 * alpha))
    p.drawText(QRectF(x + pad + dw, y, tw + 2, h), Qt.AlignVCenter | Qt.AlignLeft, text)
    return w


# ───────────── делегаты списков ─────────────
ROLE_PID = Qt.UserRole
ROLE_DATA = Qt.UserRole + 1


class PersonDelegate(QStyledItemDelegate):
    """Строка базы людей: аватар, ФИО, статус-чип, телефон."""

    def __init__(self, db, status_color, parent=None):
        super().__init__(parent)
        self.db, self.status_color = db, status_color
        self.cache = {}

    def avatar(self, pid, name, color):
        key = (pid, self.db.people.get(pid, {}).get("added_at"), name)
        if key not in self.cache:
            blob = self.db.photo(pid)
            pm = QPixmap()
            if blob:
                pm.loadFromData(blob)
            self.cache[key] = (circle_pixmap(pm, 40) if not pm.isNull() else initials_avatar(name, 40, color))
        return self.cache[key]

    def sizeHint(self, option, index):
        return QSize(option.rect.width(), 58)

    def paint(self, p, option, index):
        pid = index.data(ROLE_PID)
        person = self.db.people.get(pid)
        if person is None:
            return
        p.save()
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(option.rect).adjusted(2, 2, -2, -2)
        sel = option.state & QStyle.State_Selected
        hov = option.state & QStyle.State_MouseOver
        col = self.status_color(person["status"])
        if sel or hov:
            p.setPen(QPen(qc(C["accent"], 90 if sel else 0), 1))
            p.setBrush(qc(C["accent"], 28 if sel else 12))
            p.drawRoundedRect(r, 12, 12)
        if sel:
            p.setPen(Qt.NoPen)
            p.setBrush(qc(C["accent"]))
            p.drawRoundedRect(QRectF(r.left(), r.top() + 12, 3, r.height() - 24), 1.5, 1.5)
        p.drawPixmap(int(r.left() + 12), int(r.top() + (r.height() - 40) / 2), self.avatar(pid, person["full"], col))
        tx = r.left() + 62
        p.setFont(font(14, QFont.DemiBold))
        p.setPen(qc(C["text"]))
        fm = QFontMetrics(p.font())
        name = fm.elidedText(person["full"], Qt.ElideRight, int(r.width() - 74))
        p.drawText(QPointF(tx, r.top() + 24), name)
        x = tx
        if person["status"]:
            x += draw_chip(p, x, r.top() + 32, person["status"].upper(), col, size=9) + 8
        sub = person["phone"] or person["workplace"] or ""
        p.setFont(font(11))
        p.setPen(qc(C["text3"]))
        p.drawText(QRectF(x, r.top() + 31, r.right() - x - 8, 18), Qt.AlignVCenter | Qt.AlignLeft,
                   QFontMetrics(p.font()).elidedText(sub, Qt.ElideRight, int(r.right() - x - 10)))
        p.restore()


class SceneDelegate(QStyledItemDelegate):
    """Строка «В кадре»: цель, имя, поза, чипы предметов/действий."""

    def sizeHint(self, option, index):
        d = index.data(ROLE_DATA) or {}
        return QSize(option.rect.width(), 74 + (22 if d.get("chips") else 0))

    def paint(self, p, option, index):
        d = index.data(ROLE_DATA)
        if not d:
            return
        p.save()
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(option.rect).adjusted(2, 3, -2, -3)
        col = d["color"]
        p.setPen(QPen(qc(col, 60), 1))
        p.setBrush(qc(col, 14))
        p.drawRoundedRect(r, 12, 12)
        p.setPen(Qt.NoPen)
        p.setBrush(qc(col))
        p.drawRoundedRect(QRectF(r.left(), r.top() + 12, 3, r.height() - 24), 1.5, 1.5)
        av = d.get("avatar")
        if av is not None:
            p.drawPixmap(int(r.left() + 12), int(r.top() + 10), av)
        tx = r.left() + 62
        p.setFont(font(9, QFont.Bold, MONO, 1))
        p.setPen(qc(col))
        p.drawText(QPointF(tx, r.top() + 18), d["tag"])
        p.setFont(font(14, QFont.DemiBold))
        p.setPen(qc(C["text"]))
        fm = QFontMetrics(p.font())
        p.drawText(QPointF(tx, r.top() + 37), fm.elidedText(d["name"], Qt.ElideRight, int(r.width() - 74)))
        p.setFont(font(11))
        p.setPen(qc(C["accent2"]))
        fm = QFontMetrics(p.font())
        p.drawText(QPointF(tx, r.top() + 55), fm.elidedText(d["pose"], Qt.ElideRight, int(r.width() - 74)))
        x, y = tx, r.top() + 62
        for text, ccol in d.get("chips", []):
            w = QFontMetrics(font(9, QFont.Bold, spacing=0.4)).horizontalAdvance(text) + 26
            if x + w > r.right() - 6:
                break
            x += draw_chip(p, x, y, text, ccol, size=9) + 6
        p.restore()
