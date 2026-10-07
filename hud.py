"""Видео-экран с HUD: кадр, скелет, силуэт, карточки людей, предметы, журнал модулей."""
import math
import time
from functools import lru_cache

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal, QRectF, QPointF, QTimer
from PySide6.QtGui import (QPainter, QColor, QPen, QFont, QFontMetrics, QPixmap, QImage, QPolygonF,
                           QPainterPath, QRadialGradient, QLinearGradient, QBrush)
from PySide6.QtWidgets import QWidget

from items import CATEGORIES
from engine import HAND_LINKS, HAND_TIPS
from theme import C, MONO, qc, font, draw_chip, circle_pixmap, initials_avatar

F = lru_cache(maxsize=96)(font)

# части тела: цвета и связи между 33 точками (стороны — анатомические)
PART_COLORS = {"head": QColor("#eafff2"), "torso": QColor("#39ff88"),
               "arm_l": QColor("#00e5ff"), "arm_r": QColor("#4d8dff"),
               "leg_l": QColor("#ffb000"), "leg_r": QColor("#ff6a00")}
PART_LINKS = {
    "head": [(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10)],
    "torso": [(11, 12), (11, 23), (12, 24), (23, 24)],
    "arm_l": [(11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19)],
    "arm_r": [(12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20)],
    "leg_l": [(23, 25), (25, 27), (27, 29), (29, 31), (27, 31)],
    "leg_r": [(24, 26), (26, 28), (28, 30), (30, 32), (28, 32)],
}
PART_OF = {}
for _i in range(0, 11):
    PART_OF[_i] = "head"
for _i in (11, 12, 23, 24):
    PART_OF[_i] = "torso"
for _i in (13, 15, 17, 19, 21):
    PART_OF[_i] = "arm_l"
for _i in (14, 16, 18, 20, 22):
    PART_OF[_i] = "arm_r"
for _i in (25, 27, 29, 31):
    PART_OF[_i] = "leg_l"
for _i in (26, 28, 30, 32):
    PART_OF[_i] = "leg_r"
KEY_JOINTS = (11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)
PART_LABELS = [(13, "Л.РУКА", "arm_l"), (14, "П.РУКА", "arm_r"), (15, "Л.КИСТЬ", "arm_l"),
               (16, "П.КИСТЬ", "arm_r"), (25, "Л.НОГА", "leg_l"), (26, "П.НОГА", "leg_r"),
               (27, "Л.СТОПА", "leg_l"), (28, "П.СТОПА", "leg_r")]


def calc_age(d):
    try:
        dd, mm, yy = [int(x) for x in d.split(".")]
        t = time.localtime()
        return t.tm_year - yy - ((t.tm_mon, t.tm_mday) < (mm, dd))
    except Exception:
        return None


def status_color(status):
    s = (status or "").lower()
    if "чёрн" in s or "черн" in s or "подозр" in s:
        return QColor(C["danger"])
    if "vip" in s:
        return QColor(C["warn"])
    return QColor(C["accent"])


def to_pixmap(bgr):
    bgr = np.ascontiguousarray(bgr)
    h, w, _ = bgr.shape
    return QPixmap.fromImage(QImage(bgr.data, w, h, 3 * w, QImage.Format_BGR888))


def cat_color(cat):
    r, g, b = CATEGORIES.get(cat, CATEGORIES["other"])[1]
    return QColor(r, g, b)


class VideoWidget(QWidget):
    clicked = Signal(float, float)
    selected = Signal(object)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(640, 440)
        self.setMouseTracking(True)
        self.frame = None
        self.pix = None
        self.tracks, self.objects = [], []
        self.db = None
        self.names = {}              # ключ предмета/действия -> (рус. название, категория)
        self.message = "ЗАПУСК КАМЕРЫ…"
        self.source_label = "CAM0"
        self.stats = {"fps": 0.0, "ms": 0.0}
        self.modules = {"POSE": True, "OBJ": True, "SCAN": True}
        # слои отображения
        self.show_body = True
        self.show_skeleton = True
        self.show_silhouette = True
        self.show_labels = False
        self.show_objects = True
        self.show_cards = True
        self.show_hud = True
        self.show_scores = False
        self.enroll = None           # панель записи образцов: {"title", "sub", "progress", "n"}
        self.highlight = set()
        self.select_mode = False
        self._drag = None
        self._geom = (0.0, 0.0, 1.0)
        self._born, self._cardpos, self._cardsel = {}, {}, {}
        self._avatars = {}
        self._fps, self._t_last = 0.0, time.time()
        self._t0 = time.time()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(40)

    # ── данные ──
    def update_frame(self, frame, tracks, objects, ms=0.0):
        now = time.time()
        dt = now - self._t_last
        self._t_last = now
        if dt > 0:
            self._fps = 0.9 * self._fps + 0.1 / dt if self._fps else 1 / dt
        self.stats = {"fps": self._fps, "ms": ms}
        self.frame, self.tracks, self.objects = frame, tracks, objects
        self.pix = to_pixmap(frame)
        live = {t["tid"] for t in tracks}
        for tid in live:
            self._born.setdefault(tid, now)
        for d in (self._born, self._cardpos, self._cardsel):
            for tid in [k for k in d if k not in live]:
                del d[tid]

    def clear(self, message):
        self.frame = self.pix = None
        self.tracks, self.objects = [], []
        self.message = message

    def set_select_mode(self, on):
        self.select_mode = on
        self._drag = None
        self.setCursor(Qt.CrossCursor if on else Qt.ArrowCursor)

    # ── вспомогательное ──
    def _name(self, key):
        return self.names.get(key, (key, "other"))

    def _avatar(self, pid, name, color, size=44):
        key = (pid, size, name)
        if key not in self._avatars:
            pm = QPixmap()
            if pid is not None and self.db is not None:
                blob = self.db.photo(pid)
                if blob:
                    pm.loadFromData(blob)
            self._avatars[key] = circle_pixmap(pm, size) if not pm.isNull() else initials_avatar(name, size, color)
            if len(self._avatars) > 200:
                self._avatars.pop(next(iter(self._avatars)))
        return self._avatars[key]

    @staticmethod
    def _brackets(p, r, color, ln=16, w=2.0):
        pen = QPen(color, w, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        p.setPen(pen)
        x1, y1, x2, y2 = r.left(), r.top(), r.right(), r.bottom()
        ln = min(ln, r.width() / 2.5, r.height() / 2.5)
        for (x, y, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
            path = QPainterPath(QPointF(x + dx * ln, y))
            path.lineTo(x, y)
            path.lineTo(x, y + dy * ln)
            p.drawPath(path)

    # ── главная отрисовка ──
    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        W, H = self.width(), self.height()
        # фон
        g = QRadialGradient(W / 2, H / 2, max(W, H) * 0.7)
        g.setColorAt(0, qc("#0a1612"))
        g.setColorAt(1, qc("#03070a"))
        p.fillRect(self.rect(), g)
        margin = 10
        area = QRectF(margin, margin, W - 2 * margin, H - 2 * margin)
        clip = QPainterPath()
        clip.addRoundedRect(area, 16, 16)
        now = time.time()

        if self.pix is None:
            self._paint_idle(p, area, now)
            self._frame_border(p, area, now)
            p.end()
            return

        fw, fh = self.pix.width(), self.pix.height()
        sc = min(area.width() / fw, area.height() / fh)
        vw, vh = fw * sc, fh * sc
        ox, oy = area.left() + (area.width() - vw) / 2, area.top() + (area.height() - vh) / 2
        self._geom = (ox, oy, sc)
        vid = QRectF(ox, oy, vw, vh)
        vclip = QPainterPath()
        vclip.addRoundedRect(vid, 14, 14)
        p.save()
        p.setClipPath(vclip)
        p.drawPixmap(vid.toRect(), self.pix)
        p.drawPixmap(vid.toRect(), self._overlay(int(vw), int(vh)))  # тонировка + сканлинии + виньетка
        if self.select_mode:
            p.fillRect(vid, QColor(0, 12, 8, 90))

        S = lambda x, y: QPointF(ox + float(x) * sc, oy + float(y) * sc)
        objs = {o["oid"]: o for o in self.objects}
        placed = []
        cards = []
        known_n = 0
        for t in self.tracks:
            pid = t["pid"]
            person = self.db.people.get(pid) if (pid is not None and self.db) else None
            known_n += person is not None
            color = status_color(person["status"]) if person else QColor(C["danger"] if t["face"] is not None else C["warn"])
            born = min(1.0, (now - self._born.get(t["tid"], now)) / 0.35)
            x1, y1, x2, y2 = [float(v) for v in t["box"]]
            body = QRectF(ox + x1 * sc, oy + y1 * sc, (x2 - x1) * sc, (y2 - y1) * sc)
            hl = person is not None and pid in self.highlight
            p.setOpacity(born)
            if t.get("pose") is not None:
                self._draw_pose(p, t["pose"], color, ox, oy, sc, t.get("hands"))
            if self.show_body:
                p.setPen(QPen(qc(color, 46), 1))
                p.setBrush(Qt.NoBrush)
                p.drawRoundedRect(body, 6, 6)
                self._brackets(p, body, color, 20, 3.4 if hl else 2.2)
                draw_chip(p, body.left() + 8, body.top() + 8, f"TGT-{t['tid']:02d}", color, size=9, dot=False)
            if t["face"] is not None:
                fx, fy, fw_, fh_ = [float(v) for v in t["face"][:4]]
                fr = QRectF(ox + fx * sc, oy + fy * sc, fw_ * sc, fh_ * sc)
                self._brackets(p, fr, qc(C["accent2"]), 9, 1.8)
            self._draw_hands(p, t, ox, oy, sc)
            self._draw_hold(p, t, ox, oy, sc, objs)
            if self.show_scores or self.enroll:
                self._draw_scores(p, t, ox, oy, sc)
            p.setOpacity(1.0)
            cards.append((t, person, color, body, born, hl))

        if self.show_objects:
            for o in self.objects:
                self._draw_object(p, o, ox, oy, sc)

        if self.show_cards:
            for t, person, color, body, born, hl in sorted(cards, key=lambda c: c[3].left()):
                self._draw_card(p, t, person, color, body, born, hl, vid, placed, objs)
        p.restore()

        if self.show_hud:
            self._paint_hud(p, vid, known_n, now)
        self._frame_border(p, vid, now, rounded=14)
        if self.enroll:
            self._paint_enroll(p, vid)
        if self.select_mode:
            self._paint_select(p, vid)
        p.end()

    def _overlay(self, w, h):
        if getattr(self, "_ov_key", None) != (w, h):
            pm = QPixmap(max(1, w), max(1, h))
            pm.fill(Qt.transparent)
            q = QPainter(pm)
            q.fillRect(0, 0, w, h, QColor(0, 38, 20, 34))
            q.setPen(QPen(QColor(0, 0, 0, 26), 1))
            for y in range(0, h, 4):
                q.drawLine(0, y, w, y)
            vg = QRadialGradient(w / 2, h / 2, max(w, h) * 0.75)
            vg.setColorAt(0.55, QColor(0, 0, 0, 0))
            vg.setColorAt(1, QColor(0, 0, 0, 150))
            q.fillRect(0, 0, w, h, vg)
            q.end()
            self._ov, self._ov_key = pm, (w, h)
        return self._ov

    def _paint_idle(self, p, area, now):
        c = area.center()
        for i in range(3):
            ph = ((now * 0.6 + i / 3) % 1.0)
            p.setPen(QPen(qc(C["accent"], 120 * (1 - ph)), 1.5))
            p.setBrush(Qt.NoBrush)
            r = 26 + ph * 90
            p.drawEllipse(c, r, r)
        p.setPen(QPen(qc(C["accent"]), 2))
        p.drawEllipse(c, 16, 16)
        p.setBrush(qc(C["accent"]))
        p.drawEllipse(c, 3, 3)
        p.setFont(F(12, QFont.DemiBold, MONO, 1.5))
        p.setPen(qc(C["text2"]))
        p.drawText(QRectF(area.left(), c.y() + 110, area.width(), 70), Qt.AlignHCenter | Qt.AlignTop, self.message)

    def _frame_border(self, p, r, now, rounded=16):
        p.setBrush(Qt.NoBrush)
        for w, a in ((7, 10), (4, 18), (1.4, 120)):
            p.setPen(QPen(qc(C["accent"], a), w))
            p.drawRoundedRect(r, rounded, rounded)

    # ── поза ──
    def _draw_pose(self, p, pose, color, ox, oy, sc, hands=None):
        pts, vis = pose["pts"], pose["vis"]
        hands = hands or {}
        skip = set()
        if "L" in hands:
            skip |= {(15, 17), (15, 19), (15, 21), (17, 19)}
        if "R" in hands:
            skip |= {(16, 18), (16, 20), (16, 22), (18, 20)}
        S = lambda i: QPointF(ox + float(pts[i][0]) * sc, oy + float(pts[i][1]) * sc)
        ok = lambda i: bool(vis[i] > 0.5)
        rgb = (color.red(), color.green(), color.blue())
        if self.show_silhouette:
            for c in pose["contours"]:
                poly = QPolygonF([QPointF(ox + float(x) * sc, oy + float(y) * sc) for x, y in c])
                p.setPen(QPen(QColor(*rgb, 26), 5))
                p.setBrush(Qt.NoBrush)
                p.drawPolygon(poly)
                p.setPen(QPen(QColor(*rgb, 220), 1.4))
                p.setBrush(QColor(*rgb, 22))
                p.drawPolygon(poly)
        if self.show_skeleton:
            if all(ok(i) for i in (11, 12, 23, 24)):
                p.setPen(Qt.NoPen)
                p.setBrush(QColor(57, 255, 136, 44))
                p.drawPolygon(QPolygonF([S(11), S(12), S(24), S(23)]))
            p.setBrush(Qt.NoBrush)
            for part, links in PART_LINKS.items():
                col = PART_COLORS[part]
                glow = QPen(qc(col, 64), 7, Qt.SolidLine, Qt.RoundCap)
                line = QPen(col, 2.3, Qt.SolidLine, Qt.RoundCap)
                for a, b in links:
                    if ok(a) and ok(b) and (a, b) not in skip:
                        p.setPen(glow)
                        p.drawLine(S(a), S(b))
                        p.setPen(line)
                        p.drawLine(S(a), S(b))
            hide = {17, 19, 21} if "L" in hands else set()
            hide |= {18, 20, 22} if "R" in hands else set()
            for i in range(33):
                if ok(i) and i not in hide:
                    r = 4.4 if i in KEY_JOINTS else 2.6
                    p.setPen(QPen(PART_COLORS[PART_OF[i]], 1.5))
                    p.setBrush(QColor(2, 12, 8, 235))
                    p.drawEllipse(S(i), r, r)
        head = [i for i in (0, 2, 5, 7, 8) if ok(i)]
        hc = hr = None
        if len(head) >= 2:
            hc = QPointF(sum(S(i).x() for i in head) / len(head), sum(S(i).y() for i in head) / len(head))
            hr = max(math.hypot(S(i).x() - hc.x(), S(i).y() - hc.y()) for i in head) * 1.45 + 6
            if self.show_skeleton:
                p.setPen(QPen(PART_COLORS["head"], 1.1, Qt.DashLine))
                p.setBrush(Qt.NoBrush)
                p.drawEllipse(hc, hr, hr)
        if self.show_labels:
            labels = []
            if hc is not None:
                labels.append((QPointF(hc.x() - 22, hc.y() - hr - 6), "ГОЛОВА", "head"))
            if all(ok(i) for i in (11, 12, 23, 24)):
                cx = sum(S(i).x() for i in (11, 12, 23, 24)) / 4
                cy = sum(S(i).y() for i in (11, 12, 23, 24)) / 4
                labels.append((QPointF(cx - 30, cy), "ТУЛОВИЩЕ", "torso"))
            elif ok(11) and ok(12):
                m = QPointF((S(11).x() + S(12).x()) / 2, (S(11).y() + S(12).y()) / 2)
                labels.append((QPointF(m.x() - 18, m.y() + 24), "ПЛЕЧИ", "torso"))
            for i, text, part in PART_LABELS:
                if ok(i):
                    labels.append((QPointF(S(i).x() + 8, S(i).y() - 6), text, part))
            p.setFont(F(9, QFont.Bold, MONO))
            fm = QFontMetrics(p.font())
            for pt, text, part in labels:
                r = QRectF(pt.x() - 4, pt.y() - fm.ascent() - 2, fm.horizontalAdvance(text) + 8, fm.height() + 3)
                p.setPen(Qt.NoPen)
                p.setBrush(QColor(2, 12, 8, 205))
                p.drawRoundedRect(r, 4, 4)
                p.setPen(PART_COLORS[part])
                p.drawText(pt, text)

    # ── пальцы (21 точка кисти) ──
    def _draw_hands(self, p, t, ox, oy, sc):
        for slot, h in (t.get("hands") or {}).items():
            col = PART_COLORS["arm_l" if slot == "L" else "arm_r"]
            pts = h["pts"]
            S = lambda i: QPointF(ox + float(pts[i][0]) * sc, oy + float(pts[i][1]) * sc)
            glow = QPen(qc(col, 60), 6, Qt.SolidLine, Qt.RoundCap)
            line = QPen(col, 2.0, Qt.SolidLine, Qt.RoundCap)
            for a, b in HAND_LINKS:
                p.setPen(glow)
                p.drawLine(S(a), S(b))
                p.setPen(line)
                p.drawLine(S(a), S(b))
            for i in range(21):
                r = 3.6 if i in HAND_TIPS else 2.2
                p.setPen(QPen(col, 1.2))
                p.setBrush(QColor(2, 12, 8, 235) if i not in HAND_TIPS else qc(col, 230))
                p.drawEllipse(S(i), r, r)

    def _draw_scores(self, p, t, ox, oy, sc):
        """Что сейчас «думает» нейросеть о каждой руке (первые три гипотезы) — чтобы видеть, работает ли обучение."""
        hands, scores = t.get("hands") or {}, t.get("scores") or {}
        for slot, top in scores.items():
            h = hands.get(slot)
            if h is None or not top:
                continue
            pts = h["pts"]
            x = ox + float(pts[:, 0].max()) * sc + 8
            y = oy + float(pts[:, 1].min()) * sc
            lines = [(self._name(k)[0], pr) for k, pr in top[:3]]
            f = F(10, QFont.DemiBold, MONO)
            fm = QFontMetrics(f)
            w = max(fm.horizontalAdvance(f"{n} 100%") for n, _ in lines) + 18
            r = QRectF(x, y, w, 14 * len(lines) + 12)
            p.setPen(QPen(qc(C["line3"], 160), 1))
            p.setBrush(QColor(3, 10, 7, 215))
            p.drawRoundedRect(r, 8, 8)
            p.setFont(f)
            for i, (n, pr) in enumerate(lines):
                p.setPen(qc(C["accent"] if (i == 0 and pr >= 0.5) else C["text3"] if i else C["text2"]))
                p.drawText(QPointF(r.left() + 9, r.top() + 17 + 14 * i), f"{n} {int(pr * 100)}%")
            bar = QRectF(r.left() + 6, r.bottom() - 5, (r.width() - 12) * min(1.0, lines[0][1]), 2.4)
            p.setPen(Qt.NoPen)
            p.setBrush(qc(C["accent"], 200))
            p.drawRoundedRect(bar, 1.2, 1.2)

    # ── что в руках (CLIP) ──
    def _draw_hold(self, p, t, ox, oy, sc, objs):
        pose, hold = t.get("pose"), t.get("hold") or {}
        if pose is None or not hold:
            return
        pts, vis = pose["pts"], pose["vis"]
        hands = t.get("hands") or {}
        for slot, (key, prob) in hold.items():
            wr = 15 if slot == "L" else 16
            if slot in hands:
                hp = hands[slot]["pts"]
                anchor = hp.mean(0)
            elif vis[wr] >= 0.5:
                anchor = pts[wr]
            else:
                continue
            name, cat = self._name(key)
            hand = "ЛЕВАЯ" if slot == "L" else "ПРАВАЯ"
            # этот же предмет уже показан рамкой (YOLOX + уточнение CLIP) — не дублируем
            dup = False
            for i in t.get("objects", []):
                o = objs.get(i)
                if o and o["hand"] == hand and (o["cat"] == cat or (o.get("clip") and o["clip"][0] == key)):
                    dup = True
            if dup:
                continue
            col = cat_color(cat)
            c = QPointF(ox + float(anchor[0]) * sc, oy + float(anchor[1]) * sc)
            pulse = 0.5 + 0.5 * math.sin(time.time() * 5)
            p.setPen(QPen(qc(col, 90 + 90 * pulse), 1.6))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(c, 14 + 2 * pulse, 14 + 2 * pulse)
            text = f"{name} {int(prob * 100)}%"
            draw_chip(p, c.x() + 14, c.y() + 6, text, col, size=10, filled=True)

    # ── предметы (YOLOX) ──
    def _draw_object(self, p, o, ox, oy, sc):
        x1, y1, x2, y2 = o["box"]
        r = QRectF(ox + x1 * sc, oy + y1 * sc, (x2 - x1) * sc, (y2 - y1) * sc)
        col = cat_color(o["cat"])
        label, conf = o["name"], o["score"]
        if o.get("clip"):  # CLIP узнал предмет точнее, чем общий детектор (пульт → айкос)
            name, cat = self._name(o["clip"][0])
            label, conf, col = name, o["clip"][1], cat_color(cat)
        p.setPen(QPen(qc(col, 50), 1))
        p.setBrush(qc(col, 18))
        p.drawRoundedRect(r, 5, 5)
        self._brackets(p, r, col, 12, 2.0)
        text = f"{label} {int(conf * 100)}%"
        if o["hand"]:
            text += f" · {'Л' if o['hand'] == 'ЛЕВАЯ' else 'П'}.РУКА"
        if o["near_face"]:
            text += " · У ЛИЦА"
        w = QFontMetrics(F(10, QFont.Bold, spacing=0.4)).horizontalAdvance(text) + 36
        cy = r.top() - 24 if r.top() > oy + 30 else r.bottom() + 4
        draw_chip(p, max(ox, min(r.left(), ox + self._geom_w() - w)), cy, text, col, size=10, filled=True)

    def _geom_w(self):
        return (self.pix.width() if self.pix else 0) * self._geom[2]

    # ── карточка человека ──
    def _card_model(self, t, person, color, objs):
        if person:
            tag, title = "ОПОЗНАН", person["full"]
            rows = []
            if person["birth_date"]:
                age = calc_age(person["birth_date"])
                rows.append(("ДР", person["birth_date"] + (f"  ·  {age} л." if age is not None else "")))
            for lbl, key in (("ТЕЛ", "phone"), ("РАБОТА", "workplace"), ("АДРЕС", "address"), ("EMAIL", "email")):
                if person[key]:
                    rows.append((lbl, person[key]))
            for ln in person["notes"].splitlines()[:2]:
                if ln.strip():
                    rows.append(("•", ln.strip()))
            rows = rows[:5]
        else:
            tag = "НЕ ОПОЗНАН" if t["face"] is not None else "ЛИЦО НЕ ВИДНО"
            title = "Неизвестный"
            rows = [("", "Нажмите на человека, чтобы добавить в базу")] if t["face"] is not None else []
        chips = []
        for oid in t.get("objects", []):
            o = objs.get(oid)
            if not o:
                continue
            nm, ct = o["name"], o["cat"]
            if o.get("clip"):
                nm, ct = self._name(o["clip"][0])
            txt = nm + (f" · {'Л' if o['hand'] == 'ЛЕВАЯ' else 'П'}.РУКА" if o["hand"] else "")
            chips.append((txt, cat_color(ct), False))
        hold_cats = {objs[i]["cat"] + (objs[i]["hand"] or "") for i in t.get("objects", []) if i in objs}
        hold_keys = {(objs[i]["clip"][0], objs[i]["hand"]) for i in t.get("objects", []) if i in objs and objs[i].get("clip")}
        for slot, (key, prob) in (t.get("hold") or {}).items():
            name, cat = self._name(key)
            hand_name = "ЛЕВАЯ" if slot == "L" else "ПРАВАЯ"
            if cat + hand_name in hold_cats or (key, hand_name) in hold_keys:
                continue
            chips.append((f"{name} {int(prob * 100)}% · {slot.replace('L', 'Л').replace('R', 'П')}.РУКА",
                          cat_color(cat), False))
        if t.get("act"):
            key, prob = t["act"]
            name, cat = self._name(key)
            chips.insert(0, (f"{name}", cat_color(cat), True))
        return {"tag": tag, "title": title, "rows": rows, "chips": chips, "status": person["status"] if person else ""}

    def _draw_card(self, p, t, person, color, body, born, hl, vid, placed, objs):
        m = self._card_model(t, person, color, objs)
        pose = t.get("pose")
        info = pose["info"] if pose else None
        PAD, AV = 12, 44
        f_tag, f_name = F(9, QFont.Bold, MONO, 1.2), F(14, QFont.DemiBold)
        f_row, f_lbl = F(11), F(9, QFont.Bold, MONO, 0.8)
        f_chip = F(10, QFont.Bold, spacing=0.4)
        fm_name, fm_row, fm_chip = QFontMetrics(f_name), QFontMetrics(f_row), QFontMetrics(f_chip)
        # 1) размеры
        wmax = 330
        content_w = AV + 12 + max(fm_name.horizontalAdvance(m["title"]) + (110 if m["status"] else 0), 150)
        for lbl, val in m["rows"]:
            content_w = max(content_w, 52 + fm_row.horizontalAdvance(val))
        if info:
            content_w = max(content_w, fm_row.horizontalAdvance(info["text"]) + 4, 250)
        width = min(max(content_w, 240), wmax) + PAD * 2
        h = PAD + AV + 6
        h += len(m["rows"]) * 17 + (4 if m["rows"] else 0)
        if info:
            h += 8 + 22 + 20
        chip_lines = self._chip_layout(m["chips"], width - PAD * 2, fm_chip)
        if chip_lines:
            h += 8 + len(chip_lines) * 26
        h += PAD - 2
        # 2) место
        face = None
        if t["face"] is not None:
            ox_, oy_, sc_ = self._geom
            fx, fy, fw_, fh_ = [float(v) for v in t["face"][:4]]
            face = QRectF(ox_ + fx * sc_, oy_ + fy * sc_, fw_ * sc_, fh_ * sc_)
        r = self._place(width, h, body, vid, placed, t["tid"], face)
        pos = self._cardpos.get(t["tid"])
        if pos is None:
            pos = [r.left(), r.top()]
        pos[0] += (r.left() - pos[0]) * 0.35
        pos[1] += (r.top() - pos[1]) * 0.35
        self._cardpos[t["tid"]] = pos
        rect = QRectF(pos[0], pos[1], width, h)
        placed.append(rect)
        # 3) рисуем
        p.save()
        p.setOpacity(born)
        # выноска от карточки к человеку, если они не соприкасаются
        if not body.intersects(rect):
            bc = body.center()
            a = QPointF(min(max(bc.x(), rect.left()), rect.right()), min(max(bc.y(), rect.top()), rect.bottom()))
            tgt = QPointF(min(max(a.x(), body.left()), body.right()), min(max(a.y(), body.top()), body.bottom()))
            p.setPen(QPen(qc(color, 110), 1, Qt.DashLine))
            p.drawLine(a, tgt)
            p.setBrush(qc(color))
            p.setPen(Qt.NoPen)
            p.drawEllipse(tgt, 3, 3)
        path = QPainterPath()
        path.addRoundedRect(rect, 14, 14)
        p.setPen(QPen(qc(color, 190 if hl else 110), 1.4 if hl else 1))
        p.setBrush(QColor(5, 13, 10, 226))
        p.drawPath(path)
        p.save()
        p.setClipPath(path)
        grad = QLinearGradient(rect.left(), rect.top(), rect.right(), rect.top())
        grad.setColorAt(0, qc(color, 230))
        grad.setColorAt(1, qc(color, 0))
        p.fillRect(QRectF(rect.left(), rect.top(), rect.width(), 3), grad)
        p.restore()
        x, y = rect.left() + PAD, rect.top() + PAD
        p.drawPixmap(int(x), int(y), self._avatar(t["pid"] if person else None, m["title"], color, AV))
        p.setPen(QPen(qc(color, 200), 1.4))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(QRectF(x - 1, y - 1, AV + 2, AV + 2))
        tx = x + AV + 12
        p.setFont(f_tag)
        p.setPen(qc(color))
        p.drawText(QPointF(tx, y + 12), m["tag"] + f"  ·  TGT-{t['tid']:02d}")
        p.setFont(f_name)
        p.setPen(qc("#ffffff"))
        title = fm_name.elidedText(m["title"], Qt.ElideRight, int(rect.right() - tx - PAD - (96 if m["status"] else 0)))
        p.drawText(QPointF(tx, y + 33), title)
        if m["status"]:
            draw_chip(p, rect.right() - PAD - QFontMetrics(F(9, QFont.Bold, spacing=0.4)).horizontalAdvance(m["status"].upper()) - 26,
                      y + 15, m["status"].upper(), color, size=9)
        y += AV + 6
        for lbl, val in m["rows"]:
            p.setFont(f_lbl)
            p.setPen(qc(C["text3"]))
            p.drawText(QPointF(rect.left() + PAD, y + 12), lbl)
            p.setFont(f_row)
            p.setPen(qc(C["text"]))
            p.drawText(QPointF(rect.left() + PAD + 52, y + 12), fm_row.elidedText(val, Qt.ElideRight, int(rect.width() - PAD * 2 - 52)))
            y += 17
        if m["rows"]:
            y += 4
        if info:
            y += 8
            p.setPen(QPen(qc(C["line2"], 200), 1))
            p.drawLine(QPointF(rect.left() + PAD, y - 4), QPointF(rect.right() - PAD, y - 4))
            cx = rect.left() + PAD
            for text, lvl in (("ГОЛОВА", 2 if info["head"] else 0), ("ТОРС", info["torso"]),
                              (f"РУКИ {info['arms']}/2", info["arms"]), (f"КИСТИ {info['hands']}/2", info["hands"]),
                              (f"НОГИ {info['legs']}/2", info["legs"])):
                col = QColor(C["accent"]) if lvl >= 2 else QColor(C["warn"]) if lvl == 1 else QColor(C["text3"])
                cx += draw_chip(p, cx, y, text, col, size=9, dot=False, h=18) + 5
            y += 22
            p.setFont(F(11, QFont.DemiBold))
            p.setPen(qc(C["accent2"]))
            p.drawText(QPointF(rect.left() + PAD, y + 12), fm_row.elidedText(info["text"], Qt.ElideRight, int(rect.width() - PAD * 2)))
            y += 20
        if chip_lines:
            y += 8
            for line in chip_lines:
                cx = rect.left() + PAD
                for text, col, strong in line:
                    cx += draw_chip(p, cx, y, text, col, size=10, filled=strong, h=22) + 6
                y += 26
        p.restore()

    @staticmethod
    def _chip_layout(chips, maxw, fm):
        lines, cur, w = [], [], 0
        for text, col, strong in chips:
            cw = fm.horizontalAdvance(text) + 8 * 2 + 10 + 6
            if cur and w + cw > maxw:
                lines.append(cur)
                cur, w = [], 0
            cur.append((text, col, strong))
            w += cw
        if cur:
            lines.append(cur)
        return lines[:3]

    def _place(self, w, h, body, vid, placed, tid, face=None):
        """Выбирает место для карточки: над человеком, сбоку или у края кадра на стороне, где нет лица."""
        pad = 8
        top_lim, bot_lim = vid.top() + 50, vid.bottom() - 40

        def clamp(x, y):
            return QRectF(min(max(vid.left() + pad, x), vid.right() - w - pad),
                          min(max(top_lim, y), bot_lim - h), w, h)

        fcx = face.center().x() if face is not None else body.center().x()
        side_first = vid.right() - w - 16 if fcx < vid.center().x() else vid.left() + 16
        side_second = vid.left() + 16 if fcx < vid.center().x() else vid.right() - w - 16
        cands = [
            (body.left(), body.top() - h - 10, True),      # над человеком
            (body.right() + 10, body.top(), True),         # справа
            (body.left() - w - 10, body.top(), True),      # слева
            (side_first, top_lim + 4, False),              # у края, противоположного лицу
            (side_second, top_lim + 4, False),
            (body.left(), body.bottom() + 10, True),       # под человеком
        ]

        def valid(i):
            x, y, strict = cands[i]
            r = clamp(x, y)
            if strict and (abs(r.left() - x) > 2 or abs(r.top() - y) > 2):
                return None
            return None if any(r.intersects(o) for o in placed) else r

        order = list(range(len(cands)))
        sel = self._cardsel.get(tid)
        if sel is not None:
            order.remove(sel)
            order.insert(0, sel)
        for i in order:
            r = valid(i)
            if r is not None:
                self._cardsel[tid] = i
                return r
        return clamp(cands[3][0], cands[3][1])

    # ── HUD ──
    def _paint_hud(self, p, vid, known_n, now):
        x, y = vid.left() + 18, vid.top() + 16
        blink = 0.35 + 0.65 * (0.5 + 0.5 * math.sin(now * 4))
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(5, 13, 10, 200))
        p.drawRoundedRect(QRectF(x - 4, y - 4, 214, 30), 15, 15)
        p.setBrush(qc(C["danger"], 255 * blink))
        p.drawEllipse(QPointF(x + 12, y + 11), 5, 5)
        p.setFont(F(10, QFont.Bold, MONO, 1.2))
        p.setPen(qc(C["text"]))
        p.drawText(QPointF(x + 26, y + 15), "REC")
        p.setPen(qc(C["text2"]))
        p.setFont(F(10, QFont.Medium, MONO))
        p.drawText(QPointF(x + 62, y + 15), self.source_label[:22])
        # правый верхний блок со статистикой (сжимается, если места мало)
        items = [("FPS", f"{self.stats['fps']:.0f}"), ("МС", f"{self.stats['ms']:.0f}"),
                 ("ЦЕЛИ", str(len(self.tracks))), ("ОПОЗН.", str(known_n))]
        f = F(10, QFont.Bold, MONO, 0.6)
        fm = QFontMetrics(f)
        room = vid.width() - 36 - 214 - 14
        for variant in (items, [items[0], items[2]], []):
            total = sum(fm.horizontalAdvance(f"{a} {b}") + 20 for a, b in variant) + 6
            if total <= room:
                break
        if variant:
            xr = vid.right() - 18 - total
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(5, 13, 10, 200))
            p.drawRoundedRect(QRectF(xr - 4, y - 4, total + 8, 30), 15, 15)
            cx = xr + 10
            for a, b in variant:
                p.setFont(f)
                p.setPen(qc(C["text3"]))
                p.drawText(QPointF(cx, y + 15), a)
                cx += fm.horizontalAdvance(a + " ")
                p.setPen(qc(C["accent"]))
                p.drawText(QPointF(cx, y + 15), b)
                cx += fm.horizontalAdvance(b) + 20
        # низ: время + светодиоды модулей
        yb = vid.bottom() - 16
        p.setFont(F(10, QFont.Medium, MONO, 0.6))
        tstr = time.strftime("%d.%m.%Y   %H:%M:%S")
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(5, 13, 10, 190))
        p.drawRoundedRect(QRectF(vid.left() + 12, yb - 18, QFontMetrics(p.font()).horizontalAdvance(tstr) + 18, 26), 13, 13)
        p.setPen(qc(C["text2"], 235))
        p.drawText(QPointF(vid.left() + 21, yb), tstr)
        cx = vid.right() - 20
        mw_ = sum(QFontMetrics(p.font()).horizontalAdvance(n) + 28 for n in self.modules) + 6
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(5, 13, 10, 190))
        p.drawRoundedRect(QRectF(vid.right() - 20 - mw_ + 22, yb - 18, mw_ - 4, 26), 13, 13)
        for name, on in reversed(list(self.modules.items())):
            w = QFontMetrics(p.font()).horizontalAdvance(name)
            cx -= w
            p.setPen(qc(C["accent"] if on else C["text3"], 255 if on else 140))
            p.drawText(QPointF(cx, yb), name)
            p.setPen(Qt.NoPen)
            p.setBrush(qc(C["accent"] if on else C["text3"], 255 if on else 120))
            p.drawEllipse(QPointF(cx - 9, yb - 4), 3, 3)
            cx -= 28

    def _paint_enroll(self, p, vid):
        e = self.enroll
        w, h = min(520, vid.width() - 40), 96
        r = QRectF(vid.center().x() - w / 2, vid.top() + 56, w, h)
        pulse = 0.5 + 0.5 * math.sin(time.time() * 6)
        p.setPen(QPen(qc(C["accent2"], 120 + 100 * pulse), 1.4))
        p.setBrush(QColor(4, 12, 9, 238))
        p.drawRoundedRect(r, 16, 16)
        p.setFont(F(10, QFont.Bold, MONO, 1.6))
        p.setPen(qc(C["accent2"]))
        p.drawText(QPointF(r.left() + 20, r.top() + 24), e["title"])
        p.setFont(F(14, QFont.DemiBold))
        p.setPen(qc("#ffffff"))
        fm = QFontMetrics(p.font())
        p.drawText(QPointF(r.left() + 20, r.top() + 50), fm.elidedText(e["sub"], Qt.ElideRight, int(w - 40)))
        bar = QRectF(r.left() + 20, r.bottom() - 26, w - 40, 6)
        p.setPen(Qt.NoPen)
        p.setBrush(qc(C["line2"]))
        p.drawRoundedRect(bar, 3, 3)
        p.setBrush(qc(C["accent"]))
        p.drawRoundedRect(QRectF(bar.left(), bar.top(), bar.width() * max(0.02, e["progress"]), bar.height()), 3, 3)
        p.setFont(F(10, QFont.Medium, MONO))
        p.setPen(qc(C["text2"]))
        p.drawText(QPointF(r.left() + 20, r.bottom() - 6), f"записано образцов: {e['n']}")

    def _paint_select(self, p, vid):
        p.setFont(F(12, QFont.DemiBold))
        text = "Выделите предмет рамкой  ·  Esc — отмена"
        w = QFontMetrics(p.font()).horizontalAdvance(text) + 36
        r = QRectF(vid.center().x() - w / 2, vid.top() + 16, w, 32)
        p.setPen(QPen(qc(C["accent2"], 150), 1))
        p.setBrush(QColor(5, 13, 10, 230))
        p.drawRoundedRect(r, 16, 16)
        p.setPen(qc(C["accent2"]))
        p.drawText(r, Qt.AlignCenter, text)
        if self._drag:
            (ax, ay), (bx, by) = self._drag
            sel = QRectF(QPointF(min(ax, bx), min(ay, by)), QPointF(max(ax, bx), max(ay, by)))
            p.setPen(QPen(qc(C["accent2"]), 1.6, Qt.DashLine))
            p.setBrush(qc(C["accent2"], 30))
            p.drawRect(sel)

    # ── мышь ──
    def _to_frame(self, pos):
        ox, oy, sc = self._geom
        return (pos.x() - ox) / sc, (pos.y() - oy) / sc

    def mousePressEvent(self, e):
        if self.pix is None:
            return
        if self.select_mode:
            self._drag = [(e.position().x(), e.position().y())] * 2
        else:
            self.clicked.emit(*self._to_frame(e.position()))

    def mouseMoveEvent(self, e):
        if self.select_mode and self._drag:
            self._drag[1] = (e.position().x(), e.position().y())

    def mouseReleaseEvent(self, e):
        if self.select_mode and self._drag:
            (ax, ay), (bx, by) = self._drag
            self._drag = None
            x1, y1 = self._to_frame(QPointF(min(ax, bx), min(ay, by)))
            x2, y2 = self._to_frame(QPointF(max(ax, bx), max(ay, by)))
            if x2 - x1 > 10 and y2 - y1 > 10:
                self.selected.emit((int(x1), int(y1), int(x2), int(y2)))

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape and self.select_mode:
            self.set_select_mode(False)
            self.selected.emit(None)
