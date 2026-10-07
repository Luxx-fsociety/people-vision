"""People Vision — распознавание людей, тела, предметов и действий с камеры/видео (macOS)."""
import csv
import os
import sys
import time

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QRectF, QPointF, QSize, QSettings, QEvent
from PySide6.QtGui import QPixmap, QPainter, QColor, QFont, QPen, QFontMetrics, QKeySequence, QShortcut
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
                               QPushButton, QFileDialog, QDialog, QLineEdit, QTextEdit, QListWidget,
                               QListWidgetItem, QMessageBox, QGridLayout, QComboBox, QFrame,
                               QStackedWidget, QSlider, QStyledItemDelegate, QButtonGroup,
                               QAbstractItemView, QStyle)

import items as itemlib
from engine import Analyzer, PeopleDB, FIELDS, DATA
from hud import VideoWidget, status_color, calc_age, to_pixmap, cat_color
from items import CATEGORIES
from theme import (C, build_qss, MONO, Switch, Logo, app_icon, qc, font, initials_avatar,
                   PersonDelegate, SceneDelegate, ROLE_PID, ROLE_DATA)

STATUSES = ["", "Сотрудник", "Клиент", "Гость", "VIP", "Подозрительный", "Чёрный список"]


# ───────────── поток видео + нейросети ─────────────
class VideoThread(QThread):
    """Читает камеру или файл, прогоняет через нейросети, отдаёт кадр + цели + предметы."""
    frame_ready = Signal(object, object, object, float)
    finished_file = Signal()

    def __init__(self, analyzer):
        super().__init__()
        self.an = analyzer
        self.source = 0
        self.running = False
        self.paused = False
        self.is_file = False

    def open(self, source):
        self.stop()
        self.source, self.is_file = source, isinstance(source, str)
        self.an.mirrored = not self.is_file  # кадр с камеры отражается — левую/правую стороны меняем местами
        self.an.reset()
        self.running = True
        self.start()

    def stop(self):
        self.running = False
        self.wait(3000)

    def run(self):
        cap = cv2.VideoCapture(self.source)
        if not self.is_file:  # камере просим 720p — этого достаточно, а нагрузка в 2 раза меньше, чем на 1080p
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not cap.isOpened():
            self.running = False
            self.finished_file.emit()
            return
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        while self.running:
            if self.paused:
                time.sleep(0.05)
                continue
            t0 = time.time()
            ok, frame = cap.read()
            if not ok:
                break
            if not self.is_file:
                frame = cv2.flip(frame, 1)
            if max(frame.shape[:2]) > 1280:  # видео 4K/1080p уменьшаем: анализ и показ идут по одному кадру
                k = 1280 / max(frame.shape[:2])
                frame = cv2.resize(frame, (int(frame.shape[1] * k), int(frame.shape[0] * k)), interpolation=cv2.INTER_AREA)
            ts = (cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0) if self.is_file else time.time()
            tracks = self.an.process(frame, ts)
            ms = (time.time() - t0) * 1000
            self.frame_ready.emit(frame, [self._snap(t) for t in tracks],
                                  [self._snap_obj(o) for o in self.an.objects_visible], ms)
            if self.is_file:
                time.sleep(max(0, 1 / fps - (time.time() - t0)))
        cap.release()
        if self.running and self.is_file:
            self.finished_file.emit()
        self.running = False

    @staticmethod
    def _snap(t):
        pose = None
        if t.pose is not None:
            pose = {"pts": t.pose["pts"].copy(), "vis": t.pose["vis"].copy(),
                    "contours": t.pose["contours"], "info": t.pose["info"]}
        hands = {sl: {"pts": h["pts"].copy(), "gesture": h["gesture"]} for sl, h in t.hands.items()}
        return {"tid": t.tid, "box": t.box, "face": None if t.face is None else np.array(t.face),
                "feat": t.feat, "pid": t.person_id, "pose": pose, "hold": dict(t.hold), "act": t.act,
                "objects": [o.oid for o in t.objects], "hands": hands,
                "scores": {sl: [(k, float(v)) for k, v in top] for sl, top in t.scores.items()}}

    @staticmethod
    def _snap_obj(o):
        return {"oid": o.oid, "name": o.name, "cat": o.cat, "box": tuple(float(v) for v in o.box),
                "score": float(o.score), "owner": o.owner, "hand": o.hand, "near_face": bool(o.near_face),
                "clip": o.clip}


# ───────────── делегаты списков ─────────────
class EventDelegate(QStyledItemDelegate):
    def sizeHint(self, option, index):
        return QSize(option.rect.width(), 34)

    def paint(self, p, option, index):
        d = index.data(ROLE_DATA)
        if not d:
            return
        tm, text, col = d
        p.save()
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(option.rect)
        p.setFont(font(10, QFont.Medium, MONO))
        p.setPen(qc(C["text3"]))
        p.drawText(QRectF(r.left() + 6, r.top(), 62, r.height()), Qt.AlignVCenter | Qt.AlignLeft, tm)
        p.setPen(Qt.NoPen)
        p.setBrush(qc(col))
        p.drawEllipse(QPointF(r.left() + 82, r.center().y()), 3.5, 3.5)
        p.setFont(font(12))
        p.setPen(qc(C["text"]))
        fm = QFontMetrics(p.font())
        p.drawText(QRectF(r.left() + 96, r.top(), r.width() - 100, r.height()), Qt.AlignVCenter | Qt.AlignLeft,
                   fm.elidedText(text, Qt.ElideRight, int(r.width() - 104)))
        p.setPen(QPen(qc(C["line"]), 1))
        p.drawLine(QPointF(r.left() + 6, r.bottom()), QPointF(r.right() - 6, r.bottom()))
        p.restore()


class ItemDelegate(QStyledItemDelegate):
    """Строка каталога предметов с мини-переключателем."""

    def __init__(self, catalog, trained, parent=None):
        super().__init__(parent)
        self.catalog, self.trained = catalog, trained

    def sizeHint(self, option, index):
        return QSize(option.rect.width(), 56)

    @staticmethod
    def _switch_rect(r):
        return QRectF(r.right() - 52, r.center().y() - 10, 38, 20)

    def paint(self, p, option, index):
        it = self.catalog.find(index.data(ROLE_PID)) if self.catalog else None
        if it is None:
            return
        p.save()
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(option.rect).adjusted(2, 2, -2, -2)
        if option.state & QStyle.State_Selected:
            p.setPen(QPen(qc(C["accent"], 80), 1))
            p.setBrush(qc(C["accent"], 24))
            p.drawRoundedRect(r, 12, 12)
        col = cat_color(it["cat"])
        p.setPen(Qt.NoPen)
        p.setBrush(qc(col, 40))
        p.drawRoundedRect(QRectF(r.left() + 10, r.center().y() - 16, 32, 32), 9, 9)
        p.setBrush(qc(col))
        p.drawEllipse(QPointF(r.left() + 26, r.center().y()), 5, 5)
        on = it.get("enabled", True)
        p.setFont(font(13, QFont.DemiBold))
        p.setPen(qc(C["text"] if on else C["text3"]))
        p.drawText(QPointF(r.left() + 54, r.top() + 24), it["ru"])
        n = self.catalog.n_pos(it["key"])
        state = "  ·  ✓ обучен" if self.trained(it["key"]) else ("  ·  мало образцов" if n else "")
        sub = CATEGORIES.get(it["cat"], CATEGORIES["other"])[0] + (f"  ·  образцов: {n}" if n else "") + state
        p.setFont(font(10, QFont.Medium))
        p.setPen(qc(C["text3"]))
        p.drawText(QPointF(r.left() + 54, r.top() + 41), sub)
        sw = self._switch_rect(r)
        p.setPen(Qt.NoPen)
        p.setBrush(qc(C["accent"]) if on else qc(C["line2"]))
        p.drawRoundedRect(sw, 10, 10)
        p.setBrush(QColor("#02140a") if on else QColor("#a4cdb6"))
        p.drawEllipse(QRectF(sw.left() + (19 if on else 2), sw.top() + 2, 16, 16))
        p.restore()

    def editorEvent(self, event, model, option, index):
        if event.type() == QEvent.MouseButtonRelease:
            r = QRectF(option.rect).adjusted(2, 2, -2, -2)
            if self._switch_rect(r).adjusted(-8, -8, 8, 8).contains(event.position()):
                it = self.catalog.find(index.data(ROLE_PID))
                if it is not None:
                    it["enabled"] = not it.get("enabled", True)
                    self.catalog.save()
                    self.parent().viewport().update()
                return True
        return False


# ───────────── диалоги ─────────────
def section(text):
    lab = QLabel(text)
    lab.setObjectName("h")
    return lab


class PersonDialog(QDialog):
    LABELS = {
        "surname": "ФАМИЛИЯ", "name": "ИМЯ", "patronymic": "ОТЧЕСТВО", "birth_date": "ДАТА РОЖДЕНИЯ",
        "phone": "ТЕЛЕФОН", "email": "E-MAIL", "address": "АДРЕС", "workplace": "МЕСТО РАБОТЫ / ДОЛЖНОСТЬ",
        "status": "СТАТУС", "notes": "ЗАМЕТКИ",
    }

    def __init__(self, parent, title, data=None, photo=None):
        super().__init__(parent)
        data = data or {}
        self.setWindowTitle(title)
        self.setMinimumWidth(700)
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 22, 24, 20)
        root.setSpacing(14)
        head = QLabel(title)
        head.setStyleSheet("font-size:22px;font-weight:700;color:#ffffff;")
        sub = QLabel("Карточка субъекта — данные появятся над человеком в кадре")
        sub.setObjectName("muted")
        root.addWidget(head)
        root.addWidget(sub)

        body = QHBoxLayout()
        body.setSpacing(20)
        left = QVBoxLayout()
        self.photo = QLabel()
        self.photo.setFixedSize(168, 200)
        self.photo.setAlignment(Qt.AlignCenter)
        self.photo.setStyleSheet(f"border:1px solid {C['line2']};background:#060d0a;border-radius:16px;color:{C['text3']};")
        if photo is not None:
            pm = to_pixmap(photo) if isinstance(photo, np.ndarray) else photo
            pm = pm.scaled(168, 200, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            self.photo.setPixmap(pm.copy((pm.width() - 168) // 2, (pm.height() - 200) // 2, 168, 200))
        else:
            self.photo.setText("НЕТ ФОТО")
        left.addWidget(self.photo)
        left.addStretch(1)
        body.addLayout(left)

        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(4)
        self.w = {}
        layout_map = [("surname", 0, 0, 1), ("name", 0, 1, 1), ("patronymic", 1, 0, 1),
                      ("birth_date", 1, 1, 1), ("phone", 2, 0, 1), ("email", 2, 1, 1),
                      ("address", 3, 0, 2), ("workplace", 4, 0, 2), ("status", 5, 0, 2),
                      ("notes", 6, 0, 2)]
        for f, r, c, span in layout_map:
            lab = QLabel(self.LABELS[f])
            lab.setObjectName("h")
            if f == "status":
                w = QComboBox()
                w.setEditable(True)
                w.addItems(STATUSES)
                w.setCurrentText(data.get(f, ""))
            elif f == "notes":
                w = QTextEdit()
                w.setMinimumHeight(86)
                w.setPlainText(data.get(f, ""))
            else:
                w = QLineEdit(data.get(f, ""))
                if f == "birth_date":
                    w.setInputMask("00.00.0000;_")
                    w.setText(data.get(f, ""))
                    w.setPlaceholderText("ДД.ММ.ГГГГ")
            self.w[f] = w
            grid.addWidget(lab, r * 2, c, 1, span)
            grid.addWidget(w, r * 2 + 1, c, 1, span)
            grid.setRowMinimumHeight(r * 2 + 1, 40 if f != "notes" else 90)
        body.addLayout(grid, 1)
        root.addLayout(body)

        btns = QHBoxLayout()
        btns.addStretch(1)
        cancel = QPushButton("Отмена")
        save = QPushButton("Сохранить в базу")
        save.setObjectName("primary")
        save.setMinimumWidth(170)
        cancel.clicked.connect(self.reject)
        save.clicked.connect(self.try_accept)
        btns.addWidget(cancel)
        btns.addWidget(save)
        root.addLayout(btns)

    def try_accept(self):
        d = self.values()
        if not (d["surname"] or d["name"]):
            QMessageBox.warning(self, "Пусто", "Укажите хотя бы фамилию или имя.")
            return
        bd = d["birth_date"]
        if bd and calc_age(bd) is None:
            QMessageBox.warning(self, "Дата", "Дата рождения должна быть в формате ДД.ММ.ГГГГ.")
            return
        self.accept()

    def values(self):
        out = {}
        for f in FIELDS:
            w = self.w[f]
            v = (w.toPlainText() if isinstance(w, QTextEdit) else w.currentText() if isinstance(w, QComboBox)
                 else w.text()).strip()
            if f == "birth_date" and v.replace(".", "").strip() == "":
                v = ""
            out[f] = v
        return out


class ItemDialog(QDialog):
    """Новый предмет по описанию или запись образцов предмета, который держат в руке."""
    NEW = "＋ Новый предмет…"

    def __init__(self, parent, catalog, teach=False, n_neg=0, preselect=None):
        super().__init__(parent)
        self.catalog, self.teach = catalog, teach
        self.setWindowTitle("Обучение предмета" if teach else "Новый предмет")
        self.setMinimumWidth(540)
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 22, 24, 20)
        root.setSpacing(10)
        head = QLabel("Обучить предмет" if teach else "Новый предмет")
        head.setStyleSheet("font-size:22px;font-weight:700;color:#ffffff;")
        root.addWidget(head)
        hint = QLabel("Вы возьмёте предмет в руку перед камерой, а программа запомнит, как он выглядит именно у вас. "
                      "Сначала запишутся пустые руки — для сравнения, потом сам предмет." if teach else
                      "Опишите предмет словами — нейросеть будет искать его в руках. Позже его можно обучить по образцам.")
        hint.setObjectName("muted")
        hint.setWordWrap(True)
        root.addWidget(hint)
        self.combo = QComboBox()
        if teach:
            root.addWidget(section("ЧТО ЗАПОМИНАЕМ"))
            for it in catalog.items:
                self.combo.addItem(it["ru"], it["key"])
            self.combo.addItem(self.NEW, None)
            if preselect:
                i = self.combo.findData(preselect)
                if i >= 0:
                    self.combo.setCurrentIndex(i)
            root.addWidget(self.combo)
            root.addWidget(section("КАКОЙ РУКОЙ ДЕРЖИТЕ"))
            self.hand_combo = QComboBox()
            self.hand_combo.addItems(["Любой (определю сам)", "Левой", "Правой"])
            root.addWidget(self.hand_combo)
            self.neg_box = QPushButton()
            self.neg_box.setCheckable(True)
            self.neg_box.setChecked(n_neg < 16)
            self.neg_box.setCursor(Qt.PointingHandCursor)
            self.neg_box.clicked.connect(self._neg_text)
            self._neg_text()
            root.addSpacing(4)
            root.addWidget(self.neg_box)
        self.new_box = QWidget()
        nb = QVBoxLayout(self.new_box)
        nb.setContentsMargins(0, 0, 0, 0)
        nb.setSpacing(6)
        nb.addWidget(section("НАЗВАНИЕ"))
        self.name = QLineEdit()
        self.name.setPlaceholderText("например: Стики Terea")
        nb.addWidget(self.name)
        nb.addWidget(section("КАТЕГОРИЯ"))
        self.cat = QComboBox()
        for k, (ru, _) in CATEGORIES.items():
            self.cat.addItem(ru.capitalize(), k)
        nb.addWidget(self.cat)
        nb.addWidget(section("ОПИСАНИЕ ДЛЯ НЕЙРОСЕТИ (по-английски, необязательно)"))
        self.en = QLineEdit()
        self.en.setPlaceholderText("a pack of TEREA heated tobacco sticks")
        nb.addWidget(self.en)
        root.addWidget(self.new_box)
        self.combo.currentIndexChanged.connect(self._toggle)
        self._toggle()
        btns = QHBoxLayout()
        btns.addStretch(1)
        cancel = QPushButton("Отмена")
        ok = QPushButton("Начать запись" if teach else "Сохранить")
        ok.setObjectName("primary")
        ok.setMinimumWidth(160)
        cancel.clicked.connect(self.reject)
        ok.clicked.connect(self._accept)
        btns.addWidget(cancel)
        btns.addWidget(ok)
        root.addSpacing(6)
        root.addLayout(btns)

    def _neg_text(self):
        self.neg_box.setText(("▣  Записать пустые руки (3 секунды)" if self.neg_box.isChecked()
                              else "▢  Записать пустые руки (3 секунды)"))

    def _is_new(self):
        return (not self.teach) or self.combo.currentData() is None

    def _toggle(self):
        self.new_box.setVisible(self._is_new())

    def _accept(self):
        if self._is_new() and not self.name.text().strip():
            QMessageBox.warning(self, "Название", "Введите название предмета.")
            return
        self.accept()

    def hand(self):
        return ("any", "L", "R")[self.hand_combo.currentIndex()]

    def redo_neg(self):
        return self.neg_box.isChecked()

    def result_key(self):
        """Возвращает ключ предмета (создаёт новый при необходимости)."""
        if not self._is_new():
            return self.combo.currentData()
        en = [self.en.text().strip()] if self.en.text().strip() else []
        return self.catalog.add_item(self.name.text().strip(), en, self.cat.currentData())


# ───────────── главное окно ─────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = QSettings("PeopleVision", "App")
        self.setWindowTitle("People Vision")
        self.resize(1500, 900)
        self.setMinimumSize(1180, 720)
        self.db = PeopleDB()
        self.analyzer = Analyzer(self.db)
        self.catalog = self.analyzer.catalog
        self.thread = VideoThread(self.analyzer)
        self.video = VideoWidget()
        self.video.db = self.db
        self.video.setFocusPolicy(Qt.StrongFocus)
        self.last_frame, self.last_tracks, self.last_objects = None, [], []
        self._frame_ids = set()
        self._ui_t = 0.0
        self._scene_items = {}
        self._ev_state = {}
        self._scores_tab, self._scores_on = False, False
        self._enroll = None
        self.enroll_timer = QTimer(self)
        self.enroll_timer.setInterval(100)
        self.enroll_timer.timeout.connect(self._enroll_tick)
        self.refresh_names()

        root = QWidget()
        rl = QVBoxLayout(root)
        rl.setContentsMargins(16, 12, 16, 16)
        rl.setSpacing(12)
        rl.addWidget(self._build_header())
        body = QHBoxLayout()
        body.setSpacing(14)
        body.addWidget(self.video, 1)
        body.addWidget(self._build_side())
        rl.addLayout(body, 1)
        self.setCentralWidget(root)

        self.thread.frame_ready.connect(self.on_frame)
        self.thread.finished_file.connect(self.on_finished)
        self.video.clicked.connect(self.on_click)
        QShortcut(QKeySequence(Qt.Key_Space), self, activated=self.toggle_pause)
        QShortcut(QKeySequence("Ctrl+F"), self, activated=lambda: (self.set_tab(0), self.search.setFocus()))
        self.refresh_list()
        QTimer.singleShot(250, self.start_camera)

    # ── сборка интерфейса ──
    def _build_header(self):
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(4, 0, 4, 0)
        h.setSpacing(8)
        h.addWidget(Logo())
        h.addStretch(1)
        seg = QFrame()
        seg.setObjectName("card")
        sl = QHBoxLayout(seg)
        sl.setContentsMargins(5, 5, 5, 5)
        sl.setSpacing(2)
        self.btn_cam = QPushButton("●  Камера")
        self.btn_file = QPushButton("▤  Видеофайл…")
        for b in (self.btn_cam, self.btn_file):
            b.setProperty("src", True)
            b.setCheckable(True)
            b.setCursor(Qt.PointingHandCursor)
        self.btn_cam.setChecked(True)
        self.camsel = QComboBox()
        self.camsel.addItems(list_cameras())
        self.camsel.setCurrentIndex(min(int(self.settings.value("camera", 0)), self.camsel.count() - 1))
        self.camsel.setFixedWidth(210)
        self.camsel.setMaxVisibleItems(8)
        self.camsel.setStyleSheet("QComboBox{padding:5px 10px;border-radius:9px;}")
        sl.addWidget(self.btn_cam)
        sl.addWidget(self.camsel)
        sl.addWidget(self.btn_file)
        h.addWidget(seg)
        self.btn_pause = QPushButton("❚❚  Пауза")
        self.btn_pause.setFixedWidth(130)
        h.addWidget(self.btn_pause)
        h.addStretch(1)
        self.status = QLabel("")
        self.status.setObjectName("muted")
        self.status.setMinimumWidth(210)
        self.status.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        h.addWidget(self.status)
        src = QButtonGroup(self)  # «Камера» и «Видеофайл» — взаимоисключающие источники
        src.setExclusive(True)
        src.addButton(self.btn_cam)
        src.addButton(self.btn_file)
        self.btn_cam.clicked.connect(self.start_camera)
        self.btn_file.clicked.connect(self.open_file)
        self.camsel.activated.connect(lambda _: self.start_camera())
        self.btn_pause.clicked.connect(self.toggle_pause)
        return w

    def _sw(self, key, label, default, cb, hint=None):
        on = self.settings.value("sw/" + key, default, type=bool)

        def wrapped(v):
            self.settings.setValue("sw/" + key, v)
            cb(v)
        return Switch(label, on, hint, wrapped)

    def _build_side(self):
        panel = QFrame()
        panel.setObjectName("panel")
        panel.setFixedWidth(400)
        v = QVBoxLayout(panel)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(6)
        tabs = QHBoxLayout()
        tabs.setSpacing(2)
        self.tab_btns = []
        grp = QButtonGroup(self)
        for i, name in enumerate(("ЛЮДИ", "В КАДРЕ", "ПРЕДМЕТЫ", "ВИД")):
            b = QPushButton(name)
            b.setProperty("tab", True)
            b.setCheckable(True)
            b.setCursor(Qt.PointingHandCursor)
            grp.addButton(b, i)
            tabs.addWidget(b)
            self.tab_btns.append(b)
        self.tab_btns[0].setChecked(True)
        v.addLayout(tabs)
        self.stack = QStackedWidget()
        self.stack.addWidget(self._page_people())
        self.stack.addWidget(self._page_scene())
        self.stack.addWidget(self._page_items())
        self.stack.addWidget(self._page_view())
        v.addWidget(self.stack, 1)
        grp.idClicked.connect(self.set_tab)
        return panel

    def set_tab(self, i):
        self.stack.setCurrentIndex(i)
        self.tab_btns[i].setChecked(True)
        self._scores_tab = (i == 2)
        self.video.show_scores = self._scores_tab or self._scores_on

    def _page_people(self):
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(8)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Поиск: ФИО, телефон, статус, адрес…")
        self.search.setClearButtonEnabled(True)
        lay.addWidget(self.search)
        self.sw_inframe = Switch("Только те, кто сейчас в кадре", False)
        lay.addWidget(self.sw_inframe)
        self.count_lbl = QLabel("")
        self.count_lbl.setObjectName("h")
        lay.addWidget(self.count_lbl)
        self.listw = QListWidget()
        self.listw.setItemDelegate(PersonDelegate(self.db, status_color, self.listw))
        self.listw.setMouseTracking(True)
        self.listw.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        lay.addWidget(self.listw, 1)
        self.detail = QLabel("")
        self.detail.setObjectName("detail")
        self.detail.setWordWrap(True)
        self.detail.setTextFormat(Qt.RichText)
        self.detail.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.detail.setMinimumHeight(150)
        lay.addWidget(self.detail)
        row = QHBoxLayout()
        self.btn_edit = QPushButton("Изменить")
        self.btn_del = QPushButton("Удалить")
        self.btn_del.setObjectName("danger")
        row.addWidget(self.btn_edit)
        row.addWidget(self.btn_del)
        lay.addLayout(row)
        self.search.textChanged.connect(self.refresh_list)
        self.sw_inframe.toggled.connect(lambda _: self.refresh_list())
        self.listw.currentItemChanged.connect(self.show_detail)
        self.listw.itemDoubleClicked.connect(lambda _: self.edit_person())
        self.btn_edit.clicked.connect(self.edit_person)
        self.btn_del.clicked.connect(self.delete_person)
        return page

    def _page_scene(self):
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(8)
        lay.addWidget(section("ЦЕЛИ В КАДРЕ"))
        self.scene_list = QListWidget()
        self.scene_list.setItemDelegate(SceneDelegate(self.scene_list))
        self.scene_list.setSelectionMode(QAbstractItemView.NoSelection)
        self.scene_list.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        lay.addWidget(self.scene_list, 3)
        hr = QHBoxLayout()
        hr.addWidget(section("ЖУРНАЛ СОБЫТИЙ"))
        hr.addStretch(1)
        opn = QPushButton("Открыть CSV")
        opn.setObjectName("ghost")
        opn.setCursor(Qt.PointingHandCursor)
        clear = QPushButton("Очистить")
        clear.setObjectName("ghost")
        clear.setCursor(Qt.PointingHandCursor)
        hr.addWidget(opn)
        hr.addWidget(clear)
        lay.addLayout(hr)
        self.event_list = QListWidget()
        self.event_list.setItemDelegate(EventDelegate(self.event_list))
        self.event_list.setSelectionMode(QAbstractItemView.NoSelection)
        self.event_list.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        lay.addWidget(self.event_list, 4)
        clear.clicked.connect(self.event_list.clear)
        opn.clicked.connect(lambda: os.system(f'open "{self._csv_path()}"'))
        return page

    def _page_items(self):
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(8)
        an, v = self.analyzer, self.video
        self.sw_obj = self._sw("obj", "Предметы в кадре (телефон, напитки…)", True,
                               lambda on: (setattr(an, "use_objects", on), v.modules.__setitem__("OBJ", on)),
                               "YOLOX находит телефоны, бутылки, чашки, ноутбуки, сумки и привязывает их к людям и рукам")
        self.sw_scan = self._sw("scan", "Что в руках и что делает человек", True,
                                lambda on: (setattr(an, "use_scan", on), v.modules.__setitem__("SCAN", on)),
                                "CLIP: айкос, стики, сигареты, вейп, напитки; действия — курит, пьёт, говорит по телефону")
        lay.addWidget(self.sw_obj)
        lay.addWidget(self.sw_scan)
        lay.addWidget(section("КАТАЛОГ ПРЕДМЕТОВ ДЛЯ РУК"))
        self.item_list = QListWidget()
        self.item_list.setItemDelegate(ItemDelegate(self.catalog, self._is_trained, self.item_list))
        self.item_list.setMouseTracking(True)
        self.item_list.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        lay.addWidget(self.item_list, 1)
        sr = QHBoxLayout()
        sl = QLabel("ЧУВСТВИТЕЛЬНОСТЬ")
        sl.setObjectName("h")
        sl.setToolTip("Правее — предметы находятся чаще, но возможны ложные срабатывания")
        self.sens = QSlider(Qt.Horizontal)
        self.sens.setRange(30, 90)
        self.sens.setValue(int(self.settings.value("sensitivity", 60)))
        self.sens_lbl = QLabel("")
        self.sens_lbl.setObjectName("muted")
        sr.addWidget(sl)
        sr.addWidget(self.sens, 1)
        sr.addWidget(self.sens_lbl)
        lay.addLayout(sr)
        self.sens.valueChanged.connect(self._sens_changed)
        self._sens_changed(self.sens.value())
        self.btn_teach = QPushButton("◎  Обучить предмет")
        self.btn_teach.setObjectName("primary")
        self.btn_teach.setMinimumHeight(38)
        lay.addWidget(self.btn_teach)
        row = QHBoxLayout()
        self.btn_add_item = QPushButton("＋ Новый предмет")
        self.btn_reset_item = QPushButton("Сбросить образцы")
        self.btn_reset_item.setToolTip("Удалить записанные образцы выбранного предмета")
        self.btn_del_item = QPushButton("✕")
        self.btn_del_item.setObjectName("danger")
        self.btn_del_item.setFixedWidth(44)
        self.btn_del_item.setToolTip("Удалить предмет из каталога")
        row.addWidget(self.btn_add_item, 1)
        row.addWidget(self.btn_reset_item, 1)
        row.addWidget(self.btn_del_item)
        lay.addLayout(row)
        tip = QLabel("Если стик или устройство не находится — выберите предмет и нажмите «Обучить предмет»: "
                     "3 секунды пустых рук, затем 8 секунд с предметом в руке. Подсказки нейросети видны у рук; "
                     "ползунок вправо — находит чаще, влево — реже ошибается.")
        tip.setObjectName("muted")
        tip.setWordWrap(True)
        lay.addWidget(tip)
        self.btn_add_item.clicked.connect(self.add_item)
        self.btn_teach.clicked.connect(self.start_teach)
        self.btn_reset_item.clicked.connect(self.reset_item)
        self.btn_del_item.clicked.connect(self.delete_item)
        has = self.catalog is not None
        for w in (self.sw_scan, self.item_list, self.btn_add_item, self.btn_teach, self.btn_reset_item,
                  self.btn_del_item, self.sens):
            w.setEnabled(has)
        if not has:
            tip.setText("Модель CLIP не найдена — запустите ./run.sh, он скачает недостающие файлы.")
        self.refresh_items()
        return page

    def _page_view(self):
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(2)
        v, an = self.video, self.analyzer
        lay.addWidget(section("НЕЙРОСЕТИ"))
        lay.addWidget(self._sw("pose", "Анализ тела (поза, жесты)", True,
                               lambda on: (setattr(an, "use_pose", on), v.modules.__setitem__("POSE", on))))
        lay.addWidget(self._sw("scores", "Оценки нейросети у рук", False,
                               lambda on: (setattr(self, "_scores_on", on),
                                           setattr(v, "show_scores", self._scores_tab or on)),
                               "Показывает, что нейросеть думает о предмете в каждой руке"))
        lay.addSpacing(10)
        lay.addWidget(section("СЛОИ НА ВИДЕО"))
        for key, label, default, attr in (
                ("body", "Рамки тел", True, "show_body"), ("skel", "Скелет", True, "show_skeleton"),
                ("sil", "Силуэт тела", True, "show_silhouette"), ("lab", "Подписи частей тела", False, "show_labels"),
                ("objs", "Рамки предметов", True, "show_objects"), ("cards", "Карточки людей", True, "show_cards"),
                ("hud", "HUD-панели", True, "show_hud")):
            lay.addWidget(self._sw(key, label, default, lambda on, a=attr: setattr(v, a, on)))
        lay.addStretch(1)
        info = QLabel("Пробел — пауза  ·  Ctrl+F — поиск по базе\nКлик по человеку — добавить или изменить данные")
        info.setObjectName("muted")
        lay.addWidget(info)
        return page

    # ── источники ──
    def on_finished(self):
        if self.thread.is_file:
            self.status.setText("Видео закончилось")
        else:
            self.video.clear(f"НЕ УДАЛОСЬ ОТКРЫТЬ КАМЕРУ «{self.camsel.currentText()}»\n\n"
                             "Проверьте, что доступ к камере разрешён: Системные настройки →\n"
                             "Конфиденциальность и безопасность → Камера, или выберите другую камеру")

    def start_camera(self):
        self.status.setText("")
        self.btn_cam.setChecked(True)
        self.thread.paused = False
        self.btn_pause.setText("❚❚  Пауза")
        self.settings.setValue("camera", self.camsel.currentIndex())
        self.video.source_label = self.camsel.currentText().upper()
        self.video.message = "ЗАПУСК КАМЕРЫ…"
        self.thread.open(self.camsel.currentIndex())

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Выберите видео", "",
                                              "Видео (*.mp4 *.mov *.avi *.mkv *.m4v *.webm)")
        if path:
            self.btn_file.setChecked(True)
            self.thread.paused = False
            self.btn_pause.setText("❚❚  Пауза")
            self.video.source_label = os.path.basename(path)
            self.status.setText("")
            self.thread.open(path)
        else:
            (self.btn_file if self.thread.is_file else self.btn_cam).setChecked(True)

    def toggle_pause(self):
        self.thread.paused = not self.thread.paused
        self.btn_pause.setText("▶  Продолжить" if self.thread.paused else "❚❚  Пауза")

    # ── кадры ──
    def on_frame(self, frame, tracks, objects, ms):
        self.last_frame, self.last_tracks, self.last_objects = frame, tracks, objects
        sel = self._selected()
        self.video.highlight = {sel} if sel is not None else set()
        self.video.update_frame(frame, tracks, objects, ms)
        now = time.time()
        ids = {t["pid"] for t in tracks if t["pid"] in self.db.people}
        if self.sw_inframe.isChecked() and self._frame_ids != ids:
            self._frame_ids = ids
            self.refresh_list()
        self._frame_ids = ids
        self.track_events(tracks, objects, now)
        if now - self._ui_t > 0.25:
            self._ui_t = now
            self.update_scene(tracks, objects)

    # ── вкладка «В кадре» ──
    def update_scene(self, tracks, objects):
        objs = {o["oid"]: o for o in objects}
        live = {t["tid"] for t in tracks}
        for tid in [k for k in self._scene_items if k not in live]:
            item = self._scene_items.pop(tid)
            self.scene_list.takeItem(self.scene_list.row(item))
        for t in sorted(tracks, key=lambda t: t["tid"]):
            person = self.db.people.get(t["pid"]) if t["pid"] is not None else None
            col = status_color(person["status"]) if person else QColor(C["danger"] if t["face"] is not None else C["warn"])
            name = person["full"] if person else "Неизвестный"
            tag = f"TGT-{t['tid']:02d} · " + ("ОПОЗНАН" if person else "НЕ ОПОЗНАН" if t["face"] is not None else "ЛИЦО НЕ ВИДНО")
            pose = t["pose"]["info"]["text"] if t["pose"] else "поза не определена"
            chips = []
            if t.get("act"):
                n, c = self.video._name(t["act"][0])
                chips.append((n, cat_color(c)))
            for oid in t["objects"]:
                o = objs.get(oid)
                if o:
                    n, c = (self.video._name(o["clip"][0]) if o.get("clip") else (o["name"], o["cat"]))
                    chips.append((n, cat_color(c)))
            for slot, (key, prob) in t["hold"].items():
                n, c = self.video._name(key)
                chips.append((f"{n} {int(prob * 100)}%", cat_color(c)))
            data = {"tag": tag, "name": name, "pose": pose, "color": col, "chips": chips[:4],
                    "avatar": self.listw.itemDelegate().avatar(t["pid"], name, col) if person
                    else initials_avatar("?", 40, col)}
            item = self._scene_items.get(t["tid"])
            if item is None:
                item = QListWidgetItem()
                self.scene_list.addItem(item)
                self._scene_items[t["tid"]] = item
            item.setData(ROLE_DATA, data)
            item.setSizeHint(QSize(100, 70 + (24 if chips else 0)))
        self.scene_list.viewport().update()

    # ── журнал событий ──
    def _csv_path(self):
        return os.path.join(DATA, "events.csv")

    def _log(self, text, color, kind, who=""):
        item = QListWidgetItem()
        item.setData(ROLE_DATA, (time.strftime("%H:%M:%S"), text, color))
        self.event_list.insertItem(0, item)
        while self.event_list.count() > 300:
            self.event_list.takeItem(self.event_list.count() - 1)
        try:
            new = not os.path.exists(self._csv_path())
            with open(self._csv_path(), "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(["время", "источник", "кто", "тип", "событие"])
                w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), self.video.source_label, who, kind, text])
        except OSError:
            pass

    def track_events(self, tracks, objects, now):
        """Фиксируем устойчивые события: появление, опознание, предметы в руках, действия."""
        objs = {o["oid"]: o for o in objects}
        live = set()
        for t in tracks:
            tid = t["tid"]
            person = self.db.people.get(t["pid"]) if t["pid"] is not None else None
            who = person["full"] if person else f"TGT-{tid:02d}"
            conds = {("seen",): (f"{who} в кадре" if person else f"TGT-{tid:02d} появился в кадре",
                                 QColor(C["accent2"]), "вход", 1.0)}
            if person:
                conds[("known", t["pid"])] = (f"Опознан: {who}", status_color(person["status"]), "опознан", 0.4)
            for slot, (key, prob) in t["hold"].items():
                n, c = self.video._name(key)
                hand = "левой" if slot == "L" else "правой"
                conds[("hold", slot, key)] = (f"{who}: {n} в {hand} руке", cat_color(c), "предмет", 1.2)
            for oid in t["objects"]:
                o = objs.get(oid)
                if o:
                    where = f"в {'левой' if o['hand'] == 'ЛЕВАЯ' else 'правой'} руке" if o["hand"] else "рядом"
                    nm, ct = (self.video._name(o["clip"][0]) if o.get("clip") else (o["name"], o["cat"]))
                    conds[("obj", nm, o["hand"])] = (f"{who}: {nm} {where}", cat_color(ct), "предмет", 1.2)
            if t.get("act"):
                n, c = self.video._name(t["act"][0])
                conds[("act", t["act"][0])] = (f"{who}: {n}", cat_color(c), "действие", 1.0)
            for key, (text, col, kind, dwell) in conds.items():
                k = (tid,) + key
                live.add(k)
                st = self._ev_state.setdefault(k, {"since": now, "fired": False, "last": now, "cool": 0.0})
                st["last"] = now
                if not st["fired"] and now - st["since"] >= dwell and now > st["cool"]:
                    st["fired"] = True
                    self._log(text, col, kind, who)
        for k in list(self._ev_state):
            st = self._ev_state[k]
            if k not in live and now - st["last"] > 1.5:
                if st["fired"] and k[1] == "seen":
                    self._log(f"TGT-{k[0]:02d} вышел из кадра", QColor(C["text3"]), "выход")
                    del self._ev_state[k]
                elif st["fired"]:  # повторное срабатывание не раньше чем через 8 секунд
                    self._ev_state[k] = {"since": now, "fired": False, "last": now, "cool": now + 8.0}
                else:
                    del self._ev_state[k]

    # ── предметы ──
    def refresh_names(self):
        names = {}
        if self.catalog:
            for it in self.catalog.items + self.catalog.acts:
                names[it["key"]] = (it["ru"], it["cat"])
        self.video.names = names

    def refresh_items(self):
        if not self.catalog:
            return
        keep = self.item_list.currentItem().data(ROLE_PID) if self.item_list.currentItem() else None
        self.item_list.clear()
        for it in self.catalog.items:
            row = QListWidgetItem()
            row.setData(ROLE_PID, it["key"])
            self.item_list.addItem(row)
            if it["key"] == keep:
                self.item_list.setCurrentItem(row)
        self.refresh_names()

    def _sens_changed(self, v):
        """Ползунок — чувствительность: чем правее, тем ниже порог уверенности."""
        self.settings.setValue("sensitivity", v)
        self.sens_lbl.setText(f"{v}%")
        self.analyzer.hold_thr = 1.1 - v / 100.0
        self.analyzer.act_thr = min(0.95, max(0.72, self.analyzer.hold_thr + 0.2))

    def add_item(self):
        dlg = ItemDialog(self, self.catalog)
        if dlg.exec() == QDialog.Accepted:
            dlg.result_key()
            self.refresh_items()

    def delete_item(self):
        cur = self.item_list.currentItem()
        if cur is None:
            return
        it = self.catalog.find(cur.data(ROLE_PID))
        if it and QMessageBox.question(self, "Удалить", f"Удалить предмет «{it['ru']}» из каталога?") == QMessageBox.Yes:
            self.catalog.remove_item(it["key"])
            self.refresh_items()

    # ── обучение предмета: пустые руки → предмет в руке ──
    def _is_trained(self, key):
        sc = self.analyzer.scanner
        return bool(sc) and any(k == key for (k, _t) in sc.rec.protos)

    def reset_item(self):
        cur = self.item_list.currentItem()
        if cur is None:
            return
        key = cur.data(ROLE_PID)
        if self.catalog.n_pos(key):
            self.catalog.clear_examples(key)
            self.item_list.viewport().update()
            self.status.setText("Образцы удалены")

    def start_teach(self):
        sc = self.analyzer.scanner
        if sc is None:
            return
        if self.last_frame is None:
            QMessageBox.information(self, "Нет кадра", "Сначала запустите камеру или видео.")
            return
        if self._enroll:
            return
        cur = self.item_list.currentItem()
        n_neg = max((len(a) for a in self.catalog.examples.get(itemlib.NEG_KEY, {}).values()), default=0)
        dlg = ItemDialog(self, self.catalog, teach=True, n_neg=n_neg, preselect=cur.data(ROLE_PID) if cur else None)
        if dlg.exec() != QDialog.Accepted:
            return
        key = dlg.result_key()
        self.refresh_items()
        self._enroll = {"key": key, "slot": dlg.hand(), "need_neg": dlg.redo_neg() or n_neg < 8,
                        "stage": None, "name": self.catalog.find(key)["ru"]}
        self._enroll["old_neg"] = self.catalog.examples.pop(itemlib.NEG_KEY, None) if self._enroll["need_neg"] else None
        self.thread.paused = False
        self.btn_pause.setText("❚❚  Пауза")
        self._enroll_stage("neg" if self._enroll["need_neg"] else "pos")
        self.enroll_timer.start()

    def _enroll_stage(self, stage):
        e, sc = self._enroll, self.analyzer.scanner
        e["stage"] = stage
        if stage == "neg":
            sc.start_learn("neg", seconds=3.5, cap=70)
            e["title"], e["sub"] = "ШАГ 1 ИЗ 2", "Держите руки перед камерой БЕЗ предмета"
        else:
            sc.start_learn("pos", slot=e["slot"], seconds=8.5, cap=90)
            e["title"] = "ШАГ 2 ИЗ 2" if e["need_neg"] else "ЗАПИСЬ ОБРАЗЦОВ"
            e["sub"] = f"Возьмите «{e['name']}» в руку, поворачивайте и приближайте к камере"
        self.video.enroll = {"title": e["title"], "sub": e["sub"], "progress": 0.0, "n": 0}

    def _enroll_tick(self):
        e, sc = self._enroll, self.analyzer.scanner
        if not e or sc is None or sc.learn is None:
            self.enroll_timer.stop()
            return
        prog, n = sc.learn_progress()
        self.video.enroll = {"title": e["title"], "sub": e["sub"], "progress": prog, "n": n}
        if prog < 1.0:
            return
        samples = sc.finish_learn()
        got = max((len(a) for a in samples.values()), default=0)
        if got < 6:
            self._enroll_abort("Не получилось: нейросеть почти не видела руки в кадре. "
                               "Поднимите руки на уровень груди, ближе к камере, и повторите.")
            return
        if e["stage"] == "neg":
            self.catalog.add_examples(itemlib.NEG_KEY, samples, cap=120)
            self._enroll_stage("pos")
            return
        self.catalog.add_examples(e["key"], samples)
        self.enroll_timer.stop()
        self.video.enroll = None
        self._enroll = None
        self.refresh_items()
        self.status.setText(f"Готово: «{self.catalog.find(e['key'])['ru']}» — {got} образцов")
        self.set_tab(2)

    def _enroll_end(self):
        """Выход из записи без результата: возвращаем прежние «пустые руки», если новые не успели записаться."""
        self.enroll_timer.stop()
        sc = self.analyzer.scanner
        if sc and sc.learn is not None:
            sc.finish_learn()
        old = (self._enroll or {}).get("old_neg")
        if old is not None and itemlib.NEG_KEY not in self.catalog.examples:
            self.catalog.examples[itemlib.NEG_KEY] = old
        self.video.enroll = None
        self._enroll = None

    def _enroll_abort(self, text):
        self._enroll_end()
        QMessageBox.information(self, "Обучение", text)

    # ── клик по человеку ──
    def on_click(self, x, y):
        if self.last_frame is None:
            return
        hit = None
        for t in self.last_tracks:
            x1, y1, x2, y2 = t["box"]
            if x1 <= x <= x2 and y1 <= y <= y2:
                area = (x2 - x1) * (y2 - y1)
                if hit is None or area < (hit["box"][2] - hit["box"][0]) * (hit["box"][3] - hit["box"][1]):
                    hit = t
        if hit is None:
            return
        was_paused = self.thread.paused
        self.thread.paused = True
        frame = self.last_frame
        if hit["pid"] is not None and hit["pid"] in self.db.people:
            self.edit_by_id(hit["pid"], hit)
        elif hit["face"] is None or hit["feat"] is None:
            QMessageBox.information(self, "Лицо не видно",
                                    "Чтобы запомнить человека, лицо должно быть видно в кадре.")
        else:
            fx, fy, fw, fh = [int(v) for v in hit["face"][:4]]
            m = int(fw * 0.4)
            crop = frame[max(0, fy - m):fy + fh + m, max(0, fx - m):fx + fw + m].copy()
            dlg = PersonDialog(self, "Новый субъект", photo=crop)
            if dlg.exec() == QDialog.Accepted:
                ok, jpg = cv2.imencode(".jpg", crop)
                pid = self.db.add_person(dlg.values(), hit["feat"], jpg.tobytes() if ok else None)
                self.analyzer.reset()
                self.refresh_list(select=pid)
                self.set_tab(0)
        self.thread.paused = was_paused

    def edit_by_id(self, pid, hit=None):
        p = self.db.people[pid]
        photo = None
        blob = self.db.photo(pid)
        if blob:
            pm = QPixmap()
            pm.loadFromData(blob)
            photo = pm
        dlg = PersonDialog(self, "Редактирование", p, photo)
        if dlg.exec() == QDialog.Accepted:
            self.db.update_person(pid, dlg.values())
            if hit is not None and hit["feat"] is not None:
                self.db.add_embedding(pid, hit["feat"])  # ещё один образец лица
            self.refresh_list(select=pid)

    # ── список людей и поиск ──
    def refresh_list(self, *_, select=None):
        keep = select if select is not None else self._selected()
        ids = self.db.search(self.search.text())
        if self.sw_inframe.isChecked():
            present = {t["pid"] for t in self.last_tracks}
            ids = [i for i in ids if i in present]
        self.listw.blockSignals(True)
        self.listw.clear()
        row_to_select = None
        for pid in ids:
            it = QListWidgetItem()
            it.setData(ROLE_PID, pid)
            self.listw.addItem(it)
            if pid == keep:
                row_to_select = it
        self.listw.blockSignals(False)
        if row_to_select:
            self.listw.setCurrentItem(row_to_select)
        else:
            self.listw.setCurrentItem(None)
            self.show_detail(None)
        self.count_lbl.setText(f"ЗАПИСЕЙ  {len(ids)} / {len(self.db.people)}")

    def show_detail(self, item, *_):
        if item is None:
            self.detail.setText("Выберите запись или нажмите на человека в кадре,<br>чтобы добавить его в базу")
            return
        p = self.db.people[item.data(ROLE_PID)]
        age = calc_age(p["birth_date"]) if p["birth_date"] else None
        rows = [("Дата рождения", p["birth_date"] + (f"  ({age} л.)" if age is not None else "") if p["birth_date"] else ""),
                ("Статус", p["status"]), ("Телефон", p["phone"]), ("E-mail", p["email"]),
                ("Адрес", p["address"]), ("Работа", p["workplace"]), ("Заметки", p["notes"]),
                ("Добавлен", p["added_at"])]
        html = f"<div style='font-size:15px;font-weight:700;color:#ffffff;margin-bottom:6px'>{p['full']}</div>"
        for k, v in rows:
            if v:
                html += (f"<div style='margin:2px 0'><span style='color:{C['text3']}'>{k}&nbsp;&nbsp;</span>"
                         f"<span style='color:{C['text']}'>{v.replace(chr(10), '<br>')}</span></div>")
        self.detail.setText(html)

    def _selected(self):
        it = self.listw.currentItem()
        return it.data(ROLE_PID) if it else None

    def edit_person(self):
        pid = self._selected()
        if pid is not None:
            self.edit_by_id(pid)

    def delete_person(self):
        pid = self._selected()
        if pid is not None and QMessageBox.question(
                self, "Удалить", f"Удалить «{self.db.people[pid]['full']}» из базы?") == QMessageBox.Yes:
            self.db.delete_person(pid)
            self.analyzer.reset()
            self.refresh_list()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape and self._enroll:
            self._enroll_end()
            self.status.setText("Запись отменена")
        else:
            super().keyPressEvent(e)

    def closeEvent(self, e):
        self.thread.stop()
        if self.analyzer.scanner:
            self.analyzer.scanner.alive = False
        e.accept()


def list_cameras():
    """Названия камер в том же порядке, что и индексы OpenCV (AVFoundation)."""
    try:
        import AVFoundation as AV
        names = [str(d.localizedName()) for d in AV.AVCaptureDevice.devicesWithMediaType_(AV.AVMediaTypeVideo)]
        if names:
            return names
    except Exception:
        pass
    return [f"Камера {i}" for i in range(3)]


def request_camera_access():
    """macOS: запрашиваем доступ к камере из главного потока."""
    try:
        import AVFoundation as AV
        status = AV.AVCaptureDevice.authorizationStatusForMediaType_(AV.AVMediaTypeVideo)
        if status == 0:
            done = []
            AV.AVCaptureDevice.requestAccessForMediaType_completionHandler_(
                AV.AVMediaTypeVideo, lambda granted: done.append(granted))
            t0 = time.time()
            while not done and time.time() - t0 < 60:
                QApplication.processEvents()
                time.sleep(0.05)
    except Exception:
        pass


def dark_titlebar():
    """Тёмная рамка окна независимо от темы macOS."""
    try:
        import AppKit
        AppKit.NSApplication.sharedApplication().setAppearance_(
            AppKit.NSAppearance.appearanceNamed_("NSAppearanceNameDarkAqua"))
    except Exception:
        pass


if __name__ == "__main__":
    os.environ["OPENCV_AVFOUNDATION_SKIP_AUTH"] = "1"
    app = QApplication(sys.argv)
    app.setApplicationName("People Vision")
    app.setStyleSheet(build_qss())
    app.setWindowIcon(app_icon())
    dark_titlebar()
    request_camera_access()
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
