"""Нейросетевое ядро: детектор людей (YOLOX), лиц (YuNet), распознавание лиц (SFace) и трекер."""
import json
import os
import sqlite3
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

try:
    import items as itemlib
except Exception as _e:  # onnxruntime/tokenizers не установлены — только базовые функции
    print("items module disabled:", _e)
    itemlib = None

BASE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(BASE, "models")
DATA = os.path.join(BASE, "data")
os.makedirs(DATA, exist_ok=True)

cv2.setNumThreads(2)
COSINE_THRESHOLD = 0.363  # порог из документации OpenCV SFace


# ───────────────────────── База людей ─────────────────────────
FIELDS = ["surname", "name", "patronymic", "birth_date", "phone", "email",
          "address", "workplace", "status", "notes"]


def full_name(p):
    return " ".join(x for x in (p.get("surname"), p.get("name"), p.get("patronymic")) if x).strip() or "Без имени"


class PeopleDB:
    def __init__(self, path=os.path.join(DATA, "people.db")):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS people("
            "id INTEGER PRIMARY KEY, name TEXT, info TEXT, photo BLOB)")
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS embeddings("
            "id INTEGER PRIMARY KEY, person_id INTEGER, vec BLOB)")
        have = {r[1] for r in self.conn.execute("PRAGMA table_info(people)")}
        for f in FIELDS:  # миграция старой базы
            if f not in have:
                self.conn.execute(f"ALTER TABLE people ADD COLUMN {f} TEXT DEFAULT ''")
        if "added_at" not in have:
            self.conn.execute("ALTER TABLE people ADD COLUMN added_at TEXT DEFAULT ''")
        self.conn.execute("UPDATE people SET notes=info WHERE (notes IS NULL OR notes='') AND info IS NOT NULL")
        self.conn.commit()
        self.reload()

    def reload(self):
        self.people = {}
        cols = ", ".join(["id"] + FIELDS + ["added_at"])
        for row in self.conn.execute(f"SELECT {cols} FROM people"):
            d = {k: (v or "") for k, v in zip(["id"] + FIELDS + ["added_at"], row)}
            d["full"] = full_name(d)
            self.people[d["id"]] = d
        self.vecs = []  # (person_id, ndarray)
        for pid, blob in self.conn.execute("SELECT person_id,vec FROM embeddings"):
            self.vecs.append((pid, np.frombuffer(blob, dtype=np.float32).reshape(1, -1)))

    def add_person(self, data, feature, photo_jpg=None):
        vals = [data.get(f, "") for f in FIELDS]
        q = ",".join("?" * (len(FIELDS) + 2))
        cur = self.conn.execute(
            f"INSERT INTO people({','.join(FIELDS)},photo,added_at) VALUES({q})",
            vals + [photo_jpg, time.strftime("%d.%m.%Y %H:%M")])
        pid = cur.lastrowid
        if feature is not None:
            self.add_embedding(pid, feature, commit=False)
        self.conn.commit()
        self.reload()
        return pid

    def add_embedding(self, pid, feature, commit=True):
        self.conn.execute("INSERT INTO embeddings(person_id,vec) VALUES(?,?)",
                          (pid, feature.astype(np.float32).tobytes()))
        if commit:
            self.conn.commit()
            self.reload()

    def update_person(self, pid, data):
        sets = ",".join(f"{f}=?" for f in FIELDS)
        self.conn.execute(f"UPDATE people SET {sets} WHERE id=?", [data.get(f, "") for f in FIELDS] + [pid])
        self.conn.commit()
        self.reload()

    def delete_person(self, pid):
        self.conn.execute("DELETE FROM people WHERE id=?", (pid,))
        self.conn.execute("DELETE FROM embeddings WHERE person_id=?", (pid,))
        self.conn.commit()
        self.reload()

    def photo(self, pid):
        r = self.conn.execute("SELECT photo FROM people WHERE id=?", (pid,)).fetchone()
        return r[0] if r else None

    def search(self, query):
        """Поиск по всем полям; несколько слов — все должны встретиться."""
        words = query.lower().split()
        res = []
        for pid, p in self.people.items():
            hay = " ".join(str(p[f]) for f in FIELDS).lower()
            if all(w in hay for w in words):
                res.append(pid)
        return sorted(res, key=lambda i: self.people[i]["full"].lower())


# ───────────────────────── Детекторы ─────────────────────────
# COCO-классы, которые показываем как предметы: id → (название, категория, порог уверенности)
OBJ_CLASSES = {
    67: ("ТЕЛЕФОН", "phone", 0.35),
    39: ("БУТЫЛКА", "drink", 0.35), 40: ("БОКАЛ", "drink", 0.35), 41: ("СТАКАН / ЧАШКА", "drink", 0.35),
    63: ("НОУТБУК", "tech", 0.45), 62: ("ТЕЛЕВИЗОР", "tech", 0.5), 65: ("ПУЛЬТ", "tech", 0.45),
    64: ("МЫШЬ", "tech", 0.45), 66: ("КЛАВИАТУРА", "tech", 0.45),
    24: ("РЮКЗАК", "bag", 0.45), 26: ("СУМКА", "bag", 0.45), 28: ("ЧЕМОДАН", "bag", 0.5), 25: ("ЗОНТ", "bag", 0.45),
    73: ("КНИГА", "other", 0.45), 43: ("НОЖ", "other", 0.45), 76: ("НОЖНИЦЫ", "other", 0.45),
    38: ("РАКЕТКА", "other", 0.4), 32: ("МЯЧ", "other", 0.4),
    45: ("ТАРЕЛКА", "food", 0.5), 46: ("БАНАН", "food", 0.45), 47: ("ЯБЛОКО", "food", 0.45),
    48: ("БУТЕРБРОД", "food", 0.45), 49: ("АПЕЛЬСИН", "food", 0.45), 53: ("ПИЦЦА", "food", 0.45),
    54: ("ПОНЧИК", "food", 0.45), 55: ("ТОРТ", "food", 0.45),
}


@dataclass
class Obj:
    oid: int
    cls: int
    name: str
    cat: str
    box: object
    score: float
    hits: int = 1
    last_seen: float = 0.0
    owner: object = None       # tid человека
    hand: object = None        # "ЛЕВАЯ"/"ПРАВАЯ", если предмет в руке
    near_face: bool = False
    clip: object = None        # (ключ предмета, вероятность) по уточнению CLIP


class ObjectTracker:
    """Склеивает детекции предметов между кадрами и сглаживает рамки."""

    def __init__(self):
        self.objs: dict[int, Obj] = {}
        self.next_id = 1

    def reset(self):
        self.objs.clear()
        self.next_id = 1

    def update(self, dets, ts):
        pairs = sorted(((iou(o.box, d[2]), oid, di) for oid, o in self.objs.items()
                        for di, d in enumerate(dets) if o.cls == d[0]), reverse=True)
        used_o, used_d = set(), set()
        for sc, oid, di in pairs:
            if sc < 0.3 or oid in used_o or di in used_d:
                continue
            o, d = self.objs[oid], dets[di]
            o.box = tuple(0.5 * np.array(o.box) + 0.5 * np.array(d[2]))
            o.score = 0.5 * o.score + 0.5 * d[1]
            o.hits += 1
            o.last_seen = ts
            used_o.add(oid)
            used_d.add(di)
        for di, d in enumerate(dets):
            if di in used_d:
                continue
            name, cat, _ = OBJ_CLASSES[d[0]]
            self.objs[self.next_id] = Obj(self.next_id, d[0], name, cat, tuple(float(v) for v in d[2]), d[1], 1, ts)
            self.next_id += 1
        for oid in [k for k, o in self.objs.items() if ts - o.last_seen > 0.8]:
            del self.objs[oid]
        return [o for o in self.objs.values() if ts - o.last_seen < 0.45 and (o.hits >= 2 or o.score > 0.6)]


class OnnxModel:
    """ONNX Runtime (CoreML на Apple Silicon, иначе CPU); если его нет — запасной путь через OpenCV DNN."""

    def __init__(self, path, coreml=False):
        self.sess = self.net = None
        try:
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.log_severity_level = 3
            so.intra_op_num_threads = 2          # несколько нейросетей работают одновременно — не даём им отбирать ядра
            so.add_session_config_entry("session.intra_op.allow_spinning", "0")
            attempts = []
            if coreml:
                attempts.append([("CoreMLExecutionProvider", {"ModelFormat": "MLProgram", "MLComputeUnits": "ALL"}),
                                 "CPUExecutionProvider"])
            attempts.append(["CPUExecutionProvider"])
            for provs in attempts:
                try:
                    self.sess = ort.InferenceSession(path, so, providers=provs)
                    self.inp = self.sess.get_inputs()[0].name
                    break
                except Exception:
                    self.sess = None
        except Exception:
            self.sess = None
        if self.sess is None:
            self.net = cv2.dnn.readNet(path)
            self.names = self.net.getUnconnectedOutLayersNames()

    @property
    def backend(self):
        return ("ORT/" + self.sess.get_providers()[0].replace("ExecutionProvider", "")) if self.sess else "OpenCV"

    def run(self, blob):
        blob = np.ascontiguousarray(blob, dtype=np.float32)
        if self.sess is not None:
            return self.sess.run(None, {self.inp: blob})
        self.net.setInput(blob)
        out = self.net.forward(self.names)
        return list(out) if isinstance(out, (list, tuple)) else [out]


class PersonDetector:
    """YOLOX-S (OpenCV DNN): ищем класс 'person' (туловище целиком)."""
    SIZE = 640

    def __init__(self):
        self.model = OnnxModel(os.path.join(MODELS, "yolox.onnx"), coreml=True)
        strides = [8, 16, 32]
        grids, ss = [], []
        for s in strides:
            n = self.SIZE // s
            xv, yv = np.meshgrid(np.arange(n), np.arange(n))
            grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
            ss.append(np.full((n * n, 1), s))
        self.grids = np.concatenate(grids, 0).astype(np.float32)
        self.strides = np.concatenate(ss, 0).astype(np.float32)

    def detect(self, frame, conf=0.45, nms=0.5):
        return self.detect_all(frame, conf, nms)[0]

    def detect_all(self, frame, conf=0.45, nms=0.5):
        """Один проход сети: люди (класс 0) и предметы из OBJ_CLASSES. Возвращает (люди, предметы)."""
        h, w = frame.shape[:2]
        r = min(self.SIZE / h, self.SIZE / w)
        nh, nw = int(h * r), int(w * r)
        canvas = np.full((self.SIZE, self.SIZE, 3), 114, np.uint8)
        canvas[:nh, :nw] = cv2.resize(frame, (nw, nh))
        blob = cv2.dnn.blobFromImage(canvas, 1.0, (self.SIZE, self.SIZE), swapRB=False)
        out = self.model.run(blob)[0][0]
        xy = (out[:, :2] + self.grids) * self.strides
        wh = np.exp(out[:, 2:4]) * self.strides
        cls_scores = out[:, 4:5] * out[:, 5:]  # objectness * вероятность класса
        boxes_all = np.concatenate([xy - wh / 2, wh], 1) / r

        def clip(b):
            x, y, bw, bh = b
            return (max(0, int(x)), max(0, int(y)), min(w - 1, int(x + bw)), min(h - 1, int(y + bh)))

        people = []
        score = cls_scores[:, 0]
        keep = score > conf
        if keep.any():
            bx, sc = boxes_all[keep], score[keep]
            for i in np.array(cv2.dnn.NMSBoxes(bx.tolist(), sc.tolist(), conf, nms)).flatten():
                x1, y1, x2, y2 = clip(bx[i])
                if x2 - x1 > 20 and y2 - y1 > 40:
                    people.append((x1, y1, x2, y2))

        objs = []
        ids = list(OBJ_CLASSES)
        sub = cls_scores[:, ids]
        best = sub.argmax(1)
        bs = sub[np.arange(len(sub)), best]
        keep = bs > 0.30
        if keep.any():
            bx, sc, cl = boxes_all[keep], bs[keep], best[keep]
            idx = cv2.dnn.NMSBoxesBatched(bx.tolist(), sc.tolist(), cl.tolist(), 0.30, 0.5)
            for i in np.array(idx).flatten():
                cid = ids[int(cl[i])]
                x1, y1, x2, y2 = clip(bx[i])
                if x2 - x1 >= 8 and y2 - y1 >= 8 and sc[i] >= OBJ_CLASSES[cid][2]:
                    objs.append((cid, float(sc[i]), (x1, y1, x2, y2)))
        return people, objs


class FaceEngine:
    """YuNet — поиск лиц, SFace — вектор лица для опознания."""

    def __init__(self):
        self.det = cv2.FaceDetectorYN.create(os.path.join(MODELS, "yunet.onnx"), "", (320, 320), 0.7, 0.3, 5000)
        self.rec = cv2.FaceRecognizerSF.create(os.path.join(MODELS, "sface.onnx"), "")

    def detect(self, frame, max_side=800):
        """Лица ищем на уменьшенной копии (в 2–3 раза быстрее), координаты возвращаем в полном кадре."""
        h, w = frame.shape[:2]
        s = min(1.0, max_side / max(h, w))
        small = cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1.0 else frame
        self.det.setInputSize((small.shape[1], small.shape[0]))
        _, faces = self.det.detect(small)
        if faces is None:
            return []
        faces = faces.copy()
        faces[:, :14] /= s  # рамка и 5 точек лица
        return list(faces)

    def feature(self, frame, face):
        aligned = self.rec.alignCrop(frame, face)
        return self.rec.feature(aligned)

    def match(self, feat, db: PeopleDB):
        best, best_pid = 0.0, None
        for pid, vec in db.vecs:
            s = self.rec.match(feat, vec, cv2.FaceRecognizerSF_FR_COSINE)
            if s > best:
                best, best_pid = s, pid
        return (best_pid, best) if best >= COSINE_THRESHOLD else (None, best)


# ───────────────────────── Поза тела ─────────────────────────
# 33 точки BlazePose: 0 нос, 1-6 глаза, 7-8 уши, 9-10 рот, 11-12 плечи, 13-14 локти, 15-16 кисти,
# 17-22 мизинец/указательный/большой пальцы, 23-24 бёдра, 25-26 колени, 27-28 лодыжки, 29-32 пятки и носки
_LR = list(range(33))
for _a, _b in [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16), (17, 18),
               (19, 20), (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)]:
    _LR[_a], _LR[_b] = _b, _a


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def describe_pose(pts, vis, thr=0.5):
    """По точкам тела определяет позу, жесты и какие части тела видны. Стороны — анатомические."""
    v = lambda i: bool(vis[i] > thr)
    P = lambda i: pts[i]
    dist = lambda a, b: float(np.hypot(*(P(a) - P(b))))
    head = v(0) or (v(2) and v(5)) or (v(7) and v(8))
    sh, hips = v(11) and v(12), v(23) and v(24)
    arms = sum(1 for s, e, w in ((11, 13, 15), (12, 14, 16)) if v(s) and v(e) and v(w))
    legs = sum(1 for h, k, a in ((23, 25, 27), (24, 26, 28)) if v(h) and v(k) and v(a))
    hands = sum(1 for w in (15, 16) if v(w))
    feet = sum(1 for a in (27, 28) if v(a))
    mark = lambda ok: "✓" if ok else "✗"
    torso = "✓" if (sh and hips) else ("½" if sh or hips else "✗")
    scan = [f"ГОЛОВА {mark(head)}  ТУЛОВИЩЕ {torso}",
            f"РУКИ {arms}/2  КИСТИ {hands}/2  НОГИ {legs}/2  СТОПЫ {feet}/2"]

    posture = None
    if sh and hips:
        d = (P(23) + P(24)) / 2 - (P(11) + P(12)) / 2
        tl = float(np.hypot(*d)) + 1e-6
        ang = float(np.degrees(np.arctan2(abs(d[0]), abs(d[1]) + 1e-6)))  # отклонение торса от вертикали
        if ang > 65:
            posture = "ЛЕЖИТ"
        elif ang > 35:
            posture = "НАКЛОНЁН"
        else:
            ext = [dist(h, a) / tl for h, a in ((23, 27), (24, 28)) if v(h) and v(a)]
            thigh = [(P(k)[1] - P(h)[1]) / tl for h, k in ((23, 25), (24, 26)) if v(h) and v(k)]
            if ext and max(ext) > 1.5:      # нога вытянута почти на полную длину — человек стоит (даже в выпаде)
                posture = "СТОИТ"
            elif thigh:                      # бёдра «короткие» (смотрят в камеру или горизонтальны) — сидит
                posture = "СИДИТ" if max(thigh) < 0.55 else "СТОИТ"
    if posture is None and sh and not hips:
        posture = "ВЕРХ ТЕЛА"

    gestures = []
    if sh:
        sw = max(dist(11, 12), 1.0)
        mid_x = (P(11)[0] + P(12)[0]) / 2
        up, out = [], []
        for name, (s, w) in (("ЛЕВАЯ", (11, 15)), ("ПРАВАЯ", (12, 16))):
            if not v(w):
                continue
            dy = P(s)[1] - P(w)[1]  # > 0: кисть выше плеча
            if dy > 0.6 * sw:
                up.append(name)
            elif abs(P(w)[0] - mid_x) > abs(P(s)[0] - mid_x) + 1.0 * sw and abs(dy) < 0.6 * sw:
                out.append(name)
        if len(up) == 2:
            gestures.append("ОБЕ РУКИ ПОДНЯТЫ")
        else:
            gestures += [f"{n} РУКА ПОДНЯТА" for n in up]
        if len(out) == 2:
            gestures.append("РУКИ В СТОРОНЫ")
        else:
            gestures += [f"{n} РУКА В СТОРОНУ" for n in out]
        if all(v(i) for i in (13, 14, 15, 16)) and dist(15, 14) < 0.7 * sw and dist(16, 13) < 0.7 * sw:
            gestures.append("РУКИ СКРЕЩЕНЫ")
        elif head and v(0):
            for name, w in (("ЛЕВАЯ", 15), ("ПРАВАЯ", 16)):
                if v(w) and dist(w, 0) < 0.7 * sw and name not in up:
                    gestures.append(f"{name} РУКА У ЛИЦА")
    text = ", ".join(([posture] if posture else []) + gestures) or "НЕЙТРАЛЬНАЯ"
    return {"posture": posture, "gestures": gestures, "text": text, "scan": scan,
            "arms": arms, "legs": legs, "head": head, "hands": hands, "feet": feet,
            "torso": 2 if (sh and hips) else (1 if (sh or hips) else 0)}


class PoseEngine:
    """MediaPipe Pose (33 точки + маска силуэта). Кроп вокруг человека, повёрнутый по оси бёдра→голова."""
    SIZE = 256

    def __init__(self):
        self.model = OnnxModel(os.path.join(MODELS, "pose.onnx"))

    @staticmethod
    def roi_from_box(box, face=None, frame_h=None):
        """Первое приближение: где центр бёдер и какого размера круг, охватывающий всё тело.
        Если человек обрезан краем кадра (веб-камера, сидит за столом), рост оцениваем по размеру головы."""
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        head_h = None
        if face is not None:
            head_h = 1.35 * float(face[3])          # голова ≈ 1.35 высоты рамки лица
        cut = frame_h is not None and y2 >= frame_h - 8
        if face is not None and (cut or head_h > 0.3 * h):
            eye_y = (float(face[5]) + float(face[7])) / 2
            hip = np.array([float(face[0]) + float(face[2]) / 2, eye_y + 2.9 * head_h], np.float32)
            return hip, hip + np.array([0, -4.0 * head_h], np.float32)
        if cut and h < 2.0 * w:                     # тело обрезано, лица нет — масштаб по ширине плеч
            head_h = w / 2.3
            hip = np.array([(x1 + x2) / 2, y1 + 3.4 * head_h], np.float32)
            return hip, hip + np.array([0, -4.0 * head_h], np.float32)
        hip = np.array([(x1 + x2) / 2, y1 + 0.5 * h], np.float32)
        r = 0.6 * max(h, w)
        return hip, hip + np.array([0, -r], np.float32)

    def _run(self, frame, hip, spt, enlarge=1.0):
        r = float(np.linalg.norm(spt - hip)) * enlarge
        fh, fw = frame.shape[:2]
        if not np.isfinite(r) or r < 10 or r > 3 * max(fh, fw) or not np.isfinite(hip).all():
            return None
        rad = np.pi / 2 - np.arctan2(-(spt[1] - hip[1]), spt[0] - hip[0])
        angle = float(np.rad2deg(rad - 2 * np.pi * np.floor((rad + np.pi) / (2 * np.pi))))
        M = cv2.getRotationMatrix2D((float(hip[0]), float(hip[1])), angle, self.SIZE / (2 * r))
        M[:, 2] += (self.SIZE / 2 - hip[0], self.SIZE / 2 - hip[1])
        crop = cv2.warpAffine(frame, M, (self.SIZE, self.SIZE), flags=cv2.INTER_LINEAR)
        blob = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        lm, conf, mask, _, _ = self.model.run(blob[None])
        lm = lm[0].reshape(-1, 5)
        Minv = cv2.invertAffineTransform(M)
        pts = (lm[:, :2] @ Minv[:, :2].T + Minv[:, 2]).astype(np.float32)
        prob = _sigmoid(mask[0, :, :, 0])
        contours = []
        cnts, _ = cv2.findContours((prob > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            areas = [cv2.contourArea(c) for c in cnts]
            top = max(areas)
            for c, a in zip(cnts, areas):
                if a > 120 and a > 0.3 * top:
                    c = cv2.approxPolyDP(c, 1.2, True).reshape(-1, 2).astype(np.float32)
                    contours.append(c @ Minv[:, :2].T + Minv[:, 2])
        return {"pts": pts, "vis": _sigmoid(lm[:, 3]), "conf": float(conf[0][0]), "contours": contours}

    def estimate(self, frame, t, mirrored=False):
        res = None
        if t.pose_roi is not None:  # режим слежения: область из прошлого кадра
            res = self._run(frame, *t.pose_roi, enlarge=1.25)
        if res is None or res["conf"] < 0.5:
            hip, spt = self.roi_from_box(t.box, t.face, frame.shape[0])
            res = self._run(frame, hip, spt)
            if res is not None and res["conf"] >= 0.5:  # уточняем область по служебным точкам и повторяем
                res2 = self._run(frame, res["pts"][33], res["pts"][34], enlarge=1.25)
                if res2 is not None and res2["conf"] >= res["conf"] - 0.05:
                    res = res2
        if res is None or res["conf"] < 0.5:
            t.pose_roi, t.pose_pts = None, None
            return None
        # область следующего кадра не должна уезжать от рамки человека
        x1, y1, x2, y2 = t.box
        mx, my = 0.35 * (x2 - x1), 0.35 * (y2 - y1)
        hx, hy = res["pts"][33]
        t.pose_roi = (res["pts"][33].copy(), res["pts"][34].copy()) \
            if (x1 - mx <= hx <= x2 + mx and y1 - my <= hy <= y2 + my) else None
        pts, vis = res["pts"][:33], res["vis"][:33].copy()
        fh, fw = frame.shape[:2]
        vis[(pts[:, 0] < -2) | (pts[:, 0] > fw + 2) | (pts[:, 1] < -2) | (pts[:, 1] > fh + 2)] = 0  # за кадром
        if mirrored:  # кадр зеркальный — меняем левую и правую стороны местами
            pts, vis = pts[_LR], vis[_LR]
        if t.pose_pts is not None:  # лёгкое сглаживание, чтобы скелет не дрожал
            a = np.where(t.pose_vis > 0.5, 0.65, 1.0)[:, None]
            pts = a * pts + (1 - a) * t.pose_pts
        t.pose_pts, t.pose_vis = pts.astype(np.float32), vis
        res.update(pts=t.pose_pts, vis=vis, info=describe_pose(t.pose_pts, vis))
        return res


# ───────────────────────── Кисти рук ─────────────────────────
# 21 точка MediaPipe Hand: 0 запястье; большой 1-4; указательный 5-8; средний 9-12; безымянный 13-16; мизинец 17-20
HAND_LINKS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
              (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]
HAND_TIPS = (4, 8, 12, 16, 20)
POSE_HAND = {"L": (15, 17, 19, 21, 13), "R": (16, 18, 20, 22, 14)}  # запястье, мизинец, указательный, большой, локоть


def describe_hand(h):
    """Жест по 21 точке: кулак, открытая ладонь, указывает, победа, щепоть."""
    P = h
    d = lambda a, b: float(np.hypot(*(P[a] - P[b])))
    palm = d(0, 9) + 1e-6
    ext = [d(0, t) > d(0, t - 2) * 1.12 and d(0, t) > palm * 1.25 for t in (8, 12, 16, 20)]
    thumb_out = d(4, 5) > palm * 0.55
    pinch = d(4, 8) < palm * 0.38
    n = sum(ext)
    if pinch and n >= 2:
        return "ОК"
    if pinch:
        return "ЩЕПОТЬ"
    if n == 4:
        return "ОТКРЫТАЯ ЛАДОНЬ"
    if n == 0:
        if not thumb_out:
            return "КУЛАК"
        return "ПАЛЕЦ ВВЕРХ" if P[4][1] < P[0][1] - palm * 0.8 else "ПАЛЕЦ В СТОРОНУ"
    if ext == [True, False, False, False]:
        return "УКАЗЫВАЕТ"
    if ext == [True, True, False, False]:
        return "ПОБЕДА"
    return None


class HandEngine:
    """MediaPipe Hand: 21 точка на кисть. Область берём от запястья из позы, дальше следим по своим же точкам."""
    SIZE = 224

    def __init__(self):
        self.model = OnnxModel(os.path.join(MODELS, "hand.onnx"))

    def _run(self, frame, wrist, d, length):
        """wrist — запястье, d — единичный вектор к пальцам, length — оценка длины кисти (запястье → кончик)."""
        fh, fw = frame.shape[:2]
        if not np.isfinite(length) or length < 12 or length > 1.2 * max(fh, fw) or not np.isfinite(wrist).all():
            return None
        c = wrist + d * 0.6 * length
        side = 1.65 * length
        rad = np.pi / 2 - np.arctan2(-d[1], d[0])
        angle = float(np.rad2deg(rad - 2 * np.pi * np.floor((rad + np.pi) / (2 * np.pi))))
        M = cv2.getRotationMatrix2D((float(c[0]), float(c[1])), angle, self.SIZE / side)
        M[:, 2] += (self.SIZE / 2 - c[0], self.SIZE / 2 - c[1])
        crop = cv2.warpAffine(frame, M, (self.SIZE, self.SIZE), flags=cv2.INTER_LINEAR)
        blob = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        lm, conf, _, _ = self.model.run(blob[None])
        Minv = cv2.invertAffineTransform(M)
        pts = (lm[0].reshape(21, 3)[:, :2] @ Minv[:, :2].T + Minv[:, 2]).astype(np.float32)
        return {"pts": pts, "conf": float(conf[0][0])}

    @staticmethod
    def seed_from_pose(pts, vis, slot):
        """Приблизительное положение кисти по точкам позы: (запястье, направление, длина) или None."""
        wr, pk, ix, th, el = POSE_HAND[slot]
        if vis[wr] < 0.5:
            return None
        W = pts[wr].astype(np.float64)
        fingers = [pts[k] for k in (pk, ix) if vis[k] > 0.4]
        fore = float(np.linalg.norm(W - pts[el])) if vis[el] > 0.5 else None
        if fingers:
            mid = np.mean(fingers, 0)
            hl = float(np.linalg.norm(mid - W))
            d = (mid - W) / (hl + 1e-6)
            length = 3.3 * hl
        elif fore:
            d = (W - pts[el]) / (fore + 1e-6)
            length = 0.7 * fore
        else:
            return None
        if fore:
            length = float(np.clip(length, 0.45 * fore, 0.95 * fore))
        return W, d, max(length, 16.0)

    def estimate(self, frame, t, mirrored=False):
        """Возвращает {"L": {...}, "R": {...}} для кистей, которые удалось найти."""
        out = {}
        if t.pose is None:
            t.hand_roi = {}
            return out
        pts, vis = t.pose["pts"], t.pose["vis"]
        for slot in ("L", "R"):
            seed = self.seed_from_pose(pts, vis, slot)
            if seed is None:
                t.hand_roi.pop(slot, None)
                continue
            W0, d0, L0 = seed
            res = None
            roi = t.hand_roi.get(slot)  # режим слежения: область по точкам прошлого кадра
            if roi is not None and np.linalg.norm(roi[0] - W0) < 1.3 * roi[2]:
                res = self._run(frame, *roi)
            if res is None or res["conf"] < 0.55:
                for k in (1.0, 1.35, 0.75):
                    res = self._run(frame, W0, d0, L0 * k)
                    if res is not None and res["conf"] >= 0.55:
                        break
            if res is None or res["conf"] < 0.55:
                t.hand_roi.pop(slot, None)
                continue
            h = res["pts"]
            prev = t.hand_pts.get(slot)
            if prev is not None:  # лёгкое сглаживание
                h = 0.6 * h + 0.4 * prev
            t.hand_pts[slot] = h
            wrist = h[0].astype(np.float64)
            direction = h[9] - h[0]
            ln = float(np.linalg.norm(direction)) + 1e-6
            length = float(max(np.linalg.norm(h[12] - h[0]), np.linalg.norm(h[8] - h[0]), 1.8 * ln))
            t.hand_roi[slot] = (wrist, direction / ln, length * 1.05)
            out[slot] = {"pts": h, "conf": res["conf"], "gesture": describe_hand(h)}
        for slot in [k for k in t.hand_pts if k not in out]:
            t.hand_pts.pop(slot, None)
        return out


def itemlib_find(catalog, key):
    it = catalog.find(key)
    return (it["ru"], it["cat"]) if it else (key, "other")


# ───────────────────────── Трекер ─────────────────────────
@dataclass
class Track:
    tid: int
    box: tuple
    face: object = None            # строка YuNet или None
    feat: object = None            # последний вектор лица
    person_id: object = None       # опознанный человек из базы
    votes: dict = field(default_factory=dict)
    last_seen: float = 0.0
    last_rec: float = 0.0
    face_only: bool = False        # рамка построена по лицу, тела в кадре нет
    pose: object = None            # результат PoseEngine
    pose_roi: object = None        # (центр бёдер, точка масштаба) для слежения
    pose_pts: object = None
    pose_vis: object = None
    hands: dict = field(default_factory=dict)     # "L"/"R" -> {"pts": (21,2), "conf", "gesture"}
    hand_roi: dict = field(default_factory=dict)  # область слежения за кистью
    hand_pts: dict = field(default_factory=dict)  # сглаженные точки кисти
    objects: list = field(default_factory=list)   # предметы рядом / в руках (Obj)
    hold: dict = field(default_factory=dict)      # "L"/"R" -> (ключ предмета, вероятность) от CLIP
    act: object = None                            # (ключ действия, вероятность): курит, пьёт, говорит по телефону
    scores: dict = field(default_factory=dict)    # лучшие гипотезы по рукам (для отладки)


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0


class Analyzer:
    """Полный конвейер: кадр → список треков (тело + лицо + опознанная личность)."""

    def __init__(self, db: PeopleDB):
        self.db = db
        self.persons = PersonDetector()
        self.faces = FaceEngine()
        self.pose = PoseEngine() if os.path.exists(os.path.join(MODELS, "pose.onnx")) else None
        self.hand = HandEngine() if self.pose is not None and os.path.exists(os.path.join(MODELS, "hand.onnx")) else None
        self.use_pose = True
        self.use_objects = True      # предметы YOLOX (телефон, напитки…)
        self.use_scan = True         # CLIP: что в руках, курит/пьёт
        self.objtrack = ObjectTracker()
        self.catalog = self.scanner = None
        if itemlib is not None and all(os.path.exists(os.path.join(MODELS, f))
                                       for f in ("clip_vision.onnx", "clip_text.onnx", "clip_tokenizer.json")):
            try:
                self.catalog = itemlib.Catalog()
                self.scanner = itemlib.ScanWorker(self.catalog)
            except Exception as e:
                print("CLIP scanner disabled:", e)
                self.catalog = self.scanner = None
        self.hold_thr, self.act_thr = 0.55, 0.75
        self._last_scan = 0.0
        self.objects_visible = []
        self.mirrored = False  # камера отдаёт зеркальный кадр
        self.tracks: dict[int, Track] = {}
        self.next_id = 1
        self.frame_no = 0

    def reset(self):
        self.tracks.clear()
        self.next_id = 1
        self.objtrack.reset()
        if self.scanner:
            self.scanner.forget(set())

    def process(self, frame, ts=None):
        ts = ts if ts is not None else time.time()
        self.frame_no += 1
        pboxes, odets = self.persons.detect_all(frame)
        faces = self.faces.detect(frame)

        # 1. Сопоставление тел с существующими треками по IoU
        unmatched = list(range(len(pboxes)))
        used = set()
        pairs = sorted(((iou(t.box, b), tid, bi) for tid, t in self.tracks.items()
                        for bi, b in enumerate(pboxes)), reverse=True)
        for score, tid, bi in pairs:
            if score < 0.25 or tid in used or bi not in unmatched:
                continue
            self.tracks[tid].box = pboxes[bi]
            self.tracks[tid].face_only = False
            self.tracks[tid].last_seen = ts
            used.add(tid)
            unmatched.remove(bi)
        for bi in unmatched:
            t = Track(self.next_id, pboxes[bi], last_seen=ts)
            self.tracks[t.tid] = t
            self.next_id += 1
        # лица без тела (крупный план) — тоже трек
        for f in faces:
            fx, fy, fw, fh = f[:4]
            cx, cy = fx + fw / 2, fy + fh / 2
            if not any(t.box[0] <= cx <= t.box[2] and t.box[1] <= cy <= t.box[3]
                       for t in self.tracks.values() if t.last_seen == ts):
                box = (int(max(0, fx - fw * 0.3)), int(max(0, fy - fh * 0.3)),
                       int(fx + fw * 1.3), int(fy + fh * 1.3))
                best, best_tid = 0.25, None
                for tid, t in self.tracks.items():
                    if tid not in used and iou(t.box, box) > best:
                        best, best_tid = iou(t.box, box), tid
                if best_tid is not None:
                    self.tracks[best_tid].box = box
                    self.tracks[best_tid].face_only = True
                    self.tracks[best_tid].last_seen = ts
                    used.add(best_tid)
                else:
                    t = Track(self.next_id, box, last_seen=ts, face_only=True)
                    self.tracks[t.tid] = t
                    self.next_id += 1

        # 2. Удаляем пропавшие треки
        for tid in [k for k, t in self.tracks.items() if ts - t.last_seen > 1.5]:
            del self.tracks[tid]

        # 3. Привязываем лица к трекам и опознаём
        for t in self.tracks.values():
            if t.last_seen != ts:
                t.face = None
                continue
            best, bf = None, None
            for f in faces:
                cx, cy = f[0] + f[2] / 2, f[1] + f[3] / 2
                if t.box[0] <= cx <= t.box[2] and t.box[1] <= cy <= t.box[3]:
                    if bf is None or f[2] * f[3] > bf[2] * bf[3]:
                        bf = f
            t.face = bf
            if bf is not None and (ts - t.last_rec > 0.5):
                t.last_rec = ts
                t.feat = self.faces.feature(frame, bf)
                pid, _ = self.faces.match(t.feat, self.db)
                # голосование по последним результатам, чтобы не мигало
                t.votes[pid] = t.votes.get(pid, 0) + 1
                if sum(t.votes.values()) > 6:
                    for k in t.votes:
                        t.votes[k] *= 0.6
                top = max(t.votes, key=t.votes.get)
                t.person_id = top

        # 4. Поза тела: скелет, силуэт, жесты (для самых крупных людей в кадре)
        current = [t for t in self.tracks.values() if t.last_seen == ts]
        by_area = sorted(current, key=lambda t: (t.box[2] - t.box[0]) * (t.box[3] - t.box[1]), reverse=True)
        for i, t in enumerate(by_area):
            if self.pose is None or not self.use_pose or t.face_only or i >= 6:
                t.pose, t.pose_roi, t.pose_pts = None, None, None
                continue
            t.pose = self.pose.estimate(frame, t, self.mirrored)
            t.hands = {}
            if self.hand is not None and t.pose is not None:
                t.hands = self.hand.estimate(frame, t, self.mirrored)
                if t.hands:  # жесты кистей — в описание позы
                    info = dict(t.pose["info"])
                    gs = [f"{'Л' if sl == 'L' else 'П'}: {h['gesture']}" for sl, h in t.hands.items() if h["gesture"]]
                    if gs:
                        info["text"] = (", ".join(gs) if info["text"] == "НЕЙТРАЛЬНАЯ" else info["text"] + ", " + ", ".join(gs))
                    info["hands"] = max(info["hands"], len(t.hands))
                    t.pose = dict(t.pose, info=info)
        for t in by_area[6:]:
            t.hands = {}

        # 5. Предметы: YOLOX-детекции → привязка к людям и рукам; CLIP — что именно в руках
        objs = self.objtrack.update(odets if self.use_objects else [], ts)
        self._link_objects(objs, current)
        self.objects_visible = objs
        self._scan(frame, by_area)
        return current

    def _link_objects(self, objs, people):
        for t in people:
            t.objects = []
        for o in objs:
            o.owner, o.hand, o.near_face = None, None, False
            ox1, oy1, ox2, oy2 = o.box
            oa = max(1.0, (ox2 - ox1) * (oy2 - oy1))
            best, owner = 0.35, None
            for t in people:
                x1, y1, x2, y2 = t.box
                frac = max(0, min(ox2, x2) - max(ox1, x1)) * max(0, min(oy2, y2) - max(oy1, y1)) / oa
                if frac > best:
                    best, owner = frac, t
            if owner is None:
                continue
            o.owner = owner.tid
            size = max(ox2 - ox1, oy2 - oy1)
            m = 0.3 * size + 6
            if owner.pose is not None:
                pts, vis = owner.pose["pts"], owner.pose["vis"]
                for name, ids in (("ЛЕВАЯ", (15, 17, 19, 21)), ("ПРАВАЯ", (16, 18, 20, 22))):
                    if any(vis[i] > 0.4 and ox1 - m <= pts[i][0] <= ox2 + m and oy1 - m <= pts[i][1] <= oy2 + m
                           for i in ids):
                        o.hand = name
                        break
            if owner.face is not None:
                fx, fy, fw, fh = [float(v) for v in owner.face[:4]]
                o.near_face = (abs((ox1 + ox2) / 2 - (fx + fw / 2)) < 0.9 * fw + size / 2 and
                               abs((oy1 + oy2) / 2 - (fy + fh / 2)) < 0.9 * fh + size / 2)
            owner.objects.append(o)

    def _hand_crop_list(self, frame, t, slot):
        """Вырезки вокруг одной кисти: по точкам кисти, а если их нет — по приблизительной оценке из позы."""
        h = t.hands.get(slot)
        if h is not None:
            specs = itemlib.hand_crops(h["pts"])
        elif t.pose is not None:
            seed = HandEngine.seed_from_pose(t.pose["pts"], t.pose["vis"], slot)
            specs = itemlib.pose_hand_crops(seed) if seed is not None else []
        else:
            specs = []
        out = []
        for typ, c_, s_ in specs:
            crop = itemlib.cut(frame, c_, s_)
            if crop is not None:
                out.append((typ, crop))
        return out

    def _scan(self, frame, by_area):
        sc = self.scanner
        for t in by_area:
            t.hold, t.act, t.scores = {}, None, {}
        for o in self.objects_visible:
            o.clip = None
        if sc is None or not self.use_scan:
            return
        now = time.time()
        learning = sc.learn is not None
        if sc.ready() and now - self._last_scan > (0.1 if learning else 0.2):
            self._last_scan = now
            jobs, meta = {}, {}
            for t in by_area[:3]:
                for slot in ("L", "R"):
                    cs = self._hand_crop_list(frame, t, slot)
                    if cs:
                        jobs[(t.tid, slot)] = cs
                if t.face is not None and not learning:
                    c_, s_ = itemlib.face_crop(t.face)
                    crop = itemlib.cut(frame, c_, s_)
                    if crop is not None:
                        jobs[(t.tid, "F")] = [("F", crop)]
                if learning and sc.learn["mode"] == "neg":
                    continue  # пока записываем пустые руки, предметы и лицо не считаем — быстрее
                for o in sorted(t.objects, key=lambda o: o.hand is None)[:2]:  # предметы в руках: что это на самом деле
                    crop = itemlib.cut(frame, *itemlib.object_crop(o.box))
                    if crop is not None:
                        key = (t.tid, f"O{o.oid}")
                        jobs[key] = [("O", crop)]
                        meta[key] = {"hand": {"ЛЕВАЯ": "L", "ПРАВАЯ": "R"}.get(o.hand)}
            sc.submit(jobs, meta)
        snap = sc.snapshot()
        sc.forget({t.tid for t in self.tracks.values()})
        for t in by_area:
            for slot in ("L", "R"):
                r = snap.get((t.tid, slot))
                if r:
                    top = sorted(r["probs"].items(), key=lambda kv: -kv[1])
                    t.scores[slot] = top[:3]
                    if top and top[0][1] >= self.hold_thr:
                        t.hold[slot] = top[0]
            r = snap.get((t.tid, "F"))
            if r:
                key, p = max(r["probs"].items(), key=lambda kv: kv[1])
                if p >= self.act_thr and self._act_supported(t, key, p):
                    t.act = (key, p)
        for o in self.objects_visible:  # CLIP уточняет, что за предмет нашёл YOLOX (пульт → айкос)
            r = snap.get((o.owner, f"O{o.oid}"))
            if r:
                key, p = max(r["probs"].items(), key=lambda kv: kv[1])
                if p >= self.hold_thr:
                    o.clip = (key, p)

    def _act_supported(self, t, key, p):
        """Действие «курит/пьёт/говорит по телефону» засчитываем, если рядом есть соответствующий предмет — или нейросеть очень уверена."""
        if p >= 0.9 or self.catalog is None:
            return True
        want = {"smoke": "smoke", "vape": "smoke", "drink": "drink", "call": "phone"}.get(key)
        cats = {itemlib_find(self.catalog, k)[1] for k, _ in t.hold.values()}
        cats |= {o.cat for o in t.objects if o.near_face or o.hand}
        cats |= {itemlib_find(self.catalog, o.clip[0])[1] for o in t.objects if o.clip}
        return want in cats
