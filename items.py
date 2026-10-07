"""Распознавание предметов в руках и действий (курит, пьёт, говорит по телефону) через CLIP.

Два режима:
  • без обучения — предметы описаны словами (в каталоге есть IQOS, стики, телефон, напитки…);
  • с обучением — пользователь показывает предмет в руке, программа записывает образцы вырезок вокруг кисти
    (теми же способами, какими потом распознаёт) и пустые руки для сравнения. По ним строится
    «прототипный» классификатор: предмет в руке или нет.
"""
import json
import os
import threading
import time

import cv2
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(BASE, "models")
DATA = os.path.join(BASE, "data")
os.makedirs(DATA, exist_ok=True)

# категории предметов → название и цвет на экране (RGB)
CATEGORIES = {
    "smoke": ("КУРЕНИЕ", (255, 176, 0)),
    "phone": ("ТЕЛЕФОН", (0, 229, 255)),
    "drink": ("НАПИТОК", (90, 160, 255)),
    "tech": ("ТЕХНИКА", (190, 160, 255)),
    "bag": ("ВЕЩИ", (150, 200, 170)),
    "food": ("ЕДА", (255, 140, 200)),
    "other": ("ПРЕДМЕТ", (200, 220, 210)),
}

DEFAULT_ITEMS = [
    {"key": "iqos", "ru": "АЙКОС", "cat": "smoke", "enabled": True,
     "en": ["an IQOS heated tobacco device", "a white IQOS holder pen", "an electronic heat-not-burn tobacco device"]},
    {"key": "stick", "ru": "СТИК ДЛЯ АЙКОС", "cat": "smoke", "enabled": True,
     "en": ["a heated tobacco stick for IQOS", "a TEREA tobacco stick", "a heets tobacco stick"]},
    {"key": "pack", "ru": "ПАЧКА СТИКОВ", "cat": "smoke", "enabled": True,
     "en": ["a pack of TEREA heated tobacco sticks", "a pack of heets for IQOS", "a box of heated tobacco sticks"]},
    {"key": "cigarette", "ru": "СИГАРЕТА", "cat": "smoke", "enabled": True,
     "en": ["a cigarette", "a lit cigarette between fingers", "a pack of cigarettes"]},
    {"key": "vape", "ru": "ВЕЙП", "cat": "smoke", "enabled": True,
     "en": ["a vape pen", "an electronic cigarette", "a disposable vape device"]},
    {"key": "lighter", "ru": "ЗАЖИГАЛКА", "cat": "smoke", "enabled": True,
     "en": ["a lighter", "a cigarette lighter"]},
    {"key": "phone", "ru": "ТЕЛЕФОН", "cat": "phone", "enabled": True,
     "en": ["a smartphone", "a mobile phone held in a hand"]},
    {"key": "bottle", "ru": "БУТЫЛКА", "cat": "drink", "enabled": True,
     "en": ["a plastic bottle", "a bottle of water", "a bottle of soda"]},
    {"key": "can", "ru": "БАНКА НАПИТКА", "cat": "drink", "enabled": True,
     "en": ["a soda can", "an energy drink can", "a beer can"]},
    {"key": "cup", "ru": "ЧАШКА / СТАКАН", "cat": "drink", "enabled": True,
     "en": ["a coffee cup", "a paper cup", "a mug"]},
    {"key": "glass", "ru": "БОКАЛ", "cat": "drink", "enabled": True,
     "en": ["a glass of beer", "a wine glass", "a glass of water"]},
]
# «ничего»: отнимают вероятность, когда в руке нет интересующего предмета
NEGATIVES = ["an empty hand", "a bare hand with nothing in it", "a hand resting on a table", "fingers",
             "a sleeve of a jacket", "a pen", "a pencil", "a marker", "a microphone", "a remote control",
             "a toothbrush", "a flashlight", "a spoon", "a screwdriver", "a computer mouse", "a keyboard",
             "a piece of paper", "a book", "a wall", "clothes", "a face", "a watch on a wrist", "a ring on a finger"]

DEFAULT_ACTS = [
    {"key": "smoke", "ru": "КУРИТ", "cat": "smoke", "enabled": True,
     "en": ["a person smoking a cigarette", "a person with a cigarette in the mouth",
            "a person exhaling cigarette smoke"]},
    {"key": "vape", "ru": "ПАРИТ / АЙКОС", "cat": "smoke", "enabled": True,
     "en": ["a person vaping", "a person using an IQOS heated tobacco device", "a person exhaling vapor"]},
    {"key": "drink", "ru": "ПЬЁТ", "cat": "drink", "enabled": True,
     "en": ["a person drinking from a bottle", "a person drinking from a cup", "a person drinking from a glass"]},
    {"key": "call", "ru": "ГОВОРИТ ПО ТЕЛЕФОНУ", "cat": "phone", "enabled": True,
     "en": ["a person talking on a phone", "a person holding a phone to the ear"]},
]
ACT_NEGATIVES = ["a person with a neutral face", "a person looking at the camera", "a person talking",
                 "a close-up portrait of a face", "a person eating", "a person smiling",
                 "a person sitting at a desk looking at a computer", "a person with a hand near the face",
                 "a person with an open mouth", "a person yawning", "a webcam selfie of a person"]

TYPES = ("H1", "H2", "P", "O")      # виды вырезок: кисть, кисть с окружением, кончики пальцев, сам предмет
MIN_POS, MIN_NEG = 8, 8
NEG_KEY = "__neg__"


class Catalog:
    """Каталог предметов/действий + образцы, записанные пользователем. Хранится в data/."""

    def __init__(self):
        self.path = os.path.join(DATA, "items.json")
        self.ex_path = os.path.join(DATA, "item_examples.npz")
        self.items, self.acts = [dict(i) for i in DEFAULT_ITEMS], [dict(a) for a in DEFAULT_ACTS]
        self.examples = {}  # ключ предмета (или NEG_KEY) -> {вид вырезки: ndarray (n, 512)}
        self.version = 0
        self.load()

    def load(self):
        try:
            with open(self.path, encoding="utf-8") as f:
                d = json.load(f)
            self.items, self.acts = d["items"], d["acts"]
        except Exception:
            self.save()
        self.examples = {}
        try:
            z = np.load(self.ex_path)
            for name in z.files:
                if "__" not in name or z[name].ndim != 2 or z[name].shape[1] != 512:
                    continue  # старый формат (вырезки рамкой) — несовместим с новым распознаванием
                key, typ = name.rsplit("__", 1)
                self.examples.setdefault(key, {})[typ] = z[name]
        except Exception:
            self.examples = {}

    def save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({"items": self.items, "acts": self.acts}, f, ensure_ascii=False, indent=1)
        flat = {f"{k}__{t}": a for k, d in self.examples.items() for t, a in d.items() if len(a)}
        if flat:
            np.savez(self.ex_path, **flat)
        elif os.path.exists(self.ex_path):
            os.remove(self.ex_path)
        self.version += 1

    def find(self, key):
        return next((i for i in self.items if i["key"] == key), None)

    def add_item(self, ru, en_prompts, cat="other"):
        key = "u%d" % int(time.time() * 1000)
        self.items.append({"key": key, "ru": ru.upper(), "cat": cat, "enabled": True, "en": en_prompts})
        self.save()
        return key

    def add_examples(self, key, by_type, cap=90):
        """by_type: {вид вырезки: ndarray (n, 512)}; хранится не больше cap последних образцов каждого вида."""
        d = self.examples.setdefault(key, {})
        for typ, arr in by_type.items():
            arr = np.asarray(arr, np.float32).reshape(-1, 512)
            if len(arr):
                d[typ] = np.vstack([d[typ], arr])[-cap:] if typ in d else arr[-cap:]
        self.save()

    def clear_examples(self, key):
        self.examples.pop(key, None)
        self.save()

    def n_pos(self, key):
        d = self.examples.get(key, {})
        return max((len(a) for a in d.values()), default=0)

    def remove_item(self, key):
        self.items = [i for i in self.items if i["key"] != key]
        self.examples.pop(key, None)
        self.save()


class ClipEncoder:
    MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
    STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)

    def __init__(self):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        so = ort.SessionOptions()
        so.log_severity_level = 3
        so.intra_op_num_threads = 2
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self.vis = ort.InferenceSession(os.path.join(MODELS, "clip_vision.onnx"), so, providers=["CPUExecutionProvider"])
        self.txt = ort.InferenceSession(os.path.join(MODELS, "clip_text.onnx"), so, providers=["CPUExecutionProvider"])
        self.tok = Tokenizer.from_file(os.path.join(MODELS, "clip_tokenizer.json"))
        self.tok.enable_truncation(77)
        self.tok.enable_padding(length=77, pad_id=49407, pad_token="<|endoftext|>")
        self.vin = self.vis.get_inputs()[0].name
        self.tin = [i.name for i in self.txt.get_inputs()]

    @staticmethod
    def _norm(x):
        return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)

    def images(self, crops):
        """list of BGR crops → (n, 512) нормированные векторы."""
        if not crops:
            return np.zeros((0, 512), np.float32)
        batch = []
        for c in crops:
            c = cv2.resize(c, (224, 224), interpolation=cv2.INTER_AREA)
            c = (cv2.cvtColor(c, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 - self.MEAN) / self.STD
            batch.append(c.transpose(2, 0, 1))
        out = self.vis.run(None, {self.vin: np.stack(batch).astype(np.float32)})[0]
        return self._norm(out.astype(np.float32))

    def texts(self, prompts):
        enc = self.tok.encode_batch(prompts)
        ids = np.array([e.ids for e in enc], np.int64)
        feed = {"input_ids": ids}
        if "attention_mask" in self.tin:
            feed["attention_mask"] = np.array([e.attention_mask for e in enc], np.int64)
        out = self.txt.run(None, feed)[0]
        return self._norm(out.astype(np.float32))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class ItemRecognizer:
    """Узнаёт предметы: по описанию (без обучения) и по образцам пользователя (прототипы против «пустой руки»)."""

    def __init__(self, catalog: Catalog):
        self.cat = catalog
        self.enc = ClipEncoder()
        self._ver = -1
        self.groups = {}
        self.protos = {}
        self.rebuild()

    # ── подготовка ──
    def rebuild(self):
        def build(entries, negs):
            keys, rows = [], []
            for it in entries:
                if not it.get("enabled", True):
                    continue
                if it.get("en"):
                    v = self.enc.texts(it["en"]).mean(0)
                    v = v / np.linalg.norm(v)
                elif self.cat.n_pos(it["key"]):
                    v = np.zeros(512, np.float32)  # предмет без описания — узнаём только по образцам
                else:
                    continue
                rows.append(v)
                keys.append(it["key"])
            return keys, np.array(rows, np.float32).reshape(-1, 512), self.enc.texts(negs)
        self.groups = {"item": build(self.cat.items, NEGATIVES), "act": build(self.cat.acts, ACT_NEGATIVES)}
        self.protos = self._build_protos()
        self._ver = self.cat.version

    def _build_protos(self):
        protos = {}
        neg_all = self.cat.examples.get(NEG_KEY, {})
        for it in self.cat.items:
            k = it["key"]
            if not it.get("enabled", True):
                continue
            for typ, P in self.cat.examples.get(k, {}).items():
                if len(P) < MIN_POS:
                    continue
                pools = [neg_all.get(typ, np.zeros((0, 512), np.float32))]
                for k2, ex2 in self.cat.examples.items():  # другие предметы тоже «не этот предмет»
                    if k2 not in (k, NEG_KEY) and typ in ex2:
                        pools.append(ex2[typ][-15:])
                N = np.vstack(pools)
                if len(N) < MIN_NEG:
                    continue
                mp, mn = P.mean(0), N.mean(0)
                mp, mn = mp / np.linalg.norm(mp), mn / np.linalg.norm(mn)
                dp, dn = P @ mp - P @ mn, N @ mp - N @ mn
                gap = float(dp.mean() - dn.mean())
                if gap < 0.02:
                    continue  # предмет не отличается от пустой руки на этом виде вырезки
                sp, sn = float(dp.std()) + 1e-4, float(dn.std()) + 1e-4
                thr = float((dp.mean() * sn + dn.mean() * sp) / (sp + sn))
                protos[(k, typ)] = (mp, mn, thr, max(gap / 5.0, 0.004))
        return protos

    # ── распознавание ──
    def _zero_shot(self, e, group):
        keys, tv, nv = self.groups[group]
        logits = list(100.0 * (tv @ e)) if len(keys) else []
        neg = 100.0 * (nv @ e)
        allv = np.array(logits + [neg.max()], np.float64)
        p = np.exp(allv - allv.max())
        p /= p.sum()
        return {**{k: float(p[i]) for i, k in enumerate(keys)}, None: float(p[-1])}

    def classify_hand(self, embs):
        """embs: {вид вырезки: вектор}. Возвращает {ключ предмета: вероятность, None: «ничего»}."""
        if self._ver != self.cat.version:
            self.rebuild()
        zs = [self._zero_shot(e, "item") for e in embs.values()]
        probs = {k: float(np.mean([z[k] for z in zs])) for k in zs[0]} if zs else {None: 1.0}
        trained = {}
        for (k, typ), (mp, mn, thr, scale) in self.protos.items():
            if typ in embs:
                trained.setdefault(k, []).append(float(_sigmoid((embs[typ] @ mp - embs[typ] @ mn - thr) / scale)))
        for k, vals in trained.items():
            probs[k] = float(np.mean(vals))  # обученный предмет оцениваем только по образцам пользователя
        return probs

    def classify_act(self, e):
        if self._ver != self.cat.version:
            self.rebuild()
        return self._zero_shot(e, "act")


# ───────────── геометрия: откуда вырезать кропы ─────────────
def hand_crops(h):
    """Вырезки по 21 точке кисти: сама кисть, кисть с окружением, кончики пальцев (щепоть/стик)."""
    x1, y1 = h.min(0)
    x2, y2 = h.max(0)
    c = np.array([(x1 + x2) / 2, (y1 + y2) / 2], np.float32)
    s = float(max(x2 - x1, y2 - y1, 28.0))
    d = h[9] - h[0]
    d = d / (np.linalg.norm(d) + 1e-6)
    tips = h[[4, 8, 12]].mean(0) + d * 0.10 * s
    return [("H1", c, 1.5 * s), ("H2", c, 2.1 * s), ("P", tips, 0.95 * s)]


def pose_hand_crops(seed):
    """Запасной вариант без точек кисти: по оценке из позы (запястье, направление, длина)."""
    W, d, length = seed
    c = (W + d * 0.55 * length).astype(np.float32)
    return [("H1", c, 1.5 * length), ("H2", c, 2.1 * length)]


def face_crop(face):
    """Квадрат вокруг нижней части лица (рот), чтобы увидеть сигарету/бутылку/телефон у лица."""
    x, y, w, h = [float(v) for v in face[:4]]
    return (np.array([x + w / 2, y + h * 0.8], np.float32), 2.6 * max(w, h * 0.9))


def object_crop(box):
    x1, y1, x2, y2 = box
    return (np.array([(x1 + x2) / 2, (y1 + y2) / 2], np.float32), 1.3 * max(x2 - x1, y2 - y1, 40.0))


def cut(frame, center, side):
    """Квадратный кроп с нулевой подложкой за краем кадра."""
    side = int(max(64, side))
    x0, y0 = int(center[0] - side / 2), int(center[1] - side / 2)
    h, w = frame.shape[:2]
    if x0 >= w or y0 >= h or x0 + side <= 0 or y0 + side <= 0:
        return None
    sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x0 + side, w), min(y0 + side, h)
    canvas = np.zeros((side, side, 3), np.uint8)
    canvas[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = frame[sy0:sy1, sx0:sx1]
    return canvas


# ───────────── фоновый сканер ─────────────
class ScanWorker(threading.Thread):
    """Считает CLIP в отдельном потоке, копит сглаженные вероятности по (трек, слот) и записывает образцы обучения."""

    def __init__(self, catalog: Catalog):
        super().__init__(daemon=True)
        self.rec = ItemRecognizer(catalog)
        self.event = threading.Event()
        self.job = None
        self.busy = False
        self.lock = threading.Lock()
        self.results = {}    # (tid, слот) -> {"probs": {...}, "ts": float}
        self.learn = None    # состояние записи образцов
        self.alive = True
        self.start()

    def ready(self):
        return not self.busy and self.job is None

    def submit(self, crops, meta=None):
        """crops: {(tid, слот): [(вид, bgr), ...]}; meta: {(tid, слот): {"hand": "L"/"R"/None}}"""
        if crops:
            self.job = (crops, meta or {})
            self.event.set()

    # ── запись образцов ──
    def start_learn(self, mode, slot="any", seconds=6.0, cap=70):
        """mode: "neg" — пустые руки (обе), "pos" — предмет в руке. slot: "L"/"R"/"any"."""
        self.learn = {"mode": mode, "slot": slot, "t0": time.time(), "t_end": time.time() + seconds, "cap": cap,
                      "rows": [], "n": 0, "lock": None, "usum": {"L": 0.0, "R": 0.0}}

    def learn_progress(self):
        s = self.learn
        if not s:
            return 1.0, 0
        return min(1.0, (time.time() - s["t0"]) / (s["t_end"] - s["t0"])), s["n"]

    def finish_learn(self):
        """Завершает запись. В режиме «предмет» отбрасывает кадры, где рука не отличается от пустой
        (пока человек ещё тянулся за предметом). Возвращает {вид вырезки: ndarray (n, 512)}."""
        s, self.learn = self.learn, None
        if not s:
            return {}
        rows = s["rows"]
        if s["mode"] == "pos" and len(rows) >= 6:
            neg = self.rec.cat.examples.get(NEG_KEY, {}).get("H1")
            if neg is not None and len(neg) >= 8:
                sim = neg @ neg.T
                np.fill_diagonal(sim, -1.0)
                u_neg = 1.0 - sim.max(1)                 # насколько одна пустая рука отличается от остальных пустых
                thr = float(u_neg.mean() + 2.5 * u_neg.std())
                u = np.array([1.0 - float((neg @ r["H1"]).max()) if "H1" in r else 0.0 for r in rows])
                keep = [r for r, v in zip(rows, u) if v >= thr]
                if len(keep) < 8:                        # слишком строго — берём самую «непохожую на пустую» половину
                    keep = [rows[i] for i in np.argsort(-u)[:max(8, len(rows) // 2)]]
                rows = keep
        by_type = {}
        for r in rows:
            for typ, e in r.items():
                by_type.setdefault(typ, []).append(e)
        return {t: np.array(v, np.float32) for t, v in by_type.items()}

    def _collect(self, by_key, meta):
        s = self.learn
        if not s or time.time() > s["t_end"] or s["n"] >= s["cap"]:
            return
        hands = {k: d for k, d in by_key.items() if k[1] in ("L", "R")}
        if not hands:
            return
        if s["mode"] == "neg":
            chosen = list(hands)
        else:
            if s["slot"] in ("L", "R"):
                cand = [k for k in hands if k[1] == s["slot"]]
                if s["lock"] is None and cand:
                    s["lock"] = cand[0][0]
                chosen = [k for k in cand if k[0] == s["lock"]][:1]
            else:  # любая рука: берём ту, что сильнее всего отличается от «пустой руки»
                neg = self.rec.cat.examples.get(NEG_KEY, {}).get("H1")
                if neg is None or not len(neg):
                    chosen = list(hands)
                else:
                    for k, d in hands.items():
                        if "H1" in d:
                            s["usum"][k[1]] += float(1.0 - (neg @ d["H1"]).max())
                    best = max(s["usum"], key=s["usum"].get)
                    chosen = [k for k in hands if k[1] == best][:1]
        for k in chosen:
            row = dict(hands[k])
            if s["mode"] == "pos":  # предметы, найденные детектором в этой же руке
                for ok, od in by_key.items():
                    if ok[0] == k[0] and str(ok[1]).startswith("O") and meta.get(ok, {}).get("hand") == k[1]:
                        row["O"] = od["O"]
            s["rows"].append(row)
        s["n"] += len(chosen)

    def run(self):
        while self.alive:
            self.event.wait(0.5)
            if not self.event.is_set():
                continue
            self.event.clear()
            job, self.job = self.job, None
            if not job:
                continue
            crops, meta = job
            self.busy = True
            try:
                flat, owners = [], []
                for key, lst in crops.items():
                    for typ, c in lst:
                        flat.append(c)
                        owners.append((key, typ))
                embs = self.rec.enc.images(flat)
                by_key = {}
                for (key, typ), e in zip(owners, embs):
                    by_key.setdefault(key, {})[typ] = e
                self._collect(by_key, meta)
                now = time.time()
                for key, d in by_key.items():
                    probs = self.rec.classify_act(d["F"]) if key[1] == "F" else self.rec.classify_hand(d)
                    probs = {k: v for k, v in probs.items() if k is not None}
                    with self.lock:
                        old = self.results.get(key)
                        if old and now - old["ts"] < 1.5:  # лёгкое сглаживание во времени
                            for name in probs:
                                probs[name] = 0.6 * probs[name] + 0.4 * old["probs"].get(name, 0.0)
                        self.results[key] = {"probs": probs, "ts": now}
            except Exception as ex:  # сканер не должен ронять программу
                print("scan error:", ex)
            finally:
                self.busy = False

    def snapshot(self, max_age=1.4):
        now = time.time()
        with self.lock:
            return {k: v for k, v in self.results.items() if now - v["ts"] < max_age}

    def forget(self, alive_tids):
        with self.lock:
            for k in [k for k in self.results if k[0] not in alive_tids]:
                del self.results[k]
