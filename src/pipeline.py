"""
pipeline.py — core engine for the Student Attention System.

Loads the three models (detection + behavior + emotion), processes video frames,
tracks each student, classifies behavior + emotion per student, and computes
attention scores. Pure CPU. Used by both the Gradio app and (later) the API.

Nothing here is Kaggle-specific; it runs on your laptop / Ubuntu server.
"""

import os
import json
import time
from collections import defaultdict, deque, Counter

import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO
import onnxruntime as ort


# ----------------------------------------------------------------------------
# CONFIG — paths are relative to the project root
# ----------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODELS = os.path.join(ROOT, "models")

# Detection: a plain YOLOv8 person detector (COCO). Small = fast on CPU.
DETECTOR_WEIGHTS = os.path.join(MODELS, "yolov8s.pt")   # auto-downloads if missing
PERSON_CLASS = 0

# Behavior + emotion: our trained ONNX models
BEHAVIOR_ONNX = os.path.join(MODELS, "behavior_best.onnx")
BEHAVIOR_LABELS = os.path.join(MODELS, "behavior_class_names.json")
EMOTION_ONNX = os.path.join(MODELS, "emotion_best.onnx")
EMOTION_LABELS = os.path.join(MODELS, "emotion_class_names.json")

IMG_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Attention scoring — which behaviors/emotions count as "engaged".
# Behavior is weighted higher (behavior model ~0.93 vs emotion ~0.84).
BEHAVIOR_ATTENTION = {
    "attentive": 1.0, "writing": 1.0, "reading": 1.0, "raising_hand": 1.0,
    "discussion": 0.8, "standing": 0.5,
    "turning_around": 0.2, "sleeping": 0.0,
}
EMOTION_ATTENTION = {
    "engaged": 1.0, "confused": 0.7, "neutral": 0.5,
    "bored": 0.1, "distracted": 0.0, "drowsy": 0.0, "frustrated": 0.3,
}
W_BEHAVIOR, W_EMOTION = 0.7, 0.3          # behavior-heavy blend
SMOOTH_WINDOW = 6                          # frames of rolling majority (~1.5-2s live)
MIN_FRAMES_TO_COUNT = 5                    # a track must appear this many times to count
TRACKER = os.path.join(HERE, "bytetrack_custom.yaml")
if not os.path.exists(TRACKER):
    TRACKER = "bytetrack.yaml"


# ----------------------------------------------------------------------------
def _softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


class Classifier:
    """Thin ONNX wrapper for a MobileNetV3 image classifier."""

    def __init__(self, onnx_path, labels_path):
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        with open(labels_path) as f:
            raw = json.load(f)
        self.labels = [raw[str(i)] for i in range(len(raw))]

    def _prep(self, bgr_crop):
        img = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
        img = (img - MEAN) / STD
        return img.transpose(2, 0, 1)[None]        # (1,3,H,W)

    def predict_batch(self, crops):
        """crops: list of BGR images -> list of (label, confidence)."""
        if not crops:
            return []
        batch = np.concatenate([self._prep(c) for c in crops], axis=0).astype(np.float32)
        logits = self.session.run(None, {self.input_name: batch})[0]
        out = []
        for row in logits:
            p = _softmax(row)
            idx = int(p.argmax())
            out.append((self.labels[idx], float(p[idx])))
        return out


class AttentionPipeline:
    def __init__(self, detector_conf=0.25, imgsz=960, max_students=300):
        self.detector = YOLO(DETECTOR_WEIGHTS)
        self.behavior = Classifier(BEHAVIOR_ONNX, BEHAVIOR_LABELS)
        self.emotion = Classifier(EMOTION_ONNX, EMOTION_LABELS)
        self.conf = detector_conf
        self.imgsz = imgsz
        self.max_students = max_students
        self.reset()

    def reset(self):
        # per-track rolling history + full logs
        self.hist_behavior = defaultdict(lambda: deque(maxlen=SMOOTH_WINDOW))
        self.hist_emotion = defaultdict(lambda: deque(maxlen=SMOOTH_WINDOW))
        self.track_frames = Counter()               # how many frames each id seen
        self.timeline = defaultdict(list)           # id -> [(t, attention)]
        self.class_curve = []                       # [(t, class_attention)]
        self.beh_counter = Counter()                # class-wide behavior counts
        self.emo_counter = Counter()                # class-wide emotion counts
        self.student_beh = defaultdict(Counter)     # id -> behavior counts
        self.per_frame_counts = []                  # persons detected per frame
        self.best_crop = {}                         # id -> (area, rgb_crop)
        self.first_seen = {}                        # id -> first timestamp
        self.last_seen = {}                         # id -> last timestamp
        self.teacher_id = None
        self._teacher_stats = defaultdict(lambda: {"n": 0, "area": 0.0, "cy": 0.0})

    # --- attention math -----------------------------------------------------
    @staticmethod
    def _score(behavior, emotion):
        b = BEHAVIOR_ATTENTION.get(behavior, 0.5)
        e = EMOTION_ATTENTION.get(emotion, 0.5)
        return W_BEHAVIOR * b + W_EMOTION * e

    def _majority(self, dq):
        return Counter(dq).most_common(1)[0][0] if dq else None

    def _update_teacher_guess(self, tid, box, W, H):
        s = self._teacher_stats[tid]
        s["n"] += 1
        s["area"] += ((box[2]-box[0])*(box[3]-box[1])) / (W*H)
        s["cy"] += ((box[1]+box[3])/2) / H

    def _resolve_teacher(self):
        # only flag a teacher when there is a real group (>= 3 people);
        # for 1-2 people (e.g. online class) nobody is a "teacher".
        qualifying = [t for t, s in self._teacher_stats.items() if s["n"] >= 3]
        if len(qualifying) < 3:
            self.teacher_id = None
            return
        best, tid = -1, None
        for t in qualifying:
            s = self._teacher_stats[t]
            score = s["n"] * (s["area"]/s["n"]) * (1.2 - s["cy"]/s["n"])
            if score > best:
                best, tid = score, t
        self.teacher_id = tid

    # --- main loop ----------------------------------------------------------
    def process_video(self, source, target_fps=2, draw=True, progress=None,
                      single_person=None):
        """Yields annotated frames if draw=True; always fills internal logs.
        single_person: None=auto, True=force personal, False=force class."""
        cap = cv2.VideoCapture(source)
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        step = max(1, round(src_fps / target_fps))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.reset()

        # ultralytics track over sampled frames via manual stepping
        idx = 0
        processed = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step != 0:
                idx += 1
                continue
            t = idx / src_fps
            H, W = frame.shape[:2]

            res = self.detector.track(
                frame, classes=[PERSON_CLASS], conf=self.conf, imgsz=self.imgsz,
                max_det=self.max_students, persist=True, tracker=TRACKER,
                verbose=False)[0]

            dets = []
            if res.boxes.id is not None:
                ids = res.boxes.id.int().cpu().tolist()
                boxes = res.boxes.xyxy.cpu().numpy()
                for tid, b in zip(ids, boxes):
                    x1, y1, x2, y2 = [int(v) for v in b]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(W, x2), min(H, y2)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    dets.append((tid, (x1, y1, x2, y2)))
                    self._update_teacher_guess(tid, (x1, y1, x2, y2), W, H)

            # classify all crops in a batch
            crops = [frame[y1:y2, x1:x2] for _, (x1, y1, x2, y2) in dets]
            beh = self.behavior.predict_batch(crops)
            emo = self.emotion.predict_batch(crops)
            self.per_frame_counts.append(len(dets))

            frame_scores = []
            per_student = []
            for (tid, box), (bl, _), (el, _) in zip(dets, beh, emo):
                self.track_frames[tid] += 1
                self.hist_behavior[tid].append(bl)
                self.hist_emotion[tid].append(el)
                sb = self._majority(self.hist_behavior[tid])
                se = self._majority(self.hist_emotion[tid])
                att = self._score(sb, se)
                self.timeline[tid].append((t, att))
                self.beh_counter[sb] += 1
                self.emo_counter[se] += 1
                self.student_beh[tid][sb] += 1
                # keep the biggest crop seen of this student (for the gallery)
                x1, y1, x2, y2 = box
                area = (x2 - x1) * (y2 - y1)
                if tid not in self.best_crop or area > self.best_crop[tid][0]:
                    self.best_crop[tid] = (area, cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
                self.first_seen.setdefault(tid, t)
                self.last_seen[tid] = t
                per_student.append((tid, box, sb, se, att))
                frame_scores.append(att)

            if frame_scores:
                self.class_curve.append((t, float(np.mean(frame_scores))))

            if draw:
                yield self._draw(frame, per_student, t)
            processed += 1
            if progress:
                progress(min(idx / max(total, 1), 0.99))
            idx += 1

        cap.release()
        self._resolve_teacher()
        self.single_person = (single_person if single_person is not None
                              else self._count_students() <= 1)

    # --- real-time single frame (for live streaming) ------------------------
    def process_frame(self, frame_bgr, personal_mode=True):
        """Process ONE frame (webcam). Returns (annotated_bgr, people_list).
        people_list = [{id, box, behavior, emotion, attention}]."""
        H, W = frame_bgr.shape[:2]
        res = self.detector.track(
            frame_bgr, classes=[PERSON_CLASS], conf=self.conf, imgsz=self.imgsz,
            max_det=self.max_students, persist=True, tracker=TRACKER,
            verbose=False)[0]

        dets = []
        if res.boxes.id is not None:
            ids = res.boxes.id.int().cpu().tolist()
            boxes = res.boxes.xyxy.cpu().numpy()
            for tid, b in zip(ids, boxes):
                x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                dets.append((tid, (x1, y1, x2, y2)))

        crops = [frame_bgr[y1:y2, x1:x2] for _, (x1, y1, x2, y2) in dets]
        beh = self.behavior.predict_batch(crops)
        emo = self.emotion.predict_batch(crops)

        people = []
        for (tid, box), (bl, _), (el, _) in zip(dets, beh, emo):
            self.track_frames[tid] += 1
            self.hist_behavior[tid].append(bl)
            self.hist_emotion[tid].append(el)
            sb = self._majority(self.hist_behavior[tid])
            se = self._majority(self.hist_emotion[tid])
            people.append({"id": tid, "box": box, "behavior": sb,
                           "emotion": se, "attention": self._score(sb, se)})

        annotated = frame_bgr.copy()
        for p in people:
            x1, y1, x2, y2 = p["box"]
            att = p["attention"]
            color = (0, 180, 0) if att >= 0.6 else (0, 200, 255) if att >= 0.35 else (0, 0, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
            cv2.putText(annotated, f"#{p['id']} {p['behavior']} | {p['emotion']}",
                        (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        return annotated, people

    # --- drawing ------------------------------------------------------------
    def _draw(self, frame, per_student, t):
        for tid, (x1, y1, x2, y2), sb, se, att in per_student:
            is_teacher = (tid == self.teacher_id)
            if is_teacher:
                color = (255, 140, 0)      # orange (BGR-ish)
                label = f"TEACHER"
            else:
                color = (0, 180, 0) if att >= 0.6 else (0, 200, 255) if att >= 0.35 else (0, 0, 255)
                label = f"#{tid} {sb}/{se}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        # header
        n = self._count_students()
        cur = self.class_curve[-1][1]*100 if self.class_curve else 0
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 34), (20, 20, 20), -1)
        cv2.putText(frame, f"Students: {n}   Class attention: {cur:.0f}%   t={t:.0f}s",
                    (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)
        return frame

    # --- reporting ----------------------------------------------------------
    def _count_students(self):
        real = [tid for tid, n in self.track_frames.items()
                if n >= MIN_FRAMES_TO_COUNT and tid != self.teacher_id]
        return len(real)

    def report(self):
        real_ids = [tid for tid, n in self.track_frames.items()
                    if n >= MIN_FRAMES_TO_COUNT and tid != self.teacher_id]
        per_student = {}
        for tid in real_ids:
            atts = [a for _, a in self.timeline[tid]]
            dom = self.student_beh[tid].most_common(1)
            per_student[tid] = {
                "attention_pct": round(100*np.mean(atts), 1) if atts else 0,
                "frames": self.track_frames[tid],
                "dominant_behavior": dom[0][0] if dom else "-",
            }
        overall = (round(100*np.mean([c for _, c in self.class_curve]), 1)
                   if self.class_curve else 0)

        # robust student count: how many people are typically in frame at once
        # (this avoids ID-churn inflation from unique-track counting)
        if self.per_frame_counts:
            typical = int(np.percentile(self.per_frame_counts, 85))
            peak = int(np.max(self.per_frame_counts))
        else:
            typical = len(real_ids); peak = len(real_ids)
        est_students = max(typical - (1 if self.teacher_id is not None else 0), 0)

        # crops for the gallery (biggest crop of each real student)
        crops = [(self.best_crop[tid][1], tid, per_student[tid])
                 for tid in real_ids if tid in self.best_crop]

        # movement events (approx): appeared after start / lost before end
        dur = max(self.last_seen.values(), default=0.0)
        events = []
        for tid in real_ids:
            fs = self.first_seen.get(tid, 0.0); ls = self.last_seen.get(tid, dur)
            if fs > 1.5:
                events.append((round(fs, 1), f"student #{tid} appeared"))
            if ls < dur - 1.5:
                events.append((round(ls, 1), f"student #{tid} left / lost"))
        events.sort()

        return {
            "student_count": est_students,          # headline (robust)
            "unique_tracks": len(real_ids),         # for transparency
            "peak_in_frame": peak,
            "teacher_id": self.teacher_id,
            "overall_attention": overall,
            "per_student": per_student,
            "class_curve": self.class_curve,
            "behavior_dist": dict(self.beh_counter),
            "emotion_dist": dict(self.emo_counter),
            "timeline": {tid: self.timeline[tid] for tid in real_ids},
            "crops": crops,
            "events": events,
        }
