# Reusable student detector+tracker (COCO person + ByteTrack)
from ultralytics import YOLO
import json, os
_cfg = json.load(open(os.path.join(os.path.dirname(__file__), "detector_config.json")))
_model = YOLO(_cfg["model"])

def detect_and_track(source, device=0):
    for fi, r in enumerate(_model.track(source=source, classes=[_cfg["person_class"]],
            conf=_cfg["conf"], imgsz=_cfg["imgsz"], max_det=_cfg["max_det"],
            device=device, tracker=_cfg["tracker"], persist=True, stream=True, verbose=False)):
        rows = []
        if r.boxes.id is not None:
            for tid, b, c in zip(r.boxes.id.int().cpu().tolist(),
                                 r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                rows.append((tid, *[float(v) for v in b], float(c)))
        yield fi, rows
