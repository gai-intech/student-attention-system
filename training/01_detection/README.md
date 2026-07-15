# Student Detector (Stage 1)
Pretrained COCO YOLOv8 person detector + ByteTrack. No training (see SCB-05 diagnostic).
Config: {'model': 'yolov8m.pt', 'conf': 0.25, 'imgsz': 1280, 'max_det': 300, 'person_class': 0, 'tracker': 'bytetrack.yaml'}
Chosen because SCB-05 labels ~2 students/image while COCO detects ~14/image.
SCB-05 is reused as behavior crops in Notebook 2.

    from detect_and_track import detect_and_track
    for frame_idx, dets in detect_and_track("classroom.mp4"):
        for track_id, x1,y1,x2,y2, conf in dets: ...
