# File path: tests/test_yolo_botsort.py

import cv2
import time
import numpy as np
from core.detection.yolo_detector import YOLODetector
from core.tracking.botsort_tracker import BOTSortTracker
from core.tracking.track_utils import generate_metadata_summary, filter_tracks_by_confidence

# ===========================
# Config
# ===========================
VIDEO_PATH = "NED-KHI-P26_001_2025-09-03-08-20-01_2025-09-03-08-30-00 (online-video-cutter.com).mp4"
CAMERA_ID = 1
CONF_THRESHOLD = 0.3

# ===========================
# Initialize Models
# ===========================
yolo = YOLODetector(model_path="yolov8m.pt", device="cpu")  # or "cuda"
tracker = BOTSortTracker(camera_id=CAMERA_ID)

# ===========================
# Open Video
# ===========================
cap = cv2.VideoCapture(VIDEO_PATH)
frame_number = 0
all_metadata = []

print("Starting video processing with BOTSort tracking...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_number += 1
    timestamp = int(time.time() * 1000)  # milliseconds

    # --- 1. YOLO Detection + BOTSort Tracking ---
    results = yolo.model.track(
        frame,
        classes=[0],
        persist=True,
        verbose=False,
        tracker="botsort.yaml",
        conf=CONF_THRESHOLD
    )

    # --- 2. Update BOTSort Tracker ---
    metadata = tracker.update_tracks(
        results=results,
        frame_number=frame_number,
        timestamp=timestamp
    )
    all_metadata.extend(metadata)

    # --- 3. Filter tracks by confidence ---
    metadata_filtered = filter_tracks_by_confidence(metadata, min_confidence=CONF_THRESHOLD)

    # --- 4. Visualization ---
    vis_frame = frame.copy()
    if results and len(results) > 0 and hasattr(results[0].boxes, "id"):
        boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()
        local_ids = results[0].boxes.id.cpu().numpy()

        for box, track_id in zip(boxes_xyxy, local_ids):
            x1, y1, x2, y2 = map(int, box[:4])
            # Draw bounding box
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Draw local ID
            cv2.putText(vis_frame, f"ID {track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display frame
    cv2.imshow(f"BOTSort Tracking - Camera {CAMERA_ID}", vis_frame)

    # Quit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    # --- 5. Print per-frame summary ---
    if frame_number % 30 == 0:
        print(f"Frame {frame_number} | Active tracks: {len(metadata_filtered)}")

cap.release()
cv2.destroyAllWindows()

# ===========================
# Final Summary
# ===========================
summary = generate_metadata_summary(all_metadata)
print("\n=== VIDEO SUMMARY ===")
print(summary)

# ===========================
# Example: Print first 5 track metadata
# ===========================
print("\nFirst 5 track metadata:")
for meta in all_metadata[:5]:
    print(meta)
