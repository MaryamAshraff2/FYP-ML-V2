# File path: core/detection/yolo_detector.py

# ===========================
# Imports
# ===========================
import numpy as np
import torch
from ultralytics import YOLO
# from core.detection.preprocess import preprocess_frame
from typing import List, Dict, Union

# ===========================
# YOLO Detector Class
# ===========================
class YOLODetector:
    """
    Wrapper for YOLOv8m pretrained model for person detection.
    Detects only persons and returns bounding boxes + confidence scores.
    """

    def __init__(self, model_path: str = "yolov8m.pt", device: str = "cpu"):
        """
        Initialize YOLOv8m model
        Args:
            model_path (str): Path to YOLOv8m pretrained weights
            device (str): 'cpu' or 'cuda'
        """
        self.device = device
        self.model = YOLO(model_path)
        self.model.to(device)
        self.model.classes = [0]  # Only person class (class 0 in COCO)

    def detect(self, frame: np.ndarray, conf_threshold: float = 0.1) -> List[Dict[str, Union[np.ndarray, float]]]:
        """
        Detect persons in a single frame using YOLOv8m.track()
        Args:
            frame (np.ndarray): Image frame (H, W, C)
            conf_threshold (float): Minimum confidence to keep detection
        Returns:
            List[Dict[str, Union[np.ndarray, float]]]: bounding boxes + confidence
        """
        results = self.model.track(frame, classes=[0], persist=False, verbose=False)  # Only persons

        detections = []
        if results and len(results) > 0:
            for box, conf in zip(results[0].boxes.xyxy.cpu().numpy(),
                                 results[0].boxes.conf.cpu().numpy()):
                if conf >= conf_threshold:
                    detections.append({"bbox": box.tolist(), "confidence": float(conf)})
        return detections