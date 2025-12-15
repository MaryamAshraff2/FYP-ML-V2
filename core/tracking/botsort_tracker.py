# File path: core/tracking/botsort_tracker.py

# ===========================
# Imports
# ===========================
import numpy as np
from typing import List, Dict
from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.utils import IterableSimpleNamespace

# ===========================
# BOTSort Tracker Wrapper
# ===========================
class BOTSortTracker:
    """
    Wrapper around Ultralytics BoT-SORT tracker.

    Responsibilities:
        - Maintain camera ID
        - Convert YOLO.track() results into metadata
        - Handle local track IDs
        - Provide placeholders for embeddings for future ReID
        - Hook for multi-camera global ID assignment later
    """

    def __init__(self, camera_id: int):
        """
        Args:
            camera_id (int): Identifier for the camera
        """
        self.camera_id = camera_id
        self.active_tracks = {}    # {local_track_id: metadata dict}
        
        # Create BOTSort args with default configuration
        tracker_args = {
            'track_high_thresh': 0.5,      # High threshold for track activation
            'track_low_thresh': 0.1,       # Low threshold for track activation
            'new_track_thresh': 0.6,       # Threshold for new track creation
            'track_buffer': 30,            # Buffer for lost tracks
            'match_thresh': 0.8,           # Matching threshold
            'proximity_thresh': 0.5,       # Proximity threshold
            'appearance_thresh': 0.25,     # Appearance threshold
            'fuse_score': True,            # Fuse detection and tracking scores
            'with_reid': False,            # Use ReID features (set True if using embeddings)
            'gmc_method': 'sparseOptFlow', # Camera motion compensation method
            'frame_rate': 30               # Video frame rate
        }
        
        # Convert dict to IterableSimpleNamespace (required by BOTSORT)
        self.args = IterableSimpleNamespace(**tracker_args)
        self.tracker = BOTSORT(args=self.args)

    # ===========================
    # Update tracker per frame
    # ===========================
    def update_tracks(self, results, frame_number: int, timestamp: int) -> List[Dict]:
        """
        Update BOTSort tracker with YOLO results and get tracked objects.

        Args:
            results: YOLO results object from model inference
            frame_number: current frame number
            timestamp: current timestamp

        Returns:
            List[Dict]: metadata per track
        """
        metadata_list = []

        # Check if results contain detections and tracker has assigned IDs
        if results and len(results) > 0 and hasattr(results[0].boxes, 'id') and results[0].boxes.id is not None:
            boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            local_ids = results[0].boxes.id.cpu().numpy().astype(int)

            for box, conf, local_id in zip(boxes_xyxy, confidences, local_ids):
                metadata = {
                    "camera_id": self.camera_id,
                    "local_track_id": int(local_id),
                    "global_id": -1,          # To be assigned in identity layer
                    "frame_number": frame_number,
                    "bbox": box.tolist(),      # [x1, y1, x2, y2]
                    "confidence": float(conf),
                    "status": "active",
                    "timestamp": timestamp,
                    "embedding_dim": 0,        # Will be filled after ReID
                    "is_pooled": True,         # Main pooled embedding
                    "embedding": None          # Will be filled after ReID
                }
                metadata_list.append(metadata)
                self.active_tracks[int(local_id)] = metadata

        return metadata_list

    # ===========================
    # Optional: reset active tracks
    # ===========================
    def reset_tracks(self):
        """
        Clear currently active tracks. Useful when starting a new video or camera.
        """
        self.active_tracks = {}
        # Reinitialize tracker
        self.tracker = BOTSORT(args=self.args)