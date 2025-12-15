# File path: core/tracking/deepsort_osnet_tracker.py

import numpy as np
from typing import List, Dict
from deep_sort_realtime.deepsort_tracker import DeepSort

class DeepSortOSNetTracker:
    """
    Wrapper for DeepSort with OSNet embedder.
    Uses deep-sort-realtime library with TorchReID OSNet.
    """
    
    def __init__(self, camera_id: int, max_age: int = 30, n_init: int = 3):
        """
        Args:
            camera_id: Camera identifier
            max_age: Maximum frames to keep track alive without detection (30 like v1)
            n_init: Number of consecutive detections before track is confirmed
        """
        self.camera_id = camera_id
        self.active_tracks = {}
        
        print("Initializing DeepSort with OSNet (via TorchReID)...")
        
        # Initialize DeepSort - EXACTLY LIKE YOUR WORKING V1 CONFIG
        self.tracker = DeepSort(
            max_age=max_age,              # 30 like v1
            n_init=n_init,                # 3 confirmed
            nms_max_overlap=1.0,          # Same as v1
            max_cosine_distance=0.4,      # 0.4 like v1
            nn_budget=None,               # None like v1
            override_track_class=None,
            embedder="mobilenet",         # USE MOBILENET like v1! (not torchreid)
            half=True,                    # Same as v1
            bgr=True,                     # Same as v1
            embedder_gpu=True,            # TRUE like v1 (if you have GPU)
            polygon=False,
            today=None
        )
        
        print("✅ DeepSort with mobilenet initialized successfully!")
    
    def update_tracks(self, detections: List[Dict], frame: np.ndarray, 
                     frame_number: int, timestamp: int) -> List[Dict]:
        """
        Update tracker with YOLO detections.
        
        Args:
            detections: List of dicts with 'bbox' [x1,y1,x2,y2] and 'confidence'
            frame: Current frame image (BGR format)
            frame_number: Current frame number
            timestamp: Current timestamp
            
        Returns:
            List of metadata dicts for each tracked object
        """
        metadata_list = []
        
        # Convert YOLO detections to DeepSort format
        # DeepSort expects: ([left, top, width, height], confidence, detection_class)
        deep_sort_detections = []
        for det in detections:
            bbox = det["bbox"]  # [x1, y1, x2, y2]
            conf = det["confidence"]
            
            # Convert from [x1, y1, x2, y2] to [left, top, width, height]
            left = bbox[0]
            top = bbox[1]
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            
            # Skip invalid boxes
            if width <= 0 or height <= 0:
                continue
            
            deep_sort_detections.append(([left, top, width, height], conf, 'person'))

        # Update tracker - CRITICAL: Always pass frame even if no detections
        tracks = self.tracker.update_tracks(deep_sort_detections, frame=frame)
        
        # Process confirmed tracks
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            ltrb = track.to_ltrb()  # [left, top, right, bottom]
            
            metadata = {
                "camera_id": self.camera_id,
                "local_track_id": int(track_id),
                "global_id": -1,
                "frame_number": frame_number,
                "bbox": ltrb.tolist(),
                "confidence": float(track.get_det_conf() or 0.0),
                "status": "active",
                "timestamp": timestamp,
                "embedding_dim": 256,      # MobileNet outputs 256-dim (not 512)
                "is_pooled": True,
                "embedding": None
            }
            metadata_list.append(metadata)
            self.active_tracks[int(track_id)] = metadata

        return metadata_list
    
    def get_track_embeddings(self) -> Dict[int, np.ndarray]:
        """
        Extract OSNet embeddings from active tracks.
        
        Returns:
            Dict mapping track_id to 512-dim OSNet embedding
        """
        embeddings = {}
        
        # Access internal tracker tracks
        if hasattr(self.tracker, 'tracks'):
            for track in self.tracker.tracks:
                if track.is_confirmed() and track.time_since_update == 0:
                    track_id = track.track_id
                    # Get the latest feature
                    if hasattr(track, 'features') and len(track.features) > 0:
                        embedding = track.features[-1]
                        embeddings[track_id] = embedding
        
        return embeddings
    
    def reset_tracks(self):
        """
        Clear all active tracks.
        """
        self.active_tracks = {}
        self.tracker.delete_all_tracks()