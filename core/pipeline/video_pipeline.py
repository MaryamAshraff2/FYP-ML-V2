"""
Video Pipeline - Processes a single video stream with detection, tracking, and ReID
Adapted from your working v1 code
"""

import cv2
import numpy as np
import time
from typing import Dict, List, Optional
from collections import defaultdict

from core.detection.yolo_detector import YOLODetector
from core.tracking.deepsort_osnet_tracker import DeepSortOSNetTracker
from core.reid.feature_extractor import extract_reid_embedding


class VideoPipeline:
    """
    Processes a single video source with:
    1. YOLO person detection
    2. DeepSort tracking
    3. ReID feature extraction
    4. Embedding pooling
    """
    
    def __init__(
        self,
        camera_id: int,
        video_source: str,
        yolo_model_path: str = 'yolov8m.pt',
        pooling_threshold: int = 10,
        conf_threshold: float = 0.5,
        device: str = 'cpu'
    ):
        """
        Initialize the video processing pipeline.
        
        Args:
            camera_id: Camera identifier
            video_source: Path to video file or camera index (0 for webcam)
            yolo_model_path: Path to YOLO weights
            pooling_threshold: Number of embeddings to pool before storing
            conf_threshold: Detection confidence threshold
            device: 'cpu' or 'cuda'
        """
        self.camera_id = camera_id
        self.video_source = video_source
        self.pooling_threshold = pooling_threshold
        self.conf_threshold = conf_threshold
        self.device = device
        
        # Initialize detector
        print(f"🚀 Initializing YOLO detector for Camera {camera_id}...")
        self.detector = YOLODetector(
            model_path=yolo_model_path,
            device=device
        )
        
        # Initialize tracker
        print(f"🚀 Initializing DeepSort tracker for Camera {camera_id}...")
        self.tracker = DeepSortOSNetTracker(
            camera_id=camera_id,
            max_age=30,
            n_init=3
        )
        
        # Pooling buffers
        self.embedding_pools = defaultdict(list)  # {track_id: [embeddings]}
        self.track_metadata = defaultdict(dict)   # {track_id: metadata}
        
        # Performance tracking
        self.frame_count = 0
        self.total_processing_time = 0
        
        # Open video
        self.cap = cv2.VideoCapture(video_source)
        if not self.cap.isOpened():
            raise ValueError(f"❌ Cannot open video source: {video_source}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"📊 Video info - FPS: {self.fps:.1f}, Resolution: {self.width}x{self.height}")
    
    def extract_and_pool_embeddings(
        self, 
        frame: np.ndarray, 
        bbox: List[float], 
        track_id: int
    ) -> Optional[np.ndarray]:
        """
        Extract ReID embedding and apply mean pooling.
        
        Args:
            frame: Current frame
            bbox: Bounding box [x1, y1, x2, y2]
            track_id: Track ID
            
        Returns:
            Pooled embedding if threshold reached, None otherwise
        """
        try:
            # Extract embedding
            embedding = extract_reid_embedding(frame, bbox)
            
            if embedding is None:
                return None
            
            # Add to pool
            self.embedding_pools[track_id].append(embedding)
            
            # Check if ready for pooling
            if len(self.embedding_pools[track_id]) >= self.pooling_threshold:
                # Mean pooling
                pooled = np.mean(self.embedding_pools[track_id], axis=0)
                
                # Clear pool
                self.embedding_pools[track_id] = []
                
                return pooled
            
            return None
            
        except Exception as e:
            print(f"⚠️ Error extracting embedding for track {track_id}: {e}")
            return None
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Process a single frame through detection, tracking, and ReID.
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            Dict with processing results and metadata
        """
        start_time = time.time()
        self.frame_count += 1
        timestamp = int(time.time() * 1000)
        
        # 1. Detect persons
        detections = self.detector.detect(frame, conf_threshold=self.conf_threshold)
        
        # 2. Update tracker
        tracks_metadata = self.tracker.update_tracks(
            detections=detections,
            frame=frame,
            frame_number=self.frame_count,
            timestamp=timestamp
        )
        
        # 3. Extract and pool embeddings
        pooled_embeddings = []
        
        for track_meta in tracks_metadata:
            track_id = track_meta['local_track_id']
            bbox = track_meta['bbox']
            
            # Update track history
            if track_id not in self.track_metadata:
                self.track_metadata[track_id] = {
                    'first_seen': self.frame_count,
                    'last_seen': self.frame_count,
                    'bbox_history': [],
                    'frame_numbers': []
                }
            
            self.track_metadata[track_id]['last_seen'] = self.frame_count
            self.track_metadata[track_id]['bbox_history'].append(bbox)
            self.track_metadata[track_id]['frame_numbers'].append(self.frame_count)
            
            # Extract and pool embeddings
            pooled_emb = self.extract_and_pool_embeddings(frame, bbox, track_id)
            
            if pooled_emb is not None:
                # Calculate average bbox from recent history
                recent_bboxes = self.track_metadata[track_id]['bbox_history'][-self.pooling_threshold:]
                avg_bbox = np.mean(recent_bboxes, axis=0).tolist()
                
                # Store pooled embedding with metadata
                track_meta['embedding'] = pooled_emb
                track_meta['bbox'] = avg_bbox
                pooled_embeddings.append(track_meta)
                
                # Cleanup old history to prevent memory leak
                if len(self.track_metadata[track_id]['bbox_history']) > 100:
                    self.track_metadata[track_id]['bbox_history'] = \
                        self.track_metadata[track_id]['bbox_history'][-50:]
                    self.track_metadata[track_id]['frame_numbers'] = \
                        self.track_metadata[track_id]['frame_numbers'][-50:]
        
        # Calculate performance metrics
        processing_time = time.time() - start_time
        self.total_processing_time += processing_time
        avg_fps = self.frame_count / self.total_processing_time if self.total_processing_time > 0 else 0
        
        return {
            'frame_number': self.frame_count,
            'timestamp': timestamp,
            'detections': detections,
            'tracks': tracks_metadata,
            'pooled_embeddings': pooled_embeddings,
            'fps': avg_fps,
            'active_track_count': len(tracks_metadata)
        }
    
    def visualize_frame(self, frame: np.ndarray, results: Dict) -> np.ndarray:
        """
        Draw tracking visualization on frame.
        
        Args:
            frame: Input frame
            results: Results from process_frame()
            
        Returns:
            Annotated frame
        """
        vis_frame = frame.copy()
        
        # Draw tracks
        for track in results['tracks']:
            track_id = track['local_track_id']
            bbox = track['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            
            # Draw bounding box
            color = (0, 255, 0)  # Green
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw track ID
            cv2.putText(
                vis_frame, 
                f"ID: {track_id}", 
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                color, 
                2
            )
            
            # Draw pooling progress
            pool_size = len(self.embedding_pools.get(track_id, []))
            cv2.putText(
                vis_frame,
                f"Pool: {pool_size}/{self.pooling_threshold}",
                (x1, y2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 0),
                1
            )
        
        # Draw FPS and stats
        cv2.putText(
            vis_frame,
            f"FPS: {results['fps']:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )
        
        cv2.putText(
            vis_frame,
            f"Camera: {self.camera_id}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )
        
        cv2.putText(
            vis_frame,
            f"Active Tracks: {results['active_track_count']}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
        
        return vis_frame
    
    def run(self, visualize: bool = True, callback=None):
        """
        Main processing loop.
        
        Args:
            visualize: Whether to show visualization window
            callback: Optional callback function to handle pooled embeddings
                     Signature: callback(camera_id, pooled_embeddings_list)
        """
        print(f"🎬 Starting video pipeline for Camera {self.camera_id}")
        print("Press 'q' to quit, 'p' to pause/resume")
        
        paused = False
        
        while self.cap.isOpened():
            if not paused:
                ret, frame = self.cap.read()
                if not ret:
                    print(f"📹 End of video for Camera {self.camera_id}")
                    break
                
                # Process frame
                results = self.process_frame(frame)
                
                # Call callback with pooled embeddings
                if callback and len(results['pooled_embeddings']) > 0:
                    callback(self.camera_id, results['pooled_embeddings'])
                
                # Visualize
                if visualize:
                    vis_frame = self.visualize_frame(frame, results)
                    cv2.imshow(f'Camera {self.camera_id} - Person ReID', vis_frame)
            
            # Handle keyboard input
            if visualize:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    paused = not paused
                    print(f"{'⏸️ Paused' if paused else '▶️ Resumed'}")
        
        # Cleanup
        self.cleanup()
    
    def cleanup(self):
        """Release resources and print statistics."""
        self.cap.release()
        
        # Only try to close window if visualize was True
        try:
            if cv2.getWindowProperty(f'Camera {self.camera_id} - Person ReID', 0) >= 0:
                cv2.destroyWindow(f'Camera {self.camera_id} - Person ReID')
        except:
            pass  # Window doesn't exist or can't be accessed
        
        print(f"\n📊 Pipeline Statistics for Camera {self.camera_id}:")
        print(f"   Total frames processed: {self.frame_count}")
        if self.total_processing_time > 0:
            print(f"   Average FPS: {self.frame_count/self.total_processing_time:.1f}")
        print(f"   Total unique tracks: {len(self.track_metadata)}")
        print(f"   Active embedding pools: {len(self.embedding_pools)}")
        print("✅ Pipeline completed!")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()