"""
Quick Test Script for DeepSort Tracker
Tests if tracking is working properly on a video
"""

import cv2
import numpy as np
import time
from core.detection.yolo_detector import YOLODetector
from core.tracking.deepsort_osnet_tracker import DeepSortOSNetTracker

def test_deepsort(video_source, duration_seconds=30):
    """
    Test DeepSort tracking on a video source.
    
    Args:
        video_source: Path to video file or camera index (0 for webcam)
        duration_seconds: How long to run the test (None = full video)
    """
    print("="*60)
    print("🧪 DEEPSORT TRACKING TEST")
    print("="*60)
    
    # Initialize
    print("\n1️⃣ Initializing YOLO Detector...")
    detector = YOLODetector(model_path='yolov8m.pt', device='cpu')
    print("   ✅ YOLO initialized")
    
    print("\n2️⃣ Initializing DeepSort Tracker...")
    tracker = DeepSortOSNetTracker(camera_id=1, max_age=30, n_init=3)
    print("   ✅ DeepSort initialized")
    
    # Open video
    print(f"\n3️⃣ Opening video source: {video_source}")
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print(f"   ❌ Failed to open video source!")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"   ✅ Video opened - FPS: {fps:.1f}, Resolution: {width}x{height}")
    
    # Statistics
    frame_count = 0
    total_detections = 0
    total_tracks = 0
    unique_track_ids = set()
    start_time = time.time()
    
    print("\n4️⃣ Processing frames...")
    print("   Press 'q' to quit, 'p' to pause/resume")
    print("-"*60)
    
    paused = False
    
    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("\n   📹 End of video reached")
                break
            
            frame_count += 1
            timestamp = int(time.time() * 1000)
            
            # Check duration limit
            if duration_seconds and (time.time() - start_time) > duration_seconds:
                print(f"\n   ⏰ Duration limit reached ({duration_seconds}s)")
                break
            
            # 1. Detect persons
            detections = detector.detect(frame, conf_threshold=0.5)
            total_detections += len(detections)
            
            # 2. Update tracker
            tracks_metadata = tracker.update_tracks(
                detections=detections,
                frame=frame,
                frame_number=frame_count,
                timestamp=timestamp
            )
            
            total_tracks += len(tracks_metadata)
            
            # Collect unique track IDs
            for track in tracks_metadata:
                unique_track_ids.add(track['local_track_id'])
            
            # 3. Visualize
            vis_frame = frame.copy()
            
            # Draw detections (yellow boxes)
            for det in detections:
                bbox = det['bbox']
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 255), 1)
            
            # Draw tracks (green boxes with IDs)
            for track in tracks_metadata:
                track_id = track['local_track_id']
                bbox = track['bbox']
                x1, y1, x2, y2 = map(int, bbox)
                
                # Draw bounding box
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw track ID
                cv2.putText(
                    vis_frame,
                    f"ID: {track_id}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )
            
            # Draw statistics
            current_fps = frame_count / (time.time() - start_time) if (time.time() - start_time) > 0 else 0
            
            cv2.putText(vis_frame, f"Frame: {frame_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(vis_frame, f"FPS: {current_fps:.1f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(vis_frame, f"Detections: {len(detections)}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(vis_frame, f"Active Tracks: {len(tracks_metadata)}", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(vis_frame, f"Unique IDs: {len(unique_track_ids)}", (10, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow('DeepSort Test', vis_frame)
            
            # Print progress every 30 frames
            if frame_count % 30 == 0:
                print(f"   Frame {frame_count}: {len(detections)} detections, "
                      f"{len(tracks_metadata)} tracks, "
                      f"{len(unique_track_ids)} unique IDs")
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n   🛑 Quit requested")
            break
        elif key == ord('p'):
            paused = not paused
            print(f"   {'⏸️ Paused' if paused else '▶️ Resumed'}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Print final statistics
    elapsed_time = time.time() - start_time
    
    print("\n" + "="*60)
    print("📊 TEST RESULTS")
    print("="*60)
    print(f"✅ Frames Processed:      {frame_count}")
    print(f"✅ Total Detections:      {total_detections}")
    print(f"✅ Total Track Instances: {total_tracks}")
    print(f"✅ Unique Track IDs:      {len(unique_track_ids)}")
    print(f"✅ Average FPS:           {frame_count/elapsed_time:.2f}")
    print(f"✅ Processing Time:       {elapsed_time:.2f}s")
    
    if len(unique_track_ids) > 0:
        print(f"✅ Track IDs Found:       {sorted(unique_track_ids)}")
        print("\n🎉 DeepSort is WORKING! Tracks are being assigned.")
    else:
        print("\n⚠️ No tracks found! Possible issues:")
        print("   - No persons detected in video")
        print("   - Detection confidence too high")
        print("   - Tracking parameters too strict")
    
    print("="*60)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test DeepSort Tracking')
    parser.add_argument('--source', type=str, default='0',
                       help='Video source (0 for webcam or path to video file)')
    parser.add_argument('--duration', type=int, default=None,
                       help='Test duration in seconds (None = full video)')
    
    args = parser.parse_args()
    
    # Convert source to int if it's a digit (webcam)
    video_source = int(args.source) if args.source.isdigit() else args.source
    
    try:
        test_deepsort(video_source, args.duration)
    except KeyboardInterrupt:
        print("\n\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()