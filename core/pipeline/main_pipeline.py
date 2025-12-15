"""
Main Pipeline - Orchestrates multi-camera person ReID system with Qdrant
Updated V2 with Qdrant integration
"""

import argparse
import json
import uuid
from typing import Dict, List
from core.pipeline.video_pipeline import VideoPipeline
from core.reid.feature_extractor import init_reid_model

# Qdrant imports
from database.qdrant.client import get_qdrant_client, QdrantClientManager
from database.qdrant.collections import initialize_collection, get_collection_info
from database.qdrant.uploader import QdrantUploader
from database.qdrant.queries import QdrantQueryManager
from core.identity.clustering import IdentityClusterer


class MainPipeline:
    """
    Orchestrates the complete person ReID system:
    1. Initialize Qdrant connection and collection
    2. Process multiple camera streams
    3. Extract and pool embeddings
    4. Upload to Qdrant database in real-time
    5. Perform cross-camera clustering
    6. Assign global IDs
    """
    
    def __init__(
        self,
        pooling_threshold: int = 10,
        yolo_model: str = 'yolov8m.pt',
        reid_model: str = 'osnet_x1_0',
        device: str = 'cpu',
        use_qdrant: bool = True,
        clustering_method: str = 'connected_components'
    ):
        """
        Initialize main pipeline.
        
        Args:
            pooling_threshold: Number of frames for embedding pooling
            yolo_model: YOLO model path
            reid_model: ReID model name
            device: 'cpu' or 'cuda'
            use_qdrant: Enable Qdrant storage and clustering
            clustering_method: 'connected_components' or 'greedy'
        """
        self.pooling_threshold = pooling_threshold
        self.yolo_model = yolo_model
        self.reid_model_name = reid_model
        self.device = device
        self.use_qdrant = use_qdrant
        self.clustering_method = clustering_method
        
        # Generate unique session ID for this run
        self.video_session_id = str(uuid.uuid4())
        print(f"🆔 Session ID: {self.video_session_id}")
        
        # Initialize ReID model once (shared across cameras)
        print("🚀 Initializing shared ReID model...")
        self.reid_model = init_reid_model(reid_model, device)
        
        # Storage for embeddings (backup/fallback)
        self.all_embeddings = []
        
        # Qdrant components
        self.qdrant_uploader = None
        self.qdrant_query_manager = None
        self.clusterer = None
        
        if self.use_qdrant:
            self._initialize_qdrant()
    
    # 
    def _initialize_qdrant(self):
            """Initialize Qdrant connection, collection, and components."""
            try:
                print("\n" + "="*60)
                print("🔌 Initializing Qdrant Integration")
                print("="*60 + "\n")
                
                # 1. Initialize client connection
                qdrant_manager = QdrantClientManager()
                
                # 2. Health check
                if not qdrant_manager.health_check():
                    print("⚠️ Qdrant health check failed, disabling Qdrant")
                    self.use_qdrant = False
                    return
                
                # 3. Initialize collection
                print("📦 Initializing collection...")
                success = initialize_collection(recreate=False)
                
                if not success:
                    print("⚠️ Collection initialization failed, disabling Qdrant")
                    self.use_qdrant = False
                    return
                
                # 4. Display collection info
                try:
                    info = get_collection_info()
                    print(f"📊 Collection: {info.get('name', 'unknown')}")
                    print(f"   Points: {info.get('points_count', 0)}")
                    print(f"   Vector size: {info.get('vector_size', 'unknown')}")
                except Exception as e:
                    print(f"⚠️ Could not get collection info: {e}")
                
                # 5. Initialize components
                self.qdrant_uploader = QdrantUploader()
                self.qdrant_query_manager = QdrantQueryManager()
                
                # 6. Initialize clusterer with try-catch to handle config errors
                try:
                    self.clusterer = IdentityClusterer(method=self.clustering_method)
                except Exception as e:
                    print(f"⚠️ Could not initialize clusterer: {e}")
                    print(f"   Creating basic clusterer without advanced config...")
                    # Try creating clusterer without relying on config
                    try:
                        from core.identity.clustering import IdentityClusterer
                        self.clusterer = IdentityClusterer(
                            method=self.clustering_method,
                            similarity_threshold=0.70,
                            distance_threshold=0.30
                        )
                    except Exception as e2:
                        print(f"❌ Clusterer initialization failed completely: {e2}")
                        # Disable Qdrant if clusterer is critical
                        self.use_qdrant = False
                        return
                
                print("✅ Qdrant integration ready!\n")
                
            except Exception as e:
                print(f"❌ Qdrant initialization error: {e}")
                import traceback
                traceback.print_exc()
                print("⚠️ Falling back to in-memory storage only")
                self.use_qdrant = False
    
    def embedding_callback(self, camera_id: int, pooled_embeddings: List[Dict]):
        """
        Callback to handle pooled embeddings from video pipelines.
        
        Args:
            camera_id: Source camera ID
            pooled_embeddings: List of embedding metadata dicts
        """
        print(f"📊 Received {len(pooled_embeddings)} pooled embeddings from Camera {camera_id}")
        
        # Store in-memory (backup)
        self.all_embeddings.extend(pooled_embeddings)
        
        # Upload to Qdrant if enabled
        if self.use_qdrant and self.qdrant_uploader:
            try:
                self.qdrant_uploader.upload_embeddings(
                    camera_id=camera_id,
                    pooled_embeddings=pooled_embeddings,
                    video_session_id=self.video_session_id
                )
            except Exception as e:
                print(f"⚠️ Qdrant upload error: {e}")
    
    def process_video(
        self,
        camera_id: int,
        video_source: str,
        visualize: bool = True
    ):
        """
        Process a single video stream.
        
        Args:
            camera_id: Camera identifier
            video_source: Path to video or camera index
            visualize: Show visualization window
        """
        print(f"\n{'='*60}")
        print(f"🎬 Processing Camera {camera_id}: {video_source}")
        print(f"{'='*60}\n")
        
        # Create pipeline
        pipeline = VideoPipeline(
            camera_id=camera_id,
            video_source=video_source,
            yolo_model_path=self.yolo_model,
            pooling_threshold=self.pooling_threshold,
            device=self.device
        )
        
        # Run with callback
        pipeline.run(
            visualize=visualize,
            callback=self.embedding_callback
        )
    
    def process_multiple_videos(
        self,
        video_sources: Dict[int, str],
        visualize: bool = True
    ):
        """
        Process multiple video streams sequentially.
        
        Args:
            video_sources: Dict mapping camera_id to video_source
            visualize: Show visualization windows
        """
        for camera_id, video_source in video_sources.items():
            self.process_video(camera_id, video_source, visualize)
        
        # Flush any remaining batched embeddings to Qdrant
        if self.use_qdrant and self.qdrant_uploader:
            print("\n🔄 Flushing remaining embeddings to Qdrant...")
            self.qdrant_uploader.flush()
            
            # Display upload statistics
            stats = self.qdrant_uploader.get_statistics()
            print(f"📊 Upload Statistics:")
            print(f"   Total uploaded: {stats['total_uploaded']}")
            print(f"   Batches: {stats['batch_count']}")
        
        print(f"\n✅ All videos processed!")
        print(f"📊 Total embeddings collected: {len(self.all_embeddings)}")
    
    def perform_clustering(self) -> Dict[str, int]:
        """
        Perform cross-camera clustering on collected embeddings.
        
        Returns:
            Dict mapping point_id to global_id
        """
        if not self.use_qdrant:
            print("⚠️ Qdrant disabled - clustering not available")
            return {}
        
        try:
            # Perform clustering
            point_id_to_global_id = self.clusterer.perform_clustering()
            
            if not point_id_to_global_id:
                print("⚠️ No clusters formed")
                return {}
            
            # Update Qdrant with global IDs
            print("\n🔄 Updating Qdrant with global IDs...")
            self.qdrant_query_manager.update_global_ids(point_id_to_global_id)
            
            # Get and display summary
            summary = self.clusterer.get_cluster_summary(point_id_to_global_id)
            print(f"\n📊 Clustering Summary:")
            print(f"   Total points clustered: {summary['total_points']}")
            print(f"   Unique global IDs: {summary['total_global_ids']}")
            print(f"   Avg points per ID: {summary['avg_points_per_id']:.1f}")
            
            # Get camera statistics
            camera_stats = self.qdrant_query_manager.get_statistics_by_camera()
            print(f"\n📹 Camera Statistics:")
            for camera_id, stats in camera_stats.items():
                print(f"   Camera {camera_id}: {stats['total_points']} points, "
                      f"{stats['unique_tracks']} tracks, "
                      f"{stats['processed']} processed")
            
            return point_id_to_global_id
            
        except Exception as e:
            print(f"❌ Clustering error: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def save_results(self, output_path: str = "results.json"):
        """
        Save collected embeddings to file.
        
        Args:
            output_path: Output JSON file path
        """
        # Convert numpy arrays to lists for JSON serialization
        serializable_data = []
        for emb_data in self.all_embeddings:
            data_copy = emb_data.copy()
            if 'embedding' in data_copy and data_copy['embedding'] is not None:
                data_copy['embedding'] = data_copy['embedding'].tolist()
            serializable_data.append(data_copy)
        
        # Add session metadata
        output_data = {
            "session_id": self.video_session_id,
            "total_embeddings": len(serializable_data),
            "qdrant_enabled": self.use_qdrant,
            "embeddings": serializable_data
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"💾 Results saved to {output_path}")


def main():
    """
    Main entry point - parse arguments and run pipeline.
    """
    parser = argparse.ArgumentParser(
        description='Person ReID Pipeline v2 with Qdrant',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single video
  python main_pipeline.py --video1 path/to/video.mp4
  
  # Process two videos with clustering
  python main_pipeline.py --video1 cam1.mp4 --video2 cam2.mp4 --mode both
  
  # Disable Qdrant (local only)
  python main_pipeline.py --video1 v1.mp4 --no-qdrant
  
  # Use greedy clustering
  python main_pipeline.py --video1 v1.mp4 --video2 v2.mp4 --cluster-method greedy
        """
    )
    
    # Video sources
    parser.add_argument('--camera', type=int, default=1,
                       help='Default camera ID (default: 1)')
    parser.add_argument('--source', type=str, default='0',
                       help='Video source: 0 for webcam, or path to video')
    parser.add_argument('--video1', type=str,
                       help='Path to Camera 1 video')
    parser.add_argument('--video2', type=str,
                       help='Path to Camera 2 video')
    
    # Model parameters
    parser.add_argument('--yolo-model', type=str, default='yolov8m.pt',
                       help='Path to YOLO model weights')
    parser.add_argument('--reid-model', type=str, default='osnet_x1_0',
                       help='ReID model name')
    parser.add_argument('--pooling', type=int, default=10,
                       help='Number of frames for embedding pooling')
    
    # Processing options
    parser.add_argument('--mode', type=str, default='both',
                       choices=['process', 'cluster', 'both'],
                       help='Pipeline mode')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device for inference')
    parser.add_argument('--no-viz', action='store_true',
                       help='Disable visualization')
    
    # Qdrant options
    parser.add_argument('--no-qdrant', action='store_true',
                       help='Disable Qdrant integration')
    parser.add_argument('--cluster-method', type=str, 
                       default='connected_components',
                       choices=['connected_components', 'greedy'],
                       help='Clustering method')
    
    args = parser.parse_args()
    
    try:
        # Initialize main pipeline
        pipeline = MainPipeline(
            pooling_threshold=args.pooling,
            yolo_model=args.yolo_model,
            reid_model=args.reid_model,
            device=args.device,
            use_qdrant=not args.no_qdrant,
            clustering_method=args.cluster_method
        )
        
        # Determine video sources
        video_sources = {}
        
        if args.video1:
            video_sources[1] = args.video1
        if args.video2:
            video_sources[2] = args.video2
        
        if not video_sources:
            # Use default source
            video_source = int(args.source) if args.source.isdigit() else args.source
            video_sources[args.camera] = video_source
        
        # Process videos
        if args.mode in ['process', 'both']:
            pipeline.process_multiple_videos(
                video_sources=video_sources,
                visualize=not args.no_viz
            )
            
            # Save results
            pipeline.save_results('embeddings_output.json')
        
        # Perform clustering
        if args.mode in ['cluster', 'both']:
            pipeline.perform_clustering()
        
        print("\n✅ Pipeline completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Pipeline interrupted by user")
    except Exception as e:
        print(f"\n❌ Pipeline error: {e}")
        import traceback
        traceback.print_exc()


def main_pipeline(camera_a_file, camera_b_file, pooling_threshold=10, device='cpu'):
    """
    Streamlit-compatible function to process two camera videos with Qdrant.
    FIXED VERSION - Now properly flushes embeddings to Qdrant.
    
    Args:
        camera_a_file: Uploaded file object or path for Camera A
        camera_b_file: Uploaded file object or path for Camera B
        pooling_threshold: Number of frames for embedding pooling
        device: 'cpu' or 'cuda'
        
    Returns:
        Dict with results including embeddings, clustering, and statistics
    """
    import tempfile
    import os
    
    # Save uploaded files to temporary location
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Handle Camera A
        if hasattr(camera_a_file, 'read'):  # Streamlit UploadedFile
            camera_a_path = os.path.join(temp_dir, f"camera_a_{camera_a_file.name}")
            with open(camera_a_path, 'wb') as f:
                f.write(camera_a_file.read())
        else:
            camera_a_path = camera_a_file
        
        # Handle Camera B
        if hasattr(camera_b_file, 'read'):  # Streamlit UploadedFile
            camera_b_path = os.path.join(temp_dir, f"camera_b_{camera_b_file.name}")
            with open(camera_b_path, 'wb') as f:
                f.write(camera_b_file.read())
        else:
            camera_b_path = camera_b_file
        
        print("\n" + "="*60)
        print("🚀 Starting Streamlit Pipeline")
        print("="*60)
        
        # Initialize pipeline with Qdrant
        pipeline = MainPipeline(
            pooling_threshold=pooling_threshold,
            yolo_model='yolov8m.pt',
            reid_model='resnet50',
            device=device,
            use_qdrant=True,
            clustering_method='connected_components'
        )
        
        # Process videos
        video_sources = {
            1: camera_a_path,
            2: camera_b_path
        }
        
        print(f"\n📹 Processing {len(video_sources)} camera videos...")
        pipeline.process_multiple_videos(
            video_sources=video_sources,
            visualize=False  # No visualization in Streamlit
        )
        
        # **CRITICAL FIX**: Flush any remaining embeddings
        if pipeline.use_qdrant and pipeline.qdrant_uploader:
            print("\n🔄 Flushing final embeddings to Qdrant...")
            pipeline.qdrant_uploader.flush()
            
            # Get and display upload statistics
            stats = pipeline.qdrant_uploader.get_statistics()
            print(f"\n📊 Upload Complete:")
            print(f"   Total uploaded: {stats['total_uploaded']}")
            print(f"   Batches: {stats['batch_count']}")
            print(f"   Pending: {stats['pending_in_batch']}")
        
        # Perform clustering
        print("\n🔗 Starting clustering...")
        point_id_to_global_id = pipeline.perform_clustering()
        
        # Get final statistics
        camera_stats = {}
        if pipeline.use_qdrant and pipeline.qdrant_query_manager:
            camera_stats = pipeline.qdrant_query_manager.get_statistics_by_camera()
        
        # Prepare results
        results = {
            'status': 'success',
            'session_id': pipeline.video_session_id,
            'total_embeddings': len(pipeline.all_embeddings),
            'camera_a_embeddings': [e for e in pipeline.all_embeddings if e['camera_id'] == 1],
            'camera_b_embeddings': [e for e in pipeline.all_embeddings if e['camera_id'] == 2],
            'clustering_results': {
                'total_clusters': len(set(point_id_to_global_id.values())) if point_id_to_global_id else 0,
                'total_clustered_points': len(point_id_to_global_id)
            },
            'camera_statistics': camera_stats,
            'qdrant_enabled': pipeline.use_qdrant,
            'upload_statistics': stats if pipeline.use_qdrant else None
        }
        
        print("\n✅ Pipeline completed successfully!")
        print(f"📊 Total embeddings: {results['total_embeddings']}")
        print(f"🔗 Total clusters: {results['clustering_results']['total_clusters']}")
        
        return results
        
    except Exception as e:
        import traceback
        error_msg = str(e)
        error_trace = traceback.format_exc()
        
        print(f"\n❌ Pipeline Error: {error_msg}")
        print(error_trace)
        
        return {
            'status': 'error',
            'message': error_msg,
            'traceback': error_trace,
            'total_embeddings': 0
        }
    finally:
        # Cleanup temporary files
        import shutil
        try:
            shutil.rmtree(temp_dir)
            print("\n🧹 Cleaned up temporary files")
        except Exception as e:
            print(f"⚠️ Cleanup warning: {e}")


if __name__ == "__main__":
    main()