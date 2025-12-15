"""
Test script to verify Qdrant setup and integration
Run this before processing videos to ensure everything works
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from database.qdrant.client import QdrantClientManager, get_qdrant_client
from database.qdrant.collections import (
    setup_qdrant_collection,
    get_collection_info,
    verify_collection_existence,
    delete_collection
)
from database.qdrant.uploader import QdrantUploader
from database.qdrant.queries import QdrantQueryManager


def test_connection():
    """Test 1: Qdrant connection"""
    print("\n" + "="*60)
    print("Test 1: Testing Qdrant Connection")
    print("="*60)
    
    try:
        manager = QdrantClientManager()
        client = manager.client
        
        # Try to list collections
        collections = client.get_collections()
        print(f"✅ Connected to Qdrant successfully!")
        print(f"📊 Found {len(collections.collections)} existing collections")
        
        return True
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False


def test_collection_creation():
    """Test 2: Collection creation"""
    print("\n" + "="*60)
    print("Test 2: Testing Collection Creation")
    print("="*60)
    
    try:
        # Delete if exists
        if collection_exists():
            print("🗑️ Deleting existing test collection...")
            delete_collection()
        
        # Create new collection
        success = initialize_collection(recreate=True)
        
        if success:
            print("✅ Collection created successfully!")
            
            # Get info
            info = get_collection_info()
            print(f"📊 Collection Info:")
            print(f"   Name: {info.get('name')}")
            print(f"   Vector size: {info.get('vector_size')}")
            print(f"   Distance: {info.get('distance')}")
            
            return True
        else:
            print("❌ Collection creation failed")
            return False
            
    except Exception as e:
        print(f"❌ Collection creation error: {e}")
        return False


def test_upload():
    """Test 3: Upload embeddings"""
    print("\n" + "="*60)
    print("Test 3: Testing Embedding Upload")
    print("="*60)
    
    try:
        uploader = QdrantUploader(batch_size=5)
        
        # Create dummy embeddings
        print("📤 Creating test embeddings...")
        test_embeddings = []
        
        for camera_id in [1, 2]:
            for track_id in range(3):
                for frame in range(5):
                    embedding = np.random.randn(256).tolist()  # 256-dim
                    
                    test_embeddings.append({
                        'camera_id': camera_id,
                        'local_track_id': track_id,
                        'frame_number': frame * 10,
                        'bbox': [100, 100, 200, 300],
                        'embedding': embedding,
                        'confidence': 0.9,
                        'global_id': -1
                    })
        
        # Upload
        print(f"📤 Uploading {len(test_embeddings)} test embeddings...")
        for camera_id in [1, 2]:
            camera_embeddings = [e for e in test_embeddings if e['camera_id'] == camera_id]
            uploader.upload_embeddings(
                camera_id=camera_id,
                pooled_embeddings=camera_embeddings,
                video_session_id="test-session"
            )
        
        # Flush remaining
        uploader.flush()
        
        # Check upload stats
        stats = uploader.get_statistics()
        print(f"✅ Upload complete!")
        print(f"   Total uploaded: {stats['total_uploaded']}")
        print(f"   Batches: {stats['batch_count']}")
        
        # Verify in collection
        info = get_collection_info()
        print(f"📊 Collection now has {info.get('points_count', 0)} points")
        
        return True
        
    except Exception as e:
        print(f"❌ Upload error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_search():
    """Test 4: Similarity search"""
    print("\n" + "="*60)
    print("Test 4: Testing Similarity Search")
    print("="*60)
    
    try:
        query_mgr = QdrantQueryManager()
        
        # Create query vector
        query_vector = np.random.randn(256).tolist()
        
        print("🔍 Searching for similar embeddings...")
        results = query_mgr.search_similar_embeddings(
            query_vector=query_vector,
            camera_id=1,
            exclude_camera=True,
            top_k=5,
            score_threshold=0.0  # Low threshold for test
        )
        
        print(f"✅ Search complete!")
        print(f"   Found {len(results)} results")
        
        if results:
            print(f"   Top result similarity: {results[0].score:.3f}")
            print(f"   Top result camera: {results[0].payload.get('camera_id')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Search error: {e}")
        return False


def test_clustering():
    """Test 5: Clustering"""
    print("\n" + "="*60)
    print("Test 5: Testing Clustering")
    print("="*60)
    
    try:
        from core.identity.clustering import IdentityClusterer
        
        clusterer = IdentityClusterer(method='connected_components')
        
        print("🔗 Performing clustering on test data...")
        point_id_to_global_id = clusterer.perform_clustering()
        
        if point_id_to_global_id:
            print(f"✅ Clustering complete!")
            print(f"   Clustered points: {len(point_id_to_global_id)}")
            print(f"   Unique global IDs: {len(set(point_id_to_global_id.values()))}")
            
            # Get summary
            summary = clusterer.get_cluster_summary(point_id_to_global_id)
            print(f"   Avg points per ID: {summary['avg_points_per_id']:.1f}")
            
            return True
        else:
            print("⚠️ No clusters formed (may need more similar embeddings)")
            return True  # Not necessarily a failure
            
    except Exception as e:
        print(f"❌ Clustering error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_statistics():
    """Test 6: Get statistics"""
    print("\n" + "="*60)
    print("Test 6: Testing Statistics Retrieval")
    print("="*60)
    
    try:
        query_mgr = QdrantQueryManager()
        
        # Get camera stats
        stats = query_mgr.get_statistics_by_camera()
        
        print("📊 Camera Statistics:")
        for camera_id, camera_stats in stats.items():
            print(f"   Camera {camera_id}:")
            print(f"      Total points: {camera_stats['total_points']}")
            print(f"      Unique tracks: {camera_stats['unique_tracks']}")
            print(f"      Processed: {camera_stats['processed']}")
            print(f"      Unprocessed: {camera_stats['unprocessed']}")
        
        print("✅ Statistics retrieved successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Statistics error: {e}")
        return False


def cleanup_test_data():
    """Cleanup test collection"""
    print("\n" + "="*60)
    print("Cleanup: Removing Test Data")
    print("="*60)
    
    try:
        if collection_exists():
            print("🗑️ Deleting test collection...")
            delete_collection()
            print("✅ Test data cleaned up")
        else:
            print("✅ No test data to clean up")
        return True
    except Exception as e:
        print(f"⚠️ Cleanup warning: {e}")
        return True


def main():
    """Run all tests"""
    print("\n" + "🧪 "*20)
    print("QDRANT INTEGRATION TEST SUITE")
    print("🧪 "*20)
    
    tests = [
        ("Connection", test_connection),
        ("Collection Creation", test_collection_creation),
        ("Upload", test_upload),
        ("Search", test_search),
        ("Clustering", test_clustering),
        ("Statistics", test_statistics),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Cleanup
    cleanup_test_data()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Qdrant integration is ready.")
        return 0
    else:
        print(f"\n⚠️ {total - passed} test(s) failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)