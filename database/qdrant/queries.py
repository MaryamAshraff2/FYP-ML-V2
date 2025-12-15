"""
Qdrant Queries - Search and retrieve embeddings for clustering
"""

from qdrant_client.http.models import ScoredPoint, FieldCondition, MatchValue, Filter, Record
from database.qdrant.client import get_qdrant_client, get_qdrant_config
from typing import List, Dict, Optional, Tuple
import numpy as np


class QdrantQueryManager:
    """
    Manages similarity search and retrieval from Qdrant.
    Handles cross-camera matching queries.
    """
    
    def __init__(self):
        """Initialize query manager."""
        self.client = get_qdrant_client()
        self.config = get_qdrant_config()
        
        # Access nested config sections
        self.collection_name = self.config.get('collection', {}).get('name', 'person_reid_embeddings')
        search_config = self.config.get('search', {})
        self.similarity_threshold = search_config.get('similarity_threshold', 0.65)
        self.top_k = search_config.get('top_k', 10)
        self.cross_camera_only = search_config.get('cross_camera_only', True)
    
    def search_similar_embeddings(
        self,
        query_vector: List[float],
        camera_id: Optional[int] = None,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        exclude_camera: bool = True,
        only_unprocessed: bool = True
    ) -> List[ScoredPoint]:
        """
        Search for similar embeddings in Qdrant.
        
        Args:
            query_vector: Embedding vector to search for
            camera_id: Source camera ID (for cross-camera filtering)
            top_k: Number of results to return (default from config)
            score_threshold: Minimum similarity score (default from config)
            exclude_camera: If True, exclude results from same camera
            only_unprocessed: If True, only search unprocessed points (global_id=-1)
            
        Returns:
            List of ScoredPoint with similarity scores
        """
        if top_k is None:
            top_k = self.top_k
        
        if score_threshold is None:
            score_threshold = self.similarity_threshold
        
        # Build filter conditions
        must_conditions = []
        must_not_conditions = []
        
        # Filter unprocessed points
        if only_unprocessed:
            must_conditions.append(
                FieldCondition(
                    key="global_id",
                    match=MatchValue(value=-1)
                )
            )
        
        # Cross-camera filtering
        if exclude_camera and camera_id is not None:
            must_not_conditions.append(
                FieldCondition(
                    key="camera_id",
                    match=MatchValue(value=camera_id)
                )
            )
        
        # Create filter
        query_filter = None
        if must_conditions or must_not_conditions:
            query_filter = Filter(
                must=must_conditions if must_conditions else None,
                must_not=must_not_conditions if must_not_conditions else None
            )
        
        # Execute search
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=top_k,
                score_threshold=score_threshold
            )
            
            return results
            
        except Exception as e:
            print(f"❌ Search error: {e}")
            return []
    
    def get_all_unprocessed_points(
        self,
        camera_id: Optional[int] = None,
        limit: int = 10000
    ) -> List[Record]:
        """
        Retrieve all unprocessed points (global_id=-1) from Qdrant.
        Used for batch clustering.
        
        Args:
            camera_id: Filter by specific camera (None for all cameras)
            limit: Maximum points to retrieve
            
        Returns:
            List of Record objects with vectors and payloads
        """
        # Build filter
        must_conditions = [
            FieldCondition(
                key="global_id",
                match=MatchValue(value=-1)
            )
        ]
        
        if camera_id is not None:
            must_conditions.append(
                FieldCondition(
                    key="camera_id",
                    match=MatchValue(value=camera_id)
                )
            )
        
        query_filter = Filter(must=must_conditions)
        
        # Scroll through all matching points
        try:
            points, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=query_filter,
                limit=limit,
                with_vectors=True,
                with_payload=True
            )
            
            print(f"📊 Retrieved {len(points)} unprocessed points")
            return points
            
        except Exception as e:
            print(f"❌ Error retrieving points: {e}")
            return []
    
    def get_points_by_session(
        self,
        video_session_id: str,
        with_vectors: bool = True
    ) -> List[Record]:
        """
        Retrieve all points from a specific video processing session.
        
        Args:
            video_session_id: Session UUID
            with_vectors: Include embedding vectors
            
        Returns:
            List of Record objects
        """
        query_filter = Filter(
            must=[
                FieldCondition(
                    key="video_session_id",
                    match=MatchValue(value=video_session_id)
                )
            ]
        )
        
        try:
            points, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=query_filter,
                limit=10000,
                with_vectors=with_vectors,
                with_payload=True
            )
            
            return points
            
        except Exception as e:
            print(f"❌ Error retrieving session points: {e}")
            return []
    
    def find_cross_camera_matches(
        self,
        camera_id: int,
        local_track_id: int,
        query_vector: List[float]
    ) -> List[Tuple[ScoredPoint, float]]:
        """
        Find potential matches for a person in other cameras.
        
        Args:
            camera_id: Source camera ID
            local_track_id: Source track ID
            query_vector: Embedding vector
            
        Returns:
            List of (ScoredPoint, similarity_score) tuples
        """
        results = self.search_similar_embeddings(
            query_vector=query_vector,
            camera_id=camera_id,
            exclude_camera=True,
            only_unprocessed=True
        )
        
        # Format results
        matches = [(point, point.score) for point in results]
        
        if matches:
            print(f"🔍 Found {len(matches)} cross-camera matches for "
                  f"Camera {camera_id}, Track {local_track_id}")
        
        return matches
    
    def get_statistics_by_camera(self) -> Dict[int, Dict]:
        """
        Get statistics for each camera.
        
        Returns:
            Dict mapping camera_id to statistics
        """
        stats = {}
        
        try:
            # Get all points
            all_points, _ = self.client.scroll(
                collection_name=self.collection_name,
                limit=10000,
                with_vectors=False,
                with_payload=True
            )
            
            # Group by camera
            for point in all_points:
                camera_id = point.payload.get('camera_id')
                if camera_id not in stats:
                    stats[camera_id] = {
                        'total_points': 0,
                        'processed': 0,
                        'unprocessed': 0,
                        'unique_tracks': set()
                    }
                
                stats[camera_id]['total_points'] += 1
                
                if point.payload.get('global_id', -1) != -1:
                    stats[camera_id]['processed'] += 1
                else:
                    stats[camera_id]['unprocessed'] += 1
                
                stats[camera_id]['unique_tracks'].add(point.payload.get('local_track_id'))
            
            # Convert sets to counts
            for camera_id in stats:
                stats[camera_id]['unique_tracks'] = len(stats[camera_id]['unique_tracks'])
            
            return stats
            
        except Exception as e:
            print(f"❌ Error getting statistics: {e}")
            return {}
    
    def update_global_ids(self, point_id_to_global_id: Dict[str, int]) -> bool:
        """
        Update global_id field for multiple points after clustering.
        
        Args:
            point_id_to_global_id: Mapping of point IDs to assigned global IDs
            
        Returns:
            bool: True if successful
        """
        try:
            # Batch update payloads
            for point_id, global_id in point_id_to_global_id.items():
                self.client.set_payload(
                    collection_name=self.collection_name,
                    payload={
                        "global_id": global_id,
                        "processed": True
                    },
                    points=[point_id]
                )
            
            print(f"✅ Updated {len(point_id_to_global_id)} points with global IDs")
            return True
            
        except Exception as e:
            print(f"❌ Error updating global IDs: {e}")
            return False