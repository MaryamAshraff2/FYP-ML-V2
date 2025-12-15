"""
Clustering Module - Cross-camera identity clustering
"""

import numpy as np
from typing import List, Dict, Set, Tuple
from collections import defaultdict
from database.qdrant.queries import QdrantQueryManager
from database.qdrant.client import get_qdrant_config
from core.identity.similarity import (
    compute_cosine_similarity,
    compute_pairwise_similarities,
    find_similar_pairs
)


class IdentityClusterer:
    """
    Performs cross-camera clustering to assign global IDs.
    Implements multiple clustering strategies.
    """
    
    def __init__(self, method: str = "connected_components"):
        """
        Initialize clusterer.
        
        Args:
            method: Clustering method ('connected_components', 'greedy', 'dbscan')
        """
        self.config = get_qdrant_config()
        self.query_manager = QdrantQueryManager()
        
        self.method = method or self.config['clustering']['method']
        self.distance_threshold = self.config['clustering']['distance_threshold']
        self.similarity_threshold = 1.0 - self.distance_threshold
        self.min_cluster_size = self.config['clustering']['min_cluster_size']
        self.min_cameras = self.config['clustering']['min_cameras']
        
        print(f"🔍 Clusterer initialized (method={self.method}, "
              f"threshold={self.similarity_threshold:.2f})")
    
    def cluster_connected_components(
        self,
        embeddings: List[np.ndarray],
        metadata: List[Dict]
    ) -> Dict[int, List[int]]:
        """
        Cluster using connected components (Union-Find).
        Two embeddings are connected if similarity > threshold.
        
        Args:
            embeddings: List of embedding vectors
            metadata: List of metadata dicts with 'camera_id'
            
        Returns:
            Dict mapping cluster_id to list of point indices
        """
        n = len(embeddings)
        
        # Initialize Union-Find
        parent = list(range(n))
        rank = [0] * n
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            root_x = find(x)
            root_y = find(y)
            
            if root_x == root_y:
                return
            
            if rank[root_x] < rank[root_y]:
                parent[root_x] = root_y
            elif rank[root_x] > rank[root_y]:
                parent[root_y] = root_x
            else:
                parent[root_y] = root_x
                rank[root_x] += 1
        
        # Find similar pairs and union them
        similar_pairs = find_similar_pairs(
            embeddings,
            metadata,
            threshold=self.similarity_threshold,
            cross_camera_only=True
        )
        
        print(f"🔗 Found {len(similar_pairs)} similar pairs")
        
        for i, j, similarity in similar_pairs:
            union(i, j)
        
        # Group by root (cluster)
        clusters = defaultdict(list)
        for i in range(n):
            root = find(i)
            clusters[root].append(i)
        
        # Filter clusters by minimum size and camera diversity
        valid_clusters = {}
        cluster_id = 0
        
        for root, indices in clusters.items():
            # Check minimum size
            if len(indices) < self.min_cluster_size:
                continue
            
            # Check camera diversity
            camera_ids = set(metadata[i]['camera_id'] for i in indices)
            if len(camera_ids) < self.min_cameras:
                continue
            
            valid_clusters[cluster_id] = indices
            cluster_id += 1
        
        print(f"✅ Created {len(valid_clusters)} valid clusters")
        
        return valid_clusters
    
    def cluster_greedy(
        self,
        embeddings: List[np.ndarray],
        metadata: List[Dict]
    ) -> Dict[int, List[int]]:
        """
        Greedy clustering: assign each embedding to nearest cluster or create new.
        Fast but less accurate than connected components.
        
        Args:
            embeddings: List of embedding vectors
            metadata: List of metadata dicts
            
        Returns:
            Dict mapping cluster_id to list of point indices
        """
        clusters = {}
        cluster_centroids = {}
        next_cluster_id = 0
        
        # Sort by camera_id for better cross-camera matching
        indices = list(range(len(embeddings)))
        indices.sort(key=lambda i: metadata[i]['camera_id'])
        
        for idx in indices:
            embedding = embeddings[idx]
            camera_id = metadata[idx]['camera_id']
            
            best_cluster = None
            best_similarity = 0.0
            
            # Find best matching cluster
            for cluster_id, cluster_indices in clusters.items():
                # Skip if same camera already in cluster
                cluster_cameras = set(metadata[i]['camera_id'] for i in cluster_indices)
                if camera_id in cluster_cameras:
                    continue
                
                # Compute similarity to cluster centroid
                centroid = cluster_centroids[cluster_id]
                similarity = compute_cosine_similarity(embedding, centroid)
                
                if similarity > best_similarity and similarity >= self.similarity_threshold:
                    best_similarity = similarity
                    best_cluster = cluster_id
            
            # Assign to cluster or create new
            if best_cluster is not None:
                clusters[best_cluster].append(idx)
                
                # Update centroid
                cluster_embeddings = [embeddings[i] for i in clusters[best_cluster]]
                cluster_centroids[best_cluster] = np.mean(cluster_embeddings, axis=0)
                cluster_centroids[best_cluster] /= (np.linalg.norm(cluster_centroids[best_cluster]) + 1e-8)
            else:
                # Create new cluster
                clusters[next_cluster_id] = [idx]
                cluster_centroids[next_cluster_id] = embedding
                next_cluster_id += 1
        
        # Filter by minimum requirements
        valid_clusters = {}
        cluster_id = 0
        
        for indices in clusters.values():
            if len(indices) < self.min_cluster_size:
                continue
            
            camera_ids = set(metadata[i]['camera_id'] for i in indices)
            if len(camera_ids) < self.min_cameras:
                continue
            
            valid_clusters[cluster_id] = indices
            cluster_id += 1
        
        print(f"✅ Created {len(valid_clusters)} valid clusters (greedy)")
        
        return valid_clusters
    
    def perform_clustering(self) -> Dict[str, int]:
        """
        Main clustering workflow: retrieve unprocessed points and cluster.
        
        Returns:
            Dict mapping point_id to assigned global_id
        """
        print(f"\n{'='*60}")
        print(f"🔍 Starting Cross-Camera Clustering")
        print(f"{'='*60}\n")
        
        # 1. Retrieve all unprocessed points
        print("📥 Retrieving unprocessed points from Qdrant...")
        points = self.query_manager.get_all_unprocessed_points()
        
        if not points:
            print("⚠️ No unprocessed points found")
            return {}
        
        print(f"📊 Retrieved {len(points)} unprocessed points")
        
        # 2. Extract embeddings and metadata
        embeddings = []
        metadata = []
        point_ids = []
        
        for point in points:
            embeddings.append(np.array(point.vector))
            metadata.append(point.payload)
            point_ids.append(point.id)
        
        # 3. Perform clustering
        print(f"🔄 Clustering with method: {self.method}")
        
        if self.method == "connected_components":
            clusters = self.cluster_connected_components(embeddings, metadata)
        elif self.method == "greedy":
            clusters = self.cluster_greedy(embeddings, metadata)
        else:
            raise ValueError(f"Unknown clustering method: {self.method}")
        
        # 4. Assign global IDs
        point_id_to_global_id = {}
        next_global_id = 1
        
        for cluster_id, indices in clusters.items():
            global_id = next_global_id
            next_global_id += 1
            
            for idx in indices:
                point_id = point_ids[idx]
                point_id_to_global_id[point_id] = global_id
            
            # Print cluster info
            cameras = set(metadata[i]['camera_id'] for i in indices)
            print(f"   Cluster {cluster_id} → Global ID {global_id}: "
                  f"{len(indices)} points from cameras {sorted(cameras)}")
        
        print(f"\n✅ Assigned {len(point_id_to_global_id)} points to "
              f"{len(clusters)} global identities")
        
        return point_id_to_global_id
    
    def get_cluster_summary(
        self,
        point_id_to_global_id: Dict[str, int]
    ) -> Dict:
        """
        Generate summary statistics for clustering results.
        
        Args:
            point_id_to_global_id: Mapping from point IDs to global IDs
            
        Returns:
            Dict with clustering statistics
        """
        if not point_id_to_global_id:
            return {
                'total_points': 0,
                'total_global_ids': 0,
                'avg_points_per_id': 0
            }
        
        # Count occurrences
        global_id_counts = defaultdict(int)
        for global_id in point_id_to_global_id.values():
            global_id_counts[global_id] += 1
        
        summary = {
            'total_points': len(point_id_to_global_id),
            'total_global_ids': len(global_id_counts),
            'avg_points_per_id': len(point_id_to_global_id) / len(global_id_counts),
            'min_points_per_id': min(global_id_counts.values()),
            'max_points_per_id': max(global_id_counts.values())
        }
        
        return summary