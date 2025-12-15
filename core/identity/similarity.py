"""
Similarity Module - Distance metrics and matching logic
"""

import numpy as np
from typing import List, Tuple, Dict
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity


def compute_cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Compute cosine similarity between two embeddings.
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
        
    Returns:
        float: Cosine similarity score (0 to 1, higher is more similar)
    """
    # Handle list inputs
    if isinstance(embedding1, list):
        embedding1 = np.array(embedding1)
    if isinstance(embedding2, list):
        embedding2 = np.array(embedding2)
    
    # Normalize embeddings
    embedding1 = embedding1 / (np.linalg.norm(embedding1) + 1e-8)
    embedding2 = embedding2 / (np.linalg.norm(embedding2) + 1e-8)
    
    # Compute cosine similarity (1 - cosine distance)
    similarity = 1 - cosine(embedding1, embedding2)
    
    return float(similarity)


def compute_cosine_distance(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Compute cosine distance between two embeddings.
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
        
    Returns:
        float: Cosine distance (0 to 2, lower is more similar)
    """
    return 1.0 - compute_cosine_similarity(embedding1, embedding2)


def compute_pairwise_similarities(embeddings: List[np.ndarray]) -> np.ndarray:
    """
    Compute pairwise cosine similarities for a list of embeddings.
    
    Args:
        embeddings: List of embedding vectors
        
    Returns:
        np.ndarray: Pairwise similarity matrix (n x n)
    """
    embeddings_array = np.array(embeddings)
    
    # Normalize embeddings
    norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
    embeddings_normalized = embeddings_array / (norms + 1e-8)
    
    # Compute pairwise cosine similarities
    similarity_matrix = cosine_similarity(embeddings_normalized)
    
    return similarity_matrix


def find_similar_pairs(
    embeddings: List[np.ndarray],
    metadata: List[Dict],
    threshold: float = 0.7,
    cross_camera_only: bool = True
) -> List[Tuple[int, int, float]]:
    """
    Find pairs of similar embeddings above threshold.
    
    Args:
        embeddings: List of embedding vectors
        metadata: List of metadata dicts (must contain 'camera_id')
        threshold: Similarity threshold (0 to 1)
        cross_camera_only: Only match across different cameras
        
    Returns:
        List of (idx1, idx2, similarity) tuples
    """
    if len(embeddings) != len(metadata):
        raise ValueError("Embeddings and metadata must have same length")
    
    # Compute similarity matrix
    similarity_matrix = compute_pairwise_similarities(embeddings)
    
    similar_pairs = []
    
    # Find pairs above threshold
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            similarity = similarity_matrix[i, j]
            
            # Check threshold
            if similarity < threshold:
                continue
            
            # Check cross-camera constraint
            if cross_camera_only:
                camera_i = metadata[i].get('camera_id')
                camera_j = metadata[j].get('camera_id')
                
                if camera_i == camera_j:
                    continue
            
            similar_pairs.append((i, j, float(similarity)))
    
    # Sort by similarity (descending)
    similar_pairs.sort(key=lambda x: x[2], reverse=True)
    
    return similar_pairs


def is_match(
    embedding1: np.ndarray,
    embedding2: np.ndarray,
    threshold: float = 0.7
) -> bool:
    """
    Check if two embeddings match based on threshold.
    
    Args:
        embedding1: First embedding
        embedding2: Second embedding
        threshold: Similarity threshold
        
    Returns:
        bool: True if match, False otherwise
    """
    similarity = compute_cosine_similarity(embedding1, embedding2)
    return similarity >= threshold


def compute_cluster_centroid(embeddings: List[np.ndarray]) -> np.ndarray:
    """
    Compute the centroid (mean) of a cluster of embeddings.
    
    Args:
        embeddings: List of embedding vectors
        
    Returns:
        np.ndarray: Centroid embedding
    """
    embeddings_array = np.array(embeddings)
    centroid = np.mean(embeddings_array, axis=0)
    
    # Normalize centroid
    centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
    
    return centroid


def rank_matches(
    query_embedding: np.ndarray,
    candidate_embeddings: List[np.ndarray],
    candidate_metadata: List[Dict],
    top_k: int = 10
) -> List[Tuple[int, float, Dict]]:
    """
    Rank candidate embeddings by similarity to query.
    
    Args:
        query_embedding: Query embedding vector
        candidate_embeddings: List of candidate embeddings
        candidate_metadata: Metadata for candidates
        top_k: Number of top matches to return
        
    Returns:
        List of (idx, similarity, metadata) tuples
    """
    if not candidate_embeddings:
        return []
    
    # Compute similarities
    similarities = [
        compute_cosine_similarity(query_embedding, emb)
        for emb in candidate_embeddings
    ]
    
    # Create ranked list
    ranked = [
        (idx, sim, candidate_metadata[idx])
        for idx, sim in enumerate(similarities)
    ]
    
    # Sort by similarity
    ranked.sort(key=lambda x: x[1], reverse=True)
    
    # Return top_k
    return ranked[:top_k]