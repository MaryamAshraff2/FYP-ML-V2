# File path: core/tracking/track_utils.py

# ===========================
# Imports
# ===========================
import numpy as np
from typing import List, Dict

# ===========================
# Track Utilities
# ===========================

def filter_tracks_by_confidence(tracks_metadata: List[Dict], min_confidence: float = 0.3) -> List[Dict]:
    """
    Filter out tracks with low confidence or inactive status.

    Args:
        tracks_metadata (List[Dict]): List of track metadata dictionaries
        min_confidence (float): Minimum confidence threshold

    Returns:
        List[Dict]: Filtered list of track metadata
    """
    filtered = []
    for track in tracks_metadata:
        # Keep if confidence is above threshold and status is active
        if track.get("confidence", 0) >= min_confidence and track.get("status", "active") == "active":
            filtered.append(track)
    return filtered



def extract_final_embeddings(tracks_metadata: List[Dict]) -> Dict[str, np.ndarray]:
    """
    Extract the main pooled embeddings (smooth_feat) from each track.

    Args:
        tracks_metadata (List[Dict]): List of track metadata dictionaries

    Returns:
        Dict[str, np.ndarray]: Mapping from local_track_id to main embedding vector
    """
    embeddings = {}
    for track in tracks_metadata:
        local_id = track["local_track_id"]
        embedding = track.get("embedding")
        if embedding is not None:
            embeddings[local_id] = embedding
    return embeddings


def update_track_status(tracks_metadata: List[Dict], active_ids: List[str]) -> None:
    """
    Update track status (active/inactive) based on current active track IDs.

    Args:
        tracks_metadata (List[Dict]): List of track metadata dictionaries
        active_ids (List[str]): List of local_track_ids that are currently active
    """
    for track in tracks_metadata:
        if track["local_track_id"] in active_ids:
            track["status"] = "active"
        else:
            track["status"] = "inactive"


def generate_metadata_summary(tracks_metadata: List[Dict]) -> Dict:
    """
    Create a summary of the current tracks for logging or visualization.

    Args:
        tracks_metadata (List[Dict]): List of track metadata dictionaries

    Returns:
        Dict: Summary containing number of active tracks, average embedding dim, etc.
    """
    num_active = sum(1 for t in tracks_metadata if t.get("status") == "active")
    avg_emb_dim = int(np.mean([t.get("embedding_dim", 0) for t in tracks_metadata])) if tracks_metadata else 0

    summary = {
        "num_active_tracks": num_active,
        "average_embedding_dim": avg_emb_dim,
        "total_tracks": len(tracks_metadata)
    }
    return summary
