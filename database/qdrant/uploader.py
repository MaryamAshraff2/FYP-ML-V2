# =============================
# database/qdrant/uploader.py
# Qdrant Embedding Uploader - COMPLETE FIXED VERSION
# =============================

from qdrant_client.http.models import PointStruct, UpdateStatus
from database.qdrant.client import get_qdrant_client, get_qdrant_config
import logging
from typing import List, Dict, Any, Optional
import uuid

logger = logging.getLogger(__name__)


class QdrantUploader:
    """
    Handles batch uploading of embeddings to Qdrant.
    """

    def __init__(self):
        """
        Initializes the uploader with Qdrant client and configuration.
        """
        self.client = get_qdrant_client()
        self.config = get_qdrant_config()
        self.collection_name = self.config.get('collection', {}).get('name', 'person_reid_embeddings')

        # Upload settings
        upload_config = self.config.get('upload', {})
        self.batch_size = upload_config.get('batch_size', 50)
        self.retry_count = upload_config.get('retry_count', 3)
        self.retry_delay = upload_config.get('retry_delay', 2)
        self.wait_until_indexed = upload_config.get('wait', True)
        
        # Internal batch buffer
        self._batch_embeddings = []
        self._batch_payloads = []
        self._upload_count = 0
        self._batch_count = 0

    def upload_embeddings(
        self,
        camera_id: int,
        pooled_embeddings: List[Dict],
        video_session_id: str
    ) -> List[str]:
        """
        Upload pooled embeddings from video pipeline (signature matching main_pipeline.py).
        
        Args:
            camera_id: Source camera ID
            pooled_embeddings: List of dicts with 'embedding' and metadata
            video_session_id: Session UUID
            
        Returns:
            List of uploaded point IDs
        """
        if not pooled_embeddings:
            return []
        
        # Extract embeddings and create payloads
        embeddings = []
        payloads = []
        
        for emb_data in pooled_embeddings:
            embedding = emb_data.get('embedding')
            if embedding is None:
                logger.warning(f"Skipping embedding with no vector data")
                continue
            
            # Convert numpy array to list if needed
            if hasattr(embedding, 'tolist'):
                embedding = embedding.tolist()
            
            embeddings.append(embedding)
            
            # Create payload
            payload = {
                'camera_id': camera_id,
                'local_track_id': emb_data.get('local_track_id', -1),
                'global_id': -1,  # Unprocessed
                'video_session_id': video_session_id,
                'timestamp': emb_data.get('timestamp', 0),
                'frame_number': emb_data.get('frame_number', 0),
                'bbox': emb_data.get('bbox', [0, 0, 0, 0]),
                'processed': False
            }
            payloads.append(payload)
        
        # Add to batch
        self._batch_embeddings.extend(embeddings)
        self._batch_payloads.extend(payloads)
        
        logger.info(f"📦 Added {len(embeddings)} embeddings to batch (total: {len(self._batch_embeddings)})")
        
        # Upload if batch is full
        uploaded_ids = []
        if len(self._batch_embeddings) >= self.batch_size:
            uploaded_ids = self._upload_batch()
        
        return uploaded_ids

    def _upload_batch(self) -> List[str]:
        """
        Upload current batch to Qdrant.
        
        Returns:
            List of uploaded point IDs
        """
        if not self._batch_embeddings:
            return []
        
        # Create points
        points = []
        for i, (emb, payload) in enumerate(zip(self._batch_embeddings, self._batch_payloads)):
            point_id = str(uuid.uuid4())
            points.append(
                PointStruct(
                    id=point_id,
                    vector=emb,
                    payload=payload
                )
            )
        
        # Upload to Qdrant
        uploaded_ids = []
        try:
            operation_info = self.client.upsert(
                collection_name=self.collection_name,
                wait=self.wait_until_indexed,
                points=points
            )
            
            if operation_info.status == UpdateStatus.COMPLETED:
                uploaded_ids = [point.id for point in points]
                self._upload_count += len(points)
                self._batch_count += 1
                logger.info(f"⬆️ Batch {self._batch_count}: Uploaded {len(points)} points to '{self.collection_name}'")
            else:
                logger.warning(f"⚠️ Upload operation status: {operation_info.status}")
                
        except Exception as e:
            logger.error(f"❌ Error uploading batch to '{self.collection_name}': {e}")
        
        # Clear batch
        self._batch_embeddings = []
        self._batch_payloads = []
        
        return uploaded_ids

    def flush(self) -> List[str]:
        """
        Force upload of remaining embeddings in batch.
        Call this at the end of processing.
        
        Returns:
            List of uploaded point IDs
        """
        if not self._batch_embeddings:
            logger.info("📭 No embeddings to flush")
            return []
        
        logger.info(f"🔄 Flushing remaining {len(self._batch_embeddings)} embeddings...")
        return self._upload_batch()

    def get_statistics(self) -> Dict[str, int]:
        """
        Get upload statistics.
        
        Returns:
            Dict with upload stats
        """
        return {
            'total_uploaded': self._upload_count,
            'batch_count': self._batch_count,
            'pending_in_batch': len(self._batch_embeddings)
        }


# Module-level functions
def get_uploader() -> QdrantUploader:
    """Get a QdrantUploader instance."""
    return QdrantUploader()


def upload_from_callback(data_generator, batch_size: Optional[int] = None) -> int:
    """
    Upload embeddings using a callback generator (for backward compatibility).
    
    Args:
        data_generator: Generator that yields {'embedding': [...], 'payload': {...}}
        batch_size: Override default batch size
        
    Returns:
        Total number of points uploaded
    """
    uploader = QdrantUploader()
    
    if batch_size is None:
        batch_size = uploader.batch_size
    
    total_uploaded = 0
    batch_embeddings = []
    batch_payloads = []
    
    while True:
        try:
            data = data_generator()
            if data is None:
                break
            
            embedding = data.get('embedding')
            payload = data.get('payload')
            
            if embedding is None or payload is None:
                continue
            
            batch_embeddings.append(embedding)
            batch_payloads.append(payload)
            
            if len(batch_embeddings) >= batch_size:
                # Upload batch using raw upload method
                points = []
                for i, (emb, pay) in enumerate(zip(batch_embeddings, batch_payloads)):
                    point_id = str(uuid.uuid4())
                    points.append(
                        PointStruct(
                            id=point_id,
                            vector=emb,
                            payload=pay
                        )
                    )
                
                operation_info = uploader.client.upsert(
                    collection_name=uploader.collection_name,
                    wait=uploader.wait_until_indexed,
                    points=points
                )
                
                if operation_info.status == UpdateStatus.COMPLETED:
                    total_uploaded += len(points)
                    logger.info(f"⬆️ Uploaded batch of {len(points)} points")
                
                batch_embeddings = []
                batch_payloads = []
                
        except StopIteration:
            break
        except Exception as e:
            logger.error(f"❌ Error in callback upload: {e}")
            break
    
    # Upload remaining
    if batch_embeddings:
        points = []
        for i, (emb, pay) in enumerate(zip(batch_embeddings, batch_payloads)):
            point_id = str(uuid.uuid4())
            points.append(
                PointStruct(
                    id=point_id,
                    vector=emb,
                    payload=pay
                )
            )
        
        operation_info = uploader.client.upsert(
            collection_name=uploader.collection_name,
            wait=uploader.wait_until_indexed,
            points=points
        )
        
        if operation_info.status == UpdateStatus.COMPLETED:
            total_uploaded += len(points)
    
    logger.info(f"🏁 Callback upload complete. Total: {total_uploaded} points")
    return total_uploaded