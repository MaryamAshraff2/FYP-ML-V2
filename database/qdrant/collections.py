# =============================
# database/qdrant/collections.py
# Collection Schema and Management - FIXED VERSION
# =============================

from qdrant_client.http import models
from database.qdrant.client import get_qdrant_client, get_qdrant_config
import logging

logger = logging.getLogger(__name__)


class QdrantCollectionManager:
    """Manages Qdrant collection creation and schema."""
    
    def __init__(self):
        self.client = get_qdrant_client()
        self.config = get_qdrant_config()
        # Access nested collection config
        self.collection_config = self.config.get('collection', {})
        self.collection_name = self.collection_config.get('name', 'person_reid_embeddings')
        self.vector_config = self.collection_config
    
    def create_collection_if_not_exists(self) -> bool:
        """
        Create collection if it doesn't exist.
        
        Returns:
            bool: True if created or already exists, False on error
        """
        try:
            # Check if collection exists
            existing_collections = self.client.get_collections()
            collection_names = [c.name for c in existing_collections.collections]
            
            if self.collection_name in collection_names:
                logger.info(f"✅ Collection '{self.collection_name}' already exists")
                return True
            
            # Create collection
            logger.info(f"📝 Creating collection '{self.collection_name}'...")
            
            vectors_config = models.VectorParams(
                size=self.vector_config.get('vector_size', 256),
                distance=models.Distance[self.vector_config.get('distance', 'COSINE').upper()]
            )
            
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=vectors_config,
                on_disk_payload=self.vector_config.get('on_disk_payload', False)
            )
            
            # Create payload indices for efficient filtering
            self._create_payload_indices()
            
            logger.info(f"✅ Created collection '{self.collection_name}' with dimension {self.vector_config.get('vector_size', 256)}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error creating collection: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _create_payload_indices(self):
        """Create indices on payload fields for faster filtering."""
        try:
            # Index on camera_id (for filtering by camera)
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="camera_id",
                field_schema=models.PayloadSchemaType.INTEGER
            )
            
            # Index on global_id (for filtering by assigned identity)
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="global_id",
                field_schema=models.PayloadSchemaType.INTEGER
            )
            
            # Index on local_track_id (for filtering by local track)
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="local_track_id",
                field_schema=models.PayloadSchemaType.INTEGER
            )
            
            # Index on timestamp (for temporal filtering)
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="timestamp",
                field_schema=models.PayloadSchemaType.INTEGER
            )
            
            logger.info("✅ Created payload indices for efficient filtering")
            
        except Exception as e:
            logger.warning(f"⚠️ Error creating payload indices: {e}")
    
    def recreate_collection(self) -> bool:
        """
        Delete and recreate collection (use with caution!).
        
        Returns:
            bool: True if successful, False on error
        """
        try:
            logger.warning(f"⚠️ Recreating collection '{self.collection_name}' - all data will be lost!")
            
            vectors_config = models.VectorParams(
                size=self.vector_config.get('vector_size', 256),
                distance=models.Distance[self.vector_config.get('distance', 'COSINE').upper()]
            )
            
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=vectors_config,
                on_disk_payload=self.vector_config.get('on_disk_payload', False)
            )
            
            # Create payload indices
            self._create_payload_indices()
            
            logger.info(f"✅ Recreated collection '{self.collection_name}'")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error recreating collection: {e}")
            return False
    
    def delete_collection(self) -> bool:
        """
        Delete the collection.
        
        Returns:
            bool: True if successful, False on error
        """
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"✅ Deleted collection '{self.collection_name}'")
            return True
        except Exception as e:
            logger.error(f"❌ Error deleting collection: {e}")
            return False
    
    def get_collection_stats(self) -> dict:
        """
        Get statistics about the collection.
        FIXED: Updated to handle new Qdrant API
        
        Returns:
            dict: Collection statistics
        """
        try:
            info = self.client.get_collection(self.collection_name)
            
            # Handle both old and new API versions
            vectors_count = getattr(info, 'vectors_count', None)
            if vectors_count is None:
                # New API: points_count is the same as vectors_count
                vectors_count = getattr(info, 'points_count', 0)
            
            points_count = getattr(info, 'points_count', 0)
            
            stats = {
                'collection_name': self.collection_name,
                'name': self.collection_name,
                'vectors_count': vectors_count,
                'points_count': points_count,
                'segments_count': getattr(info, 'segments_count', 0),
                'status': str(getattr(info, 'status', 'unknown')),
                'optimizer_status': str(info.optimizer_status.status) if hasattr(info, 'optimizer_status') and info.optimizer_status else None,
                'vector_size': info.config.params.vectors.size,
                'distance': info.config.params.vectors.distance.name
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Error getting collection stats: {e}")
            import traceback
            traceback.print_exc()
            return {
                'collection_name': self.collection_name,
                'name': self.collection_name,
                'vectors_count': 0,
                'points_count': 0,
                'vector_size': self.vector_config.get('vector_size', 256),
                'distance': self.vector_config.get('distance', 'Cosine')
            }
    
    def get_collection_info(self) -> dict:
        """
        Get detailed information about the collection.
        Alias for get_collection_stats for compatibility.
        
        Returns:
            dict: Collection info
        """
        return self.get_collection_stats()
    
    def collection_exists(self) -> bool:
        """
        Check if collection exists.
        
        Returns:
            bool: True if exists, False otherwise
        """
        try:
            existing_collections = self.client.get_collections()
            collection_names = [c.name for c in existing_collections.collections]
            return self.collection_name in collection_names
        except Exception as e:
            logger.error(f"❌ Error checking collection existence: {e}")
            return False
    
    def verify_collection_schema(self) -> bool:
        """
        Verify that collection schema matches configuration.
        
        Returns:
            bool: True if schema matches, False otherwise
        """
        try:
            info = self.client.get_collection(self.collection_name)
            
            expected_size = self.vector_config.get('vector_size', 256)
            actual_size = info.config.params.vectors.size
            
            expected_distance = self.vector_config.get('distance', 'COSINE').upper()
            actual_distance = info.config.params.vectors.distance.name
            
            if actual_size != expected_size:
                logger.error(f"❌ Vector size mismatch: expected {expected_size}, got {actual_size}")
                return False
            
            if actual_distance != expected_distance:
                logger.error(f"❌ Distance metric mismatch: expected {expected_distance}, got {actual_distance}")
                return False
            
            logger.info(f"✅ Collection schema verified: {actual_size}D vectors, {actual_distance} distance")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error verifying collection schema: {e}")
            return False


# Convenience functions for backward compatibility

def setup_qdrant_collection(recreate: bool = False) -> bool:
    """
    Setup Qdrant collection (main entry point).
    
    Args:
        recreate: If True, delete and recreate collection
        
    Returns:
        bool: True if successful, False on error
    """
    manager = QdrantCollectionManager()
    
    if recreate:
        return manager.recreate_collection()
    else:
        return manager.create_collection_if_not_exists()


def initialize_collection(recreate: bool = False) -> bool:
    """Alias for setup_qdrant_collection."""
    return setup_qdrant_collection(recreate=recreate)


def get_collection_statistics() -> dict:
    """Get collection statistics."""
    manager = QdrantCollectionManager()
    return manager.get_collection_stats()


def get_collection_info() -> dict:
    """Get collection information."""
    manager = QdrantCollectionManager()
    return manager.get_collection_info()


def verify_collection() -> bool:
    """Verify collection schema."""
    manager = QdrantCollectionManager()
    return manager.verify_collection_schema()


def verify_collection_existence() -> bool:
    """Check if collection exists."""
    manager = QdrantCollectionManager()
    return manager.collection_exists()


def collection_exists() -> bool:
    """Check if collection exists."""
    manager = QdrantCollectionManager()
    return manager.collection_exists()


def delete_collection() -> bool:
    """Delete the collection."""
    manager = QdrantCollectionManager()
    return manager.delete_collection()


def get_collection_manager() -> QdrantCollectionManager:
    """Get collection manager instance."""
    return QdrantCollectionManager()