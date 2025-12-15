"""
Qdrant Database Integration Module
Provides vector storage and similarity search for person ReID embeddings.
"""

# Client module
from database.qdrant.client import (
    get_qdrant_client,
    get_qdrant_config,
    perform_health_check
)

# Collections module
from database.qdrant.collections import (
    setup_qdrant_collection,
    get_collection_statistics,
    verify_collection,
    QdrantCollectionManager
)

# Uploader module
from database.qdrant.uploader import (
    QdrantUploader,
    get_uploader,
    upload_from_callback
)

# Queries module
from database.qdrant.queries import QdrantQueryManager

__all__ = [
    # Client
    'get_qdrant_client',
    'get_qdrant_config',
    'perform_health_check',
    
    # Collections
    'setup_qdrant_collection',
    'get_collection_statistics',
    'verify_collection',
    'QdrantCollectionManager',
    
    # Uploader
    'QdrantUploader',
    'get_uploader',
    'upload_from_callback',
    
    # Queries
    'QdrantQueryManager',
]