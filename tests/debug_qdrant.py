"""
CONFIRMATION: YOUR QDRANT SETUP WORKS
"""
from qdrant_client import QdrantClient
import numpy as np
import uuid

print("🎉 CONFIRMATION: YOUR QDRANT SETUP IS WORKING!")
print("="*60)

# Connect
client = QdrantClient(
    url="https://29cd6d37-e60d-4f16-a7b1-a7afba0d80e1.europe-west3-0.gcp.cloud.qdrant.io:6333",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.h1VmZzlKmAOlrLpzDwwFWicJ9drx4KAVsa9SK08f1k0"
)

print("✅ Connected to Qdrant Cloud")

# Collection exists
collections = client.get_collections()
collection_name = "person_reid_embeddings"

if collection_name in [c.name for c in collections.collections]:
    print(f"✅ Collection '{collection_name}' exists")
else:
    print("⚠️ Collection doesn't exist (but connection works!)")

# Upload test
test_embedding = np.random.randn(256).tolist()
point_id = str(uuid.uuid4())

from qdrant_client.http import models
client.upsert(
    collection_name=collection_name,
    points=[
        models.PointStruct(
            id=point_id,
            vector=test_embedding,
            payload={"test": True, "camera_id": 1}
        )
    ]
)

print(f"✅ Uploaded embedding: {point_id}")

# Search test
results = client.query_points(
    collection_name=collection_name,
    query=test_embedding,
    limit=3
)

print(f"✅ Search found {len(results.points)} results")

print("\n" + "="*60)
print("🎯 BOTTOM LINE: IT WORKS!")
print("="*60)
print("\nYour Qdrant is READY for your FYP!")
print("\n✅ Connection: WORKING")
print("✅ Storage: WORKING") 
print("✅ Upload: WORKING")
print("✅ Search: WORKING")
print("\n🚀 Now run your main pipeline and process videos!")