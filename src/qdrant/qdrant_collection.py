from qdrant_client import QdrantClient

from qdrant.settings import qdrant_settings

qdrant_client = QdrantClient(
    url=qdrant_settings.QDRANT_URL,
    api_key=qdrant_settings.QDRANT_API_KEY,
)

print(qdrant_client.get_collections())
