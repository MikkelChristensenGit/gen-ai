import os
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient

# Load .env from the project root
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

print(qdrant_client.get_collections())
