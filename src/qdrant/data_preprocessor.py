import asyncio
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models

from qdrant.pdf_loader import load_pdfs
from qdrant.settings import qdrant_settings
from rag.retrieval.embedders.dense import DenseEmbedder
from rag.retrieval.embedders.sparse_bm25 import SparseBM25Embedder
from rag.settings import retrieval_settings


class IngestionPipeline:
    def __init__(
        self,
        dense_embedder: DenseEmbedder,
        sparse_embedder: SparseBM25Embedder,
        splitter: RecursiveCharacterTextSplitter,
        qdrant_client: QdrantClient,
        collection_name: str,
    ):
        self.dense_embedder = dense_embedder
        self.sparse_embedder = sparse_embedder
        self.splitter = splitter
        self.qdrant_client = qdrant_client
        self.collection_name = collection_name
        self.dense_vector_name = qdrant_settings.DENSE_VECTOR_NAME
        self.sparse_vector_name = qdrant_settings.SPARSE_VECTOR_NAME

    @classmethod
    def from_default(cls) -> "IngestionPipeline":
        dense = DenseEmbedder(model=retrieval_settings.DENSE_EMBED_MODEL)
        sparse = SparseBM25Embedder(
            model_name=retrieval_settings.SPARSE_EMBED_MODEL,
            batch_size=retrieval_settings.SPARSE_BATCH_SIZE,
        )
        splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=120)
        qdrant = QdrantClient(url=qdrant_settings.QDRANT_URL, api_key=qdrant_settings.QDRANT_API_KEY)
        return cls(
            dense_embedder=dense,
            sparse_embedder=sparse,
            splitter=splitter,
            qdrant_client=qdrant,
            collection_name=qdrant_settings.COLLECTION,
        )

    async def run(self, pdf_dir: Path) -> None:
        docs = load_pdfs(pdf_dir)
        if not docs:
            raise SystemExit("No documents found in the specified directory.")

        # 1) Chunk documents into retrieval-sized pieces
        chunks = self.splitter.split_documents(docs)
        texts = [doc.page_content for doc in chunks]

        # 2) Compute dense + sparse embeddings for each chunk
        dense_vectors = await self.dense_embedder.aembed_batch(texts)
        sparse_vectors = await self.sparse_embedder.aembed_batch(texts)

        dim = len(dense_vectors[0])
        # 3) Create collection with named dense + named sparse vectors
        self.qdrant_client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config={
                self.dense_vector_name: models.VectorParams(size=dim, distance=models.Distance.COSINE),
            },
            sparse_vectors_config={self.sparse_vector_name: models.SparseVectorParams()},
        )
        points = []
        for i, (doc, dense, sparse) in enumerate(zip(chunks, dense_vectors, sparse_vectors, strict=True)):
            # 4) Upsert points containing both dense and sparse vectors + payload
            points.append(
                models.PointStruct(
                    id=i,
                    vector={
                        self.dense_vector_name: dense,
                        self.sparse_vector_name: models.SparseVector(
                            indices=sparse["indices"],
                            values=sparse["values"],
                        ),
                    },
                    payload={"page_content": doc.page_content, "metadata": doc.metadata},
                )
            )
        BATCH_SIZE = 128
        for i in range(0, len(points), BATCH_SIZE):
            batch = points[i : i + BATCH_SIZE]
            self.qdrant_client.upsert(collection_name=self.collection_name, points=batch)


async def main() -> None:
    pipeline = IngestionPipeline.from_default()
    await pipeline.run(Path("data/rules"))


if __name__ == "__main__":
    asyncio.run(main())
