# Qdrant Utilities

This folder contains small utilities for ingesting documents into Qdrant and inspecting collections. It is focused on **local development** and **data preparation** for the RAG pipeline.

## What This Folder Does
- **Loads PDF documents** from a directory.
- **Chunks** documents into smaller pieces for retrieval.
- **Embeds** chunks with both **dense** and **sparse** models.
- **Creates/Recreates** a Qdrant collection with named dense + sparse vectors.
- **Upserts** points with payloads compatible with the retriever.

## Entrypoints

### 1) `data_preprocessor.py`
**Purpose:** Ingests PDFs into Qdrant with dense + sparse vectors.

**How it works (high-level):**
1. Load PDFs
2. Split into chunks
3. Create embeddings (dense + sparse)
4. Create Qdrant collection with named vectors
5. Upsert points with payload

**Run it:**
```bash
python src/qdrant/data_preprocessor.py
```

### 2) `qdrant_collection.py`
**Purpose:** Quick check of available collections in Qdrant.

**Run it:**
```bash
python src/qdrant/qdrant_collection.py
```

### 3) `pdf_loader.py`
**Purpose:** Helper for loading PDFs into `langchain` documents. Used by ingestion.

## Required Settings (.env / settings)
These are read from:
- `qdrant.settings` via `qdrant_settings` (Qdrant connection + vector field names)
- `rag.settings` via `retrieval_settings` (embedding model settings)

Qdrant (`qdrant.settings` / `.env`):
- `QDRANT_URL` (e.g. `http://localhost:6333`)
- `QDRANT_API_KEY` (if using auth)
- `COLLECTION` (collection name)
- `DENSE_VECTOR_NAME` (name of dense vector field in Qdrant)
- `SPARSE_VECTOR_NAME` (name of sparse vector field in Qdrant)

Retrieval (`rag.settings`):
- `DENSE_EMBED_MODEL` (dense model, OpenAI)
- `SPARSE_EMBED_MODEL` (sparse model, FastEmbed)
- `SPARSE_BATCH_SIZE` (sparse embed batch size)

## Vector Naming
The ingestion pipeline creates **named vectors**. That means:
- The **dense vector** is stored under `DENSE_VECTOR_NAME`
- The **sparse vector** is stored under `SPARSE_VECTOR_NAME`

Your retrieval config must use `using=<vector_name>` to target the correct vector field.

## data_preprocessor.py (Detailed)
`data_preprocessor.py` is the **ingestion pipeline** that prepares data for retrieval. It is intentionally separate from the query pipeline and is meant to be run when you want to (re)build the Qdrant collection.

### Step-by-step flow
1. **Load PDFs**
   `load_pdfs(...)` returns a list of `Document` objects containing text + metadata.

2. **Chunk documents**
   `RecursiveCharacterTextSplitter` creates smaller text chunks so retrieval is more precise and efficient.

3. **Create dense embeddings**
   `DenseEmbedder.aembed_batch(texts)` returns a list of dense vectors:
   - **Type:** `list[list[float]]`
   - **Shape:** fixed length per vector (e.g., 1536 floats)

4. **Create sparse embeddings**
   `SparseBM25Embedder.model.embed(texts, ...)` returns sparse vectors:
   - **Type:** `SparseVector` (Qdrant model)
   - **Shape:** `indices: list[int]`, `values: list[float]`
   - Only non-zero terms are stored, so each vector has variable length.

5. **Create the collection**
   The script recreates the collection with **named vectors**:
   - Dense vector name: `DENSE_VECTOR_NAME`
   - Sparse vector name: `SPARSE_VECTOR_NAME`

6. **Upsert points**
   Each chunk becomes a Qdrant point with:
   - `vector` as a dict of named vectors
   - `payload` containing `page_content` and `metadata`

### Dense vs. Sparse types (quick reference)
- **Dense**
  - Type: `list[float]`
  - Fixed length (always the same size)
- **Sparse**
  - Type: `SparseVector`
  - Variable length, stored as `indices` + `values`

### Retrieval result shape (why lists are nested)
When the retriever runs, results are nested because there are multiple layers:

1. **Multiple configs** (dense + sparse)
2. **Multiple query vectors per config** (identity + expansions)
3. **Ranked list per query vector**

So a single retriever call returns:
```
list[  # one entry per config task
  list[  # one entry per query vector
    list[tuple[Document, float]]  # ranked list for that vector
  ]
]
```

The aggregator (RRF or Simple) then flattens this into a single ranked list.

### Why named vectors matter
Because points store **both** dense and sparse vectors, you must name them.
At query time, you select the target vector using `using=<vector_name>`.

## Notes / Tips
- Ingestion **recreates** the collection each run. This will delete existing points.
- Sparse embeddings are generated via FastEmbed (default: `Qdrant/bm25`).
- Payloads must include `page_content` and `metadata` for the retriever to work.

If anything breaks, check that your Qdrant server is running and your vector names match between ingestion and retrieval.
