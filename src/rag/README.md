# RAG Retrieval Overview

This folder contains the query-time retrieval pipeline. It orchestrates parsing, embedding, retrieval, and aggregation.

## End-to-End Flow (Dense + Sparse)

### Example query
```
"Who rules the clearing?"
```

### 1) Parse the query
`ParserComponent.run()` runs multiple parsers (identity + rephrase + expansion):
```
[
  {"parser": QUERY_IDENTITY, "text": "Who rules the clearing?"},
  {"parser": QUERY_REPHRASE, "text": "Who controls the clearing according to the rules?"},
  {"parser": QUERY_EXPANSION, "text": "rules of clearing control"},
  {"parser": QUERY_EXPANSION, "text": "who controls a clearing"},
]
```
`QUERY_EXPANSION_MAX` limits how many expansion lines are used, even if the LLM returns more.

### 2) Embed dense + sparse
`EmbedComponent.run()` batches by vector type and returns a **single flat list** of embedded items:
```
[
  {"parser": QUERY_IDENTITY, "vector_type": DENSE, "text": "...", "vector": [..]},
  {"parser": QUERY_REPHRASE, "vector_type": DENSE, "text": "...", "vector": [..]},
  {"parser": QUERY_EXPANSION, "vector_type": DENSE, "text": "...", "vector": [..]},
  {"parser": QUERY_EXPANSION, "vector_type": DENSE, "text": "...", "vector": [..]},
  {"parser": QUERY_IDENTITY, "vector_type": SPARSE, "text": "...", "vector": {"indices": [...], "values": [...]}},
  {"parser": QUERY_REPHRASE, "vector_type": SPARSE, "text": "...", "vector": {"indices": [...], "values": [...]}},
  {"parser": QUERY_EXPANSION, "vector_type": SPARSE, "text": "...", "vector": {"indices": [...], "values": [...]}},
  {"parser": QUERY_EXPANSION, "vector_type": SPARSE, "text": "...", "vector": {"indices": [...], "values": [...]}},
]
```

### 3) Retrieval (Qdrant)
`RetrievalComponent.run()` executes two configs:
- **Dense config**
* Collects all dense vectors from `QUERY_IDENTITY`, `QUERY_REPHRASE`, and `QUERY_EXPANSION`
* Queries Qdrant with `using=DENSE_VECTOR_NAME`
- **Sparse config**
* Collects all sparse vectors from `QUERY_IDENTITY`, `QUERY_REPHRASE`, and `QUERY_EXPANSION`
* Queries Qdrant with `using=SPARSE_VECTOR_NAME`

Each config returns ranked lists of `(Document, score)` per vector.

#### How `sparse_batch_search` works
`sparse_batch_search` takes a list of sparse vectors (each `{indices, values}`), builds one `QueryRequest` per vector, and sends them **together** in a single `query_batch_points` call. This is more efficient than sending one HTTP request per query vector.

Why batching:
- **Fewer HTTP requests** → lower overhead and latency.
- **Qdrant is optimized for batch queries** → better throughput.
- Works naturally with query expansion (many query vectors at once).

### 4) Fusion (RRF)
RRF combines dense + sparse ranked lists by **rank position**, not raw scores. Docs appearing in both lists are ranked higher.

### 5) Final output
The top‑k fused documents are returned and used as context for the chat model.

## Batch Flow (Query Time)
Chronological order of batches for a single query:
1. **Dense query embedding batch** – `DenseEmbedder.aembed_batch(...)` on all dense query texts.
2. **Sparse query embedding batch** – `SparseBM25Embedder.aembed_batch(...)` on all sparse query texts.
3. **Dense retrieval batch** – `dense_batch_search(...)` sends one `query_batch_points` request for all dense vectors.
4. **Sparse retrieval batch** – `sparse_batch_search(...)` sends one `query_batch_points` request for all sparse vectors.

## Retrieval Settings
- Retrieval defaults live in `src/rag/settings.py`:
  - `TOP_K`: final number of documents returned after fusion.
  - `CANDIDATE_LIMIT`: per-retriever candidate depth fetched from Qdrant.
  - `QUERY_EXPANSION_MAX`: hard cap on how many query expansions are used.
  - `DENSE_EMBED_MODEL`, `SPARSE_EMBED_MODEL`, `SPARSE_BATCH_SIZE`, `CHAT_MODEL`.
- Qdrant connection + vector names live in `src/qdrant/settings.py`.
- Secrets are in `.env`.

## `CANDIDATE_LIMIT` vs `TOP_K` (Concrete Example)
Think of these as two different knobs:
- `CANDIDATE_LIMIT` controls retrieval depth (recall).
- `TOP_K` controls final output size.

Example with `QUERY_EXPANSION_MAX=3`, `CANDIDATE_LIMIT=20`, and `TOP_K=5`:
1. Parsers produce up to 5 query texts total (`identity` + `rephrase` + up to 3 expansions).
2. Dense retrieval runs one ranked list per dense query vector:
   - up to 5 dense lists
   - each list up to 20 docs
   - up to 100 dense rows before fusion
3. Sparse retrieval does the same:
   - up to 5 sparse lists
   - each list up to 20 docs
   - up to 100 sparse rows before fusion
4. RRF fuses all ranked lists (and deduplicates by document id).
5. Final truncation applies `TOP_K`, so only the best 5 fused docs are returned.

Important nuance:
- This is not one global "top 100 dense + top 100 sparse" list.
- It is multiple per-vector ranked lists that are fused together.

## Entry Point
Run the query REPL:
```bash
uv run python src/rag/retriever.py
```
