# Rerankers

This folder contains second-stage rerankers for retrieval.

Current implementation:
- `llm_pointwise.py` -> `LLMPointwiseReranker`

## What It Does
The reranker receives the fused retrieval candidates and re-scores each `(query, chunk)` pair with an LLM.
It then sorts by rerank score (descending) and returns the final top-k documents.

Tie behavior:
- If two chunks get the same rerank score, the original fused order is preserved (earlier "wins").

Failure behavior:
- If reranking fails, the pipeline falls back to fused ranking (handled in `retrieval/pipeline.py`).

## Pipeline Position
Flow:
1. Parse query
2. Embed query variants
3. Retrieve dense/sparse candidates
4. Fuse candidates (RRF/Simple)
5. Rerank fused pool (optional)
6. Return `TOP_K`

## Relevant Settings
Defined in `src/rag/settings.py`:
- `RERANK_ENABLED`: enable/disable reranker stage
- `RERANK_MODEL`: model used for pointwise scoring
- `RERANK_CANDIDATE_K`: fused pool size passed into reranker
- `RERANK_MAX_CONCURRENCY`: max concurrent LLM scoring calls
- `TOP_K`: final number of returned chunks

## Score Format
The pointwise prompt asks for:
- a single integer score in `[1, 10]`

`llm_pointwise.py` enforces this by parsing and validating model output before ranking.
