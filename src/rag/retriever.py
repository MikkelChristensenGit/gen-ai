from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

QDRANT_URL = "http://localhost:6333"
COLLECTION = "boardgame_rules_v0"

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-5"  # change if you prefer another chat model

TOP_K = 6


@dataclass
class RetrievedChunk:
    text: str
    source_name: str
    page: int | None


def _safe_page_to_human(page: int | None) -> str:
    # PyPDFLoader typically stores 0-indexed page numbers in metadata["page"]
    if page is None:
        return "?"
    return str(page + 1)


def format_context(chunks: Iterable[RetrievedChunk]) -> str:
    parts: list[str] = []
    for i, c in enumerate(chunks, start=1):
        parts.append(f"[{i}] Source: {c.source_name} (p. {_safe_page_to_human(c.page)})\n{c.text}")
    return "\n\n---\n\n".join(parts)


def main() -> None:
    # 1) Same embedding model used at ingestion time
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)

    # 2) Connect to existing Qdrant collection
    # Most setups support this constructor directly:
    vectorstore = QdrantVectorStore.from_existing_collection(
        url=QDRANT_URL,
        collection_name=COLLECTION,
        embedding=embeddings,
    )

    # 3) Retriever (start simple)
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )

    # 4) Chat model for answer generation
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)

    system_instructions = (
        "You are a board-game rules assistant.\n"
        "Answer using ONLY the provided context excerpts.\n"
        "If the answer is not in the context, say you cannot find it in the rules excerpts.\n"
        "Be concise and precise.\n"
        "At the end, include a 'Sources:' line that cites the excerpt numbers you used, "
        "e.g. Sources: [1], [3]."
    )

    print(f"Connected to Qdrant collection: {COLLECTION}")
    print("Ask a question (empty line to quit)\n")

    while True:
        question = input("Q: ").strip()
        if not question:
            break

        # 5) Retrieve relevant chunks (this embeds the query under the hood)
        docs = retriever.invoke(question)

        retrieved: list[RetrievedChunk] = []
        for d in docs:
            md = d.metadata or {}
            retrieved.append(
                RetrievedChunk(
                    text=d.page_content,
                    source_name=md.get("source_name") or md.get("source") or "unknown",
                    page=md.get("page"),
                )
            )

        context = format_context(retrieved)

        # 6) Generate grounded answer
        messages = [
            {"role": "system", "content": system_instructions},
            {
                "role": "user",
                "content": f"Question:\n{question}\n\nContext excerpts:\n{context}",
            },
        ]
        answer = llm.invoke(messages).content

        print("\nA:", answer, "\n")


if __name__ == "__main__":
    main()
