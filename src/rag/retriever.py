from __future__ import annotations

from typing import Any

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

from prompts.system_prompt import SYSTEM_PROMPT
from rag.settings import settings
from rag.utils import RetrievedChunk, format_context


def main() -> None:
    # 1) Same embedding model used at ingestion time
    embeddings = OpenAIEmbeddings(model=settings.EMBED_MODEL)

    # 2) Connect to existing Qdrant collection
    vectorstore = QdrantVectorStore.from_existing_collection(
        url=settings.QDRANT_URL,
        collection_name=settings.COLLECTION,
        embedding=embeddings,
    )

    # 3) Retriever (start simple)
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": settings.TOP_K},
    )

    # 4) Chat model for answer generation
    llm = ChatOpenAI(model=settings.CHAT_MODEL, temperature=0)

    history: list[dict[Any, Any]] = []
    print(f"Connected to Qdrant collection: {settings.COLLECTION}")
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

        messages = (
            [{"role": "system", "content": SYSTEM_PROMPT}]
            + history
            + [
                {
                    "role": "user",
                    "content": f"Question:\n{question}\n\nContext excerpts:\n{context}",
                }
            ]
        )

        answer = llm.invoke(messages).content
        print("\nA:", answer, "\n")
        print(type(question))
        print(type(answer))
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
