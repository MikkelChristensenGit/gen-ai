import asyncio
from typing import Any

from langchain_openai import ChatOpenAI

from prompts.system_prompt import SYSTEM_PROMPT
from qdrant.settings import qdrant_settings
from rag.retrieval.pipeline import RetrievalPipeline
from rag.settings import retrieval_settings
from rag.utils import RetrievedChunk, format_context


async def repl() -> None:
    pipeline = RetrievalPipeline.from_default(
        qdrant_url=qdrant_settings.QDRANT_URL,
        qdrant_api_key=qdrant_settings.QDRANT_API_KEY,
        collection_name=qdrant_settings.COLLECTION,
        dense_embed_model=retrieval_settings.DENSE_EMBED_MODEL,
        candidate_limit=retrieval_settings.TOP_K,
        top_k=retrieval_settings.TOP_K,
    )

    llm = ChatOpenAI(model=retrieval_settings.CHAT_MODEL, temperature=0)

    history: list[dict[Any, Any]] = []
    print(f"Connected to Qdrant collection: {qdrant_settings.COLLECTION}")
    print("Ask a question (empty line to quit)\n")

    while True:
        question = (await asyncio.to_thread(input, "Q: ")).strip()
        if not question:
            break

        docs = await pipeline.ainvoke(question)

        retrieved: list[RetrievedChunk] = []
        for doc, _score in docs:
            md = doc.metadata or {}
            retrieved.append(
                RetrievedChunk(
                    text=doc.page_content,
                    source_name=md.get("source_name") or md.get("source") or "unknown",
                    page=md.get("page"),
                )
            )

        context = format_context(retrieved)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Question:\n{question}\n\nContext excerpts:\n{context}",
            },
        ]
        answer = llm.invoke(messages).content
        print("\nA:", answer, "\n")
        print(type(question))
        print(type(answer))
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})


def main() -> None:
    asyncio.run(repl())


if __name__ == "__main__":
    main()
