from pathlib import Path

# from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pdf_loader import load_pdfs

QDRANT_URL = "http://localhost:6333"
COLLECTION = "boardgame_rules_v0"
PDF_DIR = Path("data/rules")
# COLLECTION = "hofor_kravspec"
# PDF_DIR = Path("pdf_downloads")

"""
def load_pdfs(pdf_dir: Path):
    docs = []
    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()  # one Document per page

        # Add basic metadata (at minimum: source file)
        for d in pages:
            d.metadata["source_name"] = pdf_path.name
        docs.extend(pages)
    return docs
"""


def main():
    """
    1) Load game rules (pdfs)
    2) Define chunker and chunk documents
    3) Define embedding model
    4) Upload embeddings to Qdrant
    """
    docs = load_pdfs(PDF_DIR)
    if not docs:
        raise SystemExit(f"No PDFs found in {PDF_DIR.resolve()}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        url=QDRANT_URL,
        collection_name=COLLECTION,
    )

    print(f"Ingested {len(chunks)} chunks into Qdrant collection '{COLLECTION}'")


if __name__ == "__main__":
    main()
