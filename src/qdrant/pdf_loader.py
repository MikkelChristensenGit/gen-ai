from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader


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
