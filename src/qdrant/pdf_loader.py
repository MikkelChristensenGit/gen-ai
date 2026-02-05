from pathlib import Path

# from langchain_community.document_loaders import PyPDFLoader
from langchain_unstructured import UnstructuredLoader


def load_pdfs(pdf_dir: Path):
    docs = []
    for pdf_path in sorted(pdf_dir.rglob("*.pdf")):
        # "hi_res" improves layout + table detection; "ocr_only" if the PDF is scanned
        loader = UnstructuredLoader(
            str(pdf_path),
            mode="elements",
            strategy="hi_res",
        )
        elements = loader.load()

        for d in elements:
            d.metadata["source_name"] = pdf_path.name
            d.metadata["source_path"] = str(pdf_path)
        docs.extend(elements)
    return docs


"""
def load_pdfs(pdf_dir: Path):
    docs = []
    for pdf_path in sorted(pdf_dir.rglob("*.pdf")):
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()  # one Document per page

        # Add basic metadata (at minimum: source file)
        for d in pages:
            d.metadata["source_name"] = pdf_path.name
        docs.extend(pages)
    return docs
"""
