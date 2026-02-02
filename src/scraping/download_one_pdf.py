import hashlib
import json
import os
from typing import Any
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

START_URL = "https://hofor-tekniskdesign.dk"
OUT_DIR = "pdf_downloads"
MAX_PAGES = 25
SEEN_FILE = os.path.join(OUT_DIR, "seen_pdfs.json")
BASE_NETLOC = urlparse(START_URL).netloc


def _href_as_str(value: Any) -> str | None:
    """
    BeautifulSoup tag attributes can be typed as str | Sequence[str] | None.
    We only want a usable string href.
    """
    if isinstance(value, str):
        v = value.strip()
        return v if v else None

    # Sometimes stubs allow sequences; pick the first string if present.
    if isinstance(value, (list | tuple)) and value:
        first = value[0]
        if isinstance(first, str):
            v = first.strip()
            return v if v else None

    return None


def links(html: str, base_url: str) -> list[str]:
    """Extract absolute URLs from anchor tags in an HTML document."""
    soup = BeautifulSoup(html, "html.parser")
    out: list[str] = []

    for a in soup.select("a[href]"):
        href = _href_as_str(a.get("href"))
        if not href:
            continue
        out.append(urljoin(base_url, href))

    return out


def folder_name(page_url: str) -> str:
    """Create a filesystem-safe folder name based on a page URL."""
    path = urlparse(page_url).path.strip("/")
    name = path.split("/")[-1] if path else "homepage"
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)
    return safe or "page"


def pdf_filename(pdf_url: str) -> str:
    """Return a .pdf filename derived from the URL path."""
    name = os.path.basename(urlparse(pdf_url).path) or "document.pdf"
    return name if name.lower().endswith(".pdf") else f"{name}.pdf"


def load_seen() -> tuple[set[str], dict[str, str]]:
    if not os.path.exists(SEEN_FILE):
        return set(), {}

    with open(SEEN_FILE, encoding="utf-8") as f:
        data: Any = json.load(f)

    if isinstance(data, dict):
        hashes_raw = data.get("hashes", [])
        by_url_raw = data.get("by_url", {})
        seen_hashes = set(h for h in hashes_raw if isinstance(h, str))
        by_url: dict[str, str] = {
            k: v for k, v in by_url_raw.items() if isinstance(k, str) and isinstance(v, str)
        }
        return seen_hashes, by_url

    if isinstance(data, list):
        return set(h for h in data if isinstance(h, str)), {}

    return set(), {}


def main() -> None:
    """Crawl site pages and download the first PDF found per page."""
    os.makedirs(OUT_DIR, exist_ok=True)

    seen_hashes, by_url = load_seen()

    visited: set[str] = set()
    queue: list[str] = [START_URL]

    while queue and len(visited) < MAX_PAGES:
        page_url = queue.pop(0)
        if page_url in visited:
            continue
        visited.add(page_url)

        page = requests.get(page_url, timeout=20)
        page.raise_for_status()

        page_pdf: str | None = None
        for link in links(page.text, page_url):
            if ".pdf" in link.lower() and page_pdf is None:
                page_pdf = link
            elif urlparse(link).netloc == BASE_NETLOC and link not in visited:
                queue.append(link)

        if not page_pdf:
            continue

        pdf = requests.get(page_pdf, timeout=30)
        pdf.raise_for_status()

        pdf_hash = hashlib.sha256(pdf.content).hexdigest()
        if pdf_hash in seen_hashes:
            continue

        page_dir = os.path.join(OUT_DIR, folder_name(page_url))
        os.makedirs(page_dir, exist_ok=True)
        out_path = os.path.join(page_dir, pdf_filename(page_pdf))

        with open(out_path, "wb") as f:
            f.write(pdf.content)

        seen_hashes.add(pdf_hash)
        by_url[page_pdf] = pdf_hash

    if seen_hashes:
        with open(SEEN_FILE, "w", encoding="utf-8") as f:
            json.dump({"hashes": sorted(seen_hashes), "by_url": by_url}, f, indent=2)


if __name__ == "__main__":
    main()
