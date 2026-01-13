# -*- coding: utf-8 -*-
"""
Kremlin transcript index crawler (English + Russian)

What this script does
---------------------
- Crawls Kremlin transcript index pages (paginated).
- Extracts each transcript's numeric ID and absolute URL from each index page.
- Appends results to a CSV: columns = [page, id, url].
- Safe to re-run: de-duplicates rows already present in the output CSV.

How to use (EN vs RU)
---------------------
Kremlin has separate transcript indexes:

1) Kremlin EN transcripts (English):
   BASE_SECTION = "http://en.kremlin.ru/events/president/transcripts/"
   Index pages look like:
   http://en.kremlin.ru/events/president/transcripts/page/1

2) Kremlin RU transcripts (Russian):
   BASE_SECTION = "http://special.kremlin.ru/events/president/transcripts/"
   Index pages look like:
   http://special.kremlin.ru/events/president/transcripts/page/1

You can run either language by setting LANG = "en" or "ru" below, or by
duplicating this script into two thin wrappers (recommended in GitHub):

- scripts/00_ids/extract_ids_kremlin.py --lang en
- scripts/00_ids/extract_ids_kremlin.py --lang ru

For now, this is a single-file version with a simple LANG switch.

Notes
-----
- Be polite: requests include a User-Agent; you should replace CONTACT_EMAIL.
- The script sleeps between pages to reduce load and avoid rate-limits.
- Retries occur on transient status codes (403/429/5xx) and network errors.

Python: 3.x
"""

from __future__ import annotations

import csv
import html
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Set, Tuple
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


# ----------------------------
# Configuration
# ----------------------------

CONTACT_EMAIL = "your_email@example.com"  # TODO: replace with your email/contact
LANG = "ru"  # "ru" for special.kremlin.ru, "en" for en.kremlin.ru

# Output file (pick a clear name per language)
OUT_CSV_BY_LANG = {
    "ru": "kremlin_russian_transcript_index.csv",
    "en": "kremlin_english_transcript_index.csv",
}

BASE_SECTION_BY_LANG = {
    # Russian transcript index
    "ru": "http://special.kremlin.ru/events/president/transcripts/",
    # English transcript index
    "en": "http://en.kremlin.ru/events/president/transcripts/",
}

# Pagination uses: {BASE_SECTION}/page/{page}
INDEX_PATH_TEMPLATE = "page/{page}"

# Networking controls
TIMEOUT_SECONDS = 10
SLEEP_BETWEEN_PAGES_SECONDS = 10
MAX_RETRIES = 3
BACKOFF_MULTIPLIER = 2.0

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36 "
        f"; contact: {CONTACT_EMAIL}"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# CSV schema
FIELDNAMES = ["page", "id", "url"]

# Extract numeric ID from transcript URLs like:
# - /events/president/transcripts/78041
# - /events/president/transcripts/78041/
ID_RE = re.compile(r"/events/president/transcripts/(\d+)(?:/|$)")


# ----------------------------
# Logging
# ----------------------------

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("kremlin-index")


# ----------------------------
# HTTP session setup
# ----------------------------

SESSION = requests.Session()
# We do our own retries, so keep adapter retries at 0
SESSION.mount("http://", requests.adapters.HTTPAdapter(max_retries=0))
SESSION.mount("https://", requests.adapters.HTTPAdapter(max_retries=0))


@dataclass(frozen=True)
class CrawlResult:
    page: int
    transcript_id: str
    url: str


def build_index_url(base_section: str, page: int) -> str:
    """
    Build the URL for a given index page number.
    Example:
      base_section = "http://special.kremlin.ru/events/president/transcripts/"
      page=1 -> "http://special.kremlin.ru/events/president/transcripts/page/1"
    """
    # Ensure base ends with "/" for urljoin safety
    base = base_section if base_section.endswith("/") else base_section + "/"
    return urljoin(base, INDEX_PATH_TEMPLATE.format(page=page))


def fetch_html_with_retry(url: str) -> Tuple[Optional[str], Optional[int], str]:
    """
    GET a URL with basic retry/backoff.

    Returns:
      (html_text or None, status_code or None, final_url)
    """
    delay = 0.0
    last_error: Optional[Exception] = None

    for attempt in range(1, MAX_RETRIES + 1):
        if delay:
            time.sleep(delay)

        try:
            resp = SESSION.get(url, headers=HEADERS, timeout=TIMEOUT_SECONDS, allow_redirects=True)
            status = resp.status_code
            final_url = resp.url

            if status >= 400:
                log.info(f"[GET] {url} -> {status}")
                # Retry on transient rate-limit / forbidden / server errors
                if status in (403, 429) or 500 <= status < 600:
                    delay = max(SLEEP_BETWEEN_PAGES_SECONDS, delay * BACKOFF_MULTIPLIER if delay else 1.0)
                    continue
                return None, status, final_url

            ctype = resp.headers.get("Content-Type", "")
            if "text/html" not in ctype:
                return None, status, final_url

            text = resp.text or ""
            if len(text) < 200:  # quick sanity check
                return None, status, final_url

            return text, status, final_url

        except requests.RequestException as e:
            last_error = e
            log.info(f"[NET] {url} -> {e.__class__.__name__}: {e}")
            delay = max(SLEEP_BETWEEN_PAGES_SECONDS, delay * BACKOFF_MULTIPLIER if delay else 1.0)
            continue

    # Exhausted retries
    if last_error:
        return None, None, url
    return None, None, url


def parse_index_page(html_text: str, base_url_for_join: str) -> List[Tuple[str, str]]:
    """
    Parse an index page and return list of (transcript_id, absolute_url).

    Kremlin index pages typically include transcript links in:
      <div class="hentry">
        <h2><a href="...">...</a></h2>
      </div>

    We select: ".hentry h2 a[href]"
    """
    soup = BeautifulSoup(html_text, "lxml")
    items: List[Tuple[str, str]] = []

    for a in soup.select(".hentry h2 a[href]"):
        href = (a.get("href") or "").strip()
        if not href:
            continue

        # Unescape entities like &amp; and join to current base
        abs_url = urljoin(base_url_for_join, html.unescape(href))

        m = ID_RE.search(abs_url)
        if not m:
            continue

        items.append((m.group(1), abs_url))

    return items


def load_existing_triples(csv_path: str) -> Set[Tuple[str, str, str]]:
    """
    Load existing (page, id, url) triples from an output CSV.
    Used to prevent duplicates across re-runs.
    """
    triples: Set[Tuple[str, str, str]] = set()
    if not os.path.exists(csv_path):
        return triples

    try:
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pg = (row.get("page") or "").strip()
                rid = (row.get("id") or row.get("ID") or "").strip()
                u = (row.get("url") or "").strip()
                if pg and rid and u:
                    triples.add((pg, rid, u))
    except Exception as e:
        log.info(f"[WARN] Could not read existing CSV for dedup: {e}")

    return triples


def append_rows(csv_path: str, rows: Sequence[CrawlResult]) -> None:
    """
    Append rows to CSV, writing header if file does not exist.
    """
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()

        for r in rows:
            writer.writerow({"page": str(r.page), "id": r.transcript_id, "url": r.url})


def crawl_index_range(
    base_section: str,
    out_csv: str,
    start_page: int,
    end_page: int,
) -> None:
    """
    Crawl index pages in [start_page, end_page] and append deduped results to CSV.
    """
    existing_triples = load_existing_triples(out_csv)
    log.info(f"[INIT] output={out_csv} existing_triples={len(existing_triples)}")
    log.info(f"[RUN ] base_section={base_section}")
    log.info(f"[RUN ] pages={start_page}..{end_page}")

    for page in range(start_page, end_page + 1):
        index_url = build_index_url(base_section, page)

        html_text, status, final_url = fetch_html_with_retry(index_url)
        if not html_text:
            log.info(f"[{page}] status={status if status is not None else 'NA'} skipped (no HTML)")
            time.sleep(SLEEP_BETWEEN_PAGES_SECONDS)
            continue

        items = parse_index_page(html_text, final_url)
        if not items:
            log.info(f"[{page}] status={status} found=0")
            time.sleep(SLEEP_BETWEEN_PAGES_SECONDS)
            continue

        batch_triples: Set[Tuple[str, str, str]] = set()
        out_batch: List[CrawlResult] = []

        for transcript_id, abs_url in items:
            triple = (str(page), transcript_id, abs_url)

            # Global dedup (previous runs)
            if triple in existing_triples:
                continue

            # Local dedup (same page)
            if triple in batch_triples:
                continue

            out_batch.append(CrawlResult(page=page, transcript_id=transcript_id, url=abs_url))
            batch_triples.add(triple)

        if out_batch:
            append_rows(out_csv, out_batch)
            existing_triples.update(batch_triples)
            log.info(f"[{page}] status={status} saved={len(out_batch)} found={len(items)} deduped={len(items)-len(out_batch)}")
        else:
            log.info(f"[{page}] status={status} saved=0 found={len(items)} (all duplicates)")

        time.sleep(SLEEP_BETWEEN_PAGES_SECONDS)


# ----------------------------
# Main (set your page range)
# ----------------------------
if __name__ == "__main__":
    if LANG not in BASE_SECTION_BY_LANG:
        raise ValueError(f"LANG must be one of {list(BASE_SECTION_BY_LANG.keys())}, got: {LANG}")

    BASE_SECTION = BASE_SECTION_BY_LANG[LANG]
    OUT_CSV = OUT_CSV_BY_LANG[LANG]

    # ---- SET YOUR RANGE HERE ----
    START_PAGE = 1
    END_PAGE = 667

    crawl_index_range(
        base_section=BASE_SECTION,
        out_csv=OUT_CSV,
        start_page=START_PAGE,
        end_page=END_PAGE,
    )
