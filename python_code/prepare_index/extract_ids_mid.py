# -*- coding: utf-8 -*-
"""
MID minister speeches index crawler (English + Russian)

What this script does
---------------------
- Crawls MID "minister_speeches" list pages.
- Extracts each item’s numeric ID, absolute URL, and title.
- Writes/append to a CSV: columns = [page, id, url, title].
- Safe to re-run: de-duplicates on `id` (unique per item).

How to use (EN vs RU)
---------------------
MID has language-specific bases (confirm the exact paths you use in your paper/pipeline):

Common pattern you were using:
- English: https://mid.ru/en/press_service/minister_speeches/
- Russian: https://mid.ru/ru/press_service/minister_speeches/

Set LANG = "en" or "ru" below and the script will select BASE + output CSV name.

Notes about MID bot protection
------------------------------
MID sometimes serves JS / TSPD challenge pages. This script:
1) Tries curl_cffi (TLS impersonation) first (better success rate)
2) Falls back to requests
3) If a challenge is detected, it retries an alternate URL form once
4) Optionally dumps empty-page HTML for debugging

Python: 3.x
"""

from __future__ import annotations

import csv
import os
import random
import re
import time
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple
from urllib.parse import urljoin

from bs4 import BeautifulSoup


# ----------------------------
# Configuration
# ----------------------------

LANG = "en"  # "en" or "ru"

BASE_BY_LANG = {
    "en": "https://mid.ru/en/press_service/minister_speeches/",
    "ru": "https://mid.ru/ru/press_service/minister_speeches/",
}

OUTPUT_CSV_BY_LANG = {
    "en": "mid_minister_speeches_index_english.csv",
    "ru": "mid_minister_speeches_index_russian.csv",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
}

# List pages are accessed via query param like:
#   https://mid.ru/en/press_service/minister_speeches/?PAGEN_1=1
PAGE_PARAM = "PAGEN_1"

# De-dup key: numeric ID at end of URL
ID_RE = re.compile(r"/(\d+)(?:/)?$")

# CSV columns
FIELDNAMES = ["page", "id", "url", "title"]

# Networking
DEFAULT_TIMEOUT_SECONDS = 45


# ----------------------------
# Data model
# ----------------------------

@dataclass(frozen=True)
class IndexItem:
    page: int
    id: str
    url: str
    title: str


# ----------------------------
# HTTP fetching (curl_cffi -> requests fallback)
# ----------------------------

def fetch_html(url: str, timeout: int = DEFAULT_TIMEOUT_SECONDS) -> str:
    """
    Fetch HTML for `url`.

    Strategy:
      1) curl_cffi with browser impersonation (best for MID challenges)
      2) fallback to requests.Session()

    Returns empty string if all methods fail.
    """
    text: Optional[str] = None

    # Try curl_cffi first
    try:
        from curl_cffi import requests as curlreq  # type: ignore
        r = curlreq.get(
            url,
            headers=HEADERS,
            timeout=timeout,
            impersonate="chrome124",
            allow_redirects=True,
        )
        if r.status_code == 200:
            text = r.text
    except Exception:
        pass

    # Fallback: requests
    if text is None:
        try:
            import requests  # type: ignore
            with requests.Session() as s:
                s.headers.update(HEADERS)
                r = s.get(url, timeout=timeout, allow_redirects=True)
                if r.status_code == 200:
                    text = r.text
        except Exception:
            text = None

    return text or ""


def looks_like_challenge(html_text: str) -> bool:
    """
    Detect common MID anti-bot / JS challenge patterns.
    """
    if not html_text:
        return True
    markers = (
        "/TSPD/",
        "Please enable JavaScript",
        "Access denied",
        "verify you are human",
    )
    return any(m in html_text for m in markers)


# ----------------------------
# Parsing
# ----------------------------

def extract_id_from_url(url: str) -> Optional[str]:
    """
    Extract numeric ID from the end of a URL.
    Example: https://mid.ru/.../12345/  -> "12345"
    """
    m = ID_RE.search(url or "")
    return m.group(1) if m else None


def parse_list_page(html_text: str, base_url: str) -> List[Tuple[str, str, str]]:
    """
    Parse a list page HTML and return list of (id, abs_url, title).

    Your original selector:
      a.announce__link.announce__link_fix_en[href]

    NOTE:
    - This is *English-specific* class naming (fix_en).
    - For Russian pages, the class may differ.
      If RU returns 0 items consistently, inspect the RU HTML and adjust selector(s).
    """
    soup = BeautifulSoup(html_text, "lxml")
    out: List[Tuple[str, str, str]] = []

    anchors = soup.select("a.announce__link.announce__link_fix_en[href]")
    for a in anchors:
        href = (a.get("href") or "").strip()
        if not href:
            continue

        abs_url = urljoin(base_url, href)
        item_id = extract_id_from_url(abs_url)
        title = " ".join((a.get_text(" ", strip=True) or "").split())

        if item_id:
            out.append((item_id, abs_url, title))

    return out


# ----------------------------
# CSV utilities
# ----------------------------

def load_existing_ids(csv_path: str) -> Set[str]:
    """
    Read existing output CSV and return a set of IDs already written.
    """
    if not os.path.exists(csv_path):
        return set()

    ids: Set[str] = set()
    try:
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rid = (row.get("id") or "").strip()
                if rid:
                    ids.add(rid)
    except Exception:
        pass

    return ids


def append_rows(csv_path: str, rows: List[IndexItem]) -> None:
    """
    Append IndexItem rows to CSV, writing header if needed.
    """
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not exists:
            w.writeheader()

        for r in rows:
            w.writerow({"page": r.page, "id": r.id, "url": r.url, "title": r.title})


# ----------------------------
# Main crawl
# ----------------------------

def crawl_index(
    base_url: str,
    output_csv: str,
    start_page: int = 1,
    end_page: Optional[int] = None,
    delay_min: float = 1.0,
    delay_max: float = 2.0,
    stop_after_empty: int = 2,
    dump_empty_html: bool = True,
) -> None:
    """
    Walk list pages and write index rows (page,id,url,title).

    Stopping logic:
      - Stop if `end_page` reached, OR
      - Stop after `stop_after_empty` consecutive pages with 0 parsed items.
        (Useful when you don't know the final page count.)
    """
    existing = load_existing_ids(output_csv)
    print(f"[init] output={output_csv} | existing IDs={len(existing)}")

    empty_streak = 0
    page = start_page

    while True:
        if end_page is not None and page > end_page:
            break

        list_url = f"{base_url}?{PAGE_PARAM}={page}"
        print(f"[page {page}] -> {list_url}")

        html_text = fetch_html(list_url)

        # If it looks like a challenge, retry an alternate URL form once
        if looks_like_challenge(html_text):
            alt_url = urljoin(base_url, f"{PAGE_PARAM}={page}")
            print(f"[page {page}] challenge/empty suspected; retrying alt -> {alt_url}")
            html_text2 = fetch_html(alt_url)
            if len(html_text2) > len(html_text):
                html_text = html_text2

        items = parse_list_page(html_text, base_url)
        print(f"[page {page}] parsed_items={len(items)}")

        if not items:
            empty_streak += 1

            if dump_empty_html:
                try:
                    dbg_path = f"mid_{LANG}_page_{page}_empty.html"
                    with open(dbg_path, "w", encoding="utf-8") as fw:
                        fw.write(html_text)
                    print(f"[debug] dumped HTML -> {dbg_path}")
                except Exception:
                    pass

            if empty_streak >= stop_after_empty:
                print(f"[stop] hit {empty_streak} consecutive empty pages. stopping.")
                break
        else:
            empty_streak = 0

            new_rows: List[IndexItem] = []
            for item_id, abs_url, title in items:
                if item_id in existing:
                    continue
                new_rows.append(IndexItem(page=page, id=item_id, url=abs_url, title=title))

            if new_rows:
                append_rows(output_csv, new_rows)
                for r in new_rows:
                    existing.add(r.id)
                print(f"[csv] wrote {len(new_rows)} new rows | total IDs now {len(existing)}")
            else:
                print("[csv] no new IDs on this page (all duplicates)")

        # Random delay to reduce bot detection
        time.sleep(random.uniform(delay_min, delay_max))
        page += 1


if __name__ == "__main__":
    if LANG not in BASE_BY_LANG:
        raise ValueError(f"LANG must be one of {list(BASE_BY_LANG.keys())}, got: {LANG}")

    BASE = BASE_BY_LANG[LANG]
    OUTPUT_CSV = OUTPUT_CSV_BY_LANG[LANG]

    # Examples:
    # crawl_index(BASE, OUTPUT_CSV, start_page=1, end_page=50)    # fixed range
    # crawl_index(BASE, OUTPUT_CSV, start_page=1, end_page=None)  # run until empty pages encountered
    crawl_index(
        base_url=BASE,
        output_csv=OUTPUT_CSV,
        start_page=1,
        end_page=None,
        delay_min=1.0,
        delay_max=2.0,
        stop_after_empty=2,
        dump_empty_html=True,
    )
