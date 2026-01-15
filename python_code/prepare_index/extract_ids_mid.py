# -*- coding: utf-8 -*-
"""
MID minister speeches ID index crawler (EN + RU) -> 2 separate CSV files

Outputs (default):
- mid_english_index.csv
- mid_russian_index.csv

Each CSV columns:
- page_num, id, url
"""

import os
import csv
import re
import time
import random
import requests
from bs4 import BeautifulSoup

# -----------------------
# CONFIG YOU MAY CHANGE
# -----------------------

LANG_CONFIG = {
    "en": {
        "base": "https://mid.ru/en/press_service/minister_speeches/",
        "out_csv": "mid_english_index.csv",
        "accept_language": "en-US,en;q=0.9",
        # set end_page to an integer if you want a fixed range, or None to run-until-empty
        "start_page": 1,
        "end_page": 507,
        "id_re": re.compile(r"/en/press_service/minister_speeches/(\d+)(?:/|$)", re.I),
    },
    "ru": {
        "base": "https://mid.ru/ru/press_service/minister_speeches/",
        "out_csv": "mid_russian_index.csv",
        "accept_language": "ru-RU,ru;q=0.9,en-US;q=0.4,en;q=0.3",
        "start_page": 1,
        "end_page": 607,
        "id_re": re.compile(r"/ru/press_service/minister_speeches/(\d+)(?:/|$)", re.I),
    },
}

USER_AGENT = "your_email@udel.edu"  # keep one contact here (or change as you like)

DELAY_BASE = 0.4
DELAY_JITTER = (0.1, 0.4)
STOP_AFTER_EMPTY = 2
MAX_PAGES = None          # optional cap (per language); None = no cap
TIMEOUT = 30
RETRIES = 2
CHECKPOINT_SIZE = 500     # flush every N new IDs (per language)

# -----------------------
# HELPERS
# -----------------------

def sleep_a_bit():
    time.sleep(DELAY_BASE + random.uniform(*DELAY_JITTER))

def build_headers(accept_language: str):
    return {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": accept_language,
        "Connection": "keep-alive",
    }

def get_soup(session: requests.Session, url: str, headers: dict) -> BeautifulSoup | None:
    """Fetch URL and return BeautifulSoup, or None on challenge/empty/error."""
    for _ in range(RETRIES):
        try:
            sleep_a_bit()
            r = session.get(url, headers=headers, timeout=TIMEOUT)
            r.raise_for_status()
            text = r.text or ""
            # Bail on obvious JS/challenge pages
            if "/TSPD/" in text or "Please enable JavaScript" in text:
                return None
            return BeautifulSoup(r.content, "html.parser")
        except requests.RequestException:
            continue
    return None

def extract_ids_from_soup(soup: BeautifulSoup | None, id_re: re.Pattern) -> list[str]:
    """Return unique IDs found on a listing page."""
    if not soup:
        return []

    seen_local = set()
    ids = []

    # Strict selector first (what you were using)
    for a in soup.select('a.announce__link.announce__link_fix_en[href]'):
        href = a.get("href") or ""
        m = id_re.search(href)
        if m:
            _id = m.group(1)
            if _id not in seen_local:
                seen_local.add(_id)
                ids.append(_id)

    # Fallback: scan HTML strictly for the configured language regex
    if not ids:
        html_text = str(soup)
        for m in id_re.finditer(html_text):
            _id = m.group(1)
            if _id not in seen_local:
                seen_local.add(_id)
                ids.append(_id)

    return ids

def url_for_id(base: str, _id: str) -> str:
    return f"{base}{_id}/"

def read_existing_ids(csv_path: str) -> set[str]:
    exist = set()
    if not os.path.exists(csv_path):
        return exist
    try:
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                rid = (row.get("id") or "").strip()
                if rid:
                    exist.add(rid)
    except Exception:
        pass
    return exist

def append_rows(csv_path: str, rows: list[dict]):
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["page_num", "id", "url"])
        if not exists:
            w.writeheader()
        for row in rows:
            w.writerow(row)

# -----------------------
# MAIN WORK (per language)
# -----------------------

def crawl_one_language(lang_key: str):
    cfg = LANG_CONFIG[lang_key]
    base = cfg["base"]
    out_csv = cfg["out_csv"]
    start_page = cfg["start_page"]
    end_page = cfg["end_page"]
    id_re = cfg["id_re"]
    headers = build_headers(cfg["accept_language"])

    print(f"\n===== LANG={lang_key.upper()} =====")
    print(f"[init] base   -> {base}")
    print(f"[init] outcsv -> {out_csv}")

    seen = read_existing_ids(out_csv)
    if seen:
        print(f"[init] already have {len(seen)} IDs in {out_csv}")

    empty_streak = 0
    page = start_page
    batch = []
    total_inserted = 0

    with requests.Session() as session:
        while True:
            if end_page is not None and page > end_page:
                break
            if MAX_PAGES and (page - start_page + 1) > MAX_PAGES:
                break

            q_url = f"{base}?PAGEN_1={page}"
            slash_url = f"{base}PAGEN_1={page}"

            soup = get_soup(session, q_url, headers) or get_soup(session, slash_url, headers)
            ids = extract_ids_from_soup(soup, id_re)

            if not ids:
                print(f"[page {page}] found 0 ids")
                empty_streak += 1
                if empty_streak >= STOP_AFTER_EMPTY:
                    print("[stop] consecutive empty pages reached. stopping.")
                    break
            else:
                empty_streak = 0
                new_rows = []
                for _id in ids:
                    if _id in seen:
                        continue
                    seen.add(_id)
                    new_rows.append({"page_num": page, "id": _id, "url": url_for_id(base, _id)})

                inserted = len(new_rows)
                if inserted:
                    batch.extend(new_rows)
                    total_inserted += inserted

                print(f"[page {page}] found={len(ids)} inserted={inserted} total_inserted={total_inserted}")

                # checkpoint flush
                if len(batch) >= CHECKPOINT_SIZE:
                    append_rows(out_csv, batch)
                    print(f"[checkpoint] wrote {len(batch)} rows -> {out_csv}")
                    batch = []
                    # reload for safest dedupe after restarts
                    seen = read_existing_ids(out_csv)

            page += 1

    if batch:
        append_rows(out_csv, batch)
        print(f"[final] wrote {len(batch)} remaining rows -> {out_csv}")

    print(f"[done] {lang_key.upper()} total inserted: {total_inserted}")

def main(run_lang: str = "both"):
    if run_lang not in ("en", "ru", "both"):
        raise ValueError("run_lang must be 'en', 'ru', or 'both'")

    if run_lang in ("en", "both"):
        crawl_one_language("en")
    if run_lang in ("ru", "both"):
        crawl_one_language("ru")

if __name__ == "__main__":
    # default: run both
    main(run_lang="both")
