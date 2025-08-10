# Build/extend data/web_corpus.jsonl from URLs and local text/html files (no PDFs)
# Install: pip install requests trafilatura beautifulsoup4 lxml html5lib tqdm

import json, hashlib, time, re, sys
from pathlib import Path
from typing import List, Dict, Any, Iterable, Optional

import requests
import trafilatura
from bs4 import BeautifulSoup
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT  = DATA / "web_corpus.jsonl"
RAW  = DATA / "raw"

CHUNK_CHARS = 1200
OVERLAP     = 180
SLEEP_SEC   = 0.4
TIMEOUT     = 25
MIN_CHARS   = 400

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

URLS = [
    #  Youth & Qualification (AFM blog archives)
    "https://academyoffencingmasters.com/blog/qualification-update-fencing-summer-nationals-2022/",
    "https://academyoffencingmasters.com/blog/fencing-summer-nationals-qualifications-for-cadet-juniors-and-divisions-1a-2-3-youth/",
    "https://academyoffencingmasters.com/blog/qualification-update-2023-fencing-summer-nationals/",
    "https://academyoffencingmasters.com/blog/your-complete-guide-to-summer-nationals-2025-in-milwaukee/",
    "https://academyoffencingmasters.com/blog/types-of-fencing-competitions-qualification-paths/",
    "https://www.usafencing.org/age-classification-eligibility",
    "https://www.usafencing.org/basics-of-competition",

    #  Rules, Penalties & Qualifications Systems
    "https://www.usafencing.org/rules-compliance",
    "https://en.wikipedia.org/wiki/Fencing_rules",
    "https://academyoffencingmasters.com/blog/fencing-penalties-101-parents/",
    "https://academyoffencingmasters.com/blog/a-dummys-guide-to-right-of-way-or-priority-in-fencing/",
    "https://academyoffencingmasters.com/blog/how-to-learn-right-of-way-priority/",
    "https://academyoffencingmasters.com/blog/understanding-the-national-ranking-system-in-fencing",

    #  Technique, Training & Youth Advice
    "https://academyoffencingmasters.com/blog/basic-fencing-parries-explained/",
    "https://academyoffencingmasters.com/blog/foil-parries-and-blade-work/",
    "https://academyoffencingmasters.com/blog/12-tips-for-brand-new-fencer/",
    "https://academyoffencingmasters.com/blog/how-to-fence-unchallenging-training-bouts/",
    "https://academyoffencingmasters.com/blog/philosophy-of-sparring-by-charles-selberg/",
    "https://academyoffencingmasters.com/blog/mastering-patience-in-fencing/",
    "https://academyoffencingmasters.com/blog/what-are-usa-fencing-divisions-1-1a-2-and-3/",
    "https://academyoffencingmasters.com/blog/referees-and-right-of-way-priority/",

    #  Wikipedia Foundations for Technique & Tactics
    "https://en.wikipedia.org/wiki/Parry_(fencing)",
    "https://en.wikipedia.org/wiki/Fencing_tactics",
]



# Optional: local text/html files (NOT pdf)
LOCAL_FILES = [
    # str(DATA / "downloads" / "usaf_division1_guide.html"),
]

def clean_text(s: str) -> str:
    s = s or ""
    s = s.replace("\u00A0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def chunkify(text: str, size: int = CHUNK_CHARS, overlap: int = OVERLAP):
    text = clean_text(text)
    if not text: return []
    if len(text) <= size: return [text]
    out, i = [], 0
    while i < len(text):
        j = min(len(text), i + size)
        out.append(text[i:j])
        if j == len(text): break
        i = max(0, j - overlap)
    return out

def sha10(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]

def make_id(source: str, idx: int) -> str:
    return f"web-{sha10(source)}-{idx}"

def infer_title(html: str, fallback: str) -> str:
    try:
        soup = BeautifulSoup(html, "lxml")
        if soup.title and soup.title.string:
            return clean_text(soup.title.string)
    except Exception:
        pass
    return fallback

def infer_tags(url: str, title: str):
    tags = []
    u = (url or "").lower()
    t = (title or "").lower()
    if "usafencing.org" in u:
        tags += ["usaf", "official"]
    if any(k in (u + " " + t) for k in ["qualification", "eligibility", "division i", "division 1", "div 1", "nac", "national championships"]):
        tags += ["qualification", "div1"]
    if "fie.org" in u or "world cup" in (u + " " + t):
        tags += ["fie", "international"]
    return list(dict.fromkeys(tags))

def read_existing_ids(path: Path) -> set:
    if not path.exists(): return set()
    ids = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                obj = json.loads(line)
                if "id" in obj: ids.add(obj["id"])
            except Exception:
                continue
    return ids

def write_records(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def fetch_html(url: str) -> Optional[str]:
    try:
        r = requests.get(url, headers=HEADERS, timeout=TIMEOUT, allow_redirects=True)
        r.raise_for_status()
        ctype = r.headers.get("Content-Type", "")
        if "application/pdf" in ctype or url.lower().endswith(".pdf"):
            print(f"[skip] pdf detected (save as HTML locally instead): {url}")
            return None
        html = r.text
        RAW.mkdir(parents=True, exist_ok=True)
        (RAW / f"{sha10(url)}.html").write_text(html, encoding="utf-8", errors="ignore")
        return html
    except Exception as e:
        print(f"[fetch-error] {url}: {e}")
        return None

def extract_text(url: str, html: str) -> str:
    # Try trafilatura first
    try:
        t_text = trafilatura.extract(
            html, include_comments=False, include_formatting=False,
            favor_precision=True, url=url
        ) or ""
        t_text = clean_text(t_text)
    except Exception:
        t_text = ""
    # Fallback: BeautifulSoup visible text
    try:
        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script","style","noscript","header","footer","nav","svg"]):
            tag.extract()
        bs_text = clean_text(soup.get_text(" "))
    except Exception:
        bs_text = ""
    return t_text if len(t_text) >= len(bs_text) else bs_text

def extract_url(url: str):
    html = fetch_html(url)
    if not html:
        return None
    title = infer_title(html, url)
    text = extract_text(url, html)
    if not text or len(text) < MIN_CHARS:
        return None
    chunks = chunkify(text)
    if not chunks:
        return None
    tags = infer_tags(url, title)
    recs = []
    for i, ch in enumerate(chunks):
        recs.append({
            "id": make_id(url, i),
            "text": ch,
            "source": url,
            "title": title,
            "tags": tags,
            "date": "",
        })
    return recs

def ingest_urls(urls: List[str], out_path: Path = OUT):
    existing = read_existing_ids(out_path)
    new_recs = []
    for url in tqdm(urls, desc="Scraping"):
        try:
            recs = extract_url(url)
            if not recs:
                print(f"[skip] no extractable text: {url}")
            else:
                for r in recs:
                    if r["id"] not in existing:
                        new_recs.append(r)
        except Exception as e:
            print(f"[error] {url}: {e}")
        time.sleep(SLEEP_SEC)
    if new_recs:
        write_records(out_path, new_recs)
    print(f"[done] Added {len(new_recs)} new chunk(s) to {out_path}")

def ingest_files(paths: List[str], out_path: Path = OUT):
    existing = read_existing_ids(out_path)
    new_recs = []
    for p in paths:
        pth = Path(p)
        if not pth.exists():
            print(f"[skip] missing file: {pth}")
            continue
        try:
            txt = pth.read_text(encoding="utf-8", errors="ignore")
            for i, ch in enumerate(chunkify(txt)):
                rec = {
                    "id": make_id(str(pth), i),
                    "text": ch,
                    "source": str(pth),
                    "title": pth.stem,
                    "tags": ["local"],
                    "date": "",
                }
                if rec["id"] not in existing:
                    new_recs.append(rec)
        except Exception as e:
            print(f"[error] {pth}: {e}")
    if new_recs:
        write_records(out_path, new_recs)
    print(f"[done] Added {len(new_recs)} new local chunk(s) to {out_path}")

if __name__ == "__main__":
    mode = (sys.argv[1].lower() if len(sys.argv) > 1 else "both")
    if mode in ("urls", "both"):
        ingest_urls(URLS)
    if mode in ("files", "both"):
        if LOCAL_FILES:
            ingest_files(LOCAL_FILES)
        else:
            print("[info] No LOCAL_FILES configured; skipping files.")
