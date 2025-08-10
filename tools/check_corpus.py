from pathlib import Path
import json, collections

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CORPUS = DATA / "web_corpus.jsonl"

def main():
    if not CORPUS.exists():
        print("No corpus yet:", CORPUS)
        return
    n = 0
    by_source = collections.Counter()
    with CORPUS.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            obj = json.loads(line)
            n += 1
            src = obj.get("source", "")
            by_source[src] += 1
    print(f"Total chunks: {n}")
    for src, cnt in by_source.most_common(30):
        print(f"{cnt:5d}  {src}")

if __name__ == "__main__":
    main()
