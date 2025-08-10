# Fencing RAG Chat (Ollama-only) — Q&A + rulebook + optional web_corpus
# Prereqs:
#   ollama pull mistral:instruct
#   ollama pull nomic-embed-text
#   pip install requests numpy

import json
import requests
import numpy as np
from pathlib import Path
from typing import List, Tuple

# ---------- Config ----------
OLLAMA_URL = "http://localhost:11434"
CHAT_MODEL = "mistral:instruct"
EMBED_MODEL = "nomic-embed-text"

TOP_K = 4
CONTEXT_MAX_CHARS = 7000
TEMPERATURE = 0.2

SYSTEM_PROMPT = (
    "You are a U.S. fencing rules assistant for foil, epee, and sabre. "
    "Prefer USA Fencing (USAF) domestic eligibility/qualification guidance over international (FIE) unless the user explicitly asks about FIE/World Cups. "
    "Quote rule/eligibility phrases if present. If unsure, say you’re unsure."
)

# ---------- Paths ----------
ROOT = Path(__file__).resolve().parents[1]   # .../Fencing AI
DATA = ROOT / "data"
QNA_PATH = DATA / "fencing_qna_dataset.json"
CHUNKS_PATH = DATA / "fencing_rulebook_chunks.json"
WEB_CORPUS_PATH = DATA / "web_corpus.jsonl"  # optional

# ---------- IO ----------
def must_exist(p: Path, label: str):
    if not p.exists():
        raise FileNotFoundError(f"Missing {label} at: {p}")

def load_qna(path: Path = QNA_PATH) -> Tuple[List[str], List[str]]:
    must_exist(path, "Q&A dataset (JSON)")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    qas = data.get("q_and_a", [])
    questions = [str(x.get("question", "")).strip() for x in qas]
    answers   = [str(x.get("answer", "")).strip() for x in qas]
    return questions, answers

def load_rulebook_chunks(path: Path = CHUNKS_PATH) -> Tuple[List[str], List[str], List[str]]:
    must_exist(path, "rulebook chunks (JSON)")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    texts, ids, sources = [], [], []
    for i, rec in enumerate(data):
        texts.append((rec.get("text") or "").strip())
        ids.append(rec.get("id", f"rule-{i}"))
        sources.append(rec.get("source", "rulebook"))
    return ids, texts, sources

def load_web_corpus(path: Path = WEB_CORPUS_PATH):
    if not path.exists():
        return [], [], [], [], []
    ids, texts, sources, titles, tags_list = [], [], [], [], []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            obj = json.loads(line)
            ids.append(obj.get("id", f"web-{len(ids)}"))
            texts.append((obj.get("text") or "").strip())
            sources.append(obj.get("source", ""))
            titles.append(obj.get("title", ""))
            tags_list.append(obj.get("tags", []))
    return ids, texts, sources, titles, tags_list

# ---------- Ollama helpers ----------
def ollama_embed(texts: List[str]) -> np.ndarray:
    vecs = []
    for t in texts:
        r = requests.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": t},
            timeout=120,
        )
        r.raise_for_status()
        vecs.append(np.asarray(r.json().get("embedding", []), dtype=np.float32))
    return np.vstack(vecs) if vecs else np.zeros((0, 1), dtype=np.float32)

def cosine_topk(qvec: np.ndarray, mat: np.ndarray, k: int):
    if mat.size == 0:
        return np.array([], dtype=int), np.array([], dtype=float)
    q = qvec / (np.linalg.norm(qvec) + 1e-8)
    M = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8)
    sims = M @ q
    idxs = np.argsort(-sims)[:k]
    return idxs, sims[idxs]

def ollama_chat(messages, model: str = CHAT_MODEL, temperature: float = TEMPERATURE) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "options": {"temperature": temperature},
        "stream": False,  # avoid NDJSON parsing
    }
    r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=180)
    r.raise_for_status()
    data = r.json()
    if "message" in data and "content" in data["message"]:
        return data["message"]["content"].strip()
    if "content" in data:
        return data["content"].strip()
    if "response" in data:
        return data["response"].strip()
    return ""

# ---------- Retrieval helpers ----------
def needs_usafocus(query: str) -> bool:
    q = query.lower()
    return any(k in q for k in ["div 1", "division 1", "division i", "qualify", "qualification", "nac", "national"])

def boosted_web_hits(qvec, web_matrix, web_texts, web_sources, web_titles, web_tags, top_k):
    idxs, sims = cosine_topk(qvec, web_matrix, max(top_k * 2, 6))
    hits = []
    for i, s in zip(idxs, sims):
        src = (web_sources[i] or "").lower()
        title = (web_titles[i] or "")
        tags = web_tags[i] or []
        boost = 0.0
        if "usafencing.org" in src: boost += 0.12
        if "official" in tags or "usaf" in tags: boost += 0.10
        if "qualification" in tags or "div1" in tags: boost += 0.08
        if any(k in title.lower() for k in ["eligibility", "qualification", "division", "nac"]): boost += 0.05
        hits.append({"i": int(i), "score": float(s) + boost})
    hits = sorted(hits, key=lambda x: x["score"], reverse=True)[:top_k]
    return hits

# ---------- Context + search ----------
def build_context(q_answers: List[str], passages: List[str]) -> str:
    parts = []
    if q_answers: parts.append("Q&A:\n" + "\n".join(q_answers))
    if passages: parts.append("\n\nReferences:\n" + "\n".join(passages))
    ctx = "".join(parts)
    return ctx[:CONTEXT_MAX_CHARS]

def search_all(
    query_vec: np.ndarray,
    question_matrix: np.ndarray, answers: List[str],
    rule_texts: List[str], rule_matrix: np.ndarray, rule_sources: List[str],
    web_texts: List[str], web_matrix: np.ndarray, web_sources: List[str], web_titles: List[str], web_tags: List[List[str]],
    query: str,
    top_k: int = TOP_K,
):
    qi, _ = cosine_topk(query_vec, question_matrix, top_k) if question_matrix.size else (np.array([], dtype=int), [])
    retrieved_answers = [answers[i] for i in qi]

    ri, rsim = cosine_topk(query_vec, rule_matrix, top_k) if rule_matrix.size else (np.array([], dtype=int), [])
    rule_hits = [{"text": rule_texts[i], "source": "rulebook", "score": float(rsim[j])} for j, i in enumerate(ri)]

    if web_matrix.size:
        if needs_usafocus(query):
            whits = boosted_web_hits(query_vec, web_matrix, web_texts, web_sources, web_titles, web_tags, top_k)
            web_hits = [{"text": web_texts[h["i"]], "source": web_sources[h["i"]], "score": h["score"]} for h in whits]
        else:
            wi, wsim = cosine_topk(query_vec, web_matrix, top_k)
            web_hits = [{"text": web_texts[i], "source": web_sources[i], "score": float(wsim[j])} for j, i in enumerate(wi)]
    else:
        web_hits = []

    blended = sorted(rule_hits + web_hits, key=lambda x: x["score"], reverse=True)[:top_k]
    retrieved_passages = [h["text"] for h in blended]
    citations = [h["source"] for h in blended]
    return retrieved_answers, retrieved_passages, citations

# ---------- QA pipeline ----------
def answer_query(
    query: str,
    question_matrix: np.ndarray, answers: List[str],
    rule_texts: List[str], rule_matrix: np.ndarray, rule_sources: List[str],
    web_texts: List[str], web_matrix: np.ndarray, web_sources: List[str], web_titles: List[str], web_tags: List[List[str]],
) -> str:
    qvec = ollama_embed([query])[0]
    qa, refs, cites = search_all(
        qvec,
        question_matrix, answers,
        rule_texts, rule_matrix, rule_sources,
        web_texts, web_matrix, web_sources, web_titles, web_tags,
        query,
        top_k=TOP_K,
    )
    context = build_context(qa, refs)
    user_msg = (
        "Use this context to answer the fencing question.\n\n"
        f"{context}\n\n"
        f"Question: {query}\n"
        "Answer in 1–3 sentences. If uncertain, say so."
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_msg},
    ]
    ans = ollama_chat(messages)
    if cites:
        uniq = []
        for c in cites:
            if c and c not in uniq:
                uniq.append(c)
        if uniq:
            ans += "\n\nSources: " + " | ".join(uniq[:5])
    return ans

# ---------- Boot ----------
def main():
    print("Loading data...")
    questions, answers = load_qna()
    rule_ids, rule_texts, rule_sources = load_rulebook_chunks()
    web_ids, web_texts, web_sources, web_titles, web_tags = load_web_corpus()

    print(f"Q&A pairs: {len(answers)} | Rulebook chunks: {len(rule_texts)} | Web chunks: {len(web_texts)}")

    print("Embedding Q&A questions with Ollama...")
    question_matrix = ollama_embed(questions) if questions else np.zeros((0, 1), dtype=np.float32)

    print("Embedding rulebook chunks with Ollama...")
    rule_matrix = ollama_embed(rule_texts) if rule_texts else np.zeros((0, 1), dtype=np.float32)

    if web_texts:
        print("Embedding web corpus with Ollama...")
    web_matrix = ollama_embed(web_texts) if web_texts else np.zeros((0, 1), dtype=np.float32)

    print("🤺 Fencing Chatbot (Ollama-only)")
    print(f"Data dir: {DATA}")
    print("Type 'exit' to quit.")
    while True:
        try:
            q = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if q.lower() in {"exit", "quit"}:
            break
        if not q:
            continue
        try:
            ans = answer_query(
                q,
                question_matrix, answers,
                rule_texts, rule_matrix, rule_sources,
                web_texts, web_matrix, web_sources, web_titles, web_tags,
            )
            print("Bot:", ans)
        except Exception as e:
            print(f"[ERROR] {e}")

if __name__ == "__main__":
    main()
