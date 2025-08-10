# app.py — Fencing AI (ChatGPT-style UI, no sidebar)
# Run: streamlit run app.py

import os, json, time
from pathlib import Path
from typing import List
import numpy as np
import requests
import streamlit as st

# =========================
# Config (no sidebar knobs)
# =========================
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "mistral:instruct")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
QNA_PATH = DATA / "fencing_qna_dataset.json"
CHUNKS_PATH = DATA / "fencing_rulebook_chunks.json"
WEB_CORPUS_PATH = DATA / "web_corpus.jsonl"  # optional

TOP_K = 4
TEMPERATURE = 0.2
CONTEXT_MAX_CHARS = 7000

SYSTEM_PROMPT = (
    "You are a U.S. fencing rules assistant for foil, epee, and sabre. "
    "Prefer USA Fencing (USAF) domestic eligibility/qualification guidance over international (FIE) unless the user explicitly asks about FIE/World Cups. "
    "Quote rule/eligibility phrases if present. If unsure, say you’re unsure."
)

# =========================
# Helpers
# =========================
def must_exist(p: Path, label: str):
    if not p.exists():
        raise FileNotFoundError(f"Missing {label} at: {p}")

@st.cache_resource(show_spinner=True)
def load_all_data():
    must_exist(QNA_PATH, "Q&A dataset (JSON)")
    must_exist(CHUNKS_PATH, "rulebook chunks (JSON)")

    # Q&A
    with QNA_PATH.open("r", encoding="utf-8") as f:
        qdata = json.load(f).get("q_and_a", [])
    questions = [str(item.get("question","")).strip() for item in qdata]
    answers   = [str(item.get("answer","")).strip() for item in qdata]

    # Rulebook chunks
    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        cdata = json.load(f)
    rule_texts   = [str(item.get("text","")).strip() for item in cdata]
    rule_ids     = [item.get("id", f"rule-{i}") for i, item in enumerate(cdata)]
    rule_sources = [item.get("source","rulebook") for item in cdata]

    # Optional web corpus
    web_ids, web_texts, web_sources, web_titles, web_tags = [], [], [], [], []
    if WEB_CORPUS_PATH.exists():
        with WEB_CORPUS_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                obj = json.loads(line)
                web_ids.append(obj.get("id",""))
                web_texts.append((obj.get("text") or "").strip())
                web_sources.append(obj.get("source",""))
                web_titles.append(obj.get("title",""))
                web_tags.append(obj.get("tags",[]))

    return (questions, answers,
            rule_texts, rule_ids, rule_sources,
            web_texts, web_sources, web_titles, web_tags)

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

@st.cache_resource(show_spinner=True)
def build_embeddings(questions, rule_texts, web_texts):
    qM = ollama_embed(questions) if questions else np.zeros((0,1), dtype=np.float32)
    rM = ollama_embed(rule_texts) if rule_texts else np.zeros((0,1), dtype=np.float32)
    wM = ollama_embed(web_texts) if web_texts else np.zeros((0,1), dtype=np.float32)
    return qM, rM, wM

def cosine_topk(qvec, mat, k):
    if mat.size == 0: return np.array([],dtype=int), np.array([],dtype=float)
    q = qvec / (np.linalg.norm(qvec)+1e-8)
    M = mat / (np.linalg.norm(mat,axis=1,keepdims=True)+1e-8)
    sims = M @ q
    idxs = np.argsort(-sims)[:k]
    return idxs, sims[idxs]

def needs_usafocus(query: str) -> bool:
    q = query.lower()
    return any(k in q for k in [
        "y10","y12","y14","y-10","y-12","y-14",
        "syc","ryc","summer nationals",
        "div 1","division 1","division i",
        "qualify","qualification","nac","national"
    ])

def boosted_web_hits(qvec, web_matrix, web_texts, web_sources, web_titles, web_tags, top_k):
    idxs, sims = cosine_topk(qvec, web_matrix, max(top_k*2, 6))
    hits = []
    for i, s in zip(idxs, sims):
        src = (web_sources[i] or "").lower()
        title = (web_titles[i] or "")
        tags = web_tags[i] or []
        boost = 0.0
        if "usafencing.org" in src: boost += 0.12
        if "official" in tags or "usaf" in tags: boost += 0.10
        if "qualification" in tags or "div1" in tags: boost += 0.08
        if any(k in (title.lower()) for k in ["eligibility","qualification","division","nac","summer nationals","y10","y12","y14"]):
            boost += 0.05
        hits.append({"i": int(i), "score": float(s)+boost})
    hits = sorted(hits, key=lambda x: x["score"], reverse=True)[:top_k]
    return hits

def build_context(q_answers: List[str], passages: List[str]) -> str:
    parts=[]
    if q_answers: parts.append("Q&A:\n" + "\n".join(q_answers))
    if passages: parts.append("\n\nReferences:\n" + "\n".join(passages))
    return ("".join(parts))[:CONTEXT_MAX_CHARS]

def ollama_chat(messages, model=CHAT_MODEL, temperature=TEMPERATURE) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "options": {"temperature": temperature},
        "stream": False
    }
    r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=180)
    r.raise_for_status()
    data = r.json()
    if "message" in data and "content" in data["message"]:
        return data["message"]["content"].strip()
    if "content" in data: return data["content"].strip()
    if "response" in data: return data["response"].strip()
    return ""

def answer_query(query: str,
                 questions, answers,
                 rule_texts, rule_ids, rule_sources,
                 web_texts, web_sources, web_titles, web_tags,
                 qM, rM, wM):
    # Embed query
    qvecs = ollama_embed([query])
    if qvecs.size == 0:
        return "Sorry, I couldn't embed your query. Check Ollama is running.", []

    qvec = qvecs[0]

    # Q&A
    qi, _ = cosine_topk(qvec, qM, TOP_K) if qM.size else (np.array([],dtype=int), [])
    retrieved_answers = [answers[i] for i in qi] if len(qi) else []

    # Rulebook
    ri, rsim = cosine_topk(qvec, rM, TOP_K) if rM.size else (np.array([],dtype=int), [])
    rule_hits = []
    if len(ri):
        for j, i in enumerate(ri):
            txt = rule_texts[i] if i < len(rule_texts) else ""
            rid = rule_ids[i] if i < len(rule_ids) else f"rule-{int(i)}"
            rsrc = rule_sources[i] if i < len(rule_sources) else "rulebook"
            rule_hits.append({
                "id": rid, "text": txt, "source": rsrc,
                "score": float(rsim[j]) if len(rsim) > j else 0.0
            })

    # Web
    web_hits = []
    if wM.size and len(web_texts):
        if needs_usafocus(query):
            whits = boosted_web_hits(qvec, wM, web_texts, web_sources, web_titles, web_tags, TOP_K)
            for h in whits:
                i = h["i"]
                web_hits.append({
                    "id": f"web-{i}",
                    "text": web_texts[i] if i < len(web_texts) else "",
                    "source": web_sources[i] if i < len(web_sources) else "",
                    "score": h["score"]
                })
        else:
            wi, wsim = cosine_topk(qvec, wM, TOP_K)
            for j, i in enumerate(wi):
                web_hits.append({
                    "id": f"web-{int(i)}",
                    "text": web_texts[i] if i < len(web_texts) else "",
                    "source": web_sources[i] if i < len(web_sources) else "",
                    "score": float(wsim[j]) if len(wsim) > j else 0.0
                })

    # Blend
    blended = sorted(rule_hits + web_hits, key=lambda x: x["score"], reverse=True)[:TOP_K]
    refs  = [h.get("text","") for h in blended if h.get("text")]
    cites = [h.get("source","") for h in blended if h.get("source")]

    # Build prompt
    context = build_context(retrieved_answers, refs)
    user_msg = (
        "Use this fencing context to answer the question.\n\n"
        f"{context}\n\n"
        f"Question: {query}\n"
        "Answer in 1–3 sentences. If uncertain, say so."
    )
    messages = [{"role":"system","content":SYSTEM_PROMPT},
                {"role":"user","content":user_msg}]
    ans = ollama_chat(messages, model=CHAT_MODEL, temperature=TEMPERATURE)

    # Unique sources
    uniq = []
    for c in cites:
        if c and c not in uniq:
            uniq.append(c)

    return ans, uniq

def render_sources(sources: List[str]):
    if not sources:
        return
    with st.expander("Show sources"):
        for s in sources[:8]:
            if s.startswith("http"):
                st.markdown(f"- [{s}]({s})")
            else:
                st.markdown(f"- {s}")

# =========================
# UI (Chat-style, no sidebar)
# =========================
st.set_page_config(page_title="Fencing AI (Local)", page_icon="🤺", layout="centered")

st.markdown(
    "<h2 style='text-align:center'>🤺 Fencing AI — Local RAG</h2>"
    "<p style='text-align:center;margin-top:-8px;'>USAF-focused answers about rules, youth qualification, penalties, and technique.</p>",
    unsafe_allow_html=True,
)

# Chat history
if "chat" not in st.session_state:
    st.session_state.chat = []

def add_msg(role, content, sources=None):
    st.session_state.chat.append({
        "role": role,
        "content": content,
        "sources": sources or [],
        "time": time.time()
    })

# Load + embed on first use (cached)
(questions, answers,
 rule_texts, rule_ids, rule_sources,
 web_texts, web_sources, web_titles, web_tags) = load_all_data()
qM, rM, wM = build_embeddings(questions, rule_texts, web_texts)

# Render history
for m in st.session_state.chat:
    with st.chat_message("user" if m["role"]=="user" else "assistant",
                         avatar="👤" if m["role"]=="user" else "🤺"):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            render_sources(m.get("sources", []))

# Input
prompt = st.chat_input("Ask about youth qualification, rules, penalties, technique…")
if prompt:
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
    add_msg("user", prompt)

    with st.chat_message("assistant", avatar="🤺"):
        with st.spinner("Thinking…"):
            try:
                answer, sources = answer_query(
                    prompt,
                    questions, answers,
                    rule_texts, rule_ids, rule_sources,
                    web_texts, web_sources, web_titles, web_tags,
                    qM, rM, wM
                )
            except Exception as e:
                answer, sources = f"Error: {e}", []
        st.markdown(answer)
        render_sources(sources)
        add_msg("assistant", answer, sources)
