#!/usr/bin/env bash
# One-click launcher for macOS — installs/starts Ollama, sets up Python venv, installs deps, pulls models, launches app.

set -euo pipefail

say()  { printf "\n\033[36m==> %s\033[0m\n" "$1"; }
warn() { printf "\n\033[33m!! %s\033[0m\n" "$1"; }

# Run from repo root (this file lives in Mac/)
REPO_ROOT="$(cd "$(dirname "$0")"/.. && pwd)"
cd "$REPO_ROOT"

# Keep Terminal open on error
trap 'warn "Something failed. Scroll up to see the error."; echo; read -p "Press Return to close… " _' ERR

# Remove quarantine on this file (first run convenience)
xattr -d com.apple.quarantine "Mac/Run Fencing Coach.command" >/dev/null 2>&1 || true

# 1) Homebrew (optional, helps installs)
if ! command -v brew >/dev/null 2>&1; then
  warn "Homebrew not found. It’s optional, but helps with installs."
  [ -x /opt/homebrew/bin/brew ] && eval "$(/opt/homebrew/bin/brew shellenv)" || true
  [ -x /usr/local/bin/brew ] && eval "$(/usr/local/bin/brew shellenv)" || true
fi

# 2) Ollama (install if missing)
if ! command -v ollama >/dev/null 2>&1; then
  say "Installing Ollama…"
  if command -v brew >/dev/null 2>&1; then brew install ollama || true; fi
  if ! command -v ollama >/dev/null 2>&1; then
    say "Falling back to direct download (you may be prompted to allow it)…"
    TMPDIR="$(mktemp -d)"
    curl -fsSL "https://ollama.com/download/Ollama-darwin.zip" -o "$TMPDIR/ollama.zip"
    ditto -xk "$TMPDIR/ollama.zip" "$TMPDIR/app"
    open "$TMPDIR/app/Ollama.app"
    warn "Finish installing Ollama (drag to Applications if asked), then press Return."
    read -r _
  fi
fi

# 3) Start Ollama if not running
OLLAMA_URL="${OLLAMA_URL:-http://127.0.0.1:11434}"
if ! curl -sSf "$OLLAMA_URL/api/tags" >/dev/null 2>&1; then
  say "Starting Ollama server…"
  (ollama serve >/dev/null 2>&1 &) || true
  sleep 5
fi
if ! curl -sSf "$OLLAMA_URL/api/tags" >/dev/null 2>&1; then
  warn "Could not reach Ollama at $OLLAMA_URL. Open the Ollama app or run:  ollama serve"
  exit 1
fi

# 4) Python 3 + venv
if ! command -v python3 >/dev/null 2>&1; then
  say "Installing Python 3…"
  if command -v brew >/dev/null 2>&1; then
    brew install python
  else
    warn "Please install Python 3 from https://www.python.org/downloads/ then re-run."
    exit 1
  fi
fi

say "Preparing virtual environment…"
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip

# 5) Deps
if [ -f requirements.txt ]; then
  say "Installing requirements.txt…"
  pip install -r requirements.txt
else
  say "Installing minimal dependencies…"
  pip install streamlit requests numpy beautifulsoup4 lxml html5lib trafilatura
fi

# 6) Pull models if missing
pull_if_missing() {
  local model="$1"
  if ! curl -s "$OLLAMA_URL/api/tags" | grep -q "$model"; then
    say "Pulling $model …"
    ollama pull "$model"
  fi
}
CHAT_MODEL="${CHAT_MODEL:-mistral:instruct}"
EMBED_MODEL="${EMBED_MODEL:-nomic-embed-text}"
pull_if_missing "$CHAT_MODEL"
pull_if_missing "$EMBED_MODEL"

# 7) Minimal data for first run
mkdir -p "$REPO_ROOT/data"
[ -f "$REPO_ROOT/data/fencing_qna_dataset.json" ] || cat > "$REPO_ROOT/data/fencing_qna_dataset.json" <<'JSON'
{
  "q_and_a": [
    { "question": "Can a 10-year-old qualify for Summer Nationals?",
      "answer": "Yes. Youth fencers can qualify via SYC/NAC points or required regional points as defined by USA Fencing for that season." },
    { "question": "What is Parry 4 in foil?",
      "answer": "Parry 4 (quarte) closes the high inside line; supinate the hand to deflect the attack to your inside high line." }
  ]
}
JSON

[ -f "$REPO_ROOT/data/fencing_rulebook_chunks.json" ] || cat > "$REPO_ROOT/data/fencing_rulebook_chunks.json" <<'JSON'
[
  { "id": "rule-001", "source": "USAF Rulebook",
    "text": "Covering target is penalized; repeated offenses escalate from yellow to red." },
  { "id": "rule-002", "source": "USAF Rulebook",
    "text": "Right of way in foil: the fencer who starts the attack first has priority unless it fails or is parried." }
]
JSON

# 8) Launch GUI
say "Launching Fencing Coach AI…"
export OLLAMA_URL CHAT_MODEL EMBED_MODEL
streamlit run "$REPO_ROOT/app.py"
