#!/usr/bin/env bash
# run_mac.sh — one-command setup & run for macOS
set -euo pipefail

proj="$(cd "$(dirname "$0")"/.. && pwd)"   # repo root (Mac/..)
cd "$proj"

say_step() { printf "\n\033[36m==> %s\033[0m\n" "$1"; }

# 1) Homebrew (optional, helps install Python/Ollama)
if ! command -v brew >/dev/null 2>&1; then
  say_step "Homebrew not found. Installing (optional)…"
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" || true
  # Add brew to PATH if installed
  if [ -x /opt/homebrew/bin/brew ]; then eval "$(/opt/homebrew/bin/brew shellenv)"; fi
  if [ -x /usr/local/bin/brew ]; then eval "$(/usr/local/bin/brew shellenv)"; fi
fi

# 2) Ollama
if ! command -v ollama >/dev/null 2>&1; then
  say_step "Installing Ollama…"
  if command -v brew >/dev/null 2>&1; then
    brew install ollama || true
  fi
  if ! command -v ollama >/dev/null 2>&1; then
    say_step "Falling back to direct download (requires user approval)…"
    curl -L "https://ollama.com/download/Ollama-darwin.zip" -o /tmp/Ollama-darwin.zip
    ditto -xk /tmp/Ollama-darwin.zip /tmp/ollama_app
    open /tmp/ollama_app/Ollama.app
    echo "Finish installing Ollama (drag to Applications if prompted), then press Enter."
    read -r _
  fi
fi

# 3) Start Ollama if not running
OLLAMA_URL="${OLLAMA_URL:-http://127.0.0.1:11434}"
if ! curl -sSf "$OLLAMA_URL/api/tags" >/dev/null 2>&1; then
  say_step "Starting Ollama server…"
  (ollama serve >/dev/null 2>&1 &) || true
  sleep 5
fi
if ! curl -sSf "$OLLAMA_URL/api/tags" >/dev/null 2>&1; then
  echo "Could not reach Ollama at $OLLAMA_URL. Open the Ollama app or run: ollama serve"
  exit 1
fi

# 4) Python 3 + venv
if ! command -v python3 >/dev/null 2>&1; then
  say_step "Installing Python 3 (via Homebrew)…"
  if command -v brew >/dev/null 2>&1; then
    brew install python
  else
    echo "Please install Python 3 (https://www.python.org/downloads/) and rerun."
    exit 1
  fi
fi

say_step "Preparing virtual environment…"
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip

if [ -f requirements.txt ]; then
  say_step "Installing requirements.txt…"
  pip install -r requirements.txt
else
  say_step "Installing minimal dependencies…"
  pip install streamlit requests numpy beautifulsoup4 lxml html5lib trafilatura
fi

# 5) Pull models if missing
pull_if_missing() {
  local model="$1"
  if ! curl -s "$OLLAMA_URL/api/tags" | grep -q "$model"; then
    say_step "Pulling $model …"
    ollama pull "$model"
  fi
}
CHAT_MODEL="${CHAT_MODEL:-mistral:instruct}"
EMBED_MODEL="${EMBED_MODEL:-nomic-embed-text}"
pull_if_missing "$CHAT_MODEL"
pull_if_missing "$EMBED_MODEL"

# 6) Ensure minimal data for first run
mkdir -p "$proj/data"
[ -f "$proj/data/fencing_qna_dataset.json" ] || cat > "$proj/data/fencing_qna_dataset.json" <<'JSON'
{
  "q_and_a": [
    { "question": "Can a 10-year-old qualify for Summer Nationals?",
      "answer": "Yes. Youth fencers can qualify via SYC/NAC points or required regional points as defined by USA Fencing for that season." },
    { "question": "What is Parry 4 in foil?",
      "answer": "Parry 4 (quarte) closes the high inside line; supinate the hand to deflect the attack to your inside high line." }
  ]
}
JSON

[ -f "$proj/data/fencing_rulebook_chunks.json" ] || cat > "$proj/data/fencing_rulebook_chunks.json" <<'JSON'
[
  { "id": "rule-001", "source": "USAF Rulebook",
    "text": "Covering target is penalized; repeated offenses escalate from yellow to red." },
  { "id": "rule-002", "source": "USAF Rulebook",
    "text": "Right of way in foil: the fencer who starts the attack first has priority unless it fails or is parried." }
]
JSON

# 7) Launch app
say_step "Launching Fencing Coach AI…"
export OLLAMA_URL CHAT_MODEL EMBED_MODEL
streamlit run "$proj/app.py"
