#!/usr/bin/env bash
# rq_eval — one-time target-machine setup (Python 3.11 + Bedrock access).
#
# Idempotent. After this, configure config.yaml (providers.mode: live, model
# ids, guardrail id/version, region/profile) and run: python smoke_test.py
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$here"

PY="${PYTHON:-python3}"
echo ">> Using interpreter: $($PY --version)"

echo ">> Installing offline core (requirements.txt)…"
"$PY" -m pip install --upgrade pip
"$PY" -m pip install -r requirements.txt

echo ">> Installing this package (editable)…"
"$PY" -m pip install -e .

echo ">> Installing live deps (requirements-live.txt)…"
"$PY" -m pip install -r requirements-live.txt

echo ">> Downloading spaCy English model (en_core_web_lg)…"
"$PY" -m spacy download en_core_web_lg

echo ">> Installing coreferee English model…"
"$PY" -m coreferee install en || echo "!! coreferee model install failed — check coreferee/spaCy versions"

echo ">> (optional) local NLI via fairseq…"
if "$PY" -m pip install "torch>=2.2,<3.0" "fairseq==0.12.2"; then
  echo "   fairseq installed — you may set models.nli: fairseq"
else
  echo "!! fairseq install failed — SKIPPING. Use models.nli: bedrock instead."
fi

echo ">> Freezing resolved versions to requirements.lock…"
"$PY" -m pip freeze > requirements.lock

cat <<'EOF'

Install complete. Next steps:
  1. cp .env.example .env   and fill in AWS profile/keys (or rely on SSO)
  2. Edit config.yaml: providers.mode: live, aws.*, models.* (judge/embed/
     guardrail id+version), models.nli: bedrock (or fairseq if installed above)
  3. python smoke_test.py   # must pass every provider before evaluating
  4. pytest                 # optional: run suite once in live mode
EOF
