#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
fresh_venv=0
if [[ ! -d .venv ]]; then
  python3 -m venv .venv
  fresh_venv=1
fi
source .venv/bin/activate
if [[ "$fresh_venv" -eq 1 || "${H3_UPDATE_DEPS:-0}" == "1" ]]; then
  python -m pip install -U pip
  python -m pip install -r requirements-gui.txt
fi
python h3_gui.py "$@"
