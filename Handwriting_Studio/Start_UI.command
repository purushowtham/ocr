#!/bin/zsh
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
  .venv/bin/python -m pip install -r ui_requirements.txt
fi
export PORT=7861
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
export XDG_CACHE_HOME="$PWD/.cache"
echo "Open http://127.0.0.1:7861 when startup completes."
exec .venv/bin/python app.py
