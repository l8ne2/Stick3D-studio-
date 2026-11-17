#!/usr/bin/env bash
set -e
echo "========================================================"
echo "  Stick3D Shorts Studio — 100% Gratuit & Sans Filigrane"
echo "========================================================"
python3 -m pip install -q -r requirements.txt
echo "Ouverture sur http://127.0.0.1:8000 ..."
(sleep 2 && (xdg-open http://127.0.0.1:8000 2>/dev/null || open http://127.0.0.1:8000 2>/dev/null || true)) &
python3 -m uvicorn app:app --host 127.0.0.1 --port 8000
