@echo off
chcp 65001 >nul
echo ========================================================
echo   Stick3D Shorts Studio — 100%% Gratuit ^& Sans Filigrane
echo ========================================================
echo Verification / installation des modules Python gratuits...
python -m pip install -q -r requirements.txt
echo.
echo Demarrage du Studio sur http://127.0.0.1:8000 ...
start "" http://127.0.0.1:8000
python -m uvicorn app:app --host 127.0.0.1 --port 8000
pause
