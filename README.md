# MarketLens
A lightweight financial analytics API (FastAPI) for stock and portfolio metrics.

## Run locally
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
uvicorn marketlens.main:app --reload --app-dir src