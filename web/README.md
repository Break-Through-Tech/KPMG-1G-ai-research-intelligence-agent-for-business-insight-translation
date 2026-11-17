# Web interface

ChatGPT-style UI for the pipeline. Chat history is saved in the browser (localStorage), no database.

## Run locally

Prerequisites: Python 3.10+, Node 18+ (check with `python3 --version` and `node --version`).

### 1. Get a free LLM key
Pick one:
- **Groq (recommended, no card needed):** sign in at https://console.groq.com/keys, click *Create API Key*, copy it.
- **Gemini:** key from https://aistudio.google.com/apikey.

### 2. Backend setup (from the repo root)
```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # large (PyTorch), takes a few minutes
cp .env.example .env
```
Edit `.env` (it is gitignored, never commit it). For Groq:
```
LLM_PROVIDER=groq
GROQ_API_KEY=<your key>
```
For Gemini, leave `LLM_PROVIDER` empty and set `GOOGLE_API_KEY=<your key>`. The other variables are optional.

Build the search index once (the first run downloads the embedding model):
```bash
python src/parse_and_chunk.py && python src/build_index.py
```

### 3. Start the API (terminal 1, venv active, repo root)
```bash
uvicorn api:app --app-dir src --port 8000
```
Check http://localhost:8000/health returns ok.

### 4. Start the UI (terminal 2)
```bash
cd web && npm install && npm run dev
```
Open http://localhost:3000. The UI expects the API on port 8000 and is served from port 3000 (CORS only allows that origin).

## How it connects
web/lib/assistant.ts -> POST /chat (src/api.py) -> Retriever.search + summarize() -> answer + sources.
No pipeline logic lives in the web layer, so improvements to retrieve.py / summarize.py show up in the UI automatically.

## Limits
- Single-turn: each question is answered on its own; earlier messages aren't sent to the model.
- History lives in one browser on one machine.

## Adding PDFs
Drag PDFs onto the chat (or click +). The API saves them to `data/` and reruns `parse_and_chunk.py` and `build_index.py`, so it takes a minute.
- Name arXiv papers like `2608.20319v1_Title.pdf` so citations and links work. Other PDFs are searchable but show as "Uploaded PDF" without an arXiv link.
- Uploads with no usable text (scanned, protected, or a duplicate of an indexed paper) are skipped.
- `data/*.pdf` is gitignored, so uploads stay on your machine.

## Try it
`demo/2611.21577v1_When Tool Names Are Not Identities...pdf` is a sample paper that is not in the seeded corpus. With the API and UI running, drag it into the chat and wait for indexing to finish (about a minute), then ask e.g. "What is provider misbinding in MCP-based agents?". The filename keeps the arXiv ID so the citation links back to the paper. Re-dropping a paper that is already indexed is skipped as a duplicate.

## Troubleshooting
- `GROQ_API_KEY is not set` / `GOOGLE_API_KEY is not set`: `.env` must be in the repo root and match `LLM_PROVIDER`.
- Groq rate-limit errors: the free tier is limited, wait a minute and retry.
- Empty answers or "index not found": run the `parse_and_chunk.py` and `build_index.py` step again.
- Port 8000 or 3000 already in use: stop the other process, the UI and CORS are hard-wired to these ports.
