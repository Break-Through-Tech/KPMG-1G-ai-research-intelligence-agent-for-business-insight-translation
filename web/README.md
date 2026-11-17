# Web interface

ChatGPT-style UI for the pipeline. Chat history is saved in the browser (localStorage), no database.

## Run (two terminals, from the repo root)

1. Build the index once if you haven't:
   python src/parse_and_chunk.py && python src/build_index.py

2. Start the API (needs GOOGLE_API_KEY in .env):
   pip install -r requirements.txt
   uvicorn api:app --app-dir src --port 8000

3. Start the UI:
   cd web && npm install && npm run dev
   open http://localhost:3000

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
`demo/2608.20318v1_AI4AI-Bench.pdf` is a sample paper. With the API and UI running, drag it into the chat, wait for indexing to finish, then ask a question about it (e.g. "What does AI4AI-Bench evaluate?"). The filename keeps the arXiv ID so the citation links back to the paper.
