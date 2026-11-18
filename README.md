# Handbook knowledge assistant

Ask a question about release procedures or expense policies. The offline API ranks Markdown paragraphs by TF-IDF cosine similarity and returns matching excerpts with filename and paragraph citations. The included handbook is fictional.

## Run the demo

Use Python 3.12. From the repository root, create a virtual environment with `python -m venv .venv`. Activate it (`.venv\Scripts\activate` on Windows or `source .venv/bin/activate` on macOS/Linux), then run:

```sh
python -m pip install -r requirements.txt
python -m pytest -q
```

```sh
python -m uvicorn assistant_api:app
```

Open http://127.0.0.1:8000/docs and send `{"question":"When is travel reimbursement due?","limit":1}` to **POST /ask**. The sample answer quotes the 30-day deadline from `expenses.md#1`. An unrelated term such as “xylophone” returns an empty source list. Add Markdown files in `handbook/` and restart the server to rebuild the index.

## Optional generation

For a generated summary, start an Ollama server on localhost port 11434, install a model of your choice, and set `OLLAMA_MODEL` to its installed name before starting the API. Add `"generate":true` to the request. The adapter sends the excerpts and question to the local model and returns its text alongside the retrieved evidence. Review generated claims against those excerpts.

Offline tests cover retrieval, abstention, validation and a simulated model response. A live model check requires a running Ollama instance. Lexical retrieval can miss paraphrases, and a shared keyword can retrieve an unrelated passage.
