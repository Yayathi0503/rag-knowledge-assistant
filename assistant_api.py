"""Search a Markdown handbook and optionally summarize retrieved passages with Ollama."""
from functools import lru_cache
import os
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer


class Index:
    def __init__(self, directory):
        self.passages = []
        for path in sorted(directory.glob("*.md")):
            for number, paragraph in enumerate(path.read_text(encoding="utf-8").split("\n\n"), 1):
                if paragraph.strip():
                    self.passages.append({"source": f"{path.name}#{number}", "text": paragraph.strip()})
        if not self.passages:
            raise ValueError("Add at least one Markdown paragraph to the corpus")
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform(p["text"] for p in self.passages)

    def search(self, question, limit):
        scores = (self.matrix @ self.vectorizer.transform([question]).T).toarray().ravel()
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])
        return [self.passages[i] for i in ranked[:limit] if scores[i] >= 0.12]


@lru_cache(maxsize=1)
def handbook():
    return Index(Path(__file__).with_name("handbook"))


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=1000, pattern=r"\S")
    limit: int = Field(default=2, ge=1, le=5)
    generate: bool = False


app = FastAPI(title="Handbook assistant")


@app.post("/ask")
def ask(request: Question):
    evidence = handbook().search(request.question, request.limit)
    if not evidence:
        return {"answer": "No matching passage found.", "mode": "no_match", "sources": []}
    context = "\n\n".join(f"[{p['source']}] {p['text']}" for p in evidence)
    if not request.generate:
        return {"answer": context, "sources": evidence, "mode": "extractive"}
    model = os.getenv("OLLAMA_MODEL")
    if not model:
        raise HTTPException(503, "Set OLLAMA_MODEL to an installed local model")
    try:
        response = httpx.post("http://127.0.0.1:11434/api/generate", timeout=60,
                              json={"model": model, "stream": False,
                                    "system": "Answer using only the supplied excerpts. Cite passage IDs. If insufficient, say so.",
                                    "prompt": f"Excerpts:\n{context}\n\nQuestion: {request.question}"})
        response.raise_for_status()
        answer = response.json()["response"]
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("Empty model answer")
    except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
        raise HTTPException(502, "The local model could not return an answer") from exc
    return {"answer": answer, "sources": evidence, "mode": "generated"}
