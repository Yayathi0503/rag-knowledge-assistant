from fastapi.testclient import TestClient
import httpx
import pytest
import assistant_api as api

client = TestClient(api.app)


def test_ranked_evidence():
    result = client.post("/ask", json={"question": "deadline for travel reimbursement", "limit": 1}).json()
    assert result["sources"][0]["source"] == "expenses.md#1"
    assert "30 days" in result["answer"]
    assert result["mode"] == "extractive"


def test_no_match_does_not_call_generator():
    result = client.post("/ask", json={"question": "xylophone", "generate": True}).json()
    assert result["mode"] == "no_match" and result["sources"] == []


@pytest.mark.parametrize("payload", [{"question": " "}, {"question": "release", "limit": 0}])
def test_validation(payload):
    assert client.post("/ask", json=payload).status_code == 422


def test_missing_model(monkeypatch):
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    assert client.post("/ask", json={"question": "rollback", "generate": True}).status_code == 503


def test_model_adapter_with_stub(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "test")
    def fake_post(url, *, timeout, json):
        assert "releases.md#2" in json["prompt"]
        return httpx.Response(200, json={"response": "Restore the last tagged release."},
                              request=httpx.Request("POST", url))
    monkeypatch.setattr(api.httpx, "post", fake_post)
    result = client.post("/ask", json={"question": "rollback", "generate": True}).json()
    assert result["mode"] == "generated"
    assert result["sources"][0]["source"] == "releases.md#2"
