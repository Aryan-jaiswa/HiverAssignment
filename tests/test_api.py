import pytest
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

def test_api_empty_input():
    response = client.post("/api/v1/support", json={"message": "   "})
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()

def test_api_missing_field():
    response = client.post("/api/v1/support", json={"wrong_field": "hello"})
    assert response.status_code == 422 # FastAPI validation error

def test_api_valid_input():
    response = client.post("/api/v1/support", json={"message": "My screen is cracked"})
    assert response.status_code == 200
    data = response.json()
    assert "intent" in data
    assert "decision" in data
    assert "reply" in data
    assert "evidence" in data
    assert isinstance(data["evidence"], list)

def test_api_explicit_escalation():
    response = client.post("/api/v1/support", json={"message": "I demand to speak to a human operator"})
    assert response.status_code == 200
    data = response.json()
    assert data["decision"] == "ESCALATE"

def test_api_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_llm_provider_failure(monkeypatch):
    from src.generation.provider import GeminiProvider
    def mock_generate_fail(*args, **kwargs):
        raise RuntimeError("Fake API network error")
    
    provider = GeminiProvider(model_name="test")
    monkeypatch.setattr(provider.client.models, "generate_content", mock_generate_fail)
    
    # generate should catch the exception and return safe escalation dict
    res = provider.generate("test prompt")
    assert res["needs_escalation"] is True
    assert "LLM generation failed" in res["reason"]

def test_malformed_llm_json(monkeypatch):
    from src.generation.provider import GeminiProvider
    class FakeResponse:
        text = "This is not json"
    
    def mock_generate_content(*args, **kwargs):
        return FakeResponse()
    
    provider = GeminiProvider(model_name="test")
    monkeypatch.setattr(provider.client.models, "generate_content", mock_generate_content)
    
    res = provider.generate("test prompt")
    assert res["needs_escalation"] is True
    assert "LLM generation failed" in res["reason"]

def test_intent_classifier_exception(monkeypatch):
    from src.api import get_agent
    agent = get_agent()
    
    def mock_predict(*args, **kwargs):
        raise ValueError("Fake model crash")
        
    monkeypatch.setattr(agent.classifier, "predict_with_confidence", mock_predict)
    
    res = agent.handle_message("Fix my phone")
    assert res["decision"] == "ESCALATE"
    assert res["intent"] == "classifier_failure"
