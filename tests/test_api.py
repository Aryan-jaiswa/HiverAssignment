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
