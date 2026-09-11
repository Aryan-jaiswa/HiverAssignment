import pytest
import numpy as np
from src.intent.baseline import MajorityBaseline, TfIdfLogisticBaseline
from src.intent.classifier import EmbeddingClassifier
from src.intent.taxonomy import is_valid_intent

def test_majority_baseline():
    clf = MajorityBaseline()
    texts = ["hello", "world", "test"]
    labels = ["device_troubleshooting", "account_access", "account_access"]
    
    clf.fit(texts, labels)
    preds = clf.predict(["unknown message"])
    
    # Should predict the majority class
    assert preds[0] == "account_access"
    
    # Confidence should be 1.0
    probs = clf.predict_proba(["unknown message"])
    assert probs.shape == (1, 1)
    assert probs[0][0] == 1.0

def test_tfidf_baseline():
    clf = TfIdfLogisticBaseline(random_state=42)
    texts = ["broken screen", "forgot password", "screen is cracked", "what is my password"]
    labels = ["device_troubleshooting", "account_access", "device_troubleshooting", "account_access"]
    
    clf.fit(texts, labels)
    
    # Empty message handling
    preds = clf.predict(["", None, "   "])
    assert len(preds) == 3
    assert all(isinstance(p, str) for p in preds)

def test_embedding_classifier():
    clf = EmbeddingClassifier()
    texts = ["broken screen", "forgot password"]
    labels = ["device_troubleshooting", "account_access"]
    
    clf.fit(texts, labels)
    
    preds = clf.predict_with_confidence(["my screen is shattered"])
    assert len(preds) == 1
    assert "intent" in preds[0]
    assert "confidence" in preds[0]
    
    assert preds[0]['intent'] == "device_troubleshooting"
    assert 0.0 <= preds[0]['confidence'] <= 1.0
    
def test_zero_shot_embedding_classifier():
    clf = EmbeddingClassifier()
    # Assuming config/intent_schema.yaml exists
    clf.fit_zero_shot("config/intent_schema.yaml")
    
    preds = clf.predict_with_confidence(["I forgot my password"])
    assert len(preds) == 1
    assert is_valid_intent(preds[0]['intent'], "config/intent_schema.yaml")
