import pytest
import numpy as np
import os
import json
import faiss

from src.retrieval.index import build_faiss_index, save_index_and_metadata, load_index_and_metadata
from src.retrieval.api import RetrievalSystem

@pytest.fixture
def mock_embeddings():
    # 3 mock embeddings of dimension 4
    np.random.seed(42)
    embs = np.random.rand(3, 4).astype('float32')
    return embs

@pytest.fixture
def mock_metadata():
    return [
        {"customer_message": "Hello", "support_reply": "Hi"},
        {"customer_message": "Broken screen", "support_reply": "Send it in"},
        {"customer_message": "Forgot password", "support_reply": "Reset it"}
    ]

def test_build_faiss_index(mock_embeddings):
    index = build_faiss_index(mock_embeddings)
    assert index.is_trained
    assert index.ntotal == 3

def test_save_and_load_index(tmp_path, mock_embeddings, mock_metadata):
    index = build_faiss_index(mock_embeddings)
    idx_path = str(tmp_path / "test.index")
    meta_path = str(tmp_path / "test_meta.json")
    
    save_index_and_metadata(index, mock_metadata, idx_path, meta_path)
    
    loaded_idx, loaded_meta = load_index_and_metadata(idx_path, meta_path)
    assert loaded_idx.ntotal == 3
    assert len(loaded_meta) == 3
    assert loaded_meta[1]['customer_message'] == "Broken screen"

def test_retrieval_system_threshold(tmp_path, mock_embeddings, mock_metadata, monkeypatch):
    # We mock the retrieval system to avoid loading actual sentence transformers and models in pure unit tests
    idx_path = str(tmp_path / "test.index")
    meta_path = str(tmp_path / "test_meta.json")
    
    index = build_faiss_index(mock_embeddings)
    save_index_and_metadata(index, mock_metadata, idx_path, meta_path)
    
    class MockModel:
        def encode(self, texts, **kwargs):
            return np.random.rand(1, 4).astype('float32')
            
    # Mock SentenceTransformer initialization to return MockModel
    monkeypatch.setattr("src.retrieval.api.SentenceTransformer", lambda x: MockModel())
    
    retriever = RetrievalSystem(index_path=idx_path, meta_path=meta_path)
    
    # Test top k
    res = retriever.retrieve_similar_cases("test", top_k=2)
    assert len(res) <= 2
    
    # Test thresholding (if threshold > 1.0, it should return empty since cosine sim is <= 1.0)
    res_empty = retriever.retrieve_similar_cases("test", threshold=1.5)
    assert len(res_empty) == 0

def test_empty_input():
    # If input is empty, API should handle safely
    # Using a mock API object so we don't need real weights
    class DummyRetriever:
        from src.retrieval.api import RetrievalSystem
        retrieve_similar_cases = RetrievalSystem.retrieve_similar_cases
        
        def __init__(self):
            pass
            
    r = DummyRetriever()
    assert r.retrieve_similar_cases(r, "") == []
    assert r.retrieve_similar_cases(r, None) == []
