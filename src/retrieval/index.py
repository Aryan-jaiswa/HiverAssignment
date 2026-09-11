import os
import json
import faiss
import numpy as np

def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Builds a FAISS index using Inner Product (equivalent to Cosine Similarity if vectors are normalized).
    """
    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D numpy array.")
        
    dimension = embeddings.shape[1]
    
    # Normalize vectors for cosine similarity
    faiss.normalize_L2(embeddings)
    
    # Use IndexFlatIP for exact search using inner product (cosine similarity on normalized vectors)
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    
    return index

def save_index_and_metadata(index: faiss.Index, metadata: list, index_path: str, meta_path: str):
    """
    Saves the FAISS index and corresponding metadata list to disk.
    """
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)
    
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

def load_index_and_metadata(index_path: str, meta_path: str):
    """
    Loads the FAISS index and metadata.
    """
    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        raise FileNotFoundError(f"Index or metadata not found at {index_path} / {meta_path}")
        
    index = faiss.read_index(index_path)
    with open(meta_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
        
    return index, metadata
