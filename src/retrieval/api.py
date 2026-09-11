import faiss
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

from src.retrieval.index import load_index_and_metadata

class RetrievalSystem:
    """
    Wrapper around the FAISS index and embedding model to easily query historical cases.
    """
    def __init__(self, index_path: str = "artifacts/retrieval.index", 
                 meta_path: str = "artifacts/retrieval_metadata.json",
                 model_name: str = "all-MiniLM-L6-v2"):
        
        self.index, self.metadata = load_index_and_metadata(index_path, meta_path)
        self.model = SentenceTransformer(model_name)
        
    def retrieve_similar_cases(self, message: str, top_k: int = 5, threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Retrieves the top_k most similar historical cases.
        Returns an empty list if the top result's similarity is strictly less than the threshold.
        """
        if not isinstance(message, str) or not message.strip():
            return []
            
        # Encode and normalize
        embedding = self.model.encode([message], show_progress_bar=False)
        faiss.normalize_L2(embedding)
        
        # Search index
        similarities, indices = self.index.search(embedding, top_k)
        
        results = []
        # Filter by threshold and format output
        for sim, idx in zip(similarities[0], indices[0]):
            if idx == -1:
                continue # FAISS returns -1 if not enough results
            if float(sim) >= threshold:
                meta = self.metadata[idx]
                results.append({
                    "customer_message": meta.get("customer_message", ""),
                    "support_reply": meta.get("support_reply", ""),
                    "conversation_id": meta.get("conversation_id", ""),
                    "tweet_id": meta.get("customer_tweet_id", ""),
                    "similarity": float(sim)
                })
                
        return results
