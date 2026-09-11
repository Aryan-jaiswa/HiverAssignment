import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

from src.retrieval.index import build_faiss_index, save_index_and_metadata

def main():
    train_path = "data/processed/applesupport_train.csv"
    if not os.path.exists(train_path):
        print(f"Error: {train_path} not found. Please run Phase 1 first.")
        return
        
    print(f"Loading training data from {train_path}...")
    df = pd.read_csv(train_path)
    
    # Filter out empty messages
    df = df.dropna(subset=['customer_message', 'support_reply'])
    
    # Take a subset if the dataset is massive, to keep memory low, but here we embed all
    # To keep it fast locally without GPU, we'll embed up to 50k rows
    max_rows = 50000
    if len(df) > max_rows:
        print(f"Subsampling {len(df)} rows down to {max_rows} for FAISS index build speed.")
        df = df.sample(n=max_rows, random_state=42)
        
    messages = df['customer_message'].tolist()
    
    print("Loading SentenceTransformer model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    print(f"Embedding {len(messages)} historical customer queries...")
    # Caching embeddings locally to prevent recompute if index needs rebuild
    emb_cache_path = "artifacts/train_embeddings.npy"
    if os.path.exists(emb_cache_path):
        print("Loading embeddings from cache...")
        embeddings = np.load(emb_cache_path)
    else:
        print("Computing embeddings... (this may take a minute)")
        embeddings = model.encode(messages, show_progress_bar=True)
        os.makedirs("artifacts", exist_ok=True)
        np.save(emb_cache_path, embeddings)
        print("Embeddings cached.")
        
    print("Building FAISS index...")
    index = build_faiss_index(embeddings)
    
    print("Constructing metadata...")
    metadata = df.to_dict('records')
    
    # Strip unnecessary heavy columns if they exist
    clean_meta = []
    for row in metadata:
        clean_meta.append({
            "customer_message": row.get('customer_message', ''),
            "support_reply": row.get('support_reply', ''),
            "conversation_id": str(row.get('conversation_id', '')),
            "customer_tweet_id": str(row.get('customer_tweet_id', ''))
        })
        
    print("Saving index and metadata to artifacts/...")
    save_index_and_metadata(
        index, 
        clean_meta, 
        "artifacts/retrieval.index", 
        "artifacts/retrieval_metadata.json"
    )
    
    print("Index successfully built and saved!")

if __name__ == "__main__":
    main()
