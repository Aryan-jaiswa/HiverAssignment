import os
import pandas as pd
from src.retrieval.api import RetrievalSystem

def evaluate_retrieval(df: pd.DataFrame, retriever: RetrievalSystem, relevance_threshold: float = 0.85):
    """
    Evaluates the retriever using Pseudo-Relevance.
    A retrieved message is considered a 'hit' if its cosine similarity to the query is >= threshold.
    """
    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_5 = 0
    mrr_sum = 0.0
    
    total = len(df)
    print(f"\nEvaluating Pseudo-Relevance (Threshold > {relevance_threshold}) on {total} queries...")
    
    for _, row in df.iterrows():
        query = row['customer_message']
        results = retriever.retrieve_similar_cases(query, top_k=5, threshold=0.0)
        
        # Determine relevance
        # Because the golden_set comes from the test set, it is strictly isolated from the FAISS train index.
        first_relevant_rank = -1
        for rank, res in enumerate(results):
            if res['similarity'] >= relevance_threshold:
                first_relevant_rank = rank + 1 # 1-indexed
                break
                
        if first_relevant_rank != -1:
            if first_relevant_rank == 1:
                hits_at_1 += 1
            if first_relevant_rank <= 3:
                hits_at_3 += 1
            if first_relevant_rank <= 5:
                hits_at_5 += 1
                
            mrr_sum += (1.0 / first_relevant_rank)
            
    print("\n=== Retrieval Evaluation Results ===")
    print(f"Recall@1: {hits_at_1 / total:.4f}")
    print(f"Recall@3: {hits_at_3 / total:.4f}")
    print(f"Recall@5: {hits_at_5 / total:.4f}")
    print(f"MRR:      {mrr_sum / total:.4f}")
    print("====================================")


def main():
    golden_path = "data/processed/golden_set.csv"
    if not os.path.exists(golden_path):
        print(f"Error: {golden_path} not found. Please run Phase 2 and sample the golden set.")
        return
        
    if not os.path.exists("artifacts/retrieval.index"):
        print("Error: FAISS index not found. Please run build_index.py first.")
        return
        
    df = pd.read_csv(golden_path).dropna(subset=['customer_message'])
    
    print("Loading Retrieval System...")
    retriever = RetrievalSystem()
    
    # 1. Investigate Similarity Thresholds
    print("\nInvestigating Score Distribution for top-1 match across test set...")
    top_1_scores = []
    for msg in df['customer_message'].head(100):
        res = retriever.retrieve_similar_cases(msg, top_k=1, threshold=0.0)
        if res:
            top_1_scores.append(res[0]['similarity'])
            
    if top_1_scores:
        avg_score = sum(top_1_scores) / len(top_1_scores)
        print(f"Average Top-1 Cosine Similarity: {avg_score:.4f}")
        print(f"Max Top-1 Cosine Similarity:     {max(top_1_scores):.4f}")
        print(f"Min Top-1 Cosine Similarity:     {min(top_1_scores):.4f}")
        
    # 2. Run Evaluation
    evaluate_retrieval(df, retriever, relevance_threshold=0.85)

if __name__ == "__main__":
    main()
