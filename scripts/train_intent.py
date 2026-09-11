import os
import pandas as pd
import joblib

from src.intent.baseline import MajorityBaseline, TfIdfLogisticBaseline
from src.intent.classifier import EmbeddingClassifier

def main():
    os.makedirs("models", exist_ok=True)
    
    # 1. Train Zero-Shot Embedding Classifier (Needs no training data!)
    print("Initializing Zero-Shot Embedding Classifier from intent_schema...")
    emb_clf = EmbeddingClassifier()
    try:
        emb_clf.fit_zero_shot("config/intent_schema.yaml")
        joblib.dump(emb_clf, "models/embedding_classifier.joblib")
        print("Saved models/embedding_classifier.joblib")
    except Exception as e:
        print(f"Error training EmbeddingClassifier: {e}")

    # 2. Train Supervised Baselines (Majority and TF-IDF)
    # This requires a labeled training set which is beyond Phase 2's manual annotations.
    train_path = "data/processed/applesupport_train_annotated.csv"
    
    if not os.path.exists(train_path):
        print(f"\n[WARNING] {train_path} not found.")
        print("TF-IDF and Majority Baselines require supervised training data.")
        print("To fully execute these baselines, please create and annotate a training dataset.")
        return
        
    print(f"\nLoading training data from {train_path}...")
    df = pd.read_csv(train_path)
    
    # We expect 'customer_message' and 'intent' columns
    if 'intent' not in df.columns:
        print("Error: 'intent' column missing from training data.")
        return
        
    # Drop rows without intent
    df = df.dropna(subset=['customer_message', 'intent'])
    texts = df['customer_message'].tolist()
    labels = df['intent'].tolist()
    
    print("Training Majority Baseline...")
    maj_clf = MajorityBaseline()
    maj_clf.fit(texts, labels)
    joblib.dump(maj_clf, "models/majority_baseline.joblib")
    
    print("Training TF-IDF + Logistic Regression Baseline...")
    tfidf_clf = TfIdfLogisticBaseline()
    tfidf_clf.fit(texts, labels)
    joblib.dump(tfidf_clf, "models/tfidf_baseline.joblib")
    
    print("All models successfully trained and saved to models/")

if __name__ == "__main__":
    main()
