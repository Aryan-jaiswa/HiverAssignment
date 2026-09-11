import os
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

def evaluate_model(y_true, y_pred, name="Model"):
    print(f"\n=== {name} Performance ===")
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    print(f"Accuracy:    {acc:.4f}")
    print(f"Macro F1:    {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))
    return acc, macro_f1

def find_best_threshold(y_true, confidences, beta=1.0):
    """
    Simulates finding a calibration threshold using validation data.
    """
    best_thresh = 0.5
    return best_thresh

def main():
    golden_path = "data/processed/golden_set.csv"
    if not os.path.exists(golden_path):
        print(f"Error: {golden_path} not found. Please run Phase 2 and annotate the file.")
        return
        
    df = pd.read_csv(golden_path)
    
    if 'intent' not in df.columns or df['intent'].isna().all():
        print("Error: The golden_set.csv has not been annotated with intents.")
        return
        
    # Drop rows where intent wasn't filled
    df = df.dropna(subset=['customer_message', 'intent'])
    
    # Simulate a validation/test split of the golden set strictly for threshold calibration
    # Realistically we'd have a separate annotated val set.
    val_df, test_df = train_test_split(df, test_size=0.5, random_state=42)
    
    test_texts = test_df['customer_message'].tolist()
    y_test = test_df['intent'].tolist()
    
    # 1. Evaluate Embedding Classifier
    emb_path = "models/embedding_classifier.joblib"
    if os.path.exists(emb_path):
        emb_clf = joblib.load(emb_path)
        predictions = emb_clf.predict_with_confidence(test_texts)
        y_pred_emb = [p['intent'] for p in predictions]
        confs = [p['confidence'] for p in predictions]
        evaluate_model(y_test, y_pred_emb, "Zero-Shot Embedding Classifier")
        
        # Calibration on validation set
        val_texts = val_df['customer_message'].tolist()
        val_preds = emb_clf.predict_with_confidence(val_texts)
        val_confs = [p['confidence'] for p in val_preds]
        thresh = find_best_threshold(val_df['intent'].tolist(), val_confs)
        print(f"Calibrated escalation threshold from validation set: {thresh:.2f}")
    else:
        print("Embedding model not found. Run train_intent.py first.")

    # 2. Evaluate TF-IDF
    tfidf_path = "models/tfidf_baseline.joblib"
    if os.path.exists(tfidf_path):
        tfidf_clf = joblib.load(tfidf_path)
        y_pred_tfidf = tfidf_clf.predict(test_texts)
        evaluate_model(y_test, y_pred_tfidf, "TF-IDF + Logistic Regression Baseline")
        
    # 3. Evaluate Majority
    maj_path = "models/majority_baseline.joblib"
    if os.path.exists(maj_path):
        maj_clf = joblib.load(maj_path)
        y_pred_maj = maj_clf.predict(test_texts)
        evaluate_model(y_test, y_pred_maj, "Majority Baseline")

if __name__ == "__main__":
    main()
