import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

class MajorityBaseline:
    """
    Trivial baseline that always predicts the most frequent intent seen during training.
    """
    def __init__(self):
        self.majority_class = None
        
    def fit(self, texts: list, labels: list):
        if not labels:
            raise ValueError("Labels list is empty")
        
        # Find the most frequent class
        from collections import Counter
        counts = Counter(labels)
        self.majority_class = counts.most_common(1)[0][0]
        
    def predict(self, texts: list) -> list:
        if self.majority_class is None:
            raise RuntimeError("Model is not fitted yet.")
        return [self.majority_class] * len(texts)
    
    def predict_proba(self, texts: list) -> np.ndarray:
        # Trivial confidence of 1.0 for the majority class
        return np.ones((len(texts), 1))


class TfIdfLogisticBaseline:
    """
    Baseline using TF-IDF and Logistic Regression.
    """
    def __init__(self, random_state: int = 42):
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                ngram_range=(1, 2), 
                max_features=5000, 
                stop_words='english'
            )),
            ('clf', LogisticRegression(
                class_weight='balanced', 
                random_state=random_state,
                max_iter=1000
            ))
        ])
        
    def fit(self, texts: list, labels: list):
        # Filter out empty texts
        clean_texts, clean_labels = [], []
        for t, l in zip(texts, labels):
            if isinstance(t, str) and t.strip():
                clean_texts.append(t)
                clean_labels.append(l)
                
        if not clean_texts:
            raise ValueError("No valid texts to train on.")
            
        self.pipeline.fit(clean_texts, clean_labels)
        
    def predict(self, texts: list) -> list:
        safe_texts = [t if isinstance(t, str) else "" for t in texts]
        return self.pipeline.predict(safe_texts).tolist()
        
    def predict_proba(self, texts: list) -> np.ndarray:
        safe_texts = [t if isinstance(t, str) else "" for t in texts]
        return self.pipeline.predict_proba(safe_texts)
