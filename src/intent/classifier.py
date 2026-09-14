import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()

class EmbeddingClassifier:
    """
    A simple and explainable intent classifier based on prototype/centroid embeddings.
    """
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        # Lightweight model suitable for CPU execution
        self.model = SentenceTransformer(model_name)
        self.centroids = {} # intent -> numpy array
        self.classes = []
        
    def fit(self, texts: list, labels: list):
        """
        Fits the classifier by computing the mean embedding (centroid) for each intent.
        """
        embeddings = self.model.encode(texts, show_progress_bar=False)
        
        unique_labels = list(set(labels))
        self.classes = sorted(unique_labels)
        
        for label in self.classes:
            # Get indices of texts for this label
            indices = [i for i, l in enumerate(labels) if l == label]
            label_embeddings = embeddings[indices]
            
            # Compute centroid (mean embedding)
            centroid = np.mean(label_embeddings, axis=0)
            
            # Normalize centroid to length 1 for simple cosine similarity
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm
                
            self.centroids[label] = centroid

    def fit_zero_shot(self, config_path: str = None):
        """
        Alternative: Zero-shot fitting using just the taxonomy descriptions.
        This is used if there is no annotated training data available.
        """
        config_path = config_path or str(PROJECT_ROOT / "config" / "intent_schema.yaml")
        with open(config_path, "r") as f:
            schema = yaml.safe_load(f)
            
        self.classes = []
        descriptions = []
        
        for intent in schema['intents']:
            self.classes.append(intent['name'])
            # We embed the name + description to form the centroid
            text_rep = f"{intent['name']}: {intent['description']}"
            descriptions.append(text_rep)
            
        embeddings = self.model.encode(descriptions, show_progress_bar=False)
        
        for label, emb in zip(self.classes, embeddings):
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            self.centroids[label] = emb

    def predict_with_confidence(self, texts: list) -> list:
        """
        Predicts the intent and returns a list of dictionaries containing:
        - intent
        - confidence (cosine similarity to centroid)
        """
        safe_texts = [t if isinstance(t, str) and t.strip() else "unknown" for t in texts]
        embeddings = self.model.encode(safe_texts, show_progress_bar=False)
        
        results = []
        
        # Prepare centroid matrix for vectorized similarity computation
        centroid_matrix = np.array([self.centroids[c] for c in self.classes])
        
        # Compute similarities
        similarities = cosine_similarity(embeddings, centroid_matrix)
        
        for i in range(len(texts)):
            best_idx = np.argmax(similarities[i])
            best_intent = self.classes[best_idx]
            confidence = float(similarities[i][best_idx])
            
            results.append({
                'intent': best_intent,
                'confidence': confidence
            })
            
        return results
