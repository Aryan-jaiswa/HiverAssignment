import os
import yaml
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans

def explore_clusters(df: pd.DataFrame, num_clusters: int = 10):
    """
    Uses TF-IDF and KMeans to cluster customer messages and prints top keywords.
    """
    print(f"Exploring {len(df)} customer messages using {num_clusters} clusters...")
    
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X = vectorizer.fit_transform(df['customer_message'].fillna(''))
    
    kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=42, n_init=3)
    kmeans.fit(X)
    
    df['cluster'] = kmeans.labels_
    
    # Get top keywords per cluster
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names_out()
    
    os.makedirs("reports", exist_ok=True)
    with open("reports/intent_exploration.txt", "w", encoding="utf-8") as f:
        for i in range(num_clusters):
            top_words = [terms[ind] for ind in order_centroids[i, :10]]
            cluster_size = len(df[df['cluster'] == i])
            
            summary = f"Cluster {i} (Size: {cluster_size}): {', '.join(top_words)}\n"
            print(summary.strip())
            f.write(summary)
            
            # Print a few examples
            examples = df[df['cluster'] == i]['customer_message'].head(3).tolist()
            for ex in examples:
                f.write(f"  - {ex}\n")
            f.write("\n")

def main():
    try:
        with open("config/config.yaml", "r") as f:
            config = yaml.safe_load(f)['data']
            
        train_path = os.path.join(config['processed_dir'], f"{config['brand'].lower()}_train.csv")
        
        if not os.path.exists(train_path):
            print(f"Error: {train_path} not found. Please run Phase 1 first.")
            return
            
        df = pd.read_csv(train_path)
        explore_clusters(df, num_clusters=10)
        print("\nExploration saved to reports/intent_exploration.txt")
        print("Use these clusters as a starting point to refine config/intent_schema.yaml.")
        
    except Exception as e:
        print(f"Error during exploration: {e}")

if __name__ == "__main__":
    main()
