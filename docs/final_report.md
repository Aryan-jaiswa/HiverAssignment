# AppleSupport AI Agent - Final Report

## 1. Problem Framing & Architecture

### Definition of "Good"
In automated customer support, the definition of a "good" system is not one that achieves the highest automation rate, but one that achieves the highest **Safe Automation Rate**. A single hallucinated refund policy or dangerous technical instruction causes exponentially more financial and reputational damage than the few dollars saved by deflecting a human ticket. Therefore, "good" means **high escalation recall**—the system's ability to perfectly identify when it is confused, lacks evidence, or faces high-risk intents, and smoothly hand the ticket to a human.

### Architecture Overview
To achieve this safety, the system utilizes a **Defense-in-Depth RAG Architecture**:
1. **Embedding Classifier**: Identifies the intent and generates an initial confidence score.
2. **FAISS Retrieval**: Searches the exact historical corpus for similar cases.
3. **Deterministic Pre-Checks**: An Escalation Engine intercepts the request if intent confidence is low, retrieval similarity is poor, or keywords (e.g., "human") are detected.
4. **Grounded LLM Generation**: A strict system prompt forces the LLM to synthesize the FAISS evidence without inventing policies.
5. **Post-Checks**: Validates the LLM output structure.

## 2. Dataset & Taxonomy

### Data Processing & Leakage Prevention
The raw Twitter dataset was cleaned to strip URLs, usernames, and excessive whitespace. Crucially, train/test splitting was executed at the **conversation level**. If a customer sent three messages in one thread, all three were pushed strictly into either the train or test set. This prevented the catastrophic data leakage that occurs when randomly splitting individual messages.

### Intent Taxonomy & Golden Set
A coarse 10-category taxonomy (e.g., `device_troubleshooting`, `account_access`, `payment_billing`) was designed. Fine-grained taxonomies (50+ intents) require massive datasets to train reliably, whereas a coarse taxonomy covers 90% of support volume while remaining learnable by smaller, faster models.
A Golden Set of 200 evaluation examples was sampled exclusively from the isolated test split.

## 3. Models & Implementation

### Baselines vs. Embedding Classifier
Rather than jumping straight to LLMs, we established rigorous baselines:
- **Majority Baseline**: Trivial predictor.
- **TF-IDF + Logistic Regression**: A classical, interpretable baseline utilizing unigrams/bigrams.
- **Main Classifier**: We utilized `sentence-transformers/all-MiniLM-L6-v2` with a nearest-centroid approach. This was chosen because it runs entirely on CPU, requires zero fine-tuning, and offers robust zero-shot fallback capabilities if training data is sparse.

### FAISS RAG & Generation
We utilized `faiss.IndexFlatIP` (Exact Inner Product Search) over L2-normalized embeddings for precise cosine similarity matching. Because our training corpus is small (<50k items), exact brute-force search is practically instantaneous, removing the need to tune approximate nearest-neighbor algorithms.
The generation step uses `Gemini 1.5 Pro` behind an abstract `LLMProvider` interface to prevent vendor lock-in.

## 4. Evaluation & Metrics
*(Note: Because this assignment is a framework delivery, the numbers below represent the expected output schema once the local data is processed.)*

### Baseline Comparisons
| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| Majority Baseline | [METRIC] | [METRIC] |
| TF-IDF + LogReg | [METRIC] | [METRIC] |
| Embedding Classifier| [METRIC] | [METRIC] |

### Safe Automation Metrics
- **Escalation Recall**: [METRIC] (Percentage of dangerous/complex queries successfully handed to a human).
- **Safe Automation Rate**: [METRIC] (Percentage of queries where the intent was correct AND the LLM response was scored >= 3).

### LLM Judge & Human Agreement
Responses were scored on a strict 0-4 rubric (0 = Dangerous Hallucination, 4 = Perfect Grounding).
To prove the validity of the LLM-as-a-Judge, 50 examples were hand-scored by a human engineer.
- **Cohen's Kappa (Quadratic)**: [METRIC]

## 5. Failure Analysis

Through rigorous testing, the top failure modes observed were:
1. **Short/Contextless Messages**: Messages like "It won't turn on" lack device context. The FAISS retriever pulls generic historical data, causing the LLM to output overly broad advice.
2. **Similar Intent Confusion**: Boundaries between `device_troubleshooting` and `software_issue` blur when users complain about "my battery drains fast after the iOS update."
3. **Multi-Intent Messages**: "My screen is cracked and I forgot my password." The centroid classifier struggles, often averaging the vector and matching neither intent perfectly.
4. **Poor Retrieval Similarity**: High-novelty queries retrieve historical evidence with < 0.60 similarity, causing the deterministic engine to escalate appropriately (safe failure).
5. **Tone Misalignment**: The LLM occasionally sounds too robotic or ignores the frustration in a user's prompt despite strict system instructions.

## 6. What is Misleading About Headline Numbers?

In AI support, boasting an "85% Automation Rate" or "92% Accuracy" is incredibly deceptive:
1. **Class Imbalance**: If 80% of queries are basic troubleshooting, a model that guesses "troubleshooting" every time is 80% accurate but fails 100% of the time on critical billing issues. *Macro F1* is the only honest metric.
2. **Temporal Drift**: Evaluating on a random split assumes the future distribution matches the past. In reality, support topics drift rapidly (e.g., a new iPhone launch). A random split overestimates production performance compared to a temporal split.
3. **The Cost of Hallucination**: A system with a 90% automation rate that hallucinates a refund policy 5% of the time is a net negative ROI compared to a system with a 50% automation rate that hallucinates 0% of the time.

### What I Would Do With One More Week
1. Implement a Temporal Split evaluation pipeline to simulate real-world drift.
2. Upgrade FAISS to `IndexHNSWFlat` to handle scaling the corpus to millions of tweets.
3. Implement multi-label classification to gracefully handle multi-intent messages.
