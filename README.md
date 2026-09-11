# AppleSupport AI Agent

A robust, defense-in-depth ML pipeline for automating customer support on Twitter. Built for the Hiver Take-Home Assignment.

## 1. Project Overview & Problem Statement
This project addresses the challenge of automating Twitter customer support for `AppleSupport`. The primary problem is balancing **Automation Rate** with **Safety**. Large Language Models (LLMs) hallucinate refund policies or dangerous technical advice when confused. This architecture solves that by wrapping the LLM in a rigid deterministic Escalation Engine and grounding its responses in retrieved historical FAISS evidence.

## 2. Architecture Diagram
```
Customer Message -> Embedding Classifier -> Intent + Confidence -> 
FAISS Retrieval -> Historical Evidence -> Escalation Pre-checks -> 
(If Safe) Gemini LLM Generation -> Validation -> Final Reply
(If Unsafe) -> ESCALATE TO HUMAN
```

## 3. Dataset & Preprocessing
- **Brand Selection**: `AppleSupport` was chosen due to its high volume and diverse intent range (hardware, software, billing) in the dataset.
- **Acquisition**: Download the `twcs.csv` dataset from [Kaggle](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter). Place it at `data/raw/twcs.csv`.
- **Preprocessing**: `scripts/build_dataset.py` cleans text, strips URLs/usernames, and structures the data into `customer_message` and `support_reply` pairs.
- **Leakage Prevention**: Train/Test splitting is strictly enforced at the **conversation level**, preventing related messages in a single thread from cross-contaminating the evaluation sets.

## 4. Intent Taxonomy & Golden Set
A 10-category taxonomy (e.g., `device_troubleshooting`, `account_access`) was built. A 200-example Golden Evaluation Set was sampled entirely from the isolated test set to ensure rigorous zero-shot evaluation.

## 5. Baselines vs Main System
- **Baselines**: Implemented Majority Class and TF-IDF + Logistic Regression baselines.
- **Main Classifier**: `all-MiniLM-L6-v2` Sentence-Transformers centroid classifier.
- **Retrieval**: `faiss-cpu` exact inner-product search over historical reference sets.
- **Generation**: Gemini 1.5 Pro via an abstract provider interface.

## 6. Evaluation & Real Results
*(Placeholder metrics to be populated by the evaluator upon running the scripts)*
- **Classification Macro F1**: `[METRIC]`
- **Safe Automation Rate**: `[METRIC]`
- **Human-LLM Agreement (Cohen's Kappa)**: `[METRIC]`

## 7. Misleading Headline Metrics & Limitations
A 90% Accuracy metric is dangerously misleading in customer support. If the model predicts "Troubleshooting" for every case, it misses critical 2% "Billing" complaints. See `docs/misleading_numbers.md` for a comprehensive critique of vanity metrics. 
**Limitations**: The system struggles with multi-intent queries and cannot answer novel questions missing from the FAISS index.

## 8. Project Structure
- `src/`: Core ML, API, and Evaluation code.
- `scripts/`: Execution scripts for building indices and evaluating.
- `tests/`: Pytest suite covering all logic.
- `docs/`: Engineering Decision Log and Reports.
- `artifacts/`: JSON metric outputs and failure analyses.

## 9. Setup Instructions
1. Install Python 3.10+
2. `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and add your `GEMINI_API_KEY`.
4. Place the Kaggle dataset at `data/raw/twcs.csv`.

## 10. How to Reproduce Results (Under 15 Minutes)
```bash
# 1. Build the dataset and train/test splits
python scripts/build_dataset.py

# 2. Extract Golden Set and Intents
python scripts/explore_intents.py
python scripts/sample_golden_set.py

# 3. Build FAISS Index
python scripts/build_index.py

# 4. Generate Human Evaluation Sample (Annotate it!)
python scripts/sample_human_eval.py

# 5. Run the Master Evaluation
python scripts/evaluate.py
```
Check `artifacts/evaluation/` for all final metrics.

## 11. How to Run Demo / API
**API (FastAPI)**:
```bash
uvicorn src.api:app --reload
```

**UI (Streamlit)**:
```bash
streamlit run src.ui.py
```

## 12. Testing
```bash
pytest
```
