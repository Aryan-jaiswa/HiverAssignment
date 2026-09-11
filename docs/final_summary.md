# AppleSupport AI Agent - Final Summary

## Architecture
The AppleSupport AI Agent is a defense-in-depth customer support pipeline. It combines:
1. **Sentence-Transformers Classification**: Fast, CPU-bound intent routing.
2. **FAISS Retrieval-Augmented Generation (RAG)**: Exact historical evidence matching.
3. **Deterministic Escalation Pre-checks**: Hardcoded safety rules that intercept dangerous intents or low-confidence predictions before LLM generation.
4. **Grounded Gemini LLM**: Synthesizes historical evidence into a coherent response.

## Final Metrics (Golden Set)
*(To be populated after running `scripts/evaluate.py`)*
- **Classification Macro F1**: [METRIC]
- **Escalation Recall**: [METRIC]
- **Safe Automation Rate**: [METRIC]
- **Human vs LLM Agreement (Cohen's Kappa)**: [METRIC]

## Biggest Strengths
- **Safety First**: The deterministic escalation engine prevents the LLM from hallucinating on novel or high-risk queries.
- **Reproducibility**: The evaluation pipeline prevents data leakage via strict conversation-level dataset splitting.
- **Explainability**: The system outputs not just a response, but the exact historical cases and confidence metrics that led to its decision.

## Biggest Weaknesses
- **Multi-Intent Failure**: The centroid classifier struggles with queries containing multiple conflicting intents.
- **Corpus Dependence**: The system is completely reliant on the historical FAISS index. If a novel issue arises, the system will reliably escalate, but it will achieve a 0% automation rate on that new issue until the FAISS index is updated.

## Prototype vs. Production
**IMPORTANT**: This is a prototype take-home assignment, NOT a production-ready system. 
- **Production Requirements Missing**: PII/PHI scrubbing before embedding, Multi-tenant indexing, Asynchronous queueing (e.g., Celery/Redis) for high-throughput scaling, and Temporal drift evaluation.

## Recommended Next Steps
1. **Implement PII Scrubbing**: Add a Presidio or Regex pipeline to strip phone numbers, names, and credit cards before text reaches the embedding models.
2. **Upgrade FAISS**: Migrate from `IndexFlatIP` to `IndexHNSWFlat` to handle datasets > 1,000,000 tweets efficiently.
3. **Continuous Evaluation**: Implement a shadow-deployment pipeline where the AI scores its own Safe Automation Rate in the background on live human traffic to detect temporal drift.
