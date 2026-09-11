# Engineering Decision Log

This document records the critical, non-obvious engineering decisions made during the construction of the AppleSupport AI Agent.

## Phase 1: Data Pipeline
1. **Conversation-Level Splitting**
   - **Decision**: Forced `train_test_split` to operate on `conversation_id` rather than individual messages.
   - **Why**: To prevent data leakage. If a customer sends three messages in one conversation, randomly splitting them could put message 1 in Train and message 3 in Test, artificially inflating retrieval performance.
   - **Alternative Considered**: Message-level splitting (simpler to implement).
   - **Tradeoff**: Takes more pandas grouping complexity, but strictly enforces zero-contamination evaluation.

2. **Storing Tweet IDs as Strings**
   - **Decision**: Converted `tweet_id` to `str` rather than keeping it as `int64`.
   - **Why**: Twitter IDs exceed Javascript's maximum safe integer limit. If the API outputs them as integers, frontend systems or other tools will truncate the last digits silently.
   - **Alternative Considered**: Keep as `int64` for faster pandas indexing.
   - **Tradeoff**: Very slight memory overhead in Python, but prevents catastrophic data corruption in downstream web clients.

## Phase 2: Intent Taxonomy
3. **Random Golden Set Sourcing**
   - **Decision**: Sampled the 200 golden examples *exclusively* from the isolated `test` split.
   - **Why**: Ensures zero contamination. If we annotated training data and evaluated on it, the LLM or retriever would essentially memorize the answers.
   - **Alternative Considered**: Sampling from the whole dataset before splitting.
   - **Tradeoff**: Required building the split pipeline earlier than initially anticipated.

4. **Coarse Intent Schema**
   - **Decision**: Drafted a coarse schema of 10 categories (e.g., `device_troubleshooting`, `account_access`) instead of 50+ micro-intents.
   - **Why**: Fine-grained intents require massive annotated datasets to train reliably. A coarse taxonomy covers 90% of support volume while remaining learnable and easily verifiable by humans.
   - **Alternative Considered**: 50-category hierarchical taxonomy.
   - **Tradeoff**: Less granularity in routing, but vastly higher classification accuracy.

5. **TF-IDF + KMeans for Discovery**
   - **Decision**: Used MiniBatchKMeans on TF-IDF vectors for initial dataset exploration instead of LLM-based clustering.
   - **Why**: Speed and privacy. TF-IDF over 1,000 features runs in seconds locally, providing enough signal to identify broad topics without sending 100,000 tweets to an external API.
   - **Alternative Considered**: OpenAI Embeddings + HDBSCAN.
   - **Tradeoff**: Lower semantic nuance during discovery, but zero API cost and instantaneous feedback.

## Phase 3: Classification
6. **Sentence-Transformers Centroid Classifier**
   - **Decision**: Chose `all-MiniLM-L6-v2` with a nearest-centroid approach for the main classifier.
   - **Why**: Extremely fast, runs entirely on CPU, requires zero fine-tuning, and provides robust zero-shot capability by simply embedding the schema descriptions.
   - **Alternative Considered**: Fine-tuning BERT or DistilBERT.
   - **Tradeoff**: Slightly lower ceiling performance than a fully fine-tuned model, but vastly superior maintainability and speed.

7. **Macro F1 as Primary Metric**
   - **Decision**: Evaluated classification primarily on Macro F1 rather than Accuracy.
   - **Why**: Customer support intents are highly imbalanced (e.g., 'complaint' is rare compared to 'device_troubleshooting'). Macro F1 ensures we measure performance across *all* intents equally.
   - **Alternative Considered**: Raw Accuracy.
   - **Tradeoff**: The reported metric number will look "lower" to non-technical stakeholders, but it honestly reflects the system's safety on minority classes.

8. **Confidence Calibration Split**
   - **Decision**: Required a validation split specifically to calibrate the classification threshold.
   - **Why**: Selecting escalation thresholds on the test set is a classic data leakage error that artificially inflates reported production performance.
   - **Alternative Considered**: Hardcoding a 0.5 threshold or tuning on the test set.
   - **Tradeoff**: Shrinks the available training data slightly, but provides statistically valid generalization bounds.

## Phase 4: Retrieval
9. **FAISS IndexFlatIP**
   - **Decision**: Selected `IndexFlatIP` (exact inner product search on L2-normalized vectors) over approximate indices like HNSW or IVF.
   - **Why**: Our training subset is small enough (<100k items) that brute-force exact search is practically instantaneous on a CPU.
   - **Alternative Considered**: `IndexHNSWFlat`.
   - **Tradeoff**: Perfect accuracy without the complexity of tuning approximation hyperparameters, though it won't scale efficiently past 1 million records without migrating to HNSW.

10. **Embedding Caching**
    - **Decision**: Cached raw training embeddings to an `.npy` file during index building.
    - **Why**: The initial embedding pass across thousands of queries takes time. Caching allows instantaneous rebuilding of the FAISS index if metadata structures are altered.
    - **Alternative Considered**: Re-embedding every time `build_index.py` is run.
    - **Tradeoff**: Costs a few hundred MB of disk space, but saves minutes of CPU time during active development.

11. **Pseudo-Relevance Evaluation Proxy**
    - **Decision**: Used a very high cosine similarity threshold (e.g., >0.85) as a proxy for "relevance" when calculating Recall@K on the Golden Set.
    - **Why**: There is no hand-labeled boolean relevance between queries. This provides an objective, code-driven way to measure Recall metrics securely on the held-out golden set without requiring hundreds of hours of manual query-matching annotations.
    - **Alternative Considered**: Having an LLM judge relevance for every retrieved pair.
    - **Tradeoff**: It's a proxy; occasionally a high-similarity query might not have exactly the same solution, but it avoids fabricating ground-truth labels.

## Phase 5: Generation & Escalation
12. **Defense-in-Depth Escalation Engine**
    - **Decision**: Placed deterministic rules (keywords, confidence scores, intent risk) *before* the LLM, and schema validation *after* the LLM.
    - **Why**: An LLM should not be the sole arbiter of automation safety. High-risk financial intents must be escalated deterministically before wasting tokens or risking hallucinations.
    - **Alternative Considered**: Asking the LLM to decide if it should escalate.
    - **Tradeoff**: Some safe queries might be falsely escalated by the strict rules, prioritizing safety over maximum automation rate.

13. **Classification vs Automation Confidence Separation**
    - **Decision**: Tracked intent confidence and retrieval similarity independently from the LLM's generative confidence.
    - **Why**: A model might be 99% confident that a query is about `payment_billing` (classification), but it might lack the historical evidence to actually resolve it (automation).
    - **Alternative Considered**: Combining them into a single "Agent Confidence" score.
    - **Tradeoff**: Requires managing multiple thresholds, but provides granular visibility into *why* the system escalated.

14. **LLM Provider Abstraction**
    - **Decision**: Used an abstract base class `LLMProvider` rather than hardcoding `google-generativeai` calls.
    - **Why**: Prevents vendor lock-in. Allows effortless swapping to OpenAI, Anthropic, or local vLLM instances simply by passing a new provider class.
    - **Alternative Considered**: Direct `genai` API calls throughout the code.
    - **Tradeoff**: Requires writing slightly more boilerplate interface code.

## Phase 6: Evaluation
15. **Safe Automation Rate Definition**
    - **Decision**: Defined `Safe Automation Rate = Correctly Auto-handled / Total Cases` where "correct" requires the intent to be perfectly classified AND the LLM Judge scoring the response >= 3.
    - **Why**: An automation rate that ignores response safety is a vanity metric. If the system hallucinates, it is not "automated"; it is "broken".
    - **Alternative Considered**: Raw Automation Rate (Cases not escalated / Total).
    - **Tradeoff**: Makes the system's performance look lower on paper, but reflects true business value.

16. **Quadratic Cohen's Kappa for Agreement**
    - **Decision**: Chose Quadratic Cohen's Kappa for Human vs LLM Agreement over simple exact match.
    - **Why**: On a 0-4 scale, confusing a 3 for a 4 is less severe than confusing a 0 for a 4. Quadratic Kappa heavily penalizes extreme disagreements, providing a mathematically robust measure of inter-rater reliability.
    - **Alternative Considered**: Exact Match Accuracy.
    - **Tradeoff**: Harder to explain to non-technical stakeholders than "80% agreement", but statistically far superior.

17. **Two-Step Human Evaluation Orchestration**
    - **Decision**: Built a specific script (`sample_human_eval.py`) that dumps an empty CSV for the human engineer to score manually before the final evaluation runs.
    - **Why**: To adhere strictly to the rule against fabricating metrics, the system must wait for explicit ground-truth labels from a human rather than generating simulated numbers.
    - **Alternative Considered**: Randomly generating mock human scores in the test script.
    - **Tradeoff**: Slows down the evaluation loop since it requires a human-in-the-loop pause, but guarantees authentic compliance.
