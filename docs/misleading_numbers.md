# What is Misleading About My Headline Number?

In AI-driven customer support, it is common to boast about a "92% Accuracy" or "85% Automation Rate." However, these headline numbers are extremely misleading when viewed in isolation. Here is an objective analysis of the weaknesses and hidden risks within our system's metrics.

## 1. Class Imbalance and "Accuracy"
Support queries follow a long-tail distribution. For example, 70% of AppleSupport queries might be "device_troubleshooting" (e.g., cracked screens, battery issues), while only 2% are "payment_billing" or "security_complaint". 
If the classifier simply predicts "device_troubleshooting" for every single query, it achieves a headline Accuracy of 70%. But its Macro F1 score would be near zero, and its practical usefulness is negative because it completely fails to identify critical, high-risk financial issues. **Macro F1 is the only honest metric for imbalanced intents.**

## 2. The Golden Set is Too Small
The evaluation was run on a hand-annotated "Golden Set" of ~200 examples. While this prevents train/test leakage, 200 examples across 10 intents means some intents are represented by fewer than 5 examples. An F1 score calculated on 5 examples is highly sensitive to noise and does not guarantee statistical significance.

## 3. High Automation Rate vs. Safety Tradeoff
A system can achieve a 90% Automation Rate simply by removing the escalation thresholds. But what is the cost? 
If the system auto-handles a critical legal threat or hallucinates a refund policy, the financial and reputational damage far outweighs the saved human hours. **High Escalation Recall (catching every dangerous case) is drastically more important than a high Automation Rate.** The "Safe Automation Rate" metric attempts to capture this, but it is limited by the judge's ability to catch hallucinations.

## 4. LLM-as-a-Judge Limitations
We rely on an LLM Judge to calculate the "Safe Automation Rate" and Response Quality. 
- **Blind spots**: LLM Judges suffer from "positional bias" and "verbosity bias" (scoring longer answers higher, even if they are subtly incorrect).
- **Agreement**: If the LLM Judge only agrees with human annotators 60% of the time (Cohen's Kappa < 0.4), the response quality metrics are largely unreliable. This is why the Human Agreement evaluation is mandatory.

## 5. Temporal Drift vs. Random Splits
The Golden Set was built via a *random* split. In reality, support topics drift temporally. A new iOS update drops, and suddenly 50% of queries are about a bug that never existed in the historical FAISS index. 
A random split guarantees that the test set shares the exact same distribution as the training set. A **Temporal Split** (e.g., train on Jan-May, test on June) would reveal that the FAISS retriever fails catastrophically on novel issues. The current metrics overestimate production performance.

## Conclusion
Do not trust the top-line Accuracy or Automation Rate. The true viability of this system rests on:
1. **Macro F1** (proving it can find rare, high-risk intents).
2. **Escalation Recall** (proving it safely refuses to guess when confused).
3. **Cohen's Kappa** (proving the LLM Judge aligns with human judgment).
