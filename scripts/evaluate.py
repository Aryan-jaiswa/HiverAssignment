import os
import sys
import json
import pandas as pd
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import SupportAgent
from src.generation.provider import GeminiProvider
from src.evaluation.metrics import (
    calculate_classification_metrics, 
    calculate_escalation_metrics, 
    calculate_safe_automation_rate
)
from src.evaluation.judge import LLMJudge
from src.evaluation.agreement import calculate_agreement

def main():
    print("=== Phase 6: Full System Evaluation ===")
    
    golden_path = "data/processed/golden_set.csv"
    human_eval_path = "data/processed/human_eval_sample.csv"
    
    if not os.path.exists(golden_path):
        print(f"Error: {golden_path} not found.")
        return
        
    df = pd.read_csv(golden_path).dropna(subset=['customer_message', 'intent'])
    
    if len(df) == 0:
        print("Error: Golden set is empty or missing 'intent' annotations.")
        return

    print("Initializing Pipeline...")
    agent = SupportAgent()
    provider = GeminiProvider()
    judge = LLMJudge(provider)
    
    # 1. Classification Evaluation
    print("Evaluating Classification...")
    y_true_intent = df['intent'].tolist()
    y_pred_intent = []
    
    for msg in df['customer_message']:
        try:
            pred = agent.classifier.predict_with_confidence([msg])[0]['intent']
        except Exception:
            pred = "unknown"
        y_pred_intent.append(pred)
        
    class_metrics = calculate_classification_metrics(y_true_intent, y_pred_intent)
    with open("artifacts/evaluation/classification_results.json", "w") as f:
        json.dump(class_metrics, f, indent=2)
        
    print(f"  Macro F1: {class_metrics.get('macro_f1', 0):.4f}")
    
    # 2. Pipeline Execution & Generation
    print("Running Full Pipeline on Golden Set... (This will take time)")
    eval_results = []
    failures = []
    
    # For escalation metrics
    y_true_escalate = df['should_escalate'].tolist() if 'should_escalate' in df.columns else [False] * len(df)
    y_pred_escalate = []
    
    # To avoid API rate limits during massive runs, we will only run LLM on a subset 
    # if it's too large, but for the assignment, we process the 200 items.
    
    for idx, row in df.iterrows():
        msg = row['customer_message']
        expected_intent = row['intent']
        
        # We need the full agent response
        res = agent.handle_message(msg)
        
        pred_escalate = res['decision'] == 'ESCALATE'
        y_pred_escalate.append(pred_escalate)
        
        # Score the generated response if it was auto-handled
        judge_score = 0
        if not pred_escalate and os.getenv("GEMINI_API_KEY"):
            judge_res = judge.evaluate(msg, res['retrieved_evidence'], res['reply'])
            judge_score = judge_res['score']
        
        intent_correct = res['intent'] == expected_intent
        
        record = {
            "customer_message": msg,
            "expected_intent": expected_intent,
            "predicted_intent": res['intent'],
            "decision": res['decision'],
            "reason": res['reason'],
            "reply": res['reply'],
            "intent_correct": intent_correct,
            "llm_judge_score": judge_score
        }
        eval_results.append(record)
        
        # Track failures (Wrong intent, dangerous LLM output, or false auto-handling)
        if not intent_correct or judge_score <= 1:
            failures.append({
                "issue": "Intent Mismatch" if not intent_correct else "Poor LLM Generation",
                "customer_message": msg,
                "expected_intent": expected_intent,
                "predicted_intent": res['intent'],
                "decision": res['decision'],
                "reply": res['reply'],
                "judge_score": judge_score
            })
            
    # 3. Escalation Metrics
    esc_metrics = calculate_escalation_metrics(y_true_escalate, y_pred_escalate)
    with open("artifacts/evaluation/escalation_results.json", "w") as f:
        json.dump(esc_metrics, f, indent=2)
        
    print(f"  Escalation F1: {esc_metrics.get('f1', 0):.4f}")
    
    # 4. Safe Automation Rate
    safe_rate = calculate_safe_automation_rate(eval_results)
    print(f"  Safe Automation Rate: {safe_rate:.4f}")
    
    # 5. Failure Analysis Extraction
    top_failures = failures[:5]
    with open("artifacts/evaluation/failure_analysis.json", "w") as f:
        json.dump(top_failures, f, indent=2)
        
    # 6. Human vs LLM Agreement
    if os.path.exists(human_eval_path):
        human_df = pd.read_csv(human_eval_path)
        if 'human_score_0_to_4' in human_df.columns and not human_df['human_score_0_to_4'].isna().all():
            print("Calculating Human vs LLM Agreement...")
            human_scores = []
            llm_scores = []
            
            for _, row in human_df.dropna(subset=['human_score_0_to_4']).iterrows():
                h_score = int(row['human_score_0_to_4'])
                # Re-run judge for perfect alignment (or look it up if we cached)
                j_res = judge.evaluate(row['customer_message'], [], row['agent_response'])
                human_scores.append(h_score)
                llm_scores.append(j_res['score'])
                
            agreement = calculate_agreement(human_scores, llm_scores)
            with open("artifacts/evaluation/judge_results.json", "w") as f:
                json.dump(agreement, f, indent=2)
            print(f"  Cohen's Kappa: {agreement.get('cohens_kappa_quadratic', 0):.4f}")
        else:
            print("Skipping Human Agreement: 'human_score_0_to_4' column is empty.")
    else:
        print(f"Skipping Human Agreement: {human_eval_path} not found.")
        
    print("\nAll evaluation artifacts saved to artifacts/evaluation/")

if __name__ == "__main__":
    main()
