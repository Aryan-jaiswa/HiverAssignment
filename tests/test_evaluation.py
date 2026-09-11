import pytest
from src.evaluation.metrics import calculate_classification_metrics, calculate_escalation_metrics, calculate_safe_automation_rate
from src.evaluation.agreement import calculate_agreement

def test_classification_metrics():
    y_true = ["billing", "billing", "tech"]
    y_pred = ["billing", "tech", "tech"]
    
    res = calculate_classification_metrics(y_true, y_pred)
    assert res["accuracy"] == pytest.approx(0.666, 0.01)
    
def test_escalation_metrics():
    y_true = [True, True, False, False]
    y_pred = [True, False, False, False]
    
    res = calculate_escalation_metrics(y_true, y_pred)
    # Precision: 1.0 (Only guessed true once, and it was correct)
    assert res["precision"] == 1.0
    # Recall: 0.5 (Guessed true once out of two actual trues)
    assert res["recall"] == 0.5
    # False auto-handle rate: 1 FN out of 4 total cases = 0.25
    assert res["false_auto_handle_rate"] == 0.25

def test_safe_automation_rate():
    results = [
        {"decision": "AUTO_HANDLE", "intent_correct": True, "llm_judge_score": 4},
        {"decision": "AUTO_HANDLE", "intent_correct": True, "llm_judge_score": 2}, # Low quality
        {"decision": "AUTO_HANDLE", "intent_correct": False, "llm_judge_score": 4}, # Wrong intent
        {"decision": "ESCALATE", "intent_correct": True, "llm_judge_score": 0},
    ]
    
    rate = calculate_safe_automation_rate(results)
    # Only the first one is safely auto-handled
    assert rate == 0.25

def test_agreement_metrics():
    h = [4, 3, 0, 4]
    l = [4, 4, 0, 2]
    
    res = calculate_agreement(h, l)
    assert res["exact_agreement"] == 0.5 # 2 out of 4
    assert res["within_one_point_agreement"] == 0.75 # 3 out of 4 (4 vs 4, 3 vs 4, 0 vs 0)
