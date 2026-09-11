from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
import numpy as np

def calculate_classification_metrics(y_true: list, y_pred: list) -> dict:
    """
    Calculates Accuracy, Macro F1, Weighted F1, Per-class F1, and Confusion Matrix.
    """
    if not y_true or not y_pred:
        return {}
        
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Per-class F1
    labels = sorted(list(set(y_true + y_pred)))
    f1s = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    per_class = {label: float(f1) for label, f1 in zip(labels, f1s)}
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class_f1": per_class,
        "labels": labels,
        "confusion_matrix": cm.tolist()
    }

def calculate_escalation_metrics(y_true_escalate: list, y_pred_escalate: list) -> dict:
    """
    Calculates precision, recall, f1, and false auto-handling rate for escalation decisions.
    Escalation is considered the positive class (True=Escalate).
    """
    if not y_true_escalate or not y_pred_escalate:
        return {}
        
    precision = precision_score(y_true_escalate, y_pred_escalate, zero_division=0)
    recall = recall_score(y_true_escalate, y_pred_escalate, zero_division=0)
    f1 = f1_score(y_true_escalate, y_pred_escalate, zero_division=0)
    
    total = len(y_true_escalate)
    automated = total - sum(y_pred_escalate)
    automation_rate = automated / total if total > 0 else 0
    
    # False auto-handling rate (Cases that SHOULD have been escalated but were auto-handled)
    # y_true = True (escalate), y_pred = False (auto-handle) -> False Negatives
    false_auto_handle = sum([1 for yt, yp in zip(y_true_escalate, y_pred_escalate) if yt is True and yp is False])
    false_auto_handle_rate = false_auto_handle / total if total > 0 else 0
    
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "automation_rate": float(automation_rate),
        "false_auto_handle_rate": float(false_auto_handle_rate)
    }

def calculate_safe_automation_rate(eval_results: list) -> float:
    """
    Safe Automation Rate = Correctly Auto-handled / Total Cases
    Correctly Auto-handled: decision == 'AUTO_HANDLE' AND intent_correct == True AND llm_judge_score >= 3.
    """
    if not eval_results:
        return 0.0
        
    safe_auto_handled = 0
    for res in eval_results:
        is_auto = res.get('decision') == 'AUTO_HANDLE'
        is_correct_intent = res.get('intent_correct', False)
        is_high_quality = res.get('llm_judge_score', 0) >= 3
        
        if is_auto and is_correct_intent and is_high_quality:
            safe_auto_handled += 1
            
    return safe_auto_handled / len(eval_results)
