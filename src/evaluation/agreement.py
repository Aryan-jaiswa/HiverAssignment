from sklearn.metrics import cohen_kappa_score, accuracy_score

def calculate_agreement(human_scores: list, llm_scores: list) -> dict:
    """
    Calculates agreement metrics between Human labels (0-4) and LLM Judge labels (0-4).
    """
    if not human_scores or not llm_scores or len(human_scores) != len(llm_scores):
        return {}
        
    exact_match = accuracy_score(human_scores, llm_scores)
    
    # Cohen's kappa is ideal for measuring inter-rater agreement for categorical items.
    # It accounts for the possibility of the agreement occurring by chance.
    try:
        kappa = cohen_kappa_score(human_scores, llm_scores, weights='quadratic')
    except Exception:
        kappa = 0.0
        
    # Percentage of times the scores are within 1 point of each other (e.g. 3 vs 4)
    within_one = sum(1 for h, l in zip(human_scores, llm_scores) if abs(h - l) <= 1)
    within_one_rate = within_one / len(human_scores)
    
    return {
        "exact_agreement": float(exact_match),
        "cohens_kappa_quadratic": float(kappa),
        "within_one_point_agreement": float(within_one_rate)
    }
