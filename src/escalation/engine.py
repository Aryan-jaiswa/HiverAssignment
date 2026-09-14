import re

class DecisionEngine:
    """
    Deterministic escalation rules applied before and after LLM generation.
    """
    def __init__(self, 
                 intent_confidence_threshold=0.50, 
                 retrieval_similarity_threshold=0.60,
                 high_risk_intents=None):
                 
        self.intent_confidence_threshold = intent_confidence_threshold
        self.retrieval_similarity_threshold = retrieval_similarity_threshold
        self.high_risk_intents = high_risk_intents or ['account_access', 'payment_billing', 'complaint']
        
        self.human_pattern = re.compile(r'\b(human|agent|operator|representative|person|manager|call me|speak to)\b', re.IGNORECASE)
        self.risk_pattern = re.compile(r'\b(stolen|hacked|unauthorized|fraud|charge|charged|payment|bank|credit card|lawsuit|attorney|sue)\b', re.IGNORECASE)

    def pre_generation_check(self, message: str, intent: str, intent_confidence: float, top_similarity: float) -> dict:
        if intent == "classifier_failure":
            return {"needs_escalation": True, "reason": "Classifier infrastructure failed."}

        if self.human_pattern.search(message):
            return {"needs_escalation": True, "reason": "Explicit human request detected."}
            
        if self.risk_pattern.search(message):
            return {"needs_escalation": True, "reason": "Message contains high-risk security/financial keywords."}
            
        is_confident_high_risk = intent in self.high_risk_intents and intent_confidence >= self.intent_confidence_threshold
        if is_confident_high_risk:
            return {"needs_escalation": True, "reason": f"Intent '{intent}' requires human handling."}
            
        is_intent_uncertain = intent_confidence < self.intent_confidence_threshold
        is_retrieval_weak = top_similarity < self.retrieval_similarity_threshold
        
        if is_intent_uncertain and is_retrieval_weak:
            return {"needs_escalation": True, "reason": f"Low intent confidence ({intent_confidence:.2f}) and weak retrieval evidence ({top_similarity:.2f})."}

        return {"needs_escalation": False, "reason": ""}

    def post_generation_check(self, llm_response: dict) -> dict:
        """
        Validates the LLM output safely.
        """
        if not llm_response:
            return {"needs_escalation": True, "reason": "LLM returned empty or malformed output."}
            
        if llm_response.get("needs_escalation") is True:
            return {"needs_escalation": True, "reason": "LLM self-escalated: " + llm_response.get("reason", "Unknown reason")}
            
        reply = llm_response.get("reply", "").strip()
        if not reply:
            return {"needs_escalation": True, "reason": "LLM returned an empty reply string."}
            
        return {"needs_escalation": False, "reason": ""}
