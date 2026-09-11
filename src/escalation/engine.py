import re

class DecisionEngine:
    """
    Deterministic escalation rules applied before and after LLM generation.
    """
    def __init__(self, 
                 intent_confidence_threshold=0.65, 
                 retrieval_similarity_threshold=0.70,
                 high_risk_intents=None):
                 
        self.intent_confidence_threshold = intent_confidence_threshold
        self.retrieval_similarity_threshold = retrieval_similarity_threshold
        self.high_risk_intents = high_risk_intents or ['account_access', 'payment_billing', 'complaint']
        
        # Regex to catch explicit requests for humans
        self.human_pattern = re.compile(r'\b(human|agent|operator|representative|person|manager|call me)\b', re.IGNORECASE)

    def pre_generation_check(self, message: str, intent: str, intent_confidence: float, top_similarity: float) -> dict:
        """
        Checks if we should skip the LLM and escalate immediately.
        Returns a dict: {"needs_escalation": bool, "reason": str}
        """
        # 1. Explicit human request
        if self.human_pattern.search(message):
            return {"needs_escalation": True, "reason": "Explicit human request detected."}
            
        # 2. Low intent confidence
        if intent_confidence < self.intent_confidence_threshold:
            return {"needs_escalation": True, "reason": f"Intent confidence ({intent_confidence:.2f}) below threshold ({self.intent_confidence_threshold})."}
            
        # 3. High risk intent
        if intent in self.high_risk_intents:
            return {"needs_escalation": True, "reason": f"Intent '{intent}' requires human handling."}
            
        # 4. Low retrieval similarity
        if top_similarity < self.retrieval_similarity_threshold:
            return {"needs_escalation": True, "reason": f"No highly similar historical evidence found (max sim: {top_similarity:.2f})."}
            
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
