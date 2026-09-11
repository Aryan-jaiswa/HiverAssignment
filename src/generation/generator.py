from src.generation.provider import LLMProvider
from src.generation.prompts import build_prompt
from src.escalation.engine import DecisionEngine

class LLMGenerator:
    """
    Orchestrates building the prompt, calling the LLM, and validating the response.
    """
    def __init__(self, provider: LLMProvider, decision_engine: DecisionEngine):
        self.provider = provider
        self.decision_engine = decision_engine
        
    def generate_response(self, customer_message: str, retrieved_evidence: list) -> dict:
        """
        Returns the finalized structured output after LLM generation and post-validation.
        """
        prompt = build_prompt(customer_message, retrieved_evidence)
        
        # Call LLM
        llm_response = self.provider.generate(prompt)
        
        # Validate response format and safety
        validation_result = self.decision_engine.post_generation_check(llm_response)
        
        if validation_result["needs_escalation"]:
            # Override LLM decision if validation fails
            return {
                "reply": "",
                "confidence": 0.0,
                "needs_escalation": True,
                "reason": validation_result["reason"]
            }
            
        return {
            "reply": llm_response.get("reply", ""),
            "confidence": llm_response.get("confidence", 0.0),
            "needs_escalation": llm_response.get("needs_escalation", False),
            "reason": llm_response.get("reason", "Auto-handled successfully.")
        }
