import json
from src.generation.provider import LLMProvider

LLM_JUDGE_PROMPT = """
You are an impartial evaluator judging a customer support response.
You will be provided with:
1. The Customer Message
2. The Historical Evidence available to the agent
3. The Agent's Response

Evaluate the agent's response on Relevance, Grounding, Correctness, Helpfulness, and Tone.
Provide a final score from 0 to 4 based on this strict rubric:

[SCORE 4 - Excellent]: The response perfectly addresses the customer's query, is completely grounded in the evidence (no hallucination), adopts a polite tone, and fully resolves the issue as best as possible given the evidence.
[SCORE 3 - Good]: The response is helpful and grounded, but might be slightly brief or slightly misaligned in tone. No false promises are made.
[SCORE 2 - Borderline]: The response addresses the query but lacks clarity, relies too heavily on generic statements, or misses a key detail from the evidence.
[SCORE 1 - Poor]: The response is barely relevant, unhelpful, or hallucinates minor details.
[SCORE 0 - Dangerous]: The response invents policies, promises refunds/compensation not supported by evidence, hallucinates URLs/phone numbers, or gives dangerous technical advice.

Customer Message:
{customer_message}

Historical Evidence:
{evidence}

Agent Response:
{agent_response}

Output ONLY a JSON object exactly matching this schema:
{{
  "score": 3,
  "reasoning": "Brief explanation..."
}}
"""

class LLMJudge:
    def __init__(self, provider: LLMProvider):
        self.provider = provider
        
    def evaluate(self, customer_message: str, evidence: list, agent_response: str) -> dict:
        """
        Returns a dict with 'score' (0-4) and 'reasoning'.
        """
        if not agent_response:
            return {"score": 0, "reasoning": "No response generated."}
            
        evidence_text = ""
        for i, ev in enumerate(evidence):
            evidence_text += f"\nExample {i+1}:\n"
            evidence_text += f"Customer asked: {ev.get('customer_message', '')}\n"
            evidence_text += f"AppleSupport replied: {ev.get('support_reply', '')}\n"
            
        prompt = LLM_JUDGE_PROMPT.format(
            customer_message=customer_message,
            evidence=evidence_text,
            agent_response=agent_response
        )
        
        response = self.provider.generate(prompt)
        
        try:
            score = int(response.get("score", 0))
            # clamp to 0-4
            score = max(0, min(4, score))
        except (ValueError, TypeError):
            score = 0
            
        return {
            "score": score,
            "reasoning": response.get("reasoning", "Failed to parse reasoning.")
        }
