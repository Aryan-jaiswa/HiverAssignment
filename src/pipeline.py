import logging
from typing import Dict, Any

from src.intent.classifier import EmbeddingClassifier
from src.retrieval.api import RetrievalSystem
from src.escalation.engine import DecisionEngine
from src.generation.provider import LLMProvider, GeminiProvider
from src.generation.generator import LLMGenerator

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SupportAgent:
    """
    The unified Phase 5 Support Agent Pipeline.
    """
    def __init__(self, provider: LLMProvider = None):
        logger.info("Initializing Support Agent pipeline components...")
        
        # 1. Intent Classification
        self.classifier = EmbeddingClassifier()
        # Initialize zero-shot in case training wasn't run
        try:
            self.classifier.fit_zero_shot("config/intent_schema.yaml")
        except Exception as e:
            logger.warning(f"Could not load zero-shot intents: {e}")
            
        # 2. Retrieval
        try:
            self.retriever = RetrievalSystem()
        except FileNotFoundError:
            logger.warning("FAISS index not found. Retrieval will return empty until index is built.")
            self.retriever = None
            
        # 3. Escalation Engine
        self.decision_engine = DecisionEngine(
            intent_confidence_threshold=0.65,
            retrieval_similarity_threshold=0.70
        )
        
        # 4. Generation
        provider = provider or GeminiProvider()
        self.generator = LLMGenerator(provider, self.decision_engine)
        
        logger.info("Pipeline initialized successfully.")

    def handle_message(self, message: str) -> Dict[str, Any]:
        """
        Processes a customer message end-to-end.
        """
        logger.info(f"Processing new message: '{message[:50]}...'")
        
        # 1. Intent Classification
        try:
            intent_pred = self.classifier.predict_with_confidence([message])[0]
            intent = intent_pred["intent"]
            intent_confidence = intent_pred["confidence"]
        except Exception as e:
            logger.error(f"Intent classification failed: {e}")
            intent = "unknown"
            intent_confidence = 0.0
            
        logger.info(f"Intent: {intent} (Confidence: {intent_confidence:.2f})")

        # 2. Historical Retrieval
        evidence = []
        top_sim = 0.0
        if self.retriever:
            evidence = self.retriever.retrieve_similar_cases(message, top_k=3, threshold=0.0)
            if evidence:
                top_sim = evidence[0]["similarity"]
                
        logger.info(f"Retrieved {len(evidence)} historical cases. Max similarity: {top_sim:.2f}")

        # 3. Escalation Pre-Check
        pre_check = self.decision_engine.pre_generation_check(message, intent, intent_confidence, top_sim)
        if pre_check["needs_escalation"]:
            logger.info(f"PRE-CHECK ESCALATION: {pre_check['reason']}")
            return self._build_response(intent, intent_confidence, evidence, decision="ESCALATE", reason=pre_check["reason"])

        # 4. LLM Generation & Post-Check Validation
        logger.info("Pre-checks passed. Generating response...")
        llm_response = self.generator.generate_response(message, evidence)
        
        if llm_response["needs_escalation"]:
            logger.info(f"LLM/POST-CHECK ESCALATION: {llm_response['reason']}")
            return self._build_response(intent, intent_confidence, evidence, decision="ESCALATE", reason=llm_response["reason"])
            
        # 5. Final Auto-Handle
        logger.info("Successfully generated auto-handle response.")
        return self._build_response(
            intent=intent,
            intent_confidence=intent_confidence,
            evidence=evidence,
            reply=llm_response["reply"],
            decision="AUTO_HANDLE",
            reason=llm_response["reason"]
        )

    def _build_response(self, intent: str, intent_confidence: float, evidence: list, decision: str, reason: str, reply: str = "") -> dict:
        return {
            "intent": intent,
            "intent_confidence": intent_confidence,
            "retrieved_evidence": evidence,
            "reply": reply,
            "decision": decision,
            "reason": reason
        }
