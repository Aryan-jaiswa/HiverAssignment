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
            self.classifier.fit_zero_shot()
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
            return self._build_response(
                intent="classifier_failure",
                intent_confidence=0.0,
                evidence=[],
                decision="ESCALATE",
                reason="Classifier infrastructure failed.",
                message=message
            )
            
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
            return self._build_response(intent, intent_confidence, evidence, decision="ESCALATE", reason=pre_check["reason"], message=message)

        # 4. LLM Generation & Post-Check Validation
        logger.info("Pre-checks passed. Generating response...")
        llm_response = self.generator.generate_response(message, evidence)
        
        if llm_response["needs_escalation"]:
            logger.info(f"LLM/POST-CHECK ESCALATION: {llm_response['reason']}")
            return self._build_response(intent, intent_confidence, evidence, decision="ESCALATE", reason=llm_response["reason"], message=message)
            
        # 5. Final Auto-Handle
        logger.info("Successfully generated auto-handle response.")
        return self._build_response(
            intent=intent,
            intent_confidence=intent_confidence,
            evidence=evidence,
            reply=llm_response["reply"],
            decision="AUTO_HANDLE",
            reason=llm_response["reason"],
            message=message
        )

    def _generate_human_explanation(self, message: str, intent: str, top_sim: float, decision: str, raw_reason: str) -> dict:
        msg = message.lower()
        
        # 1. Contextual issue description
        subject = "a device or service issue"
        if "wifi" in msg or "wi-fi" in msg or "disconnect" in msg:
            subject = "a Wi-Fi connectivity problem"
        elif "battery" in msg or "drain" in msg:
            subject = "a battery drain issue"
        elif "airpods" in msg or "sound" in msg or "hear" in msg:
            subject = "an audio or AirPods issue"
        elif "password" in msg or "apple id" in msg:
            subject = "an Apple ID or password issue"
        elif "payment" in msg or "charge" in msg or "recognize" in msg:
            subject = "an unrecognized payment or billing issue"
        elif "locked" in msg or "unlock" in msg:
            subject = "a locked device or account issue"
            
        intent_map = {
            "device_troubleshooting": "is asking for troubleshooting help with",
            "software_issue": "is reporting",
            "repair_status": "is inquiring about",
            "account_access": "is requesting assistance with",
            "payment_billing": "is asking about",
            "information_request": "is requesting information regarding",
            "complaint": "is expressing frustration about"
        }
        action = intent_map.get(intent, "is asking about")
        
        issue_context = f"The customer {action} {subject}."
        
        # 2. Historical Evidence
        if top_sim >= 0.60:
            evidence_desc = "Several similar historical AppleSupport conversations were found, including a strong match for this type of issue."
        elif top_sim > 0.0:
            evidence_desc = "Some historical cases were found, but they were not strong matches for this specific request."
        else:
            evidence_desc = "No highly relevant historical support cases were found for this specific issue."
            
        # 3. Final Decision Reason
        if decision == "AUTO_HANDLE":
            reason = f"The request appears to be a routine support question and does not contain signals requiring human verification. {evidence_desc} Because the request is low-risk and supported by relevant historical evidence, the agent can provide an initial response automatically."
        else:
            if "human request" in raw_reason.lower():
                reason = "The customer has explicitly requested human assistance, so the automated agent will not continue with an automated resolution. The request should therefore be handled by a human agent."
            elif "security" in raw_reason.lower() or "financial" in raw_reason.lower() or "risk" in raw_reason.lower() or "human handling" in raw_reason.lower():
                reason = "The request involves a potentially sensitive account/payment issue that may require verification or information unavailable to the automated agent. To ensure accurate handling, the case is being escalated to a human support agent."
            elif "confidence" in raw_reason.lower() or "evidence" in raw_reason.lower():
                reason = "The system could not identify the issue with sufficient confidence and did not find strong enough historical evidence to support a reliable response. Rather than guessing, the case is being escalated for human review."
            else:
                reason = f"The case requires human review. ({raw_reason})"
                
        return {
            "issue_title": intent.replace('_', ' ').capitalize() if intent != "classifier_failure" else "Classification Error",
            "issue_context": issue_context,
            "evidence_context": evidence_desc,
            "decision_reason": reason
        }

    def _build_response(self, intent: str, intent_confidence: float, evidence: list, decision: str, reason: str, reply: str = "", message: str = "") -> dict:
        top_sim = evidence[0]["similarity"] if evidence else 0.0
        human_fields = self._generate_human_explanation(message, intent, top_sim, decision, reason)
        
        return {
            "intent": intent,
            "intent_confidence": intent_confidence,
            "retrieved_evidence": evidence,
            "reply": reply,
            "decision": decision,
            "reason": reason,
            "human_fields": human_fields
        }
