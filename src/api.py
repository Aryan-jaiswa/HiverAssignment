import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import traceback
from contextlib import asynccontextmanager

from src.pipeline import SupportAgent

logger = logging.getLogger(__name__)

# Singleton pattern for the agent to avoid reloading models on every request
agent_instance = None
agent_init_error = None

def get_agent():
    global agent_instance, agent_init_error
    if agent_instance is None:
        try:
            agent_instance = SupportAgent()
            agent_init_error = None
        except Exception as e:
            agent_init_error = str(e)
            logger.error(f"Failed to initialize SupportAgent: {e}")
            raise RuntimeError(f"Model initialization failed: {e}")
    return agent_instance

class SupportRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000, description="The incoming customer message from Twitter.")

class EvidenceItem(BaseModel):
    similarity: float
    conversation_id: str
    customer_message: str

class SupportResponse(BaseModel):
    intent: str
    intent_confidence: float
    decision: str
    reason: str
    reply: str
    evidence: list[EvidenceItem]

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-load models on startup
    try:
        get_agent()
    except Exception as e:
        logger.error(f"Failed to load agent on startup. System is degraded. Reason: {e}")
    yield

app = FastAPI(title="AppleSupport AI Agent API", version="1.0.0", lifespan=lifespan)

@app.get("/health")
async def health_check():
    if agent_init_error:
        raise HTTPException(status_code=503, detail=f"Unhealthy: {agent_init_error}")
    if not agent_instance:
        raise HTTPException(status_code=503, detail="Unhealthy: Agent not initialized")
    return {"status": "healthy"}

@app.post("/api/v1/support", response_model=SupportResponse)
async def handle_support_request(request: SupportRequest):
    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty or whitespace.")
        
    try:
        agent = get_agent()
    except Exception as e:
        raise HTTPException(status_code=503, detail="Service Unavailable: Model failed to load.")
        
    try:
        result = agent.handle_message(request.message)
        
        # Format evidence safely
        formatted_evidence = []
        for ev in result.get("retrieved_evidence", []):
            formatted_evidence.append(EvidenceItem(
                similarity=ev.get("similarity", 0.0),
                conversation_id=ev.get("conversation_id", ""),
                customer_message=ev.get("customer_message", "")
            ))
            
        return SupportResponse(
            intent=result.get("intent", "unknown"),
            intent_confidence=result.get("intent_confidence", 0.0),
            decision=result.get("decision", "ESCALATE"),
            reason=result.get("reason", ""),
            reply=result.get("reply", ""),
            evidence=formatted_evidence
        )
        
    except Exception as e:
        logger.error(f"Error handling request: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Internal Server Error during processing.")
