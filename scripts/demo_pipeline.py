import os
import sys
import json
# pyrefly: ignore [missing-import]

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# pyrefly: ignore [missing-import]
from src.pipeline import SupportAgent

def main():
    print("=== AppleSupport Pipeline Demo ===")
    
    if not os.getenv("GEMINI_API_KEY"):
        print("\n[WARNING] GEMINI_API_KEY not found in environment.")
        print("The system will fail on the generation step. Please configure .env")
        print("Continuing with execution to demonstrate escalation pre-checks...\n")
        
    try:
        agent = SupportAgent()
    except Exception as e:
        print(f"Initialization error: {e}")
        return

    test_messages = [
        "I need to speak to a human manager right now, this is ridiculous.",
        "I forgot my apple ID password, how do I reset it?",
        "My macbook screen is completely shattered after I dropped it.",
        "You charged my credit card twice for my iCloud subscription, refund me immediately.",
        "My iphone 11 battery is draining way too fast after the latest ios 15 update. Can you help?"
    ]

    for msg in test_messages:
        print(f"\n--- MESSAGE ---")
        print(f"Customer: {msg}")
        
        result = agent.handle_message(msg)
        
        print("\n--- DECISION ---")
        print(f"Status: {result['decision']}")
        print(f"Intent: {result['intent']} ({result['intent_confidence']:.2f})")
        print(f"Reason: {result['reason']}")
        
        if result['decision'] == "AUTO_HANDLE":
            print(f"Reply:  {result['reply']}")
            
        print("-" * 50)

if __name__ == "__main__":
    main()
