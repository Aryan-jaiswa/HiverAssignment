import streamlit as st
import os
import sys

# Ensure src can be imported regardless of where streamlit is run from
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import SupportAgent

st.set_page_config(page_title="AppleSupport AI Agent", layout="centered")

# Initialize Singleton Agent
@st.cache_resource
def load_agent():
    return SupportAgent()

st.title("🍎 AppleSupport AI Agent")
st.markdown("This dashboard demonstrates the end-to-end classification, FAISS retrieval, and Gemini-grounded generation pipeline.")

if not os.getenv("GEMINI_API_KEY"):
    st.warning("GEMINI_API_KEY is not set in `.env`. The LLM generation step will fail and immediately escalate.")

agent = load_agent()

# Chat interface
user_input = st.text_area("Customer Message (Tweet):", placeholder="e.g. My iPhone 15 battery is draining way too fast after the update.")

if st.button("Process Message"):
    if not user_input.strip():
        st.error("Please enter a message.")
    else:
        with st.spinner("Processing through the AI pipeline..."):
            try:
                result = agent.handle_message(user_input)
                
                # Layout
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("1. Intent Classification")
                    st.write(f"**Predicted Intent:** `{result['intent']}`")
                    st.write(f"**Confidence:** `{result['intent_confidence']:.2f}`")
                    
                with col2:
                    st.subheader("2. Final Decision")
                    if result['decision'] == "AUTO_HANDLE":
                        st.success("✅ AUTO_HANDLE")
                    else:
                        st.error("🚨 ESCALATE TO HUMAN")
                    st.write(f"**Reasoning:** {result['reason']}")
                
                st.subheader("3. Agent Reply")
                if result['reply']:
                    st.info(result['reply'])
                else:
                    st.warning("No reply generated due to escalation.")
                
                st.subheader("4. Historical FAISS Evidence")
                if not result['retrieved_evidence']:
                    st.write("No similar historical evidence found above threshold.")
                else:
                    for i, ev in enumerate(result['retrieved_evidence']):
                        with st.expander(f"Historical Case {i+1} (Similarity: {ev['similarity']:.2f})"):
                            st.write(f"**Customer:** {ev['customer_message']}")
                            st.write(f"**Agent:** {ev['support_reply']}")
                            
            except Exception as e:
                st.error(f"Pipeline Error: {e}")
