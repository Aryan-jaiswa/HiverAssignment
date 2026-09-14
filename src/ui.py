import streamlit as st
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import SupportAgent

st.set_page_config(page_title="AppleSupport AI Agent", layout="centered", initial_sidebar_state="collapsed")

# Inject Custom Glassmorphic CSS
def inject_css():
    st.markdown("""
    <style>
    /* Global App Background */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Central Typography */
    .main-title {
        text-align: center;
        font-weight: 700;
        font-size: 3rem;
        color: #1d1d1f;
        margin-bottom: 0px;
        padding-top: 2rem;
    }
    .sub-title {
        text-align: center;
        font-weight: 400;
        font-size: 1.5rem;
        color: #1d1d1f;
        margin-bottom: 2rem;
    }
    
    /* Glassmorphic Container */
    .glass-card {
        background: rgba(255, 255, 255, 0.4);
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.5);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.1);
        margin-bottom: 1.5rem;
        color: #1d1d1f;
    }
    
    /* Quick Action Pills */
    .pill-container {
        display: flex;
        justify-content: center;
        flex-wrap: wrap;
        gap: 10px;
        margin-bottom: 2rem;
        margin-top: 1rem;
    }
    .pill {
        background: rgba(255, 255, 255, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.8);
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 500;
        color: #333;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        cursor: default;
    }
    
    /* Streamlit overrides */
    .stMarkdown, .stMarkdown p, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown strong {
        color: #1d1d1f !important;
    }
    .stExpander, .stExpander p, .stExpander span, .stExpander div {
        color: #1d1d1f !important;
    }
    .stAlert p {
        color: #1d1d1f !important;
    }
    .stTextArea label {
        display: none !important;
    }
    div[data-testid="stWidgetLabel"] {
        display: none !important;
    }
    .stTextArea > div {
        background-color: transparent !important;
    }
    .stTextArea textarea {
        background-color: rgba(255, 255, 255, 0.8) !important;
        color: #1d1d1f !important;
        -webkit-text-fill-color: #1d1d1f !important;
        border: 1px solid rgba(255, 255, 255, 0.8) !important;
        border-radius: 15px !important;
        font-size: 1.1rem !important;
        padding: 15px !important;
    }
    .stButton>button {
        border-radius: 20px !important;
        background: #1d1d1f !important;
        color: white !important;
        font-weight: 600 !important;
        border: none !important;
        padding: 10px 24px !important;
        transition: all 0.3s ease;
        margin-top: 10px;
    }
    .stButton>button:hover {
        background: #333336 !important;
        transform: translateY(-2px);
    }
    
    /* Logo Header */
    .header-container {
        display: flex;
        align-items: center;
        margin-bottom: -1rem;
    }
    .header-logo {
        font-size: 1.5rem;
        margin-right: 10px;
    }
    .header-text {
        font-weight: 600;
        font-size: 1.2rem;
        color: #1d1d1f;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

inject_css()

@st.cache_resource
def load_agent():
    return SupportAgent()

# Header
st.markdown("""
<div class="header-container">
    <div class="header-logo">🍎 💬</div>
    <div class="header-text">AppleSupport AI Agent</div>
</div>
""", unsafe_allow_html=True)

# Main Titles
st.markdown('<div class="main-title">AppleSupport AI Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Ready to assist?</div>', unsafe_allow_html=True)

agent = load_agent()

# Input section inside a glass card
with st.container():
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    user_input = st.text_area("Enter customer message...", height=100, label_visibility="collapsed", placeholder="e.g. My iPhone 15 battery is draining way too fast after the update.")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        submit_btn = st.button("Process Message", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Quick action pills
st.markdown("""
<div class="pill-container">
    <div class="pill">Troubleshooting</div>
    <div class="pill">Billing</div>
    <div class="pill">Account Access</div>
    <div class="pill">Device Setup</div>
    <div class="pill">Software Update</div>
</div>
""", unsafe_allow_html=True)

if not os.getenv("GEMINI_API_KEY"):
    st.warning("GEMINI_API_KEY is not set in `.env`. The LLM generation step will fail and immediately escalate.")

if submit_btn and user_input:
    with st.spinner("Analyzing message through AI pipeline..."):
        try:
            result = agent.handle_message(user_input)
            human = result.get('human_fields', {})
            
            # Start Results Glass Card
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            
            st.markdown("### 1. Issue Identified")
            st.markdown(f"**{human.get('issue_title', 'Unknown')}**")
            st.markdown(human.get('issue_context', 'No context available.'))
            st.markdown("<hr style='border: 0.5px solid rgba(0,0,0,0.1); margin: 15px 0;'>", unsafe_allow_html=True)
            
            st.markdown("### 2. Historical Support Evidence")
            st.markdown(human.get('evidence_context', 'No evidence available.'))
            st.markdown("<hr style='border: 0.5px solid rgba(0,0,0,0.1); margin: 15px 0;'>", unsafe_allow_html=True)

            st.markdown("### 3. Final Decision")
            if result['decision'] == "AUTO_HANDLE":
                st.success("✅ AUTO-HANDLE")
                st.markdown("**Why we can handle this automatically**")
                st.markdown(human.get('decision_reason', ''))
            else:
                st.error("🚨 ESCALATE TO HUMAN")
                st.markdown("**Why we're escalating**")
                st.markdown(human.get('decision_reason', ''))
            st.markdown("<hr style='border: 0.5px solid rgba(0,0,0,0.1); margin: 15px 0;'>", unsafe_allow_html=True)
            
            st.markdown("### 4. Agent Reply")
            if result['decision'] == "AUTO_HANDLE":
                st.info(result['reply'])
            else:
                st.warning("No automated resolution was sent.")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Technical expander
            with st.expander("View Technical Signals (Debugging)"):
                st.write(f"**Intent:** `{result['intent']}`")
                st.write(f"**Classification score:** `{result['intent_confidence']:.2f}`")
                top_sim = result['retrieved_evidence'][0]['similarity'] if result['retrieved_evidence'] else 0.0
                st.write(f"**Historical match:** `{top_sim:.2f}`")
                st.write(f"**Risk flags:** None explicitly tracked here")
                if result['retrieved_evidence']:
                    st.write("**Retrieved cases:**")
                    for i, ev in enumerate(result['retrieved_evidence']):
                        st.write(f"Case {i+1} (Sim: {ev['similarity']:.2f}): {ev['customer_message']} -> {ev['support_reply']}")
                        
        except Exception as e:
            st.error(f"Pipeline Error: {e}")
