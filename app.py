# === app.py ===
import streamlit as st
import os
import json
import pandas as pd
from datetime import datetime
from oss_power_analyser import analyze_multiple_repos_with_logs, save_all
from history_tracker import save_history_entry, get_past_runs
from rag_utils import query_vector_index
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq

st.set_page_config(page_title="RECOPS", layout="wide")

st.markdown("""
    <style>
        body { background-color: #f9f9f9; color: #333; }
        .main-title { font-size: 48px; font-weight: bold; color: #1e3a8a; text-align: center; margin-top: 40px; }
        .section-title { font-size: 24px; font-weight: 600; margin-top: 30px; color: #0f172a; }
        .footer { position: fixed; bottom: 10px; left: 0; width: 100%; text-align: center; color: #666; font-size: 14px; }
        .floating-spinner {
            position: fixed;
            top: 80px;
            right: 40px;
            background: #facc15;
            color: #000;
            font-weight: bold;
            padding: 8px 14px;
            border-radius: 50px;
            z-index: 9999;
            box-shadow: 2px 2px 6px rgba(0,0,0,0.2);
            animation: pulse 1.5s infinite;
        }
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.1); }
            100% { transform: scale(1); }
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">RECOPS: Resilience and Cost-benefits of Open Source Software in the Power Sector</div>', unsafe_allow_html=True)

page = st.sidebar.radio("Navigate", ["Home", "Feature Analysis", "Ask LLM", "Past Analyses"])

if "analysis_thread" not in st.session_state:
    st.session_state.analysis_thread = None
    st.session_state.analysis_done = False
    st.session_state.analysis_result = []

if page == "Home":
    st.markdown("""
    ### Welcome to RECOPS
    This tool helps analyze open-source software in the power systems domain for features like:
    - Technical metadata
    - Security vulnerabilities
    - Lock-in risk
    - Domain applicability
    - Compliance with power sector standards
    """)

elif page == "Feature Analysis":
    st.markdown("<div class='section-title'>🔍 OSS Feature Analyzer</div>", unsafe_allow_html=True)
    urls_input = st.text_area("Enter up to 200 GitHub Repository URLs (one per line)")
    urls = [u.strip() for u in urls_input.strip().split("\n") if u.strip()][:200]
    log_container = st.empty()

    if urls and st.button("Start Analysis"):
        log_lines = []
        def log_fn(msg):
            log_lines.append(msg)
            log_container.markdown("  \n".join(log_lines[-12:]))

        st.markdown('<div class="floating-spinner">⚙️ Analysing...</div>', unsafe_allow_html=True)
        st.info("⏳ Analysis started. Please do not refresh or switch away until complete.")

        result = analyze_multiple_repos_with_logs(urls, log_fn)
        st.session_state.analysis_result = result

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        full_filename = f"features_output_{timestamp}"
        essential_filename = f"essential_features_{timestamp}"

        save_all(result, full_filename, essential_filename)
        save_history_entry(full_filename)

        st.session_state.full_csv = f"{full_filename}.csv"
        st.session_state.full_json = f"{full_filename}.json"
        st.session_state.ess_csv = f"{essential_filename}.csv"
        st.session_state.ess_json = f"{essential_filename}.json"

        st.session_state.analysis_done = True

    if st.session_state.analysis_thread and st.session_state.analysis_thread.is_alive():
        st.warning("🔄 Analysis in progress...")
    elif st.session_state.analysis_done:
        st.success("✅ Analysis complete!")
        st.subheader("🔧 Extracted Features")
        st.json(st.session_state.analysis_result)

        st.download_button("📥 Download ESSENTIAL FEATURES (CSV)", open(st.session_state.ess_csv, "rb"), file_name=st.session_state.ess_csv)
        st.download_button("📥 Download ESSENTIAL FEATURES (JSON)", open(st.session_state.ess_json, "rb"), file_name=st.session_state.ess_json)
        st.download_button("📥 Download FULL FEATURES (CSV)", open(st.session_state.full_csv, "rb"), file_name=st.session_state.full_csv)
        st.download_button("📥 Download FULL FEATURES (JSON)", open(st.session_state.full_json, "rb"), file_name=st.session_state.full_json)

elif page == "Ask LLM":
    st.markdown("<div class='section-title'>🤖 Ask LLM About a Project</div>", unsafe_allow_html=True)
    st.markdown("#### 1. Ask custom questions on OSS behavior")
    question = st.text_input("Enter your question (based on analyzed project code):")
    if question:
        with st.spinner("Thinking..."):
            answer = query_vector_index(question)
        st.success("LLM Answer:")
        st.markdown(f"`{answer}`")

    st.markdown("---")
    st.markdown("#### 2. Bonus Insights (LLM-based project summaries)")
    sample_questions = [
        "What unique power sector capability does this OSS offer?",
        "Does it help reduce vendor lock-in?",
        "Is this software suitable for real-time grid operations?",
        "How scalable is this solution for a national utility?",
        "Can this be integrated with existing EMS or SCADA?"
    ]
    selected_question = st.selectbox("Pick a common question to evaluate:", sample_questions)
    if st.button("Run this insight"):
        if "analysis_result" in st.session_state and st.session_state.analysis_result:
            try:
                context_block = json.dumps(st.session_state.analysis_result[0])[:5000]
                llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="llama3-8b-8192")
                prompt = PromptTemplate(
                    input_variables=["context", "q"],
                    template="""
Based on the OSS project metadata below, answer the following question:
Context:
{context}

Question:
{q}

Answer:
"""
                )
                chain = LLMChain(llm=llm, prompt=prompt)
                result = chain.run({"context": context_block, "q": selected_question})
                st.success("Insight:")
                st.write(result.strip())
            except Exception as e:
                st.error(f"LLM processing failed: {str(e)}")
        else:
            st.warning("❗ Please run feature analysis first from the sidebar.")

elif page == "Past Analyses":
    st.markdown("<div class='section-title'>📁 Past Runs</div>", unsafe_allow_html=True)
    past_runs = get_past_runs()
    selected = st.selectbox("Choose one:", past_runs)
    if selected:
        st.download_button("📥 Download CSV", open(f"{selected}.csv", "rb"), file_name=f"{selected}.csv")
        st.download_button("📥 Download JSON", open(f"{selected}.json", "rb"), file_name=f"{selected}.json")

st.markdown("<div class='footer'>Developed by Shashank</div>", unsafe_allow_html=True)
