import os
import re
import json
import shutil
import subprocess
import stat
import pandas as pd
from datetime import datetime
# from dotenv import load_dotenv
from github import Github
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFaceHub

# Load environment variables
# load_dotenv()

# --- Content from history_tracker.py ---
HISTORY_DIR = "data"
HISTORY_FILE = os.path.join(HISTORY_DIR, "analysis_history.txt")

def ensure_history_dir():
    """Ensure the history directory exists (create if needed)."""
    if not os.path.exists(HISTORY_DIR):
        os.makedirs(HISTORY_DIR)

def save_history_entry(filename):
    ensure_history_dir()
    try:
        with open(HISTORY_FILE, "a", encoding="utf-8") as f:
            f.write(filename + "\n")
    except Exception as e:
        print(f"[History Error] Could not write entry: {e}")

def get_past_runs(max_entries=10):
    ensure_history_dir()
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            entries = [line.strip() for line in f.readlines()][-max_entries:]
        return entries[::-1]  # Show latest first
    except Exception as e:
        print(f"[History Error] Could not read history: {e}")
        return []

# --- Content from rag_utils.py ---
# Embedding Model and Vector DB Path
EMBED_MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L6-v2"
VECTOR_DB_DIR = "vector_db"

# === Prompt & LLM Chain Setup ===
prompt = PromptTemplate(
    input_variables=["context", "q"],
    template="Answer the question based on the context:\n\n{context}\n\nQuestion: {q}"
)
llm = HuggingFaceHub(repo_id="google/flan-t5-base", huggingfacehub_api_token='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', task="text2text-generation")
chain = LLMChain(llm=llm, prompt=prompt)

# === Load Embeddings ===
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_NAME,
        # model_kwargs={"local_files_only": True}
    )

embedding = load_embeddings()

# === Initialize Vector Store ===
if not os.path.exists(VECTOR_DB_DIR):
    os.makedirs(VECTOR_DB_DIR)

if os.path.exists(f"{VECTOR_DB_DIR}/index.faiss"):
    vectorstore = FAISS.load_local(VECTOR_DB_DIR, embeddings=embedding, index_name="oss", allow_dangerous_deserialization=True)
else:
    vectorstore = None

# === Store Project Metadata ===
def store_to_vector_index(project_name, metadata_dict):
    global vectorstore

    full_text = "\n".join([f"{k}: {v}" for k, v in metadata_dict.items() if isinstance(v, (str, int, float))])
    doc = Document(page_content=full_text, metadata={"project": project_name})

    if vectorstore is None:
        vectorstore = FAISS.from_documents([doc], embedding)
    else:
        vectorstore.add_documents([doc])

    vectorstore.save_local(VECTOR_DB_DIR, index_name="oss")

# === Query Vector Store ===
def query_vector_index(question: str) -> str:
    try:
        if vectorstore is None:
            return "❌ Vector index is empty. Please analyze a repository first."
        results = vectorstore.similarity_search(question, k=1)
        return results[0].page_content if results else "⚠ No relevant match found."
    except Exception as e:
        return f"❌ Error during vector search: {str(e)}"

# === Run Prompt Chain ===
def run_chain(context_block, selected_question):
    return chain.run({"context": context_block, "q": selected_question})

# --- Content from oss_power_analyser.py ---
# load_dotenv()
# GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GITHUB_TOKEN = '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'
# GROQ_API_KEY = 'xai-AQluiJN8n4xW6zw0racIjrm4r6gyF5cBCPgT8vkvI1BJUQsZ2shtpLxkuRMkmAnkLWtxswYC9sJckbDs'
GROQ_API_KEY = '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'


def handle_remove_readonly(func, path, exc_info):
    os.chmod(path, stat.S_IWRITE)
    func(path)

def clone_repo(repo_url, clone_dir="cloned_repo"):
    if os.path.exists(clone_dir):
        shutil.rmtree(clone_dir, onerror=handle_remove_readonly)
    subprocess.run(["git", "clone", repo_url, clone_dir], check=True)
    return clone_dir

def extract_static_metadata(repo_path):
    metadata = {}
    readme_text = ""
    for name in ["README.md", "README.rst"]:
        path = os.path.join(repo_path, name)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                readme_text = f.read()
            break
    metadata.update({
        "Detailed description": readme_text[:1000] if readme_text else "NF",
        "Code Used": "Yes" if any(file.endswith(('.py','.cpp','.c','.java')) for root, _, files in os.walk(repo_path) for file in files) else "NF",
        "Licence": "NF",
        "has_contributing": os.path.exists(os.path.join(repo_path, "CONTRIBUTING.md")),
        "has_code_of_conduct": os.path.exists(os.path.join(repo_path, "CODE_OF_CONDUCT.md")),
        "has_tests": any(re.match(r"test", name, re.I) for root, dirs, files in os.walk(repo_path) for name in dirs + files),
        "module_count": len([d for d in os.listdir(repo_path) if os.path.isdir(os.path.join(repo_path, d)) and d != ".git"]),
        "platforms": [p for p in ["Linux", "Windows", "CUDA", "GPU"] if p in readme_text],
        "README": readme_text
    })
    return metadata

def get_github_metadata(repo_url):
    # g = Github(GITHUB_TOKEN)
    from github import Auth
    g = Github(auth=Auth.Token(GITHUB_TOKEN))
    user_repo = "/".join(repo_url.split("/")[-2:]).replace(".git", "")
    repo = g.get_repo(user_repo)
    commits = list(repo.get_commits())
    return {
        "github repo": user_repo,
        "First update date": commits[-1].commit.author.date.isoformat() if commits else "NF",
        " last update": commits[0].commit.author.date.isoformat() if commits else "NF",
        "Releases": repo.get_releases().totalCount,
        "Active Contributors": repo.get_contributors().totalCount,
        "Open/closed issues": repo.get_issues(state="all").totalCount,
        "Fork Count": repo.forks_count,
        "GitHub Stars/Forks": f"{repo.stargazers_count} stars / {repo.forks_count} forks",
        "License Type": repo.get_license().license.spdx_id if repo.license else "NF"
    }

def collect_full_repo_code_text(repo_path):
    extensions = [".py", ".java", ".cpp", ".c", ".js", ".ts", ".html", ".css", ".rs", ".go", ".rb", ".json", ".xml"]
    code_chunks = []
    for root, _, files in os.walk(repo_path):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                try:
                    with open(os.path.join(root, file), "r", encoding="utf-8", errors="ignore") as f:
                        code_chunks.append(f.read())
                except:
                    continue
    return "\n\n".join(code_chunks)

def safe_json_parse(text):
    try:
        match = re.search(r"\{[\s\S]+\}", text)
        if match:
            return json.loads(match.group())
    except Exception as e:
        print("⚠ JSON Parse Error:", e)
    return {}

def llm_extract_features(readme, license, module_count, dependencies, code):
    llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.1-8b-instant")

    full_context = f"""
README:
{readme[:3000]}

LICENSE: {license}
Modules: {module_count}
Dependencies: {dependencies}
CODE:
{code[:10000]}
"""

    prompt = ChatPromptTemplate.from_template(
        """
From the given README, LICENSE, and CODE extract the following features if available (use "NF" if not found):
- Scale
- Time criticality
- Software function
- Simulation Accuracy
- Real-Time Processing
- Fault Tolerance Mechanism
- Standards Interoperability
- Platform Support
- SCADA/EMS Presence
- ROI for OSS
- DER Type
- Voltage Level
- Node/Buses Count
- Control Architecture
- Redundancy
- Cyber-Physical Integration
- Resilience Strategy
- Energy Not Supplied (ENS)
- Licensing Savings
- Maintenance Cost
- TCO (Total Cost of Ownership)
- Customer Diversity
- Reusability Value
- Simulation Tools
- Real-Time Control Systems
- Planning / Optimization Models
- Energy Management Systems
- Data Analytics & Forecasting Tools

Respond in JSON format only.
        """
    )

    chain = prompt | llm
    try:
        response = chain.invoke({
            "readme": readme,
            "license": license,
            "modules": module_count,
            "deps": dependencies,
            "code": code
        })

        parsed = safe_json_parse(response.content)

        EXPECTED_KEYS = [
            "Scale", "Time criticality", "Software function", "Simulation Accuracy",
            "Real-Time Processing", "Fault Tolerance Mechanism", "Standards Interoperability",
            "Platform Support", "SCADA/EMS Presence", "ROI for OSS", "DER Type", "Voltage Level",
            "Node/Buses Count", "Control Architecture", "Redundancy", "Cyber-Physical Integration",
            "Resilience Strategy", "Energy Not Supplied (ENS)", "Licensing Savings", "Maintenance Cost",
            "TCO (Total Cost of Ownership)", "Customer Diversity", "Reusability Value", "Simulation Tools",
            "Real-Time Control Systems", "Planning / Optimization Models", "Energy Management Systems",
            "Data Analytics & Forecasting Tools"
        ]

        for key in EXPECTED_KEYS:
            parsed.setdefault(key, "NF")

        return parsed

    except Exception as e:
        print(f"🛑 Error in llm_extract_features: {e}")
        return {k: "NF" for k in EXPECTED_KEYS}

def analyze_multiple_repos_with_logs(repo_urls, log_fn):
    all_features = []
    for url in repo_urls:
        try:
            log_fn(f"📥 Cloning repo: {url}")
            path = clone_repo(url)

            log_fn("📄 Extracting metadata...")
            static_meta = extract_static_metadata(path)
            gh_meta = get_github_metadata(url)
            code = collect_full_repo_code_text(path)

            # log_fn("⚙ Running Syft + Grype...")
            # syft = run_syft(path)
            # grype = run_grype(path)
            # dependencies = len(syft.get("artifacts", []))
            # vulnerabilities = len(grype.get("matches", []))

            log_fn("⚠ Skipping Syft + Grype (Streamlit Cloud)")
            dependencies = "NF"
            vulnerabilities = "NF"

            log_fn("🧠 Running LLM inference...")
            llm_features = llm_extract_features(
                static_meta.get("README", ""),
                gh_meta.get("License Type", "NF"),
                static_meta.get("module_count", 0),
                dependencies,
                code
            )

            combined = {
                **gh_meta,
                "Title": "NF",
                "Link": url,
                "Creator": "NF",
                "Creator specific": "NF",
                **static_meta,
                "dependency_count": dependencies,
                "vulnerabilities": vulnerabilities,
                **llm_features
            }
            all_features.append(combined)
            store_to_vector_index(combined["github repo"], combined)
            log_fn(f"✅ Done: {url}")
        except Exception as e:
            log_fn(f"❌ Failed for {url}: {str(e)}")
    return all_features

def save_all(features_list, full_filename="features_output", essential_filename="essential_features"):
    ESSENTIAL_FEATURES = [
        "Title", "Link", "github repo", "Creator", "Creator specific", "License Type", "Fork Count", "GitHub Stars/Forks",
        "First update date", " last update", "Releases", "Active Contributors",
        "Open/closed issues", "Detailed description", "Programming Language used", 
        "has_contributing", "has_code_of_conduct", "has_tests", "module_count", "platforms", "README",
        "dependency_count", "vulnerabilities", "Third-party Integrations", "Downloads / Installs Count", "User Community Size",
        "DER Type", "Version Release Frequency", "Vendor Diversity", "SCADA/EMS Presence", "Data Analytics & Forecasting Tools",
        "ROI for OSS", "Integration Cost", "Vendor Lock-in Avoidance", "Customer Diversity"
    ]

    FULL_FEATURE_LIST = list(set(ESSENTIAL_FEATURES + [
        "Community Support", "Documentation Completeness", "Maintenance History",
        "Dependency Freshness", "Code Modularity", "CI/CD Availability", "Issue Resolution Time",
        "Code Review Coverage", "Test Coverage", "Commit Frequency Trend", "Bus Factor",
        "Simulation Accuracy", "API Integration", "Real-Time Processing", "Platform Support",
        "Hardware Interfacing", "Scalability", "Fault Tolerance Mechanism", "Validation Availability",
        "Standards Interoperability", "Model Abstraction", "Extensibility / Plugin Support", "Resource Efficiency",
        "Security Features", "Deployment Modes", "Containerization Support", "Real-World Use Cases",
        "Institutional Backing", "Citations", "Educational Usage", "Benchmarks Participation",
        "Language Localization", "Codebase Size", "Commercial Support Availability", "Social Media Mentions",
        "Voltage Level", "Network Topology", "Control Architecture", "Redundancy",
        "Node/Buses Count", "Cyber-Physical Integration", "Resilience Strategy",
        "Fault Recovery Time", "MTBF (Mean Time Between Failures)",
        "SAIDI/SAIFI", "Load Stability", "Voltage/Frequency Stability", "DER Hosting Capacity",
        "Power Quality", "Islanding Accuracy", "Energy Not Supplied (ENS)", "Control Responsiveness",
        "Licensing Savings", "Maintenance Cost", "TCO (Total Cost of Ownership)", "Downtime Cost",
        "Reusability Value", "Training Cost", "Collaborative Cost Sharing",
        "Simulation Tools", "Real-Time Control Systems", "Planning / Optimization Models",
        "Energy Management Systems"
    ]))

    # Save full features
    full_data = []
    for project in features_list:
        entry = {key: project.get(key, "NF") for key in FULL_FEATURE_LIST}
        full_data.append(entry)
    pd.DataFrame(full_data).to_csv(f"{full_filename}.csv", index=False)
    with open(f"{full_filename}.json", "w") as f:
        json.dump(full_data, f, indent=2)

    # Save essential features
    essential_data = []
    for project in features_list:
        entry = {key: project.get(key, "NF") for key in ESSENTIAL_FEATURES}
        essential_data.append(entry)
    pd.DataFrame(essential_data).to_csv(f"{essential_filename}.csv", index=False)
    with open(f"{essential_filename}.json", "w") as f:
        json.dump(essential_data, f, indent=2)

# --- Content from app.py (Main Streamlit App) ---
st.write("✅ App started successfully.")

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

        st.markdown('<div class="floating-spinner">⚙ Analysing...</div>', unsafe_allow_html=True)
        st.info("⏳ Analysis started. Please do not refresh or switch away until complete.")

        result = analyze_multiple_repos_with_logs(urls, log_fn)
        st.session_state.analysis_result = result

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        full_filename = f"features_output_{timestamp}"
        essential_filename = f"essential_features_{timestamp}"

        save_all(result, full_filename, essential_filename)
        # save_history_entry(full_filename)

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
        st.markdown(f"{answer}")

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
                llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="llama-3.1-8b-instant")
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









# import os
# import re
# import json
# import shutil
# import subprocess
# import stat
# import pandas as pd
# from datetime import datetime
# # from dotenv import load_dotenv
# from github import Github
# from langchain_groq import ChatGroq
# from langchain.prompts import ChatPromptTemplate
# import streamlit as st
# from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain.schema import Document
# from langchain.chains import LLMChain
# from langchain.prompts import PromptTemplate
# from langchain.llms import HuggingFaceHub

# # Load environment variables
# # load_dotenv()

# # --- Content from history_tracker.py ---
# HISTORY_DIR = "data"
# HISTORY_FILE = os.path.join(HISTORY_DIR, "analysis_history.txt")

# def ensure_history_dir():
#     """Ensure the history directory exists (create if needed)."""
#     if not os.path.exists(HISTORY_DIR):
#         os.makedirs(HISTORY_DIR)

# def save_history_entry(filename):
#     ensure_history_dir()
#     try:
#         with open(HISTORY_FILE, "a", encoding="utf-8") as f:
#             f.write(filename + "\n")
#     except Exception as e:
#         print(f"[History Error] Could not write entry: {e}")

# def get_past_runs(max_entries=10):
#     ensure_history_dir()
#     if not os.path.exists(HISTORY_FILE):
#         return []
#     try:
#         with open(HISTORY_FILE, "r", encoding="utf-8") as f:
#             entries = [line.strip() for line in f.readlines()][-max_entries:]
#         return entries[::-1]  # Show latest first
#     except Exception as e:
#         print(f"[History Error] Could not read history: {e}")
#         return []

# # --- Content from rag_utils.py ---
# # Embedding Model and Vector DB Path
# EMBED_MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L6-v2"
# VECTOR_DB_DIR = "vector_db"

# # === Prompt & LLM Chain Setup ===
# prompt = PromptTemplate(
#     input_variables=["context", "q"],
#     template="Answer the question based on the context:\n\n{context}\n\nQuestion: {q}"
# )
# llm = HuggingFaceHub(repo_id="google/flan-t5-base", huggingfacehub_api_token='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', task="text2text-generation")
# chain = LLMChain(llm=llm, prompt=prompt)

# # === Load Embeddings ===
# # @st.cache_resource
# # def load_embeddings():
# #     return HuggingFaceEmbeddings(
# #         model_name=EMBED_MODEL_NAME,
# #         # model_kwargs={"local_files_only": True}
# #     )

# # embedding = load_embeddings()

# # === Initialize Vector Store ===
# if not os.path.exists(VECTOR_DB_DIR):
#     os.makedirs(VECTOR_DB_DIR)

# if os.path.exists(f"{VECTOR_DB_DIR}/index.faiss"):
#     vectorstore = FAISS.load_local(VECTOR_DB_DIR, embeddings=embedding, index_name="oss", allow_dangerous_deserialization=True)
# else:
#     vectorstore = None

# # === Store Project Metadata ===
# def store_to_vector_index(project_name, metadata_dict):
#     global vectorstore

#     full_text = "\n".join([f"{k}: {v}" for k, v in metadata_dict.items() if isinstance(v, (str, int, float))])
#     doc = Document(page_content=full_text, metadata={"project": project_name})

#     if vectorstore is None:
#         vectorstore = FAISS.from_documents([doc], embedding)
#     else:
#         vectorstore.add_documents([doc])

#     vectorstore.save_local(VECTOR_DB_DIR, index_name="oss")

# # === Query Vector Store ===
# def query_vector_index(question: str) -> str:
#     try:
#         if vectorstore is None:
#             return "❌ Vector index is empty. Please analyze a repository first."
#         results = vectorstore.similarity_search(question, k=1)
#         return results[0].page_content if results else "⚠️ No relevant match found."
#     except Exception as e:
#         return f"❌ Error during vector search: {str(e)}"

# # === Run Prompt Chain ===
# def run_chain(context_block, selected_question):
#     return chain.run({"context": context_block, "q": selected_question})

# # --- Content from oss_power_analyser.py ---
# # load_dotenv()
# # GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
# # GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# GITHUB_TOKEN = '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'
# GROQ_API_KEY = 'xai-AQluiJN8n4xW6zw0racIjrm4r6gyF5cBCPgT8vkvI1BJUQsZ2shtpLxkuRMkmAnkLWtxswYC9sJckbDs'
# # GROQ_API_KEY = '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'


# def handle_remove_readonly(func, path, exc_info):
#     os.chmod(path, stat.S_IWRITE)
#     func(path)

# def clone_repo(repo_url, clone_dir="cloned_repo"):
#     if os.path.exists(clone_dir):
#         shutil.rmtree(clone_dir, onerror=handle_remove_readonly)
#     subprocess.run(["git", "clone", repo_url, clone_dir], check=True)
#     return clone_dir

# def extract_static_metadata(repo_path):
#     metadata = {}
#     readme_text = ""
#     for name in ["README.md", "README.rst"]:
#         path = os.path.join(repo_path, name)
#         if os.path.exists(path):
#             with open(path, encoding="utf-8") as f:
#                 readme_text = f.read()
#             break
#     metadata.update({
#         "Detailed description": readme_text[:1000] if readme_text else "NF",
#         "Code Used": "Yes" if any(file.endswith(('.py','.cpp','.c','.java')) for root, _, files in os.walk(repo_path) for file in files) else "NF",
#         "Licence": "NF",
#         "has_contributing": os.path.exists(os.path.join(repo_path, "CONTRIBUTING.md")),
#         "has_code_of_conduct": os.path.exists(os.path.join(repo_path, "CODE_OF_CONDUCT.md")),
#         "has_tests": any(re.match(r"test", name, re.I) for root, dirs, files in os.walk(repo_path) for name in dirs + files),
#         "module_count": len([d for d in os.listdir(repo_path) if os.path.isdir(os.path.join(repo_path, d)) and d != ".git"]),
#         "platforms": [p for p in ["Linux", "Windows", "CUDA", "GPU"] if p in readme_text],
#         "README": readme_text
#     })
#     return metadata

# def get_github_metadata(repo_url):
#     g = Github(GITHUB_TOKEN)
#     user_repo = "/".join(repo_url.split("/")[-2:]).replace(".git", "")
#     repo = g.get_repo(user_repo)
#     commits = list(repo.get_commits())
#     return {
#         "github repo": user_repo,
#         "First update date": commits[-1].commit.author.date.isoformat() if commits else "NF",
#         " last update": commits[0].commit.author.date.isoformat() if commits else "NF",
#         "Releases": repo.get_releases().totalCount,
#         "Active Contributors": repo.get_contributors().totalCount,
#         "Open/closed issues": repo.get_issues(state="all").totalCount,
#         "Fork Count": repo.forks_count,
#         "GitHub Stars/Forks": f"{repo.stargazers_count} stars / {repo.forks_count} forks",
#         "License Type": repo.get_license().license.spdx_id if repo.license else "NF"
#     }

# def collect_full_repo_code_text(repo_path):
#     extensions = [".py", ".java", ".cpp", ".c", ".js", ".ts", ".html", ".css", ".rs", ".go", ".rb", ".json", ".xml"]
#     code_chunks = []
#     for root, _, files in os.walk(repo_path):
#         for file in files:
#             if any(file.endswith(ext) for ext in extensions):
#                 try:
#                     with open(os.path.join(root, file), "r", encoding="utf-8", errors="ignore") as f:
#                         code_chunks.append(f.read())
#                 except:
#                     continue
#     return "\n\n".join(code_chunks)

# def safe_json_parse(text):
#     try:
#         match = re.search(r"\{[\s\S]+\}", text)
#         if match:
#             return json.loads(match.group())
#     except Exception as e:
#         print("⚠️ JSON Parse Error:", e)
#     return {}

# def llm_extract_features(readme, license, module_count, dependencies, code):
#     llm = ChatGroq(api_key=GROQ_API_KEY, model="llama3-8b-8192")

#     full_context = f"""
# README:
# {readme[:3000]}

# LICENSE: {license}
# Modules: {module_count}
# Dependencies: {dependencies}
# CODE:
# {code[:10000]}
# """

#     prompt = ChatPromptTemplate.from_template(
#         """
# From the given README, LICENSE, and CODE extract the following features if available (use "NF" if not found):
# - Scale
# - Time criticality
# - Software function
# - Simulation Accuracy
# - Real-Time Processing
# - Fault Tolerance Mechanism
# - Standards Interoperability
# - Platform Support
# - SCADA/EMS Presence
# - ROI for OSS
# - DER Type
# - Voltage Level
# - Node/Buses Count
# - Control Architecture
# - Redundancy
# - Cyber-Physical Integration
# - Resilience Strategy
# - Energy Not Supplied (ENS)
# - Licensing Savings
# - Maintenance Cost
# - TCO (Total Cost of Ownership)
# - Customer Diversity
# - Reusability Value
# - Simulation Tools
# - Real-Time Control Systems
# - Planning / Optimization Models
# - Energy Management Systems
# - Data Analytics & Forecasting Tools

# Respond in JSON format only.
#         """
#     )

#     chain = prompt | llm
#     try:
#         response = chain.invoke({
#             "readme": readme,
#             "license": license,
#             "modules": module_count,
#             "deps": dependencies,
#             "code": code
#         })

#         parsed = safe_json_parse(response.content)

#         EXPECTED_KEYS = [
#             "Scale", "Time criticality", "Software function", "Simulation Accuracy",
#             "Real-Time Processing", "Fault Tolerance Mechanism", "Standards Interoperability",
#             "Platform Support", "SCADA/EMS Presence", "ROI for OSS", "DER Type", "Voltage Level",
#             "Node/Buses Count", "Control Architecture", "Redundancy", "Cyber-Physical Integration",
#             "Resilience Strategy", "Energy Not Supplied (ENS)", "Licensing Savings", "Maintenance Cost",
#             "TCO (Total Cost of Ownership)", "Customer Diversity", "Reusability Value", "Simulation Tools",
#             "Real-Time Control Systems", "Planning / Optimization Models", "Energy Management Systems",
#             "Data Analytics & Forecasting Tools"
#         ]

#         for key in EXPECTED_KEYS:
#             parsed.setdefault(key, "NF")

#         return parsed

#     except Exception as e:
#         print(f"🛑 Error in llm_extract_features: {e}")
#         return {k: "NF" for k in EXPECTED_KEYS}

# def analyze_multiple_repos_with_logs(repo_urls, log_fn):
#     all_features = []
#     for url in repo_urls:
#         try:
#             log_fn(f"📥 Cloning repo: `{url}`")
#             path = clone_repo(url)

#             log_fn("📄 Extracting metadata...")
#             static_meta = extract_static_metadata(path)
#             gh_meta = get_github_metadata(url)
#             code = collect_full_repo_code_text(path)

#             # log_fn("⚙️ Running Syft + Grype...")
#             # syft = run_syft(path)
#             # grype = run_grype(path)
#             # dependencies = len(syft.get("artifacts", []))
#             # vulnerabilities = len(grype.get("matches", []))

#             log_fn("⚠️ Skipping Syft + Grype (Streamlit Cloud)")
#             dependencies = "NF"
#             vulnerabilities = "NF"

#             log_fn("🧠 Running LLM inference...")
#             llm_features = llm_extract_features(
#                 static_meta.get("README", ""),
#                 gh_meta.get("License Type", "NF"),
#                 static_meta.get("module_count", 0),
#                 dependencies,
#                 code
#             )

#             combined = {
#                 **gh_meta,
#                 "Title": "NF",
#                 "Link": url,
#                 "Creator": "NF",
#                 "Creator specific": "NF",
#                 **static_meta,
#                 "dependency_count": dependencies,
#                 "vulnerabilities": vulnerabilities,
#                 **llm_features
#             }
#             all_features.append(combined)
#             store_to_vector_index(combined["github repo"], combined)
#             log_fn(f"✅ Done: `{url}`")
#         except Exception as e:
#             log_fn(f"❌ Failed for `{url}`: {str(e)}")
#     return all_features

# def save_all(features_list, full_filename="features_output", essential_filename="essential_features"):
#     ESSENTIAL_FEATURES = [
#         "Title", "Link", "github repo", "Creator", "Creator specific", "License Type", "Fork Count", "GitHub Stars/Forks",
#         "First update date", " last update", "Releases", "Active Contributors",
#         "Open/closed issues", "Detailed description", "Programming Language used", 
#         "has_contributing", "has_code_of_conduct", "has_tests", "module_count", "platforms", "README",
#         "dependency_count", "vulnerabilities", "Third-party Integrations", "Downloads / Installs Count", "User Community Size",
#         "DER Type", "Version Release Frequency", "Vendor Diversity", "SCADA/EMS Presence", "Data Analytics & Forecasting Tools",
#         "ROI for OSS", "Integration Cost", "Vendor Lock-in Avoidance", "Customer Diversity"
#     ]

#     FULL_FEATURE_LIST = list(set(ESSENTIAL_FEATURES + [
#         "Community Support", "Documentation Completeness", "Maintenance History",
#         "Dependency Freshness", "Code Modularity", "CI/CD Availability", "Issue Resolution Time",
#         "Code Review Coverage", "Test Coverage", "Commit Frequency Trend", "Bus Factor",
#         "Simulation Accuracy", "API Integration", "Real-Time Processing", "Platform Support",
#         "Hardware Interfacing", "Scalability", "Fault Tolerance Mechanism", "Validation Availability",
#         "Standards Interoperability", "Model Abstraction", "Extensibility / Plugin Support", "Resource Efficiency",
#         "Security Features", "Deployment Modes", "Containerization Support", "Real-World Use Cases",
#         "Institutional Backing", "Citations", "Educational Usage", "Benchmarks Participation",
#         "Language Localization", "Codebase Size", "Commercial Support Availability", "Social Media Mentions",
#         "Voltage Level", "Network Topology", "Control Architecture", "Redundancy",
#         "Node/Buses Count", "Cyber-Physical Integration", "Resilience Strategy",
#         "Fault Recovery Time", "MTBF (Mean Time Between Failures)",
#         "SAIDI/SAIFI", "Load Stability", "Voltage/Frequency Stability", "DER Hosting Capacity",
#         "Power Quality", "Islanding Accuracy", "Energy Not Supplied (ENS)", "Control Responsiveness",
#         "Licensing Savings", "Maintenance Cost", "TCO (Total Cost of Ownership)", "Downtime Cost",
#         "Reusability Value", "Training Cost", "Collaborative Cost Sharing",
#         "Simulation Tools", "Real-Time Control Systems", "Planning / Optimization Models",
#         "Energy Management Systems"
#     ]))

#     # Save full features
#     full_data = []
#     for project in features_list:
#         entry = {key: project.get(key, "NF") for key in FULL_FEATURE_LIST}
#         full_data.append(entry)
#     pd.DataFrame(full_data).to_csv(f"{full_filename}.csv", index=False)
#     with open(f"{full_filename}.json", "w") as f:
#         json.dump(full_data, f, indent=2)

#     # Save essential features
#     essential_data = []
#     for project in features_list:
#         entry = {key: project.get(key, "NF") for key in ESSENTIAL_FEATURES}
#         essential_data.append(entry)
#     pd.DataFrame(essential_data).to_csv(f"{essential_filename}.csv", index=False)
#     with open(f"{essential_filename}.json", "w") as f:
#         json.dump(essential_data, f, indent=2)

# # --- Content from app.py (Main Streamlit App) ---
# st.write("✅ App started successfully.")

# st.set_page_config(page_title="RECOPS", layout="wide")

# st.markdown("""
#     <style>
#         body { background-color: #f9f9f9; color: #333; }
#         .main-title { font-size: 48px; font-weight: bold; color: #1e3a8a; text-align: center; margin-top: 40px; }
#         .section-title { font-size: 24px; font-weight: 600; margin-top: 30px; color: #0f172a; }
#         .footer { position: fixed; bottom: 10px; left: 0; width: 100%; text-align: center; color: #666; font-size: 14px; }
#         .floating-spinner {
#             position: fixed;
#             top: 80px;
#             right: 40px;
#             background: #facc15;
#             color: #000;
#             font-weight: bold;
#             padding: 8px 14px;
#             border-radius: 50px;
#             z-index: 9999;
#             box-shadow: 2px 2px 6px rgba(0,0,0,0.2);
#             animation: pulse 1.5s infinite;
#         }
#         @keyframes pulse {
#             0% { transform: scale(1); }
#             50% { transform: scale(1.1); }
#             100% { transform: scale(1); }
#         }
#     </style>
# """, unsafe_allow_html=True)

# st.markdown('<div class="main-title">RECOPS: Resilience and Cost-benefits of Open Source Software in the Power Sector</div>', unsafe_allow_html=True)

# page = st.sidebar.radio("Navigate", ["Home", "Feature Analysis", "Ask LLM", "Past Analyses"])

# if "analysis_thread" not in st.session_state:
#     st.session_state.analysis_thread = None
#     st.session_state.analysis_done = False
#     st.session_state.analysis_result = []

# if page == "Home":
#     st.markdown("""
#     ### Welcome to RECOPS
#     This tool helps analyze open-source software in the power systems domain for features like:
#     - Technical metadata
#     - Security vulnerabilities
#     - Lock-in risk
#     - Domain applicability
#     - Compliance with power sector standards
#     """)

# elif page == "Feature Analysis":
#     st.markdown("<div class='section-title'>🔍 OSS Feature Analyzer</div>", unsafe_allow_html=True)
#     urls_input = st.text_area("Enter up to 200 GitHub Repository URLs (one per line)")
#     urls = [u.strip() for u in urls_input.strip().split("\n") if u.strip()][:200]
#     log_container = st.empty()

#     if urls and st.button("Start Analysis"):
#         log_lines = []
#         def log_fn(msg):
#             log_lines.append(msg)
#             log_container.markdown("  \n".join(log_lines[-12:]))

#         st.markdown('<div class="floating-spinner">⚙️ Analysing...</div>', unsafe_allow_html=True)
#         st.info("⏳ Analysis started. Please do not refresh or switch away until complete.")

#         result = analyze_multiple_repos_with_logs(urls, log_fn)
#         st.session_state.analysis_result = result

#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         full_filename = f"features_output_{timestamp}"
#         essential_filename = f"essential_features_{timestamp}"

#         save_all(result, full_filename, essential_filename)
#         # save_history_entry(full_filename)

#         st.session_state.full_csv = f"{full_filename}.csv"
#         st.session_state.full_json = f"{full_filename}.json"
#         st.session_state.ess_csv = f"{essential_filename}.csv"
#         st.session_state.ess_json = f"{essential_filename}.json"

#         st.session_state.analysis_done = True

#     if st.session_state.analysis_thread and st.session_state.analysis_thread.is_alive():
#         st.warning("🔄 Analysis in progress...")
#     elif st.session_state.analysis_done:
#         st.success("✅ Analysis complete!")
#         st.subheader("🔧 Extracted Features")
#         st.json(st.session_state.analysis_result)

#         st.download_button("📥 Download ESSENTIAL FEATURES (CSV)", open(st.session_state.ess_csv, "rb"), file_name=st.session_state.ess_csv)
#         st.download_button("📥 Download ESSENTIAL FEATURES (JSON)", open(st.session_state.ess_json, "rb"), file_name=st.session_state.ess_json)
#         st.download_button("📥 Download FULL FEATURES (CSV)", open(st.session_state.full_csv, "rb"), file_name=st.session_state.full_csv)
#         st.download_button("📥 Download FULL FEATURES (JSON)", open(st.session_state.full_json, "rb"), file_name=st.session_state.full_json)

# elif page == "Ask LLM":
#     st.markdown("<div class='section-title'>🤖 Ask LLM About a Project</div>", unsafe_allow_html=True)
#     st.markdown("#### 1. Ask custom questions on OSS behavior")
#     question = st.text_input("Enter your question (based on analyzed project code):")
#     if question:
#         with st.spinner("Thinking..."):
#             answer = query_vector_index(question)
#         st.success("LLM Answer:")
#         st.markdown(f"`{answer}`")

#     st.markdown("---")
#     st.markdown("#### 2. Bonus Insights (LLM-based project summaries)")
#     sample_questions = [
#         "What unique power sector capability does this OSS offer?",
#         "Does it help reduce vendor lock-in?",
#         "Is this software suitable for real-time grid operations?",
#         "How scalable is this solution for a national utility?",
#         "Can this be integrated with existing EMS or SCADA?"
#     ]
#     selected_question = st.selectbox("Pick a common question to evaluate:", sample_questions)
#     if st.button("Run this insight"):
#         if "analysis_result" in st.session_state and st.session_state.analysis_result:
#             try:
#                 context_block = json.dumps(st.session_state.analysis_result[0])[:5000]
#                 llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="llama3-8b-8192")
#                 prompt = PromptTemplate(
#                     input_variables=["context", "q"],
#                     template="""
# Based on the OSS project metadata below, answer the following question:
# Context:
# {context}

# Question:
# {q}

# Answer:
# """
#                 )
#                 chain = LLMChain(llm=llm, prompt=prompt)
#                 result = chain.run({"context": context_block, "q": selected_question})
#                 st.success("Insight:")
#                 st.write(result.strip())
#             except Exception as e:
#                 st.error(f"LLM processing failed: {str(e)}")
#         else:
#             st.warning("❗ Please run feature analysis first from the sidebar.")

# elif page == "Past Analyses":
#     st.markdown("<div class='section-title'>📁 Past Runs</div>", unsafe_allow_html=True)
#     past_runs = get_past_runs()
#     selected = st.selectbox("Choose one:", past_runs)
#     if selected:
#         st.download_button("📥 Download CSV", open(f"{selected}.csv", "rb"), file_name=f"{selected}.csv")
#         st.download_button("📥 Download JSON", open(f"{selected}.json", "rb"), file_name=f"{selected}.json")

# st.markdown("<div class='footer'>Developed by Shashank</div>", unsafe_allow_html=True)
