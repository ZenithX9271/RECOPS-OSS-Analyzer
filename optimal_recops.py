import os
import warnings
import logging

# 1. Kill standard Python warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Accessing `__path__`.*")

# 2. Force Hugging Face to only report critical errors at the OS level
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# 3. Mute the specific Python loggers causing the noise
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)

import re
import json
import shutil
import subprocess
import stat
import pandas as pd
from datetime import datetime
import concurrent.futures
from github import Github, Auth
import streamlit as st
from dotenv import load_dotenv

# --- LangChain Imports ---
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# 🚨 IMPORTANT: Do not hardcode your real keys here if pushing to GitHub! Use the .env file.
GITHUB_TOKEN = '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' #'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' 
GROQ_API_KEY = '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'
# GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "") 
# GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# --- Page Configuration ---
st.set_page_config(page_title="RECOPS Fast Extractor", page_icon="⚡", layout="wide")

# --- Custom CSS for Enhanced UI ---
st.markdown("""
    <style>
        /* Global Styles */
        body { background-color: #f8fafc; }
        
        /* Typography & Headers */
        .main-header {
            background: -webkit-linear-gradient(45deg, #1e3a8a, #3b82f6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3rem;
            font-weight: 800;
            margin-bottom: 0.5rem;
        }
        .sub-header {
            color: #64748b;
            font-size: 1.2rem;
            margin-bottom: 2rem;
        }
        
        /* Cards */
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: transparent;
            border-radius: 4px 4px 0px 0px;
            gap: 1px;
            padding-top: 10px;
            padding-bottom: 10px;
        }
        
        /* Styled Terminal Box */
        .terminal-box {
            background-color: #0f172a;
            color: #10b981;
            font-family: 'Courier New', Courier, monospace;
            padding: 15px;
            border-radius: 8px;
            height: 250px;
            overflow-y: auto;
            border: 1px solid #334155;
            box-shadow: inset 0 2px 4px rgba(0,0,0,0.5);
        }
        
        /* Metrics Styling */
        div[data-testid="metric-container"] {
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
    </style>
""", unsafe_allow_html=True)

# --- Utilities ---
def handle_remove_readonly(func, path, exc_info):
    os.chmod(path, stat.S_IWRITE)
    func(path)

def shallow_clone(repo_url, clone_dir):
    if os.path.exists(clone_dir):
        shutil.rmtree(clone_dir, onerror=handle_remove_readonly)
    try:
        subprocess.run(["git", "clone", "--depth", "1", repo_url, clone_dir], check=True, capture_output=True)
        return clone_dir
    except subprocess.CalledProcessError as e:
        return None

def extract_local_metadata(repo_path):
    if not repo_path or not os.path.exists(repo_path):
        return {"README": "", "Detailed description": "NF", "has_tests": False, "module_count": 0}

    readme_text = ""
    for name in ["README.md", "README.rst", "readme.md"]:
        path = os.path.join(repo_path, name)
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8", errors="ignore") as f:
                    readme_text = f.read()
                break
            except Exception:
                continue

    return {
        "Detailed description": readme_text[:1000] if readme_text else "NF",
        "has_contributing": os.path.exists(os.path.join(repo_path, "CONTRIBUTING.md")),
        "has_code_of_conduct": os.path.exists(os.path.join(repo_path, "CODE_OF_CONDUCT.md")),
        "has_tests": any(re.match(r"test", name, re.I) for root, dirs, files in os.walk(repo_path) for name in dirs + files),
        "module_count": len([d for d in os.listdir(repo_path) if os.path.isdir(os.path.join(repo_path, d)) and d != ".git"]),
        "platforms": [p for p in ["Linux", "Windows", "CUDA", "GPU"] if p in readme_text],
        "README": readme_text
    }

def get_github_metadata(repo_url):
    if not GITHUB_TOKEN:
        return {"github repo": repo_url, "API_ERROR": "No Token"}
    try:
        g = Github(auth=Auth.Token(GITHUB_TOKEN))
        user_repo = "/".join(repo_url.split("/")[-2:]).replace(".git", "")
        repo = g.get_repo(user_repo)
        return {
            "github repo": user_repo,
            "Releases": repo.get_releases().totalCount,
            "Active Contributors": repo.get_contributors().totalCount,
            "Open/closed issues": repo.get_issues(state="all").totalCount,
            "Fork Count": repo.forks_count,
            "GitHub Stars/Forks": f"{repo.stargazers_count} stars / {repo.forks_count} forks",
            "License Type": repo.get_license().license.spdx_id if repo.license else "NF"
        }
    except Exception as e:
        return {"github repo": repo_url, "GitHub API Error": str(e)}

def safe_json_parse(text):
    try:
        match = re.search(r"\{[\s\S]+\}", text)
        if match:
            return json.loads(match.group())
    except Exception:
        pass
    return {}

def fast_llm_extraction(readme):
    if not GROQ_API_KEY or not readme:
        return {}

    llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.1-8b-instant")
    prompt = ChatPromptTemplate.from_template(
        """
Analyze the following GitHub README for an open-source power system software.
Extract the following technical features. If the feature is NOT explicitly mentioned in the text, you MUST output "NF". Do not guess.

README Text (First 5000 chars):
{readme}

Extract into strictly formatted JSON:
- Software function
- Platform Support
- SCADA/EMS Presence
- DER Type (Distributed Energy Resources)
- Voltage Level
- Node/Buses Count
- Control Architecture
- Simulation Tools
- Data Analytics & Forecasting Tools
- Standards Interoperability
        """
    )
    chain = prompt | llm
    try:
        response = chain.invoke({"readme": readme[:5000]})
        return safe_json_parse(response.content)
    except Exception:
        return {}

def process_single_repo(url):
    repo_name = url.split('/')[-1].replace(".git", "")
    clone_dir = f"temp_clones/clone_{repo_name}_{datetime.now().strftime('%M%S%f')}"
    
    try:
        path = shallow_clone(url, clone_dir)
        local_meta = extract_local_metadata(path)
        gh_meta = get_github_metadata(url)
        llm_meta = fast_llm_extraction(local_meta.get("README", "")) if local_meta.get("README") else {}

        if os.path.exists(clone_dir):
            shutil.rmtree(clone_dir, onerror=handle_remove_readonly)

        return {
            "Link": url,
            "Status": "Success",
            **gh_meta,
            **{k: v for k, v in local_meta.items() if k != "README"},
            **llm_meta
        }
    except Exception as e:
        if os.path.exists(clone_dir):
            shutil.rmtree(clone_dir, onerror=handle_remove_readonly)
        return {"Link": url, "Status": f"Failed: {str(e)}"}

# --- Streamlit UI App ---
st.markdown('<div class="main-header">⚡ RECOPS Fast Extractor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">High-speed ETL pipeline for Power System OSS metadata extraction.</div>', unsafe_allow_html=True)

if not GITHUB_TOKEN or not GROQ_API_KEY:
    st.error("🚨 Missing API Keys! Please configure GITHUB_TOKEN and GROQ_API_KEY.")
    st.stop()

# Input Section with Tabs
tab1, tab2 = st.tabs(["📝 Manual Entry", "📁 CSV Bulk Upload"])
urls = []

with tab1:
    st.markdown("### Paste Repository URLs")
    urls_input = st.text_area("Enter GitHub Repository URLs (one per line):", height=150, placeholder="https://github.com/e2nIEE/pandapower\nhttps://github.com/NREL/OpenDSS...")
    if urls_input:
        urls = [u.strip() for u in urls_input.strip().split("\n") if u.strip()]

with tab2:
    st.markdown("### Upload a CSV File")
    st.info("Ensure your CSV file has a column named **'URL'**, **'Link'**, or **'Repository'**.")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            st.dataframe(df_upload.head(3), use_container_width=True)
            
            # Smart column detection
            url_col = None
            for col in df_upload.columns:
                if col.strip().lower() in ['url', 'link', 'repository', 'github_url', 'repo']:
                    url_col = col
                    break
            
            if url_col:
                urls = df_upload[url_col].dropna().astype(str).tolist()
                st.success(f"✅ Extracted {len(urls)} URLs from column '{url_col}'.")
            else:
                # Fallback: Just take the first column
                urls = df_upload.iloc[:, 0].dropna().astype(str).tolist()
                st.warning(f"Could not auto-detect URL column. Using first column. Found {len(urls)} URLs.")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

# Action Section
st.divider()

if st.button("🚀 Run Extraction Pipeline", type="primary", use_container_width=True):
    if not urls:
        st.warning("Please provide repository URLs to begin.")
    else:
        # Prepare tracking
        if not os.path.exists("temp_clones"):
            os.makedirs("temp_clones")

        progress_bar = st.progress(0)
        
        # Terminal UI setup
        st.markdown("### 🖥️ Processing Terminal")
        terminal_container = st.empty()
        log_text = f"> Initiating parallel workers for {len(urls)} repositories...\n"
        terminal_container.markdown(f'<div class="terminal-box">{log_text}</div>', unsafe_allow_html=True)
        
        results = []
        success_count = 0
        fail_count = 0
        
        # Parallel Execution
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_url = {executor.submit(process_single_repo, url): url for url in urls}
            
            for i, future in enumerate(concurrent.futures.as_completed(future_to_url)):
                url = future_to_url[future]
                repo_short_name = url.split('/')[-1]
                
                try:
                    data = future.result()
                    results.append(data)
                    
                    if data.get("Status") == "Success":
                        success_count += 1
                        log_text += f"> [{datetime.now().strftime('%H:%M:%S')}] ✅ Success: Analyzed {repo_short_name}\n"
                    else:
                        fail_count += 1
                        log_text += f"> [{datetime.now().strftime('%H:%M:%S')}] ❌ Failed: {repo_short_name} -> {data.get('Status')}\n"
                        
                except Exception as exc:
                    fail_count += 1
                    log_text += f"> [{datetime.now().strftime('%H:%M:%S')}] ❌ Error: {repo_short_name} generated an exception.\n"
                
                # Update UI elements
                progress = (i + 1) / len(urls)
                progress_bar.progress(progress)
                terminal_container.markdown(f'<div class="terminal-box">{log_text}</div>', unsafe_allow_html=True)

        st.success("🎉 Extraction Pipeline Finished!")
        
        # Metrics Display
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Processed", len(urls))
        col2.metric("Successful", success_count)
        col3.metric("Failed", fail_count)
        
        st.divider()
        
        # Output Results
        # FIX: Force all nested lists/dictionaries into strings so Arrow doesn't crash
        for row in results:
            for key, value in row.items():
                if isinstance(value, (list, dict)):
                    row[key] = str(value)

        df_results = pd.DataFrame(results)
        st.markdown("### 📊 Extracted Data")
        
        # FIX: Update Streamlit syntax to remove the deprecation warnings
        st.dataframe(df_results, width="stretch")

        # Download
        csv_data = df_results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Extracted Features (CSV)",
            data=csv_data,
            file_name=f"recops_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        # Final cleanup
        if os.path.exists("temp_clones"):
            shutil.rmtree("temp_clones", onerror=handle_remove_readonly)









# import os
# import warnings
# import logging

# # 1. Kill standard Python warnings
# warnings.filterwarnings("ignore", category=UserWarning)
# warnings.filterwarnings("ignore", message=".*Accessing `__path__`.*")

# # 2. Force Hugging Face to only report critical errors at the OS level
# os.environ["TRANSFORMERS_VERBOSITY"] = "error"
# os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# # 3. Mute the specific Python loggers causing the noise
# logging.getLogger("transformers").setLevel(logging.ERROR)
# logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)

# import re
# import json
# import shutil
# import subprocess
# import stat
# import pandas as pd
# from datetime import datetime
# import concurrent.futures
# from github import Github, Auth
# import streamlit as st
# from dotenv import load_dotenv

# # --- LangChain Imports ---
# from langchain_groq import ChatGroq
# from langchain_core.prompts import ChatPromptTemplate

# # Load environment variables (Make sure you have a .env file!)
# load_dotenv()

# GITHUB_TOKEN = '' 
# GROQ_API_KEY = ''

# st.set_page_config(page_title="RECOPS Fast Extractor", layout="wide")

# # --- Utilities ---
# def handle_remove_readonly(func, path, exc_info):
#     """Helper to remove read-only files during git clone cleanup."""
#     os.chmod(path, stat.S_IWRITE)
#     func(path)

# def shallow_clone(repo_url, clone_dir):
#     """Clones ONLY the latest commit to save massive amounts of time and bandwidth."""
#     if os.path.exists(clone_dir):
#         shutil.rmtree(clone_dir, onerror=handle_remove_readonly)
#     try:
#         # --depth 1 makes it 10x faster
#         subprocess.run(["git", "clone", "--depth", "1", repo_url, clone_dir], check=True, capture_output=True)
#         return clone_dir
#     except subprocess.CalledProcessError as e:
#         print(f"Clone failed for {repo_url}: {e}")
#         return None

# def extract_local_metadata(repo_path):
#     """Extracts metadata purely from the local file structure."""
#     if not repo_path or not os.path.exists(repo_path):
#         return {"README": "", "Detailed description": "NF", "has_tests": False, "module_count": 0}

#     readme_text = ""
#     for name in ["README.md", "README.rst", "readme.md"]:
#         path = os.path.join(repo_path, name)
#         if os.path.exists(path):
#             try:
#                 with open(path, encoding="utf-8", errors="ignore") as f:
#                     readme_text = f.read()
#                 break
#             except Exception:
#                 continue

#     return {
#         "Detailed description": readme_text[:1000] if readme_text else "NF",
#         "has_contributing": os.path.exists(os.path.join(repo_path, "CONTRIBUTING.md")),
#         "has_code_of_conduct": os.path.exists(os.path.join(repo_path, "CODE_OF_CONDUCT.md")),
#         "has_tests": any(re.match(r"test", name, re.I) for root, dirs, files in os.walk(repo_path) for name in dirs + files),
#         "module_count": len([d for d in os.listdir(repo_path) if os.path.isdir(os.path.join(repo_path, d)) and d != ".git"]),
#         "platforms": [p for p in ["Linux", "Windows", "CUDA", "GPU"] if p in readme_text],
#         "README": readme_text
#     }

# def get_github_metadata(repo_url):
#     """Fetches high-level metrics via GitHub API."""
#     if not GITHUB_TOKEN:
#         return {"github repo": repo_url, "API_ERROR": "No Token"}
    
#     try:
#         g = Github(auth=Auth.Token(GITHUB_TOKEN))
#         user_repo = "/".join(repo_url.split("/")[-2:]).replace(".git", "")
#         repo = g.get_repo(user_repo)
        
#         return {
#             "github repo": user_repo,
#             "Releases": repo.get_releases().totalCount,
#             "Active Contributors": repo.get_contributors().totalCount,
#             "Open/closed issues": repo.get_issues(state="all").totalCount,
#             "Fork Count": repo.forks_count,
#             "GitHub Stars/Forks": f"{repo.stargazers_count} stars / {repo.forks_count} forks",
#             "License Type": repo.get_license().license.spdx_id if repo.license else "NF"
#         }
#     except Exception as e:
#         return {"github repo": repo_url, "GitHub API Error": str(e)}

# def safe_json_parse(text):
#     try:
#         match = re.search(r"\{[\s\S]+\}", text)
#         if match:
#             return json.loads(match.group())
#     except Exception:
#         pass
#     return {}

# def fast_llm_extraction(readme):
#     """Uses Groq to quickly extract ONLY technical features found in documentation."""
#     if not GROQ_API_KEY or not readme:
#         return {}

#     llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.1-8b-instant")

#     prompt = ChatPromptTemplate.from_template(
#         """
# Analyze the following GitHub README for an open-source power system software.
# Extract the following technical features. If the feature is NOT explicitly mentioned in the text, you MUST output "NF". Do not guess.

# README Text (First 5000 chars):
# {readme}

# Extract into strictly formatted JSON:
# - Software function
# - Platform Support
# - SCADA/EMS Presence
# - DER Type (Distributed Energy Resources)
# - Voltage Level
# - Node/Buses Count
# - Control Architecture
# - Simulation Tools
# - Data Analytics & Forecasting Tools
# - Standards Interoperability
#         """
#     )

#     chain = prompt | llm
    
#     try:
#         # Only pass the first 5000 chars of README to save time/tokens
#         response = chain.invoke({"readme": readme[:5000]})
#         return safe_json_parse(response.content)
#     except Exception as e:
#         print(f"LLM Error: {e}")
#         return {}

# def process_single_repo(url):
#     """The master pipeline for a single URL."""
#     # Create a unique folder name to avoid thread collisions
#     repo_name = url.split('/')[-1].replace(".git", "")
#     clone_dir = f"temp_clones/clone_{repo_name}_{datetime.now().strftime('%M%S%f')}"
    
#     try:
#         path = shallow_clone(url, clone_dir)
#         local_meta = extract_local_metadata(path)
#         gh_meta = get_github_metadata(url)
        
#         # Only run LLM if we found a README
#         llm_meta = fast_llm_extraction(local_meta.get("README", "")) if local_meta.get("README") else {}

#         # Cleanup the clone immediately to save disk space
#         if os.path.exists(clone_dir):
#             shutil.rmtree(clone_dir, onerror=handle_remove_readonly)

#         # Merge everything
#         return {
#             "Link": url,
#             **gh_meta,
#             **{k: v for k, v in local_meta.items() if k != "README"}, # Don't bloat the CSV with the full README text
#             **llm_meta
#         }
#     except Exception as e:
#         if os.path.exists(clone_dir):
#             shutil.rmtree(clone_dir, onerror=handle_remove_readonly)
#         return {"Link": url, "Error": str(e)}

# # --- Streamlit UI ---
# st.markdown("""
#     <style>
#         .main-title { font-size: 36px; font-weight: bold; color: #1e3a8a; margin-top: 20px; }
#         .sub-title { font-size: 18px; color: #64748b; margin-bottom: 30px;}
#     </style>
# """, unsafe_allow_html=True)

# st.markdown('<div class="main-title">⚡ RECOPS Fast Extractor</div>', unsafe_allow_html=True)
# st.markdown('<div class="sub-title">High-speed ETL pipeline for Power System OSS metadata.</div>', unsafe_allow_html=True)

# if not GITHUB_TOKEN or not GROQ_API_KEY:
#     st.error("🚨 Missing API Keys! Please check your `.env` file for GITHUB_TOKEN and GROQ_API_KEY.")
#     st.stop()

# urls_input = st.text_area("Enter GitHub Repository URLs (one per line):", height=200)
# urls = [u.strip() for u in urls_input.strip().split("\n") if u.strip()]

# if st.button("🚀 Start Fast Extraction", type="primary"):
#     if not urls:
#         st.warning("Please enter at least one URL.")
#     else:
#         # Create temp dir for cloning
#         if not os.path.exists("temp_clones"):
#             os.makedirs("temp_clones")

#         progress_bar = st.progress(0)
#         status_text = st.empty()
        
#         results = []
        
#         # Multithreading magic: Process up to 5 repos at the exact same time
#         status_text.text(f"Starting parallel processing for {len(urls)} repositories...")
        
#         with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
#             # Map the function to the URLs
#             future_to_url = {executor.submit(process_single_repo, url): url for url in urls}
            
#             for i, future in enumerate(concurrent.futures.as_completed(future_to_url)):
#                 url = future_to_url[future]
#                 try:
#                     data = future.result()
#                     results.append(data)
#                 except Exception as exc:
#                     st.error(f"{url} generated an exception: {exc}")
                
#                 # Update UI
#                 progress = (i + 1) / len(urls)
#                 progress_bar.progress(progress)
#                 status_text.text(f"Processed {i+1} of {len(urls)} repos...")

#         st.success("✅ Extraction Complete!")
        
#         # Convert to Pandas and show
#         df = pd.DataFrame(results)
#         st.dataframe(df)

#         # Download buttons
#         csv = df.to_csv(index=False).encode('utf-8')
#         st.download_button(
#             label="📥 Download Data as CSV",
#             data=csv,
#             file_name=f"recops_fast_extract_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
#             mime="text/csv",
#         )
        
#         # Final cleanup of temp directory
#         if os.path.exists("temp_clones"):
#             shutil.rmtree("temp_clones", onerror=handle_remove_readonly)
