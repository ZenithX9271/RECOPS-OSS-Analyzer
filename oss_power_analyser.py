# === oss_power_analyser.py ===
import os
import re
import json
import shutil
import subprocess
import stat
import pandas as pd
from github import Github
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from datetime import datetime
from rag_utils import store_to_vector_index
import streamlit as st

def get_tokens():
    import streamlit as st
    return st.secrets["GITHUB_TOKEN"], st.secrets["GROQ_API_KEY"]

GITHUB_TOKEN, GROQ_API_KEY = get_tokens()

# === Load tokens securely ===
load_dotenv()
# GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# GITHUB_TOKEN = st.secrets['GITHUB_TOKEN']
# GROQ_API_KEY = st.secrets['GROQ_API_KEY']

def handle_remove_readonly(func, path, exc_info):
    os.chmod(path, stat.S_IWRITE)
    func(path)

def clone_repo(repo_url, clone_dir="cloned_repo"):
    if os.path.exists(clone_dir):
        shutil.rmtree(clone_dir, onerror=handle_remove_readonly)
    subprocess.run(["git", "clone", repo_url, clone_dir], check=True)
    return clone_dir

def run_syft(repo_path):
    result_file = "sbom_syft.json"
    subprocess.run(["syft", repo_path, "-o", "json", "--output", f"json={result_file}"], check=True)
    with open(result_file) as f:
        return json.load(f)

def run_grype(repo_path):
    result_file = "vulns_grype.json"
    subprocess.run(["grype", repo_path, "-o", "json", "--output", f"json={result_file}"], check=True)
    with open(result_file) as f:
        return json.load(f)

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
    g = Github(GITHUB_TOKEN)
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
        print("⚠️ JSON Parse Error:", e)
    return {}

def llm_extract_features(readme, license, module_count, dependencies, code):
    llm = ChatGroq(api_key=GROQ_API_KEY, model="llama3-8b-8192")

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
            log_fn(f"📥 Cloning repo: `{url}`")
            path = clone_repo(url)

            log_fn("📄 Extracting metadata...")
            static_meta = extract_static_metadata(path)
            gh_meta = get_github_metadata(url)
            code = collect_full_repo_code_text(path)

            log_fn("⚙️ Running Syft + Grype...")
            syft = run_syft(path)
            grype = run_grype(path)
            dependencies = len(syft.get("artifacts", []))
            vulnerabilities = len(grype.get("matches", []))

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
            log_fn(f"✅ Done: `{url}`")
        except Exception as e:
            log_fn(f"❌ Failed for `{url}`: {str(e)}")
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
