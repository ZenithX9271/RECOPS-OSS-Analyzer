# === FILE: oss_power_analyser.py ===
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
from datetime import datetime
from rag_utils import store_to_vector_index

# Load environment variables
dotenv_loaded = load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


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
    except Exception:
        pass
    return {}


def llm_extract_features(readme, license, module_count, dependencies, code):
    llm = ChatGroq(api_key=GROQ_API_KEY, model="llama3-8b-8192")
    prompt_text = (
        f"README:\n{readme[:3000]}\n\n"
        f"LICENSE: {license}\nModules: {module_count}\nDependencies: {dependencies}\n"
        f"CODE:\n{code[:10000]}\n"
        "\nExtract features in JSON:"
    )
    try:
        resp = llm.client.completions.create(prompt=prompt_text)
        parsed = safe_json_parse(resp.choices[0].text)
        for key in [
            "Scale", "Time criticality", "Software function", "Simulation Accuracy",
            "Real-Time Processing", "Fault Tolerance Mechanism", "Standards Interoperability",
            "Platform Support", "SCADA/EMS Presence", "ROI for OSS", "DER Type", "Voltage Level",
            "Node/Buses Count", "Control Architecture", "Redundancy", "Cyber-Physical Integration",
            "Resilience Strategy", "Energy Not Supplied (ENS)", "Licensing Savings", "Maintenance Cost",
            "TCO (Total Cost of Ownership)", "Customer Diversity", "Reusability Value", "Simulation Tools",
            "Real-Time Control Systems", "Planning / Optimization Models", "Energy Management Systems",
            "Data Analytics & Forecasting Tools"
        ]:
            parsed.setdefault(key, "NF")
        return parsed
    except Exception:
        return {k: "NF" for k in key}


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

            log_fn("⚠️ Skipping Syft + Grype")
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

            combined = {**gh_meta, "Link": url, **static_meta,
                        "dependency_count": dependencies, "vulnerabilities": vulnerabilities,
                        **llm_features}
            all_features.append(combined)
            store_to_vector_index(combined["github repo"], combined)
            log_fn(f"✅ Done: `{url}`")
        except Exception as e:
            log_fn(f"❌ Failed for `{url}`: {e}")
    return all_features
