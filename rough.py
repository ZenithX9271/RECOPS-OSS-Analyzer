#!/usr/bin/env python3
"""
RECOPS Feature Extractor - Streamlit + CLI friendly single-file app.

Usage:
  - Streamlit GUI:
      streamlit run recops_app_GPT.py
    Fill repo URL, options and click Run.

  - CLI:
      export GITHUB_TOKEN="ghp_xxx"
      python recops_app_GPT.py --repo https://github.com/OpenEMS/openems

Outputs:
  - JSON and CSV files saved into ./outputs/
Notes:
  - Many features are heuristics and may be "NF" (not found) where not determinable.
  - Optional LLM blocks are disabled unless keys are provided and packages installed.
"""
import os
import re
import json
import shutil
import subprocess
import stat
import csv
import math
import sys
from pathlib import Path
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from typing import Dict, Any, List, Tuple, Optional

# third-party
try:
    import pandas as pd
except Exception:
    pd = None

# progress helpers (used in CLI; Streamlit uses its own spinner)
from tqdm import tqdm
from dateutil import parser as dateparser

# Git imports
try:
    from git import Repo, GitCommandError
except Exception:
    Repo = None
    GitCommandError = Exception

# PyGitHub
try:
    from github import Github, GithubException
    from github import Auth
except Exception:
    Github = None
    Auth = None
    GithubException = Exception

# -------------------------
# Configuration / envvars
# -------------------------
DEFAULT_OUTPUT_DIR = Path("outputs")
DEFAULT_CLONE_BASE = Path("cloned_repos")

# create output dirs
DEFAULT_OUTPUT_DIR.mkdir(exist_ok=True)
DEFAULT_CLONE_BASE.mkdir(exist_ok=True)

# -------------------------
# Utilities
# -------------------------
def ensure_clean_clone_dir(repo_name: str, base: Path = DEFAULT_CLONE_BASE) -> Path:
    path = base / repo_name.replace("/", "_")
    if path.exists():
        try:
            shutil.rmtree(path, onerror=handle_remove_readonly)
        except Exception:
            # fallback: attempt to remove readonly bits then remove
            for root, dirs, files in os.walk(path):
                for fname in files:
                    fp = Path(root) / fname
                    try:
                        fp.chmod(stat.S_IWRITE)
                    except Exception:
                        pass
            shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    return path

def handle_remove_readonly(func, path, exc_info):
    try:
        os.chmod(path, stat.S_IWRITE)
    except Exception:
        pass
    func(path)

def safe_read_text(path: Path, max_chars: Optional[int] = None) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
        return text if max_chars is None else text[:max_chars]
    except Exception:
        return ""

def list_files(repo_path: Path):
    for root, dirs, files in os.walk(repo_path):
        for f in files:
            yield Path(root) / f

def top_k(seq, k=5):
    return sorted(seq, reverse=True)[:k]

# -------------------------
# GitHub helpers
# -------------------------
def github_client(token: Optional[str]):
    if not token:
        raise ValueError("GITHUB_TOKEN not set")
    if Github is None or Auth is None:
        raise RuntimeError("PyGithub not installed. Install with `pip install PyGithub`")
    return Github(auth=Auth.Token(token), per_page=100)

def repo_fullname_from_url(url: str) -> str:
    parts = url.strip().rstrip("/").split("/")
    # support git@github.com:owner/repo.git
    if ":" in parts[-1] and parts[0].startswith("git@"):
        # git@github.com:owner/repo.git
        tail = parts[-1]
        tail = tail.split(":", 1)[-1]
        return tail.replace(".git", "")
    return "/".join(parts[-2:]).replace(".git", "")

# -------------------------
# Cloning & local collection
# -------------------------
def clone_repo(repo_url: str, clone_dir: Optional[Path] = None) -> Path:
    if Repo is None:
        raise RuntimeError("GitPython not installed. Install with `pip install GitPython`")
    repo_name = repo_fullname_from_url(repo_url)
    if clone_dir is None:
        clone_dir = ensure_clean_clone_dir(repo_name)
    try:
        print(f"[git] cloning {repo_url} -> {clone_dir}")
        Repo.clone_from(repo_url, str(clone_dir))
        return clone_dir
    except GitCommandError as e:
        raise RuntimeError(f"git clone failed: {e}")
    except Exception as e:
        raise RuntimeError(f"git clone failed: {e}")

def collect_readme(repo_path: Path) -> str:
    if not repo_path:
        return ""
    for name in ("README.md", "README.rst", "README.MD", "readme.md", "Readme.md"):
        p = repo_path / name
        if p.exists():
            return safe_read_text(p, max_chars=50_000)
    # search a little deeper
    for f in repo_path.rglob("README*"):
        if f.is_file():
            return safe_read_text(f, max_chars=50_000)
    return ""

# -------------------------
# Static repo analysis
# -------------------------
SOURCE_EXTENSIONS = [".py", ".java", ".cpp", ".c", ".js", ".ts", ".rs", ".go", ".rb", ".html", ".css", ".xml", ".json", ".ipynb"]

def get_codebase_size(repo_path: Path) -> Dict[str, int]:
    total_files = 0
    total_lines = 0
    if not repo_path:
        return {"total_files": "NF", "total_lines": "NF"}
    for f in repo_path.rglob("*"):
        if f.is_file() and any(f.name.endswith(ext) for ext in SOURCE_EXTENSIONS):
            total_files += 1
            try:
                with f.open("r", errors="ignore", encoding="utf-8") as fh:
                    total_lines += sum(1 for _ in fh)
            except Exception:
                continue
    return {"total_files": total_files, "total_lines": total_lines}

def detect_programming_languages(repo_path: Path) -> List[str]:
    if not repo_path:
        return ["NF"]
    exts = Counter()
    for f in repo_path.rglob("*"):
        if f.is_file():
            exts[f.suffix.lower()] += 1
    mapping = {
        ".py": "Python", ".java": "Java", ".cpp": "C++", ".c": "C", ".js": "JavaScript",
        ".ts": "TypeScript", ".rs": "Rust", ".go": "Go", ".rb": "Ruby", ".html": "HTML/CSS",
        ".ipynb": "Jupyter"
    }
    langs = []
    for ext, cnt in exts.most_common(10):
        if ext in mapping:
            langs.append(mapping[ext])
    return langs or ["NF"]

def detect_container_support(repo_path: Path) -> str:
    if not repo_path:
        return "NF"
    candidate_files = ["Dockerfile", "docker-compose.yml", "docker-compose.yaml"]
    for cf in candidate_files:
        if (repo_path / cf).exists():
            return "Yes"
    for d in ("k8s", "helm", "charts"):
        if (repo_path / d).exists():
            return "Yes"
    return "No"

def detect_ci_cd(repo_path: Path) -> Dict[str, Any]:
    result = {"ci_present": False, "ci_providers": []}
    if not repo_path:
        return result
    # GitHub Actions
    gha = repo_path / ".github" / "workflows"
    if gha.exists():
        result["ci_present"] = True
        result["ci_providers"].append("GitHub Actions")
        wf = [p.name for p in gha.glob("*.y*ml")]
        result["workflows"] = wf
    # Gitlab-CI
    if (repo_path / ".gitlab-ci.yml").exists():
        result["ci_present"] = True
        result["ci_providers"].append("GitLab CI")
    # travis
    if (repo_path / ".travis.yml").exists():
        result["ci_present"] = True
        result["ci_providers"].append("Travis CI")
    return result

def detect_tests(repo_path: Path) -> bool:
    if not repo_path:
        return False
    for d in ("tests", "test"):
        p = repo_path / d
        if p.exists() and any(p.iterdir()):
            return True
    # search for unittest/pytest keywords
    for f in repo_path.rglob("*.py"):
        try:
            t = f.read_text(encoding="utf-8", errors="ignore").lower()
            if "pytest" in t or "unittest" in t or "nose" in t:
                return True
        except Exception:
            continue
    return False

def find_contributing_and_code_of_conduct(repo_path: Path) -> Tuple[bool, bool]:
    if not repo_path:
        return False, False
    has_contrib = (repo_path / "CONTRIBUTING.md").exists() or (repo_path / "CONTRIBUTING.rst").exists()
    has_coc = (repo_path / "CODE_OF_CONDUCT.md").exists() or (repo_path / "CODE_OF_CONDUCT.rst").exists()
    return has_contrib, has_coc

def detect_benchmarks(repo_path: Path, readme_text: str) -> bool:
    if not repo_path and not readme_text:
        return False
    if repo_path and (repo_path / "benchmarks").exists():
        return True
    keywords = ["benchmark", "benchmarks", "performance comparison", "evaluation"]
    if any(k in readme_text.lower() for k in keywords):
        return True
    return False

def detect_simulation_tools(readme_text: str) -> List[str]:
    tools = []
    known = ["psse", "opendss", "pandapower", "pyomo", "pypower", "gridlab-d", "matpower", "digsilent", "openems"]
    for k in known:
        if k.lower() in readme_text.lower():
            tools.append(k)
    return tools

# -------------------------
# GitHub API derived features
# -------------------------
def get_github_metadata(repo_url: str, gh_client: Any) -> Dict[str, Any]:
    # Assumes gh_client is a valid PyGithub Github instance
    repo_name = repo_fullname_from_url(repo_url)
    try:
        repo = gh_client.get_repo(repo_name)
    except Exception as e:
        raise RuntimeError(f"Failed to get repo metadata via GitHub API: {e}")

    # try to fetch commits (capped)
    commits = []
    try:
        commits = list(repo.get_commits()[:500])
    except Exception:
        # silent fallback
        pass

    # releases, contributors, issues
    try:
        releases = repo.get_releases().totalCount
    except Exception:
        releases = "NF"
    try:
        contributors_count = repo.get_contributors().totalCount
    except Exception:
        contributors_count = "NF"
    try:
        issues = repo.get_issues(state="all").totalCount
    except Exception:
        issues = "NF"

    license_type = "NF"
    try:
        if getattr(repo, "license", None):
            # safe obtain license id
            try:
                license_type = repo.get_license().license.spdx_id if repo.get_license() and repo.get_license().license else repo.license.spdx_id if repo.license else "NF"
            except Exception:
                license_type = getattr(repo.license, "spdx_id", "NF")
    except Exception:
        license_type = "NF"

    # first & last commit dates (if commits retrieved)
    if commits:
        try:
            first_update = commits[-1].commit.author.date.isoformat()
            last_update = commits[0].commit.author.date.isoformat()
        except Exception:
            first_update = "NF"
            last_update = "NF"
    else:
        first_update = "NF"
        last_update = "NF"

    stars = getattr(repo, "stargazers_count", "NF")
    forks = getattr(repo, "forks_count", "NF")
    watchers = getattr(repo, "subscribers_count", "NF")

    # releases frequency (approx)
    try:
        rels = list(repo.get_releases()[:50])
        if len(rels) >= 2:
            dates = sorted([r.created_at for r in rels])
            diffs = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
            rel_freq_days = sum(diffs)/len(diffs)
        else:
            rel_freq_days = "NF"
    except Exception:
        rel_freq_days = "NF"

    code_review_cov = get_code_review_coverage(repo)
    bus_factor = get_bus_factor(repo)
    active_contribs, community_score = get_active_contributors_and_community_score(repo)

    gh_meta = {
        "github repo": repo_name,
        "First update date": first_update,
        "last update": last_update,
        "Releases": releases,
        "Active Contributors": contributors_count,
        "Open/closed issues": issues,
        "Fork Count": forks,
        "GitHub Stars/Forks": f"{stars} stars / {forks} forks",
        "License Type": license_type,
        "release_frequency_days_approx": rel_freq_days,
        "stars": stars,
        "forks": forks,
        "watchers": watchers,
        "Code Review Coverage": code_review_cov,
        "Bus Factor": bus_factor,
        "Active Contributors (recent)": active_contribs,
        "User Community Size Score": community_score,
    }
    return gh_meta

def get_code_review_coverage(repo) -> str:
    try:
        pulls = repo.get_pulls(state="all")
        total_prs = 0
        reviewed_prs = 0
        for pr in pulls[:200]:
            total_prs += 1
            try:
                if pr.get_reviews().totalCount > 0:
                    reviewed_prs += 1
            except Exception:
                continue
        if total_prs == 0:
            return "NF"
        coverage = (reviewed_prs / total_prs) * 100
        return f"{coverage:.1f}%"
    except Exception:
        return "NF"

def get_bus_factor(repo) -> Any:
    try:
        contributors = list(repo.get_contributors()[:500])
        if not contributors:
            return "NF"
        total_commits = sum(getattr(c, "contributions", 0) for c in contributors)
        contributors_sorted = sorted(contributors, key=lambda c: getattr(c, "contributions", 0), reverse=True)
        cumulative = 0
        count = 0
        for c in contributors_sorted:
            cumulative += getattr(c, "contributions", 0)
            count += 1
            if cumulative >= 0.75 * total_commits:
                break
        return count
    except Exception:
        return "NF"

def get_active_contributors_and_community_score(repo, months=6) -> Tuple[Any, Any]:
    try:
        cutoff = datetime.utcnow() - timedelta(days=30*months)
        commits = repo.get_commits(since=cutoff)
        active_set = set()
        for c in commits:
            try:
                if c.author:
                    active_set.add(c.author.login)
            except Exception:
                continue
        active_count = len(active_set)
        stars = getattr(repo, "stargazers_count", 0) or 0
        forks = getattr(repo, "forks_count", 0) or 0
        try:
            total_contribs = repo.get_contributors().totalCount
        except Exception:
            total_contribs = 0
        score = 1
        if stars > 500 or forks > 200 or total_contribs > 200:
            score = 5
        elif stars > 200 or forks > 100 or total_contribs > 100:
            score = 4
        elif stars > 100 or forks > 50 or total_contribs > 50:
            score = 3
        elif stars > 50 or forks > 20 or total_contribs > 20:
            score = 2
        return active_count, score
    except Exception:
        return "NF", "NF"

# -------------------------
# Issues / resolution time
# -------------------------
def get_avg_issue_resolution_time(repo, max_issues=100) -> Any:
    try:
        closed_issues = repo.get_issues(state="closed")
        durations = []
        for issue in closed_issues[:max_issues]:
            if getattr(issue, "created_at", None) and getattr(issue, "closed_at", None):
                durations.append((issue.closed_at - issue.created_at).total_seconds())
        if durations:
            avg_days = sum(durations)/(len(durations)*86400)
            return round(avg_days, 2)
        return "NF"
    except Exception:
        return "NF"

# -------------------------
# PR review metrics
# -------------------------
def get_pr_review_metrics(repo, max_prs=200) -> Dict[str, Any]:
    try:
        total_reviews = 0
        positive_reviews = 0
        negative_reviews = 0
        count_prs = 0
        for pr in repo.get_pulls(state="all")[:max_prs]:
            count_prs += 1
            reviews = pr.get_reviews()
            total_reviews += reviews.totalCount
            for r in reviews:
                if getattr(r, "state", "").upper() == "APPROVED":
                    positive_reviews += 1
                elif getattr(r, "state", "").upper() == "CHANGES_REQUESTED":
                    negative_reviews += 1
        if count_prs == 0:
            return {"Total Reviews": "NF", "Positive Reviews": "NF", "Negative Reviews": "NF"}
        return {
            "Total Reviews": total_reviews,
            "Positive Reviews": positive_reviews,
            "Negative Reviews": negative_reviews
        }
    except Exception:
        return {"Total Reviews": "NF", "Positive Reviews": "NF", "Negative Reviews": "NF"}

# -------------------------
# Semantic / keyword detectors
# -------------------------
def keyword_search_in_texts(patterns: List[str], texts: List[str]) -> Dict[str, bool]:
    found = {}
    lower_join = "\n".join(t.lower() for t in texts)
    for p in patterns:
        found[p] = (p.lower() in lower_join)
    return found

def detect_hardware_interfacing(repo_path: Path, readme_text: str) -> Dict[str, Any]:
    keywords = ["hardware-in-the-loop", "hil", "modbus", "opc-ua", "gpio", "pyserial", "serial", "can bus", "iec61850", "dnp3", "ethernet"]
    present = [k for k in keywords if k in readme_text.lower()]
    libs = []
    if repo_path:
        for f in repo_path.rglob("*.py"):
            try:
                t = f.read_text(encoding="utf-8", errors="ignore").lower()
                if "pyserial" in t:
                    libs.append("pyserial")
                if "pymodbus" in t or "modbus" in t:
                    libs.append("modbus")
                if "opcua" in t or "opc_ua" in t:
                    libs.append("opcua")
            except Exception:
                continue
    return {"keywords_found": present, "libs_detected": list(set(libs)), "hardware_interface_present": bool(present or libs)}

def detect_voltage_freq_capability(readme_text: str, code_sample: Optional[str] = None) -> Dict[str, Any]:
    keywords = ["voltage stability", "frequency stability", "transient", "small-signal", "load flow", "dynamic simulation", "power flow", "swing equation"]
    present = [k for k in keywords if k in readme_text.lower()]
    code_indicators = False
    if code_sample:
        code_indicators = any(k in code_sample.lower() for k in ["powerflow", "pflow", "runpf", "swing", "transient", "frequency"])
    return {"voltage_frequency_terms": present, "code_indicators": code_indicators, "voltage_frequency_support": bool(present or code_indicators)}

def detect_fault_tolerance(repo_path: Path, readme_text: str) -> Dict[str, Any]:
    try_blocks = 0
    files_with_try = 0
    total_files = 0
    retry_patterns = ["retry", "backoff", "checkpoint", "failover", "graceful", "redundant", "watchdog"]
    if repo_path:
        for f in repo_path.rglob("*.py"):
            total_files += 1
            try:
                t = f.read_text(encoding="utf-8", errors="ignore")
                if "try:" in t:
                    try_blocks += t.count("try:")
                    files_with_try += 1
            except Exception:
                continue
    readme_hits = [k for k in retry_patterns if k in readme_text.lower()]
    score = 1
    if files_with_try > 10 or len(readme_hits) >= 2:
        score = 4
    elif files_with_try > 3 or len(readme_hits) >= 1:
        score = 3
    return {"try_blocks": try_blocks, "files_with_try": files_with_try, "readme_fault_keywords": readme_hits, "fault_tolerance_score_1_5": score}

def detect_api_integration(repo_path: Path, readme_text: str) -> Dict[str, Any]:
    doc_api = False
    code_api = False
    api_terms = ["rest api", "openapi", "swagger", "flask", "fastapi", "django", "@app.route", "flask.run", "uvicorn", "GET", "POST", "/api/"]
    for k in api_terms:
        if k.lower() in readme_text.lower():
            doc_api = True
            break
    if repo_path:
        for f in repo_path.rglob("*"):
            if f.suffix in (".py", ".js", ".ts"):
                try:
                    t = f.read_text(encoding="utf-8", errors="ignore").lower()
                    if any(term.lower() in t for term in api_terms):
                        code_api = True
                        break
                except Exception:
                    continue
    return {"api_doc_detected": doc_api, "api_code_detected": code_api, "api_present": doc_api or code_api}

def detect_standards_interoperability(readme_text: str, repo_path: Path) -> Dict[str, Any]:
    standards = ["iec 61850", "iec 61970", "iec 62325", "ieee 1547", "cim", "opendss", "modbus", "dnp3", "iec 61850-7-3"]
    found = [s for s in standards if s in readme_text.lower()]
    return {"standards_found": found, "standards_support": bool(found)}

def detect_vendor_diversity(readme_text: str) -> Dict[str, Any]:
    vendors = ["siemens", "schneider", "abb", "schneider electric", "GE", "hitachi", "mitsubishi"]
    found = [v for v in vendors if v.lower() in readme_text.lower()]
    return {"vendor_mentions": found, "vendor_diversity": len(set(found)) >= 2}

def detect_deployment_modes(readme_text: str, repo_path: Path) -> List[str]:
    modes = []
    if "web" in readme_text.lower() or any(f.suffix in (".html", ".js") for f in (repo_path or Path('.')).rglob("*")):
        modes.append("web")
    if detect_container_support(repo_path) == "Yes":
        modes.append("container")
    if any(f.name.lower().endswith((".exe", ".msi")) for f in (repo_path or Path('.')).rglob("*")):
        modes.append("desktop")
    if "embedded" in readme_text.lower() or "microcontroller" in readme_text.lower():
        modes.append("embedded")
    return modes or ["NF"]

def detect_validation_availability(readme_text: str) -> Dict[str, Any]:
    keywords = ["validation", "validated", "benchmark", "tested against", "evaluated against"]
    found = [k for k in keywords if k in readme_text.lower()]
    return {"validation_terms_found": found, "validation_available": bool(found)}

def detect_real_world_use_cases(readme_text: str) -> Dict[str, Any]:
    keywords = ["production", "utility", "used by", "deployed in", "industry", "contract", "pilot"]
    found = [k for k in keywords if k in readme_text.lower()]
    return {"real_world_use_cases_keywords": found, "real_world_use_cases": bool(found)}

# -------------------------
# Social / external mentions (optional placeholders)
# -------------------------
def aggregate_social_mentions(project_name: str, reddit_creds=None, youtube_api_key=None) -> Dict[str, Any]:
    """
    Placeholder for social scraping. Returns NF unless APIs enabled.
    """
    return {"Twitter Mentions": "NF", "Reddit Mentions": "NF", "YouTube Mentions": "NF", "Total Social Media Mentions": "NF"}

# -------------------------
# LLM assisted extraction (optional) - returns NF unless configured
# -------------------------
def llm_extract_semantic_features(readme: str, code_samples: str, llm_api_key: Optional[str]=None) -> Dict[str, Any]:
    if not llm_api_key:
        return {"LLM_used": False, "semantic_features": "NF"}
    return {"LLM_used": False, "semantic_features": "NF (LLM available but not implemented in this build)"}

# -------------------------
# Putting it all together
# -------------------------
ESSENTIAL_FEATURES = [
    "Title", "Link", "github repo", "Creator", "Creator specific", "License Type", "Fork Count",
    "GitHub Stars/Forks", "First update date", "last update", "Releases", "Active Contributors",
    "Open/closed issues", "Detailed description", "Programming Language used",
    "has_contributing", "has_code_of_conduct", "has_tests", "module_count", "platforms",
    "dependency_count", "Third-party Integrations", "Downloads / Installs Count",
    "User Community Size", "DER Type", "Version Release Frequency", "Vendor Diversity", "SCADA/EMS Presence",
    "Data Analytics & Forecasting Tools", "ROI for OSS", "Integration Cost", "Vendor Lock-in Avoidance", "Customer Diversity"
]

def analyze_repo(repo_url: str, run_clone=True, enable_llm=False, llm_api_key=None, github_token: Optional[str]=None, output_dir: Path = DEFAULT_OUTPUT_DIR):
    # Setup GitHub client
    gh = None
    if github_token:
        try:
            gh = github_client(github_token)
        except Exception as e:
            raise RuntimeError(f"GitHub client initialization failed: {e}")

    repo_name = repo_fullname_from_url(repo_url)
    print(f"[analyze] {repo_name}")
    repo_clone_path = None
    if run_clone:
        repo_clone_path = clone_repo(repo_url)
    readme_text = collect_readme(repo_clone_path) if repo_clone_path else ""

    gh_meta = {}
    if gh:
        try:
            gh_meta = get_github_metadata(repo_url, gh)
        except Exception as e:
            gh_meta = {"github_error": str(e)}

    codebase = get_codebase_size(repo_clone_path) if repo_clone_path else {"total_files": "NF", "total_lines": "NF"}
    languages = detect_programming_languages(repo_clone_path) if repo_clone_path else ["NF"]
    container = detect_container_support(repo_clone_path) if repo_clone_path else "NF"
    ci_info = detect_ci_cd(repo_clone_path) if repo_clone_path else {}
    has_tests = detect_tests(repo_clone_path) if repo_clone_path else False
    has_contrib, has_coc = find_contributing_and_code_of_conduct(repo_clone_path) if repo_clone_path else (False, False)
    bench = detect_benchmarks(repo_clone_path, readme_text)
    sim_tools = detect_simulation_tools(readme_text)
    hw_if = detect_hardware_interfacing(repo_clone_path, readme_text) if repo_clone_path else {"hardware_interface_present": "NF"}
    vfs = detect_voltage_freq_capability(readme_text, code_sample=collect_code_samples(repo_clone_path, max_chars=20000) if repo_clone_path else "")
    fault = detect_fault_tolerance(repo_clone_path, readme_text) if repo_clone_path else {}
    api_info = detect_api_integration(repo_clone_path, readme_text) if repo_clone_path else {}
    standards = detect_standards_interoperability(readme_text, repo_clone_path)
    vendor_div = detect_vendor_diversity(readme_text)
    deployment_modes = detect_deployment_modes(readme_text, repo_clone_path)
    validation = detect_validation_availability(readme_text)
    real_world = detect_real_world_use_cases(readme_text)
    pr_metrics = {}
    issues_resolution = "NF"
    if gh:
        try:
            pr_metrics = get_pr_review_metrics(gh.get_repo(repo_name))
        except Exception:
            pr_metrics = {}
        try:
            issues_resolution = get_avg_issue_resolution_time(gh.get_repo(repo_name))
        except Exception:
            issues_resolution = "NF"
    llm_feats = llm_extract_semantic_features(readme_text, collect_code_samples(repo_clone_path, max_chars=100000) if repo_clone_path else "", llm_api_key) if enable_llm else {"LLM_used": False}
    social = aggregate_social_mentions(repo_name)

    combined = {}
    combined.update(gh_meta)
    combined.update({
        "Link": repo_url,
        "Title": repo_name.split("/")[-1],
        "Detailed description": readme_text[:5000] or "NF",
        "Programming Language used": languages,
        "Codebase Size": codebase,
        "Containerization Support": container,
        "CI/CD Availability": ci_info,
        "has_tests": has_tests,
        "has_contributing": has_contrib,
        "has_code_of_conduct": has_coc,
        "Benchmarks Participation": bench,
        "Simulation Tools": sim_tools,
        "Hardware Interfacing": hw_if,
        "Voltage/Frequency Stability": vfs,
        "Fault Tolerance Mechanism": fault,
        "API Integration": api_info,
        "Standards Interoperability": standards,
        "Vendor Diversity": vendor_div,
        "Deployment Modes": deployment_modes,
        "Validation Availability": validation,
        "Real-World Use Cases": real_world,
        "PR Review Metrics": pr_metrics,
        "Issue Resolution Time (days avg)": issues_resolution,
        "LLM_features": llm_feats,
        "Social Mentions": social
    })

    # Additional heuristic features
    combined["module_count"] = count_python_modules(repo_clone_path) if repo_clone_path else "NF"
    combined["Programming Language Summary"] = ", ".join(languages) if isinstance(languages, list) else languages
    combined["User Community Size"] = combined.get("User Community Size Score", "NF")
    combined["Third-party Integrations"] = detect_third_party_integrations(repo_clone_path)
    combined["Downloads / Installs Count"] = detect_downloads_metric(repo_url, repo_clone_path)

    # Save per-repo outputs
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_json = output_dir / f"{repo_name.replace('/', '_')}_{timestamp}.json"
    out_csv = output_dir / f"{repo_name.replace('/', '_')}_{timestamp}.csv"
    try:
        with out_json.open("w", encoding="utf-8") as fh:
            json.dump(combined, fh, indent=2, default=str)
        # flatten
        flat = {k: (v if isinstance(v, (str, int, float)) else json.dumps(v) ) for k, v in combined.items()}
        if pd:
            pd.DataFrame([flat]).to_csv(out_csv, index=False)
        else:
            # fallback CSV writer
            with out_csv.open("w", newline='', encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(flat.keys()))
                writer.writeheader()
                writer.writerow(flat)
    except Exception as e:
        print(f"[warning] failed to save outputs: {e}")

    print(f"Saved outputs: {out_json} , {out_csv}")
    return combined, out_json, out_csv

# -------------------------
# Helper small utilities used above
# -------------------------
def collect_code_samples(repo_path: Path, max_chars=20000) -> str:
    texts = []
    count = 0
    if not repo_path:
        return ""
    for f in repo_path.rglob("*"):
        if f.is_file() and any(str(f).endswith(ext) for ext in SOURCE_EXTENSIONS):
            try:
                t = f.read_text(encoding="utf-8", errors="ignore")
                texts.append(t)
                count += len(t)
                if count > max_chars:
                    break
            except Exception:
                continue
    return "\n\n".join(texts)[:max_chars]

def count_python_modules(repo_path: Path) -> int:
    if not repo_path:
        return "NF"
    cnt = 0
    try:
        for d in [p for p in repo_path.iterdir() if p.is_dir()]:
            if (d / "__init__.py").exists() or any(str(f).endswith(".py") for f in d.glob("*.py")):
                cnt += 1
    except Exception:
        return "NF"
    return cnt

def detect_third_party_integrations(repo_path: Path) -> List[str]:
    integrations = []
    if not repo_path:
        return ["NF"]
    if (repo_path / "requirements.txt").exists():
        text = safe_read_text(repo_path / "requirements.txt")
        integrations += [line.split("==")[0] for line in text.splitlines() if line and not line.strip().startswith("#")]
    if (repo_path / "pyproject.toml").exists():
        text = safe_read_text(repo_path / "pyproject.toml")
        integrations += re.findall(r'["\']([a-zA-Z0-9_\-]+)["\']', text)
    if (repo_path / "package.json").exists():
        text = safe_read_text(repo_path / "package.json")
        try:
            pj = json.loads(text)
            deps = pj.get("dependencies", {}).keys()
            integrations += list(deps)
        except Exception:
            pass
    return list(set(integrations))[:200] if integrations else ["NF"]

def detect_downloads_metric(repo_url: str, repo_path: Path) -> Any:
    try:
        token = os.environ.get("GITHUB_TOKEN")
        if not token:
            return "NF"
        gh = github_client(token)
        repo = gh.get_repo(repo_fullname_from_url(repo_url))
        total_downloads = 0
        for r in repo.get_releases()[:50]:
            for a in r.get_assets():
                total_downloads += getattr(a, "download_count", 0) or 0
        if total_downloads > 0:
            return total_downloads
    except Exception:
        pass
    return "NF"

# -------------------------
# CLI / Streamlit glue
# -------------------------
def run_cli_mode(args):
    token = args.github_token or os.environ.get("GITHUB_TOKEN")
    if not token:
        print("ERROR: GITHUB_TOKEN missing. Set env var GITHUB_TOKEN or pass --github-token")
        sys.exit(1)
    combined, jpath, cpath = analyze_repo(args.repo, run_clone=(not args.no_clone), enable_llm=args.enable_llm, llm_api_key=args.llm_api_key, github_token=token)
    print(json.dumps(combined, indent=2, default=str)[:10000])

# Streamlit UI
def run_streamlit_app():
    try:
        import streamlit as st
    except Exception:
        print("Streamlit not installed. Install with `pip install streamlit`")
        sys.exit(1)

    st.set_page_config(page_title="RECOPS Feature Extractor", layout="wide")
    st.title("RECOPS Feature Extractor (Streamlit)")
    st.write("Enter a GitHub repo URL and click Run. Outputs (JSON + CSV) are saved to ./outputs/")

    col1, col2 = st.columns([3,1])
    with col1:
        repo_url = st.text_input("GitHub repo URL", value="https://github.com/OpenEMS/openems")
        run_clone = st.checkbox("Clone repository locally (recommended)", value=True)
        enable_llm = st.checkbox("Enable LLM-assisted extraction (requires API key & libs)", value=False)
        github_token_input = st.text_input("GITHUB_TOKEN (or set env GITHUB_TOKEN)", value=os.environ.get("GITHUB_TOKEN",""), type="password")
        llm_api_key = st.text_input("LLM API key (optional)", value=os.environ.get("GROQ_API_KEY",""), type="password")
    with col2:
        st.write("Options")
        no_clone = st.checkbox("Skip cloning (fast, but limited metadata)", value=False)
        output_dir = st.text_input("Outputs directory", value=str(DEFAULT_OUTPUT_DIR))
        run_button = st.button("Run Analysis")

    if not github_token_input:
        st.warning("GITHUB_TOKEN not set. Some features requiring GitHub API will be unavailable. Provide token for full metadata.")

    if run_button:
        # minimal validation
        if not repo_url or "github.com" not in repo_url:
            st.error("Please enter a valid GitHub repo URL (https://github.com/owner/repo)")
        else:
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            # Run analysis
            with st.spinner("Running analysis (this may take a while for large repos)..."):
                try:
                    combined, jpath, cpath = analyze_repo(repo_url, run_clone=(not no_clone), enable_llm=enable_llm, llm_api_key=llm_api_key or None, github_token=github_token_input or None, output_dir=out_dir)
                except Exception as e:
                    st.exception(e)
                    return
            st.success("Analysis complete")
            st.subheader("Summary")
            try:
                # show a few key fields
                st.write("Repo:", combined.get("github repo", "NF"))
                st.write("Stars:", combined.get("stars", "NF"), "Forks:", combined.get("forks", "NF"))
                st.write("Languages:", combined.get("Programming Language used", "NF"))
                st.write("Containerization:", combined.get("Containerization Support", "NF"))
                st.write("Has tests:", combined.get("has_tests", False))
            except Exception:
                pass

            st.subheader("Outputs")
            st.write(f"Saved JSON: `{jpath}`")
            st.write(f"Saved CSV: `{cpath}`")
            try:
                st.download_button("Download JSON", data=open(jpath, "rb").read(), file_name=jpath.name, mime="application/json")
                st.download_button("Download CSV", data=open(cpath, "rb").read(), file_name=cpath.name, mime="text/csv")
            except Exception:
                st.warning("Could not prepare download buttons (files may be large). Files are saved in outputs directory.")

# -------------------------
# Entry point
# -------------------------
if __name__ == "__main__":
    # Determine if running under Streamlit
    running_under_streamlit = any("streamlit" in p for p in sys.argv) or ("STREAMLIT_RUN" in os.environ) or ("STREAMLIT_SERVER_RUNNING" in os.environ)

    # A more robust detection is to check for STREAMLIT_SERVER_RUN or importability, but simpler:
    try:
        import streamlit as _st  # type: ignore
        streamlit_available = True
    except Exception:
        streamlit_available = False

    # If streamlit run was used, prefer Streamlit UI
    if streamlit_available and ("streamlit" in sys.argv[0] or "streamlit" in " ".join(sys.argv) or os.environ.get("STREAMLIT") or os.environ.get("STREAMLIT_SERVER_RUNNING")):
        run_streamlit_app()
    else:
        # CLI path
        import argparse
        parser = argparse.ArgumentParser(description="RECOPS comprehensive repo feature extractor")
        parser.add_argument("--repo", type=str, required=True, help="GitHub repo URL (https.../owner/repo)")
        parser.add_argument("--no-clone", action="store_true", help="Skip cloning the repo (only GitHub metadata will be fetched)")
        parser.add_argument("--enable-llm", action="store_true", help="Enable LLM-assisted semantic extraction if API key provided")
        parser.add_argument("--llm-api-key", type=str, default=os.environ.get("GROQ_API_KEY"), help="LLM API key (optional)")
        parser.add_argument("--github-token", type=str, default=os.environ.get("GITHUB_TOKEN"), help="GitHub token (or set GITHUB_TOKEN env)")
        args = parser.parse_args()
        run_cli_mode(args)
