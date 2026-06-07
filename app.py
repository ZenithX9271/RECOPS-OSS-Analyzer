#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RECOPS Scorecard Extractor (Streamlit Cloud + simple password gate)
====================================================================

Scores power-system open-source repositories on the six RECOPS families /
27 sub-indicators, with an interactive LLM panel.

Access is protected by a single shared username + password defined in this
file. After signing in, the rest of the app is identical to the local
version. API keys are read from Streamlit secrets / environment variables
and are hidden from the UI so users cannot read them out of the sidebar.

DEPLOY (Streamlit Community Cloud)
----------------------------------
1. Push this file plus the six recops_repo_scores_*.py modules and
   requirements.txt to a GitHub repo.
2. On https://share.streamlit.io connect the repo and deploy.
3. In App settings -> Secrets, set ONLY the two API keys:

       GITHUB_TOKEN  = "ghp_..."
       GROQ_API_KEY  = "gsk_..."

4. Share the URL. Visitors must enter the username and password below to
   reach the app.
"""

from __future__ import annotations

import os
import sys
import io
import re
import json
import hmac
import shutil
import stat
import tempfile
import importlib
import importlib.util
import subprocess
import traceback
import concurrent.futures
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# =========================================================================== #
#                  SHARED ACCESS CREDENTIALS  (fixed, in-file)                #
#   Anyone who reads app.py can read these values. If you ever push the repo  #
#   to a public URL, treat the password as known to the world and rely on it  #
#   only as a polite gate, not real security.                                 #
# =========================================================================== #
APP_USERNAME = "recops_oss"
APP_PASSWORD = "KTH@recops2527"


# --------------------------------------------------------------------------- #
# API credentials -- read from env var first, then st.secrets.                 #
# Streamlit Cloud exposes secrets as both, so this works in either context.   #
# --------------------------------------------------------------------------- #
def _from_secrets(key: str) -> str:
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return ""


def resolve_github_token() -> str:
    return os.environ.get("GITHUB_TOKEN", "") or _from_secrets("GITHUB_TOKEN")


def resolve_groq_key() -> str:
    return os.environ.get("GROQ_API_KEY", "") or _from_secrets("GROQ_API_KEY")


GROQ_MODELS = ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"]

# --------------------------------------------------------------------------- #
# Locate the six scoring modules robustly, wherever the app is launched from. #
# --------------------------------------------------------------------------- #
_THIS_DIR = Path(__file__).resolve().parent
_MODULE_SEARCH_DIRS = [_THIS_DIR, Path.cwd()]
for _d in _MODULE_SEARCH_DIRS:
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

try:
    import requests
except Exception:  # pragma: no cover
    requests = None  # type: ignore


# =========================================================================== #
#                        RECOPS  SCORING  ENGINE  (core)                      #
# =========================================================================== #
FAMILY_REGISTRY: Dict[str, Dict[str, Any]] = {
    "EQ":  {"module": "recops_repo_scores_EQ",  "fn": "compute_eq",
            "subs": ["CQI", "ARS", "PER", "SPS", "SRSW"], "name": "Engineering Quality"},
    "AE":  {"module": "recops_repo_scores_AE",  "fn": "compute_ae",
            "subs": ["CEI", "EVS", "RWIS", "AQS"], "name": "Adoption & Ecosystem"},
    "GC":  {"module": "recops_repo_scores_GC",  "fn": "compute_gc",
            "subs": ["GOS", "SCS", "ITI", "RVS"], "name": "Governance & Credibility"},
    "DI":  {"module": "recops_repo_scores_DI",  "fn": "compute_di",
            "subs": ["DRS", "ICI", "PFS", "OBS"], "name": "Deployability & Interoperability"},
    "PH":  {"module": "recops_repo_scores_PH",  "fn": "compute_ph",
            "subs": ["PVI", "MQS", "FCS", "MRD", "BUS_FACTOR"], "name": "Project Health"},
    "EDF": {"module": "recops_repo_scores_EDF", "fn": "compute_edf",
            "subs": ["ESCI", "CPRI", "ENSC", "EDVS"], "name": "Energy-Domain Fitness"},
}
ALL_FAMILIES = list(FAMILY_REGISTRY.keys())
_MODULE_CACHE: Dict[str, Any] = {}


def _find_module_file(stem: str) -> Optional[Path]:
    candidates: List[Path] = []
    for d in _MODULE_SEARCH_DIRS:
        candidates.append(d / f"{stem}.py")
    for d in _MODULE_SEARCH_DIRS:
        candidates += list(d.glob(f"*/{stem}.py"))
    for c in candidates:
        if c.exists():
            return c
    return None


def _get_module(family: str):
    stem = FAMILY_REGISTRY[family]["module"]
    if stem in _MODULE_CACHE:
        return _MODULE_CACHE[stem]
    try:
        mod = importlib.import_module(stem)
    except Exception:
        path = _find_module_file(stem)
        if not path:
            looked = ", ".join(str(d) for d in _MODULE_SEARCH_DIRS)
            raise ModuleNotFoundError(
                f"Could not find {stem}.py. Put all six recops_repo_scores_*.py "
                f"files in the same folder as this app. Looked in: {looked}")
        spec = importlib.util.spec_from_file_location(stem, path)
        mod = importlib.util.module_from_spec(spec)  # type: ignore
        sys.modules[stem] = mod
        spec.loader.exec_module(mod)  # type: ignore
    _MODULE_CACHE[stem] = mod
    return mod


def verify_modules() -> Dict[str, Optional[str]]:
    status: Dict[str, Optional[str]] = {}
    for fam in ALL_FAMILIES:
        try:
            _get_module(fam)
            status[fam] = None
        except Exception as e:
            status[fam] = str(e)
    return status


# --------------------------------------------------------------------------- #
# Repo URL helpers                                                            #
# --------------------------------------------------------------------------- #
_GH_RX = re.compile(r"github\.com[/:]+([^/]+)/([^/#?]+)", re.I)


def parse_owner_repo(url: str) -> Tuple[Optional[str], Optional[str]]:
    if not url:
        return None, None
    m = _GH_RX.search(url.strip())
    if not m:
        return None, None
    owner, repo = m.group(1), m.group(2)
    if repo.endswith(".git"):
        repo = repo[:-4]
    return owner, repo


def normalize_repo_url(url: str) -> str:
    owner, repo = parse_owner_repo(url)
    if owner and repo:
        return f"https://github.com/{owner}/{repo}"
    return url.strip()


# --------------------------------------------------------------------------- #
# Filesystem cleanup (Windows-safe)                                           #
# --------------------------------------------------------------------------- #
def _handle_remove_readonly(func, path, _exc_info):
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception:
        pass


def cleanup_path(path: Optional[Path]) -> None:
    if not path:
        return
    try:
        p = Path(path)
        target = p
        if p.name == "repo" and p.parent.name.startswith("recops_clone_"):
            target = p.parent
        if target.exists():
            shutil.rmtree(target, onerror=_handle_remove_readonly)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Robust single clone (full history + tags) reused by every family            #
# --------------------------------------------------------------------------- #
def clone_repo(url: str, timeout_s: int = 900) -> Path:
    url = normalize_repo_url(url)
    temp = Path(tempfile.mkdtemp(prefix="recops_clone_"))
    target = temp / "repo"

    git_ok = False
    try:
        subprocess.run(["git", "--version"], check=True, capture_output=True)
        git_ok = True
    except Exception:
        git_ok = False

    if git_ok:
        for extra in (["--filter=blob:none", "--tags"], ["--depth", "1"]):
            try:
                proc = subprocess.run(
                    ["git", "clone", *extra, url, str(target)],
                    cwd=str(temp), capture_output=True, timeout=timeout_s)
                if proc.returncode == 0 and target.exists():
                    return target
            except Exception:
                continue

    if requests is None:
        raise RuntimeError("Need git or the requests package to fetch a repository.")

    owner, repo = parse_owner_repo(url)
    if not owner or not repo:
        raise RuntimeError(f"Cannot parse owner/repo from URL: {url}")

    for branch in ("main", "master"):
        zip_url = f"https://github.com/{owner}/{repo}/archive/refs/heads/{branch}.zip"
        try:
            r = requests.get(zip_url, timeout=120)
            if r.status_code == 200:
                zpath = temp / "repo.zip"
                zpath.write_bytes(r.content)
                shutil.unpack_archive(str(zpath), str(temp))
                dirs = [p for p in temp.iterdir() if p.is_dir() and p.name != "repo"]
                if dirs:
                    return dirs[0]
        except Exception:
            continue

    raise RuntimeError(f"Failed to clone or download repository: {url}")


# --------------------------------------------------------------------------- #
# GitHub metadata with anonymous fallback when token is bad                   #
# --------------------------------------------------------------------------- #
def _gh_get(owner: str, repo: str, token: Optional[str]):
    base = f"https://api.github.com/repos/{owner}/{repo}"
    attempts = []
    if token:
        attempts.append({"Accept": "application/vnd.github+json",
                         "Authorization": f"Bearer {token}"})
    attempts.append({"Accept": "application/vnd.github+json"})
    last = None
    for headers in attempts:
        try:
            r = requests.get(base, headers=headers, timeout=30)
        except Exception as e:
            last = ("error", str(e)); continue
        if r.status_code == 200:
            authed = "Authorization" in headers
            return r, (None if (authed or len(attempts) == 1)
                       else "token_invalid_used_anonymous")
        last = ("status", r.status_code)
    return None, (f"github_api_{last[0]}={last[1]}" if last else "github_api_failed")


def github_metadata(url: str, token: Optional[str]) -> Dict[str, Any]:
    owner, repo = parse_owner_repo(url)
    out: Dict[str, Any] = {"github_repo": f"{owner}/{repo}" if owner else url}
    if requests is None or not owner or not repo:
        return out
    r, note = _gh_get(owner, repo, token)
    if note:
        out["github_api_note"] = note
    if r is not None and r.status_code == 200:
        d = r.json()
        lic = (d.get("license") or {})
        out.update({
            "stars": d.get("stargazers_count"),
            "forks": d.get("forks_count"),
            "open_issues": d.get("open_issues_count"),
            "watchers": d.get("subscribers_count"),
            "license": lic.get("spdx_id") or lic.get("name") or "NF",
            "language": d.get("language"),
            "archived": d.get("archived"),
            "pushed_at": d.get("pushed_at"),
            "created_at": d.get("created_at"),
            "default_branch": d.get("default_branch"),
            "description": d.get("description"),
        })
    return out


def github_token_status(token: Optional[str]) -> Tuple[bool, str]:
    if not token:
        return False, "No token set. Anonymous calls work but with a 60/hr rate limit."
    if requests is None:
        return False, "The requests package is not installed."
    try:
        r = requests.get("https://api.github.com/user",
                         headers={"Authorization": f"Bearer {token}"}, timeout=20)
    except Exception as e:
        return False, f"Network error: {e}"
    if r.status_code == 200:
        login = r.json().get("login", "?")
        rem = r.headers.get("X-RateLimit-Remaining", "?")
        return True, f"Token valid (user: {login}). API calls remaining: {rem}."
    if r.status_code == 401:
        return False, ("Token is invalid or revoked. Generate a new one at "
                       "github.com -> Settings -> Developer settings -> "
                       "Personal access tokens.")
    return False, f"Unexpected status {r.status_code} from GitHub."


def local_metadata(repo_path: Path) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "has_readme": False, "has_tests": False, "has_contributing": False,
        "has_code_of_conduct": False, "has_license_file": False,
        "module_count": 0, "readme_chars": 0,
    }
    if not repo_path or not Path(repo_path).exists():
        return meta
    root = Path(repo_path)

    for name in ("README.md", "README.rst", "readme.md", "README.txt", "README"):
        p = root / name
        if p.exists():
            meta["has_readme"] = True
            try:
                meta["readme_chars"] = len(p.read_text(encoding="utf-8", errors="ignore"))
            except Exception:
                pass
            break

    meta["has_contributing"] = any((root / n).exists()
                                   for n in ("CONTRIBUTING.md", "CONTRIBUTING.rst"))
    meta["has_code_of_conduct"] = any((root / n).exists()
                                      for n in ("CODE_OF_CONDUCT.md", "CODE_OF_CONDUCT.rst"))
    meta["has_license_file"] = any((root / n).exists()
                                   for n in ("LICENSE", "LICENSE.md", "LICENSE.txt", "COPYING"))
    try:
        meta["module_count"] = len([d for d in os.listdir(root)
                                    if os.path.isdir(root / d) and d != ".git"])
    except Exception:
        pass

    try:
        for dirpath, dirs, files in os.walk(root):
            if ".git" in dirpath:
                continue
            if any(re.match(r"tests?$", d, re.I) for d in dirs) or \
               any(re.match(r"test_.*\.py$", f, re.I) for f in files):
                meta["has_tests"] = True
                break
    except Exception:
        pass
    return meta


# --------------------------------------------------------------------------- #
# Family runners                                                              #
# --------------------------------------------------------------------------- #
def run_family(family: str, repo_path: Path, repo_url: str,
               token: Optional[str], aggregation: str = "geometric") -> Dict[str, Any]:
    mod = _get_module(family)
    fn = getattr(mod, FAMILY_REGISTRY[family]["fn"])
    try:
        if family == "EQ":
            return fn(repo_path, None, aggregation)
        elif family == "PH":
            return fn(repo_path, repo_url, token, aggregation)
        else:
            return fn(repo_path, repo_url, token, None, None, aggregation)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(limit=3)}


def run_all_families(repo_path: Path, repo_url: str, token: Optional[str],
                     families: Optional[List[str]] = None,
                     aggregation: str = "geometric",
                     per_family_timeout_s: int = 600) -> Dict[str, Dict[str, Any]]:
    families = families or ALL_FAMILIES
    results: Dict[str, Dict[str, Any]] = {}
    for fam in families:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(run_family, fam, repo_path, repo_url, token, aggregation)
            try:
                results[fam] = fut.result(timeout=per_family_timeout_s)
            except concurrent.futures.TimeoutError:
                results[fam] = {"error": f"timeout after {per_family_timeout_s}s"}
            except Exception as e:
                results[fam] = {"error": f"{type(e).__name__}: {e}"}
    return results


# --------------------------------------------------------------------------- #
# Flatten                                                                     #
# --------------------------------------------------------------------------- #
def _round(v: Any, n: int = 2) -> Any:
    if isinstance(v, (int, float)):
        return round(float(v), n)
    return v


def flatten_scorecard(repo_url: str,
                      family_results: Dict[str, Dict[str, Any]],
                      gh_meta: Dict[str, Any],
                      loc_meta: Dict[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {"Link": repo_url, "Status": "Success"}
    row["GitHub Repo"] = gh_meta.get("github_repo")
    row["Stars"] = gh_meta.get("stars")
    row["Forks"] = gh_meta.get("forks")
    row["Open Issues"] = gh_meta.get("open_issues")
    row["License"] = gh_meta.get("license")
    row["Primary Language"] = gh_meta.get("language")
    row["Archived"] = gh_meta.get("archived")
    row["Last Push"] = gh_meta.get("pushed_at")
    row["Description"] = gh_meta.get("description")
    row["Has README"] = loc_meta.get("has_readme")
    row["Has Tests"] = loc_meta.get("has_tests")
    row["Has CONTRIBUTING"] = loc_meta.get("has_contributing")
    row["Has Code of Conduct"] = loc_meta.get("has_code_of_conduct")
    row["Module Count"] = loc_meta.get("module_count")

    any_error = []
    for fam in ALL_FAMILIES:
        res = family_results.get(fam)
        if not res:
            continue
        if "error" in res:
            row[f"{fam}"] = None
            row[f"{fam}_band"] = "ERROR"
            any_error.append(f"{fam}:{res['error']}")
            for sub in FAMILY_REGISTRY[fam]["subs"]:
                row[sub] = None
            continue
        details = res.get("details", {}) or {}
        row[f"{fam}"] = _round(res.get("score"))
        row[f"{fam}_band"] = res.get("band")
        for sub in FAMILY_REGISTRY[fam]["subs"]:
            row[sub] = _round(details.get(sub))

    if any_error:
        row["Status"] = "Partial (" + "; ".join(any_error)[:300] + ")"
    return row


# --------------------------------------------------------------------------- #
# LLM helpers                                                                 #
# --------------------------------------------------------------------------- #
def build_llm_context(flat_rows: List[Dict[str, Any]], max_repos: int = 25) -> str:
    keep_meta = ["GitHub Repo", "Stars", "Forks", "License", "Primary Language", "Archived"]
    compact = []
    for row in flat_rows[:max_repos]:
        item: Dict[str, Any] = {"repo": row.get("Link")}
        for k in keep_meta:
            if row.get(k) is not None:
                item[k] = row.get(k)
        for fam in ALL_FAMILIES:
            item[fam] = row.get(fam)
            item[f"{fam}_band"] = row.get(f"{fam}_band")
            for sub in FAMILY_REGISTRY[fam]["subs"]:
                item[sub] = row.get(sub)
        compact.append(item)
    legend = {fam: {"name": FAMILY_REGISTRY[fam]["name"],
                    "subs": FAMILY_REGISTRY[fam]["subs"]} for fam in ALL_FAMILIES}
    return json.dumps({"legend": legend, "scorecard": compact}, default=str)


def _friendly_llm_error(err: Exception) -> str:
    msg = str(err); low = msg.lower()
    if "invalid_api_key" in low or "invalid api key" in low or "401" in low:
        return ("Your Groq API key is invalid or expired. Generate a new one at "
                "console.groq.com (API Keys) and update the GROQ_API_KEY secret.")
    if "rate" in low and "limit" in low:
        return "Groq rate limit hit. Wait a moment and try again, or use a smaller model."
    if "model" in low and ("decommission" in low or "not found" in low or "does not exist" in low):
        return ("That Groq model is unavailable. Pick a different model in the sidebar "
                "(for example llama-3.1-8b-instant).")
    return f"LLM error: {msg}"


def llm_narrative(repo_url: str, flat_row: Dict[str, Any],
                  groq_api_key: Optional[str],
                  model: str = "llama-3.1-8b-instant") -> Optional[str]:
    if not groq_api_key:
        return None
    try:
        from langchain_groq import ChatGroq
        from langchain_core.prompts import ChatPromptTemplate
    except Exception:
        return None
    fam_lines = []
    for fam in ALL_FAMILIES:
        subs = {s: flat_row.get(s) for s in FAMILY_REGISTRY[fam]["subs"]}
        fam_lines.append(f"{fam} ({FAMILY_REGISTRY[fam]['name']}): "
                         f"composite={flat_row.get(fam)} band={flat_row.get(f'{fam}_band')} "
                         f"subs={subs}")
    signal_block = "\n".join(fam_lines)
    meta_block = (f"stars={flat_row.get('Stars')} forks={flat_row.get('Forks')} "
                  f"license={flat_row.get('License')} language={flat_row.get('Primary Language')} "
                  f"archived={flat_row.get('Archived')}")
    try:
        llm = ChatGroq(api_key=groq_api_key, model=model, temperature=0.2)
        prompt = ChatPromptTemplate.from_template(
            "You are summarising an automated RECOPS scorecard for the open-source "
            "power-system repository {url}. All numbers are 0-100 (higher is better) "
            "and were computed deterministically. METADATA: {meta}\n\n"
            "FAMILY SCORES:\n{signals}\n\n"
            "Write 4-6 sentences: strengths, weaknesses, and one or two concrete "
            "recommendations, referencing the actual scores. Do not invent features."
        )
        resp = (prompt | llm).invoke({"url": repo_url, "meta": meta_block,
                                      "signals": signal_block})
        return getattr(resp, "content", str(resp)).strip()
    except Exception as e:
        return f"({_friendly_llm_error(e)})"


def llm_chat_answer(question: str, context_json: str,
                    history: List[Dict[str, str]], groq_api_key: str,
                    model: str = "llama-3.1-8b-instant") -> str:
    try:
        from langchain_groq import ChatGroq
        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
    except Exception as e:
        return f"(LLM unavailable: install langchain-groq. {e})"
    try:
        llm = ChatGroq(api_key=groq_api_key, model=model, temperature=0.3)
        system = SystemMessage(content=(
            "You are a RECOPS analyst assistant. You answer questions about the "
            "repository scorecard JSON below. Scores are 0-100 (higher is better); "
            "each family has sub-indicators (see 'legend'). Be concise and accurate, "
            "reference actual numbers, compare repositories when asked, and never "
            "invent data that is not present. If a value is null it was not "
            "measurable for that repo.\n\nSCORECARD DATA:\n" + context_json))
        msgs: List[Any] = [system]
        for turn in history:
            if turn.get("role") == "user":
                msgs.append(HumanMessage(content=turn["content"]))
            else:
                msgs.append(AIMessage(content=turn["content"]))
        msgs.append(HumanMessage(content=question))
        resp = llm.invoke(msgs)
        return getattr(resp, "content", str(resp)).strip()
    except Exception as e:
        return f"({_friendly_llm_error(e)})"


def groq_key_ok(groq_api_key: str, model: str = "llama-3.1-8b-instant") -> Tuple[bool, str]:
    if not groq_api_key:
        return False, "No key provided."
    try:
        from langchain_groq import ChatGroq
        from langchain_core.messages import HumanMessage
    except Exception as e:
        return False, f"langchain-groq not installed: {e}"
    try:
        llm = ChatGroq(api_key=groq_api_key, model=model, temperature=0.0, max_tokens=5)
        llm.invoke([HumanMessage(content="ping")])
        return True, "Key is valid."
    except Exception as e:
        return False, _friendly_llm_error(e)


# --------------------------------------------------------------------------- #
# Per-repository orchestration                                                #
# --------------------------------------------------------------------------- #
def process_single_repo(repo_url: str,
                        token: Optional[str] = None,
                        groq_api_key: Optional[str] = None,
                        families: Optional[List[str]] = None,
                        aggregation: str = "geometric",
                        want_narrative: bool = False,
                        model: str = "llama-3.1-8b-instant",
                        per_family_timeout_s: int = 600) -> Dict[str, Any]:
    repo_url = normalize_repo_url(repo_url)
    repo_path: Optional[Path] = None
    try:
        repo_path = clone_repo(repo_url)
        gh_meta = github_metadata(repo_url, token)
        loc_meta = local_metadata(repo_path)
        fam_results = run_all_families(repo_path, repo_url, token,
                                       families=families, aggregation=aggregation,
                                       per_family_timeout_s=per_family_timeout_s)
        flat = flatten_scorecard(repo_url, fam_results, gh_meta, loc_meta)
        narrative = (llm_narrative(repo_url, flat, groq_api_key, model)
                     if want_narrative else None)
        if narrative:
            flat["Summary"] = narrative
        return {"flat": flat,
                "full": {"repo_url": repo_url, "metadata": {**gh_meta, **loc_meta},
                         "families": fam_results},
                "narrative": narrative}
    except Exception as e:
        return {"flat": {"Link": repo_url, "Status": f"Failed: {type(e).__name__}: {e}"},
                "full": {"repo_url": repo_url, "error": traceback.format_exc(limit=4)},
                "narrative": None}
    finally:
        cleanup_path(repo_path)


# =========================================================================== #
#                       SIMPLE USERNAME + PASSWORD GATE                       #
# =========================================================================== #
def _check_credentials(u: str, p: str) -> bool:
    """Constant-time comparison so we don't leak match length via timing."""
    return (hmac.compare_digest(u or "", APP_USERNAME)
            and hmac.compare_digest(p or "", APP_PASSWORD))


def _password_gate() -> None:
    """Block the app until the user submits the correct username + password."""
    import streamlit as st
    ss = st.session_state

    if ss.get("authenticated"):
        return  # already signed in this session

    # Render a small, centred login form.
    st.markdown("# RECOPS Scorecard Extractor")
    st.markdown("This application is private. Please sign in to continue.")
    st.write("")

    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        submitted = st.form_submit_button("Sign in", type="primary")

    if submitted:
        if _check_credentials(username, password):
            ss["authenticated"] = True
            ss["username"] = username
            # Wipe the typed password from session_state for hygiene.
            ss.pop("login_password", None)
            st.rerun()
        else:
            st.error("Invalid username or password.")

    st.stop()


def _render_user_pill() -> None:
    """Show the signed-in user + a Sign-out button in the sidebar."""
    import streamlit as st
    if not st.session_state.get("authenticated"):
        return
    with st.sidebar:
        st.divider()
        st.caption(
            f"Signed in as **{st.session_state.get('username', 'user')}**")
        if st.button("Sign out", use_container_width=True):
            for k in ("authenticated", "username", "login_username", "login_password"):
                st.session_state.pop(k, None)
            st.rerun()


# =========================================================================== #
#                              CLI  ENTRYPOINT                                #
# =========================================================================== #
def _run_cli(argv: List[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser(
        description="RECOPS Scorecard Extractor (headless).")
    ap.add_argument("urls", nargs="+")
    ap.add_argument("--token", default=resolve_github_token())
    ap.add_argument("--groq-key", default=resolve_groq_key())
    ap.add_argument("--families", default="all")
    ap.add_argument("--aggregation", choices=["geometric", "sum"], default="geometric")
    ap.add_argument("--narrative", action="store_true")
    ap.add_argument("--model", default=GROQ_MODELS[0])
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", default=None)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args(argv)

    missing = {f: e for f, e in verify_modules().items() if e}
    if missing:
        print("[RECOPS] Cannot import scoring modules:")
        for f, e in missing.items():
            print(f"   {f}: {e}")
        return 2

    fams = (ALL_FAMILIES if args.families.strip().lower() == "all"
            else [f.strip().upper() for f in args.families.split(",") if f.strip()])

    print(f"[RECOPS] Scoring {len(args.urls)} repo(s) | families={fams} | "
          f"token={'yes' if args.token else 'no'}")

    flat_rows, full_blobs = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futures = {ex.submit(process_single_repo, u, args.token, args.groq_key,
                             fams, args.aggregation, args.narrative, args.model): u
                   for u in args.urls}
        for fut in concurrent.futures.as_completed(futures):
            u = futures[fut]; res = fut.result()
            flat_rows.append(res["flat"]); full_blobs.append(res["full"])
            print(f"  - {u}: {res['flat'].get('Status', '?')}")

    try:
        import pandas as pd
        df = pd.DataFrame(flat_rows)
        for c in df.columns:
            df[c] = df[c].apply(lambda v: str(v) if isinstance(v, (list, dict)) else v)
        if args.out:
            df.to_csv(args.out, index=False); print(f"[RECOPS] CSV -> {args.out}")
        else:
            print(df.to_string(index=False))
    except Exception as e:
        print(f"[RECOPS] (pandas unavailable: {e})")
        print(json.dumps(flat_rows, indent=2, default=str))

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(full_blobs, indent=2, default=str),
                                       encoding="utf-8")
        print(f"[RECOPS] Full JSON -> {args.json_out}")
    return 0


# =========================================================================== #
#                            STREAMLIT  GUI  APP                              #
# =========================================================================== #
def _run_streamlit() -> None:
    import streamlit as st
    import pandas as pd

    st.set_page_config(page_title="RECOPS Scorecard Extractor", layout="wide")

    # ---- GATE FIRST: nothing renders until the user signs in ----
    _password_gate()

    st.markdown("""
        <style>
            .main-header {
                background: -webkit-linear-gradient(45deg, #1e3a8a, #3b82f6);
                -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                font-size: 2.6rem; font-weight: 800; margin-bottom: 0.25rem;
            }
            .sub-header { color: #64748b; font-size: 1.1rem; margin-bottom: 1.0rem; }
            .terminal-box {
                background-color: #0f172a; color: #10b981;
                font-family: 'Courier New', Courier, monospace;
                padding: 15px; border-radius: 8px; height: 260px;
                overflow-y: auto; border: 1px solid #334155;
                box-shadow: inset 0 2px 4px rgba(0,0,0,0.5); white-space: pre-wrap;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-header">RECOPS Scorecard Extractor</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Deterministic 6-family / 27-indicator '
                'scoring for power-system open-source repositories.</div>',
                unsafe_allow_html=True)

    ss = st.session_state
    ss.setdefault("results", None)
    ss.setdefault("chat", [])
    ss.setdefault("model", GROQ_MODELS[0])

    # Module health check
    missing = {f: e for f, e in verify_modules().items() if e}
    if missing:
        st.error("The six RECOPS scoring modules could not be imported. Make sure "
                 "recops_repo_scores_EQ/AE/GC/DI/PH/EDF.py are in the SAME folder "
                 "as this app (or committed alongside it in the repo).")
        with st.expander("Details"):
            for f, e in missing.items():
                st.write(f"- {f}: {e}")

    # ----------------------------- Sidebar ------------------------------ #
    secret_token = resolve_github_token()
    secret_groq = resolve_groq_key()

    with st.sidebar:
        st.header("Configuration")

        # If a secret is configured, use it silently and hide the input.
        # Otherwise show an editable field for local development.
        if secret_token:
            github_token = secret_token
            st.caption("GitHub token: configured from secrets.")
        else:
            github_token = st.text_input(
                "GitHub Token", value="", type="password",
                help="Set GITHUB_TOKEN as an env var or in st.secrets.")
            if st.button("Test GitHub token", use_container_width=True):
                ok, msg = github_token_status(github_token)
                (st.success if ok else st.warning)(msg)

        if secret_groq:
            groq_key = secret_groq
            st.caption("Groq key: configured from secrets.")
        else:
            groq_key = st.text_input(
                "Groq API Key (optional)", value="", type="password",
                help="Set GROQ_API_KEY as an env var or in st.secrets.")
            if st.button("Test Groq key", use_container_width=True):
                ok, msg = groq_key_ok(groq_key, ss.model)
                (st.success if ok else st.error)(msg)

        model = st.selectbox("Groq model", GROQ_MODELS, index=0)
        ss.model = model
        st.divider()

        aggregation = st.selectbox("Aggregation", ["geometric", "sum"], index=0,
                                   help="Geometric is non-compensatory and recommended.")
        chosen_families = st.multiselect(
            "Families to compute", options=ALL_FAMILIES, default=ALL_FAMILIES,
            format_func=lambda f: f"{f} - {FAMILY_REGISTRY[f]['name']}")
        workers = st.slider("Parallel repositories", 1, 8, 3)
        want_narrative = st.checkbox("Add per-repo LLM summary (needs Groq key)",
                                     value=True)

    # Signed-in user pill (bottom of sidebar)
    _render_user_pill()

    if not chosen_families:
        st.warning("Select at least one family in the sidebar.")
        st.stop()

    # ----------------------------- Inputs ------------------------------- #
    tab1, tab2 = st.tabs(["Manual Entry", "CSV Bulk Upload"])
    urls: List[str] = []
    with tab1:
        txt = st.text_area("GitHub repository URLs (one per line):", height=150,
                           placeholder="https://github.com/e2nIEE/pandapower\n"
                                       "https://github.com/PyPSA/PyPSA")
        if txt:
            urls = [u.strip() for u in txt.splitlines() if u.strip()]
    with tab2:
        st.info("CSV needs a column named one of: URL, Link, Repository, repo, github_url.")
        up = st.file_uploader("Choose a CSV file", type="csv")
        if up is not None:
            try:
                dfu = pd.read_csv(up)
                st.dataframe(dfu.head(3), use_container_width=True)
                col = None
                for c in dfu.columns:
                    if c.strip().lower() in ("url", "link", "repository", "repo",
                                             "github_url", "github url"):
                        col = c; break
                if col:
                    urls = dfu[col].dropna().astype(str).tolist()
                    st.success(f"Found {len(urls)} URLs in column '{col}'.")
                else:
                    urls = dfu.iloc[:, 0].dropna().astype(str).tolist()
                    st.warning(f"No obvious URL column; using the first column "
                               f"({len(urls)} URLs).")
            except Exception as e:
                st.error(f"Error reading CSV: {e}")

    st.divider()

    # ----------------------------- Run ---------------------------------- #
    if st.button("Run RECOPS Scoring", type="primary", use_container_width=True):
        urls = [normalize_repo_url(u) for u in urls if u.strip()]
        seen, deduped = set(), []
        for u in urls:
            if u not in seen:
                seen.add(u); deduped.append(u)
        urls = deduped
        if not urls:
            st.warning("Please provide at least one repository URL."); st.stop()

        progress = st.progress(0.0)
        st.markdown("### Processing Log")
        term = st.empty()
        log = f"> Spinning up {workers} worker(s) for {len(urls)} repositories...\n"
        log += f"> Families: {', '.join(chosen_families)} | Aggregation: {aggregation}\n"
        term.markdown(f'<div class="terminal-box">{log}</div>', unsafe_allow_html=True)

        flat_rows, full_blobs = [], []
        ok = fail = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(process_single_repo, u, github_token, groq_key,
                                 chosen_families, aggregation, want_narrative, model): u
                       for u in urls}
            for i, fut in enumerate(concurrent.futures.as_completed(futures)):
                u = futures[fut]; short = u.split("/")[-1]
                try:
                    res = fut.result()
                    flat_rows.append(res["flat"]); full_blobs.append(res["full"])
                    status = res["flat"].get("Status", "")
                    if status.startswith("Success"):
                        ok += 1
                        log += f"> [{datetime.now():%H:%M:%S}] [OK]   {short} -> {status[:80]}\n"
                    elif status.startswith("Partial"):
                        ok += 1
                        log += f"> [{datetime.now():%H:%M:%S}] [WARN] {short} -> {status[:80]}\n"
                    else:
                        fail += 1
                        log += f"> [{datetime.now():%H:%M:%S}] [FAIL] {short} -> {status[:80]}\n"
                except Exception as exc:
                    fail += 1
                    flat_rows.append({"Link": u, "Status": f"Failed: {exc}"})
                    log += f"> [{datetime.now():%H:%M:%S}] [FAIL] {short} raised {exc}\n"
                progress.progress((i + 1) / len(urls))
                term.markdown(f'<div class="terminal-box">{log}</div>',
                              unsafe_allow_html=True)

        df = pd.DataFrame(flat_rows)
        for c in df.columns:
            df[c] = df[c].apply(lambda v: str(v) if isinstance(v, (list, dict)) else v)
        composites = [f for f in ALL_FAMILIES if f in df.columns]
        comp_bands = [f"{f}_band" for f in composites if f"{f}_band" in df.columns]
        subs = [s for f in ALL_FAMILIES for s in FAMILY_REGISTRY[f]["subs"] if s in df.columns]
        front = ["Link", "Status", "GitHub Repo", "Stars", "Forks", "License"]
        ordered = ([c for c in front if c in df.columns] + composites + comp_bands + subs)
        rest = [c for c in df.columns if c not in ordered]
        df = df[[c for c in ordered if c in df.columns] + rest]

        ss.results = {"df": df, "flat_rows": flat_rows, "full_blobs": full_blobs,
                      "composites": composites, "ok": ok, "fail": fail, "total": len(urls)}
        ss.chat = []

    # ----------------------------- Results ------------------------------ #
    if ss.results:
        r = ss.results
        df = r["df"]
        st.success("Scoring complete.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Processed", r["total"]); c2.metric("Succeeded", r["ok"])
        c3.metric("Failed", r["fail"])
        st.divider()

        st.markdown("### RECOPS Scorecard")
        st.dataframe(df, use_container_width=True)

        try:
            num = df.set_index("Link")[r["composites"]].apply(pd.to_numeric, errors="coerce")
            if not num.empty:
                st.markdown("### Composite Family Scores (0-100)")
                st.bar_chart(num)
        except Exception:
            pass

        narratives = [(b.get("repo_url"), row.get("Summary"))
                      for b, row in zip(r["full_blobs"], r["flat_rows"])
                      if row.get("Summary")]
        if narratives:
            st.markdown("### LLM Summaries")
            for repo, summary in narratives:
                with st.expander(repo, expanded=(len(narratives) == 1)):
                    st.write(summary)

        with st.expander("Per-Repository Raw Detail (JSON)"):
            for blob in r["full_blobs"]:
                st.json(blob)

        st.divider()
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        col_a, col_b, col_c = st.columns(3)
        col_a.download_button("Download Scorecard (CSV)", data=csv_bytes,
                              file_name=f"recops_scorecard_{ts}.csv",
                              mime="text/csv", use_container_width=True)
        try:
            xbuf = io.BytesIO()
            with pd.ExcelWriter(xbuf, engine="openpyxl") as w:
                df.to_excel(w, index=False, sheet_name="scorecard")
            col_b.download_button("Download Scorecard (Excel)", data=xbuf.getvalue(),
                                  file_name=f"recops_scorecard_{ts}.xlsx",
                                  mime=("application/vnd.openxmlformats-officedocument."
                                        "spreadsheetml.sheet"),
                                  use_container_width=True)
        except Exception:
            col_b.caption("Install openpyxl for Excel export.")
        col_c.download_button("Download Full JSON",
                              data=json.dumps(r["full_blobs"], indent=2, default=str).encode(),
                              file_name=f"recops_full_{ts}.json",
                              mime="application/json", use_container_width=True)

        # ----------------- INTERACTIVE LLM PANEL ----------------- #
        st.divider()
        st.markdown("### Ask the LLM about these results")
        if not groq_key:
            st.info("Groq API key not configured -- the chat panel needs one.")
        else:
            st.caption("Examples: \"Which repo has the best Project Health and why?\", "
                       "\"Compare EQ vs DI across all repos\", "
                       "\"What should pypsa improve first?\"")
            quick = st.columns(3)
            preset = None
            if quick[0].button("Rank repos overall", use_container_width=True):
                preset = "Rank the repositories from best to worst overall and justify the ranking using the family scores."
            if quick[1].button("Biggest weakness each", use_container_width=True):
                preset = "For each repository, name its single biggest weakness (lowest family or sub-indicator) and one concrete fix."
            if quick[2].button("Energy-domain fit", use_container_width=True):
                preset = "Which repositories are the strongest fit for energy-system use based on EDF and its sub-indicators?"

            for msg in ss.chat:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            typed = st.chat_input("Ask a question about the scorecard")
            question = preset or typed
            if question:
                with st.chat_message("user"):
                    st.markdown(question)
                ss.chat.append({"role": "user", "content": question})
                ctx = build_llm_context(r["flat_rows"])
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        answer = llm_chat_answer(question, ctx, ss.chat[:-1],
                                                 groq_key, ss.model)
                    st.markdown(answer)
                ss.chat.append({"role": "assistant", "content": answer})

            if ss.chat and st.button("Clear chat"):
                ss.chat = []; st.rerun()


# =========================================================================== #
#                                  DISPATCH                                   #
# =========================================================================== #
def _running_under_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


if __name__ == "__main__":
    if _running_under_streamlit():
        _run_streamlit()
    elif len(sys.argv) > 1:
        raise SystemExit(_run_cli(sys.argv[1:]))
    else:
        print(__doc__)
        print("\nNo URLs given. Examples:")
        print("  streamlit run app.py")
        print("  python app.py https://github.com/PyPSA/PyPSA")
