#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RECOPS Repository Scorer v2 — Adoption & Ecosystem (AE)

Computes the four AE indicators from the RECOPS v2 architecture:
- CEI  — Community Engagement Index
- EVS  — Ecosystem Vitality Score
- RWIS — Real-World Impact Score
- AQS  — Adoption Quality Score

Design choices
--------------
- No fixed formula weights such as 0.40/0.30/0.30 inside indicators.
- Default aggregation is weighted_geometric with data-driven weights.
- Block weights can be learned from a corpus; otherwise a per-repository
  dispersion fallback is used.
- Missing data are kept as None and excluded with weight renormalization.
- Observable absence is scored as 0.0.
- Signals are extracted from local Git history, GitHub API metadata, package
  registry APIs where possible, and repository files/docs.

Optional dependencies
---------------------
- requests: GitHub API, package registry APIs, fallback zip download.
- PyYAML: optional parsing of YAML metadata.

Recommended installation:
    pip install requests pyyaml

Example usage:
    python recops_repo_scores_AE.py https://github.com/PyPSA/PyPSA --score all
    python recops_repo_scores_AE.py https://github.com/PyPSA/PyPSA --score cei
    python recops_repo_scores_AE.py --local ./PyPSA --score evs
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean as _py_mean
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import quote

try:
    import requests
except Exception:  # optional dependency
    requests = None

try:
    import yaml
except Exception:  # optional dependency
    yaml = None


# =========================================================
# Math and aggregation utilities
# =========================================================

def clamp01(x: float) -> float:
    try:
        xf = float(x)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, xf))


def mean(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def weighted_geometric(pairs: Iterable[Tuple[Optional[float], float]], eps: float = 1e-6) -> Optional[float]:
    """
    Weighted geometric mean with explicit missingness.

    - None values are ignored.
    - Remaining weights are renormalized.
    - A zero value remains near-zero through eps, preserving non-compensation.
    """
    vals: List[Tuple[float, float]] = []
    for v, w in pairs:
        if v is None or w is None:
            continue
        try:
            vf = clamp01(float(v))
            wf = float(w)
        except Exception:
            continue
        if wf <= 0:
            continue
        vals.append((vf, wf))
    if not vals:
        return None
    wsum = sum(w for _, w in vals)
    if wsum <= 0:
        return None
    # Adaptive floor: preserves geometric non-compensation while preventing
    # numerical collapse to near-zero in sparse evidence cases.
    positive = sorted([v for v, _ in vals if v > 0.0])
    if positive:
        floor = min(0.05, max(eps, 0.5 * positive[0]))
    else:
        floor = eps
    acc = 0.0
    for v, w in vals:
        acc += (w / wsum) * math.log(max(floor, v))
    return clamp01(math.exp(acc))


def weighted_sum(pairs: Iterable[Tuple[Optional[float], float]]) -> Optional[float]:
    vals: List[Tuple[float, float]] = []
    for v, w in pairs:
        if v is None or w is None:
            continue
        try:
            vf = clamp01(float(v))
            wf = float(w)
        except Exception:
            continue
        if wf <= 0:
            continue
        vals.append((vf, wf))
    if not vals:
        return None
    wsum = sum(w for _, w in vals)
    if wsum <= 0:
        return None
    return clamp01(sum(v * w for v, w in vals) / wsum)


def ratio01(num: Optional[float], den: Optional[float]) -> Optional[float]:
    if num is None or den is None:
        return None
    try:
        n = float(num)
        d = float(den)
    except Exception:
        return None
    if d <= 0:
        return None
    return clamp01(n / d)


def sat_exp(x: Optional[float], scale: float = 1.0) -> Optional[float]:
    """
    Smooth volume saturation: 1 - exp(-x/scale).

    Scale is a soft characteristic count, not a pass/fail threshold.
    """
    if x is None:
        return None
    try:
        xx = max(0.0, float(x))
        ss = max(1e-9, float(scale))
    except Exception:
        return None
    return clamp01(1.0 - math.exp(-xx / ss))


def exp_recency(days: Optional[float], half_life_days: float) -> Optional[float]:
    """Continuous recency decay: score halves every half_life_days."""
    if days is None:
        return None
    try:
        d = max(0.0, float(days))
        h = max(1e-9, float(half_life_days))
    except Exception:
        return None
    return clamp01(math.exp(-math.log(2.0) * d / h))


def _exp_decay_days(days: Optional[float], scale_days: float) -> Optional[float]:
    """Continuous timing decay: exp(-days/scale)."""
    if days is None:
        return None
    try:
        d = max(0.0, float(days))
        s = max(1e-9, float(scale_days))
    except Exception:
        return None
    return clamp01(math.exp(-d / s))


def stability_from_cv(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    if len(vals) < 2:
        return 1.0
    m = _py_mean(vals)
    if m <= 0:
        return None
    var = _py_mean([(x - m) ** 2 for x in vals])
    cv = math.sqrt(max(0.0, var)) / m
    return clamp01(math.exp(-cv))


def entropy_normalized(counts: Iterable[int]) -> Optional[float]:
    vals = [float(v) for v in counts if v is not None and float(v) > 0]
    if not vals:
        return None
    if len(vals) == 1:
        return 0.0
    s = sum(vals)
    probs = [v / s for v in vals]
    h = -sum(p * math.log(p) for p in probs if p > 0)
    return clamp01(h / math.log(len(vals)))


def _log_repo_signal(hits: Optional[float], ref: Optional[float]) -> Optional[float]:
    """
    Repository-relative logarithmic normalization:
        log(1+h) / log(1+h_ref)
    """
    if hits is None or ref is None:
        return None
    try:
        h = max(0.0, float(hits))
        r = max(0.0, float(ref))
    except Exception:
        return None
    if h <= 0 or r <= 0:
        return 0.0
    return clamp01(math.log1p(h) / math.log1p(r))


def percentile_score(value: Optional[float], corpus_values: Optional[List[float]]) -> Optional[float]:
    """
    Corpus-calibrated percentile score.
    Used only when the user supplies an empirical calibration distribution.
    """
    if value is None or not corpus_values:
        return None
    vals = sorted(float(v) for v in corpus_values if v is not None)
    if not vals:
        return None
    return ratio01(sum(1 for v in vals if v <= float(value)), len(vals))


def band(score_0_100: Optional[float]) -> Optional[str]:
    if score_0_100 is None:
        return None
    if score_0_100 >= 80:
        return "Excellent"
    if score_0_100 >= 60:
        return "Good"
    if score_0_100 >= 40:
        return "Moderate"
    if score_0_100 >= 20:
        return "Poor"
    return "Very Poor"


# =========================================================
# Subprocess / repository acquisition
# =========================================================

def run(cmd: List[str], cwd: Optional[Path] = None, timeout_s: int = 120) -> Tuple[int, str, str]:
    try:
        r = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd is not None else None,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        return r.returncode, r.stdout, r.stderr
    except Exception as e:
        return 1, "", str(e)


def parse_github_owner_repo(url: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if not url:
        return None, None
    m = re.search(r"github\.com/([^/]+)/([^/]+)", url)
    if not m:
        return None, None
    owner = m.group(1)
    repo = m.group(2).replace(".git", "")
    return owner, repo


def github_to_zip(url: str) -> Tuple[str, str, str]:
    owner, repo = parse_github_owner_repo(url)
    if not owner or not repo:
        raise ValueError("Invalid GitHub URL")
    return owner, repo, f"https://github.com/{owner}/{repo}/archive/refs/heads/main.zip"


def clone_or_download(github_url: str) -> Path:
    temp = Path(tempfile.mkdtemp(prefix="recops_ae_"))
    target = temp / "repo"

    code, _, _ = run(["git", "--version"])
    if code == 0:
        # Keep history because AE uses contributor retention, repeat activity, tags, and age.
        c, _, _ = run(["git", "clone", "--filter=blob:none", "--tags", github_url, str(target)], cwd=temp, timeout_s=900)
        if c == 0 and target.exists():
            return target

    if requests is None:
        raise RuntimeError("Need git or requests to fetch a GitHub repository")

    _, _, zip_url = github_to_zip(github_url)
    zpath = temp / "repo.zip"
    r = requests.get(zip_url, timeout=60)
    if r.status_code != 200:
        zip_url = zip_url.replace("/main.zip", "/master.zip")
        r = requests.get(zip_url, timeout=60)
    r.raise_for_status()
    zpath.write_bytes(r.content)
    shutil.unpack_archive(str(zpath), str(temp))
    dirs = [p for p in temp.iterdir() if p.is_dir() and p.name != "repo"]
    if dirs:
        return dirs[0]
    raise RuntimeError("Cannot extract repository archive")


# =========================================================
# File scanning
# =========================================================

IGNORE_DIRS = {
    ".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "node_modules", "dist", "build", ".venv", "venv", ".idea", ".vscode",
}

TEXT_SUFFIXES = {
    ".md", ".rst", ".txt", ".cff", ".toml", ".json", ".yaml", ".yml",
    ".ini", ".cfg", ".py", ".js", ".ts", ".java", ".jl", ".c", ".cpp",
    ".ipynb", ".bib", ".ris", ".csv",
}

DOC_DIR_NAMES = {
    "docs", "doc", "examples", "example", "notebooks", "tutorials", "tutorial",
    "case_studies", "case-study", "use_cases", "use-cases", ".github", "community",
    "papers", "publications", "benchmarks", "benchmark",
}

SOURCE_SUFFIXES = {
    ".py", ".js", ".ts", ".java", ".cpp", ".c", ".cs", ".jl", ".r", ".m",
    ".gms", ".mod", ".dat", ".inc", ".lp", ".mps",
}


def iter_files(root: Path) -> Iterable[Path]:
    root = Path(root)
    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        for f in files:
            yield Path(base) / f


def _safe_read_text(path: Path) -> Optional[str]:
    try:
        return Path(path).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None


def _safe_json_load(path: Path) -> Optional[Any]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return None


def _safe_yaml_load_text(text: Optional[str]) -> Optional[Any]:
    if text is None or yaml is None:
        return None
    try:
        return yaml.safe_load(text)
    except Exception:
        return None


def exists_any(root: Path, names: Iterable[str]) -> float:
    wanted = {n.lower() for n in names}
    for p in iter_files(root):
        if p.name.lower() in wanted:
            return 1.0
    return 0.0


def has_dir(root: Path, names: Iterable[str]) -> float:
    wanted = {n.lower() for n in names}
    for p in Path(root).iterdir():
        if p.is_dir() and p.name.lower() in wanted:
            return 1.0
    return 0.0


def find_files_by_names(root: Path, names: Iterable[str]) -> List[Path]:
    wanted = {n.lower() for n in names}
    return [p for p in iter_files(root) if p.name.lower() in wanted]


def is_doc_like_path(root: Path, path: Path) -> bool:
    try:
        rel = path.relative_to(root)
    except Exception:
        return False
    parts = [x.lower() for x in rel.parts]
    name = path.name.lower()
    if name.startswith("readme") or name in {
        "citation.cff", "codemeta.json", "zenodo.json", ".zenodo.json",
        "acknowledgements", "acknowledgments", "publications.md", "papers.md",
        "users.md", "use_cases.md", "use-cases.md", "case_studies.md", "case-studies.md",
    }:
        return True
    return any(part in DOC_DIR_NAMES for part in parts[:-1])


def candidate_texts(root: Path) -> Dict[str, str]:
    """Read texts relevant to adoption/ecosystem evidence."""
    out: Dict[str, str] = {}
    for p in iter_files(root):
        if p.suffix.lower() not in TEXT_SUFFIXES and p.name.lower() not in {"acknowledgements", "acknowledgments"}:
            continue
        if not is_doc_like_path(root, p):
            # Include root metadata files only.
            try:
                rel = p.relative_to(root)
            except Exception:
                continue
            if len(rel.parts) != 1 or p.name.lower() not in {
                "pyproject.toml", "package.json", "citation.cff", "codemeta.json", "readme.md", "readme.rst",
            }:
                continue
        txt = _safe_read_text(p)
        if txt:
            try:
                out[str(p.relative_to(root))] = txt
            except Exception:
                out[str(p)] = txt
    return out


def count_regex_hits_in_texts(texts: Dict[str, str], patterns: Iterable[str]) -> int:
    rx = [re.compile(p, flags=re.I | re.M) for p in patterns]
    hits = 0
    for txt in texts.values():
        for r in rx:
            hits += len(r.findall(txt or ""))
    return hits


def count_regex_hits_repo(root: Path, patterns: Iterable[str], suffixes: Optional[set] = None) -> int:
    suffixes = suffixes or TEXT_SUFFIXES
    rx = [re.compile(p, flags=re.I | re.M) for p in patterns]
    hits = 0
    for p in iter_files(root):
        if p.suffix.lower() not in suffixes:
            continue
        txt = _safe_read_text(p)
        if not txt:
            continue
        for r in rx:
            hits += len(r.findall(txt))
    return hits


def count_path_hits(root: Path, patterns: Iterable[str]) -> int:
    rx = [re.compile(p, flags=re.I) for p in patterns]
    hits = 0
    for p in iter_files(root):
        try:
            rel = str(p.relative_to(root)).replace("\\", "/")
        except Exception:
            rel = str(p)
        for r in rx:
            hits += len(r.findall(rel))
    return hits


def repo_hit_ref(*counts: Optional[int]) -> float:
    return max([1.0] + [float(c) for c in counts if c is not None])


# =========================================================
# Git helpers
# =========================================================

def is_git_repo(repo: Path) -> bool:
    code, out, _ = run(["git", "rev-parse", "--is-inside-work-tree"], cwd=repo)
    return code == 0 and out.strip().lower() == "true"


def git_epoch_list(repo: Path, args: List[str], timeout_s: int = 180) -> List[int]:
    code, out, _ = run(["git", *args], cwd=repo, timeout_s=timeout_s)
    if code != 0:
        return []
    ts: List[int] = []
    for line in out.splitlines():
        s = line.strip()
        if s.isdigit():
            ts.append(int(s))
    return ts


def month_key_from_epoch(ts: int) -> str:
    d = dt.datetime.utcfromtimestamp(int(ts))
    return f"{d.year:04d}-{d.month:02d}"


def add_months(d: dt.datetime, delta_months: int) -> dt.datetime:
    y = d.year
    m = d.month + delta_months
    y += (m - 1) // 12
    m = (m - 1) % 12 + 1
    return dt.datetime(y, m, 1)


def rolling_month_keys(months: int, now: Optional[dt.datetime] = None) -> List[str]:
    now = now or dt.datetime.utcnow()
    end_month = dt.datetime(now.year, now.month, 1)
    start_month = add_months(end_month, -(max(1, months) - 1))
    keys = []
    cursor = start_month
    for _ in range(max(1, months)):
        keys.append(f"{cursor.year:04d}-{cursor.month:02d}")
        cursor = add_months(cursor, 1)
    return keys


def git_commit_authors(repo: Path, months: Optional[int] = None) -> List[Tuple[str, int]]:
    if not is_git_repo(repo):
        return []
    args = ["log", "--format=%ae%x09%ct"]
    if months is not None:
        since = (dt.datetime.utcnow() - dt.timedelta(days=int(30.4375 * months))).strftime("%Y-%m-%d")
        args.insert(1, f"--since={since}")
    code, out, _ = run(["git", *args], cwd=repo, timeout_s=240)
    if code != 0:
        return []
    rows: List[Tuple[str, int]] = []
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) != 2:
            continue
        email = parts[0].strip().lower()
        ts = parts[1].strip()
        if email and "@" in email and ts.isdigit():
            rows.append((email, int(ts)))
    return rows


def git_commit_activity(repo: Path, months: int = 12) -> Dict[str, Any]:
    if not is_git_repo(repo):
        return {"available": False, "reason": "not_a_git_repo"}
    now = dt.datetime.utcnow()
    since = (now - dt.timedelta(days=int(30.4375 * months))).strftime("%Y-%m-%d")
    recent_ts = git_epoch_list(repo, ["log", f"--since={since}", "--pretty=format:%ct"])
    last_ts = git_epoch_list(repo, ["log", "-1", "--pretty=format:%ct"])
    first_ts = git_epoch_list(repo, ["log", "--reverse", "--pretty=format:%ct"], timeout_s=240)
    last_days = None
    if last_ts:
        last_days = (now - dt.datetime.utcfromtimestamp(last_ts[0])).total_seconds() / 86400.0
    age_days = None
    if first_ts:
        age_days = (now - dt.datetime.utcfromtimestamp(first_ts[0])).total_seconds() / 86400.0
    monthly_counter = Counter(month_key_from_epoch(t) for t in recent_ts)
    keys = rolling_month_keys(months, now=now)
    monthly_counts = {k: int(monthly_counter.get(k, 0)) for k in keys}
    return {
        "available": True,
        "total_commits": len(recent_ts),
        "last_commit_days_ago": last_days,
        "repo_age_days": age_days,
        "monthly_counts": monthly_counts,
        "active_months": sum(1 for v in monthly_counts.values() if v > 0),
        "window_months": months,
    }


def contributor_sets(repo: Path, recent_months: int = 12, prior_months: int = 24) -> Dict[str, Any]:
    if not is_git_repo(repo):
        return {"available": False, "reason": "not_a_git_repo"}
    now = dt.datetime.utcnow()
    recent_since = now - dt.timedelta(days=int(30.4375 * recent_months))
    prior_since = now - dt.timedelta(days=int(30.4375 * prior_months))
    rows = git_commit_authors(repo, months=prior_months)
    recent = set()
    prior = set()
    all_window = set()
    per_author_months: Dict[str, set] = defaultdict(set)
    for email, ts in rows:
        d = dt.datetime.utcfromtimestamp(ts)
        all_window.add(email)
        per_author_months[email].add(f"{d.year:04d}-{d.month:02d}")
        if d >= recent_since:
            recent.add(email)
        elif d >= prior_since:
            prior.add(email)
    # all-time contributors, useful to avoid interpreting a mature stable repo as weak.
    all_authors = {email for email, _ in git_commit_authors(repo, months=None)}
    new_recent = recent - prior
    retained_recent = recent.intersection(prior)
    return {
        "available": True,
        "recent_contributors": recent,
        "prior_contributors": prior,
        "window_contributors": all_window,
        "all_time_contributors": all_authors,
        "new_recent_contributors": new_recent,
        "retained_recent_contributors": retained_recent,
        "per_author_months": per_author_months,
        "recent_months": recent_months,
        "prior_months": prior_months,
    }


def git_file_years(repo_path: Path, rel_path: str) -> List[int]:
    code, out, _ = run(["git", "log", "--follow", "--format=%ct", "--", rel_path], cwd=repo_path, timeout_s=120)
    if code != 0:
        return []
    years = set()
    for line in out.splitlines():
        s = line.strip()
        if s.isdigit():
            years.add(dt.datetime.utcfromtimestamp(int(s)).year)
    return sorted(years)


# =========================================================
# GitHub API helpers
# =========================================================

def gh_api_get(url: str, token: Optional[str] = None, accept: Optional[str] = None, timeout_s: int = 30) -> Optional[Any]:
    if requests is None:
        return None
    headers = {"User-Agent": "recops-ae/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if accept:
        headers["Accept"] = accept
    try:
        r = requests.get(url, headers=headers, timeout=timeout_s)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


def gh_paginate(url: str, token: Optional[str] = None, accept: Optional[str] = None, per_page: int = 100, max_pages: int = 10) -> Optional[List[Any]]:
    """
    None when the first page cannot be retrieved; [] when the API successfully returned no rows.
    """
    out: List[Any] = []
    if requests is None:
        return None
    for page in range(1, max_pages + 1):
        sep = "&" if "?" in url else "?"
        data = gh_api_get(f"{url}{sep}per_page={per_page}&page={page}", token=token, accept=accept)
        if data is None:
            return None if page == 1 else out
        if not isinstance(data, list):
            return None if page == 1 else out
        if len(data) == 0:
            break
        out.extend(data)
        if len(data) < per_page:
            break
    return out


def parse_iso8601(s: Optional[str]) -> Optional[dt.datetime]:
    if not s:
        return None
    try:
        if s.endswith("Z"):
            return dt.datetime.fromisoformat(s[:-1]).replace(tzinfo=dt.timezone.utc).astimezone(dt.timezone.utc).replace(tzinfo=None)
        return dt.datetime.fromisoformat(s).astimezone(dt.timezone.utc).replace(tzinfo=None)
    except Exception:
        return None


def github_repo_info(repo_url: Optional[str], token: Optional[str] = None) -> Optional[Dict[str, Any]]:
    owner, repo = parse_github_owner_repo(repo_url)
    if not owner or not repo:
        return None
    data = gh_api_get(f"https://api.github.com/repos/{owner}/{repo}", token=token)
    return data if isinstance(data, dict) else None


def github_topics(repo_url: Optional[str], token: Optional[str] = None) -> Optional[List[str]]:
    owner, repo = parse_github_owner_repo(repo_url)
    if not owner or not repo:
        return []
    data = gh_api_get(f"https://api.github.com/repos/{owner}/{repo}/topics", token=token, accept="application/vnd.github+json")
    if data is None:
        return None
    if isinstance(data, dict) and isinstance(data.get("names"), list):
        return [str(x).lower() for x in data["names"]]
    return []


def github_issues_or_prs(
    repo_url: Optional[str], token: Optional[str], kind: str = "issues", months: int = 12, max_pages: int = 10
) -> Optional[List[Dict[str, Any]]]:
    owner, repo = parse_github_owner_repo(repo_url)
    if not owner or not repo:
        return []
    since = (dt.datetime.utcnow() - dt.timedelta(days=int(30.4375 * months))).replace(microsecond=0).isoformat() + "Z"
    if kind == "pulls":
        items = gh_paginate(f"https://api.github.com/repos/{owner}/{repo}/pulls?state=all&sort=updated&direction=desc", token=token, max_pages=max_pages)
        if items is None:
            return None
        out = []
        cutoff = dt.datetime.utcnow() - dt.timedelta(days=int(30.4375 * months))
        for it in items:
            if not isinstance(it, dict):
                continue
            updated = parse_iso8601(it.get("updated_at"))
            if updated and updated >= cutoff:
                out.append(it)
        return out
    items = gh_paginate(f"https://api.github.com/repos/{owner}/{repo}/issues?state=all&since={since}", token=token, max_pages=max_pages)
    if items is None:
        return None
    out = []
    for it in items:
        if isinstance(it, dict) and "pull_request" not in it:
            out.append(it)
    return out


def github_stars_gained(repo_url: Optional[str], token: Optional[str], months: int = 12, max_pages: int = 10) -> Optional[int]:
    owner, repo = parse_github_owner_repo(repo_url)
    if not owner or not repo or requests is None:
        return None
    cutoff = dt.datetime.utcnow() - dt.timedelta(days=int(30.4375 * months))
    gained = 0
    accept = "application/vnd.github.star+json"
    for page in range(1, max_pages + 1):
        data = gh_api_get(f"https://api.github.com/repos/{owner}/{repo}/stargazers?per_page=100&page={page}", token=token, accept=accept)
        if data is None:
            return None if page == 1 else gained
        if not isinstance(data, list) or not data:
            break
        stop = False
        for it in data:
            if not isinstance(it, dict):
                continue
            starred = parse_iso8601(it.get("starred_at"))
            if starred is None:
                continue
            if starred >= cutoff:
                gained += 1
            else:
                stop = True
                break
        if stop or len(data) < 100:
            break
    return gained


def github_forks_gained(repo_url: Optional[str], token: Optional[str], months: int = 12, max_pages: int = 10) -> Optional[int]:
    owner, repo = parse_github_owner_repo(repo_url)
    if not owner or not repo or requests is None:
        return None
    cutoff = dt.datetime.utcnow() - dt.timedelta(days=int(30.4375 * months))
    gained = 0
    for page in range(1, max_pages + 1):
        data = gh_api_get(f"https://api.github.com/repos/{owner}/{repo}/forks?sort=newest&per_page=100&page={page}", token=token)
        if data is None:
            return None if page == 1 else gained
        if not isinstance(data, list) or not data:
            break
        stop = False
        for it in data:
            if not isinstance(it, dict):
                continue
            created = parse_iso8601(it.get("created_at"))
            if created is None:
                continue
            if created >= cutoff:
                gained += 1
            else:
                stop = True
                break
        if stop or len(data) < 100:
            break
    return gained


# =========================================================
# Package and citation metadata helpers
# =========================================================

DOI_RX = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+", flags=re.I)
ARXIV_RX = re.compile(r"\barXiv[: ]?(\d{4}\.\d{4,5}(?:v\d+)?)\b", flags=re.I)


def _package_names_from_metadata(root: Path) -> Dict[str, List[str]]:
    """Extract likely package names from common metadata files."""
    out: Dict[str, List[str]] = {"pypi": [], "npm": [], "julia": [], "rust": []}

    pyproject = root / "pyproject.toml"
    if pyproject.exists():
        txt = _safe_read_text(pyproject) or ""
        # First `name = "..."` is usually enough for PEP 621 / Poetry metadata.
        m = re.search(r"(?im)^\s*name\s*=\s*['\"]([^'\"]+)['\"]", txt)
        if m:
            out["pypi"].append(m.group(1).strip())

    setup_cfg = root / "setup.cfg"
    if setup_cfg.exists():
        txt = _safe_read_text(setup_cfg) or ""
        m = re.search(r"(?im)^\s*name\s*=\s*([^\n#]+)", txt)
        if m:
            out["pypi"].append(m.group(1).strip())

    setup_py = root / "setup.py"
    if setup_py.exists():
        txt = _safe_read_text(setup_py) or ""
        m = re.search(r"name\s*=\s*['\"]([^'\"]+)['\"]", txt)
        if m:
            out["pypi"].append(m.group(1).strip())

    package_json = root / "package.json"
    if package_json.exists():
        data = _safe_json_load(package_json)
        if isinstance(data, dict) and isinstance(data.get("name"), str):
            out["npm"].append(data["name"].strip())

    project_toml = root / "Project.toml"
    if project_toml.exists():
        txt = _safe_read_text(project_toml) or ""
        m = re.search(r"(?im)^\s*name\s*=\s*['\"]([^'\"]+)['\"]", txt)
        if m:
            out["julia"].append(m.group(1).strip())

    cargo = root / "Cargo.toml"
    if cargo.exists():
        txt = _safe_read_text(cargo) or ""
        m = re.search(r"(?im)^\s*name\s*=\s*['\"]([^'\"]+)['\"]", txt)
        if m:
            out["rust"].append(m.group(1).strip())

    return {k: sorted(set(vv for vv in vals if vv)) for k, vals in out.items()}


def _pypi_stats(name: str) -> Dict[str, Any]:
    if requests is None or not name:
        return {"available": False}
    try:
        r = requests.get(f"https://pypi.org/pypi/{quote(name)}/json", timeout=20)
        if r.status_code != 200:
            return {"available": False}
        data = r.json()
        releases = data.get("releases", {}) if isinstance(data, dict) else {}
        release_count = len(releases) if isinstance(releases, dict) else None
        # PyPI JSON does not expose total downloads anymore. Release count is stable metadata.
        return {"available": True, "package": name, "release_count": release_count}
    except Exception:
        return {"available": False}


def _npm_stats(name: str) -> Dict[str, Any]:
    if requests is None or not name:
        return {"available": False}
    out: Dict[str, Any] = {"available": False, "package": name}
    try:
        r = requests.get(f"https://registry.npmjs.org/{quote(name, safe='@/')}", timeout=20)
        if r.status_code == 200:
            data = r.json()
            versions = data.get("versions", {}) if isinstance(data, dict) else {}
            out.update({"available": True, "version_count": len(versions) if isinstance(versions, dict) else None})
    except Exception:
        pass
    try:
        d = requests.get(f"https://api.npmjs.org/downloads/point/last-month/{quote(name, safe='@/')}", timeout=20)
        if d.status_code == 200:
            dd = d.json()
            if isinstance(dd, dict) and isinstance(dd.get("downloads"), int):
                out["downloads_last_month"] = dd["downloads"]
                out["available"] = True
    except Exception:
        pass
    return out


def package_registry_evidence(root: Path) -> Dict[str, Any]:
    names = _package_names_from_metadata(root)
    pypi = [_pypi_stats(n) for n in names.get("pypi", [])[:3]]
    npm = [_npm_stats(n) for n in names.get("npm", [])[:3]]
    return {"package_names": names, "pypi": pypi, "npm": npm}


def extract_publication_metadata(root: Path) -> Dict[str, Any]:
    texts = candidate_texts(root)
    blob = "\n".join(texts.values())
    bib_files = [p for p in iter_files(root) if p.suffix.lower() in {".bib", ".ris"}]
    citation_files = find_files_by_names(root, ["CITATION.cff", "citation.cff", "codemeta.json", "zenodo.json", ".zenodo.json"])
    for p in bib_files + citation_files:
        blob += "\n" + (_safe_read_text(p) or "")
    dois = sorted(set(m.rstrip(".,;)]}") for m in DOI_RX.findall(blob)))
    arxivs = sorted(set(ARXIV_RX.findall(blob)))
    bib_entries = 0
    for p in bib_files:
        bib_entries += len(re.findall(r"@(?:article|inproceedings|proceedings|book|misc|software|dataset)\s*\{", _safe_read_text(p) or "", re.I))
    return {"dois": dois, "arxivs": arxivs, "bib_entries": bib_entries, "citation_files": citation_files, "bib_files": bib_files}


def crossref_citation_counts(dois: List[str], max_dois: int = 5) -> Dict[str, Any]:
    if requests is None or not dois:
        return {"external_metadata_available": 0.0, "checked": 0, "is_referenced_by_total": None}
    total = 0
    checked = 0
    venues = []
    for doi in dois[:max_dois]:
        try:
            r = requests.get(f"https://api.crossref.org/works/{doi}", headers={"User-Agent": "recops-ae/1.0"}, timeout=20)
            if r.status_code != 200:
                continue
            data = r.json()
            msg = data.get("message") if isinstance(data, dict) else None
            if not isinstance(msg, dict):
                continue
            checked += 1
            if isinstance(msg.get("is-referenced-by-count"), int):
                total += int(msg["is-referenced-by-count"])
            for key in ["container-title", "short-container-title"]:
                val = msg.get(key)
                if isinstance(val, list):
                    venues.extend(str(x) for x in val if x)
        except Exception:
            continue
    return {
        "external_metadata_available": 1.0 if checked else 0.0,
        "checked": checked,
        "is_referenced_by_total": total if checked else None,
        "venues": sorted(set(venues)),
    }


# =========================================================
# Data-driven weight learning
# =========================================================

def data_driven_weights_from_blocks(blocks: Dict[str, Optional[float]]) -> Dict[str, float]:
    """
    Repository-level fallback.

    Observed blocks farther from the repository's cross-block mean receive more
    weight because they discriminate this repository more. If all observed blocks
    are identical, equal weights are used across observed blocks. Missing blocks
    receive zero weight.
    """
    observed = {k: float(v) for k, v in blocks.items() if v is not None}
    out = {k: 0.0 for k in blocks}
    if not observed:
        return out
    mu = _py_mean(list(observed.values()))
    raw = {k: abs(v - mu) for k, v in observed.items()}
    z = sum(raw.values())
    if z <= 0:
        raw = {k: 1.0 for k in observed}
        z = float(len(observed))
    # Smooth toward uniform to reduce over-concentration from one outlier block.
    n = float(len(observed))
    smooth = 0.30
    uniform = 1.0 / n
    for k, v in raw.items():
        out[k] = (1.0 - smooth) * (v / z) + smooth * uniform
    return out


def robust_variance(vals: List[float]) -> float:
    if len(vals) < 2:
        return 0.0
    m = _py_mean(vals)
    return _py_mean([(x - m) ** 2 for x in vals])


def pairwise_abs_corr(a: List[Optional[float]], b: List[Optional[float]]) -> Optional[float]:
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if len(pairs) < 3:
        return None
    xs = [float(x) for x, _ in pairs]
    ys = [float(y) for _, y in pairs]
    mx, my = _py_mean(xs), _py_mean(ys)
    vx = _py_mean([(x - mx) ** 2 for x in xs])
    vy = _py_mean([(y - my) ** 2 for y in ys])
    if vx <= 0 or vy <= 0:
        return None
    cov = _py_mean([(x - mx) * (y - my) for x, y in pairs])
    return abs(cov / math.sqrt(vx * vy))


def norm_weights_from_rows(rows: List[Dict[str, Optional[float]]], keys: List[str]) -> Dict[str, float]:
    """
    Learn weights from a corpus:
        raw(k) = variance(k) × coverage(k) × uniqueness(k)
    """
    vals = {k: [r.get(k) for r in rows] for k in keys}

    def coverage(arr: List[Optional[float]]) -> float:
        return sum(1 for x in arr if x is not None) / max(1, len(arr))

    info = {k: robust_variance([float(x) for x in vals[k] if x is not None]) for k in keys}
    covg = {k: coverage(vals[k]) for k in keys}
    uniq: Dict[str, float] = {}
    for k in keys:
        corrs = [pairwise_abs_corr(vals[k], vals[j]) for j in keys if j != k]
        c = mean(corrs)
        uniq[k] = clamp01(1.0 - c) if c is not None else 1.0
    raw = {k: max(1e-9, info[k] * covg[k] * uniq[k]) for k in keys}
    z = sum(raw.values()) or 1.0
    return {k: raw[k] / z for k in keys}


def final_score_from_blocks(
    blocks: Dict[str, Optional[float]],
    learned_weights: Optional[Dict[str, float]] = None,
    aggregation: str = "geometric",
) -> Tuple[Optional[float], Dict[str, float], str, str]:
    weights_source = "learned_from_corpus" if learned_weights is not None else "repo_data_driven_fallback"
    weights = learned_weights or data_driven_weights_from_blocks(blocks)
    pairs = [(blocks[k], float(weights.get(k, 0.0))) for k in blocks]
    agg = (aggregation or "geometric").strip().lower()
    if agg == "sum":
        val = weighted_sum(pairs)
        agg_name = "weighted_sum"
    else:
        val = weighted_geometric(pairs)
        agg_name = "weighted_geometric"
    score = 100.0 * val if val is not None else None
    return score, weights, weights_source, agg_name


# =========================================================
# AE.1 CEI — Community Engagement Index
# =========================================================

def compute_contributor_engagement(repo_path: Path) -> Tuple[Optional[float], Dict[str, Any]]:
    """
    Contributor_Engagement = mean(active base, new contributors, retention, persistence, concentration safety)

    It measures participation dynamics, not general project health. Commit volume
    is avoided to reduce redundancy with PH/PVI.
    """
    raw = contributor_sets(repo_path, recent_months=12, prior_months=24)
    if not raw.get("available"):
        return None, {"available": False, "reason": raw.get("reason")}

    recent = raw["recent_contributors"]
    prior = raw["prior_contributors"]
    all_time = raw["all_time_contributors"]
    new_recent = raw["new_recent_contributors"]
    retained = raw["retained_recent_contributors"]
    per_author_months = raw["per_author_months"]

    ce_active_base = ratio01(len(recent), len(all_time)) if all_time else None
    ce_newcomer_ratio = ratio01(len(new_recent), len(recent)) if recent else 0.0
    ce_retention = ratio01(len(retained), len(prior)) if prior else None
    active_author_months = sum(len(mths) for author, mths in per_author_months.items() if author in recent)
    ce_persistence = ratio01(active_author_months, max(1, 12 * max(1, len(recent)))) if recent else 0.0

    rows = git_commit_authors(repo_path, months=12)
    counts = Counter(email for email, _ in rows)
    top_share = None
    if counts:
        top_share = max(counts.values()) / sum(counts.values())
    ce_concentration_safety = clamp01(1.0 - top_share) if top_share is not None else None

    block = mean([ce_active_base, ce_newcomer_ratio, ce_retention, ce_persistence, ce_concentration_safety])
    return block, {
        "available": True,
        "block": block,
        "sub": {
            "CE_active_base_ratio": ce_active_base,
            "CE_newcomer_ratio": ce_newcomer_ratio,
            "CE_retention_ratio": ce_retention,
            "CE_participation_persistence": ce_persistence,
            "CE_concentration_safety": ce_concentration_safety,
        },
        "raw": {
            "recent_contributors_12m": len(recent),
            "prior_contributors_12_to_24m": len(prior),
            "all_time_contributors": len(all_time),
            "new_recent_contributors": len(new_recent),
            "retained_recent_contributors": len(retained),
            "active_author_months": active_author_months,
            "top_contributor_commit_share_12m": top_share,
        },
        "missingness_note": "Contributor signals require a git repository with history; missing values are not forced to zero.",
    }


def compute_user_engagement(repo_path: Path, repo_url: Optional[str], token: Optional[str], calibration: Optional[Dict[str, Any]]) -> Tuple[Optional[float], Dict[str, Any]]:
    """
    User_Engagement uses weak but observable external interest signals.
    It avoids duplicating PH community growth by combining current stock with recent deltas and package/download evidence.
    """
    info = github_repo_info(repo_url, token) if repo_url else None
    stars = forks = watchers = subscribers = None
    if isinstance(info, dict):
        stars = info.get("stargazers_count")
        forks = info.get("forks_count")
        watchers = info.get("watchers_count")
        subscribers = info.get("subscribers_count")

    stars_gained = github_stars_gained(repo_url, token, months=12) if repo_url else None
    forks_gained = github_forks_gained(repo_url, token, months=12) if repo_url else None

    # Percentile calibration if provided; otherwise use size-adjusted growth ratios and log relative balance.
    ue_star_stock = percentile_score(stars, (calibration or {}).get("stars_total") if isinstance(calibration, dict) else None)
    ue_fork_stock = percentile_score(forks, (calibration or {}).get("forks_total") if isinstance(calibration, dict) else None)
    ue_recent_star_growth = ratio01(stars_gained, stars) if stars_gained is not None and stars is not None else None
    ue_recent_fork_growth = ratio01(forks_gained, forks) if forks_gained is not None and forks is not None else None
    ue_watch_subscribe = ratio01(subscribers, watchers) if subscribers is not None and watchers is not None else None

    registries = package_registry_evidence(repo_path)
    npm_downloads = [x.get("downloads_last_month") for x in registries["npm"] if isinstance(x, dict) and x.get("downloads_last_month") is not None]
    npm_download_total = sum(int(x) for x in npm_downloads) if npm_downloads else None
    ue_downloads = percentile_score(npm_download_total, (calibration or {}).get("npm_downloads_last_month") if isinstance(calibration, dict) else None)
    # If no corpus is supplied, release/version count is still weak package-registry evidence.
    version_counts = []
    for x in registries["pypi"]:
        if isinstance(x, dict) and x.get("release_count") is not None:
            version_counts.append(int(x["release_count"]))
    for x in registries["npm"]:
        if isinstance(x, dict) and x.get("version_count") is not None:
            version_counts.append(int(x["version_count"]))
    ue_registry_presence = mean([1.0 if registries["pypi"] or registries["npm"] else 0.0, sat_exp(sum(version_counts), scale=10.0) if version_counts else None])

    block = mean([ue_star_stock, ue_fork_stock, ue_recent_star_growth, ue_recent_fork_growth, ue_watch_subscribe, ue_downloads, ue_registry_presence])
    return block, {
        "available": True,
        "block": block,
        "sub": {
            "UE_star_stock_calibrated": ue_star_stock,
            "UE_fork_stock_calibrated": ue_fork_stock,
            "UE_recent_star_growth_ratio": ue_recent_star_growth,
            "UE_recent_fork_growth_ratio": ue_recent_fork_growth,
            "UE_watch_subscribe_ratio": ue_watch_subscribe,
            "UE_downloads_calibrated": ue_downloads,
            "UE_registry_presence": ue_registry_presence,
        },
        "raw": {
            "stars_total": stars,
            "forks_total": forks,
            "watchers_count": watchers,
            "subscribers_count": subscribers,
            "stars_gained_12m": stars_gained,
            "forks_gained_12m": forks_gained,
            "registry_evidence": registries,
            "npm_downloads_last_month_total": npm_download_total,
        },
        "missingness_note": "Stock popularity is only scored with a supplied corpus calibration; otherwise raw totals are reported and growth/registry evidence are used.",
    }


def compute_communication_activity(repo_url: Optional[str], token: Optional[str]) -> Tuple[Optional[float], Dict[str, Any]]:
    """
    Communication_Activity measures interaction, discussion depth, and response coverage.
    Requires GitHub API. It is distinct from PH maintenance quality: it measures community communication intensity, not resolution speed.
    """
    issues = github_issues_or_prs(repo_url, token, kind="issues", months=12, max_pages=10) if repo_url else []
    prs = github_issues_or_prs(repo_url, token, kind="pulls", months=12, max_pages=10) if repo_url else []
    if repo_url and (issues is None or prs is None):
        return None, {
            "available": False,
            "reason": "github_issues_or_pulls_api_unavailable",
        }

    total_threads = len(issues) + len(prs)
    issue_comments = [it.get("comments") for it in issues if isinstance(it.get("comments"), int)]
    pr_comments = [it.get("comments") for it in prs if isinstance(it.get("comments"), int)]
    all_comments = issue_comments + pr_comments
    avg_comments = _py_mean(all_comments) if all_comments else None
    with_comments = sum(1 for x in all_comments if isinstance(x, int) and x > 0)

    ca_discussion_depth = sat_exp(avg_comments, scale=5.0) if avg_comments is not None else None
    ca_response_coverage = ratio01(with_comments, total_threads) if total_threads else 0.0
    ca_thread_flow = sat_exp(total_threads, scale=120.0)  # soft volume, not pass/fail

    # Activity persistence across months from issue/PR updated_at.
    months = set()
    for it in issues + prs:
        u = parse_iso8601(it.get("updated_at"))
        if u is not None:
            months.add(f"{u.year:04d}-{u.month:02d}")
    ca_persistence = ratio01(len(months), 12)

    block = mean([ca_discussion_depth, ca_response_coverage, ca_thread_flow, ca_persistence])
    return block, {
        "available": True,
        "block": block,
        "sub": {
            "CA_discussion_depth": ca_discussion_depth,
            "CA_response_coverage": ca_response_coverage,
            "CA_thread_flow": ca_thread_flow,
            "CA_activity_persistence": ca_persistence,
        },
        "raw": {
            "issues_12m_sampled": len(issues),
            "prs_12m_sampled": len(prs),
            "total_threads_12m": total_threads,
            "avg_comments_per_thread": avg_comments,
            "threads_with_comments": with_comments,
            "active_discussion_months": len(months),
        },
        "missingness_note": "GitHub API availability affects issue/PR communication signals; failed list fetch yields unavailable (None), while empty lists are treated as observed low thread volume.",
    }


def compute_cei(
    repo_path: Path,
    repo_url: Optional[str] = None,
    github_token: Optional[str] = None,
    learned_weights: Optional[Dict[str, float]] = None,
    calibration: Optional[Dict[str, Any]] = None,
    aggregation: str = "geometric",
) -> Dict[str, Any]:
    n_contrib, contrib_d = compute_contributor_engagement(repo_path)
    n_user, user_d = compute_user_engagement(repo_path, repo_url, github_token, calibration=calibration)
    n_comm, comm_d = compute_communication_activity(repo_url, github_token)
    blocks = {
        "contributor_engagement": n_contrib,
        "user_engagement": n_user,
        "communication_activity": n_comm,
    }
    score, weights, weights_source, agg_name = final_score_from_blocks(blocks, learned_weights, aggregation)
    return {
        "repository_path": str(repo_path),
        "repository_url": repo_url,
        "score": score,
        "band": band(score),
        "aggregation": agg_name,
        "details": {
            "CEI": score,
            "N_contributor_engagement": n_contrib,
            "N_user_engagement": n_user,
            "N_communication_activity": n_comm,
            "weights": weights,
            "weights_source": weights_source,
            "blocks": {
                "contributor_engagement": contrib_d,
                "user_engagement": user_d,
                "communication_activity": comm_d,
            },
            "method_note": "CEI focuses on engagement and communication, not general project health or code quality.",
        },
    }


# =========================================================
# AE.2 EVS — Ecosystem Vitality Score
# =========================================================

def compute_dependent_projects(repo_path: Path, repo_url: Optional[str], token: Optional[str], calibration: Optional[Dict[str, Any]]) -> Tuple[Optional[float], Dict[str, Any]]:
    """Dependent projects are difficult to fetch reliably from GitHub REST; use registry and repo/package evidence."""
    info = github_repo_info(repo_url, token) if repo_url else None
    forks = info.get("forks_count") if isinstance(info, dict) else None
    forks_score = percentile_score(forks, (calibration or {}).get("forks_total") if isinstance(calibration, dict) else None)

    registries = package_registry_evidence(repo_path)
    packages_detected = sum(len(v) for v in registries["package_names"].values())
    dp_package_published = sat_exp(packages_detected, scale=1.0)

    version_counts = []
    for x in registries["pypi"]:
        if isinstance(x, dict) and x.get("release_count") is not None:
            version_counts.append(int(x["release_count"]))
    for x in registries["npm"]:
        if isinstance(x, dict) and x.get("version_count") is not None:
            version_counts.append(int(x["version_count"]))
    dp_package_maturity = sat_exp(sum(version_counts), scale=10.0) if version_counts else None

    texts = candidate_texts(repo_path)
    dependent_hits = count_regex_hits_in_texts(texts, [
        r"\bdependent project(?:s)?\b", r"\bused by\b", r"\bdownstream\b", r"\bplugin(?:s)?\b",
        r"\bextension(?:s)?\b", r"\bpackage ecosystem\b", r"\bimport(?:ed)? by\b",
    ])
    dp_downstream_docs = _log_repo_signal(dependent_hits, max(1.0, dependent_hits))

    block = mean([forks_score, dp_package_published, dp_package_maturity, dp_downstream_docs])
    return block, {
        "available": True,
        "block": block,
        "sub": {
            "DP_forks_calibrated": forks_score,
            "DP_package_published": dp_package_published,
            "DP_package_maturity": dp_package_maturity,
            "DP_downstream_docs": dp_downstream_docs,
        },
        "raw": {
            "forks_total": forks,
            "package_registry_evidence": registries,
            "dependent_keyword_hits": dependent_hits,
        },
        "missingness_note": "GitHub dependent graph is not available in the REST API; package/registry/downstream documentation evidence is used instead.",
    }


def compute_citations_usage(repo_path: Path, calibration: Optional[Dict[str, Any]]) -> Tuple[Optional[float], Dict[str, Any]]:
    meta = extract_publication_metadata(repo_path)
    external = crossref_citation_counts(meta["dois"])
    cu_identifier_presence = mean([sat_exp(len(meta["dois"]), scale=1.0), sat_exp(len(meta["arxivs"]), scale=1.0)])
    cu_bibliography = sat_exp(meta["bib_entries"], scale=1.0) if meta["bib_files"] else 0.0
    cu_external_citations = percentile_score(external.get("is_referenced_by_total"), (calibration or {}).get("crossref_citations") if isinstance(calibration, dict) else None)
    texts = candidate_texts(repo_path)
    usage_hits = count_regex_hits_in_texts(texts, [
        r"\bcited by\b", r"\bcitation(?:s)?\b", r"\bpublication(?:s)?\b", r"\bcase stud(?:y|ies)\b",
        r"\bused in\b", r"\bapplication(?:s)?\b", r"\breal[- ]world\b",
    ])
    cu_usage_mentions = _log_repo_signal(usage_hits, max(1.0, usage_hits))
    block = mean([cu_identifier_presence, cu_bibliography, external["external_metadata_available"], cu_external_citations, cu_usage_mentions])
    return block, {
        "available": True,
        "block": block,
        "sub": {
            "CU_identifier_presence": cu_identifier_presence,
            "CU_bibliography": cu_bibliography,
            "CU_external_metadata": external["external_metadata_available"],
            "CU_external_citations_calibrated": cu_external_citations,
            "CU_usage_mentions": cu_usage_mentions,
        },
        "raw": {
            "doi_count": len(meta["dois"]),
            "dois_sample": meta["dois"][:20],
            "arxiv_ids": meta["arxivs"],
            "bib_entries": meta["bib_entries"],
            "citation_files": [str(p.relative_to(repo_path)) for p in meta["citation_files"]],
            "external": external,
            "usage_keyword_hits": usage_hits,
        },
        "missingness_note": "Citation impact is scored only with a supplied calibration distribution; otherwise raw citation counts are reported.",
    }


def compute_integration_ecosystem(repo_path: Path, repo_url: Optional[str], token: Optional[str]) -> Tuple[Optional[float], Dict[str, Any]]:
    texts = candidate_texts(repo_path)
    topics = github_topics(repo_url, token) if repo_url else []

    integration_patterns = [
        r"\bintegration(?:s)?\b", r"\bconnector(?:s)?\b", r"\bplugin(?:s)?\b", r"\bextension(?:s)?\b",
        r"\bAPI\b", r"\bimporter(?:s)?\b", r"\bexporter(?:s)?\b", r"\bconverter(?:s)?\b", r"\badapter(?:s)?\b",
        r"\binterface(?:s)?\b", r"\bcompatible with\b", r"\bsupports .* format",
    ]
    data_format_patterns = [r"\bcsv\b", r"\bjson\b", r"\bhdf5\b", r"\bnetcdf\b", r"\bexcel\b", r"\bxml\b", r"\bparquet\b", r"\bgeojson\b"]
    tool_patterns = [r"\bpandas\b", r"\bnumpy\b", r"\bscipy\b", r"\bxarray\b", r"\bmatplotlib\b", r"\bplotly\b", r"\bjupyter\b", r"\bdocker\b", r"\bqgis\b"]

    integration_hits = count_regex_hits_in_texts(texts, integration_patterns)
    format_hits = count_regex_hits_in_texts(texts, data_format_patterns)
    tool_hits = count_regex_hits_in_texts(texts, tool_patterns)
    if topics is None:
        ref = repo_hit_ref(integration_hits, format_hits, tool_hits, 0)
        ie_topic_signal = None
        topic_hits = 0
        family_presence = sum(1 for x in [integration_hits, format_hits, tool_hits] if x and x > 0)
        ie_family_diversity = ratio01(family_presence, 3)
    else:
        topic_hits = sum(1 for t in topics if any(k in t for k in ["plugin", "integration", "api", "connector", "data", "jupyter"]))
        ref = repo_hit_ref(integration_hits, format_hits, tool_hits, topic_hits)
        ie_topic_signal = _log_repo_signal(topic_hits, ref)
        family_presence = sum(1 for x in [integration_hits, format_hits, tool_hits, topic_hits] if x and x > 0)
        ie_family_diversity = ratio01(family_presence, 4)

    ie_integration_docs = _log_repo_signal(integration_hits, ref)
    ie_format_breadth = _log_repo_signal(format_hits, ref)
    ie_tooling_breadth = _log_repo_signal(tool_hits, ref)

    block = mean([ie_integration_docs, ie_format_breadth, ie_tooling_breadth, ie_topic_signal, ie_family_diversity])
    return block, {
        "available": True,
        "block": block,
        "sub": {
            "IE_integration_docs": ie_integration_docs,
            "IE_format_breadth": ie_format_breadth,
            "IE_tooling_breadth": ie_tooling_breadth,
            "IE_topic_signal": ie_topic_signal,
            "IE_family_diversity": ie_family_diversity,
        },
        "raw": {
            "integration_hits": integration_hits,
            "format_hits": format_hits,
            "tool_hits": tool_hits,
            "topic_hits": topic_hits,
            "github_topics": topics,
            "hit_ref": ref,
        },
    }


def compute_evs(
    repo_path: Path,
    repo_url: Optional[str] = None,
    github_token: Optional[str] = None,
    learned_weights: Optional[Dict[str, float]] = None,
    calibration: Optional[Dict[str, Any]] = None,
    aggregation: str = "geometric",
) -> Dict[str, Any]:
    n_dep, dep_d = compute_dependent_projects(repo_path, repo_url, github_token, calibration)
    n_cit, cit_d = compute_citations_usage(repo_path, calibration)
    n_int, int_d = compute_integration_ecosystem(repo_path, repo_url, github_token)
    blocks = {
        "dependent_projects": n_dep,
        "citations_usage": n_cit,
        "integration_ecosystem": n_int,
    }
    score, weights, weights_source, agg_name = final_score_from_blocks(blocks, learned_weights, aggregation)
    return {
        "repository_path": str(repo_path),
        "repository_url": repo_url,
        "score": score,
        "band": band(score),
        "aggregation": agg_name,
        "details": {
            "EVS": score,
            "N_dependent_projects": n_dep,
            "N_citations_usage": n_cit,
            "N_integration_ecosystem": n_int,
            "weights": weights,
            "weights_source": weights_source,
            "blocks": {
                "dependent_projects": dep_d,
                "citations_usage": cit_d,
                "integration_ecosystem": int_d,
            },
            "method_note": "EVS measures external embedding of the project in a wider software/scientific ecosystem.",
        },
    }


# =========================================================
# AE.3 RWIS — Real-World Impact Score
# =========================================================

USE_CASE_PATTERNS = [
    r"\buse case(?:s)?\b", r"\bcase stud(?:y|ies)\b", r"\bexample application(?:s)?\b",
    r"\bdemonstration(?:s)?\b", r"\bapplied to\b", r"\bapplication(?:s)?\b",
    r"\bscenario(?:s)?\b", r"\bpilot(?:s)?\b", r"\bdeployment(?:s)?\b",
]
EDUCATION_PATTERNS = [
    r"\bteaching\b", r"\bcourse(?:s)?\b", r"\blecture(?:s)?\b", r"\btutorial(?:s)?\b",
    r"\bworkshop(?:s)?\b", r"\bsummer school\b", r"\buniversity\b", r"\bmaster'?s?\b",
    r"\bphd\b", r"\bstudent(?:s)?\b", r"\bclassroom\b",
]
INDUSTRY_PATTERNS = [
    r"\bindustry\b", r"\butility\b", r"\butilities\b", r"\boperator(?:s)?\b", r"\bTSO\b", r"\bDSO\b",
    r"\bconsult(?:ing|ancy|ant)?\b", r"\bcommercial\b", r"\bcompany\b", r"\bgovernment\b", r"\bministry\b",
    r"\bregulator\b", r"\benergy agency\b", r"\btransmission operator\b", r"\bdistribution operator\b",
]


def compute_use_case_score(repo_path: Path) -> Tuple[Optional[float], Dict[str, Any]]:
    texts = candidate_texts(repo_path)
    use_case_hits = count_regex_hits_in_texts(texts, USE_CASE_PATTERNS)
    path_hits = count_path_hits(repo_path, [r"use[-_ ]?cases?", r"case[-_ ]?stud", r"examples?", r"demos?", r"scenarios?"])
    evidence_files = []
    for p in iter_files(repo_path):
        rel = str(p.relative_to(repo_path)).replace("\\", "/").lower()
        if p.suffix.lower() in {".md", ".rst", ".txt", ".ipynb"} and re.search(r"use[-_ ]?cases?|case[-_ ]?stud|scenario|demo", rel):
            evidence_files.append(p)

    ref = repo_hit_ref(use_case_hits, path_hits, len(evidence_files))
    uc_docs = _log_repo_signal(use_case_hits, ref)
    uc_path_evidence = _log_repo_signal(path_hits, ref)
    uc_file_evidence = _log_repo_signal(len(evidence_files), ref)
    uc_examples_dir = has_dir(repo_path, ["examples", "example", "demos", "demo", "case_studies", "use_cases", "scenarios"])

    block = mean([uc_docs, uc_path_evidence, uc_file_evidence, uc_examples_dir])
    return block, {
        "available": True,
        "block": block,
        "sub": {
            "UC_documented_use_cases": uc_docs,
            "UC_path_evidence": uc_path_evidence,
            "UC_file_evidence": uc_file_evidence,
            "UC_examples_directory": uc_examples_dir,
        },
        "raw": {
            "use_case_keyword_hits": use_case_hits,
            "use_case_path_hits": path_hits,
            "use_case_files_count": len(evidence_files),
            "use_case_files_sample": [str(p.relative_to(repo_path)) for p in evidence_files[:50]],
            "hit_ref": ref,
        },
    }


def compute_educational_use(repo_path: Path) -> Tuple[Optional[float], Dict[str, Any]]:
    texts = candidate_texts(repo_path)
    edu_hits = count_regex_hits_in_texts(texts, EDUCATION_PATTERNS)
    notebook_count = sum(1 for p in iter_files(repo_path) if p.suffix.lower() == ".ipynb")
    tutorial_dirs = has_dir(repo_path, ["tutorials", "tutorial", "notebooks", "examples", "lectures", "workshops"])
    teaching_files = []
    for p in iter_files(repo_path):
        rel = str(p.relative_to(repo_path)).replace("\\", "/").lower()
        if p.suffix.lower() in {".md", ".rst", ".txt", ".ipynb"} and re.search(r"tutorial|teaching|course|lecture|workshop|notebook", rel):
            teaching_files.append(p)
    ref = repo_hit_ref(edu_hits, notebook_count, len(teaching_files))
    eu_docs = _log_repo_signal(edu_hits, ref)
    eu_notebooks = _log_repo_signal(notebook_count, ref)
    eu_teaching_files = _log_repo_signal(len(teaching_files), ref)
    eu_tutorial_dirs = tutorial_dirs
    block = mean([eu_docs, eu_notebooks, eu_teaching_files, eu_tutorial_dirs])
    return block, {
        "available": True,
        "block": block,
        "sub": {
            "EU_education_docs": eu_docs,
            "EU_notebooks": eu_notebooks,
            "EU_teaching_files": eu_teaching_files,
            "EU_tutorial_dirs": eu_tutorial_dirs,
        },
        "raw": {
            "education_keyword_hits": edu_hits,
            "notebook_count": notebook_count,
            "teaching_files_count": len(teaching_files),
            "teaching_files_sample": [str(p.relative_to(repo_path)) for p in teaching_files[:50]],
            "hit_ref": ref,
        },
    }


def compute_industry_adoption(repo_path: Path) -> Tuple[Optional[float], Dict[str, Any]]:
    texts = candidate_texts(repo_path)
    industry_hits = count_regex_hits_in_texts(texts, INDUSTRY_PATTERNS)
    # Count distinct organization-like named mentions in user/adoption files if present.
    adoption_files = []
    for p in iter_files(repo_path):
        name = p.name.lower()
        rel = str(p.relative_to(repo_path)).replace("\\", "/").lower()
        if name in {"users.md", "adopters.md", "case_studies.md", "use_cases.md", "acknowledgements", "acknowledgments"} or re.search(r"users|adopters|industry|case[-_ ]stud", rel):
            if p.suffix.lower() in {".md", ".rst", ".txt", ".csv"} or name in {"acknowledgements", "acknowledgments"}:
                adoption_files.append(p)
    organization_markers = count_regex_hits_in_texts({str(p): _safe_read_text(p) or "" for p in adoption_files}, [
        r"\b(?:Inc\.|Ltd\.|LLC|GmbH|SAS|SA|AB|Corp\.|Corporation|Company|University|Institute|Agency|Ministry|Operator)\b",
        r"\b[A-Z][A-Za-z0-9&\-.]+\s+(?:Energy|Power|Grid|Electric|Utility|Utilities|Networks|Transmission|Distribution)\b",
    ])
    ref = repo_hit_ref(industry_hits, len(adoption_files), organization_markers)
    ia_industry_docs = _log_repo_signal(industry_hits, ref)
    ia_adoption_files = _log_repo_signal(len(adoption_files), ref)
    ia_org_mentions = _log_repo_signal(organization_markers, ref)
    block = mean([ia_industry_docs, ia_adoption_files, ia_org_mentions])
    return block, {
        "available": True,
        "block": block,
        "sub": {
            "IA_industry_docs": ia_industry_docs,
            "IA_adoption_files": ia_adoption_files,
            "IA_organization_mentions": ia_org_mentions,
        },
        "raw": {
            "industry_keyword_hits": industry_hits,
            "adoption_files_count": len(adoption_files),
            "adoption_files_sample": [str(p.relative_to(repo_path)) for p in adoption_files[:50]],
            "organization_marker_hits": organization_markers,
            "hit_ref": ref,
        },
        "missingness_note": "Industry adoption is based on documented evidence inside the repository; it does not infer private commercial use.",
    }


def compute_rwis(
    repo_path: Path,
    repo_url: Optional[str] = None,
    github_token: Optional[str] = None,
    learned_weights: Optional[Dict[str, float]] = None,
    calibration: Optional[Dict[str, Any]] = None,
    aggregation: str = "geometric",
) -> Dict[str, Any]:
    n_use, use_d = compute_use_case_score(repo_path)
    n_edu, edu_d = compute_educational_use(repo_path)
    n_ind, ind_d = compute_industry_adoption(repo_path)
    blocks = {
        "use_case_score": n_use,
        "educational_use": n_edu,
        "industry_adoption": n_ind,
    }
    score, weights, weights_source, agg_name = final_score_from_blocks(blocks, learned_weights, aggregation)
    return {
        "repository_path": str(repo_path),
        "repository_url": repo_url,
        "score": score,
        "band": band(score),
        "aggregation": agg_name,
        "details": {
            "RWIS": score,
            "N_use_case_score": n_use,
            "N_educational_use": n_edu,
            "N_industry_adoption": n_ind,
            "weights": weights,
            "weights_source": weights_source,
            "blocks": {
                "use_case_score": use_d,
                "educational_use": edu_d,
                "industry_adoption": ind_d,
            },
            "method_note": "RWIS measures documented real-world use evidence, not generic GitHub popularity.",
        },
    }


# =========================================================
# AE.4 AQS — Adoption Quality Score
# =========================================================

MISSION_PATTERNS = [
    r"\bcritical infrastructure\b", r"\bmission[- ]critical\b", r"\boperational\b", r"\breal[- ]world\b",
    r"\bproduction\b", r"\bdeployment\b", r"\bgrid operation\b", r"\bpower system operation\b",
    r"\bplanning study\b", r"\bpolicy analysis\b", r"\bregulatory\b", r"\bsecurity of supply\b",
    r"\bresilience\b", r"\bdecarboni[sz]ation\b", r"\breliability\b",
]
REPEAT_USE_PATTERNS = [
    r"\bongoing\b", r"\brepeated\b", r"\bcontinued use\b", r"\blong[- ]term\b", r"\bannual\b", r"\brecurring\b",
    r"\bmultiple projects\b", r"\bseveral studies\b", r"\bused across\b", r"\bworkflow\b", r"\bpipeline\b",
]
SECTOR_PATTERNS = {
    "academia": [r"\buniversity\b", r"\binstitute\b", r"\blaborator(?:y|ies)\b", r"\bresearch\b"],
    "industry": [r"\bindustry\b", r"\bcompany\b", r"\bconsult(?:ing|ancy)?\b", r"\bcommercial\b"],
    "utility_operator": [r"\butility\b", r"\boperator\b", r"\bTSO\b", r"\bDSO\b", r"\btransmission\b", r"\bdistribution\b"],
    "government": [r"\bgovernment\b", r"\bministry\b", r"\bagency\b", r"\bregulator\b", r"\bpublic authority\b"],
    "education": [r"\bteaching\b", r"\bcourse\b", r"\bworkshop\b", r"\bstudent\b"],
    "community": [r"\bcommunity\b", r"\bopen source\b", r"\busers\b", r"\badopters\b"],
}
GEOGRAPHIC_PATTERNS = [
    r"\bEurope(?:an)?\b", r"\bEU\b", r"\bUnited States\b", r"\bUS\b", r"\bChina\b", r"\bIndia\b", r"\bAfrica\b",
    r"\bAsia\b", r"\bNordic\b", r"\bGermany\b", r"\bFrance\b", r"\bUK\b", r"\bSweden\b", r"\bglobal\b",
]


def compute_mission_relevance(repo_path: Path) -> Tuple[Optional[float], Dict[str, Any]]:
    texts = candidate_texts(repo_path)
    mission_hits = count_regex_hits_in_texts(texts, MISSION_PATTERNS)
    use_case_hits = count_regex_hits_in_texts(texts, USE_CASE_PATTERNS)
    # Mission relevance is stronger when consequential-context mentions co-occur with use-case evidence.
    mr_context = _log_repo_signal(mission_hits, max(1.0, mission_hits))
    mr_use_context = _log_repo_signal(min(mission_hits, use_case_hits), max(1.0, max(mission_hits, use_case_hits)))
    topic_files = count_path_hits(repo_path, [r"critical", r"operational", r"deployment", r"resilience", r"planning", r"policy", r"reliability"])
    mr_path_context = _log_repo_signal(topic_files, max(1.0, topic_files))
    block = mean([mr_context, mr_use_context, mr_path_context])
    return block, {
        "available": True,
        "block": block,
        "sub": {
            "MR_consequential_context": mr_context,
            "MR_use_case_context_overlap": mr_use_context,
            "MR_path_context": mr_path_context,
        },
        "raw": {
            "mission_keyword_hits": mission_hits,
            "use_case_keyword_hits": use_case_hits,
            "mission_path_hits": topic_files,
        },
    }


def compute_repeat_use(repo_path: Path, repo_url: Optional[str], token: Optional[str]) -> Tuple[Optional[float], Dict[str, Any]]:
    texts = candidate_texts(repo_path)
    repeat_hits = count_regex_hits_in_texts(texts, REPEAT_USE_PATTERNS)
    ru_repeat_docs = _log_repo_signal(repeat_hits, max(1.0, repeat_hits))

    # Repeated external use proxy: issue/PR communication persistence and release/version maturity.
    issues = github_issues_or_prs(repo_url, token, kind="issues", months=24, max_pages=10) if repo_url else []
    active_months = set()
    unique_external_authors = set()
    owner, repo = parse_github_owner_repo(repo_url)
    if repo_url and issues is not None:
        for it in issues:
            u = parse_iso8601(it.get("updated_at"))
            if u:
                active_months.add(f"{u.year:04d}-{u.month:02d}")
            user = it.get("user") if isinstance(it.get("user"), dict) else None
            login = user.get("login") if isinstance(user, dict) else None
            if login and login.lower() not in {str(owner or "").lower(), "dependabot[bot]", "github-actions[bot]"}:
                unique_external_authors.add(login)
    if repo_url and issues is None:
        ru_issue_persistence = None
        ru_external_user_breadth = None
    elif repo_url:
        ru_issue_persistence = ratio01(len(active_months), 24)
        ru_external_user_breadth = sat_exp(len(unique_external_authors), scale=10.0)
    else:
        ru_issue_persistence = None
        ru_external_user_breadth = None

    registries = package_registry_evidence(repo_path)
    release_counts = []
    for x in registries["pypi"]:
        if isinstance(x, dict) and x.get("release_count") is not None:
            release_counts.append(int(x["release_count"]))
    for x in registries["npm"]:
        if isinstance(x, dict) and x.get("version_count") is not None:
            release_counts.append(int(x["version_count"]))
    ru_registry_versioning = sat_exp(sum(release_counts), scale=10.0) if release_counts else None

    block = mean([ru_repeat_docs, ru_issue_persistence, ru_external_user_breadth, ru_registry_versioning])
    return block, {
        "available": True,
        "block": block,
        "sub": {
            "RU_repeat_docs": ru_repeat_docs,
            "RU_issue_persistence": ru_issue_persistence,
            "RU_external_user_breadth": ru_external_user_breadth,
            "RU_registry_versioning": ru_registry_versioning,
        },
        "raw": {
            "repeat_keyword_hits": repeat_hits,
            "issue_active_months_24m": len(active_months),
            "unique_external_issue_authors_24m": len(unique_external_authors),
            "registry_release_or_version_counts": release_counts,
        },
        "missingness_note": "Repeat-use proxies combine documentation, GitHub issue persistence, and package version history when available.",
    }


def compute_diversity_of_use(repo_path: Path, repo_url: Optional[str], token: Optional[str]) -> Tuple[Optional[float], Dict[str, Any]]:
    texts = candidate_texts(repo_path)
    blob = "\n".join(texts.values())
    sector_counts = {}
    for sector, pats in SECTOR_PATTERNS.items():
        sector_counts[sector] = count_regex_hits_in_texts({"blob": blob}, pats)
    sector_present = [k for k, v in sector_counts.items() if v > 0]
    du_sector_diversity = ratio01(len(sector_present), len(SECTOR_PATTERNS))
    du_sector_entropy = entropy_normalized([v for v in sector_counts.values() if v > 0])

    geo_hits = count_regex_hits_in_texts({"blob": blob}, GEOGRAPHIC_PATTERNS)
    du_geo_breadth = sat_exp(geo_hits, scale=5.0)

    topics = github_topics(repo_url, token) if repo_url else []
    topic_families = set()
    if topics is not None:
        for t in topics:
            if any(k in t for k in ["energy", "power", "grid", "electric"]):
                topic_families.add("energy")
            if any(k in t for k in ["optimization", "simulation", "forecast", "planning"]):
                topic_families.add("method")
            if any(k in t for k in ["python", "julia", "data", "api"]):
                topic_families.add("technical")
            if any(k in t for k in ["education", "research", "academic"]):
                topic_families.add("research_education")
    if topics is None:
        du_topic_diversity = None
    else:
        du_topic_diversity = ratio01(len(topic_families), 4)

    block = mean([du_sector_diversity, du_sector_entropy, du_geo_breadth, du_topic_diversity])
    return block, {
        "available": True,
        "block": block,
        "sub": {
            "DU_sector_diversity": du_sector_diversity,
            "DU_sector_entropy": du_sector_entropy,
            "DU_geographic_breadth": du_geo_breadth,
            "DU_topic_diversity": du_topic_diversity,
        },
        "raw": {
            "sector_counts": sector_counts,
            "sector_families_present": sector_present,
            "geographic_keyword_hits": geo_hits,
            "github_topics": topics,
            "topic_families": sorted(topic_families),
        },
    }


def compute_aqs(
    repo_path: Path,
    repo_url: Optional[str] = None,
    github_token: Optional[str] = None,
    learned_weights: Optional[Dict[str, float]] = None,
    calibration: Optional[Dict[str, Any]] = None,
    aggregation: str = "geometric",
) -> Dict[str, Any]:
    n_mr, mr_d = compute_mission_relevance(repo_path)
    n_ru, ru_d = compute_repeat_use(repo_path, repo_url, github_token)
    n_du, du_d = compute_diversity_of_use(repo_path, repo_url, github_token)
    blocks = {
        "mission_relevance": n_mr,
        "repeat_use": n_ru,
        "diversity_of_use": n_du,
    }
    score, weights, weights_source, agg_name = final_score_from_blocks(blocks, learned_weights, aggregation)
    return {
        "repository_path": str(repo_path),
        "repository_url": repo_url,
        "score": score,
        "band": band(score),
        "aggregation": agg_name,
        "details": {
            "AQS": score,
            "N_mission_relevance": n_mr,
            "N_repeat_use": n_ru,
            "N_diversity_of_use": n_du,
            "weights": weights,
            "weights_source": weights_source,
            "blocks": {
                "mission_relevance": mr_d,
                "repeat_use": ru_d,
                "diversity_of_use": du_d,
            },
            "method_note": "AQS de-noises adoption by emphasizing depth, persistence, and diversity rather than vanity metrics alone.",
        },
    }


# =========================================================
# AE category aggregator
# =========================================================

def compute_ae(
    repo_path: Path,
    repo_url: Optional[str] = None,
    github_token: Optional[str] = None,
    learned_weights: Optional[Dict[str, float]] = None,
    calibration: Optional[Dict[str, Any]] = None,
    aggregation: str = "geometric",
) -> Dict[str, Any]:
    cei = compute_cei(repo_path, repo_url, github_token, learned_weights=None, calibration=calibration, aggregation=aggregation)
    evs = compute_evs(repo_path, repo_url, github_token, learned_weights=None, calibration=calibration, aggregation=aggregation)
    rwis = compute_rwis(repo_path, repo_url, github_token, learned_weights=None, calibration=calibration, aggregation=aggregation)
    aqs = compute_aqs(repo_path, repo_url, github_token, learned_weights=None, calibration=calibration, aggregation=aggregation)

    blocks = {
        "CEI": cei.get("score") / 100.0 if cei.get("score") is not None else None,
        "EVS": evs.get("score") / 100.0 if evs.get("score") is not None else None,
        "RWIS": rwis.get("score") / 100.0 if rwis.get("score") is not None else None,
        "AQS": aqs.get("score") / 100.0 if aqs.get("score") is not None else None,
    }
    score, weights, weights_source, agg_name = final_score_from_blocks(blocks, learned_weights, aggregation)
    return {
        "repository_path": str(repo_path),
        "repository_url": repo_url,
        "score": score,
        "band": band(score),
        "aggregation": agg_name,
        "details": {
            "AE": score,
            "CEI": cei.get("score"),
            "EVS": evs.get("score"),
            "RWIS": rwis.get("score"),
            "AQS": aqs.get("score"),
            "weights": weights,
            "weights_source": weights_source,
            "scores": {"CEI": cei, "EVS": evs, "RWIS": rwis, "AQS": aqs},
            "missingness_note": "Unavailable values remain None and are excluded with weight renormalization; observable absence is scored as 0.0.",
            "method_note": "AE evaluates adoption and ecosystem embedding; it deliberately separates engagement, external ecosystem, real-world impact, and adoption quality to reduce redundancy.",
        },
    }


# =========================================================
# Corpus weight-learning entry points
# =========================================================

def learn_cei_weights(dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = ["contributor_engagement", "user_engagement", "communication_activity"]
    rows = [{k: row.get(k, row.get("N_" + k)) for k in keys} for row in dataset]
    return {"weights": norm_weights_from_rows(rows, keys), "n_repos": len(dataset)}


def learn_evs_weights(dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = ["dependent_projects", "citations_usage", "integration_ecosystem"]
    rows = [{k: row.get(k, row.get("N_" + k)) for k in keys} for row in dataset]
    return {"weights": norm_weights_from_rows(rows, keys), "n_repos": len(dataset)}


def learn_rwis_weights(dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = ["use_case_score", "educational_use", "industry_adoption"]
    rows = [{k: row.get(k, row.get("N_" + k)) for k in keys} for row in dataset]
    return {"weights": norm_weights_from_rows(rows, keys), "n_repos": len(dataset)}


def learn_aqs_weights(dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = ["mission_relevance", "repeat_use", "diversity_of_use"]
    rows = [{k: row.get(k, row.get("N_" + k)) for k in keys} for row in dataset]
    return {"weights": norm_weights_from_rows(rows, keys), "n_repos": len(dataset)}


def learn_ae_weights(dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = ["CEI", "EVS", "RWIS", "AQS"]
    rows = [{k: row.get(k) for k in keys} for row in dataset]
    return {"weights": norm_weights_from_rows(rows, keys), "n_repos": len(dataset)}


# =========================================================
# CLI
# =========================================================

def load_json_file(path: Optional[str]) -> Optional[Any]:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def split_weights_for_score(score: str, loaded: Optional[Dict[str, Any]]) -> Optional[Dict[str, float]]:
    if not isinstance(loaded, dict):
        return None
    if "weights" in loaded and isinstance(loaded["weights"], dict):
        return loaded["weights"]
    # Allow one JSON file with named sections.
    section = loaded.get(score.upper()) or loaded.get(score.lower())
    if isinstance(section, dict):
        return section.get("weights", section)
    return loaded


def main() -> None:
    parser = argparse.ArgumentParser(description="RECOPS Adoption & Ecosystem scores: CEI, EVS, RWIS, AQS, AE")
    parser.add_argument("repo_url", nargs="?", help="GitHub repository URL, e.g. https://github.com/owner/repo")
    parser.add_argument("--local", dest="local_repo", help="Analyze an already-local repository path instead of cloning")
    parser.add_argument("--token", dest="github_token", default=os.environ.get("GITHUB_TOKEN"), help="GitHub token; defaults to env GITHUB_TOKEN")
    parser.add_argument("--score", choices=["cei", "evs", "rwis", "aqs", "ae", "all"], default="all", help="Which AE score to compute")
    parser.add_argument("--weights-json", dest="weights_json", help="Optional learned weights JSON for the selected score")
    parser.add_argument("--calibration-json", dest="calibration_json", help="Optional corpus calibration JSON for popularity/citation/download percentiles")
    parser.add_argument("--aggregation", choices=["geometric", "sum"], default="geometric", help="Final aggregation; geometric is recommended")
    parser.add_argument("--out", dest="out", help="Optional output JSON path")
    parser.add_argument("--indent", type=int, default=2, help="JSON indentation")
    args = parser.parse_args()

    if not args.local_repo and not args.repo_url:
        parser.error("Provide either a GitHub repo_url or --local PATH")

    if args.local_repo:
        repo_path = Path(args.local_repo).expanduser().resolve()
        if not repo_path.exists() or not repo_path.is_dir():
            raise SystemExit(f"Local repository path does not exist or is not a directory: {repo_path}")
    else:
        repo_path = clone_or_download(args.repo_url)

    weights_loaded = load_json_file(args.weights_json) if args.weights_json else None
    calibration = load_json_file(args.calibration_json) if args.calibration_json else None
    if calibration is not None and not isinstance(calibration, dict):
        raise SystemExit("--calibration-json must contain a JSON object")

    sc = args.score.lower()
    if sc == "cei":
        result = compute_cei(repo_path, args.repo_url, args.github_token, split_weights_for_score("cei", weights_loaded), calibration, args.aggregation)
    elif sc == "evs":
        result = compute_evs(repo_path, args.repo_url, args.github_token, split_weights_for_score("evs", weights_loaded), calibration, args.aggregation)
    elif sc == "rwis":
        result = compute_rwis(repo_path, args.repo_url, args.github_token, split_weights_for_score("rwis", weights_loaded), calibration, args.aggregation)
    elif sc == "aqs":
        result = compute_aqs(repo_path, args.repo_url, args.github_token, split_weights_for_score("aqs", weights_loaded), calibration, args.aggregation)
    elif sc == "ae":
        result = compute_ae(repo_path, args.repo_url, args.github_token, split_weights_for_score("ae", weights_loaded), calibration, args.aggregation)
    else:
        result = {
            "repository_path": str(repo_path),
            "repository_url": args.repo_url,
            "scores": {
                "CEI": compute_cei(repo_path, args.repo_url, args.github_token, None, calibration, args.aggregation),
                "EVS": compute_evs(repo_path, args.repo_url, args.github_token, None, calibration, args.aggregation),
                "RWIS": compute_rwis(repo_path, args.repo_url, args.github_token, None, calibration, args.aggregation),
                "AQS": compute_aqs(repo_path, args.repo_url, args.github_token, None, calibration, args.aggregation),
                "AE": compute_ae(repo_path, args.repo_url, args.github_token, None, calibration, args.aggregation),
            },
            "note": "With --score all, --weights-json is ignored to avoid applying one weight file to multiple formulas.",
        }

    txt = json.dumps(result, indent=args.indent, ensure_ascii=False, sort_keys=False)
    if args.out:
        Path(args.out).write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
