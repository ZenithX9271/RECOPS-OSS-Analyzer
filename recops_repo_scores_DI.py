#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RECOPS Repository Scorer v2 — Deployability & Interoperability (DI)

Computes the four DI indicators from the RECOPS v2 architecture:
- DRS — Deployment Readiness Score
- ICI — Integration Capability Index
- PFS — Platform Flexibility Score
- OBS — Observability Score
- DI  — Deployability & Interoperability aggregate

Design choices
--------------
- No fixed formula weights such as 0.35/0.30/0.20/0.15 inside indicators.
- Each formula is decomposed into non-redundant evidence blocks.
- Default aggregation is weighted_geometric with data-driven weights.
- Block weights can be learned from a corpus; otherwise a per-repository
  dispersion fallback is used.
- Missing data are kept as None and excluded with weight renormalization.
- Observable absence is scored as 0.0.
- Signals are extracted from repository files, configuration files, CI matrices,
  package metadata, container/deployment artifacts, docs, and optional GitHub API data.

Scientific interpretation
-------------------------
DI measures whether software can realistically be installed, configured,
integrated, deployed across environments, and operated with observability.
It deliberately avoids re-scoring general code quality, project health,
adoption/popularity, or domain capability.

Optional dependencies
---------------------
- requests: GitHub API and fallback zip download.
- PyYAML: parsing YAML CI/deployment/config files.

Recommended installation:
    pip install requests pyyaml

Example usage:
    python recops_repo_scores_DI.py https://github.com/PyPSA/PyPSA --score all
    python recops_repo_scores_DI.py https://github.com/PyPSA/PyPSA --score drs
    python recops_repo_scores_DI.py --local ./PyPSA --score di --out di_results.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from statistics import mean as _py_mean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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
    # Adaptive floor keeps geometric non-compensation while avoiding
    # pathological collapse to near-zero in sparse evidence settings.
    positive = sorted([v for v, _ in vals if v > 0.0])
    if positive:
        floor = min(0.10, max(eps, 0.5 * positive[0]))
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
    if x is None:
        return None
    try:
        xx = max(0.0, float(x))
        ss = max(1e-9, float(scale))
    except Exception:
        return None
    return clamp01(1.0 - math.exp(-xx / ss))


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
    """Repository-relative logarithmic normalization: log(1+h) / log(1+h_ref)."""
    if hits is None or ref is None:
        return None
    try:
        h = max(0.0, float(hits))
        r = max(0.0, float(ref))
    except Exception:
        return None
    if h <= 0:
        return 0.0
    if r <= 0:
        return 0.0
    return clamp01(math.log1p(h) / math.log1p(r))


def repo_hit_ref(*counts: Optional[int]) -> float:
    return max([1.0] + [float(c) for c in counts if c is not None])


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
# Data-driven weights
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
    smooth = 0.50
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

    High-variance, well-covered, low-correlation blocks are weighted more.
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
# Repository acquisition and file scanning
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


def clone_or_download(github_url: str) -> Path:
    temp = Path(tempfile.mkdtemp(prefix="recops_di_"))
    target = temp / "repo"
    code, _, _ = run(["git", "--version"])
    if code == 0:
        c, _, _ = run(["git", "clone", "--filter=blob:none", "--tags", github_url, str(target)], cwd=temp, timeout_s=900)
        if c == 0 and target.exists():
            return target
    if requests is None:
        raise RuntimeError("Need git or requests to fetch a GitHub repository")
    owner, repo = parse_github_owner_repo(github_url)
    if not owner or not repo:
        raise ValueError("Invalid GitHub URL")
    for branch in ["main", "master"]:
        zip_url = f"https://github.com/{owner}/{repo}/archive/refs/heads/{branch}.zip"
        try:
            r = requests.get(zip_url, timeout=60)
            if r.status_code == 200:
                zpath = temp / "repo.zip"
                zpath.write_bytes(r.content)
                shutil.unpack_archive(str(zpath), str(temp))
                dirs = [p for p in temp.iterdir() if p.is_dir() and p.name != "repo"]
                if dirs:
                    return dirs[0]
        except Exception:
            pass
    raise RuntimeError("Cannot fetch repository")


IGNORE_DIRS = {
    ".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "node_modules", "dist", "build", ".venv", "venv", ".idea", ".vscode",
    "target", ".tox", ".nox",
}

TEXT_SUFFIXES = {
    ".md", ".rst", ".txt", ".toml", ".json", ".yaml", ".yml", ".ini", ".cfg",
    ".py", ".js", ".ts", ".java", ".jl", ".c", ".cpp", ".cs", ".go", ".rs",
    ".sh", ".bat", ".ps1", ".ipynb", ".xml", ".csv", ".dockerfile",
}

SOURCE_SUFFIXES = {
    ".py", ".js", ".ts", ".java", ".jl", ".c", ".cpp", ".cs", ".go", ".rs", ".rb", ".php",
    ".gms", ".mod", ".dat", ".inc", ".lp", ".mps",
}

# Below this count, per-file density ratios are treated as non-observable (None), not as evidence of absence.
MIN_SOURCE_FILES_FOR_RUNTIME_DENSITY = 8

DOC_DIR_NAMES = {
    "docs", "doc", "examples", "example", "notebooks", "tutorials", "tutorial", ".github",
    "deployment", "deploy", "operations", "ops", "config", "configs", "helm", "k8s", "kubernetes",
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


def _safe_yaml_load(path: Path) -> Optional[Any]:
    if yaml is None:
        return None
    try:
        return yaml.safe_load(Path(path).read_text(encoding="utf-8", errors="ignore"))
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
    for base, dirs, _ in os.walk(root):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        for d in dirs:
            if d.lower() in wanted:
                return 1.0
    return 0.0


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


def is_doc_like_path(root: Path, path: Path) -> bool:
    try:
        rel = path.relative_to(root)
    except Exception:
        return False
    parts = [x.lower() for x in rel.parts]
    name = path.name.lower()
    if len(parts) == 1 and (name.startswith("readme") or name in {
        "pyproject.toml", "package.json", "setup.py", "setup.cfg", "environment.yml", "requirements.txt",
        "dockerfile", "docker-compose.yml", "docker-compose.yaml", "compose.yml", "compose.yaml",
    }):
        return True
    return any(part in DOC_DIR_NAMES for part in parts[:-1]) or any(part == ".github" for part in parts[:-1])


def candidate_texts(root: Path, include_source: bool = False) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for p in iter_files(root):
        if p.suffix.lower() not in TEXT_SUFFIXES and p.name.lower() not in {"dockerfile", "makefile"}:
            continue
        if not include_source and not is_doc_like_path(root, p):
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


def files_matching(root: Path, patterns: Iterable[str], suffixes: Optional[Sequence[str]] = None) -> List[Path]:
    rx = [re.compile(p, flags=re.I) for p in patterns]
    allowed = set(s.lower() for s in suffixes) if suffixes else None
    out = []
    for p in iter_files(root):
        if allowed is not None and p.suffix.lower() not in allowed and p.name.lower() not in allowed:
            continue
        rel = str(p.relative_to(root)).replace("\\", "/")
        if any(r.search(rel) for r in rx):
            out.append(p)
    return out


# =========================================================
# GitHub API helpers
# =========================================================

def gh_api_get(url: str, token: Optional[str] = None, accept: Optional[str] = None, timeout_s: int = 30) -> Optional[Any]:
    if requests is None:
        return None
    headers = {"User-Agent": "recops-di/1.0"}
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


def github_topics(repo_url: Optional[str], token: Optional[str] = None) -> List[str]:
    owner, repo = parse_github_owner_repo(repo_url)
    if not owner or not repo:
        return []
    data = gh_api_get(f"https://api.github.com/repos/{owner}/{repo}/topics", token=token, accept="application/vnd.github+json")
    if isinstance(data, dict) and isinstance(data.get("names"), list):
        return [str(x).lower() for x in data["names"]]
    return []


# =========================================================
# Shared evidence extractors
# =========================================================

def package_metadata(root: Path) -> Dict[str, Any]:
    # Search repository-wide for dependency manifests to avoid under-counting
    # monorepos or projects that keep manifests outside the root folder.
    names = {
        "pyproject": {"pyproject.toml"},
        "setup_py": {"setup.py"},
        "setup_cfg": {"setup.cfg"},
        "requirements": {"requirements.txt", "requirements-dev.txt"},
        "environment_yml": {"environment.yml", "environment.yaml"},
        "package_json": {"package.json"},
        "project_toml": {"project.toml", "manifest.toml"},
        "cargo_toml": {"cargo.toml", "cargo.lock"},
        "go_mod": {"go.mod"},
    }
    present = {k: False for k in names}
    for p in iter_files(root):
        low = p.name.lower()
        for k, allowed in names.items():
            if low in allowed:
                present[k] = True
    package_managers = []
    if present["pyproject"] or present["setup_py"] or present["setup_cfg"]:
        package_managers.append("python_package")
    if present["requirements"] or present["environment_yml"]:
        package_managers.append("python_env")
    if present["package_json"]:
        package_managers.append("npm")
    if present["project_toml"]:
        package_managers.append("julia")
    if present["cargo_toml"]:
        package_managers.append("rust")
    if present["go_mod"]:
        package_managers.append("go")
    return {"present": present, "package_managers": sorted(set(package_managers))}


def workflow_files(root: Path) -> List[Path]:
    gh = root / ".github" / "workflows"
    if not gh.exists():
        return []
    return [p for p in gh.iterdir() if p.is_file() and p.suffix.lower() in {".yml", ".yaml"}]


def parse_ci_matrices(root: Path) -> Dict[str, Any]:
    files = workflow_files(root)
    os_tokens = Counter()
    python_versions = set()
    node_versions = set()
    julia_versions = set()
    matrix_files = []
    for p in files:
        txt = _safe_read_text(p) or ""
        if re.search(r"matrix\s*:", txt, flags=re.I):
            matrix_files.append(p)
        for os_name, pat in {
            "linux": r"ubuntu|linux",
            "windows": r"windows",
            "macos": r"macos|mac-os|darwin",
        }.items():
            if re.search(pat, txt, flags=re.I):
                os_tokens[os_name] += 1
        for m in re.findall(r"python-version[^\n]*[:=]\s*\[?([^\n\]]+)", txt, flags=re.I):
            python_versions.update(re.findall(r"\d+(?:\.\d+)?", m))
        for m in re.findall(r"node-version[^\n]*[:=]\s*\[?([^\n\]]+)", txt, flags=re.I):
            node_versions.update(re.findall(r"\d+(?:\.\d+)?", m))
        for m in re.findall(r"julia-version[^\n]*[:=]\s*\[?([^\n\]]+)", txt, flags=re.I):
            julia_versions.update(re.findall(r"\d+(?:\.\d+)?", m))
    return {
        "workflow_count": len(files),
        "matrix_file_count": len(matrix_files),
        "os_tokens": dict(os_tokens),
        "python_versions": sorted(python_versions),
        "node_versions": sorted(node_versions),
        "julia_versions": sorted(julia_versions),
        "workflow_files": [str(p.relative_to(root)) for p in files],
    }


def source_file_count(root: Path) -> int:
    return sum(1 for p in iter_files(root) if p.suffix.lower() in SOURCE_SUFFIXES)


def source_texts(root: Path, max_files: int = 2000) -> Dict[str, str]:
    out = {}
    n = 0
    for p in iter_files(root):
        if p.suffix.lower() not in SOURCE_SUFFIXES:
            continue
        txt = _safe_read_text(p)
        if txt:
            out[str(p.relative_to(root))] = txt
            n += 1
        if n >= max_files:
            break
    return out


# =========================================================
# DI.1 DRS — Deployment Readiness Score
# =========================================================

INSTALL_DOC_PATTERNS = [
    r"\binstall(?:ation)?\b", r"\bquick\s*start\b", r"\bgetting started\b", r"\bpip install\b",
    r"\bconda install\b", r"\bnpm install\b", r"\bdocker run\b", r"\bsetup\b", r"\bbuild from source\b",
]
CONFIG_PATTERNS = [
    r"\bconfiguration\b", r"\bconfig(?:uration)? file\b", r"\benvironment variable(?:s)?\b", r"\b\.env\b",
    r"\bdefault(?:s)?\b", r"\bparameter(?:s)?\b", r"\bsettings\b", r"\boptions\b",
]
ERROR_PATTERNS = [
    r"\btry\s*:", r"\bexcept\b", r"\braise\s+", r"\bValueError\b", r"\bTypeError\b", r"\bassert\b",
    r"\bvalidate\b", r"\bvalidation\b", r"\berror message\b", r"\bgraceful\b",
    r"\babort\b", r"\bfail(?:ed|ure)?\b", r"\bwarning\b", r"\bcheck\b",
]
BASIC_LOG_PATTERNS = [r"\blogging\b", r"\blogger\b", r"\bgetLogger\b", r"\blog\." , r"\bverbose\b", r"\bdebug\b"]


def compute_installation_ease(repo_path: Path) -> Tuple[Optional[float], Dict[str, Any]]:
    texts = candidate_texts(repo_path)
    if not texts:
        return None, {"available": False, "reason": "no_candidate_documentation_scanned"}
    metadata = package_metadata(repo_path)
    package_manager_score = sat_exp(len(metadata["package_managers"]), scale=1.0)
    install_hits = count_regex_hits_in_texts(texts, INSTALL_DOC_PATTERNS)
    install_docs = _log_repo_signal(install_hits, max(1.0, install_hits))
    quickstart_files = files_matching(repo_path, [r"readme", r"install", r"quickstart", r"getting[-_ ]started"], suffixes=[".md", ".rst", ".txt"])
    quickstart_score = sat_exp(len(quickstart_files), scale=1.0)
    example_run_hits = count_regex_hits_in_texts(texts, [r"\bpython -m\b", r"\bpytest\b", r"\bmake\b", r"\bCLI\b", r"\bcommand line\b", r"\bexample command\b"])
    runnable_examples = _log_repo_signal(example_run_hits, max(1.0, example_run_hits))
    block = mean([package_manager_score, install_docs, quickstart_score, runnable_examples])
    return block, {
        "available": True,
        "block": block,
        "sub": {
            "IE_package_manager_evidence": package_manager_score,
            "IE_installation_documentation": install_docs,
            "IE_quickstart_files": quickstart_score,
            "IE_runnable_command_examples": runnable_examples,
        },
        "raw": {
            "package_managers": metadata["package_managers"],
            "package_metadata_present": metadata["present"],
            "installation_keyword_hits": install_hits,
            "quickstart_files": [str(p.relative_to(repo_path)) for p in quickstart_files[:50]],
            "runnable_example_hits": example_run_hits,
        },
        "method_note": "Installation ease uses package-manager evidence, install docs, quick-start artifacts, and executable examples, not generic documentation volume.",
    }


def compute_configuration_management(repo_path: Path) -> Tuple[Optional[float], Dict[str, Any]]:
    texts = candidate_texts(repo_path)
    if not texts:
        return None, {"available": False, "reason": "no_candidate_documentation_scanned"}
    config_files = files_matching(repo_path, [
        r"(^|/)config", r"settings", r"\.env", r"parameters?", r"defaults?", r"example.*\.ya?ml", r"example.*\.json",
    ], suffixes=[".yml", ".yaml", ".json", ".toml", ".ini", ".cfg", ".env", ".txt", ".md"])
    config_dirs = has_dir(repo_path, ["config", "configs", "settings"])
    config_hits = count_regex_hits_in_texts(texts, CONFIG_PATTERNS)
    env_hits = count_regex_hits_in_texts(texts, [r"\b[A-Z][A-Z0-9_]{3,}\b", r"\bos\.environ\b", r"\bprocess\.env\b", r"\bgetenv\b"])
    defaults_hits = count_regex_hits_in_texts(texts, [r"\bdefault(?:s)?\b", r"\bsensible defaults\b", r"\bexample config\b", r"\btemplate\b"])
    ref = repo_hit_ref(len(config_files), config_hits, env_hits, defaults_hits)
    cm_config_artifacts = mean([_log_repo_signal(len(config_files), ref), config_dirs])
    cm_config_docs = _log_repo_signal(config_hits, ref)
    cm_environment_variables = _log_repo_signal(env_hits, ref)
    cm_defaults_templates = _log_repo_signal(defaults_hits, ref)
    block = mean([cm_config_artifacts, cm_config_docs, cm_environment_variables, cm_defaults_templates])
    return block, {
        "available": True,
        "block": block,
        "sub": {
            "CM_config_artifacts": cm_config_artifacts,
            "CM_configuration_documentation": cm_config_docs,
            "CM_environment_variable_support": cm_environment_variables,
            "CM_defaults_and_templates": cm_defaults_templates,
        },
        "raw": {
            "config_files_count": len(config_files),
            "config_files_sample": [str(p.relative_to(repo_path)) for p in config_files[:50]],
            "config_directory_present": config_dirs,
            "configuration_keyword_hits": config_hits,
            "environment_variable_hits": env_hits,
            "defaults_template_hits": defaults_hits,
            "hit_ref": ref,
        },
    }


def compute_runtime_safety(repo_path: Path) -> Tuple[Optional[float], Dict[str, Any]]:
    src = source_texts(repo_path)
    if not src:
        return None, {"available": False, "reason": "no_source_files_detected"}
    source_files = len(src)
    if source_files < MIN_SOURCE_FILES_FOR_RUNTIME_DENSITY:
        return None, {
            "available": False,
            "reason": "insufficient_source_sample_for_density",
            "source_files_sampled": source_files,
        }
    error_hits = count_regex_hits_in_texts(src, ERROR_PATTERNS)
    validation_hits = count_regex_hits_in_texts(
        src,
        [
            r"\bvalidate\b", r"\bschema\b", r"\bjsonschema\b", r"\bpydantic\b", r"\bmarshmallow\b", r"\bcerberus\b",
            r"\bconstraint\b", r"\binfeasib", r"\bbounds?\b", r"\bconsistency check\b",
        ],
    )
    user_message_hits = count_regex_hits_in_texts(
        src,
        [
            r"raise\s+\w*Error\s*\([^\)]{8,}", r"ValueError\s*\([^\)]{8,}", r"click\.BadParameter", r"argparse",
            r"\blog(?:ger)?\.(?:error|warning|critical)\(", r"\bprint\s*\(\s*[\"'][^\"']{10,}",
        ],
    )
    # Density is normalized by repository source size, avoiding larger repos scoring high simply by volume.
    eh_density = ratio01(error_hits, max(1, source_files * 8))
    val_density = ratio01(validation_hits, max(1, source_files * 2))
    msg_density = ratio01(user_message_hits, max(1, source_files * 2))
    block = mean([eh_density, val_density, msg_density])
    return block, {
        "available": True,
        "block": block,
        "sub": {
            "RS_error_handling_density": eh_density,
            "RS_input_validation_density": val_density,
            "RS_actionable_message_density": msg_density,
        },
        "raw": {
            "source_files_sampled": source_files,
            "error_handling_hits": error_hits,
            "validation_hits": validation_hits,
            "actionable_message_hits": user_message_hits,
        },
        "method_note": "Runtime safety is source-normalized to reduce redundancy with repository size and general code volume.",
    }


def compute_deployment_logging(repo_path: Path) -> Tuple[Optional[float], Dict[str, Any]]:
    docs = candidate_texts(repo_path)
    src = source_texts(repo_path)
    all_text = dict(docs)
    all_text.update({f"src::{k}": v for k, v in src.items()})
    log_hits = count_regex_hits_in_texts(all_text, BASIC_LOG_PATTERNS)
    cli_verbosity_hits = count_regex_hits_in_texts(all_text, [r"--verbose", r"--debug", r"--log-level", r"LOG_LEVEL", r"verbosity"])
    operation_doc_hits = count_regex_hits_in_texts(docs, [r"\blog(?:ging|s)?\b", r"\bmonitor(?:ing)?\b", r"\btroubleshoot(?:ing)?\b", r"\bdiagnostic(?:s)?\b"])
    source_files = max(1, len(src))
    log_density = ratio01(log_hits, source_files * 5)
    cli_logging = sat_exp(cli_verbosity_hits, scale=2.0)
    ops_docs = _log_repo_signal(operation_doc_hits, max(1.0, operation_doc_hits))
    block = mean([log_density, cli_logging, ops_docs])
    return block, {
        "available": True,
        "block": block,
        "sub": {
            "DL_logging_implementation_density": log_density,
            "DL_runtime_verbosity_controls": cli_logging,
            "DL_operational_logging_docs": ops_docs,
        },
        "raw": {
            "logging_hits": log_hits,
            "cli_verbosity_hits": cli_verbosity_hits,
            "operation_doc_hits": operation_doc_hits,
            "source_files_sampled": len(src),
        },
    }


def compute_drs(
    repo_path: Path,
    repo_url: Optional[str] = None,
    github_token: Optional[str] = None,
    learned_weights: Optional[Dict[str, float]] = None,
    calibration: Optional[Dict[str, Any]] = None,
    aggregation: str = "geometric",
) -> Dict[str, Any]:
    install, install_d = compute_installation_ease(repo_path)
    config, config_d = compute_configuration_management(repo_path)
    runtime, runtime_d = compute_runtime_safety(repo_path)
    logging, logging_d = compute_deployment_logging(repo_path)
    blocks = {
        "installation_ease": install,
        "configuration_management": config,
        "runtime_safety": runtime,
        "deployment_logging": logging,
    }
    score, weights, weights_source, agg_name = final_score_from_blocks(blocks, learned_weights, aggregation)
    return {
        "repository_path": str(repo_path),
        "repository_url": repo_url,
        "score": score,
        "band": band(score),
        "aggregation": agg_name,
        "details": {
            "DRS": score,
            "N_installation_ease": install,
            "N_configuration_management": config,
            "N_runtime_safety": runtime,
            "N_deployment_logging": logging,
            "weights": weights,
            "weights_source": weights_source,
            "blocks": {
                "installation_ease": install_d,
                "configuration_management": config_d,
                "runtime_safety": runtime_d,
                "deployment_logging": logging_d,
            },
            "method_note": "DRS measures installability, configurability, runtime safety, and basic deployment logging. It avoids full observability maturity, which belongs to OBS.",
        },
    }


# =========================================================
# DI.2 ICI — Integration Capability Index
# =========================================================

API_PATTERNS = [
    r"\bAPI\b", r"\bREST\b", r"\bGraphQL\b", r"\bSDK\b", r"\bclient library\b", r"\bpublic interface\b",
    r"\bcommand line interface\b", r"\bCLI\b", r"\bimport .* from\b", r"\bfrom .* import\b",
]
STANDARD_PATTERNS = [
    r"\bIEC\s?61850\b", r"\bIEC\s?61970\b", r"\bIEC\s?61968\b", r"\bCIM\b", r"\bCommon Information Model\b",
    r"\bIEEE\b", r"\bENTSO[- ]?E\b", r"\bCGMES\b", r"\bOpenADR\b", r"\bOPC\s?UA\b", r"\bModbus\b",
    r"\bDNP3\b", r"\bMQTT\b", r"\bSunSpec\b", r"\bOCPP\b",
]
THIRD_PARTY_PATTERNS = [
    r"\bintegration(?:s)?\b", r"\bconnector(?:s)?\b", r"\bplugin(?:s)?\b", r"\badapter(?:s)?\b",
    r"\bimporter(?:s)?\b", r"\bexporter(?:s)?\b", r"\bcompatible with\b", r"\binterface to\b",
]
FORMAT_PATTERNS = {
    "tabular": [r"\bCSV\b", r"\bExcel\b", r"\bXLSX\b", r"\bParquet\b", r"\bFeather\b"],
    "structured": [r"\bJSON\b", r"\bYAML\b", r"\bXML\b", r"\bTOML\b"],
    "scientific": [r"\bHDF5\b", r"\bNetCDF\b", r"\bZarr\b", r"\bMatpower\b", r"\bPSS/E\b", r"\bRAW\b"],
    "geospatial": [r"\bGeoJSON\b", r"\bShapefile\b", r"\bGeoPackage\b", r"\bGIS\b"],
    "database_stream": [r"\bSQL\b", r"\bPostgreSQL\b", r"\bSQLite\b", r"\bKafka\b", r"\bMQTT\b"],
}


def compute_api_surface(repo_path: Path) -> Tuple[Optional[float], Dict[str, Any]]:
    texts = candidate_texts(repo_path, include_source=False)
    src = source_texts(repo_path)
    api_doc_hits = count_regex_hits_in_texts(texts, API_PATTERNS)
    cli_files = files_matching(repo_path, [r"(^|/)cli\.", r"command", r"console_scripts", r"entry_points"], suffixes=[".py", ".toml", ".cfg", ".json", ".md"])
    openapi_files = files_matching(repo_path, [r"openapi", r"swagger", r"api[-_ ]?spec"], suffixes=[".yml", ".yaml", ".json", ".md"])
    python_public_defs = 0
    for rel, txt in src.items():
        if rel.endswith(".py"):
            python_public_defs += len(re.findall(r"(?m)^def\s+(?!_)[A-Za-z_]\w+\s*\(", txt))
            python_public_defs += len(re.findall(r"(?m)^class\s+(?!_)[A-Za-z_]\w+", txt))
    src_count = max(1, len(src))
    api_docs = _log_repo_signal(api_doc_hits, max(1.0, api_doc_hits))
    cli_surface = sat_exp(len(cli_files), scale=1.0)
    formal_spec = sat_exp(len(openapi_files), scale=1.0)
    public_api_density = (
        ratio01(python_public_defs, src_count * 10)
        if src and len(src) >= MIN_SOURCE_FILES_FOR_RUNTIME_DENSITY
        else None
    )
    block = mean([api_docs, cli_surface, formal_spec, public_api_density])
    return block, {
        "available": True,
        "block": block,
        "sub": {
            "API_documented_interface": api_docs,
            "API_cli_or_entrypoints": cli_surface,
            "API_formal_specification": formal_spec,
            "API_public_surface_density": public_api_density,
        },
        "raw": {
            "api_documentation_hits": api_doc_hits,
            "cli_or_entrypoint_files": [str(p.relative_to(repo_path)) for p in cli_files[:50]],
            "openapi_or_swagger_files": [str(p.relative_to(repo_path)) for p in openapi_files[:50]],
            "python_public_defs": python_public_defs,
            "source_files_sampled": len(src),
        },
    }


def compute_standards_compliance(repo_path: Path) -> Tuple[Optional[float], Dict[str, Any]]:
    texts = candidate_texts(repo_path, include_source=False)
    hits = count_regex_hits_in_texts(texts, STANDARD_PATTERNS)
    standard_files = files_matching(repo_path, [r"standard", r"iec", r"ieee", r"cim", r"cgmes", r"openadr", r"modbus", r"opc"], suffixes=[".md", ".rst", ".txt", ".yml", ".yaml", ".json", ".xml"])
    standard_families = {}
    for fam, pats in {
        "power_information_models": [r"CIM", r"IEC\s?61970", r"IEC\s?61968", r"CGMES"],
        "grid_communication": [r"IEC\s?61850", r"OPC\s?UA", r"Modbus", r"DNP3", r"MQTT"],
        "market_policy_data": [r"ENTSO", r"OpenADR", r"OCPP"],
        "general_engineering": [r"IEEE", r"ISO", r"NIST"],
    }.items():
        standard_families[fam] = count_regex_hits_in_texts(texts, pats)
    presence = ratio01(sum(1 for v in standard_families.values() if v > 0), len(standard_families))
    intensity = _log_repo_signal(hits, max(1.0, hits))
    file_evidence = sat_exp(len(standard_files), scale=1.0)
    diversity = entropy_normalized([v for v in standard_families.values() if v > 0])
    block = mean([intensity, file_evidence, presence, diversity])
    return block, {
        "available": True,
        "block": block,
        "sub": {
            "SC_standard_reference_intensity": intensity,
            "SC_standard_artifact_files": file_evidence,
            "SC_standard_family_coverage": presence,
            "SC_standard_family_entropy": diversity,
        },
        "raw": {
            "standard_keyword_hits": hits,
            "standard_files": [str(p.relative_to(repo_path)) for p in standard_files[:50]],
            "standard_family_counts": standard_families,
        },
    }


def compute_third_party_support(repo_path: Path, repo_url: Optional[str], token: Optional[str]) -> Tuple[Optional[float], Dict[str, Any]]:
    texts = candidate_texts(repo_path, include_source=False)
    topics = github_topics(repo_url, token) if repo_url else []
    integration_hits = count_regex_hits_in_texts(texts, THIRD_PARTY_PATTERNS)
    integration_files = files_matching(repo_path, [r"integration", r"connector", r"plugin", r"adapter", r"importer", r"exporter"], suffixes=[".md", ".rst", ".txt", ".py", ".jl", ".js", ".ts"])
    plugin_dirs = has_dir(repo_path, ["plugins", "plugin", "integrations", "connectors", "adapters", "interfaces"])
    topic_hits = sum(1 for t in topics if any(k in t for k in ["plugin", "integration", "connector", "api", "adapter"]))
    ref = repo_hit_ref(integration_hits, len(integration_files), topic_hits)
    docs = _log_repo_signal(integration_hits, ref)
    files = _log_repo_signal(len(integration_files), ref)
    topic_signal = _log_repo_signal(topic_hits, ref)
    block = mean([docs, files, plugin_dirs, topic_signal])
    return block, {
        "available": True,
        "block": block,
        "sub": {
            "TPS_integration_documentation": docs,
            "TPS_integration_artifacts": files,
            "TPS_plugin_or_connector_dirs": plugin_dirs,
            "TPS_integration_topics": topic_signal,
        },
        "raw": {
            "integration_keyword_hits": integration_hits,
            "integration_files_count": len(integration_files),
            "integration_files_sample": [str(p.relative_to(repo_path)) for p in integration_files[:50]],
            "plugin_dirs_present": plugin_dirs,
            "github_topics": topics,
            "integration_topic_hits": topic_hits,
            "hit_ref": ref,
        },
    }


def compute_data_format_support(repo_path: Path) -> Tuple[Optional[float], Dict[str, Any]]:
    texts = candidate_texts(repo_path, include_source=True)
    family_counts = {}
    for family, pats in FORMAT_PATTERNS.items():
        family_counts[family] = count_regex_hits_in_texts(texts, pats)
    present_families = [k for k, v in family_counts.items() if v > 0]
    total_hits = sum(family_counts.values())
    format_files = files_matching(repo_path, [r"format", r"converter", r"importer", r"exporter", r"io", r"read_", r"write_"], suffixes=[".md", ".rst", ".txt", ".py", ".jl", ".js", ".ts"])
    coverage = ratio01(len(present_families), len(FORMAT_PATTERNS))
    intensity = _log_repo_signal(total_hits, max(1.0, total_hits))
    entropy = entropy_normalized([v for v in family_counts.values() if v > 0])
    io_artifacts = sat_exp(len(format_files), scale=3.0)
    block = mean([coverage, intensity, entropy, io_artifacts])
    return block, {
        "available": True,
        "block": block,
        "sub": {
            "DFS_format_family_coverage": coverage,
            "DFS_format_reference_intensity": intensity,
            "DFS_format_family_entropy": entropy,
            "DFS_io_artifacts": io_artifacts,
        },
        "raw": {
            "format_family_counts": family_counts,
            "format_families_present": present_families,
            "total_format_hits": total_hits,
            "io_artifacts_count": len(format_files),
            "io_artifacts_sample": [str(p.relative_to(repo_path)) for p in format_files[:50]],
        },
    }


def compute_ici(
    repo_path: Path,
    repo_url: Optional[str] = None,
    github_token: Optional[str] = None,
    learned_weights: Optional[Dict[str, float]] = None,
    calibration: Optional[Dict[str, Any]] = None,
    aggregation: str = "geometric",
) -> Dict[str, Any]:
    api, api_d = compute_api_surface(repo_path)
    standards, standards_d = compute_standards_compliance(repo_path)
    third, third_d = compute_third_party_support(repo_path, repo_url, github_token)
    formats, formats_d = compute_data_format_support(repo_path)
    blocks = {
        "api_surface": api,
        "standards_compliance": standards,
        "third_party_support": third,
        "data_format_support": formats,
    }
    score, weights, weights_source, agg_name = final_score_from_blocks(blocks, learned_weights, aggregation)
    return {
        "repository_path": str(repo_path),
        "repository_url": repo_url,
        "score": score,
        "band": band(score),
        "aggregation": agg_name,
        "details": {
            "ICI": score,
            "N_api_surface": api,
            "N_standards_compliance": standards,
            "N_third_party_support": third,
            "N_data_format_support": formats,
            "weights": weights,
            "weights_source": weights_source,
            "blocks": {
                "api_surface": api_d,
                "standards_compliance": standards_d,
                "third_party_support": third_d,
                "data_format_support": formats_d,
            },
            "method_note": "ICI measures integration mechanisms and compatibility evidence, separated from adoption/popularity and energy-domain capability.",
        },
    }


# =========================================================
# DI.3 PFS — Platform Flexibility Score
# =========================================================

DEPLOYMENT_PATTERNS = {
    "local": [r"\blocal\b", r"\bworkstation\b", r"\bdesktop\b"],
    "server": [r"\bserver\b", r"\bservice\b", r"\bdaemon\b", r"\bbackend\b"],
    "cloud": [r"\bcloud\b", r"\bAWS\b", r"\bAzure\b", r"\bGCP\b", r"\bS3\b"],
    "hpc": [r"\bHPC\b", r"\bcluster\b", r"\bSLURM\b", r"\bMPI\b"],
    "edge": [r"\bedge\b", r"\bembedded\b", r"\bRaspberry Pi\b", r"\bIoT\b"],
    "notebook": [r"\bJupyter\b", r"\bnotebook\b", r"\bColab\b"],
}


def compute_os_support(repo_path: Path) -> Tuple[Optional[float], Dict[str, Any]]:
    ci = parse_ci_matrices(repo_path)
    docs = candidate_texts(repo_path)
    doc_counts = {
        "linux": count_regex_hits_in_texts(docs, [r"\bLinux\b", r"\bUbuntu\b", r"\bDebian\b"]),
        "windows": count_regex_hits_in_texts(docs, [r"\bWindows\b", r"\bPowerShell\b"]),
        "macos": count_regex_hits_in_texts(docs, [r"\bmacOS\b", r"\bMac OS\b", r"\bOSX\b"]),
    }
    ci_os_present = {k: 1 for k, v in ci["os_tokens"].items() if v > 0}
    combined = {osn: int((ci_os_present.get(osn, 0) > 0) or (doc_counts.get(osn, 0) > 0)) for osn in ["linux", "windows", "macos"]}
    family_coverage = ratio01(sum(combined.values()), 3)
    ci_matrix_score = ratio01(sum(1 for v in ci["os_tokens"].values() if v > 0), 3) if ci["workflow_count"] else None
    docs_score = ratio01(sum(1 for v in doc_counts.values() if v > 0), 3)
    block = mean([family_coverage, ci_matrix_score, docs_score])
    return block, {
        "available": True,
        "block": block,
        "sub": {
            "OS_combined_platform_coverage": family_coverage,
            "OS_ci_matrix_coverage": ci_matrix_score,
            "OS_documented_platform_coverage": docs_score,
        },
        "raw": {
            "ci": ci,
            "documented_os_counts": doc_counts,
            "combined_os_presence": combined,
        },
    }


def compute_deployment_modes(repo_path: Path) -> Tuple[Optional[float], Dict[str, Any]]:
    texts = candidate_texts(repo_path)
    mode_counts = {mode: count_regex_hits_in_texts(texts, pats) for mode, pats in DEPLOYMENT_PATTERNS.items()}
    deployment_files = files_matching(repo_path, [r"deploy", r"deployment", r"service", r"systemd", r"terraform", r"ansible", r"helm", r"k8s", r"kubernetes"], suffixes=[".md", ".rst", ".txt", ".yml", ".yaml", ".tf", ".service", ".sh"])
    present = [k for k, v in mode_counts.items() if v > 0]
    coverage = ratio01(len(present), len(DEPLOYMENT_PATTERNS))
    entropy = entropy_normalized([v for v in mode_counts.values() if v > 0])
    artifact_score = sat_exp(len(deployment_files), scale=2.0)
    intensity = _log_repo_signal(sum(mode_counts.values()), max(1.0, sum(mode_counts.values())))
    block = mean([coverage, entropy, artifact_score, intensity])
    return block, {
        "available": True,
        "block": block,
        "sub": {
            "DM_deployment_mode_coverage": coverage,
            "DM_deployment_mode_entropy": entropy,
            "DM_deployment_artifacts": artifact_score,
            "DM_deployment_documentation_intensity": intensity,
        },
        "raw": {
            "deployment_mode_counts": mode_counts,
            "deployment_modes_present": present,
            "deployment_files_count": len(deployment_files),
            "deployment_files_sample": [str(p.relative_to(repo_path)) for p in deployment_files[:50]],
        },
    }


def compute_containerization(repo_path: Path) -> Tuple[Optional[float], Dict[str, Any]]:
    dockerfiles = files_matching(repo_path, [r"(^|/)Dockerfile$", r"Dockerfile"], suffixes=["dockerfile", ""])
    compose_files = files_matching(repo_path, [r"docker-compose", r"compose\.ya?ml"], suffixes=[".yml", ".yaml"])
    k8s_files = files_matching(repo_path, [r"k8s", r"kubernetes", r"deployment\.ya?ml", r"service\.ya?ml", r"helm", r"chart\.ya?ml"], suffixes=[".yml", ".yaml", ".tpl"])
    container_docs = count_regex_hits_in_texts(candidate_texts(repo_path), [r"\bDocker\b", r"\bcontainer\b", r"\bdocker compose\b", r"\bKubernetes\b", r"\bHelm\b"])
    docker_score = sat_exp(len(dockerfiles), scale=1.0)
    compose_score = sat_exp(len(compose_files), scale=1.0)
    orchestration_score = sat_exp(len(k8s_files), scale=2.0)
    docs_score = _log_repo_signal(container_docs, max(1.0, container_docs))
    block = mean([docker_score, compose_score, orchestration_score, docs_score])
    return block, {
        "available": True,
        "block": block,
        "sub": {
            "CT_dockerfile_presence": docker_score,
            "CT_compose_presence": compose_score,
            "CT_orchestration_artifacts": orchestration_score,
            "CT_container_documentation": docs_score,
        },
        "raw": {
            "dockerfiles": [str(p.relative_to(repo_path)) for p in dockerfiles[:50]],
            "compose_files": [str(p.relative_to(repo_path)) for p in compose_files[:50]],
            "kubernetes_or_helm_files": [str(p.relative_to(repo_path)) for p in k8s_files[:50]],
            "container_doc_hits": container_docs,
        },
    }


def compute_runtime_environment_variability(repo_path: Path) -> Tuple[Optional[float], Dict[str, Any]]:
    ci = parse_ci_matrices(repo_path)
    metadata = package_metadata(repo_path)
    version_families = 0
    if len(ci["python_versions"]) > 1:
        version_families += 1
    if len(ci["node_versions"]) > 1:
        version_families += 1
    if len(ci["julia_versions"]) > 1:
        version_families += 1
    env_files = files_matching(repo_path, [r"requirements", r"environment", r"conda", r"poetry", r"package-lock", r"pnpm-lock", r"uv.lock", r"nix", r"flake\.nix"], suffixes=[".txt", ".yml", ".yaml", ".toml", ".lock", ".nix", ".json"])
    lang_runtime_score = sat_exp(version_families, scale=1.0)
    env_repro_score = sat_exp(len(env_files), scale=2.0)
    package_manager_breadth = sat_exp(len(metadata["package_managers"]), scale=2.0)
    block = mean([lang_runtime_score, env_repro_score, package_manager_breadth])
    return block, {
        "available": True,
        "block": block,
        "sub": {
            "REV_language_version_matrix": lang_runtime_score,
            "REV_environment_reproducibility_files": env_repro_score,
            "REV_package_manager_breadth": package_manager_breadth,
        },
        "raw": {
            "ci_versions": {
                "python": ci["python_versions"],
                "node": ci["node_versions"],
                "julia": ci["julia_versions"],
            },
            "environment_files_count": len(env_files),
            "environment_files_sample": [str(p.relative_to(repo_path)) for p in env_files[:50]],
            "package_managers": metadata["package_managers"],
        },
        "method_note": "Runtime environment variability captures tested/interpretable environment breadth without rewarding mere deployment mode claims.",
    }


def compute_pfs(
    repo_path: Path,
    repo_url: Optional[str] = None,
    github_token: Optional[str] = None,
    learned_weights: Optional[Dict[str, float]] = None,
    calibration: Optional[Dict[str, Any]] = None,
    aggregation: str = "geometric",
) -> Dict[str, Any]:
    os_support, os_d = compute_os_support(repo_path)
    deployment_modes, dm_d = compute_deployment_modes(repo_path)
    containerization, ct_d = compute_containerization(repo_path)
    runtime_env, rev_d = compute_runtime_environment_variability(repo_path)
    blocks = {
        "os_support": os_support,
        "deployment_modes": deployment_modes,
        "containerization": containerization,
        "runtime_environment_variability": runtime_env,
    }
    score, weights, weights_source, agg_name = final_score_from_blocks(blocks, learned_weights, aggregation)
    return {
        "repository_path": str(repo_path),
        "repository_url": repo_url,
        "score": score,
        "band": band(score),
        "aggregation": agg_name,
        "details": {
            "PFS": score,
            "N_os_support": os_support,
            "N_deployment_modes": deployment_modes,
            "N_containerization": containerization,
            "N_runtime_environment_variability": runtime_env,
            "weights": weights,
            "weights_source": weights_source,
            "blocks": {
                "os_support": os_d,
                "deployment_modes": dm_d,
                "containerization": ct_d,
                "runtime_environment_variability": rev_d,
            },
            "method_note": "PFS measures platform breadth, deployment mode breadth, container support, and runtime environment variability. It avoids duplicating DRS installability.",
        },
    }


# =========================================================
# DI.4 OBS — Observability Score
# =========================================================

STRUCTURED_LOG_PATTERNS = [
    r"structlog", r"logging\.config", r"dictConfig", r"JSONFormatter", r"jsonlogger", r"loguru", r"winston", r"pino",
    r"structured log", r"log format", r"correlation id", r"trace id", r"request id",
]
METRICS_PATTERNS = [
    r"prometheus", r"metrics endpoint", r"/metrics", r"OpenTelemetry", r"opentelemetry", r"StatsD", r"Grafana",
    r"Datadog", r"counter\(", r"gauge\(", r"histogram\(", r"MeterProvider", r"tracer",
]
HEALTH_PATTERNS = [
    r"health check", r"healthcheck", r"/health", r"/ready", r"readiness", r"liveness", r"self[- ]test",
    r"diagnostic", r"status endpoint", r"service status",
]
ALERT_PATTERNS = [
    r"alert", r"alertmanager", r"PagerDuty", r"Opsgenie", r"Slack webhook", r"webhook", r"incident", r"Sentry",
    r"notification hook", r"on-call", r"monitoring alert",
]


def compute_logging_quality(repo_path: Path) -> Tuple[Optional[float], Dict[str, Any]]:
    docs = candidate_texts(repo_path)
    src = source_texts(repo_path)
    combined = dict(docs)
    combined.update({f"src::{k}": v for k, v in src.items()})
    basic_hits = count_regex_hits_in_texts(combined, BASIC_LOG_PATTERNS)
    structured_hits = count_regex_hits_in_texts(combined, STRUCTURED_LOG_PATTERNS)
    level_hits = count_regex_hits_in_texts(combined, [r"log[-_ ]?level", r"DEBUG", r"INFO", r"WARNING", r"ERROR", r"CRITICAL", r"--log-level"])
    docs_hits = count_regex_hits_in_texts(docs, [r"logging", r"logs", r"troubleshooting", r"diagnostics"])
    src_count = max(1, len(src))
    basic_density = ratio01(basic_hits, src_count * 5) if src else None
    structured = sat_exp(structured_hits, scale=2.0)
    levels = sat_exp(level_hits, scale=5.0)
    docs_score = _log_repo_signal(docs_hits, max(1.0, docs_hits))
    block = mean([basic_density, structured, levels, docs_score])
    return block, {
        "available": True,
        "block": block,
        "sub": {
            "LQ_basic_logging_density": basic_density,
            "LQ_structured_logging_evidence": structured,
            "LQ_log_level_controls": levels,
            "LQ_logging_documentation": docs_score,
        },
        "raw": {
            "basic_logging_hits": basic_hits,
            "structured_logging_hits": structured_hits,
            "log_level_hits": level_hits,
            "logging_doc_hits": docs_hits,
            "source_files_sampled": len(src),
        },
    }


def compute_metrics_exposure(repo_path: Path) -> Tuple[Optional[float], Dict[str, Any]]:
    docs = candidate_texts(repo_path)
    src = source_texts(repo_path)
    combined = dict(docs)
    combined.update({f"src::{k}": v for k, v in src.items()})
    metrics_hits = count_regex_hits_in_texts(combined, METRICS_PATTERNS)
    metrics_files = files_matching(repo_path, [r"metrics", r"prometheus", r"grafana", r"opentelemetry", r"otel"], suffixes=[".md", ".rst", ".txt", ".py", ".js", ".ts", ".yml", ".yaml", ".json"])
    dashboard_files = files_matching(repo_path, [r"dashboard", r"grafana", r"prometheus"], suffixes=[".json", ".yml", ".yaml", ".md"])
    docs_hits = count_regex_hits_in_texts(docs, [r"metrics", r"monitoring", r"telemetry", r"prometheus", r"grafana"])
    implementation = sat_exp(metrics_hits, scale=3.0)
    artifact_score = sat_exp(len(metrics_files), scale=2.0)
    dashboard_score = sat_exp(len(dashboard_files), scale=1.0)
    docs_score = _log_repo_signal(docs_hits, max(1.0, docs_hits))
    block = mean([implementation, artifact_score, dashboard_score, docs_score])
    return block, {
        "available": True,
        "block": block,
        "sub": {
            "ME_metrics_implementation": implementation,
            "ME_metrics_artifacts": artifact_score,
            "ME_dashboard_or_scrape_configs": dashboard_score,
            "ME_metrics_documentation": docs_score,
        },
        "raw": {
            "metrics_hits": metrics_hits,
            "metrics_files_count": len(metrics_files),
            "metrics_files_sample": [str(p.relative_to(repo_path)) for p in metrics_files[:50]],
            "dashboard_files_count": len(dashboard_files),
            "dashboard_files_sample": [str(p.relative_to(repo_path)) for p in dashboard_files[:50]],
            "metrics_doc_hits": docs_hits,
        },
    }


def compute_health_checks(repo_path: Path) -> Tuple[Optional[float], Dict[str, Any]]:
    docs = candidate_texts(repo_path)
    src = source_texts(repo_path)
    combined = dict(docs)
    combined.update({f"src::{k}": v for k, v in src.items()})
    health_hits = count_regex_hits_in_texts(combined, HEALTH_PATTERNS)
    docker_health_hits = count_regex_hits_in_texts(combined, [r"HEALTHCHECK", r"healthcheck:", r"livenessProbe", r"readinessProbe"])
    health_files = files_matching(repo_path, [r"health", r"status", r"diagnostic", r"selftest", r"self-test"], suffixes=[".md", ".rst", ".txt", ".py", ".js", ".ts", ".yml", ".yaml"])
    docs_hits = count_regex_hits_in_texts(docs, [r"health check", r"readiness", r"liveness", r"self-test", r"diagnostic"])
    endpoint_score = sat_exp(health_hits, scale=3.0)
    deployment_probe_score = sat_exp(docker_health_hits, scale=1.0)
    file_score = sat_exp(len(health_files), scale=2.0)
    docs_score = _log_repo_signal(docs_hits, max(1.0, docs_hits))
    block = mean([endpoint_score, deployment_probe_score, file_score, docs_score])
    return block, {
        "available": True,
        "block": block,
        "sub": {
            "HC_health_endpoint_or_selftest": endpoint_score,
            "HC_deployment_probe_evidence": deployment_probe_score,
            "HC_health_artifact_files": file_score,
            "HC_health_documentation": docs_score,
        },
        "raw": {
            "health_hits": health_hits,
            "deployment_probe_hits": docker_health_hits,
            "health_files_count": len(health_files),
            "health_files_sample": [str(p.relative_to(repo_path)) for p in health_files[:50]],
            "health_doc_hits": docs_hits,
        },
    }


def compute_alert_hooks(repo_path: Path) -> Tuple[Optional[float], Dict[str, Any]]:
    docs = candidate_texts(repo_path)
    src = source_texts(repo_path)
    combined = dict(docs)
    combined.update({f"src::{k}": v for k, v in src.items()})
    alert_hits = count_regex_hits_in_texts(combined, ALERT_PATTERNS)
    webhook_files = files_matching(repo_path, [r"alert", r"webhook", r"incident", r"sentry", r"pagerduty", r"opsgenie"], suffixes=[".md", ".rst", ".txt", ".py", ".js", ".ts", ".yml", ".yaml", ".json"])
    ci_notification_hits = count_regex_hits_in_texts(combined, [r"slack", r"teams", r"discord", r"email notification", r"notify", r"webhook"])
    implementation = sat_exp(alert_hits, scale=3.0)
    artifact_score = sat_exp(len(webhook_files), scale=1.0)
    notification_score = sat_exp(ci_notification_hits, scale=4.0)
    block = mean([implementation, artifact_score, notification_score])
    return block, {
        "available": True,
        "block": block,
        "sub": {
            "AH_alerting_integration_evidence": implementation,
            "AH_alert_or_webhook_artifacts": artifact_score,
            "AH_notification_hook_evidence": notification_score,
        },
        "raw": {
            "alert_hits": alert_hits,
            "alert_or_webhook_files_count": len(webhook_files),
            "alert_or_webhook_files_sample": [str(p.relative_to(repo_path)) for p in webhook_files[:50]],
            "notification_hook_hits": ci_notification_hits,
        },
    }


def compute_obs(
    repo_path: Path,
    repo_url: Optional[str] = None,
    github_token: Optional[str] = None,
    learned_weights: Optional[Dict[str, float]] = None,
    calibration: Optional[Dict[str, Any]] = None,
    aggregation: str = "geometric",
) -> Dict[str, Any]:
    logging_q, lq_d = compute_logging_quality(repo_path)
    metrics, me_d = compute_metrics_exposure(repo_path)
    health, hc_d = compute_health_checks(repo_path)
    alerts, ah_d = compute_alert_hooks(repo_path)
    blocks = {
        "logging_quality": logging_q,
        "metrics_exposure": metrics,
        "health_checks": health,
        "alert_hooks": alerts,
    }
    score, weights, weights_source, agg_name = final_score_from_blocks(blocks, learned_weights, aggregation)
    return {
        "repository_path": str(repo_path),
        "repository_url": repo_url,
        "score": score,
        "band": band(score),
        "aggregation": agg_name,
        "details": {
            "OBS": score,
            "N_logging_quality": logging_q,
            "N_metrics_exposure": metrics,
            "N_health_checks": health,
            "N_alert_hooks": alerts,
            "weights": weights,
            "weights_source": weights_source,
            "blocks": {
                "logging_quality": lq_d,
                "metrics_exposure": me_d,
                "health_checks": hc_d,
                "alert_hooks": ah_d,
            },
            "method_note": "OBS measures production observability maturity. Unlike DRS deployment logging, it requires operational signals such as metrics, health checks, and alert hooks.",
        },
    }


# =========================================================
# DI category aggregate
# =========================================================

def compute_di(
    repo_path: Path,
    repo_url: Optional[str] = None,
    github_token: Optional[str] = None,
    learned_weights: Optional[Dict[str, float]] = None,
    calibration: Optional[Dict[str, Any]] = None,
    aggregation: str = "geometric",
) -> Dict[str, Any]:
    drs = compute_drs(repo_path, repo_url, github_token, learned_weights=None, calibration=calibration, aggregation=aggregation)
    ici = compute_ici(repo_path, repo_url, github_token, learned_weights=None, calibration=calibration, aggregation=aggregation)
    pfs = compute_pfs(repo_path, repo_url, github_token, learned_weights=None, calibration=calibration, aggregation=aggregation)
    obs = compute_obs(repo_path, repo_url, github_token, learned_weights=None, calibration=calibration, aggregation=aggregation)
    blocks = {
        "DRS": drs.get("score") / 100.0 if drs.get("score") is not None else None,
        "ICI": ici.get("score") / 100.0 if ici.get("score") is not None else None,
        "PFS": pfs.get("score") / 100.0 if pfs.get("score") is not None else None,
        "OBS": obs.get("score") / 100.0 if obs.get("score") is not None else None,
    }
    score, weights, weights_source, agg_name = final_score_from_blocks(blocks, learned_weights, aggregation)
    return {
        "repository_path": str(repo_path),
        "repository_url": repo_url,
        "score": score,
        "band": band(score),
        "aggregation": agg_name,
        "details": {
            "DI": score,
            "DRS": drs.get("score"),
            "ICI": ici.get("score"),
            "PFS": pfs.get("score"),
            "OBS": obs.get("score"),
            "weights": weights,
            "weights_source": weights_source,
            "scores": {"DRS": drs, "ICI": ici, "PFS": pfs, "OBS": obs},
            "missingness_note": "Unavailable values remain None and are excluded with weight renormalization; observable absence is scored as 0.0.",
            "method_note": "DI evaluates realistic deployment and interoperability readiness through four non-redundant dimensions: deployment readiness, integration capability, platform flexibility, and observability.",
        },
    }


# =========================================================
# Corpus weight-learning entry points
# =========================================================

def learn_drs_weights(dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = ["installation_ease", "configuration_management", "runtime_safety", "deployment_logging"]
    rows = [{k: row.get(k, row.get("N_" + k)) for k in keys} for row in dataset]
    return {"weights": norm_weights_from_rows(rows, keys), "n_repos": len(dataset)}


def learn_ici_weights(dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = ["api_surface", "standards_compliance", "third_party_support", "data_format_support"]
    rows = [{k: row.get(k, row.get("N_" + k)) for k in keys} for row in dataset]
    return {"weights": norm_weights_from_rows(rows, keys), "n_repos": len(dataset)}


def learn_pfs_weights(dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = ["os_support", "deployment_modes", "containerization", "runtime_environment_variability"]
    rows = [{k: row.get(k, row.get("N_" + k)) for k in keys} for row in dataset]
    return {"weights": norm_weights_from_rows(rows, keys), "n_repos": len(dataset)}


def learn_obs_weights(dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = ["logging_quality", "metrics_exposure", "health_checks", "alert_hooks"]
    rows = [{k: row.get(k, row.get("N_" + k)) for k in keys} for row in dataset]
    return {"weights": norm_weights_from_rows(rows, keys), "n_repos": len(dataset)}


def learn_di_weights(dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = ["DRS", "ICI", "PFS", "OBS"]
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
    section = loaded.get(score.upper()) or loaded.get(score.lower())
    if isinstance(section, dict):
        return section.get("weights", section)
    return loaded


def main() -> None:
    parser = argparse.ArgumentParser(description="RECOPS Deployability & Interoperability scores: DRS, ICI, PFS, OBS, DI")
    parser.add_argument("repo_url", nargs="?", help="GitHub repository URL, e.g. https://github.com/owner/repo")
    parser.add_argument("--local", dest="local_repo", help="Analyze an already-local repository path instead of cloning")
    parser.add_argument("--token", dest="github_token", default=os.environ.get("GITHUB_TOKEN"), help="GitHub token; defaults to env GITHUB_TOKEN")
    parser.add_argument("--score", choices=["drs", "ici", "pfs", "obs", "di", "all"], default="all", help="Which DI score to compute")
    parser.add_argument("--weights-json", dest="weights_json", help="Optional learned weights JSON for the selected score")
    parser.add_argument("--calibration-json", dest="calibration_json", help="Reserved for optional corpus calibration JSON")
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
    if sc == "drs":
        result = compute_drs(repo_path, args.repo_url, args.github_token, split_weights_for_score("drs", weights_loaded), calibration, args.aggregation)
    elif sc == "ici":
        result = compute_ici(repo_path, args.repo_url, args.github_token, split_weights_for_score("ici", weights_loaded), calibration, args.aggregation)
    elif sc == "pfs":
        result = compute_pfs(repo_path, args.repo_url, args.github_token, split_weights_for_score("pfs", weights_loaded), calibration, args.aggregation)
    elif sc == "obs":
        result = compute_obs(repo_path, args.repo_url, args.github_token, split_weights_for_score("obs", weights_loaded), calibration, args.aggregation)
    elif sc == "di":
        result = compute_di(repo_path, args.repo_url, args.github_token, split_weights_for_score("di", weights_loaded), calibration, args.aggregation)
    else:
        result = {
            "repository_path": str(repo_path),
            "repository_url": args.repo_url,
            "scores": {
                "DRS": compute_drs(repo_path, args.repo_url, args.github_token, None, calibration, args.aggregation),
                "ICI": compute_ici(repo_path, args.repo_url, args.github_token, None, calibration, args.aggregation),
                "PFS": compute_pfs(repo_path, args.repo_url, args.github_token, None, calibration, args.aggregation),
                "OBS": compute_obs(repo_path, args.repo_url, args.github_token, None, calibration, args.aggregation),
                "DI": compute_di(repo_path, args.repo_url, args.github_token, None, calibration, args.aggregation),
            },
            "note": "With --score all, --weights-json is ignored to avoid applying one weight file to multiple formulas.",
        }

    txt = json.dumps(result, indent=args.indent, ensure_ascii=False, sort_keys=False)
    if args.out:
        Path(args.out).write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
