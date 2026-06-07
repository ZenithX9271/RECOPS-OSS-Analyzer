#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Optional dependencies
---------------------
- requests: only needed if you pass a GitHub URL and want fallback zip download or
  GitHub API metadata.
- PyYAML: optional, improves YAML parsing for CITATION.cff, funding/community files.

Recommended installation:
    pip install requests pyyaml

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
from pathlib import Path
from statistics import mean as _py_mean
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import requests
except Exception:  # pragma: no cover - optional dependency
    requests = None

try:
    import yaml
except Exception:  # pragma: no cover - optional dependency
    yaml = None


# =========================================================
# Basic math utilities
# =========================================================

def clamp01(x: float) -> float:
    """Clamp a numeric value to the [0, 1] interval."""
    try:
        xf = float(x)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, xf))


def mean(values: Iterable[Optional[float]]) -> Optional[float]:
    """Arithmetic mean, skipping None values. Returns None if all values are None."""
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def weighted_geometric(pairs: Iterable[Tuple[Optional[float], float]], eps: float = 1e-6) -> Optional[float]:
    """
    Weighted geometric mean with explicit missingness.

    - None values are dropped.
    - Remaining weights are renormalized.
    - A value of 0 remains near-zero through eps, preserving non-compensation while
      avoiding log(0) numerical failure.
    """
    vals: List[Tuple[float, float]] = []
    for v, w in pairs:
        if v is None or w is None:
            continue
        try:
            vf = float(v)
            wf = float(w)
        except Exception:
            continue
        if wf <= 0:
            continue
        vals.append((clamp01(vf), wf))

    if not vals:
        return None

    wsum = sum(w for _, w in vals)
    if wsum <= 0:
        return None

    # Adaptive floor: keeps geometric non-compensation while avoiding
    # pathological near-zero collapse under sparse evidence.
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
    """Weighted arithmetic mean with None values removed and weights renormalized."""
    vals: List[Tuple[float, float]] = []
    for v, w in pairs:
        if v is None or w is None:
            continue
        try:
            vf = float(v)
            wf = float(w)
        except Exception:
            continue
        if wf <= 0:
            continue
        vals.append((clamp01(vf), wf))

    if not vals:
        return None
    wsum = sum(w for _, w in vals)
    if wsum <= 0:
        return None
    return clamp01(sum(v * w for v, w in vals) / wsum)


def sat_exp(x: Optional[float], scale: float = 1.0) -> Optional[float]:
    """
    Continuous saturation for non-negative counts: 1 - exp(-x/scale).

    Used only for small discrete evidence counts where a single artifact already
    carries meaningful evidence, such as the existence of one or more templates.
    """
    if x is None:
        return None
    try:
        xf = max(0.0, float(x))
        sf = max(1e-9, float(scale))
    except Exception:
        return None
    return clamp01(1.0 - math.exp(-xf / sf))


def _log_repo_signal(hits: Optional[float], ref: Optional[float]) -> Optional[float]:
    """
    Repository-relative logarithmic normalization.

    signal(h, h_ref) = log(1+h) / log(1+h_ref)

    This is useful for keyword evidence because it avoids both hard thresholds and
    trivial saturation in large repositories. h_ref should be the largest relevant
    hit count observed in the same repository.
    """
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


def ratio01(num: Optional[float], den: Optional[float]) -> Optional[float]:
    """Safe clamped ratio num/den. Returns None if denominator is unavailable or <= 0."""
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


def band(score_0_100: Optional[float]) -> Optional[str]:
    """RECOPS qualitative rating scale."""
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
    """Run a subprocess safely and return (returncode, stdout, stderr)."""
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
    """
    Fetch a GitHub repository.

    Prefer git clone because GOS uses git history for recency and traceability signals.
    Falls back to a zip download when git is unavailable. In zip mode, history-based
    recency signals become None instead of being forced to 0.
    """
    temp = Path(tempfile.mkdtemp(prefix="recops_gos_"))
    target = temp / "repo"

    code, _, _ = run(["git", "--version"])
    if code == 0:
        # Full history is not strictly required, but keeping tags/history improves
        # governance recency and ADR/roadmap modification evidence.
        c, _, _ = run(["git", "clone", "--filter=blob:none", "--tags", github_url, str(target)], cwd=temp, timeout_s=600)
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

    with open(zpath, "wb") as f:
        f.write(r.content)
    shutil.unpack_archive(str(zpath), str(temp))

    dirs = [p for p in temp.iterdir() if p.is_dir() and p.name != "repo"]
    if dirs:
        return dirs[0]
    raise RuntimeError("Cannot extract repository archive")


# =========================================================
# File scanning helpers
# =========================================================

IGNORE_DIRS = {
    ".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "node_modules", "dist", "build", ".venv", "venv", ".idea", ".vscode",
}

TEXT_SUFFIXES = {
    ".md", ".rst", ".txt", ".cff", ".toml", ".json", ".yaml", ".yml",
    ".ini", ".cfg", ".py", ".js", ".ts", ".java", ".jl", ".c", ".cpp",
}

DOC_DIR_NAMES = {"docs", "doc", ".github", "github", "community", "governance", "adr", "adrs", "rfcs", "rfc"}


def iter_files(root: Path) -> Iterable[Path]:
    """Recursive file iterator with noisy/generated directories ignored."""
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


def find_files_by_names(root: Path, names: Iterable[str]) -> List[Path]:
    wanted = {n.lower() for n in names}
    return [p for p in iter_files(root) if p.name.lower() in wanted]


def has_dir(root: Path, names: Iterable[str]) -> float:
    wanted = {n.lower() for n in names}
    try:
        for p in Path(root).iterdir():
            if p.is_dir() and p.name.lower() in wanted:
                return 1.0
    except Exception:
        pass
    return 0.0


def _is_doc_like_path(root: Path, path: Path) -> bool:
    """True for README/root governance files or files under documentation-like dirs."""
    try:
        rel = path.relative_to(root)
    except Exception:
        return False
    parts = [x.lower() for x in rel.parts]
    name = path.name.lower()
    if name.startswith("readme"):
        return True
    if any(part in DOC_DIR_NAMES for part in parts[:-1]):
        return True
    if name in {
        "license", "license.md", "license.txt", "copying", "notice", "contributing.md",
        "governance.md", "code_of_conduct.md", "security.md", "support.md",
        "maintainers", "maintainers.md", "owners", "owners.md", "roadmap.md",
    }:
        return True
    return False


def candidate_texts(root: Path) -> Dict[str, str]:
    """
    Read repository texts relevant to governance/openness.

    This intentionally avoids scanning every generated source file as documentation,
    while still allowing root metadata and docs/.github governance files to count.
    """
    out: Dict[str, str] = {}
    for p in iter_files(root):
        if p.suffix.lower() not in TEXT_SUFFIXES and p.name.lower() not in {"license", "copying", "notice", "maintainers", "owners"}:
            continue
        if not _is_doc_like_path(root, p):
            # Still include root metadata files because license/contribution info often lives there.
            try:
                rel = p.relative_to(root)
            except Exception:
                continue
            if len(rel.parts) != 1 or p.name.lower() not in {"pyproject.toml", "package.json", "citation.cff", "codemeta.json"}:
                continue
        txt = _safe_read_text(p)
        if txt:
            try:
                out[str(p.relative_to(root))] = txt
            except Exception:
                out[str(p)] = txt
    return out


def _count_regex_hits_in_texts(texts: Dict[str, str], patterns: Iterable[str]) -> int:
    rx = [re.compile(p, flags=re.I | re.M) for p in patterns]
    hits = 0
    for txt in texts.values():
        for r in rx:
            hits += len(r.findall(txt or ""))
    return hits


def _count_regex_hits_in_paths(root: Path, patterns: Iterable[str]) -> int:
    rx = [re.compile(p, flags=re.I) for p in patterns]
    hits = 0
    for p in iter_files(root):
        try:
            s = str(p.relative_to(root)).replace("\\", "/")
        except Exception:
            s = str(p)
        for r in rx:
            hits += len(r.findall(s))
    return hits


def _repo_hit_ref(*counts: Optional[int]) -> float:
    vals = [float(c) for c in counts if c is not None]
    return max([1.0, *vals])


# =========================================================
# Git helpers
# =========================================================

def _is_git_repo(repo: Path) -> bool:
    code, out, _ = run(["git", "rev-parse", "--is-inside-work-tree"], cwd=repo)
    return code == 0 and out.strip().lower() == "true"


def _git_last_change_days(repo_path: Path, file_path: Path, now: Optional[dt.datetime] = None) -> Optional[float]:
    """
    Age in days since the file last changed in git.

    Falls back to filesystem mtime only when git history is unavailable.
    """
    now = now or dt.datetime.utcnow()
    try:
        rel = str(file_path.relative_to(repo_path))
    except Exception:
        rel = str(file_path)

    code, out, _ = run(["git", "log", "-1", "--format=%ct", "--", rel], cwd=repo_path)
    if code == 0 and out.strip().isdigit():
        ts = int(out.strip())
        return max(0.0, (now - dt.datetime.utcfromtimestamp(ts)).total_seconds() / 86400.0)

    try:
        return max(0.0, (now - dt.datetime.utcfromtimestamp(int(file_path.stat().st_mtime))).total_seconds() / 86400.0)
    except Exception:
        return None


def _median(vals: List[float]) -> Optional[float]:
    xs = sorted([float(v) for v in vals if v is not None])
    if not xs:
        return None
    n = len(xs)
    mid = n // 2
    if n % 2 == 1:
        return xs[mid]
    return 0.5 * (xs[mid - 1] + xs[mid])


def _file_recency_score(repo_path: Path, files: List[Path], scale_days: float = 365.0) -> Tuple[Optional[float], Optional[float]]:
    """
    Continuous recency score for a group of governance files.

    score = exp(-median_age_days / scale_days)

    The scale controls smooth decay, not a pass/fail threshold. None is returned
    when no files are present or age cannot be estimated.
    """
    if not files:
        return None, None
    now = dt.datetime.utcnow()
    days: List[float] = []
    for p in files:
        d = _git_last_change_days(repo_path, p, now=now)
        if d is not None:
            days.append(float(d))
    med = _median(days)
    if med is None:
        return None, None
    return clamp01(math.exp(-med / float(scale_days))), med


# =========================================================
# GitHub API helpers — optional
# =========================================================

def gh_api_get(url: str, token: Optional[str] = None, accept: Optional[str] = None, timeout_s: int = 30) -> Optional[Any]:
    if requests is None:
        return None
    headers = {"User-Agent": "recops-gos/1.0"}
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


def github_repo_info(repo_url: Optional[str], github_token: Optional[str] = None) -> Optional[Dict[str, Any]]:
    owner, repo = parse_github_owner_repo(repo_url)
    if not owner or not repo:
        return None
    data = gh_api_get(f"https://api.github.com/repos/{owner}/{repo}", token=github_token)
    return data if isinstance(data, dict) else None


def github_community_profile(repo_url: Optional[str], github_token: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    GitHub community profile endpoint.

    This is only used as supplementary evidence; the score is not copied from
    GitHub's health_percentage because GOS should remain transparent and formula-based.
    """
    owner, repo = parse_github_owner_repo(repo_url)
    if not owner or not repo:
        return None
    data = gh_api_get(f"https://api.github.com/repos/{owner}/{repo}/community/profile", token=github_token)
    return data if isinstance(data, dict) else None


# =========================================================
# License parsing helpers
# =========================================================

SPDX_LICENSE_IDS = {
    # Common OSS licenses. This is intentionally not a ranking.
    "MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "ISC", "MPL-2.0",
    "GPL-2.0", "GPL-3.0", "LGPL-2.1", "LGPL-3.0", "AGPL-3.0",
    "EPL-2.0", "CDDL-1.0", "Unlicense", "CC0-1.0", "BSL-1.0", "Zlib",
    "Artistic-2.0", "EUPL-1.2", "BlueOak-1.0.0",
}

LICENSE_FILE_NAMES = {
    "license", "license.md", "license.txt", "licence", "licence.md", "licence.txt",
    "copying", "copying.md", "notice", "notice.md", "notice.txt",
}

LICENSE_TEXT_HINTS = [
    ("MIT", r"permission is hereby granted, free of charge"),
    ("Apache-2.0", r"apache license\s+version\s+2\.0"),
    ("BSD-3-Clause", r"redistribution and use in source and binary forms"),
    ("GPL-3.0", r"gnu general public license\s+version\s+3"),
    ("GPL-2.0", r"gnu general public license\s+version\s+2"),
    ("LGPL", r"gnu lesser general public license"),
    ("AGPL-3.0", r"gnu affero general public license"),
    ("MPL-2.0", r"mozilla public license\s+version\s+2\.0"),
    ("EPL-2.0", r"eclipse public license"),
]


def _extract_spdx_identifiers(text: str) -> List[str]:
    if not text:
        return []
    found: List[str] = []
    # Explicit SPDX lines.
    for m in re.findall(r"SPDX-License-Identifier:\s*([^\s]+)", text, flags=re.I):
        found.append(m.strip())
    # Direct common SPDX IDs in metadata.
    for lic in SPDX_LICENSE_IDS:
        if re.search(rf"(?<![A-Za-z0-9_.-]){re.escape(lic)}(?![A-Za-z0-9_.-])", text):
            found.append(lic)
    return sorted(set(found))


def _detect_license_from_text(text: str) -> List[str]:
    detected = set(_extract_spdx_identifiers(text))
    low = text.lower()
    for label, pattern in LICENSE_TEXT_HINTS:
        if re.search(pattern, low, flags=re.I | re.M):
            detected.add(label)
    return sorted(detected)


def _license_metadata_values(root: Path) -> List[str]:
    """Extract declared license values from common package/research metadata files."""
    values: List[str] = []

    pyproject = root / "pyproject.toml"
    if pyproject.exists():
        txt = _safe_read_text(pyproject) or ""
        # PEP 621: license = {text = "MIT"} or license = "MIT"
        for m in re.findall(r"(?im)^\s*license\s*=\s*['\"]([^'\"]+)['\"]", txt):
            values.append(m.strip())
        for m in re.findall(r"(?im)^\s*license\s*=\s*\{[^}]*text\s*=\s*['\"]([^'\"]+)['\"]", txt):
            values.append(m.strip())
        for m in re.findall(r"(?im)^\s*license\s*=\s*\{[^}]*file\s*=\s*['\"]([^'\"]+)['\"]", txt):
            values.append(m.strip())

    package_json = root / "package.json"
    if package_json.exists():
        data = _safe_json_load(package_json)
        if isinstance(data, dict):
            lic = data.get("license")
            if isinstance(lic, str):
                values.append(lic.strip())
            elif isinstance(lic, dict) and isinstance(lic.get("type"), str):
                values.append(lic["type"].strip())

    codemeta = root / "codemeta.json"
    if codemeta.exists():
        data = _safe_json_load(codemeta)
        if isinstance(data, dict):
            lic = data.get("license")
            if isinstance(lic, str):
                values.append(lic.strip())
            elif isinstance(lic, list):
                values.extend([str(x).strip() for x in lic if x])

    citation = root / "CITATION.cff"
    if citation.exists():
        txt = _safe_read_text(citation)
        parsed = _safe_yaml_load_text(txt)
        if isinstance(parsed, dict) and parsed.get("license"):
            values.append(str(parsed.get("license")).strip())
        elif txt:
            for m in re.findall(r"(?im)^\s*license\s*:\s*([^\n]+)", txt):
                values.append(m.strip().strip('"\''))

    return sorted(set(v for v in values if v))


def _license_consistency(local_detected: List[str], metadata_values: List[str], api_license: Optional[str]) -> Optional[float]:
    """
    Consistency score among available license declarations.

    It does not judge license type. It only checks whether declared identifiers are
    mutually compatible/overlapping.
    """
    sources: List[set] = []
    if local_detected:
        sources.append(set(local_detected))
    meta_detected: List[str] = []
    for v in metadata_values:
        meta_detected.extend(_extract_spdx_identifiers(v) or [v])
    if meta_detected:
        sources.append(set(meta_detected))
    if api_license:
        sources.append({api_license})

    if len(sources) < 2:
        return None

    # Pairwise overlap ratio: objective agreement measure across available sources.
    pairs = 0
    agree = 0
    for i in range(len(sources)):
        for j in range(i + 1, len(sources)):
            pairs += 1
            if sources[i].intersection(sources[j]):
                agree += 1
    return ratio01(agree, pairs)


# =========================================================
# GOS block 1 — License openness and legal clarity
# =========================================================

def compute_license_openness(
    repo_path: Path,
    repo_url: Optional[str] = None,
    github_token: Optional[str] = None,
) -> Tuple[Optional[float], Dict[str, Any]]:
    """
    N_license = mean(L_file_presence, L_identifier_detected, L_metadata_declared,
                     L_consistency, L_recency)

    This block measures whether reuse rights are explicit and machine/human readable.
    It deliberately does not rank permissive vs copyleft licenses.
    """
    repo_path = Path(repo_path)
    license_files = [p for p in iter_files(repo_path) if p.name.lower() in LICENSE_FILE_NAMES]
    l_file_presence = 1.0 if license_files else 0.0

    local_text = "\n".join((_safe_read_text(p) or "") for p in license_files)
    local_detected = _detect_license_from_text(local_text) if local_text else []
    l_identifier_detected = 1.0 if local_detected else 0.0

    metadata_values = _license_metadata_values(repo_path)
    l_metadata_declared = 1.0 if metadata_values else 0.0

    repo_info = github_repo_info(repo_url, github_token=github_token) if repo_url else None
    api_license = None
    if isinstance(repo_info, dict) and isinstance(repo_info.get("license"), dict):
        api_license = repo_info.get("license", {}).get("spdx_id")
        if api_license == "NOASSERTION":
            api_license = None

    l_consistency = _license_consistency(local_detected, metadata_values, api_license)
    l_recency, median_license_days = _file_recency_score(repo_path, license_files, scale_days=365.0)

    block = mean([l_file_presence, l_identifier_detected, l_metadata_declared, l_consistency, l_recency])
    return block, {
        "available": True,
        "block": block,
        "sub": {
            "L_file_presence": l_file_presence,
            "L_identifier_detected": l_identifier_detected,
            "L_metadata_declared": l_metadata_declared,
            "L_consistency": l_consistency,
            "L_recency": l_recency,
        },
        "raw": {
            "license_files": [str(p.relative_to(repo_path)) for p in license_files],
            "local_detected_licenses": local_detected,
            "metadata_license_values": metadata_values,
            "github_api_license_spdx": api_license,
            "median_license_file_age_days": median_license_days,
        },
        "missingness_note": "L_consistency is None unless at least two independent license declarations are available.",
    }


# =========================================================
# GOS block 2 — Contribution openness
# =========================================================

CONTRIBUTING_NAMES = {
    "contributing", "contributing.md", "contributing.rst", "contributing.txt",
    "development.md", "developers.md", "developer.md",
}

PR_TEMPLATE_NAMES = {
    "pull_request_template.md", "pull_request_template.txt", "pull_request_template.yml", "pull_request_template.yaml",
}

ISSUE_TEMPLATE_HINTS = [
    r"(^|/)\.github/issue_template(/|$)",
    r"issue_template",
    r"bug_report",
    r"feature_request",
]

CONTRIBUTION_KEYWORDS = [
    r"\bcontribut(?:e|ing|ion|or|ors)\b",
    r"\bpull request\b",
    r"\bmerge request\b",
    r"\bdeveloper guide\b",
    r"\bdevelopment setup\b",
    r"\bcode review\b",
    r"\bissue\b",
    r"\bbug report\b",
    r"\bfeature request\b",
    r"\bgood first issue\b",
]


def _template_counts(repo_path: Path) -> Dict[str, int]:
    issue_templates = 0
    pr_templates = 0
    config_templates = 0
    for p in iter_files(repo_path):
        try:
            rel = str(p.relative_to(repo_path)).replace("\\", "/").lower()
        except Exception:
            rel = str(p).lower()
        name = p.name.lower()
        if any(re.search(rx, rel) for rx in ISSUE_TEMPLATE_HINTS):
            issue_templates += 1
        if name in PR_TEMPLATE_NAMES or "/pull_request_template" in rel:
            pr_templates += 1
        if ".github/issue_template/config" in rel:
            config_templates += 1
    return {
        "issue_templates": issue_templates,
        "pr_templates": pr_templates,
        "config_templates": config_templates,
        "total_templates": issue_templates + pr_templates + config_templates,
    }


def compute_contribution_openness(repo_path: Path) -> Tuple[Optional[float], Dict[str, Any]]:
    """
    N_contribution = mean(C_contributing_file, C_templates, C_pr_template,
                          C_guidance_density, C_onboarding_recency)

    This block measures whether external users can understand how to contribute.
    """
    repo_path = Path(repo_path)
    texts = candidate_texts(repo_path)
    contributing_files = [p for p in iter_files(repo_path) if p.name.lower() in CONTRIBUTING_NAMES]
    c_contributing_file = 1.0 if contributing_files else 0.0

    tpl = _template_counts(repo_path)
    # Objective ratio over observed template families: issue template and PR template.
    c_issue_templates = 1.0 if tpl["issue_templates"] > 0 else 0.0
    c_pr_template = 1.0 if tpl["pr_templates"] > 0 else 0.0
    c_templates = mean([c_issue_templates, c_pr_template])

    guidance_hits = _count_regex_hits_in_texts(texts, CONTRIBUTION_KEYWORDS)
    # Compare guidance density to other governance keyword families observed later.
    # Here the block-local ref is at least guidance_hits, so any non-zero guidance can
    # get evidence without hard thresholds. Cross-block refs are not needed.
    c_guidance_density = _log_repo_signal(guidance_hits, max(1.0, guidance_hits))

    onboarding_files = contributing_files[:]
    for p in iter_files(repo_path):
        name = p.name.lower()
        try:
            rel = str(p.relative_to(repo_path)).replace("\\", "/").lower()
        except Exception:
            rel = name
        if name in PR_TEMPLATE_NAMES or "issue_template" in rel:
            onboarding_files.append(p)
    c_onboarding_recency, median_onboarding_days = _file_recency_score(repo_path, onboarding_files, scale_days=365.0)

    block = mean([c_contributing_file, c_templates, c_pr_template, c_guidance_density, c_onboarding_recency])
    return block, {
        "available": True,
        "block": block,
        "sub": {
            "C_contributing_file": c_contributing_file,
            "C_templates": c_templates,
            "C_pr_template": c_pr_template,
            "C_guidance_density": c_guidance_density,
            "C_onboarding_recency": c_onboarding_recency,
        },
        "raw": {
            "contributing_files": [str(p.relative_to(repo_path)) for p in contributing_files],
            "template_counts": tpl,
            "guidance_keyword_hits": guidance_hits,
            "median_onboarding_file_age_days": median_onboarding_days,
        },
        "missingness_note": "C_onboarding_recency is None when no contribution-related files/templates exist.",
    }


# =========================================================
# GOS block 3 — Community and responsible-disclosure safeguards
# =========================================================

CODE_OF_CONDUCT_NAMES = {
    "code_of_conduct.md", "code-of-conduct.md", "code_of_conduct.rst", "code-of-conduct.rst",
}
SECURITY_NAMES = {"security.md", "security.rst", "security.txt"}
SUPPORT_NAMES = {"support.md", "support.rst", "support.txt"}

SAFEGUARD_KEYWORDS = [
    r"\bcode of conduct\b",
    r"\bcovenant\b",
    r"\bmoderation\b",
    r"\babuse\b",
    r"\bharassment\b",
    r"\bresponsible disclosure\b",
    r"\bvulnerability disclosure\b",
    r"\bsecurity policy\b",
    r"\bsecurity advisory\b",
    r"\bsupport policy\b",
    r"\bcontact\b",
]


def compute_community_safeguards(repo_path: Path) -> Tuple[Optional[float], Dict[str, Any]]:
    """
    N_safeguards = mean(S_code_of_conduct, S_security_policy, S_support_policy,
                        S_safeguard_guidance, S_policy_recency)

    This block captures whether the project documents community norms and secure
    vulnerability-reporting/support paths.
    """
    repo_path = Path(repo_path)
    texts = candidate_texts(repo_path)

    coc_files = [p for p in iter_files(repo_path) if p.name.lower() in CODE_OF_CONDUCT_NAMES]
    sec_files = [p for p in iter_files(repo_path) if p.name.lower() in SECURITY_NAMES]
    support_files = [p for p in iter_files(repo_path) if p.name.lower() in SUPPORT_NAMES]

    s_code_of_conduct = 1.0 if coc_files else 0.0
    s_security_policy = 1.0 if sec_files else 0.0
    s_support_policy = 1.0 if support_files else 0.0

    safeguard_hits = _count_regex_hits_in_texts(texts, SAFEGUARD_KEYWORDS)
    s_safeguard_guidance = _log_repo_signal(safeguard_hits, max(1.0, safeguard_hits))

    policy_files = coc_files + sec_files + support_files
    s_policy_recency, median_policy_days = _file_recency_score(repo_path, policy_files, scale_days=365.0)

    block = mean([s_code_of_conduct, s_security_policy, s_support_policy, s_safeguard_guidance, s_policy_recency])
    return block, {
        "available": True,
        "block": block,
        "sub": {
            "S_code_of_conduct": s_code_of_conduct,
            "S_security_policy": s_security_policy,
            "S_support_policy": s_support_policy,
            "S_safeguard_guidance": s_safeguard_guidance,
            "S_policy_recency": s_policy_recency,
        },
        "raw": {
            "code_of_conduct_files": [str(p.relative_to(repo_path)) for p in coc_files],
            "security_files": [str(p.relative_to(repo_path)) for p in sec_files],
            "support_files": [str(p.relative_to(repo_path)) for p in support_files],
            "safeguard_keyword_hits": safeguard_hits,
            "median_policy_file_age_days": median_policy_days,
        },
        "missingness_note": "Policy recency is None when no community/security/support policy file exists.",
    }


# =========================================================
# GOS block 4 — Governance transparency and decision traceability
# =========================================================

GOVERNANCE_NAMES = {
    "governance.md", "governance.rst", "charter.md", "charter.rst",
}
MAINTAINER_NAMES = {
    "maintainers", "maintainers.md", "maintainers.rst", "owners", "owners.md",
    "owners.rst", "codeowners",
}
ROADMAP_NAMES = {
    "roadmap.md", "roadmap.rst", "roadmap.txt", "milestones.md", "milestones.rst",
}

TRANSPARENCY_KEYWORDS = [
    r"\bgovernance\b",
    r"\bmaintainer(?:s)?\b",
    r"\bsteering committee\b",
    r"\btechnical committee\b",
    r"\bdecision(?:s)?\b",
    r"\bvote\b",
    r"\brfc\b",
    r"\badr\b",
    r"\barchitecture decision record\b",
    r"\broadmap\b",
    r"\brelease plan\b",
]


def _adr_rfc_files(repo_path: Path) -> List[Path]:
    out: List[Path] = []
    for p in iter_files(repo_path):
        try:
            rel = str(p.relative_to(repo_path)).replace("\\", "/").lower()
        except Exception:
            rel = str(p).lower()
        name = p.name.lower()
        if "/adr/" in f"/{rel}" or "/adrs/" in f"/{rel}" or "/rfcs/" in f"/{rel}" or "/rfc/" in f"/{rel}":
            if p.suffix.lower() in {".md", ".rst", ".txt"}:
                out.append(p)
        elif re.search(r"(^|[-_])adr[-_]?\d+", name) or re.search(r"(^|[-_])rfc[-_]?\d+", name):
            out.append(p)
    return out


def compute_governance_transparency(repo_path: Path) -> Tuple[Optional[float], Dict[str, Any]]:
    """
    N_transparency = mean(T_governance_doc, T_maintainers_doc, T_roadmap,
                          T_decision_records, T_transparency_guidance,
                          T_governance_recency)

    This block measures whether decision rights, maintainers, roadmap, and design
    decisions are visible and traceable.
    """
    repo_path = Path(repo_path)
    texts = candidate_texts(repo_path)

    governance_files = [p for p in iter_files(repo_path) if p.name.lower() in GOVERNANCE_NAMES]
    maintainer_files = [p for p in iter_files(repo_path) if p.name.lower() in MAINTAINER_NAMES]
    roadmap_files = [p for p in iter_files(repo_path) if p.name.lower() in ROADMAP_NAMES]
    adr_files = _adr_rfc_files(repo_path)

    t_governance_doc = 1.0 if governance_files else 0.0
    t_maintainers_doc = 1.0 if maintainer_files else 0.0
    t_roadmap = 1.0 if roadmap_files else 0.0

    # Objective continuous evidence: more decision records add evidence, but with
    # diminishing returns. Scale=1 means the first ADR/RFC carries strong evidence.
    t_decision_records = sat_exp(len(adr_files), scale=1.0)

    transparency_hits = _count_regex_hits_in_texts(texts, TRANSPARENCY_KEYWORDS)
    t_transparency_guidance = _log_repo_signal(transparency_hits, max(1.0, transparency_hits))

    governance_related_files = governance_files + maintainer_files + roadmap_files + adr_files
    t_governance_recency, median_gov_days = _file_recency_score(repo_path, governance_related_files, scale_days=365.0)

    block = mean([
        t_governance_doc,
        t_maintainers_doc,
        t_roadmap,
        t_decision_records,
        t_transparency_guidance,
        t_governance_recency,
    ])
    return block, {
        "available": True,
        "block": block,
        "sub": {
            "T_governance_doc": t_governance_doc,
            "T_maintainers_doc": t_maintainers_doc,
            "T_roadmap": t_roadmap,
            "T_decision_records": t_decision_records,
            "T_transparency_guidance": t_transparency_guidance,
            "T_governance_recency": t_governance_recency,
        },
        "raw": {
            "governance_files": [str(p.relative_to(repo_path)) for p in governance_files],
            "maintainer_files": [str(p.relative_to(repo_path)) for p in maintainer_files],
            "roadmap_files": [str(p.relative_to(repo_path)) for p in roadmap_files],
            "adr_rfc_files_count": len(adr_files),
            "adr_rfc_files_sample": [str(p.relative_to(repo_path)) for p in adr_files[:50]],
            "transparency_keyword_hits": transparency_hits,
            "median_governance_file_age_days": median_gov_days,
        },
        "missingness_note": "Governance recency is None when no governance/maintainer/roadmap/ADR file exists.",
    }


# =========================================================

# =========================================================
# SCS — Scientific Credibility Score
# =========================================================

SOURCE_SUFFIXES = {
    ".py", ".js", ".ts", ".java", ".cpp", ".c", ".cs", ".jl", ".r", ".m",
    ".gms", ".mod", ".dat", ".inc", ".lp", ".mps",
}
DOI_RX = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+", flags=re.I)
ARXIV_RX = re.compile(r"\barXiv[: ]?(\d{4}\.\d{4,5}(?:v\d+)?)\b", flags=re.I)

PUBLICATION_PATTERNS = [
    r"\bdoi\b", r"\bjournal\b", r"\bconference\b", r"\bproceedings\b", r"\bpreprint\b",
    r"\barxiv\b", r"\bzenodo\b", r"\bpublication(?:s)?\b", r"\bcite\b", r"\bcitation\b",
    r"\bpeer[- ]reviewed\b", r"\bpaper(?:s)?\b", r"\barticle(?:s)?\b",
]
VALIDATION_PATTERNS = [
    r"\bvalidation\b", r"\bverification\b", r"\bvalidate[sd]?\b", r"\bverified\b",
    r"\bunit test\b", r"\bintegration test\b", r"\bregression test\b", r"\bbenchmark\b",
    r"\bcomparison\b", r"\bbaseline\b", r"\breference result\b", r"\bexpected output\b",
    r"\bgolden\b", r"\baccuracy\b", r"\berror metric\b",
]
METHODOLOGY_PATTERNS = [
    r"\bmethodology\b", r"\bmethod\b", r"\balgorithm\b", r"\bmodel formulation\b",
    r"\bassumption(?:s)?\b", r"\bequation(?:s)?\b", r"\bderivation\b", r"\breproducib",
    r"\breplication\b", r"\bexample(?:s)?\b", r"\btutorial\b", r"\bnotebook(?:s)?\b",
    r"\bdataset(?:s)?\b", r"\binput data\b", r"\bcase stud(?:y|ies)\b",
]


def source_files(root: Path) -> List[Path]:
    return [p for p in iter_files(root) if p.suffix.lower() in SOURCE_SUFFIXES]


def count_source_loc(root: Path) -> int:
    total = 0
    for p in source_files(root):
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                total += sum(1 for _ in f)
        except Exception:
            pass
    return total


def _workflow_texts(root: Path) -> List[str]:
    wf = Path(root) / ".github" / "workflows"
    if not wf.exists():
        return []
    texts = []
    for p in wf.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".yml", ".yaml"}:
            txt = _safe_read_text(p)
            if txt:
                texts.append(txt)
    return texts


def _workflow_signal(root: Path, patterns: Iterable[str]) -> float:
    texts = _workflow_texts(root)
    if not texts:
        return 0.0
    rx = [re.compile(p, flags=re.I | re.M) for p in patterns]
    hits = sum(1 for txt in texts if any(r.search(txt or "") for r in rx))
    return sat_exp(hits, scale=2.0) or 0.0


def count_test_files(root: Path) -> int:
    pats = [
        re.compile(r"(^|/|\\)tests?(/|\\)", re.I),
        re.compile(r"(^|/|\\)test_", re.I),
        re.compile(r"_test\.", re.I),
        re.compile(r"\.test\.", re.I),
        re.compile(r"\.spec\.", re.I),
    ]
    count = 0
    for p in iter_files(root):
        if p.suffix.lower() not in SOURCE_SUFFIXES | {".md", ".rst", ".txt"}:
            continue
        rel = str(p.relative_to(root)).replace("\\", "/")
        if any(rx.search(rel) for rx in pats):
            count += 1
    return count


def _count_pattern_hits_repo(root: Path, patterns: Iterable[str], suffixes: Optional[set] = None) -> int:
    suffixes = suffixes or (TEXT_SUFFIXES | SOURCE_SUFFIXES | {".ipynb", ".bib", ".ris"})
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


def _extract_publication_metadata(repo_path: Path) -> Dict[str, Any]:
    texts = candidate_texts(repo_path)
    citation_files = find_files_by_names(repo_path, ["CITATION.cff", "citation.cff", "codemeta.json", "zenodo.json", ".zenodo.json"])
    bibliography_files = [p for p in iter_files(repo_path) if p.suffix.lower() in {".bib", ".ris"}]
    blob = "\n".join(texts.values())
    for p in citation_files + bibliography_files:
        blob += "\n" + (_safe_read_text(p) or "")
    dois = sorted(set(m.rstrip(".,;)]}") for m in DOI_RX.findall(blob)))
    arxivs = sorted(set(ARXIV_RX.findall(blob)))
    structured_records = 0
    citation_cff_complete = None
    cff = repo_path / "CITATION.cff"
    if cff.exists():
        structured_records += 1
        parsed = _safe_yaml_load_text(_safe_read_text(cff))
        if isinstance(parsed, dict):
            required = ["title", "authors", "date-released"]
            citation_cff_complete = ratio01(sum(1 for k in required if parsed.get(k)), len(required))
            if parsed.get("doi"):
                dois.append(str(parsed.get("doi")).strip())
    codemeta = repo_path / "codemeta.json"
    if codemeta.exists():
        structured_records += 1
        data = _safe_json_load(codemeta)
        if isinstance(data, dict):
            for key in ["identifier", "sameAs", "citation"]:
                val = data.get(key)
                vals = val if isinstance(val, list) else [val]
                for x in vals:
                    if isinstance(x, str):
                        dois.extend(DOI_RX.findall(x))
    bib_entries = 0
    for p in bibliography_files:
        bib_entries += len(re.findall(r"@(?:article|inproceedings|proceedings|book|misc|software|dataset)\s*\{", _safe_read_text(p) or "", flags=re.I))
    return {
        "texts": texts,
        "citation_files": citation_files,
        "bibliography_files": bibliography_files,
        "structured_records": structured_records,
        "citation_cff_completeness": citation_cff_complete,
        "dois": sorted(set(d for d in dois if d)),
        "arxivs": arxivs,
        "bib_entries": bib_entries,
        "publication_keyword_hits": _count_regex_hits_in_texts(texts, PUBLICATION_PATTERNS),
    }


def _fetch_crossref_work(doi: str) -> Optional[Dict[str, Any]]:
    if requests is None or not doi:
        return None
    try:
        r = requests.get(f"https://api.crossref.org/works/{doi}", headers={"User-Agent": "recops-scs/1.0"}, timeout=20)
        if r.status_code != 200:
            return None
        data = r.json()
        msg = data.get("message") if isinstance(data, dict) else None
        return msg if isinstance(msg, dict) else None
    except Exception:
        return None


def _external_publication_evidence(dois: List[str], max_dois: int = 5) -> Dict[str, Any]:
    works = []
    for doi in dois[:max_dois]:
        w = _fetch_crossref_work(doi)
        if isinstance(w, dict):
            works.append(w)
    if not works:
        return {"external_metadata_available": 0.0, "external_works_checked": 0, "total_reference_count": None, "total_is_referenced_by_count": None, "venue_titles": [], "published_years": []}
    refs, cited, venues, years = [], [], [], []
    for w in works:
        if isinstance(w.get("reference-count"), int):
            refs.append(w["reference-count"])
        if isinstance(w.get("is-referenced-by-count"), int):
            cited.append(w["is-referenced-by-count"])
        for key in ["container-title", "short-container-title"]:
            val = w.get(key)
            if isinstance(val, list):
                venues.extend(str(x) for x in val if x)
        issued = w.get("published-print") or w.get("published-online") or w.get("issued")
        parts = issued.get("date-parts") if isinstance(issued, dict) else None
        if isinstance(parts, list) and parts and isinstance(parts[0], list) and parts[0]:
            try:
                years.append(int(parts[0][0]))
            except Exception:
                pass
    return {"external_metadata_available": 1.0, "external_works_checked": len(works), "total_reference_count": sum(refs) if refs else None, "total_is_referenced_by_count": sum(cited) if cited else None, "venue_titles": sorted(set(venues)), "published_years": sorted(set(years))}


def _calibrated_percentile_score(value: Optional[float], corpus_values: Optional[List[float]]) -> Optional[float]:
    if value is None or not corpus_values:
        return None
    vals = sorted(float(v) for v in corpus_values if v is not None)
    if not vals:
        return None
    return ratio01(sum(1 for v in vals if v <= float(value)), len(vals))


def compute_publication_record(repo_path: Path, calibration: Optional[Dict[str, Any]] = None) -> Tuple[Optional[float], Dict[str, Any]]:
    meta = _extract_publication_metadata(repo_path)
    external = _external_publication_evidence(meta["dois"])
    p_citation_file = 1.0 if meta["citation_files"] else 0.0
    p_identifier_density = mean([sat_exp(len(meta["dois"]), scale=1.0), sat_exp(len(meta["arxivs"]), scale=1.0)])
    p_structured_metadata = mean([sat_exp(meta["structured_records"], scale=1.0), meta["citation_cff_completeness"]])
    p_bibliography = sat_exp(meta["bib_entries"], scale=1.0) if meta["bibliography_files"] else 0.0
    p_publication_mentions = _log_repo_signal(meta["publication_keyword_hits"], max(1.0, meta["publication_keyword_hits"]))
    p_external_metadata = external["external_metadata_available"]
    p_citation_impact = _calibrated_percentile_score(external.get("total_is_referenced_by_count"), (calibration or {}).get("citation_counts") if isinstance(calibration, dict) else None)
    block = mean([p_citation_file, p_identifier_density, p_structured_metadata, p_bibliography, p_publication_mentions, p_external_metadata, p_citation_impact])
    return block, {"available": True, "block": block, "sub": {"P_citation_file": p_citation_file, "P_identifier_density": p_identifier_density, "P_structured_metadata": p_structured_metadata, "P_bibliography": p_bibliography, "P_publication_mentions": p_publication_mentions, "P_external_metadata": p_external_metadata, "P_citation_impact_calibrated": p_citation_impact}, "raw": {"citation_files": [str(p.relative_to(repo_path)) for p in meta["citation_files"]], "bibliography_files": [str(p.relative_to(repo_path)) for p in meta["bibliography_files"]], "doi_count": len(meta["dois"]), "dois_sample": meta["dois"][:20], "arxiv_ids": meta["arxivs"], "bib_entries": meta["bib_entries"], "publication_keyword_hits": meta["publication_keyword_hits"], "external": external}, "missingness_note": "Citation impact is scored only with a supplied corpus distribution; otherwise raw citation counts are reported but not normalized arbitrarily."}


def compute_validation_verification(repo_path: Path) -> Tuple[Optional[float], Dict[str, Any]]:
    tests = count_test_files(repo_path)
    src_files = len(source_files(repo_path))
    loc = count_source_loc(repo_path)
    v_test_density = ratio01(tests, max(1, math.sqrt(max(1, src_files))))
    v_ci_validation = _workflow_signal(repo_path, [r"\bpytest\b", r"\bunittest\b", r"\bnpm test\b", r"\byarn test\b", r"\bpnpm test\b", r"\bmvn test\b", r"\bgradle test\b", r"\bcoverage\b", r"\bbenchmark\b", r"\bvalidation\b"])
    benchmark_path_hits = _count_regex_hits_in_paths(repo_path, [r"benchmark", r"benchmarks", r"asv"])
    benchmark_text_hits = _count_pattern_hits_repo(repo_path, [r"benchmark", r"pytest-benchmark", r"asv", r"performance", r"runtime", r"speed"])
    v_benchmark_evidence = mean([sat_exp(benchmark_path_hits, scale=1.0), _log_repo_signal(benchmark_text_hits, max(1.0, benchmark_text_hits))])
    texts = candidate_texts(repo_path)
    validation_doc_hits = _count_regex_hits_in_texts(texts, VALIDATION_PATTERNS)
    v_validation_docs = _log_repo_signal(validation_doc_hits, max(1.0, validation_doc_hits))
    reference_artifacts = 0
    for p in iter_files(repo_path):
        rel = str(p.relative_to(repo_path)).replace("\\", "/").lower()
        if re.search(r"(reference|expected|baseline|golden|validation|verification|case)[-_ ]", rel) and p.suffix.lower() in {".csv", ".json", ".yaml", ".yml", ".txt", ".npz", ".h5", ".nc", ".parquet"}:
            reference_artifacts += 1
    v_reference_artifacts = sat_exp(reference_artifacts, scale=1.0)
    repro_files = sum(1 for p in iter_files(repo_path) if p.name.lower() in {"environment.yml", "environment.yaml", "requirements.txt", "pyproject.toml", "poetry.lock", "package-lock.json", "dockerfile", "docker-compose.yml"})
    examples_present = has_dir(repo_path, ["examples", "example", "notebooks", "tutorials", "cases", "case_studies"])
    v_reproducible_execution = mean([sat_exp(repro_files, scale=2.0), examples_present])
    block = mean([v_test_density, v_ci_validation, v_benchmark_evidence, v_validation_docs, v_reference_artifacts, v_reproducible_execution])
    return block, {"available": True, "block": block, "sub": {"V_test_density": v_test_density, "V_ci_validation": v_ci_validation, "V_benchmark_evidence": v_benchmark_evidence, "V_validation_docs": v_validation_docs, "V_reference_artifacts": v_reference_artifacts, "V_reproducible_execution": v_reproducible_execution}, "raw": {"test_files": tests, "source_files": src_files, "source_loc": loc, "benchmark_path_hits": benchmark_path_hits, "benchmark_text_hits": benchmark_text_hits, "validation_doc_hits": validation_doc_hits, "reference_artifact_count": reference_artifacts, "reproducibility_file_count": repro_files}, "missingness_note": "Validation signals are repository-observable; absence of tests/CI/artifacts is scored 0, not None."}


def compute_methodology_transparency(repo_path: Path) -> Tuple[Optional[float], Dict[str, Any]]:
    texts = candidate_texts(repo_path)
    method_files = []
    for p in iter_files(repo_path):
        rel = str(p.relative_to(repo_path)).replace("\\", "/").lower()
        if p.suffix.lower() in {".md", ".rst", ".txt", ".ipynb"} and re.search(r"method|algorithm|model|theory|formulation|reproduc", rel):
            method_files.append(p)
    m_methods_doc = sat_exp(len(method_files), scale=1.0)
    method_hits = _count_regex_hits_in_texts(texts, METHODOLOGY_PATTERNS)
    m_algorithm_density = _log_repo_signal(method_hits, max(1.0, method_hits))
    example_dirs = has_dir(repo_path, ["examples", "example", "notebooks", "tutorials", "demo", "demos", "cases", "case_studies"])
    notebook_count = sum(1 for p in iter_files(repo_path) if p.suffix.lower() == ".ipynb")
    m_examples = mean([example_dirs, sat_exp(notebook_count, scale=2.0)])
    repro_files = find_files_by_names(repo_path, ["requirements.txt", "environment.yml", "environment.yaml", "pyproject.toml", "poetry.lock", "Pipfile.lock", "package-lock.json", "renv.lock", "Manifest.toml", "Dockerfile", "docker-compose.yml", "reproducibility.md", "reproduce.md"])
    m_reproducibility = sat_exp(len(repro_files), scale=2.0)
    data_hits = _count_regex_hits_in_texts(texts, [r"\bdataset(?:s)?\b", r"\bdata source(?:s)?\b", r"\binput data\b", r"\bzenodo\b", r"\bfigshare\b", r"\bopen data\b", r"\blicense(?:d)? data\b", r"\bcase data\b"])
    data_dirs = has_dir(repo_path, ["data", "datasets", "inputs", "cases", "case_data"])
    m_data_transparency = mean([_log_repo_signal(data_hits, max(1.0, data_hits)), data_dirs])
    if _is_git_repo(repo_path):
        code, out, _ = run(["git", "rev-list", "--count", "HEAD"], cwd=repo_path)
        commit_count = int(out.strip()) if code == 0 and out.strip().isdigit() else None
        m_history_openness = 1.0 if commit_count and commit_count > 1 else 0.0
    else:
        commit_count = None
        m_history_openness = None
    block = mean([m_methods_doc, m_algorithm_density, m_examples, m_reproducibility, m_data_transparency, m_history_openness])
    return block, {"available": True, "block": block, "sub": {"M_methods_doc": m_methods_doc, "M_algorithm_density": m_algorithm_density, "M_examples": m_examples, "M_reproducibility": m_reproducibility, "M_data_transparency": m_data_transparency, "M_history_openness": m_history_openness}, "raw": {"method_files": [str(p.relative_to(repo_path)) for p in method_files[:50]], "method_file_count": len(method_files), "methodology_keyword_hits": method_hits, "notebook_count": notebook_count, "reproducibility_files": [str(p.relative_to(repo_path)) for p in repro_files[:50]], "data_keyword_hits": data_hits, "git_commit_count": commit_count}, "missingness_note": "Git history openness is None only when the analyzed input is not a git repository."}


def learn_scs_weights(dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    rows = []
    for row in dataset:
        rows.append({"publication_record": row.get("publication_record", row.get("N_publication")), "validation_verification": row.get("validation_verification", row.get("N_validation")), "methodology_transparency": row.get("methodology_transparency", row.get("N_methodology"))})
    weights = _norm_weights_from_rows(rows, ["publication_record", "validation_verification", "methodology_transparency"])
    return {"weights": weights, "n_repos": len(dataset)}


def compute_scs(repo_path: Path, repo_url: Optional[str] = None, github_token: Optional[str] = None, learned_weights: Optional[Dict[str, float]] = None, calibration: Optional[Dict[str, Any]] = None, aggregation: str = "geometric") -> Dict[str, Any]:
    repo_path = Path(repo_path)
    n_publication, publication_details = compute_publication_record(repo_path, calibration=calibration)
    n_validation, validation_details = compute_validation_verification(repo_path)
    n_methodology, methodology_details = compute_methodology_transparency(repo_path)
    blocks = {"publication_record": n_publication, "validation_verification": n_validation, "methodology_transparency": n_methodology}
    weights_source = "learned_from_corpus" if learned_weights is not None else "repo_data_driven_fallback"
    weights = learned_weights or _data_driven_weights_from_blocks(blocks)
    pairs = [(n_publication, float(weights.get("publication_record", weights.get("N_publication", 0.0)))), (n_validation, float(weights.get("validation_verification", weights.get("N_validation", 0.0)))), (n_methodology, float(weights.get("methodology_transparency", weights.get("N_methodology", 0.0))))]
    final_0_1 = weighted_sum(pairs) if (aggregation or "").strip().lower() == "sum" else weighted_geometric(pairs)
    score = 100.0 * float(final_0_1) if final_0_1 is not None else None
    return {"repository_path": str(repo_path), "repository_url": repo_url, "score": score, "band": band(score), "aggregation": "weighted_sum" if (aggregation or "").strip().lower() == "sum" else "weighted_geometric", "details": {"SCS": score, "N_publication": n_publication, "N_validation": n_validation, "N_methodology": n_methodology, "weights": weights, "weights_source": weights_source, "blocks": {"publication_record": publication_details, "validation_verification": validation_details, "methodology_transparency": methodology_details}, "missingness_note": "Unavailable values remain None and are excluded from means/weighted aggregation; observable absence is scored as 0.0.", "method_note": "SCS evaluates observable scientific-credibility evidence in the repository and optional DOI metadata; it does not infer actual scientific truth or peer-review quality without data."}}
# Data-driven weight learning and fallback
# =========================================================

def _data_driven_weights_from_blocks(blocks: Dict[str, Optional[float]]) -> Dict[str, float]:
    """
    Repository-level fallback when corpus-learned weights are unavailable.

    Principle copied from the PH design: observed blocks farther from the repository's
    cross-block mean receive more weight because they are more discriminative for
    that repository. If all observed blocks are identical, use equal weights across
    observed blocks. Missing blocks get weight 0.
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


def _robust_variance(vals: List[float]) -> float:
    if len(vals) < 2:
        return 0.0
    m = _py_mean(vals)
    return _py_mean([(x - m) ** 2 for x in vals])


def _pairwise_abs_corr(a: List[Optional[float]], b: List[Optional[float]]) -> Optional[float]:
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if len(pairs) < 3:
        return None
    xs = [float(x) for x, _ in pairs]
    ys = [float(y) for _, y in pairs]
    mx = _py_mean(xs)
    my = _py_mean(ys)
    vx = _py_mean([(x - mx) ** 2 for x in xs])
    vy = _py_mean([(y - my) ** 2 for y in ys])
    if vx <= 0 or vy <= 0:
        return None
    cov = _py_mean([(x - mx) * (y - my) for x, y in pairs])
    return abs(cov / math.sqrt(vx * vy))


def _norm_weights_from_rows(rows: List[Dict[str, Optional[float]]], keys: List[str]) -> Dict[str, float]:
    """
    Learn weights from a corpus using:
    raw_weight(k) = variance(k) * coverage(k) * uniqueness(k)

    - variance: more discriminative blocks matter more.
    - coverage: more often available blocks matter more.
    - uniqueness: less redundant blocks matter more.
    """
    vals = {k: [r.get(k) for r in rows] for k in keys}

    def coverage(arr: List[Optional[float]]) -> float:
        return sum(1 for x in arr if x is not None) / max(1, len(arr))

    info = {k: _robust_variance([float(x) for x in vals[k] if x is not None]) for k in keys}
    covg = {k: coverage(vals[k]) for k in keys}
    uniq: Dict[str, float] = {}

    for k in keys:
        corrs: List[Optional[float]] = []
        for j in keys:
            if j == k:
                continue
            corrs.append(_pairwise_abs_corr(vals[k], vals[j]))
        c = mean(corrs)
        uniq[k] = clamp01(1.0 - c) if c is not None else 1.0

    raw = {k: max(1e-9, info[k] * covg[k] * uniq[k]) for k in keys}
    z = sum(raw.values()) or 1.0
    return {k: raw[k] / z for k in keys}


def learn_gos_weights(dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Learn GOS block weights from a corpus of repository block values.

    Accepted row keys:
    - license_openness or N_license
    - contribution_openness or N_contribution
    - community_safeguards or N_safeguards
    - governance_transparency or N_transparency
    """
    rows: List[Dict[str, Optional[float]]] = []
    for row in dataset:
        rows.append(
            {
                "license_openness": row.get("license_openness", row.get("N_license")),
                "contribution_openness": row.get("contribution_openness", row.get("N_contribution")),
                "community_safeguards": row.get("community_safeguards", row.get("N_safeguards")),
                "governance_transparency": row.get("governance_transparency", row.get("N_transparency")),
            }
        )
    weights = _norm_weights_from_rows(
        rows,
        ["license_openness", "contribution_openness", "community_safeguards", "governance_transparency"],
    )
    return {"weights": weights, "n_repos": len(dataset)}


# =========================================================
# Final GOS computation
# =========================================================

def compute_gos(
    repo_path: Path,
    repo_url: Optional[str] = None,
    github_token: Optional[str] = None,
    learned_weights: Optional[Dict[str, float]] = None,
    aggregation: str = "geometric",
) -> Dict[str, Any]:
    """
    Compute Governance & Openness Score for one repository.

    Weighting priority:
    1) learned_weights when supplied
    2) repository-level data-driven fallback based on observed block dispersion
    """
    repo_path = Path(repo_path)

    n_license, license_details = compute_license_openness(repo_path, repo_url=repo_url, github_token=github_token)
    n_contribution, contribution_details = compute_contribution_openness(repo_path)
    n_safeguards, safeguards_details = compute_community_safeguards(repo_path)
    n_transparency, transparency_details = compute_governance_transparency(repo_path)

    blocks = {
        "license_openness": n_license,
        "contribution_openness": n_contribution,
        "community_safeguards": n_safeguards,
        "governance_transparency": n_transparency,
    }

    weights_source = "learned_from_corpus" if learned_weights is not None else "repo_data_driven_fallback"
    weights = learned_weights or _data_driven_weights_from_blocks(blocks)

    pairs = [
        (n_license, float(weights.get("license_openness", weights.get("N_license", 0.0)))),
        (n_contribution, float(weights.get("contribution_openness", weights.get("N_contribution", 0.0)))),
        (n_safeguards, float(weights.get("community_safeguards", weights.get("N_safeguards", 0.0)))),
        (n_transparency, float(weights.get("governance_transparency", weights.get("N_transparency", 0.0)))),
    ]

    agg = (aggregation or "geometric").strip().lower()
    if agg == "sum":
        final_0_1 = weighted_sum(pairs)
        agg_name = "weighted_sum"
    else:
        final_0_1 = weighted_geometric(pairs)
        agg_name = "weighted_geometric"

    score = 100.0 * float(final_0_1) if final_0_1 is not None else None

    return {
        "repository_path": str(repo_path),
        "repository_url": repo_url,
        "score": score,
        "band": band(score),
        "aggregation": agg_name,
        "details": {
            "GOS": score,
            "N_license": n_license,
            "N_contribution": n_contribution,
            "N_safeguards": n_safeguards,
            "N_transparency": n_transparency,
            "weights": weights,
            "weights_source": weights_source,
            "blocks": {
                "license_openness": license_details,
                "contribution_openness": contribution_details,
                "community_safeguards": safeguards_details,
                "governance_transparency": transparency_details,
            },
            "missingness_note": "Unavailable values remain None and are excluded from means/weighted aggregation; observable absence is scored as 0.0.",
            "method_note": "GOS evaluates observable repository governance/openness evidence, not legal advice or actual community behavior outside GitHub.",
        },
    }




# =========================================================
# ITI — Institutional Trust Index
# =========================================================

INSTITUTIONAL_KEYWORDS = [r"\buniversity\b", r"\binstitute\b", r"\blaborator(?:y|ies)\b", r"\bresearch\b", r"\bfoundation\b", r"\bconsortium\b", r"\bagency\b", r"\bministry\b", r"\bcnrs\b", r"\bcea\b", r"\binria\b", r"\bnrel\b", r"\bdoe\b", r"\bnsf\b", r"\beuropean commission\b", r"\bhorizon europe\b", r"\bkth\b", r"\beth\b"]
PROFESSIONAL_NETWORK_PATTERNS = [r"\bpartner(?:s|ship)?\b", r"\bcollaborat(?:e|ion|or|ors)\b", r"\bconsortium\b", r"\bindustry\b", r"\bgovernment\b", r"\buniversity\b", r"\bresearch group\b", r"\bworking group\b", r"\bproject partner\b", r"\bsteering\b"]
REPUTATION_PATTERNS = [r"\baward(?:s)?\b", r"\bgrant(?:s)?\b", r"\bfunded by\b", r"\bcase stud(?:y|ies)\b", r"\bused by\b", r"\badopted by\b", r"\bcited by\b", r"\breference implementation\b", r"\bproduction\b", r"\bdeployment\b", r"\brelease(?:s)?\b"]
DOMAIN_RX = re.compile(r"https?://([^/\s\)\]\}>\"']+)", flags=re.I)
INSTITUTIONAL_DOMAIN_RX = re.compile(r"(\.edu($|\.)|\.ac\.\w+|\.gov($|\.)|\.gouv\.fr|europa\.eu|cnrs\.fr|inria\.fr|kth\.se|ethz\.ch)", flags=re.I)
PROFESSIONAL_DOMAIN_RX = re.compile(r"(\.edu($|\.)|\.ac\.\w+|\.gov($|\.)|\.org($|\.)|\.eu($|\.)|europa\.eu|ieee\.org|acm\.org|zenodo\.org)", flags=re.I)


def _extract_domains_from_texts(texts: Dict[str, str]) -> List[str]:
    domains = []
    for txt in texts.values():
        for m in DOMAIN_RX.findall(txt or ""):
            d = str(m).lower().split(":", 1)[0].strip()
            if d.startswith("www."):
                d = d[4:]
            if d:
                domains.append(d)
    return sorted(set(domains))


def _git_first_commit_age_days(repo_path: Path) -> Optional[float]:
    if not _is_git_repo(repo_path):
        return None
    code, out, _ = run(["git", "log", "--reverse", "--format=%ct", "--max-count=1"], cwd=repo_path)
    if code == 0 and out.strip().isdigit():
        first = dt.datetime.utcfromtimestamp(int(out.strip()))
        return max(0.0, (dt.datetime.utcnow() - first).total_seconds() / 86400.0)
    return None


def _git_recent_commit_email_domains(repo_path: Path, months: int = 24) -> Optional[List[str]]:
    if not _is_git_repo(repo_path):
        return None
    since = (dt.datetime.utcnow() - dt.timedelta(days=int(30.4375 * months))).strftime("%Y-%m-%d")
    code, out, _ = run(["git", "log", f"--since={since}", "--format=%ae"], cwd=repo_path)
    if code != 0:
        return None
    domains = []
    for line in out.splitlines():
        email = line.strip().lower()
        if "@" not in email or "noreply.github.com" in email:
            continue
        domains.append(email.split("@", 1)[1])
    return domains


def compute_institution_quality(repo_path: Path, repo_url: Optional[str] = None, github_token: Optional[str] = None) -> Tuple[Optional[float], Dict[str, Any]]:
    texts = candidate_texts(repo_path)
    if not texts:
        return None, {"available": False, "reason": "no_candidate_documentation_scanned"}
    owner, _ = parse_github_owner_repo(repo_url)
    repo_info = github_repo_info(repo_url, github_token=github_token) if repo_url else None
    owner_meta = None
    if owner:
        owner_meta = gh_api_get(f"https://api.github.com/users/{owner}", token=github_token)
        if not isinstance(owner_meta, dict):
            owner_meta = gh_api_get(f"https://api.github.com/orgs/{owner}", token=github_token)
    owner_type = owner_meta.get("type") if isinstance(owner_meta, dict) else None
    iq_owner_org = 1.0 if owner_type == "Organization" else (0.0 if owner_type == "User" else None)
    meta_blob = ""
    if isinstance(owner_meta, dict):
        meta_blob = " ".join(str(owner_meta.get(k, "")) for k in ["name", "company", "bio", "blog", "location"])
    if isinstance(repo_info, dict):
        meta_blob += " " + " ".join(str(repo_info.get(k, "")) for k in ["description", "homepage"])
    doc_hits = _count_regex_hits_in_texts(texts, INSTITUTIONAL_KEYWORDS)
    meta_hits = sum(len(re.findall(p, meta_blob, flags=re.I)) for p in INSTITUTIONAL_KEYWORDS)
    iq_institutional_mentions = _log_repo_signal(doc_hits + meta_hits, max(1.0, doc_hits + meta_hits))
    domains = _extract_domains_from_texts(texts)
    if isinstance(owner_meta, dict):
        for k in ["blog", "company"]:
            val = owner_meta.get(k)
            if isinstance(val, str):
                domains.extend(_extract_domains_from_texts({k: val}))
    unique_domains = sorted(set(domains))
    inst_domains = sorted(set(d for d in unique_domains if INSTITUTIONAL_DOMAIN_RX.search(d)))
    iq_institutional_domains = ratio01(len(inst_domains), max(1, len(unique_domains))) if unique_domains else None
    email_domains = _git_recent_commit_email_domains(repo_path, months=24)
    iq_commit_affiliation = ratio01(sum(1 for d in email_domains if INSTITUTIONAL_DOMAIN_RX.search(d)), len(email_domains)) if email_domains else None
    block = mean([iq_owner_org, iq_institutional_mentions, iq_institutional_domains, iq_commit_affiliation])
    return block, {"available": True, "block": block, "sub": {"IQ_owner_organization": iq_owner_org, "IQ_institutional_mentions": iq_institutional_mentions, "IQ_institutional_domains": iq_institutional_domains, "IQ_commit_affiliation": iq_commit_affiliation}, "raw": {"owner_type": owner_type, "institutional_doc_hits": doc_hits, "institutional_owner_metadata_hits": meta_hits, "domains_total": len(unique_domains), "institutional_domains": inst_domains[:50], "recent_commit_email_domains_count": len(email_domains) if email_domains is not None else None}, "missingness_note": "Commit affiliation is None when git history or usable email domains are unavailable; no institution ranking is inferred."}


def compute_professional_network(repo_path: Path, repo_url: Optional[str] = None, github_token: Optional[str] = None) -> Tuple[Optional[float], Dict[str, Any]]:
    texts = candidate_texts(repo_path)
    domains = _extract_domains_from_texts(texts)
    unique_domains = sorted(set(domains))
    professional_domains = sorted(set(d for d in unique_domains if PROFESSIONAL_DOMAIN_RX.search(d)))
    pn_domain_diversity = ratio01(len(professional_domains), max(1, len(unique_domains))) if unique_domains else None
    network_hits = _count_regex_hits_in_texts(texts, PROFESSIONAL_NETWORK_PATTERNS)
    pn_network_mentions = _log_repo_signal(network_hits, max(1.0, network_hits))
    org_files = []
    for pth in iter_files(repo_path):
        name = pth.name.lower()
        rel = str(pth.relative_to(repo_path)).replace("\\", "/").lower()
        if name in {"citation.cff", "codemeta.json", "contributors.md", "authors.md", "acknowledgements.md", "acknowledgments.md"} or "partners" in rel or "collaborat" in rel:
            org_files.append(pth)
    pn_structured_affiliations = sat_exp(len(org_files), scale=1.0)
    owner, _ = parse_github_owner_repo(repo_url)
    followers = repos = None
    pn_owner_network = None
    if owner:
        owner_meta = gh_api_get(f"https://api.github.com/users/{owner}", token=github_token)
        if isinstance(owner_meta, dict):
            followers = owner_meta.get("followers") if isinstance(owner_meta.get("followers"), int) else None
            repos = owner_meta.get("public_repos") if isinstance(owner_meta.get("public_repos"), int) else None
            pn_owner_network = mean([sat_exp(followers, scale=50.0) if followers is not None else None, sat_exp(repos, scale=20.0) if repos is not None else None])
    block = mean([pn_domain_diversity, pn_network_mentions, pn_structured_affiliations, pn_owner_network])
    return block, {"available": True, "block": block, "sub": {"PN_domain_diversity": pn_domain_diversity, "PN_network_mentions": pn_network_mentions, "PN_structured_affiliations": pn_structured_affiliations, "PN_owner_network_context": pn_owner_network}, "raw": {"domains_total": len(unique_domains), "professional_domains": professional_domains[:50], "network_keyword_hits": network_hits, "structured_affiliation_files": [str(p.relative_to(repo_path)) for p in org_files[:50]], "owner_followers": followers, "owner_public_repos": repos}, "missingness_note": "Owner network context is optional GitHub metadata; raw counts are reported for calibration."}


def compute_track_record(repo_path: Path, repo_url: Optional[str] = None, github_token: Optional[str] = None) -> Tuple[Optional[float], Dict[str, Any]]:
    first_age_days = _git_first_commit_age_days(repo_path)
    tr_project_age = sat_exp(first_age_days / 365.25, scale=2.0) if first_age_days is not None else None
    tag_count = 0
    if _is_git_repo(repo_path):
        code, out, _ = run(["git", "tag", "--list"], cwd=repo_path)
        if code == 0:
            tag_count = len([x for x in out.splitlines() if x.strip()])
    tr_release_history = sat_exp(tag_count, scale=3.0)
    tr_history_continuity = active_months = total_months = None
    if _is_git_repo(repo_path):
        code, out, _ = run(["git", "log", "--format=%ct"], cwd=repo_path, timeout_s=180)
        if code == 0 and out.strip():
            stamps = [int(x.strip()) for x in out.splitlines() if x.strip().isdigit()]
            if stamps:
                months = {dt.datetime.utcfromtimestamp(t).strftime("%Y-%m") for t in stamps}
                first = dt.datetime.utcfromtimestamp(min(stamps))
                now = dt.datetime.utcnow()
                total_months = max(1, (now.year - first.year) * 12 + (now.month - first.month) + 1)
                active_months = len(months)
                tr_history_continuity = ratio01(active_months, total_months)
    texts = candidate_texts(repo_path)
    reputation_hits = _count_regex_hits_in_texts(texts, REPUTATION_PATTERNS)
    tr_documented_reputation = _log_repo_signal(reputation_hits, max(1.0, reputation_hits))
    repo_info = github_repo_info(repo_url, github_token=github_token) if repo_url else None
    stars = repo_info.get("stargazers_count") if isinstance(repo_info, dict) and isinstance(repo_info.get("stargazers_count"), int) else None
    forks = repo_info.get("forks_count") if isinstance(repo_info, dict) and isinstance(repo_info.get("forks_count"), int) else None
    tr_github_reputation_context = mean([sat_exp(stars, scale=100.0) if stars is not None else None, sat_exp(forks, scale=30.0) if forks is not None else None])
    block = mean([tr_project_age, tr_release_history, tr_history_continuity, tr_documented_reputation, tr_github_reputation_context])
    return block, {"available": True, "block": block, "sub": {"TR_project_age": tr_project_age, "TR_release_history": tr_release_history, "TR_history_continuity": tr_history_continuity, "TR_documented_reputation": tr_documented_reputation, "TR_github_reputation_context": tr_github_reputation_context}, "raw": {"first_commit_age_days": first_age_days, "tag_count": tag_count, "active_months_full_history": active_months, "total_months_full_history": total_months, "reputation_keyword_hits": reputation_hits, "github_stars": stars, "github_forks": forks}, "missingness_note": "GitHub popularity counts are context only and should be recalibrated on a corpus when available."}


def learn_iti_weights(dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    rows = [{"institution_quality": r.get("institution_quality", r.get("N_institution_quality")), "professional_network": r.get("professional_network", r.get("N_professional_network")), "track_record": r.get("track_record", r.get("N_track_record"))} for r in dataset]
    return {"weights": _norm_weights_from_rows(rows, ["institution_quality", "professional_network", "track_record"]), "n_repos": len(dataset)}


def compute_iti(repo_path: Path, repo_url: Optional[str] = None, github_token: Optional[str] = None, learned_weights: Optional[Dict[str, float]] = None, aggregation: str = "geometric") -> Dict[str, Any]:
    n_iq, iq_details = compute_institution_quality(repo_path, repo_url=repo_url, github_token=github_token)
    n_pn, pn_details = compute_professional_network(repo_path, repo_url=repo_url, github_token=github_token)
    n_tr, tr_details = compute_track_record(repo_path, repo_url=repo_url, github_token=github_token)
    blocks = {"institution_quality": n_iq, "professional_network": n_pn, "track_record": n_tr}
    weights = learned_weights or _data_driven_weights_from_blocks(blocks)
    pairs = [(n_iq, float(weights.get("institution_quality", 0.0))), (n_pn, float(weights.get("professional_network", 0.0))), (n_tr, float(weights.get("track_record", 0.0)))]
    final_0_1 = weighted_sum(pairs) if (aggregation or "").strip().lower() == "sum" else weighted_geometric(pairs)
    score = 100.0 * float(final_0_1) if final_0_1 is not None else None
    return {"repository_path": str(repo_path), "repository_url": repo_url, "score": score, "band": band(score), "aggregation": "weighted_sum" if (aggregation or "").strip().lower() == "sum" else "weighted_geometric", "details": {"ITI": score, "N_institution_quality": n_iq, "N_professional_network": n_pn, "N_track_record": n_tr, "weights": weights, "weights_source": "learned_from_corpus" if learned_weights is not None else "repo_data_driven_fallback", "blocks": {"institution_quality": iq_details, "professional_network": pn_details, "track_record": tr_details}, "missingness_note": "Unavailable signals remain None; observable absence is scored 0.0; no institution ranking is hardcoded.", "method_note": "ITI measures observable institutional trust evidence, not actual institutional prestige unless calibrated externally."}}


# =========================================================
# RVS — Reproducibility & Validation Score
# =========================================================

EXAMPLE_RUN_PATTERNS = [r"\bpython\s+[-\w/\\.]+\.py\b", r"\bjupyter\b", r"\bnotebook\b", r"\bmake\s+\w+", r"\bsnakemake\b", r"\bnox\b", r"\btox\b", r"\bpytest\b", r"\bdocker\s+run\b", r"\bexpected output\b", r"\breproduce\b", r"\breproducib"]
BENCHMARK_REPRO_PATTERNS = [r"\bbenchmark\b", r"\basv\b", r"\bpytest-benchmark\b", r"\bperformance claim\b", r"\bruntime\b", r"\btiming\b", r"\bprofile\b", r"\bbaseline\b", r"\bcomparison\b"]
DATASET_TRANSPARENCY_PATTERNS = [r"\bdataset(?:s)?\b", r"\bdata source(?:s)?\b", r"\binput data\b", r"\bopen data\b", r"\bsynthetic data\b", r"\bcase data\b", r"\blicense(?:d)? data\b", r"\bzenodo\b", r"\bfigshare\b", r"\bdoi\b", r"\bdownload\b"]
DATA_FILE_SUFFIXES = {".csv", ".tsv", ".json", ".yaml", ".yml", ".h5", ".hdf5", ".nc", ".npz", ".parquet", ".mat", ".xlsx"}


def _count_files_in_dirs(repo_path: Path, dir_keywords: Iterable[str], suffixes: Optional[set] = None) -> int:
    kws = [k.lower() for k in dir_keywords]
    count = 0
    for pth in iter_files(repo_path):
        if suffixes and pth.suffix.lower() not in suffixes:
            continue
        rel = str(pth.relative_to(repo_path)).replace("\\", "/").lower()
        if any(f"/{k}/" in f"/{rel}" or rel.startswith(f"{k}/") for k in kws):
            count += 1
    return count


def compute_reproducible_examples(repo_path: Path) -> Tuple[Optional[float], Dict[str, Any]]:
    texts = candidate_texts(repo_path)
    example_files = _count_files_in_dirs(repo_path, ["examples", "example", "notebooks", "tutorials", "tutorial", "demo", "demos", "cases", "case_studies"], suffixes=TEXT_SUFFIXES | {".ipynb"})
    re_example_presence = sat_exp(example_files, scale=2.0)
    run_hits = _count_regex_hits_in_texts(texts, EXAMPLE_RUN_PATTERNS)
    re_runnable_instructions = _log_repo_signal(run_hits, max(1.0, run_hits))
    notebook_count = sum(1 for p in iter_files(repo_path) if p.suffix.lower() == ".ipynb")
    re_notebooks = sat_exp(notebook_count, scale=2.0)
    expected_artifacts = _count_pattern_hits_repo(repo_path, [r"expected output", r"reference result", r"baseline", r"golden", r"assert"], suffixes=TEXT_SUFFIXES | SOURCE_SUFFIXES)
    re_expected_outputs = _log_repo_signal(expected_artifacts, max(1.0, expected_artifacts))
    env_files = find_files_by_names(repo_path, ["requirements.txt", "environment.yml", "environment.yaml", "pyproject.toml", "poetry.lock", "Pipfile.lock", "Manifest.toml", "Dockerfile", "docker-compose.yml"])
    re_environment_capture = sat_exp(len(env_files), scale=2.0)
    block = mean([re_example_presence, re_runnable_instructions, re_notebooks, re_expected_outputs, re_environment_capture])
    return block, {"available": True, "block": block, "sub": {"RE_example_presence": re_example_presence, "RE_runnable_instructions": re_runnable_instructions, "RE_notebooks": re_notebooks, "RE_expected_outputs": re_expected_outputs, "RE_environment_capture": re_environment_capture}, "raw": {"example_file_count": example_files, "run_instruction_hits": run_hits, "notebook_count": notebook_count, "expected_artifact_hits": expected_artifacts, "environment_files": [str(p.relative_to(repo_path)) for p in env_files[:50]]}, "missingness_note": "Runnable examples are scored from repository-visible artifacts only."}


def compute_benchmark_reproducibility(repo_path: Path) -> Tuple[Optional[float], Dict[str, Any]]:
    texts = candidate_texts(repo_path)
    benchmark_files = _count_files_in_dirs(repo_path, ["benchmarks", "benchmark", "asv", "performance", "perf"], suffixes=TEXT_SUFFIXES | SOURCE_SUFFIXES | {".json", ".ipynb"})
    br_benchmark_suite = sat_exp(benchmark_files, scale=1.0)
    bench_config_files = find_files_by_names(repo_path, ["asv.conf.json", "asv.conf.toml", "pytest.ini", "tox.ini", "noxfile.py", "Makefile"])
    br_benchmark_config = sat_exp(len(bench_config_files), scale=1.0)
    bench_hits = _count_regex_hits_in_texts(texts, BENCHMARK_REPRO_PATTERNS)
    br_documented_protocol = _log_repo_signal(bench_hits, max(1.0, bench_hits))
    bench_ci = _workflow_signal(repo_path, [r"benchmark", r"pytest-benchmark", r"asv", r"performance", r"perf", r"runtime"])
    result_artifacts = 0
    for pth in iter_files(repo_path):
        rel = str(pth.relative_to(repo_path)).replace("\\", "/").lower()
        if re.search(r"benchmark|asv|performance|perf|baseline", rel) and pth.suffix.lower() in {".json", ".csv", ".txt", ".md", ".rst"}:
            result_artifacts += 1
    br_result_artifacts = sat_exp(result_artifacts, scale=2.0)
    block = mean([br_benchmark_suite, br_benchmark_config, br_documented_protocol, bench_ci, br_result_artifacts])
    return block, {"available": True, "block": block, "sub": {"BR_benchmark_suite": br_benchmark_suite, "BR_benchmark_config": br_benchmark_config, "BR_documented_protocol": br_documented_protocol, "BR_ci_reexecution": bench_ci, "BR_result_artifacts": br_result_artifacts}, "raw": {"benchmark_file_count": benchmark_files, "benchmark_config_files": [str(p.relative_to(repo_path)) for p in bench_config_files[:50]], "benchmark_doc_hits": bench_hits, "benchmark_result_artifacts": result_artifacts}, "missingness_note": "Benchmark reproducibility measures rerunnable benchmark evidence, not absolute performance."}


def compute_dataset_transparency(repo_path: Path) -> Tuple[Optional[float], Dict[str, Any]]:
    texts = candidate_texts(repo_path)
    data_files = _count_files_in_dirs(repo_path, ["data", "datasets", "dataset", "inputs", "cases", "case_data", "test_data", "fixtures"], suffixes=DATA_FILE_SUFFIXES)
    dt_data_artifacts = sat_exp(data_files, scale=3.0)
    data_hits = _count_regex_hits_in_texts(texts, DATASET_TRANSPARENCY_PATTERNS)
    dt_data_documentation = _log_repo_signal(data_hits, max(1.0, data_hits))
    doi_hits = _count_pattern_hits_repo(repo_path, [r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+", r"\bzenodo\b", r"\bfigshare\b"], suffixes=TEXT_SUFFIXES | {".bib", ".cff"})
    dt_public_identifiers = sat_exp(doi_hits, scale=1.0)
    license_data_files = []
    for pth in iter_files(repo_path):
        rel = str(pth.relative_to(repo_path)).replace("\\", "/").lower()
        if ("data" in rel or "dataset" in rel or "case" in rel) and pth.name.lower() in {"license", "license.md", "license.txt", "readme.md", "readme.rst"}:
            license_data_files.append(pth)
    dt_data_license_clarity = sat_exp(len(license_data_files), scale=1.0)
    synthetic_hits = _count_regex_hits_in_texts(texts, [r"synthetic", r"generated data", r"toy case", r"example dataset", r"test case"])
    dt_synthetic_reproducibility = _log_repo_signal(synthetic_hits, max(1.0, synthetic_hits))
    block = mean([dt_data_artifacts, dt_data_documentation, dt_public_identifiers, dt_data_license_clarity, dt_synthetic_reproducibility])
    return block, {"available": True, "block": block, "sub": {"DT_data_artifacts": dt_data_artifacts, "DT_data_documentation": dt_data_documentation, "DT_public_identifiers": dt_public_identifiers, "DT_data_license_clarity": dt_data_license_clarity, "DT_synthetic_reproducibility": dt_synthetic_reproducibility}, "raw": {"data_file_count": data_files, "data_doc_hits": data_hits, "public_identifier_hits": doi_hits, "data_license_files": [str(p.relative_to(repo_path)) for p in license_data_files[:50]], "synthetic_data_hits": synthetic_hits}, "missingness_note": "Dataset transparency includes both public data identifiers and reproducible synthetic/test datasets."}


def learn_rvs_weights(dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    rows = [{"reproducible_examples": r.get("reproducible_examples", r.get("N_reproducible_examples")), "benchmark_reproducibility": r.get("benchmark_reproducibility", r.get("N_benchmark_reproducibility")), "dataset_transparency": r.get("dataset_transparency", r.get("N_dataset_transparency"))} for r in dataset]
    return {"weights": _norm_weights_from_rows(rows, ["reproducible_examples", "benchmark_reproducibility", "dataset_transparency"]), "n_repos": len(dataset)}


def learn_gc_weights(dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = ["GOS", "SCS", "ITI", "RVS"]
    rows = []
    for row in dataset:
        src = row.get("GC", row)
        if isinstance(src, dict) and "scores" in src:
            src = src["scores"]
        out: Dict[str, Optional[float]] = {}
        for k in keys:
            blk = src.get(k) if isinstance(src, dict) else None
            val = blk.get("score") if isinstance(blk, dict) else None
            if isinstance(val, (int, float)):
                out[k] = float(val) / 100.0 if float(val) > 1.0 else float(val)
            else:
                out[k] = None
        rows.append(out)
    return {"weights": _norm_weights_from_rows(rows, keys), "n_repos": len(dataset)}


def compute_gc(
    repo_path: Path,
    repo_url: Optional[str] = None,
    github_token: Optional[str] = None,
    learned_weights: Optional[Dict[str, float]] = None,
    calibration: Optional[Dict[str, Any]] = None,
    aggregation: str = "geometric",
) -> Dict[str, Any]:
    """Governance & Credibility family composite over GOS, SCS, ITI, RVS."""
    gos = compute_gos(repo_path, repo_url, github_token, learned_weights=None, aggregation=aggregation)
    scs = compute_scs(repo_path, repo_url, github_token, learned_weights=None, calibration=calibration, aggregation=aggregation)
    iti = compute_iti(repo_path, repo_url, github_token, learned_weights=None, aggregation=aggregation)
    rvs = compute_rvs(repo_path, repo_url, github_token, learned_weights=None, aggregation=aggregation)

    def _blk(res: Dict[str, Any]) -> Optional[float]:
        s = res.get("score")
        return float(s) / 100.0 if isinstance(s, (int, float)) else None

    blocks = {"GOS": _blk(gos), "SCS": _blk(scs), "ITI": _blk(iti), "RVS": _blk(rvs)}
    family_weights = None
    if isinstance(learned_weights, dict):
        family_weights = learned_weights.get("GC") or learned_weights.get("gc")
        if family_weights is None and all(k in learned_weights for k in blocks):
            family_weights = learned_weights

    weights = family_weights or _data_driven_weights_from_blocks(blocks)
    pairs = [(blocks[k], float(weights.get(k, 0.0))) for k in blocks]
    agg = (aggregation or "geometric").strip().lower()
    if agg == "sum":
        final_0_1 = weighted_sum(pairs)
        agg_name = "weighted_sum"
    else:
        final_0_1 = weighted_geometric(pairs)
        agg_name = "weighted_geometric"
    score = 100.0 * float(final_0_1) if final_0_1 is not None else None
    weights_source = "learned_from_corpus" if family_weights is not None else "repo_data_driven_fallback"

    return {
        "repository_path": str(repo_path),
        "repository_url": repo_url,
        "score": score,
        "band": band(score),
        "aggregation": agg_name,
        "details": {
            "GC": score,
            "GOS": gos.get("score"),
            "SCS": scs.get("score"),
            "ITI": iti.get("score"),
            "RVS": rvs.get("score"),
            "weights": weights,
            "weights_source": weights_source,
            "scores": {"GOS": gos, "SCS": scs, "ITI": iti, "RVS": rvs},
            "missingness_note": "Unavailable values remain None and are excluded with weight renormalization; observable absence is scored as 0.0.",
            "method_note": "GC aggregates governance and credibility headline indicators (GOS, SCS, ITI, RVS).",
        },
    }


def compute_rvs(repo_path: Path, repo_url: Optional[str] = None, github_token: Optional[str] = None, learned_weights: Optional[Dict[str, float]] = None, aggregation: str = "geometric") -> Dict[str, Any]:
    n_re, re_details = compute_reproducible_examples(repo_path)
    n_br, br_details = compute_benchmark_reproducibility(repo_path)
    n_dt, dt_details = compute_dataset_transparency(repo_path)
    blocks = {"reproducible_examples": n_re, "benchmark_reproducibility": n_br, "dataset_transparency": n_dt}
    weights = learned_weights or _data_driven_weights_from_blocks(blocks)
    pairs = [(n_re, float(weights.get("reproducible_examples", 0.0))), (n_br, float(weights.get("benchmark_reproducibility", 0.0))), (n_dt, float(weights.get("dataset_transparency", 0.0)))]
    final_0_1 = weighted_sum(pairs) if (aggregation or "").strip().lower() == "sum" else weighted_geometric(pairs)
    score = 100.0 * float(final_0_1) if final_0_1 is not None else None
    return {"repository_path": str(repo_path), "repository_url": repo_url, "score": score, "band": band(score), "aggregation": "weighted_sum" if (aggregation or "").strip().lower() == "sum" else "weighted_geometric", "details": {"RVS": score, "N_reproducible_examples": n_re, "N_benchmark_reproducibility": n_br, "N_dataset_transparency": n_dt, "weights": weights, "weights_source": "learned_from_corpus" if learned_weights is not None else "repo_data_driven_fallback", "blocks": {"reproducible_examples": re_details, "benchmark_reproducibility": br_details, "dataset_transparency": dt_details}, "missingness_note": "Unavailable signals remain None; observable absence is scored 0.0.", "method_note": "RVS measures whether scientific/engineering claims can be independently rerun using repository evidence."}}


# =========================================================
# CLI
# =========================================================

def _load_json_file(path: Optional[str]) -> Optional[Any]:
    if not path:
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise RuntimeError(f"Cannot read JSON file {path}: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RECOPS Governance & Credibility scores: GOS, SCS, ITI and RVS")
    parser.add_argument("repo_url", nargs="?", help="GitHub repository URL, e.g. https://github.com/owner/repo")
    parser.add_argument("--local", dest="local_repo", help="Analyze an already-local repository path instead of cloning")
    parser.add_argument("--token", dest="github_token", default=os.environ.get("GITHUB_TOKEN"), help="GitHub token; defaults to env GITHUB_TOKEN")
    parser.add_argument("--score", choices=["gos", "scs", "iti", "rvs", "gc", "all"], default="all", help="Which GC score to compute")
    parser.add_argument("--weights-json", dest="weights_json", help="Optional JSON file containing learned weights for the selected score")
    parser.add_argument("--calibration-json", dest="calibration_json", help="Optional SCS calibration JSON, e.g. citation_counts distribution")
    parser.add_argument("--aggregation", choices=["geometric", "sum"], default="geometric", help="Final aggregation; geometric is recommended")
    parser.add_argument("--out", dest="out", help="Optional output JSON path")
    parser.add_argument("--indent", type=int, default=2, help="JSON indentation")
    args = parser.parse_args()

    if not args.local_repo and not args.repo_url:
        parser.error("Provide either a GitHub repo_url or --local PATH")

    repo_path: Path
    if args.local_repo:
        repo_path = Path(args.local_repo).expanduser().resolve()
        if not repo_path.exists() or not repo_path.is_dir():
            raise SystemExit(f"Local repository path does not exist or is not a directory: {repo_path}")
    else:
        repo_path = clone_or_download(args.repo_url)

    learned_weights = None
    if args.weights_json:
        loaded = _load_json_file(args.weights_json)
        if isinstance(loaded, dict):
            learned_weights = loaded.get("weights", loaded)
        else:
            raise SystemExit("--weights-json must contain a JSON object")

    calibration = None
    if args.calibration_json:
        loaded_cal = _load_json_file(args.calibration_json)
        if isinstance(loaded_cal, dict):
            calibration = loaded_cal
        else:
            raise SystemExit("--calibration-json must contain a JSON object")

    sc = (args.score or "all").lower()
    if sc == "gos":
        result = compute_gos(repo_path=repo_path, repo_url=args.repo_url, github_token=args.github_token, learned_weights=learned_weights, aggregation=args.aggregation)
    elif sc == "scs":
        result = compute_scs(repo_path=repo_path, repo_url=args.repo_url, github_token=args.github_token, learned_weights=learned_weights, calibration=calibration, aggregation=args.aggregation)
    elif sc == "iti":
        result = compute_iti(repo_path=repo_path, repo_url=args.repo_url, github_token=args.github_token, learned_weights=learned_weights, aggregation=args.aggregation)
    elif sc == "rvs":
        result = compute_rvs(repo_path=repo_path, repo_url=args.repo_url, github_token=args.github_token, learned_weights=learned_weights, aggregation=args.aggregation)
    elif sc == "gc":
        result = compute_gc(repo_path=repo_path, repo_url=args.repo_url, github_token=args.github_token, learned_weights=learned_weights, calibration=calibration, aggregation=args.aggregation)
    else:
        result = {
            "repository_path": str(repo_path),
            "repository_url": args.repo_url,
            "scores": {
                "GOS": compute_gos(repo_path=repo_path, repo_url=args.repo_url, github_token=args.github_token, learned_weights=None, aggregation=args.aggregation),
                "SCS": compute_scs(repo_path=repo_path, repo_url=args.repo_url, github_token=args.github_token, learned_weights=None, calibration=calibration, aggregation=args.aggregation),
                "ITI": compute_iti(repo_path=repo_path, repo_url=args.repo_url, github_token=args.github_token, learned_weights=None, aggregation=args.aggregation),
                "RVS": compute_rvs(repo_path=repo_path, repo_url=args.repo_url, github_token=args.github_token, learned_weights=None, aggregation=args.aggregation),
                "GC": compute_gc(repo_path=repo_path, repo_url=args.repo_url, github_token=args.github_token, learned_weights=None, calibration=calibration, aggregation=args.aggregation),
            },
            "note": "When --score all is used, --weights-json is ignored to avoid applying one weight file to different formulas."
        }

    txt = json.dumps(result, indent=args.indent, ensure_ascii=False, sort_keys=False)
    if args.out:
        out_path = Path(args.out)
        out_path.write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
