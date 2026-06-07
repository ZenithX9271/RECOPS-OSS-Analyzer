#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RECOPS Repository Scorer v2 — Project Vitality Index (PVI)

PVI measures observable project vitality from activity metadata:
- local Git history (commits, tags)
- GitHub API signals when available (releases, issues, community interest)

Methodological constraints:
- No hard thresholds like "commits > N => score=1"
- Continuous normalization (recency decay, volume saturation, regularity via dispersion, ratios for growth/closure)
- Block-level aggregation is arithmetic mean of sub-signals (stability)
- Final PVI aggregation is, by default, weighted geometric (non-compensatory)
- Missingness is explicit: unavailable signals are kept as None and excluded from means
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
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
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

try:
    import requests
except Exception:
    requests = None

try:
    import yaml
except Exception:
    yaml = None


# =========================================================
# Utils (standalone, adapted from recops_repo_scores_EQ.py)
# =========================================================

def clamp01(x: float) -> float:
    try:
        x = float(x)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, x))


def mean(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def geometric(values: Iterable[Optional[float]], eps: float = 1e-9) -> Optional[float]:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    if any(v <= 0 for v in vals):
        return 0.0
    p = 1.0
    for v in vals:
        p *= max(eps, v)
    return p ** (1.0 / len(vals))


def weighted_geometric(pairs: Iterable[Tuple[Optional[float], float]], eps: float = 1e-6) -> Optional[float]:
    """
    Weighted geometric mean with missingness:
    - pairs: (value, weight)
    - None values are ignored and weights are renormalized.
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
    # Data-adaptive floor: keeps non-compensatory behavior while avoiding
    # collapse to machine-near-zero when one observable block is exactly 0.
    positive = sorted([v for v, _ in vals if v > 0.0])
    if positive:
        floor = min(0.10, max(eps, 0.5 * positive[0]))
    else:
        floor = eps
    acc = 0.0
    for v, w in vals:
        acc += (w / wsum) * math.log(max(floor, float(v)))
    return math.exp(acc)


def weighted_sum(pairs: Iterable[Tuple[Optional[float], float]]) -> Optional[float]:
    vals: List[Tuple[float, float]] = [(float(v), float(w)) for v, w in pairs if v is not None and w is not None and float(w) > 0]
    if not vals:
        return None
    wsum = sum(w for _, w in vals)
    if wsum <= 0:
        return None
    return sum(v * w for v, w in vals) / wsum


def band(score_0_100: float) -> str:
    if score_0_100 >= 80:
        return "Excellent"
    if score_0_100 >= 60:
        return "Good"
    if score_0_100 >= 40:
        return "Moderate"
    if score_0_100 >= 20:
        return "Poor"
    return "Very Poor"


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


# =========================================================
# Repo acquisition (GitHub URL -> local path)
# =========================================================

def github_to_zip(url: str) -> Tuple[str, str, str]:
    m = re.search(r"github\.com/([^/]+)/([^/]+)", url)
    if not m:
        raise ValueError("Invalid GitHub URL")
    owner = m.group(1)
    repo = m.group(2).replace(".git", "")
    # try main first, then master as fallback
    return owner, repo, f"https://github.com/{owner}/{repo}/archive/refs/heads/main.zip"


def clone_or_download(github_url: str) -> Path:
    """
    Mirror of the EQ scorer behavior:
    - prefer full-history partial git clone when git is available
    - fallback to downloading a zip (requires requests)

    Returns a local directory path.
    """
    temp = Path(tempfile.mkdtemp(prefix="recops_pvi_"))
    target = temp / "repo"

    code, _, _ = run(["git", "--version"])
    if code == 0:
        # PVI needs history (commits over months, tags cadence).
        # Use partial clone to reduce bandwidth while keeping commit/tag history.
        c, _, _ = run(["git", "clone", "--filter=blob:none", "--tags", github_url, str(target)], cwd=temp, timeout_s=600)
        if c == 0 and target.exists():
            return target

    if requests is None:
        raise RuntimeError("Need git or requests to fetch repository")

    owner, repo, zip_url = github_to_zip(github_url)
    zpath = temp / "repo.zip"

    r = requests.get(zip_url, timeout=60)
    if r.status_code != 200:
        # fallback to master
        zip_url = zip_url.replace("/main.zip", "/master.zip")
        r = requests.get(zip_url, timeout=60)
    r.raise_for_status()

    with open(zpath, "wb") as f:
        f.write(r.content)

    shutil.unpack_archive(str(zpath), str(temp))
    # GitHub archives unpack to <repo>-<branch> directory
    dirs = [p for p in temp.iterdir() if p.is_dir()]
    for d in dirs:
        if d.name != "repo":
            return d
    raise RuntimeError("Cannot extract repo archive")


# =========================================================
# Normalizations (continuous, no hard thresholds)
# =========================================================

def exp_recency(days: Optional[float], half_life_days: float) -> Optional[float]:
    """
    Recency decay in [0,1], continuous.
    half_life_days: score halves every half-life.
    """
    if days is None:
        return None
    try:
        d = max(0.0, float(days))
    except Exception:
        return None
    if half_life_days <= 0:
        return None
    # exp(-ln(2) * d / half_life)
    return clamp01(math.exp(-math.log(2.0) * d / float(half_life_days)))


def sat_exp(x: Optional[float], scale: float) -> Optional[float]:
    """
    Saturation for non-negative volumes: 1 - exp(-x/scale).
    scale is a soft characteristic scale, not a threshold.
    """
    if x is None:
        return None
    try:
        v = max(0.0, float(x))
    except Exception:
        return None
    s = max(1e-9, float(scale))
    return clamp01(1.0 - math.exp(-v / s))


def stability_from_cv(values: List[float]) -> Optional[float]:
    """
    Stability score in [0,1] from coefficient of variation (CV = std/mean),
    mapped continuously as exp(-CV). Lower dispersion -> higher score.
    """
    if not values:
        return None
    vals = [float(v) for v in values if v is not None]
    if len(vals) < 2:
        return 1.0
    m = _py_mean(vals)
    if m <= 0:
        return None
    # population std (stable for small n)
    var = _py_mean([(v - m) ** 2 for v in vals])
    std = math.sqrt(max(0.0, var))
    cv = std / m
    return clamp01(math.exp(-cv))


def ratio01(num: Optional[float], den: Optional[float]) -> Optional[float]:
    if num is None or den is None:
        return None
    try:
        den = float(den)
        num = float(num)
    except Exception:
        return None
    if den <= 0:
        return None
    return clamp01(num / den)


def exists_any(root: Path, names: List[str]) -> bool:
    """
    Case-insensitive file presence check (recursive).
    """
    wanted = {n.lower() for n in names}
    try:
        for p in root.rglob("*"):
            if p.is_file() and p.name.lower() in wanted:
                return True
    except Exception:
        return False
    return False


def iter_files(root: Path) -> Iterable[Path]:
    """
    Recursive iterator over files. (Standalone; do not depend on EQ module.)
    """
    try:
        for p in root.rglob("*"):
            if p.is_file():
                yield p
    except Exception:
        return


def _workflow_texts(root: Path) -> List[str]:
    wf = root / ".github" / "workflows"
    texts: List[str] = []
    if not wf.exists():
        return texts
    for p in wf.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".yml", ".yaml"}:
            try:
                texts.append(p.read_text(encoding="utf-8", errors="ignore"))
            except Exception:
                pass
    return texts


def workflow_signal(root: Path, patterns: List[str]) -> Optional[float]:
    """
    Returns a saturating score for presence of CI workflow patterns.
    Missing workflows => 0 (evidence absent, not "data unavailable").
    """
    texts = _workflow_texts(root)
    if not texts:
        return 0.0
    rx = [re.compile(p, flags=re.I | re.M) for p in patterns]
    hits = 0
    for txt in texts:
        if any(r.search(txt) for r in rx):
            hits += 1
    return sat_exp(hits, scale=2.0)


# =========================================================
# Git signals (local)
# =========================================================

def _is_git_repo(repo: Path) -> bool:
    code, out, _ = run(["git", "rev-parse", "--is-inside-work-tree"], cwd=repo)
    return code == 0 and out.strip().lower() == "true"


def _git_epoch_list(repo: Path, args: List[str], timeout_s: int = 180) -> List[int]:
    code, out, _ = run(["git", *args], cwd=repo, timeout_s=timeout_s)
    if code != 0:
        return []
    ts: List[int] = []
    for line in out.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.isdigit():
            ts.append(int(s))
    return ts


def _month_key_from_epoch(ts: int) -> str:
    d = dt.datetime.utcfromtimestamp(int(ts))
    return f"{d.year:04d}-{d.month:02d}"


def git_commit_activity(repo: Path, months: int = 12) -> Dict[str, Any]:
    """
    Returns raw commit activity over the last `months` months:
    - total_commits
    - last_commit_days_ago
    - monthly_counts (dict YYYY-MM -> commits)
    - active_months
    """
    if not _is_git_repo(repo):
        return {
            "available": False,
            "reason": "not_a_git_repo",
        }

    now = dt.datetime.utcnow()
    since = now - dt.timedelta(days=int(30.4375 * months))
    since_iso = since.strftime("%Y-%m-%d")

    recent_ts = _git_epoch_list(repo, ["log", f"--since={since_iso}", "--pretty=format:%ct"])
    last_ts = _git_epoch_list(repo, ["log", "-1", "--pretty=format:%ct"])
    last_commit_days_ago = None
    if last_ts:
        last_commit_days_ago = (now - dt.datetime.utcfromtimestamp(last_ts[0])).total_seconds() / 86400.0

    monthly = Counter(_month_key_from_epoch(t) for t in recent_ts)

    # ensure the last `months` calendar months INCLUDING the current month exist as zeros
    def _add_months(d: dt.datetime, delta_months: int) -> dt.datetime:
        y = d.year
        m = d.month + delta_months
        y += (m - 1) // 12
        m = (m - 1) % 12 + 1
        return dt.datetime(y, m, 1)

    end_month = dt.datetime(now.year, now.month, 1)
    start_month = _add_months(end_month, -(max(1, months) - 1))

    months_keys: List[str] = []
    cursor = start_month
    for _ in range(max(1, months)):
        months_keys.append(f"{cursor.year:04d}-{cursor.month:02d}")
        cursor = _add_months(cursor, 1)

    monthly_counts = {k: int(monthly.get(k, 0)) for k in months_keys}
    active_months = sum(1 for v in monthly_counts.values() if v > 0)

    return {
        "available": True,
        "window_months": months,
        "total_commits": int(len(recent_ts)),
        "last_commit_days_ago": last_commit_days_ago,
        "monthly_counts": monthly_counts,
        "active_months": int(active_months),
    }


def git_contributor_distribution(repo: Path, months: int = 12) -> Dict[str, Any]:
    """
    Contributor commit distribution over the last `months` months.
    Used for bus factor and concentration metrics.
    """
    if not _is_git_repo(repo):
        return {"available": False, "reason": "not_a_git_repo"}

    now = dt.datetime.utcnow()
    since = now - dt.timedelta(days=int(30.4375 * months))
    since_iso = since.strftime("%Y-%m-%d")

    code, out, _ = run(["git", "log", f"--since={since_iso}", "--format=%ae"], cwd=repo, timeout_s=180)
    if code != 0:
        return {"available": False, "reason": "git_log_failed"}

    authors: List[str] = []
    for line in out.splitlines():
        s = line.strip().lower()
        if s and "@" in s:
            authors.append(s)

    total_commits = len(authors)
    if total_commits <= 0:
        return {
            "available": True,
            "window_months": months,
            "total_commits": 0,
            "active_contributors": 0,
            "contributor_commit_counts": {},
            "commit_distribution": {},
            "contributor_concentration": 0.0,
            "bus_factor": 0,
        }

    counts = Counter(authors)
    sorted_counts = sorted(counts.values(), reverse=True)

    cumulative = 0
    bus_factor = 0
    target_mass = 0.5 * float(total_commits)
    for c in sorted_counts:
        cumulative += int(c)
        bus_factor += 1
        if cumulative >= target_mass:
            break

    distribution = {k: (float(v) / float(total_commits)) for k, v in counts.items()}
    top_share = float(sorted_counts[0]) / float(total_commits) if sorted_counts else 0.0

    return {
        "available": True,
        "window_months": months,
        "total_commits": int(total_commits),
        "active_contributors": int(len(counts)),
        "contributor_commit_counts": {k: int(v) for k, v in counts.items()},
        "commit_distribution": distribution,
        "contributor_concentration": top_share,
        "bus_factor": int(bus_factor),
    }


def compute_bus_factor(repo_path: Path, months: int = 24) -> Dict[str, Any]:
    """
    BUS_FACTOR indicator (0-100):
    - raw bus_factor is computed as the minimum number of authors accounting for 50% of commits in window.
    - normalized score uses the share of active contributors needed to reach that 50% mass:
        N_bus_factor = bus_factor / active_contributors  (clamped to [0,1])
      This is ratio-based (no fixed weights/scales) and robust across repo sizes.
    """
    repo_path = Path(repo_path)
    raw = git_contributor_distribution(repo_path, months=months)
    if not raw.get("available"):
        return {"score": None, "details": {"available": False, "reason": raw.get("reason")}}

    bus_factor = raw.get("bus_factor")
    active = raw.get("active_contributors")
    total_commits = raw.get("total_commits")

    if total_commits == 0:
        # Observable: no commits in window => bus factor is 0 by definition.
        return {
            "score": 0.0,
            "details": {
                "available": True,
                "BUS_FACTOR": 0.0,
                "N_bus_factor": 0.0,
                "raw": {"bus_factor": 0, "active_contributors": 0, "total_commits": 0, "window_months": months},
                "missingness_note": "No commits observed in the window; bus factor is defined as 0 (observable absence).",
            },
        }

    n_bus = ratio01(bus_factor, active)
    score = 100.0 * float(n_bus) if n_bus is not None else None
    return {
        "score": score,
        "details": {
            "available": True,
            "BUS_FACTOR": score,
            "N_bus_factor": n_bus,
            "raw": {
                "bus_factor": int(bus_factor) if bus_factor is not None else None,
                "active_contributors": int(active) if active is not None else None,
                "total_commits": int(total_commits) if total_commits is not None else None,
                "contributor_concentration": raw.get("contributor_concentration"),
                "window_months": raw.get("window_months"),
            },
            "missingness_note": "Score is None only when contributor counts are unavailable; otherwise ratio-based normalization is used.",
        },
    }


def git_tag_dates(repo: Path) -> List[Tuple[str, Optional[int]]]:
    """
    Try to obtain tag names and creator dates (epoch seconds) from git.
    Tags without a creatordate are returned with None.
    """
    if not _is_git_repo(repo):
        return []
    code, out, _ = run(
        [
            "git",
            "for-each-ref",
            "--sort=creatordate",
            "--format=%(refname:strip=2)\t%(creatordate:unix)",
            "refs/tags",
        ],
        cwd=repo,
        timeout_s=120,
    )
    if code != 0:
        return []
    tags: List[Tuple[str, Optional[int]]] = []
    for line in out.splitlines():
        parts = line.strip().split("\t")
        if not parts or not parts[0]:
            continue
        name = parts[0]
        ts = None
        if len(parts) > 1 and parts[1].strip().isdigit():
            ts = int(parts[1].strip())
        tags.append((name, ts))
    return tags


SEMVER_RX = re.compile(
    r"^(?:v)?(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:[-+][0-9A-Za-z\.-]+)?$"
)


def is_semver_tag(tag: str) -> bool:
    return bool(SEMVER_RX.match((tag or "").strip()))


# =========================================================
# GitHub API helpers (optional)
# =========================================================

def parse_github_owner_repo(url: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if not url:
        return None, None
    m = re.search(r"github\.com/([^/]+)/([^/]+)", url)
    if not m:
        return None, None
    owner = m.group(1)
    repo = m.group(2).replace(".git", "")
    return owner, repo


def gh_api_get(url: str, token: Optional[str] = None, accept: Optional[str] = None, timeout_s: int = 30) -> Optional[Any]:
    if requests is None:
        return None
    headers = {"User-Agent": "recops-pvi/1.0"}
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
    Simple page-based pagination for GitHub REST.
    Returns None when the first page cannot be fetched (network/auth/list shape); otherwise a list (possibly empty when the API successfully returned no rows).
    Later-page failures return accumulated rows instead of None.
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


def _parse_iso8601(s: Optional[str]) -> Optional[dt.datetime]:
    if not s:
        return None
    try:
        # GitHub: "2024-01-01T12:34:56Z"
        if s.endswith("Z"):
            return dt.datetime.fromisoformat(s[:-1]).replace(tzinfo=dt.timezone.utc).astimezone(dt.timezone.utc).replace(tzinfo=None)
        return dt.datetime.fromisoformat(s).astimezone(dt.timezone.utc).replace(tzinfo=None)
    except Exception:
        return None


# =========================================================
# PVI blocks
# =========================================================

def compute_commit_activity(repo: Path, months: int = 12) -> Tuple[Optional[float], Dict[str, Any]]:
    raw = git_commit_activity(repo, months=months)
    if not raw.get("available"):
        return None, {"available": False, "reason": raw.get("reason")}

    total = raw["total_commits"]
    active_months = raw["active_months"]
    last_days = raw["last_commit_days_ago"]
    monthly_counts = raw["monthly_counts"]

    n_volume = sat_exp(total, scale=max(10.0, 4.0 * months))
    n_recency = exp_recency(last_days, half_life_days=30.0)
    n_persistence = ratio01(active_months, months)
    n_regularity = stability_from_cv(list(monthly_counts.values()))
    contrib = git_contributor_distribution(repo, months=months)
    bus_factor = int(contrib.get("bus_factor", 0)) if contrib.get("available") else None
    active_contributors = int(contrib.get("active_contributors", 0)) if contrib.get("available") else None
    # Concentration-resilience proxy: fraction of contributors required to reach 50% commit mass.
    n_bfri = ratio01(bus_factor, active_contributors) if (bus_factor is not None and active_contributors is not None) else None

    block = mean([n_volume, n_recency, n_persistence, n_regularity, n_bfri])
    return block, {
        "available": True,
        "block": block,
        "sub": {
            "N_commit_volume": n_volume,
            "N_commit_recency": n_recency,
            "N_commit_persistence": n_persistence,
            "N_commit_regularity": n_regularity,
            "PH3_BFRI": n_bfri,
        },
        "raw": {
            "commits_last_window": total,
            "active_months": active_months,
            "last_commit_days_ago": last_days,
            "monthly_counts": monthly_counts,
            "window_months": months,
            "active_contributors": active_contributors,
            "contributor_concentration": contrib.get("contributor_concentration") if contrib.get("available") else None,
            "Bus_Factor": bus_factor,
            "PH3_BFRI_ratio_0_1": n_bfri,
        },
    }


def _release_dates_from_github(owner: str, repo: str, token: Optional[str]) -> Optional[List[dt.datetime]]:
    releases = gh_paginate(f"https://api.github.com/repos/{owner}/{repo}/releases", token=token, per_page=100, max_pages=5)
    if releases is None:
        return None
    if not releases:
        return None
    dates: List[dt.datetime] = []
    for r in releases:
        if not isinstance(r, dict):
            continue
        d = _parse_iso8601(r.get("published_at")) or _parse_iso8601(r.get("created_at"))
        if d is not None:
            dates.append(d)
    dates = sorted(set(dates))
    return dates


def _release_dates_from_git_tags(repo_path: Path) -> List[dt.datetime]:
    tags = git_tag_dates(repo_path)
    dates: List[dt.datetime] = []
    for _, ts in tags:
        if ts is None:
            continue
        dates.append(dt.datetime.utcfromtimestamp(int(ts)))
    dates = sorted(set(dates))
    return dates


def compute_release_regularity(
    repo_path: Path,
    github_url: Optional[str] = None,
    github_token: Optional[str] = None,
    months: int = 24,
) -> Tuple[Optional[float], Dict[str, Any]]:
    now = dt.datetime.utcnow()
    cutoff = now - dt.timedelta(days=int(30.4375 * months))

    owner, repo = parse_github_owner_repo(github_url)
    dates: Optional[List[dt.datetime]] = None
    source = None
    if owner and repo:
        dates = _release_dates_from_github(owner, repo, token=github_token)
        if dates is not None:
            source = "github_releases"

    if dates is None:
        dates = _release_dates_from_git_tags(repo_path)
        source = "git_tags" if dates else None

    if not dates:
        return None, {
            "available": False,
            "reason": "no_release_dates_observed_or_unavailable",
            "source": source,
        }

    recent = [d for d in dates if d >= cutoff]
    if not recent:
        # We observed releases, but none in the target window.
        last_days = (now - max(dates)).total_seconds() / 86400.0
        n_recency = exp_recency(last_days, half_life_days=120.0)
        # keep other sub-signals as 0-like but explicit
        block = mean([n_recency, 0.0, None, None])
        return block, {
            "available": True,
            "source": source,
            "block": block,
            "sub": {
                "N_release_volume": 0.0,
                "N_release_recency": n_recency,
                "N_release_regularity": None,
                "N_release_semver": None,
            },
            "raw": {
                "releases_total_observed": len(dates),
                "releases_in_window": 0,
                "last_release_days_ago": last_days,
                "window_months": months,
            },
            "missingness_note": "Releases/tags exist but none in the scoring window; regularity/semver window signals are left as None.",
        }

    last_days = (now - max(recent)).total_seconds() / 86400.0
    n_volume = sat_exp(len(recent), scale=max(2.0, months / 3.0))  # soft: ~8 releases / 24m saturates
    n_recency = exp_recency(last_days, half_life_days=120.0)

    # regularity from inter-release intervals
    recent_sorted = sorted(recent)
    intervals = []
    for a, b in zip(recent_sorted[:-1], recent_sorted[1:]):
        intervals.append((b - a).total_seconds() / 86400.0)
    n_regularity = stability_from_cv(intervals) if intervals else 1.0

    # semver discipline: among observed tags in window (if available)
    semver_score = None
    tags = git_tag_dates(repo_path)
    if tags:
        tag_names = [t for t, ts in tags if ts is not None and dt.datetime.utcfromtimestamp(int(ts)) >= cutoff]
        if tag_names:
            semver_score = clamp01(sum(1 for t in tag_names if is_semver_tag(t)) / len(tag_names))

    block = mean([n_volume, n_recency, n_regularity, semver_score])
    return block, {
        "available": True,
        "source": source,
        "block": block,
        "sub": {
            "N_release_volume": n_volume,
            "N_release_recency": n_recency,
            "N_release_regularity": n_regularity,
            "N_release_semver": semver_score,
        },
        "raw": {
            "releases_in_window": len(recent),
            "last_release_days_ago": last_days,
            "window_months": months,
            "intervals_days": intervals,
        },
        "missingness_note": "If GitHub releases are unavailable, git tags are used as a proxy for releases.",
    }


def _issues_from_github(owner: str, repo: str, token: Optional[str], since: dt.datetime) -> Optional[List[Dict[str, Any]]]:
    # GitHub issues endpoint returns PRs too; filter them out.
    since_iso = since.replace(microsecond=0).isoformat() + "Z"
    items = gh_paginate(
        f"https://api.github.com/repos/{owner}/{repo}/issues?state=all&since={since_iso}",
        token=token,
        per_page=100,
        max_pages=10,
    )
    if items is None:
        return None
    out: List[Dict[str, Any]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        if "pull_request" in it:
            continue
        out.append(it)
    return out


def _issue_first_comment_delay_days(owner: str, repo: str, issue_number: int, token: Optional[str]) -> Optional[float]:
    """
    Proxy for "time to first response": first comment time - issue creation time.
    Uses only the first page of comments to keep it bounded.
    """
    issue = gh_api_get(f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}", token=token)
    if not isinstance(issue, dict):
        return None
    created = _parse_iso8601(issue.get("created_at"))
    if created is None:
        return None
    comments = gh_api_get(f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}/comments?per_page=1&page=1", token=token)
    if not isinstance(comments, list) or not comments:
        return None
    first = comments[0] if isinstance(comments[0], dict) else None
    if not first:
        return None
    first_dt = _parse_iso8601(first.get("created_at"))
    if first_dt is None:
        return None
    return max(0.0, (first_dt - created).total_seconds() / 86400.0)


def _issue_resolution_time_days(owner: str, repo: str, issue_number: int, token: Optional[str]) -> Optional[float]:
    """
    Resolution time proxy: closed_at - created_at for a closed issue.
    Fetches the issue object (bounded, per-issue).
    """
    issue = gh_api_get(f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}", token=token)
    if not isinstance(issue, dict):
        return None
    created = _parse_iso8601(issue.get("created_at"))
    closed = _parse_iso8601(issue.get("closed_at"))
    if created is None or closed is None:
        return None
    return max(0.0, (closed - created).total_seconds() / 86400.0)


def _issues_open_backlog(owner: str, repo: str, token: Optional[str], max_pages: int = 10) -> Optional[List[Dict[str, Any]]]:
    """
    Fetch open issues (excluding PRs). This is used for backlog control.
    Bounded by max_pages.
    """
    items = gh_paginate(
        f"https://api.github.com/repos/{owner}/{repo}/issues?state=open&sort=created&direction=asc",
        token=token,
        per_page=100,
        max_pages=max_pages,
    )
    if items is None:
        return None
    out: List[Dict[str, Any]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        if "pull_request" in it:
            continue
        out.append(it)
    return out


def compute_issue_activity(
    github_url: Optional[str],
    github_token: Optional[str],
    months: int = 12,
    max_issues_for_response_time: int = 50,
) -> Tuple[Optional[float], Dict[str, Any]]:
    owner, repo = parse_github_owner_repo(github_url)
    if not owner or not repo:
        return None, {
            "available": False,
            "reason": "github_url_missing_or_unparseable",
        }
    if requests is None:
        return None, {
            "available": False,
            "reason": "requests_not_installed",
        }
    has_issues = _github_has_issues(owner, repo, token=github_token)
    if has_issues is False:
        return None, {
            "available": False,
            "reason": "github_issues_disabled",
        }

    now = dt.datetime.utcnow()
    since = now - dt.timedelta(days=int(30.4375 * months))
    issues = _issues_from_github(owner, repo, token=github_token, since=since)
    if issues is None:
        return None, {
            "available": False,
            "reason": "github_issues_api_unavailable",
            "hint": "If this is a large repo, you may be rate-limited without a GitHub token (set env GITHUB_TOKEN).",
        }

    opened = 0
    closed = 0
    comment_counts: List[int] = []
    activity_months = set()

    for it in issues:
        created = _parse_iso8601(it.get("created_at"))
        updated = _parse_iso8601(it.get("updated_at")) or created
        closed_at = _parse_iso8601(it.get("closed_at"))
        if created and created >= since:
            opened += 1
        if closed_at and closed_at >= since:
            closed += 1
        if updated and updated >= since:
            activity_months.add(f"{updated.year:04d}-{updated.month:02d}")
        c = it.get("comments")
        if isinstance(c, int):
            comment_counts.append(c)

    total_flow = opened + closed
    n_flow = sat_exp(total_flow, scale=max(25.0, 2.0 * months * 10.0))
    n_closure = ratio01(closed, total_flow)  # closure share of total observed flow

    avg_comments = (_py_mean(comment_counts) if comment_counts else None)
    n_discussion = sat_exp(avg_comments, scale=5.0) if avg_comments is not None else None

    n_persistence = ratio01(len(activity_months), months)

    # Response time proxy (bounded sampling)
    delays: List[float] = []
    sampled = 0
    for it in issues:
        if sampled >= max_issues_for_response_time:
            break
        num = it.get("number")
        if not isinstance(num, int):
            continue
        d = _issue_first_comment_delay_days(owner, repo, num, token=github_token)
        if d is not None:
            delays.append(d)
        sampled += 1

    median_delay = None
    if delays:
        delays_sorted = sorted(delays)
        median_delay = delays_sorted[len(delays_sorted) // 2]
    n_responsiveness = exp_recency(median_delay, half_life_days=7.0) if median_delay is not None else None

    block = mean([n_flow, n_closure, n_responsiveness, n_discussion, n_persistence])
    return block, {
        "available": True,
        "block": block,
        "sub": {
            "N_issue_flow": n_flow,
            "N_issue_closure": n_closure,
            "N_issue_responsiveness": n_responsiveness,
            "N_issue_discussion": n_discussion,
            "N_issue_persistence": n_persistence,
        },
        "raw": {
            "issues_opened_window": opened,
            "issues_closed_window": closed,
            "issues_total_flow": total_flow,
            "avg_comments_per_issue": avg_comments,
            "active_months": len(activity_months),
            "median_first_comment_delay_days_proxy": median_delay,
            "sampled_issues_for_response_time": sampled,
            "window_months": months,
        },
        "missingness_note": "Issue signals require GitHub API access; missing values are kept as None (not forced to zero). First response time is a bounded proxy based on first comment.",
    }


def _git_contributors(repo: Path, since_iso: Optional[str] = None) -> List[str]:
    args = ["log", "--format=%ae"]
    if since_iso:
        args.insert(1, f"--since={since_iso}")
    code, out, _ = run(["git", *args], cwd=repo, timeout_s=180)
    if code != 0:
        return []
    seen = set()
    for line in out.splitlines():
        s = line.strip().lower()
        if s and "@" in s:
            seen.add(s)
    return sorted(seen)


def _github_repo_info(owner: str, repo: str, token: Optional[str]) -> Optional[Dict[str, Any]]:
    data = gh_api_get(f"https://api.github.com/repos/{owner}/{repo}", token=token)
    return data if isinstance(data, dict) else None


def _github_has_issues(owner: str, repo: str, token: Optional[str]) -> Optional[bool]:
    info = _github_repo_info(owner, repo, token=token)
    if not isinstance(info, dict):
        return None
    v = info.get("has_issues")
    if isinstance(v, bool):
        return v
    return None


def _github_stars_gained(owner: str, repo: str, token: Optional[str], since: dt.datetime, max_pages: int = 10) -> Optional[int]:
    """
    Counts stargazers since cutoff using the "star+json" media type which includes starred_at.
    Bounded by max_pages to avoid heavy crawling.
    """
    accept = "application/vnd.github.star+json"
    cutoff = since
    gained = 0
    for page in range(1, max_pages + 1):
        data = gh_api_get(
            f"https://api.github.com/repos/{owner}/{repo}/stargazers?per_page=100&page={page}",
            token=token,
            accept=accept,
        )
        if data is None:
            return None if page == 1 else gained
        if not isinstance(data, list) or not data:
            break
        stop = False
        for it in data:
            if not isinstance(it, dict):
                continue
            starred_at = _parse_iso8601(it.get("starred_at"))
            if starred_at is None:
                continue
            if starred_at >= cutoff:
                gained += 1
            else:
                stop = True
                break
        if stop or len(data) < 100:
            break
    return gained


def _github_forks_gained(owner: str, repo: str, token: Optional[str], since: dt.datetime, max_pages: int = 10) -> Optional[int]:
    """
    Counts forks since cutoff via /forks sorted by newest.
    Bounded by max_pages.
    """
    gained = 0
    cutoff = since
    for page in range(1, max_pages + 1):
        data = gh_api_get(
            f"https://api.github.com/repos/{owner}/{repo}/forks?sort=newest&per_page=100&page={page}",
            token=token,
        )
        if data is None:
            return None if page == 1 else gained
        if not isinstance(data, list) or not data:
            break
        stop = False
        for it in data:
            if not isinstance(it, dict):
                continue
            created = _parse_iso8601(it.get("created_at"))
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


def compute_community_growth(
    repo_path: Path,
    github_url: Optional[str] = None,
    github_token: Optional[str] = None,
    months: int = 12,
) -> Tuple[Optional[float], Dict[str, Any]]:
    now = dt.datetime.utcnow()
    since = now - dt.timedelta(days=int(30.4375 * months))
    since_iso = since.strftime("%Y-%m-%d")

    if not _is_git_repo(repo_path):
        return None, {"available": False, "reason": "not_a_git_repo"}

    recent_contribs = _git_contributors(repo_path, since_iso=since_iso)
    total_contribs = _git_contributors(repo_path, since_iso=None)
    n_contrib_growth = ratio01(len(recent_contribs), len(total_contribs)) if total_contribs else None

    owner, repo = parse_github_owner_repo(github_url)
    repo_info = _github_repo_info(owner, repo, token=github_token) if owner and repo else None

    stars_total = repo_info.get("stargazers_count") if isinstance(repo_info, dict) else None
    forks_total = repo_info.get("forks_count") if isinstance(repo_info, dict) else None

    stars_gained = None
    forks_gained = None
    if owner and repo and requests is not None:
        stars_gained = _github_stars_gained(owner, repo, token=github_token, since=since, max_pages=10)
        forks_gained = _github_forks_gained(owner, repo, token=github_token, since=since, max_pages=10)

    n_star_growth = ratio01(stars_gained, stars_total) if (stars_gained is not None and stars_total is not None) else None
    n_fork_growth = ratio01(forks_gained, forks_total) if (forks_gained is not None and forks_total is not None) else None

    # soft volume of interest growth (recent deltas)
    interest_delta = None
    if stars_gained is not None and forks_gained is not None:
        interest_delta = float(stars_gained + forks_gained)
    elif stars_gained is not None:
        interest_delta = float(stars_gained)
    elif forks_gained is not None:
        interest_delta = float(forks_gained)
    n_interest = sat_exp(interest_delta, scale=max(10.0, 2.0 * months))

    block = mean([n_contrib_growth, n_star_growth, n_fork_growth, n_interest])
    return block, {
        "available": True,
        "block": block,
        "sub": {
            "N_contributor_growth": n_contrib_growth,
            "N_star_growth_rel": n_star_growth,
            "N_fork_growth_rel": n_fork_growth,
            "N_interest_delta": n_interest,
        },
        "raw": {
            "contributors_recent": len(recent_contribs),
            "contributors_total": len(total_contribs),
            "stars_total": stars_total,
            "forks_total": forks_total,
            "stars_gained_window": stars_gained,
            "forks_gained_window": forks_gained,
            "window_months": months,
        },
        "missingness_note": "Contributor growth is derived from local git authors (recent/total). Star/fork growth requires GitHub API and may be None if unavailable; missing values are not forced to zero.",
    }


# =========================================================
# MQS — Maintenance Quality Score
# =========================================================

def _exp_decay_days(days: Optional[float], scale_days: float) -> Optional[float]:
    """
    MQS-specific decay requested by design: exp(-days / scale).
    """
    if days is None:
        return None
    try:
        d = max(0.0, float(days))
    except Exception:
        return None
    if scale_days <= 0:
        return None
    return clamp01(math.exp(-d / float(scale_days)))


def compute_issue_management(
    repo_url: Optional[str],
    github_token: Optional[str] = None,
    months: int = 12,
    max_issues_sample: int = 100,
) -> Tuple[Optional[float], Dict[str, Any]]:
    """
    Data-driven, extractible issue management:
    IM1 first response speed: exp(-median_first_response_days / 7)
    IM2 resolution speed: exp(-median_resolution_days / 30)
    IM3 closure efficiency: min(closed/opened, 1)
    IM4 backlog pressure: exp(-median_open_issue_age_days / 90)
    """
    owner, repo = parse_github_owner_repo(repo_url)
    if not owner or not repo:
        return None, {"available": False, "reason": "github_url_missing_or_unparseable"}
    if requests is None:
        return None, {"available": False, "reason": "requests_not_installed"}
    has_issues = _github_has_issues(owner, repo, token=github_token)
    if has_issues is False:
        return None, {
            "available": False,
            "reason": "github_issues_disabled",
        }

    now = dt.datetime.utcnow()
    since = now - dt.timedelta(days=int(30.4375 * months))

    issues = _issues_from_github(owner, repo, token=github_token, since=since)
    if issues is None:
        return None, {
            "available": False,
            "reason": "github_issues_api_unavailable",
            "hint": "You may be rate-limited without a GitHub token (set env GITHUB_TOKEN).",
        }

    opened_count = 0
    closed_count = 0
    issue_numbers: List[int] = []
    for it in issues:
        created = _parse_iso8601(it.get("created_at"))
        closed_at = _parse_iso8601(it.get("closed_at"))
        if created and created >= since:
            opened_count += 1
            n = it.get("number")
            if isinstance(n, int):
                issue_numbers.append(n)
        if closed_at and closed_at >= since:
            closed_count += 1

    delays_first: List[float] = []
    delays_resolution: List[float] = []
    sampled = 0
    for num in issue_numbers:
        if sampled >= max_issues_sample:
            break
        d_first = _issue_first_comment_delay_days(owner, repo, num, token=github_token)
        if d_first is not None:
            delays_first.append(d_first)
        d_res = _issue_resolution_time_days(owner, repo, num, token=github_token)
        if d_res is not None:
            delays_resolution.append(d_res)
        sampled += 1

    median_first = sorted(delays_first)[len(delays_first) // 2] if delays_first else None
    median_res = sorted(delays_resolution)[len(delays_resolution) // 2] if delays_resolution else None

    im1 = _exp_decay_days(median_first, 7.0)
    im2 = _exp_decay_days(median_res, 30.0)
    im3 = clamp01(min(float(closed_count) / max(float(opened_count), 1.0), 1.0))

    open_issues = _issues_open_backlog(owner, repo, token=github_token, max_pages=10)
    open_ages: List[float] = []
    median_open_age: Optional[float] = None
    if open_issues is None:
        im4 = None
    elif not open_issues:
        # Successful fetch with zero open issues: backlog pressure is maximally favorable.
        im4 = 1.0
    else:
        for it in open_issues:
            created = _parse_iso8601(it.get("created_at"))
            if created is None:
                continue
            open_ages.append(max(0.0, (now - created).total_seconds() / 86400.0))
        median_open_age = sorted(open_ages)[len(open_ages) // 2] if open_ages else None
        im4 = _exp_decay_days(median_open_age, 90.0)

    block = mean([im1, im2, im3, im4])
    return block, {
        "available": True,
        "block": block,
        "sub": {
            "IM1_first_response_speed": im1,
            "IM2_resolution_speed": im2,
            "IM3_closure_efficiency": im3,
            "IM4_backlog_pressure": im4,
        },
        "raw": {
            "issues_opened_12m": opened_count,
            "issues_closed_12m": closed_count,
            "median_first_response_days_proxy": median_first,
            "median_resolution_days": median_res,
            "median_open_issue_age_days": median_open_age,
            "sampled_issues_for_timings": sampled,
            "window_months": months,
        },
    }


def _dependency_files(repo_path: Path) -> List[Path]:
    names = {
        "requirements.txt",
        "requirements-dev.txt",
        "pyproject.toml",
        "setup.py",
        "setup.cfg",
        "poetry.lock",
        "pipfile",
        "pipfile.lock",
        "environment.yml",
        "environment.yaml",
        "package.json",
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "cargo.toml",
        "cargo.lock",
        "pom.xml",
    }
    files: List[Path] = []
    for p in iter_files(repo_path):
        if p.name.lower() in names:
            files.append(p)
    return files


def _git_last_change_days(repo_path: Path, file_path: Path, now: dt.datetime) -> Optional[float]:
    """
    Uses git log for recency when available; falls back to filesystem mtime.
    """
    try:
        rel = str(file_path)
        code, out, _ = run(["git", "log", "-1", "--format=%ct", "--", rel], cwd=repo_path)
        if code == 0 and out.strip().isdigit():
            ts = int(out.strip())
            return max(0.0, (now - dt.datetime.utcfromtimestamp(ts)).total_seconds() / 86400.0)
    except Exception:
        pass
    try:
        mtime = file_path.stat().st_mtime
        return max(0.0, (now - dt.datetime.utcfromtimestamp(int(mtime))).total_seconds() / 86400.0)
    except Exception:
        return None


def dependency_update_freshness(repo_path: Path) -> Tuple[Optional[float], Dict[str, Any]]:
    """
    MQS UC4 requested form:
    - find dependency files
    - use the most recent dependency-file commit change
    - score = exp(-days_since_dep_update / 90)
    """
    if not _is_git_repo(repo_path):
        return None, {"available": False, "reason": "not_a_git_repo"}
    now = dt.datetime.utcnow()
    dep_files = _dependency_files(repo_path)
    if not dep_files:
        return None, {"available": False, "reason": "no_dependency_manifests_found"}

    days: List[float] = []
    for f in dep_files:
        d = _git_last_change_days(repo_path, f, now=now)
        if d is not None:
            days.append(float(d))
    if not days:
        return None, {"available": False, "reason": "dependency_recency_unavailable"}

    most_recent_days = min(days)
    score = _exp_decay_days(most_recent_days, 90.0)
    return score, {
        "available": True,
        "score": score,
        "raw": {
            "dependency_files_count": len(dep_files),
            "most_recent_dep_change_days": most_recent_days,
            "dep_files": [str(p.relative_to(repo_path)) for p in dep_files[:50]],
        },
        "missingness_note": "If no dependency manifests are present, this sub-signal is None (not forced to 0).",
    }


def compute_update_consistency(repo_path: Path) -> Tuple[Optional[float], Dict[str, Any]]:
    """
    Update consistency:
    - commit frequency (saturating)
    - active-month persistence
    - commit regularity (dispersion-based stability)
    - dependency update freshness (median recency of dep files)
    """
    commit_raw = git_commit_activity(repo_path, months=12)
    if not commit_raw.get("available"):
        return None, {"available": False, "reason": commit_raw.get("reason")}

    total = commit_raw["total_commits"]
    active_months = commit_raw["active_months"]
    monthly_counts = commit_raw["monthly_counts"]

    last_days = commit_raw["last_commit_days_ago"]
    uc1 = _exp_decay_days(last_days, 30.0)
    uc2 = ratio01(active_months, 12)
    uc3 = stability_from_cv(list(monthly_counts.values()))
    uc4, dep_details = dependency_update_freshness(repo_path)

    block = mean([uc1, uc2, uc3, uc4])
    return block, {
        "available": True,
        "block": block,
        "sub": {
            "UC1_commit_recency": uc1,
            "UC2_active_months_ratio": uc2,
            "UC3_commit_regularity": uc3,
            "UC4_dependency_maintenance": uc4,
        },
        "raw": {
            "commits_last_12m": total,
            "active_months_12m": active_months,
            "monthly_counts": monthly_counts,
            "days_since_last_commit": last_days,
        },
        "dependency_details": dep_details,
    }


def _security_policy_presence(repo_path: Path) -> float:
    return 1.0 if (repo_path / "SECURITY.md").exists() or (repo_path / ".github" / "SECURITY.md").exists() else 0.0


def _security_tools_from_workflows(repo_path: Path) -> Tuple[int, List[str]]:
    """
    Parse GitHub workflow YAMLs and detect security tooling keywords.
    Returns unique-tool count and list of matched tools.
    """
    workflow_dir = repo_path / ".github" / "workflows"
    if not workflow_dir.exists():
        return 0, []

    patterns = {
        "dependabot": re.compile(r"\bdependabot\b", re.I),
        "codeql": re.compile(r"\bcodeql\b", re.I),
        "pip-audit": re.compile(r"\bpip-audit\b", re.I),
        "npm audit": re.compile(r"\bnpm audit\b", re.I),
        "osv-scanner": re.compile(r"\bosv-scanner\b", re.I),
        "snyk": re.compile(r"\bsnyk\b", re.I),
        "safety": re.compile(r"\bsafety\b", re.I),
    }
    found = set()
    for p in workflow_dir.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in {".yml", ".yaml"}:
            continue
        txt = ""
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            pass
        if yaml is not None:
            try:
                parsed = yaml.safe_load(txt)
                txt = json.dumps(parsed, ensure_ascii=True)
            except Exception:
                pass
        for label, rx in patterns.items():
            if rx.search(txt or ""):
                found.add(label)
    return len(found), sorted(found)


def _security_automation_presence(repo_path: Path) -> float:
    if (repo_path / ".github" / "dependabot.yml").exists():
        return 1.0
    if (repo_path / "renovate.json").exists() or (repo_path / "renovate.json5").exists():
        return 1.0
    return 0.0


def _last_security_change_days(repo_path: Path) -> Optional[float]:
    now = dt.datetime.utcnow()
    candidates: List[Path] = []
    for rel in [
        "SECURITY.md",
        ".github/SECURITY.md",
        ".github/dependabot.yml",
        "renovate.json",
        "renovate.json5",
    ]:
        p = repo_path / rel
        if p.exists():
            candidates.append(p)

    wf = repo_path / ".github" / "workflows"
    if wf.exists():
        for p in wf.rglob("*"):
            if p.is_file() and p.suffix.lower() in {".yml", ".yaml"}:
                try:
                    txt = p.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    txt = ""
                if re.search(r"\b(dependabot|codeql|pip-audit|npm audit|osv-scanner|snyk|safety)\b", txt, re.I):
                    candidates.append(p)

    if not candidates:
        return None
    days = [_git_last_change_days(repo_path, p, now) for p in candidates]
    vals = [d for d in days if d is not None]
    if not vals:
        return None
    return min(vals)


def compute_security_responsiveness(repo_path: Path, github_url: Optional[str] = None, github_token: Optional[str] = None) -> Tuple[Optional[float], Dict[str, Any]]:
    """
    Security responsiveness (maintenance posture):
    - security policy presence (SECURITY.md / GH metadata)
    - security scanning automation (workflows: dependabot/renovate/osv/pip-audit/npm audit/etc.)
    - open vulnerability pressure (optional: pip-audit if available) -> None if not measurable
    """
    if not repo_path.exists():
        return None, {"available": False, "reason": "repo_path_missing"}

    sr1 = _security_policy_presence(repo_path)
    tool_count, tools = _security_tools_from_workflows(repo_path)
    # Continuous saturation from observed tool diversity (no piecewise bins).
    sr2 = sat_exp(tool_count, scale=2.0)
    sr3 = _security_automation_presence(repo_path)
    sec_days = _last_security_change_days(repo_path)
    sr4 = _exp_decay_days(sec_days, 180.0) if sec_days is not None else None

    block = mean([sr1, sr2, sr3, sr4])
    return block, {
        "available": True,
        "block": block,
        "sub": {
            "SR1_security_policy_presence": sr1,
            "SR2_automated_security_scanning": sr2,
            "SR3_dependency_automation": sr3,
            "SR4_security_update_recency": sr4,
        },
        "raw": {
            "security_tools_detected_count": tool_count,
            "security_tools_detected": tools,
            "days_since_security_change": sec_days,
        },
        "missingness_note": "SR4 is None when no security-related files/workflows are detected; other security sub-signals are observable 0/1-style evidence.",
    }


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


def learn_mqs_weights(dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Learn MQS block weights from a RECOPS corpus.
    Expected entries contain block values under one of:
    - {"IM":..., "UC":..., "SR":...}
    - {"N_issue_management":..., "N_update_consistency":..., "N_security_responsiveness":...}
    """
    im_vals: List[Optional[float]] = []
    uc_vals: List[Optional[float]] = []
    sr_vals: List[Optional[float]] = []

    for row in dataset:
        im = row.get("IM", row.get("N_issue_management"))
        uc = row.get("UC", row.get("N_update_consistency"))
        sr = row.get("SR", row.get("N_security_responsiveness"))
        im_vals.append(float(im) if im is not None else None)
        uc_vals.append(float(uc) if uc is not None else None)
        sr_vals.append(float(sr) if sr is not None else None)

    def _coverage(vals: List[Optional[float]]) -> float:
        return sum(1 for v in vals if v is not None) / max(1, len(vals))

    im_obs = [v for v in im_vals if v is not None]
    uc_obs = [v for v in uc_vals if v is not None]
    sr_obs = [v for v in sr_vals if v is not None]

    info = {
        "issue_management": _robust_variance(im_obs),
        "update_consistency": _robust_variance(uc_obs),
        "security_responsiveness": _robust_variance(sr_obs),
    }
    coverage = {
        "issue_management": _coverage(im_vals),
        "update_consistency": _coverage(uc_vals),
        "security_responsiveness": _coverage(sr_vals),
    }

    corr_im_uc = _pairwise_abs_corr(im_vals, uc_vals)
    corr_im_sr = _pairwise_abs_corr(im_vals, sr_vals)
    corr_uc_sr = _pairwise_abs_corr(uc_vals, sr_vals)
    uniq = {
        "issue_management": clamp01(1.0 - mean([corr_im_uc, corr_im_sr]) if mean([corr_im_uc, corr_im_sr]) is not None else 1.0),
        "update_consistency": clamp01(1.0 - mean([corr_im_uc, corr_uc_sr]) if mean([corr_im_uc, corr_uc_sr]) is not None else 1.0),
        "security_responsiveness": clamp01(1.0 - mean([corr_im_sr, corr_uc_sr]) if mean([corr_im_sr, corr_uc_sr]) is not None else 1.0),
    }

    raw = {
        k: max(1e-9, info[k] * coverage[k] * uniq[k])
        for k in ["issue_management", "update_consistency", "security_responsiveness"]
    }
    total_raw = sum(raw.values())
    weights = {k: raw[k] / total_raw for k in raw}

    return {
        "weights": weights,
        "raw_components": {"info": info, "coverage": coverage, "uniq": uniq, "raw": raw},
        "n_repos": len(dataset),
    }


def _compute_mqs_blocks(repo_url: str, github_token: Optional[str] = None, local_repo: Optional[Path] = None) -> Tuple[Dict[str, Optional[float]], Dict[str, Any], Path]:
    repo_path = Path(local_repo) if local_repo is not None else clone_or_download(repo_url)
    im, im_d = compute_issue_management(repo_url=repo_url, github_token=github_token, months=12)
    uc, uc_d = compute_update_consistency(repo_path)
    sr, sr_d = compute_security_responsiveness(repo_path, github_url=repo_url, github_token=github_token)
    blocks = {"issue_management": im, "update_consistency": uc, "security_responsiveness": sr}
    details = {"issue_management": im_d, "update_consistency": uc_d, "security_responsiveness": sr_d}
    return blocks, details, repo_path


def compute_mqs(
    repo_url: str,
    github_token: Optional[str] = None,
    aggregation: str = "geometric",
    learned_weights: Optional[Dict[str, float]] = None,
    local_repo: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Compute MQS for one repository URL.
    Weighting priority:
    1) corpus-learned weights when provided
    2) repository-level data-driven fallback from observed block dispersion
    """
    blocks, details, repo_path = _compute_mqs_blocks(repo_url=repo_url, github_token=github_token, local_repo=local_repo)

    weights_source = "learned_from_corpus" if learned_weights is not None else "repo_data_driven_fallback"
    weights = learned_weights or _data_driven_weights_from_blocks(
        {
            "issue_management": blocks["issue_management"],
            "update_consistency": blocks["update_consistency"],
            "security_responsiveness": blocks["security_responsiveness"],
        }
    )
    pairs = [
        (blocks["issue_management"], float(weights.get("issue_management", 0.0))),
        (blocks["update_consistency"], float(weights.get("update_consistency", 0.0))),
        (blocks["security_responsiveness"], float(weights.get("security_responsiveness", 0.0))),
    ]

    agg = (aggregation or "").strip().lower()
    if agg == "sum":
        final_0_1 = weighted_sum(pairs)
        agg_name = "weighted_sum"
    else:
        final_0_1 = weighted_geometric(pairs)
        agg_name = "weighted_geometric"
    score = 100.0 * float(final_0_1) if final_0_1 is not None else None

    return {
        "repository_path": str(repo_path),
        "score": score,
        "band": band(score) if score is not None else None,
        "aggregation": agg_name,
        "details": {
            "MQS": score,
            "N_issue_management": blocks["issue_management"],
            "N_update_consistency": blocks["update_consistency"],
            "N_security_responsiveness": blocks["security_responsiveness"],
            "weights": weights,
            "weights_source": weights_source,
            "blocks": details,
            "missingness_note": "Missing block values remain None and are excluded from weighted aggregation with renormalized effective weights.",
        },
    }


# =========================================================
# FCS (Funding Continuity Score)
# =========================================================

_RX_URL = re.compile(r"https?://[^\s\)\]\}>\"']+")
_INSTITUTIONAL_EMAIL_RX = re.compile(r"\.(edu|ac\.[a-z]{2}|gov|gouv\.fr)$|\.org$|europa\.eu$|\.edu\.[a-z]{2}$", re.I)
_INSTITUTIONAL_DOMAIN_RX = re.compile(r"\.edu$|\.gov$|\.ac\.[a-z]{2}$|europa\.eu$|cnrs\.fr$|inria\.fr$", re.I)
_COMMERCIAL_HINT_RX = re.compile(r"\b(inc|ltd|llc|gmbh|sas|sa|ab|bv|corp|company|technologies|systems|solutions|software|analytics|energy|consulting)\b", re.I)
_GRANT_ID_RX = re.compile(
    r"(?:grant\s*(?:no\.?|number)|award\s*number|contract\s*number|agreement\s*no\.?|ga\s*no\.?|no\.\s*\d{6,}|[A-Z]{2,}-\d{3,}|\b\d{6,}\b)",
    re.I,
)
_FCS_FALLBACK_SCALES: Dict[str, float] = {
    "IB2_scale": 50.0,
    "IB3_scale": 5.0,
    "CS1_scale": 20.0,
    "CS2_scale": 3.0,
    "CS4_scale": 15.0,
    "SP2_scale": 2.0,
    "SP3_scale": 5.0,
    "SP4_half_life_days": 365.0,
    "GC1_scale": 30.0,
    "GC2_scale": 5.0,
    "GC3_scale": 3.0,
    "GC4_scale": 3.0,
}


def _safe_read_text(p: Path) -> Optional[str]:
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None


def _median_positive(vals: List[float]) -> Optional[float]:
    pos = sorted([float(v) for v in vals if v is not None and float(v) > 0])
    if not pos:
        return None
    n = len(pos)
    mid = n // 2
    if n % 2 == 1:
        return pos[mid]
    return 0.5 * (pos[mid - 1] + pos[mid])


def _percentile_positive(vals: List[float], q: float) -> Optional[float]:
    pos = sorted([float(v) for v in vals if v is not None and float(v) > 0])
    if not pos:
        return None
    if len(pos) == 1:
        return pos[0]
    idx = clamp01(q) * (len(pos) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return pos[lo]
    w = idx - lo
    return pos[lo] * (1.0 - w) + pos[hi] * w


def _fcs_effective_scale(calibration: Optional[Dict[str, Any]], key: str) -> Optional[float]:
    sig = ((calibration or {}).get("signals", {}) or {})
    learned = sig.get(key)
    if learned is not None:
        try:
            if float(learned) > 0:
                return float(learned)
        except Exception:
            pass
    return _FCS_FALLBACK_SCALES.get(key)


def _calibrated_count_score(raw: Optional[float], scale: Optional[float]) -> Optional[float]:
    if raw is None:
        return None
    if scale is None or float(scale) <= 0:
        return None
    return sat_exp(raw, float(scale))


def _calibrated_recency_score(days: Optional[float], half_life: Optional[float]) -> Optional[float]:
    if days is None:
        return None
    if half_life is None or float(half_life) <= 0:
        return None
    return exp_recency(days, float(half_life))


def _extract_urls(text: Optional[str]) -> List[str]:
    if not text:
        return []
    return _RX_URL.findall(text)


def _url_domain(url: str) -> Optional[str]:
    try:
        netloc = (urlparse(url).netloc or "").lower().strip()
    except Exception:
        return None
    if not netloc:
        return None
    if "@" in netloc:
        netloc = netloc.split("@", 1)[-1]
    if ":" in netloc:
        netloc = netloc.split(":", 1)[0]
    return netloc or None


def _is_institutional_domain(d: str) -> bool:
    if _INSTITUTIONAL_DOMAIN_RX.search(d or ""):
        return True
    known = [
        "university",
        "institute",
        "research",
        "national-lab",
        "mit.edu",
        "stanford.edu",
        "ethz.ch",
        "kth.se",
    ]
    lo = (d or "").lower()
    return any(k in lo for k in known)


def _read_candidate_texts(repo_path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    candidates = [
        "README.md",
        "README.rst",
        "CITATION.cff",
        "codemeta.json",
        "pyproject.toml",
        "package.json",
        ".github/FUNDING.yml",
        "ACKNOWLEDGEMENTS",
        "ACKNOWLEDGMENTS",
        "FUNDING",
    ]
    for rel in candidates:
        p = repo_path / rel
        if p.exists() and p.is_file():
            txt = _safe_read_text(p)
            if txt is not None:
                out[str(p.relative_to(repo_path))] = txt
    for drel in ["docs", "doc", ".github"]:
        d = repo_path / drel
        if not d.exists() or not d.is_dir():
            continue
        for p in d.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in {".md", ".rst", ".txt", ".yml", ".yaml", ".json", ".toml", ".cff"}:
                continue
            txt = _safe_read_text(p)
            if txt is not None:
                out[str(p.relative_to(repo_path))] = txt
    return out


def _keyword_hits(texts: Dict[str, str], keywords: List[str]) -> int:
    if not texts:
        return 0
    pats = [re.compile(re.escape(k), re.I) for k in keywords]
    total = 0
    for txt in texts.values():
        for rx in pats:
            total += len(rx.findall(txt or ""))
    return total


def _gh_repo_file_text(owner: str, repo: str, path: str, token: Optional[str]) -> Optional[str]:
    data = gh_api_get(f"https://api.github.com/repos/{owner}/{repo}/contents/{path}", token=token)
    if not isinstance(data, dict):
        return None
    content = data.get("content")
    enc = (data.get("encoding") or "").lower()
    if isinstance(content, str) and enc == "base64":
        try:
            return base64.b64decode(content).decode("utf-8", errors="ignore")
        except Exception:
            return None
    return None


def _git_last_change_days_rel(repo_path: Path, rel_path: str, now: Optional[dt.datetime] = None) -> Optional[float]:
    now = now or dt.datetime.utcnow()
    code, out, _ = run(["git", "log", "-1", "--format=%ct", "--", rel_path], cwd=repo_path)
    if code != 0:
        return None
    ts = (out or "").strip()
    if not ts.isdigit():
        return None
    t = dt.datetime.utcfromtimestamp(int(ts))
    return max(0.0, (now - t).total_seconds() / 86400.0)


def _git_file_years(repo_path: Path, rel_path: str) -> List[int]:
    code, out, _ = run(["git", "log", "--follow", "--format=%ct", "--", rel_path], cwd=repo_path)
    if code != 0:
        return []
    years = set()
    for line in (out or "").splitlines():
        s = line.strip()
        if s.isdigit():
            years.add(dt.datetime.utcfromtimestamp(int(s)).year)
    return sorted(years)


def _git_recent_commit_email_domains(repo_path: Path, months: int = 24) -> Optional[List[str]]:
    since = (dt.datetime.utcnow() - dt.timedelta(days=int(months * 30.4375))).strftime("%Y-%m-%d")
    code, out, _ = run(["git", "log", f"--since={since}", "--format=%ae"], cwd=repo_path)
    if code != 0:
        return None
    domains: List[str] = []
    for line in (out or "").splitlines():
        email = line.strip().lower()
        if "@" not in email:
            continue
        dom = email.split("@", 1)[1]
        domains.append(dom)
    return domains


def _norm_weights_from_rows(rows: List[Dict[str, Optional[float]]], keys: List[str]) -> Dict[str, float]:
    vals = {k: [r.get(k) for r in rows] for k in keys}

    def _coverage(arr: List[Optional[float]]) -> float:
        return sum(1 for x in arr if x is not None) / max(1, len(arr))

    info = {k: _robust_variance([x for x in vals[k] if x is not None]) for k in keys}
    coverage = {k: _coverage(vals[k]) for k in keys}
    uniq: Dict[str, float] = {}
    for k in keys:
        corr_others: List[Optional[float]] = []
        for j in keys:
            if j == k:
                continue
            corr_others.append(_pairwise_abs_corr(vals[k], vals[j]))
        c = mean(corr_others)
        uniq[k] = clamp01(1.0 - c) if c is not None else 1.0
    raw = {k: max(1e-9, info[k] * coverage[k] * uniq[k]) for k in keys}
    z = sum(raw.values()) or 1.0
    return {k: raw[k] / z for k in keys}


def _data_driven_weights_from_blocks(blocks: Dict[str, Optional[float]]) -> Dict[str, float]:
    """
    Repository-level fallback when corpus-learned weights are unavailable:
    - availability-driven: only observed blocks can get weight
    - dispersion-driven: farther from observed mean gets more weight
    """
    observed = {k: float(v) for k, v in blocks.items() if v is not None}
    if not observed:
        return {k: 0.0 for k in blocks}
    mu = _py_mean(list(observed.values()))
    raw = {k: abs(v - mu) for k, v in observed.items()}
    z = sum(raw.values())
    if z <= 0:
        # If observed blocks are identical, fallback to availability-only normalization.
        raw = {k: 1.0 for k in observed}
        z = float(len(observed))
    # Smooth toward uniform weights to avoid over-concentration on one
    # extreme block in sparse-evidence situations.
    n = float(len(observed))
    smooth = 0.50
    uniform = 1.0 / n
    out = {k: 0.0 for k in blocks}
    for k, rv in raw.items():
        out[k] = (1.0 - smooth) * (rv / z) + smooth * uniform
    return out


def compute_institutional_backing(
    repo_path: Path,
    repo_url: str,
    github_token: Optional[str] = None,
    calibration: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[float], Dict[str, Any]]:
    owner, repo = parse_github_owner_repo(repo_url)
    texts = _read_candidate_texts(repo_path)

    owner_meta = None
    if owner:
        owner_meta = gh_api_get(f"https://api.github.com/users/{owner}", token=github_token)
        if not isinstance(owner_meta, dict):
            owner_meta = gh_api_get(f"https://api.github.com/orgs/{owner}", token=github_token)
    owner_type = owner_meta.get("type") if isinstance(owner_meta, dict) else None
    ib1 = 1.0 if owner_type == "Organization" else (0.0 if owner_type == "User" else None)

    institutional_keywords = [
        "university", "universite", "institute", "laboratory", "lab", "research center", "centre de recherche",
        "foundation", "consortium", "agency", "department", "ministry", "national lab", "cnrs", "cea",
        "inria", "nrel", "doe", "eu horizon", "european commission", "nsf", "ukri", "fraunhofer",
        "mit", "stanford", "eth", "kth",
    ]
    ib2_raw = _keyword_hits(texts, institutional_keywords)

    urls = set()
    for txt in texts.values():
        urls.update(_extract_urls(txt))
    if isinstance(owner_meta, dict):
        for k in ["blog", "company"]:
            if isinstance(owner_meta.get(k), str):
                urls.update(_extract_urls(owner_meta.get(k)))
    inst_domains = set()
    for u in urls:
        d = _url_domain(u)
        if d and _is_institutional_domain(d):
            inst_domains.add(d)
    ib3_raw = len(inst_domains)

    domains = _git_recent_commit_email_domains(repo_path, months=24)
    ib4 = None
    if domains is not None:
        valid = [d for d in domains if d and "noreply.github.com" not in d]
        if valid:
            ib4 = clamp01(sum(1 for d in valid if _INSTITUTIONAL_EMAIL_RX.search(d or "")) / len(valid))

    ib2 = _calibrated_count_score(ib2_raw, _fcs_effective_scale(calibration, "IB2_scale"))
    ib3 = _calibrated_count_score(ib3_raw, _fcs_effective_scale(calibration, "IB3_scale"))
    ib = mean([ib1, ib2, ib3, ib4])
    return ib, {"sub": {"IB1": ib1, "IB2": ib2, "IB3": ib3, "IB4": ib4}, "raw": {"IB2_hits": ib2_raw, "IB3_domains": ib3_raw, "IB4_recent_commit_share": ib4, "owner_type": owner_type}}


def compute_commercial_support(
    repo_path: Path,
    repo_url: str,
    github_token: Optional[str] = None,
    calibration: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[float], Dict[str, Any]]:
    owner, repo = parse_github_owner_repo(repo_url)
    texts = _read_candidate_texts(repo_path)
    text_blob = "\n".join(texts.values())
    owner_meta = gh_api_get(f"https://api.github.com/users/{owner}", token=github_token) if owner else None

    kw1 = [
        "commercial support", "enterprise support", "professional support", "paid support", "consulting", "consultancy",
        "training", "workshop", "service contract", "support contract", "sla", "managed service", "hosted service",
        "cloud service", "pricing", "contact sales", "book a demo",
    ]
    cs1_raw = _keyword_hits(texts, kw1)

    cu_patterns = [r"pricing", r"enterprise", r"support", r"services", r"consulting", r"training", r"contact", r"sales", r"demo"]
    url_count = 0
    for u in set(_extract_urls(text_blob)):
        if any(re.search(p, u, re.I) for p in cu_patterns):
            url_count += 1
    cs2_raw = url_count

    cs3 = None
    if isinstance(owner_meta, dict):
        if owner_meta.get("type") == "Organization":
            hint_blob = " ".join(str(owner_meta.get(k, "")) for k in ["company", "name", "bio", "blog"])
            cs3 = 1.0 if _COMMERCIAL_HINT_RX.search(hint_blob or "") else 0.0
        elif owner_meta.get("type") == "User":
            cs3 = 0.0

    kw4 = [
        "cloud", "hosted", "saas", "subscription", "license key", "enterprise edition", "professional edition",
        "paid plan", "commercial license", "dual license",
    ]
    cs4_raw = _keyword_hits(texts, kw4)

    cs1 = _calibrated_count_score(cs1_raw, _fcs_effective_scale(calibration, "CS1_scale"))
    cs2 = _calibrated_count_score(cs2_raw, _fcs_effective_scale(calibration, "CS2_scale"))
    cs4 = _calibrated_count_score(cs4_raw, _fcs_effective_scale(calibration, "CS4_scale"))
    cs = mean([cs1, cs2, cs3, cs4])
    return cs, {"sub": {"CS1": cs1, "CS2": cs2, "CS3": cs3, "CS4": cs4}, "raw": {"CS1_hits": cs1_raw, "CS2_url_count": cs2_raw, "CS4_hits": cs4_raw}}


def compute_sponsorship_programs(
    repo_path: Path,
    repo_url: str,
    github_token: Optional[str] = None,
    calibration: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[float], Dict[str, Any]]:
    owner, repo = parse_github_owner_repo(repo_url)
    texts = _read_candidate_texts(repo_path)
    funding_path = repo_path / ".github" / "FUNDING.yml"
    funding_txt = _safe_read_text(funding_path) if funding_path.exists() else None
    if funding_txt is None and owner and repo:
        funding_txt = _gh_repo_file_text(owner, repo, ".github/FUNDING.yml", token=github_token)
    sp1 = 1.0 if funding_txt is not None else 0.0

    platform_count: Optional[int] = 0
    funding_parsed = False
    if funding_txt and yaml is not None:
        try:
            y = yaml.safe_load(funding_txt) or {}
            if isinstance(y, dict):
                funding_parsed = True
                fields = ["github", "patreon", "open_collective", "ko_fi", "tidelift", "community_bridge", "liberapay", "issuehunt", "otechie", "custom"]
                for f in fields:
                    v = y.get(f)
                    if isinstance(v, list):
                        if any(str(x).strip() for x in v):
                            platform_count += 1
                    elif isinstance(v, str):
                        if v.strip():
                            platform_count += 1
        except Exception:
            platform_count = None
    elif funding_txt and yaml is None:
        platform_count = None
    sp2_raw = platform_count

    sp3_kw = ["sponsor", "sponsorship", "donate", "donation", "funding", "support us", "opencollective", "github sponsors", "patreon", "liberapay", "ko-fi", "tidelift"]
    sp3_raw = _keyword_hits(texts, sp3_kw)

    sp4_days = _git_last_change_days_rel(repo_path, ".github/FUNDING.yml") if sp1 > 0 else None

    sp2 = _calibrated_count_score(sp2_raw, _fcs_effective_scale(calibration, "SP2_scale")) if sp2_raw is not None else None
    sp3 = _calibrated_count_score(sp3_raw, _fcs_effective_scale(calibration, "SP3_scale"))
    sp4 = _calibrated_recency_score(sp4_days, _fcs_effective_scale(calibration, "SP4_half_life_days"))
    sp = mean([sp1, sp2, sp3, sp4])
    return sp, {"sub": {"SP1": sp1, "SP2": sp2, "SP3": sp3, "SP4": sp4}, "raw": {"SP2_platform_count": sp2_raw, "SP3_hits": sp3_raw, "SP4_days_since_funding_change": sp4_days, "SP2_funding_parsed": funding_parsed}}


def compute_grant_continuity(
    repo_path: Path,
    repo_url: str,
    github_token: Optional[str] = None,
    calibration: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[float], Dict[str, Any]]:
    texts = _read_candidate_texts(repo_path)
    grant_kw = [
        "grant", "funded by", "funding from", "acknowledgement", "acknowledgment", "supported by", "award", "project number",
        "contract number", "horizon europe", "h2020", "european commission", "erc", "marie curie", "nsf", "doe", "arpa-e",
        "ukri", "epsrc", "innovate uk", "anr", "dfg", "bmbf", "nwo", "vinnova", "formas", "energimyndigheten",
    ]
    gc1_raw = _keyword_hits(texts, grant_kw)

    ids = set()
    for txt in texts.values():
        for m in _GRANT_ID_RX.findall(txt or ""):
            ids.add(str(m).strip().lower())
    gc2_raw = len(ids)

    family_dict = {
        "EU": ["horizon europe", "h2020", "european commission", "erc", "marie curie"],
        "US federal": ["nsf", "doe", "arpa-e"],
        "UK research council": ["ukri", "epsrc", "innovate uk"],
        "French agency": ["anr"],
        "German agency": ["dfg", "bmbf", "fraunhofer"],
        "Nordic agency": ["vinnova", "formas", "energimyndigheten"],
        "university": ["university", "universite", "institute"],
        "foundation": ["foundation"],
        "industry consortium": ["consortium"],
    }
    blob = "\n".join(texts.values()).lower()
    families = {fam for fam, terms in family_dict.items() if any(t in blob for t in terms)}
    gc3_raw = len(families)

    grant_files = []
    for rel, txt in texts.items():
        if _keyword_hits({rel: txt}, grant_kw) > 0 or _GRANT_ID_RX.search(txt or ""):
            grant_files.append(rel)
    years = set()
    for rel in grant_files:
        years.update(_git_file_years(repo_path, rel))
    gc4_raw = len(years)

    gc1 = _calibrated_count_score(gc1_raw, _fcs_effective_scale(calibration, "GC1_scale"))
    gc2 = _calibrated_count_score(gc2_raw, _fcs_effective_scale(calibration, "GC2_scale"))
    gc3 = _calibrated_count_score(gc3_raw, _fcs_effective_scale(calibration, "GC3_scale"))
    gc4 = _calibrated_count_score(gc4_raw, _fcs_effective_scale(calibration, "GC4_scale"))
    gc = mean([gc1, gc2, gc3, gc4])
    return gc, {"sub": {"GC1": gc1, "GC2": gc2, "GC3": gc3, "GC4": gc4}, "raw": {"GC1_hits": gc1_raw, "GC2_identifier_count": gc2_raw, "GC3_funder_family_count": gc3_raw, "GC4_grant_evidence_years": gc4_raw}}


def learn_fcs_calibration(dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    sig_keys = [
        "IB2_scale", "IB3_scale", "CS1_scale", "CS2_scale", "CS4_scale",
        "SP2_scale", "SP3_scale", "SP4_half_life_days",
        "GC1_scale", "GC2_scale", "GC3_scale", "GC4_scale",
    ]
    raw_map = {
        "IB2_scale": "IB2_hits",
        "IB3_scale": "IB3_domains",
        "CS1_scale": "CS1_hits",
        "CS2_scale": "CS2_url_count",
        "CS4_scale": "CS4_hits",
        "SP2_scale": "SP2_platform_count",
        "SP3_scale": "SP3_hits",
        "SP4_half_life_days": "SP4_days_since_funding_change",
        "GC1_scale": "GC1_hits",
        "GC2_scale": "GC2_identifier_count",
        "GC3_scale": "GC3_funder_family_count",
        "GC4_scale": "GC4_grant_evidence_years",
    }
    vals: Dict[str, List[float]] = {k: [] for k in sig_keys}
    rows_for_weights: List[Dict[str, Optional[float]]] = []

    for row in dataset:
        raw = row.get("raw", row)
        for sk in sig_keys:
            rk = raw_map[sk]
            v = raw.get(rk)
            if v is not None:
                try:
                    vals[sk].append(float(v))
                except Exception:
                    pass
        rows_for_weights.append(
            {
                "IB1": row.get("IB1"), "IB2": row.get("IB2"), "IB3": row.get("IB3"), "IB4": row.get("IB4"),
                "CS1": row.get("CS1"), "CS2": row.get("CS2"), "CS3": row.get("CS3"), "CS4": row.get("CS4"),
                "SP1": row.get("SP1"), "SP2": row.get("SP2"), "SP3": row.get("SP3"), "SP4": row.get("SP4"),
                "GC1": row.get("GC1"), "GC2": row.get("GC2"), "GC3": row.get("GC3"), "GC4": row.get("GC4"),
                "IB": row.get("IB", row.get("N_institutional_backing")),
                "CS": row.get("CS", row.get("N_commercial_support")),
                "SP": row.get("SP", row.get("N_sponsorship_programs")),
                "GC": row.get("GC", row.get("N_grant_continuity")),
            }
        )

    signals: Dict[str, Optional[float]] = {}
    for sk in sig_keys:
        med = _median_positive(vals[sk])
        if med is None or med <= 0:
            med = _percentile_positive(vals[sk], 0.75)
        signals[sk] = med if med is not None and med > 0 else None

    internal = {
        "IB": _norm_weights_from_rows(rows_for_weights, ["IB1", "IB2", "IB3", "IB4"]),
        "CS": _norm_weights_from_rows(rows_for_weights, ["CS1", "CS2", "CS3", "CS4"]),
        "SP": _norm_weights_from_rows(rows_for_weights, ["SP1", "SP2", "SP3", "SP4"]),
        "GC": _norm_weights_from_rows(rows_for_weights, ["GC1", "GC2", "GC3", "GC4"]),
    }
    blocks = _norm_weights_from_rows(rows_for_weights, ["IB", "CS", "SP", "GC"])
    return {"signals": signals, "internal_weights": internal, "block_weights": blocks, "n_repos": len(dataset)}


def learn_fcs_weights(dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    rows = []
    for row in dataset:
        rows.append(
            {
                "IB": row.get("IB", row.get("N_institutional_backing")),
                "CS": row.get("CS", row.get("N_commercial_support")),
                "SP": row.get("SP", row.get("N_sponsorship_programs")),
                "GC": row.get("GC", row.get("N_grant_continuity")),
            }
        )
    weights = _norm_weights_from_rows(rows, ["IB", "CS", "SP", "GC"])
    return {"weights": {"institutional_backing": weights["IB"], "commercial_support": weights["CS"], "sponsorship_programs": weights["SP"], "grant_continuity": weights["GC"]}, "n_repos": len(dataset)}


def compute_fcs(
    repo_path: Path,
    repo_url: str,
    github_token: Optional[str] = None,
    learned_weights: Optional[Dict[str, float]] = None,
    calibration: Optional[Dict[str, Any]] = None,
    aggregation: str = "geometric",
) -> Dict[str, Any]:
    repo_path = Path(repo_path)
    ib, ib_d = compute_institutional_backing(repo_path, repo_url, github_token=github_token, calibration=calibration)
    cs, cs_d = compute_commercial_support(repo_path, repo_url, github_token=github_token, calibration=calibration)
    sp, sp_d = compute_sponsorship_programs(repo_path, repo_url, github_token=github_token, calibration=calibration)
    gc, gc_d = compute_grant_continuity(repo_path, repo_url, github_token=github_token, calibration=calibration)

    learned_block_weights = learned_weights or (calibration or {}).get("block_weights")
    used_learned_weights = learned_block_weights is not None
    if learned_block_weights is None:
        dyn = _data_driven_weights_from_blocks(
            {"institutional_backing": ib, "commercial_support": cs, "sponsorship_programs": sp, "grant_continuity": gc}
        )
        bw = dyn
    else:
        bw = learned_block_weights
    pairs = [
        (ib, float(bw.get("institutional_backing", bw.get("IB", 0.0)))),
        (cs, float(bw.get("commercial_support", bw.get("CS", 0.0)))),
        (sp, float(bw.get("sponsorship_programs", bw.get("SP", 0.0)))),
        (gc, float(bw.get("grant_continuity", bw.get("GC", 0.0)))),
    ]
    agg = (aggregation or "").strip().lower()
    final_0_1 = weighted_sum(pairs) if agg == "sum" else weighted_geometric(pairs)
    score = 100.0 * float(final_0_1) if final_0_1 is not None else None
    return {
        "FCS": score,
        "N_institutional_backing": ib,
        "N_commercial_support": cs,
        "N_sponsorship_programs": sp,
        "N_grant_continuity": gc,
        "weights": bw,
        "weights_source": "learned_from_corpus" if used_learned_weights else "repo_data_driven_fallback",
        "calibration": calibration,
        "calibration_source": "learned_corpus_calibration" if calibration else "fallback_default_scales",
        "raw": {
            "institutional_backing": ib_d.get("raw"),
            "commercial_support": cs_d.get("raw"),
            "sponsorship_programs": sp_d.get("raw"),
            "grant_continuity": gc_d.get("raw"),
        },
        "blocks": {
            "institutional_backing": ib_d,
            "commercial_support": cs_d,
            "sponsorship_programs": sp_d,
            "grant_continuity": gc_d,
        },
        "aggregation": "weighted_sum" if agg == "sum" else "weighted_geometric",
        "missingness_note": "Unavailable data are kept as None; observable absence is scored 0; present evidence yields >0 normalized scores.",
    }


# =========================================================
# PVI final
# =========================================================


def learn_pvi_weights(dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    rows = []
    for row in dataset:
        rows.append(
            {
                "commit_activity": row.get("commit_activity", row.get("N_commit_activity")),
                "release_regularity": row.get("release_regularity", row.get("N_release_regularity")),
                "issue_activity": row.get("issue_activity", row.get("N_issue_activity")),
                "community_growth": row.get("community_growth", row.get("N_community_growth")),
            }
        )
    weights = _norm_weights_from_rows(rows, ["commit_activity", "release_regularity", "issue_activity", "community_growth"])
    return {"weights": weights, "n_repos": len(dataset)}


_MRD_FALLBACK_CALIB: Dict[str, float] = {
    "IFR1_half_life_days": 7.0,
    "PRD2_half_life_days": 7.0,
    "PRD3_half_life_days": 30.0,
    "PRD4_half_life_days": 90.0,
    "MP1_scale": 3.0,
    "MP3_scale": 50.0,
}


def _mrd_effective_scale(calibration: Optional[Dict[str, Any]], key: str) -> Optional[float]:
    sig = ((calibration or {}).get("signals", {}) or {})
    v = sig.get(key)
    try:
        if v is not None and float(v) > 0:
            return float(v)
    except Exception:
        pass
    return _MRD_FALLBACK_CALIB.get(key)


def _median(vals: List[float]) -> Optional[float]:
    xs = sorted([float(v) for v in vals if v is not None])
    if not xs:
        return None
    n = len(xs)
    m = n // 2
    return xs[m] if n % 2 == 1 else 0.5 * (xs[m - 1] + xs[m])


def _month_key_dt(d: dt.datetime) -> str:
    return f"{d.year:04d}-{d.month:02d}"


def _learned_or_mean(sub_scores: Dict[str, Optional[float]], learned_sub_weights: Optional[Dict[str, float]]) -> Optional[float]:
    if learned_sub_weights:
        pairs = [(sub_scores.get(k), float(w)) for k, w in learned_sub_weights.items()]
        return weighted_sum(pairs)
    return mean(list(sub_scores.values()))


def compute_issue_first_response(
    repo_url: Optional[str],
    github_token: Optional[str] = None,
    months: int = 12,
    calibration: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[float], Dict[str, Any]]:
    owner, repo = parse_github_owner_repo(repo_url)
    if not owner or not repo:
        return None, {"available": False, "reason": "github_url_missing_or_unparseable"}
    if requests is None:
        return None, {"available": False, "reason": "requests_not_installed"}

    now = dt.datetime.utcnow()
    since = now - dt.timedelta(days=int(30.4375 * months))
    issues = _issues_from_github(owner, repo, token=github_token, since=since)
    if issues is None:
        return None, {"available": False, "reason": "github_issues_api_unavailable"}

    issues_12m = []
    for it in issues:
        created = _parse_iso8601(it.get("created_at"))
        if created and created >= since:
            issues_12m.append((it, created))
    created_n = len(issues_12m)
    if created_n == 0:
        sub = {"IFR1_median_first_response_speed": None, "IFR2_response_coverage": None, "IFR3_unanswered_issue_pressure": None, "IFR4_first_response_stability": None}
        return None, {"available": True, "block": None, "sub": sub, "raw": {"issues_created_12m": 0, "issues_with_first_comment": 0, "unanswered_issues_12m": 0, "median_first_response_days": None, "response_delay_cv": None, "window_months": months}}

    delays: List[float] = []
    with_first = 0
    unanswered = 0
    for it, created in issues_12m:
        num = it.get("number")
        if not isinstance(num, int):
            unanswered += 1
            continue
        comments = gh_api_get(f"https://api.github.com/repos/{owner}/{repo}/issues/{num}/comments?per_page=1&page=1", token=github_token)
        if isinstance(comments, list) and comments and isinstance(comments[0], dict):
            first_dt = _parse_iso8601(comments[0].get("created_at"))
            if first_dt is not None:
                with_first += 1
                delays.append(max(0.0, (first_dt - created).total_seconds() / 86400.0))
            else:
                unanswered += 1
        else:
            unanswered += 1

    median_delay = _median(delays)
    cv = None
    if len(delays) >= 2 and _py_mean(delays) > 0:
        m = _py_mean(delays)
        cv = math.sqrt(_py_mean([(d - m) ** 2 for d in delays])) / m

    sub = {
        "IFR1_median_first_response_speed": exp_recency(median_delay, _mrd_effective_scale(calibration, "IFR1_half_life_days")),
        "IFR2_response_coverage": ratio01(with_first, created_n),
        "IFR3_unanswered_issue_pressure": clamp01(1.0 - (float(unanswered) / float(created_n))),
        "IFR4_first_response_stability": clamp01(math.exp(-cv)) if cv is not None else None,
    }
    iw = (((calibration or {}).get("internal_weights", {}) or {}).get("IFR")) if calibration else None
    block = _learned_or_mean(sub, iw)
    return block, {
        "available": True,
        "block": block,
        "sub": sub,
        "raw": {
            "issues_created_12m": created_n,
            "issues_with_first_comment": with_first,
            "unanswered_issues_12m": unanswered,
            "median_first_response_days": median_delay,
            "response_delay_cv": cv,
            "window_months": months,
        },
    }


def compute_pr_review_discipline(
    repo_url: Optional[str],
    github_token: Optional[str] = None,
    months: int = 12,
    calibration: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[float], Dict[str, Any]]:
    owner, repo = parse_github_owner_repo(repo_url)
    if not owner or not repo:
        return None, {"available": False, "reason": "github_url_missing_or_unparseable"}
    if requests is None:
        return None, {"available": False, "reason": "requests_not_installed"}

    now = dt.datetime.utcnow()
    since = now - dt.timedelta(days=int(30.4375 * months))
    pulls = gh_paginate(f"https://api.github.com/repos/{owner}/{repo}/pulls?state=all&sort=created&direction=desc", token=github_token, per_page=100, max_pages=10)
    if pulls is None:
        return None, {"available": False, "reason": "github_prs_unavailable"}

    prs = []
    for pr in pulls:
        if not isinstance(pr, dict):
            continue
        c = _parse_iso8601(pr.get("created_at"))
        if c and c >= since:
            prs.append((pr, c))
    prs_n = len(prs)
    if prs_n == 0:
        sub = {"PRD1_review_coverage": None, "PRD2_median_time_to_first_review": None, "PRD3_pr_resolution_discipline": None, "PRD4_stale_pr_pressure": None, "PRD5_formal_review_ratio": None}
        return None, {
            "available": True,
            "block": None,
            "sub": sub,
            "raw": {
                "prs_created_12m": 0,
                "prs_with_review_or_comment": 0,
                "prs_with_formal_review": 0,
                "median_first_review_delay_days": None,
                "median_pr_resolution_days": None,
                "median_open_pr_age_days": None,
                "window_months": months,
            },
            "missingness_note": "No PR created in the target window; PR review discipline is treated as non-observable (None), not as strong performance.",
        }

    with_interaction = 0
    with_formal_review = 0
    review_delays: List[float] = []
    resolution_delays: List[float] = []
    open_ages: List[float] = []
    for pr, created in prs:
        num = pr.get("number")
        if not isinstance(num, int):
            continue
        reviews = gh_api_get(f"https://api.github.com/repos/{owner}/{repo}/pulls/{num}/reviews?per_page=100&page=1", token=github_token)
        issue_comments = gh_api_get(f"https://api.github.com/repos/{owner}/{repo}/issues/{num}/comments?per_page=1&page=1", token=github_token)
        first_review = None
        if isinstance(reviews, list) and reviews:
            rev_times = [_parse_iso8601(r.get("submitted_at")) for r in reviews if isinstance(r, dict)]
            rev_times = [t for t in rev_times if t is not None]
            if rev_times:
                first_review = min(rev_times)
                with_formal_review += 1
        first_comment = None
        if isinstance(issue_comments, list) and issue_comments and isinstance(issue_comments[0], dict):
            first_comment = _parse_iso8601(issue_comments[0].get("created_at"))
        first_interaction = first_review or first_comment
        if first_interaction is not None:
            with_interaction += 1
            review_delays.append(max(0.0, (first_interaction - created).total_seconds() / 86400.0))

        merged = _parse_iso8601(pr.get("merged_at"))
        closed = _parse_iso8601(pr.get("closed_at"))
        resolved = merged or closed
        if resolved is not None:
            resolution_delays.append(max(0.0, (resolved - created).total_seconds() / 86400.0))
        elif str(pr.get("state", "")).lower() == "open":
            open_ages.append(max(0.0, (now - created).total_seconds() / 86400.0))

    sub = {
        "PRD1_review_coverage": ratio01(with_interaction, prs_n),
        "PRD2_median_time_to_first_review": exp_recency(_median(review_delays), _mrd_effective_scale(calibration, "PRD2_half_life_days")),
        "PRD3_pr_resolution_discipline": exp_recency(_median(resolution_delays), _mrd_effective_scale(calibration, "PRD3_half_life_days")),
        "PRD4_stale_pr_pressure": exp_recency(_median(open_ages), _mrd_effective_scale(calibration, "PRD4_half_life_days")) if open_ages else 1.0,
        "PRD5_formal_review_ratio": ratio01(with_formal_review, prs_n),
    }
    iw = (((calibration or {}).get("internal_weights", {}) or {}).get("PRD")) if calibration else None
    block = _learned_or_mean(sub, iw)
    return block, {
        "available": True,
        "block": block,
        "sub": sub,
        "raw": {
            "prs_created_12m": prs_n,
            "prs_with_review_or_comment": with_interaction,
            "prs_with_formal_review": with_formal_review,
            "median_first_review_delay_days": _median(review_delays),
            "median_pr_resolution_days": _median(resolution_delays),
            "median_open_pr_age_days": _median(open_ages),
            "window_months": months,
        },
    }


def compute_maintainer_presence(
    repo_path: Path,
    repo_url: Optional[str],
    github_token: Optional[str] = None,
    months: int = 12,
    calibration: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[float], Dict[str, Any]]:
    now = dt.datetime.utcnow()
    since = now - dt.timedelta(days=int(30.4375 * months))
    since_iso = since.strftime("%Y-%m-%d")
    owner, repo = parse_github_owner_repo(repo_url)

    commit_authors = set(_git_contributors(Path(repo_path), since_iso=since_iso))
    maintainer_actions: Counter = Counter()
    maintainer_months: Dict[str, set] = {}
    for a in commit_authors:
        maintainer_actions[a] += 1
    commit_raw = git_commit_activity(Path(repo_path), months=months)
    if commit_raw.get("available"):
        for mk, c in (commit_raw.get("monthly_counts") or {}).items():
            if c > 0:
                for a in commit_authors:
                    maintainer_months.setdefault(a, set()).add(mk)

    reviews_n = 0
    merged_n = 0
    closed_issues_n = 0
    pulls_api_unavailable = False
    issues_api_unavailable = False
    if owner and repo and requests is not None:
        pulls = gh_paginate(f"https://api.github.com/repos/{owner}/{repo}/pulls?state=all&sort=created&direction=desc", token=github_token, per_page=100, max_pages=10)
        pulls_api_unavailable = pulls is None
        pulls_iter = pulls if pulls is not None else []
        for pr in pulls_iter:
            if not isinstance(pr, dict):
                continue
            created = _parse_iso8601(pr.get("created_at"))
            if created is None or created < since:
                continue
            num = pr.get("number")
            if not isinstance(num, int):
                continue
            mk = _month_key_dt(created)
            merged_at = _parse_iso8601(pr.get("merged_at"))
            if merged_at is not None:
                merged_n += 1
                u = pr.get("merged_by") or {}
                login = (u.get("login") or "").lower() if isinstance(u, dict) else ""
                if login:
                    maintainer_actions[login] += 1
                    maintainer_months.setdefault(login, set()).add(mk)
            reviews = gh_api_get(f"https://api.github.com/repos/{owner}/{repo}/pulls/{num}/reviews?per_page=100&page=1", token=github_token)
            if isinstance(reviews, list):
                for r in reviews:
                    if not isinstance(r, dict):
                        continue
                    user = r.get("user") or {}
                    login = (user.get("login") or "").lower() if isinstance(user, dict) else ""
                    submitted = _parse_iso8601(r.get("submitted_at"))
                    if login and submitted and submitted >= since:
                        reviews_n += 1
                        maintainer_actions[login] += 1
                        maintainer_months.setdefault(login, set()).add(_month_key_dt(submitted))

        issues = _issues_from_github(owner, repo, token=github_token, since=since)
        issues_api_unavailable = issues is None
        for it in issues or []:
            closed = _parse_iso8601(it.get("closed_at"))
            if closed is None or closed < since:
                continue
            u = it.get("closed_by") or {}
            login = (u.get("login") or "").lower() if isinstance(u, dict) else ""
            if login:
                closed_issues_n += 1
                maintainer_actions[login] += 1
                maintainer_months.setdefault(login, set()).add(_month_key_dt(closed))

    maintainers = {k for k, v in maintainer_actions.items() if v > 0}
    maint_actions_total = sum(maintainer_actions.values())
    active_maintainer_months = len({m for s in maintainer_months.values() for m in s})
    max_active_months = max((len(s) for s in maintainer_months.values()), default=0)
    top_share = (max(maintainer_actions.values()) / maint_actions_total) if maint_actions_total > 0 else 1.0

    sub = {
        "MP1_active_maintainer_count": sat_exp(len(maintainers), _mrd_effective_scale(calibration, "MP1_scale")),
        "MP2_maintainer_activity_persistence": ratio01(active_maintainer_months, 12),
        "MP3_maintainer_action_volume": sat_exp(maint_actions_total, _mrd_effective_scale(calibration, "MP3_scale")),
        "MP4_maintainer_continuity": ratio01(max_active_months, 12),
        "MP5_maintainer_concentration_safety": clamp01(1.0 - top_share),
    }
    iw = (((calibration or {}).get("internal_weights", {}) or {}).get("MP")) if calibration else None
    block = _learned_or_mean(sub, iw)
    return block, {
        "available": True,
        "block": block,
        "sub": sub,
        "raw": {
            "active_maintainer_count_12m": len(maintainers),
            "active_maintainer_months_12m": active_maintainer_months,
            "maintainer_action_count_12m": maint_actions_total,
            "max_active_months": max_active_months,
            "top_maintainer_action_share": top_share,
            "maintainer_commits_proxy": len(commit_authors),
            "merged_prs": merged_n,
            "closed_issues": closed_issues_n,
            "formal_reviews": reviews_n,
            "github_pull_pagination_unavailable": pulls_api_unavailable,
            "github_issues_timeline_unavailable": issues_api_unavailable,
            "window_months": months,
        },
    }


def learn_mrd_weights(dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    rows = []
    ifr_rows = []
    prd_rows = []
    mp_rows = []
    for row in dataset:
        rows.append(
            {
                "IFR": row.get("IFR", row.get("N_issue_first_response")),
                "PRD": row.get("PRD", row.get("N_pr_review_discipline")),
                "MP": row.get("MP", row.get("N_maintainer_presence")),
            }
        )
        ifr_rows.append({k: row.get(k) for k in ["IFR1_median_first_response_speed", "IFR2_response_coverage", "IFR3_unanswered_issue_pressure", "IFR4_first_response_stability"]})
        prd_rows.append({k: row.get(k) for k in ["PRD1_review_coverage", "PRD2_median_time_to_first_review", "PRD3_pr_resolution_discipline", "PRD4_stale_pr_pressure", "PRD5_formal_review_ratio"]})
        mp_rows.append({k: row.get(k) for k in ["MP1_active_maintainer_count", "MP2_maintainer_activity_persistence", "MP3_maintainer_action_volume", "MP4_maintainer_continuity", "MP5_maintainer_concentration_safety"]})

    block_weights = _norm_weights_from_rows(rows, ["IFR", "PRD", "MP"])
    internal = {
        "IFR": _norm_weights_from_rows(ifr_rows, ["IFR1_median_first_response_speed", "IFR2_response_coverage", "IFR3_unanswered_issue_pressure", "IFR4_first_response_stability"]),
        "PRD": _norm_weights_from_rows(prd_rows, ["PRD1_review_coverage", "PRD2_median_time_to_first_review", "PRD3_pr_resolution_discipline", "PRD4_stale_pr_pressure", "PRD5_formal_review_ratio"]),
        "MP": _norm_weights_from_rows(mp_rows, ["MP1_active_maintainer_count", "MP2_maintainer_activity_persistence", "MP3_maintainer_action_volume", "MP4_maintainer_continuity", "MP5_maintainer_concentration_safety"]),
    }
    return {"weights": {"issue_first_response": block_weights["IFR"], "pr_review_discipline": block_weights["PRD"], "maintainer_presence": block_weights["MP"]}, "block_weights": block_weights, "internal_weights": internal, "n_repos": len(dataset)}


def learn_mrd_calibration(dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    raw_map = {
        "IFR1_half_life_days": "median_first_response_days",
        "PRD2_half_life_days": "median_first_review_delay_days",
        "PRD3_half_life_days": "median_pr_resolution_days",
        "PRD4_half_life_days": "median_open_pr_age_days",
        "MP1_scale": "active_maintainer_count_12m",
        "MP3_scale": "maintainer_action_count_12m",
    }
    vals: Dict[str, List[float]] = {k: [] for k in raw_map}
    for row in dataset:
        raw = row.get("raw", row)
        for sk, rk in raw_map.items():
            v = raw.get(rk)
            if v is not None:
                try:
                    vals[sk].append(float(v))
                except Exception:
                    pass
    signals = {k: (_median_positive(vs) or _percentile_positive(vs, 0.75)) for k, vs in vals.items()}
    return {"signals": signals, "n_repos": len(dataset)}


def compute_mrd(
    repo_path: Path,
    repo_url: str,
    github_token: Optional[str] = None,
    learned_weights: Optional[Dict[str, Any]] = None,
    calibration: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cal = dict(calibration or {})
    if isinstance(learned_weights, dict) and isinstance(learned_weights.get("internal_weights"), dict):
        cal["internal_weights"] = learned_weights["internal_weights"]

    ifr, ifr_d = compute_issue_first_response(repo_url=repo_url, github_token=github_token, months=12, calibration=cal)
    prd, prd_d = compute_pr_review_discipline(repo_url=repo_url, github_token=github_token, months=12, calibration=cal)
    mp, mp_d = compute_maintainer_presence(repo_path=Path(repo_path), repo_url=repo_url, github_token=github_token, months=12, calibration=cal)

    learned_block = None
    if isinstance(learned_weights, dict):
        learned_block = learned_weights.get("weights") or learned_weights.get("block_weights") or learned_weights
    if isinstance(learned_block, dict) and learned_block:
        weights = {
            "issue_first_response": float(learned_block.get("issue_first_response", learned_block.get("IFR", 0.0))),
            "pr_review_discipline": float(learned_block.get("pr_review_discipline", learned_block.get("PRD", 0.0))),
            "maintainer_presence": float(learned_block.get("maintainer_presence", learned_block.get("MP", 0.0))),
        }
        weights_source = "learned_from_corpus"
    else:
        weights = _data_driven_weights_from_blocks(
            {"issue_first_response": ifr, "pr_review_discipline": prd, "maintainer_presence": mp}
        )
        weights_source = "repo_data_driven_fallback"

    final_0_1 = weighted_geometric(
        [
            (ifr, weights["issue_first_response"]),
            (prd, weights["pr_review_discipline"]),
            (mp, weights["maintainer_presence"]),
        ]
    )
    score = 100.0 * float(final_0_1) if final_0_1 is not None else None
    return {
        "MRD": score,
        "N_issue_first_response": ifr,
        "N_pr_review_discipline": prd,
        "N_maintainer_presence": mp,
        "weights": weights,
        "weights_source": weights_source,
        "calibration": calibration,
        "blocks": {
            "issue_first_response": ifr_d,
            "pr_review_discipline": prd_d,
            "maintainer_presence": mp_d,
        },
        "aggregation": "weighted_geometric",
        "missingness_note": "Unavailable data are kept as None; observable absence is scored 0; present evidence yields >0 normalized scores.",
    }


def compute_pvi(
    repo_path: Path,
    github_url: Optional[str] = None,
    github_token: Optional[str] = None,
    aggregation: str = "geometric",
    learned_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    aggregation:
    - "sum": strict weighted sum (architecture text)
    - "geometric": weighted geometric (recommended, less compensatory)
    """
    repo_path = Path(repo_path)

    commit_block, commit_details = compute_commit_activity(repo_path, months=12)
    release_block, release_details = compute_release_regularity(repo_path, github_url=github_url, github_token=github_token, months=24)
    issue_block, issue_details = compute_issue_activity(github_url=github_url, github_token=github_token, months=12)
    community_block, community_details = compute_community_growth(repo_path, github_url=github_url, github_token=github_token, months=12)

    blocks = {
        "commit_activity": commit_block,
        "release_regularity": release_block,
        "issue_activity": issue_block,
        "community_growth": community_block,
    }
    weights_source = "learned_from_corpus" if learned_weights is not None else "repo_data_driven_fallback"
    weights = learned_weights or _data_driven_weights_from_blocks(blocks)
    pairs = [
        (commit_block, float(weights.get("commit_activity", 0.0))),
        (release_block, float(weights.get("release_regularity", 0.0))),
        (issue_block, float(weights.get("issue_activity", 0.0))),
        (community_block, float(weights.get("community_growth", 0.0))),
    ]

    agg = (aggregation or "").strip().lower()
    if agg == "sum":
        final_0_1 = weighted_sum(pairs)
        agg_name = "weighted_sum"
    else:
        final_0_1 = weighted_geometric(pairs)
        agg_name = "weighted_geometric"

    score = 100.0 * float(final_0_1) if final_0_1 is not None else None

    return {
        "score": score,
        "band": band(score) if score is not None else None,
        "aggregation": agg_name,
        "details": {
            "PVI": score,
            "N_commit_activity": commit_block,
            "N_release_regularity": release_block,
            "N_issue_activity": issue_block,
            "N_community_growth": community_block,
            "weights": weights,
            "weights_source": weights_source,
            "blocks": {
                "commit_activity": commit_details,
                "release_regularity": release_details,
                "issue_activity": issue_details,
                "community_growth": community_details,
            },
            "missingness_note": (
                "PVI keeps missing signals as None and excludes them from block means and final aggregation. "
                "This distinguishes 'no evidence observed' from 'data unavailable'."
            ),
        },
    }


def _ph_headline_score(obj: Any) -> Optional[float]:
    """Extract 0--100 headline score from heterogeneous PH indicator payloads."""
    if not isinstance(obj, dict):
        return None
    s = obj.get("score")
    if isinstance(s, (int, float)):
        return float(s)
    for k in ("PVI", "MQS", "BUS_FACTOR", "MRD", "FCS"):
        v = obj.get(k)
        if isinstance(v, (int, float)):
            return float(v)
    return None


def learn_ph_weights(dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = ["PVI", "MQS", "FCS", "MRD", "BUS_FACTOR"]
    rows = []
    for row in dataset:
        src = row.get("PH", row)
        if isinstance(src, dict) and "scores" in src:
            src = src["scores"]
        out: Dict[str, Optional[float]] = {}
        for k in keys:
            val = _ph_headline_score(src.get(k) if isinstance(src, dict) else None)
            if isinstance(val, (int, float)):
                out[k] = float(val) / 100.0 if float(val) > 1.0 else float(val)
            else:
                out[k] = None
        rows.append(out)
    return {"weights": _norm_weights_from_rows(rows, keys), "n_repos": len(dataset)}


def compute_ph(
    repo_path: Path,
    repo_url: str,
    github_token: Optional[str] = None,
    aggregation: str = "geometric",
    learned_weights: Optional[Dict[str, float]] = None,
    fcs_calibration: Optional[Dict[str, Any]] = None,
    mrd_calibration: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Project Health family composite over PVI, MQS, FCS, MRD, BUS_FACTOR."""
    pvi = compute_pvi(repo_path, github_url=repo_url, github_token=github_token, aggregation=aggregation, learned_weights=None)
    mqs = compute_mqs(repo_url=repo_url, github_token=github_token, aggregation=aggregation, learned_weights=None, local_repo=repo_path)
    fcs = compute_fcs(repo_path, repo_url, github_token, learned_weights=None, calibration=fcs_calibration, aggregation=aggregation)
    mrd = compute_mrd(repo_path, repo_url, github_token, learned_weights=None, calibration=mrd_calibration)
    bus = compute_bus_factor(repo_path, months=24)

    def _blk(obj: Any) -> Optional[float]:
        s = _ph_headline_score(obj)
        return float(s) / 100.0 if isinstance(s, (int, float)) else None

    blocks = {
        "PVI": _blk(pvi),
        "MQS": _blk(mqs),
        "FCS": _blk(fcs),
        "MRD": _blk(mrd),
        "BUS_FACTOR": _blk(bus),
    }
    family_weights = None
    if isinstance(learned_weights, dict):
        family_weights = learned_weights.get("PH") or learned_weights.get("ph")
        if family_weights is None and all(k in learned_weights for k in blocks):
            family_weights = learned_weights

    weights_source = "learned_from_corpus" if family_weights is not None else "repo_data_driven_fallback"
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

    return {
        "repository_path": str(repo_path),
        "github_url": repo_url,
        "score": score,
        "band": band(score) if score is not None else None,
        "aggregation": agg_name,
        "details": {
            "PH": score,
            "PVI": _ph_headline_score(pvi),
            "MQS": _ph_headline_score(mqs),
            "FCS": _ph_headline_score(fcs),
            "MRD": _ph_headline_score(mrd),
            "BUS_FACTOR": _ph_headline_score(bus),
            "weights": weights,
            "weights_source": weights_source,
            "scores": {"PVI": pvi, "MQS": mqs, "FCS": fcs, "MRD": mrd, "BUS_FACTOR": bus},
            "missingness_note": "Unavailable indicators remain None and are excluded with weight renormalization.",
            "method_note": "PH aggregates project-health headline indicators (PVI, MQS, FCS, MRD, BUS_FACTOR).",
        },
    }


# =========================================================
# CLI
# =========================================================

def main() -> None:
    ap = argparse.ArgumentParser(description="Compute RECOPS PVI / MQS / FCS / MRD / BUS_FACTOR / PH from GitHub repositories.")
    ap.add_argument("--github-url", required=False, default=None, help="GitHub repository URL.")
    ap.add_argument("--repo-path", default=None, help="Optional path to an existing local clone; if omitted, the repo is fetched automatically.")
    ap.add_argument("--github-token", default=os.environ.get("GITHUB_TOKEN"), help="GitHub token (or set env GITHUB_TOKEN).")
    ap.add_argument("--aggregation", default="geometric", choices=["geometric", "sum"], help="Final aggregation method.")
    ap.add_argument("--indicator", default="all", choices=["all", "pvi", "mqs", "fcs", "mrd", "bus_factor", "ph"], help="Which indicator(s) to compute.")
    ap.add_argument("--pvi-weights-json", default=None, help="Optional JSON file with learned PVI weights.")
    ap.add_argument("--learn-pvi-dataset", default=None, help="Optional JSON dataset to learn PVI weights.")
    ap.add_argument("--mqs-weights-json", default=None, help="Optional JSON file with learned MQS weights.")
    ap.add_argument("--learn-mqs-dataset", default=None, help="Optional JSON dataset to learn MQS weights.")
    ap.add_argument("--fcs-weights-json", default=None, help="Optional JSON file with learned FCS block weights.")
    ap.add_argument("--fcs-calibration-json", default=None, help="Optional JSON file with learned FCS calibration.")
    ap.add_argument("--learn-fcs-dataset", default=None, help="Optional JSON dataset to learn FCS calibration/weights.")
    ap.add_argument("--mrd-weights-json", default=None, help="Optional JSON file with learned MRD weights.")
    ap.add_argument("--mrd-calibration-json", default=None, help="Optional JSON file with learned MRD calibration.")
    ap.add_argument("--learn-mrd-dataset", default=None, help="Optional JSON dataset to learn MRD calibration/weights.")
    ap.add_argument("--output-json", default=None, help="Write JSON output to file.")
    args = ap.parse_args()

    if args.indicator in {"all", "pvi", "mqs", "fcs", "mrd", "ph"} and not args.github_url:
        raise SystemExit("--github-url is required for scoring.")
    if args.indicator == "bus_factor" and (not args.github_url and not args.repo_path):
        raise SystemExit("--github-url or --repo-path is required for bus_factor.")

    if args.repo_path:
        repo_path = Path(args.repo_path)
    elif args.github_url:
        repo_path = clone_or_download(args.github_url)
    else:
        repo_path = None

    out: Dict[str, Any] = {
        "repository_path": str(Path(repo_path).resolve()) if repo_path is not None else None,
        "github_url": args.github_url,
    }

    learned_weights = None
    pvi_weights = None
    fcs_weights = None
    fcs_calibration = None
    mrd_weights = None
    mrd_calibration = None
    if args.learn_pvi_dataset:
        with open(args.learn_pvi_dataset, "r", encoding="utf-8") as f:
            pds = json.load(f)
        if not isinstance(pds, list):
            raise SystemExit("--learn-pvi-dataset must be a JSON list.")
        pvi_learned = learn_pvi_weights(pds)
        pvi_weights = pvi_learned.get("weights")
        out["PVI_learned_weights"] = pvi_learned
    if args.learn_mqs_dataset:
        with open(args.learn_mqs_dataset, "r", encoding="utf-8") as f:
            ds = json.load(f)
        if not isinstance(ds, list):
            raise SystemExit("--learn-mqs-dataset must be a JSON list.")
        if ds and isinstance(ds[0], dict) and ("IM" in ds[0] or "N_issue_management" in ds[0]):
            learned = learn_mqs_weights(ds)
        else:
            rows: List[Dict[str, Any]] = []
            for entry in ds:
                if isinstance(entry, str):
                    u = entry
                elif isinstance(entry, dict):
                    u = entry.get("github_url") or entry.get("url")
                else:
                    u = None
                if not u:
                    continue
                blocks, _, _ = _compute_mqs_blocks(repo_url=u, github_token=args.github_token)
                rows.append(
                    {
                        "N_issue_management": blocks["issue_management"],
                        "N_update_consistency": blocks["update_consistency"],
                        "N_security_responsiveness": blocks["security_responsiveness"],
                    }
                )
            learned = learn_mqs_weights(rows)
        learned_weights = learned.get("weights")
        out["MQS_learned_weights"] = learned

    if args.mqs_weights_json:
        with open(args.mqs_weights_json, "r", encoding="utf-8") as f:
            wj = json.load(f)
        if isinstance(wj, dict) and isinstance(wj.get("weights"), dict):
            learned_weights = wj["weights"]
        elif isinstance(wj, dict):
            learned_weights = wj

    if args.pvi_weights_json:
        with open(args.pvi_weights_json, "r", encoding="utf-8") as f:
            pw = json.load(f)
        if isinstance(pw, dict) and isinstance(pw.get("weights"), dict):
            pvi_weights = pw["weights"]
        elif isinstance(pw, dict):
            pvi_weights = pw

    if args.learn_fcs_dataset:
        with open(args.learn_fcs_dataset, "r", encoding="utf-8") as f:
            fds = json.load(f)
        if not isinstance(fds, list):
            raise SystemExit("--learn-fcs-dataset must be a JSON list.")
        fcs_calibration = learn_fcs_calibration(fds)
        fcs_weights_learned = learn_fcs_weights(fds)
        fcs_weights = fcs_weights_learned.get("weights")
        out["FCS_learned_calibration"] = fcs_calibration
        out["FCS_learned_weights"] = fcs_weights_learned

    if args.fcs_calibration_json:
        with open(args.fcs_calibration_json, "r", encoding="utf-8") as f:
            cj = json.load(f)
        if isinstance(cj, dict):
            fcs_calibration = cj

    if args.fcs_weights_json:
        with open(args.fcs_weights_json, "r", encoding="utf-8") as f:
            fw = json.load(f)
        if isinstance(fw, dict) and isinstance(fw.get("weights"), dict):
            fcs_weights = fw["weights"]
        elif isinstance(fw, dict):
            fcs_weights = fw

    if args.learn_mrd_dataset:
        with open(args.learn_mrd_dataset, "r", encoding="utf-8") as f:
            mds = json.load(f)
        if not isinstance(mds, list):
            raise SystemExit("--learn-mrd-dataset must be a JSON list.")
        mrd_calibration = learn_mrd_calibration(mds)
        mrd_weights = learn_mrd_weights(mds)
        out["MRD_learned_calibration"] = mrd_calibration
        out["MRD_learned_weights"] = mrd_weights

    if args.mrd_calibration_json:
        with open(args.mrd_calibration_json, "r", encoding="utf-8") as f:
            mcj = json.load(f)
        if isinstance(mcj, dict):
            mrd_calibration = mcj

    if args.mrd_weights_json:
        with open(args.mrd_weights_json, "r", encoding="utf-8") as f:
            mw = json.load(f)
        if isinstance(mw, dict):
            mrd_weights = mw

    ind = args.indicator.lower()
    if ind == "ph":
        if repo_path is None or not args.github_url:
            raise SystemExit("--github-url and --repo-path are required for ph.")
        out = compute_ph(
            repo_path=Path(repo_path),
            repo_url=args.github_url,
            github_token=args.github_token,
            aggregation=args.aggregation,
            fcs_calibration=fcs_calibration,
            mrd_calibration=mrd_calibration,
        )
    elif ind == "pvi" and repo_path is not None:
        out = compute_pvi(
            repo_path=Path(repo_path),
            github_url=args.github_url,
            github_token=args.github_token,
            aggregation=args.aggregation,
            learned_weights=pvi_weights,
        )
    elif ind == "mqs" and args.github_url:
        out = compute_mqs(
            repo_url=args.github_url,
            github_token=args.github_token,
            aggregation=args.aggregation,
            learned_weights=learned_weights,
            local_repo=Path(repo_path) if repo_path is not None else None,
        )
    elif ind == "fcs" and args.github_url and repo_path is not None:
        out = compute_fcs(
            repo_path=Path(repo_path),
            repo_url=args.github_url,
            github_token=args.github_token,
            learned_weights=fcs_weights,
            calibration=fcs_calibration,
            aggregation=args.aggregation,
        )
    elif ind == "bus_factor" and repo_path is not None:
        out = compute_bus_factor(repo_path=Path(repo_path), months=24)
    elif ind == "mrd" and args.github_url and repo_path is not None:
        out = compute_mrd(
            repo_path=Path(repo_path),
            repo_url=args.github_url,
            github_token=args.github_token,
            learned_weights=mrd_weights,
            calibration=mrd_calibration,
        )
    elif ind == "all":
        if repo_path is None or not args.github_url:
            raise SystemExit("--github-url and --repo-path are required for indicator all.")
        rp = Path(repo_path)
        out = {
            "repository_path": str(rp.resolve()),
            "github_url": args.github_url,
            "scores": {
                "PVI": compute_pvi(rp, github_url=args.github_url, github_token=args.github_token, aggregation=args.aggregation, learned_weights=None),
                "MQS": compute_mqs(repo_url=args.github_url, github_token=args.github_token, aggregation=args.aggregation, learned_weights=None, local_repo=rp),
                "FCS": compute_fcs(rp, args.github_url, args.github_token, learned_weights=None, calibration=fcs_calibration, aggregation=args.aggregation),
                "MRD": compute_mrd(rp, args.github_url, args.github_token, learned_weights=None, calibration=mrd_calibration),
                "BUS_FACTOR": compute_bus_factor(repo_path=rp, months=24),
                "PH": compute_ph(rp, args.github_url, args.github_token, aggregation=args.aggregation, fcs_calibration=fcs_calibration, mrd_calibration=mrd_calibration),
            },
            "note": "With --indicator all, per-indicator weight JSON files are ignored to avoid applying one weight file to multiple formulas.",
        }
    else:
        raise SystemExit(f"Cannot compute indicator={args.indicator} with the provided paths.")

    print(json.dumps(out, indent=2))
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()

