#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RECOPS Repository Scorer v2 — Energy-Domain Fitness (EDF)

Computes the four EDF indicators from the RECOPS v2 architecture:
- ESCI — Energy System Capability Index
- CPRI — Cyber-Physical Readiness Index
- ENSC — ENS & Reliability Capability
- EDVS — Energy-Domain Validation Score

Design choices
--------------
- No fixed formula weights such as 0.35/0.25/0.20/0.20 inside EDF.
- Default aggregation is weighted_geometric with data-driven weights.
- Block weights can be learned from a corpus; otherwise a per-repository
  dispersion fallback is used.
- Missing data are kept as None and excluded with weight renormalization.
- Observable absence is scored as 0.0.
- EDF is intentionally non-redundant with DI: it does not reward generic
  containerization, generic API presence, or generic deployment maturity. It
  rewards power-system capability, cyber-physical suitability, reliability
  modelling, and energy-domain validation evidence.

Recommended installation:
    pip install requests pyyaml

Example usage:
    python recops_repo_scores_EDF.py https://github.com/PyPSA/PyPSA --score all
    python recops_repo_scores_EDF.py https://github.com/PyPSA/PyPSA --score esci
    python recops_repo_scores_EDF.py --local ./PyPSA --score edf --out edf_results.json
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
    """Weighted geometric mean with explicit missingness and weight renormalization."""
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
    # Adaptive floor: keeps non-compensatory behavior but avoids pathological
    # near-zero collapse when one block is exactly zero.
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
    """Smooth evidence saturation: 1 - exp(-x/scale)."""
    if x is None:
        return None
    try:
        xx = max(0.0, float(x))
        ss = max(1e-9, float(scale))
    except Exception:
        return None
    return clamp01(1.0 - math.exp(-xx / ss))


def exp_recency(days: Optional[float], half_life_days: float = 365.0) -> Optional[float]:
    if days is None:
        return None
    try:
        d = max(0.0, float(days))
        h = max(1e-9, float(half_life_days))
    except Exception:
        return None
    return clamp01(math.exp(-math.log(2.0) * d / h))


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
    """Repository-relative logarithmic normalization: log(1+h)/log(1+h_ref)."""
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


def percentile_score(value: Optional[float], corpus_values: Optional[List[float]]) -> Optional[float]:
    """Optional corpus percentile score; returns None when no calibration distribution is supplied."""
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
# Repository acquisition and GitHub helpers
# =========================================================

def run(cmd: List[str], cwd: Optional[Path] = None, timeout_s: int = 120) -> Tuple[int, str, str]:
    try:
        r = subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True, timeout=timeout_s)
        return r.returncode, r.stdout, r.stderr
    except Exception as e:
        return 1, "", str(e)


def parse_github_owner_repo(url: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if not url:
        return None, None
    m = re.search(r"github\.com/([^/]+)/([^/]+)", url)
    if not m:
        return None, None
    return m.group(1), m.group(2).replace(".git", "")


def clone_or_download(github_url: str) -> Path:
    temp = Path(tempfile.mkdtemp(prefix="recops_edf_"))
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
        url = f"https://github.com/{owner}/{repo}/archive/refs/heads/{branch}.zip"
        r = requests.get(url, timeout=60)
        if r.status_code == 200:
            zpath = temp / "repo.zip"
            zpath.write_bytes(r.content)
            shutil.unpack_archive(str(zpath), str(temp))
            dirs = [p for p in temp.iterdir() if p.is_dir() and p.name != "repo"]
            if dirs:
                return dirs[0]
    raise RuntimeError("Cannot fetch repository")


def gh_api_get(url: str, token: Optional[str] = None, accept: Optional[str] = None, timeout_s: int = 30) -> Optional[Any]:
    if requests is None:
        return None
    headers = {"User-Agent": "recops-edf/1.0"}
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
# File scanning utilities
# =========================================================

IGNORE_DIRS = {
    ".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "node_modules", "dist", "build", ".venv", "venv", ".idea", ".vscode",
}

TEXT_SUFFIXES = {
    ".md", ".rst", ".txt", ".toml", ".json", ".yaml", ".yml", ".ini", ".cfg",
    ".py", ".js", ".ts", ".java", ".jl", ".c", ".cpp", ".h", ".hpp", ".m",
    ".ipynb", ".bib", ".csv", ".xml", ".rdf", ".ttl",
}

SOURCE_SUFFIXES = {
    ".py", ".js", ".ts", ".java", ".jl", ".c", ".cpp", ".h", ".hpp", ".m", ".r",
    ".gms", ".mod", ".dat", ".inc", ".lp", ".mps",
}

DOC_DIR_NAMES = {
    "docs", "doc", "examples", "example", "notebooks", "tutorials", "tutorial", "cases", "case_studies",
    "benchmarks", "benchmark", "validation", "data", "datasets", "test", "tests", "models", "scenarios",
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


def is_doc_like_path(root: Path, path: Path) -> bool:
    try:
        rel = path.relative_to(root)
    except Exception:
        return False
    parts = [x.lower() for x in rel.parts]
    name = path.name.lower()
    if len(rel.parts) == 1 and (name.startswith("readme") or name in {"citation.cff", "codemeta.json", "pyproject.toml", "package.json"}):
        return True
    return any(part in DOC_DIR_NAMES for part in parts[:-1])


def candidate_texts(root: Path, include_source: bool = False) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for p in iter_files(root):
        if p.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if include_source or is_doc_like_path(root, p):
            txt = _safe_read_text(p)
            if txt:
                try:
                    out[str(p.relative_to(root))] = txt
                except Exception:
                    out[str(p)] = txt
    return out


def source_texts(root: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for p in iter_files(root):
        if p.suffix.lower() in SOURCE_SUFFIXES:
            txt = _safe_read_text(p)
            if txt:
                try:
                    out[str(p.relative_to(root))] = txt
                except Exception:
                    out[str(p)] = txt
    return out


MIN_SOURCE_FILES_FOR_IMPLEMENTATION_SIGNAL = 8


def count_regex_hits_in_texts(texts: Dict[str, str], patterns: Iterable[str]) -> int:
    rx = [re.compile(p, flags=re.I | re.M) for p in patterns]
    hits = 0
    for txt in texts.values():
        for r in rx:
            hits += len(r.findall(txt or ""))
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


def files_matching(root: Path, path_patterns: Iterable[str], suffixes: Optional[set] = None) -> List[Path]:
    rx = [re.compile(p, flags=re.I) for p in path_patterns]
    out: List[Path] = []
    for p in iter_files(root):
        if suffixes and p.suffix.lower() not in suffixes:
            continue
        try:
            rel = str(p.relative_to(root)).replace("\\", "/")
        except Exception:
            rel = str(p)
        if any(r.search(rel) for r in rx):
            out.append(p)
    return out


def repo_hit_ref(*counts: Optional[int]) -> float:
    return max([1.0] + [float(c) for c in counts if c is not None])


def family_counts(texts: Dict[str, str], families: Dict[str, List[str]]) -> Dict[str, int]:
    return {name: count_regex_hits_in_texts(texts, pats) for name, pats in families.items()}


def family_diversity_score(counts: Dict[str, int]) -> Optional[float]:
    return ratio01(sum(1 for v in counts.values() if v > 0), len(counts))


def family_entropy_score(counts: Dict[str, int]) -> Optional[float]:
    return entropy_normalized([v for v in counts.values() if v > 0])


# =========================================================
# Git helpers
# =========================================================

def is_git_repo(repo: Path) -> bool:
    code, out, _ = run(["git", "rev-parse", "--is-inside-work-tree"], cwd=repo)
    return code == 0 and out.strip().lower() == "true"


def file_age_days(repo: Path, path: Path) -> Optional[float]:
    now = dt.datetime.utcnow()
    try:
        rel = str(path.relative_to(repo))
    except Exception:
        rel = str(path)
    if is_git_repo(repo):
        code, out, _ = run(["git", "log", "-1", "--format=%ct", "--", rel], cwd=repo, timeout_s=60)
        if code == 0 and out.strip().isdigit():
            ts = int(out.strip())
            return (now - dt.datetime.utcfromtimestamp(ts)).total_seconds() / 86400.0
    try:
        return (now - dt.datetime.utcfromtimestamp(path.stat().st_mtime)).total_seconds() / 86400.0
    except Exception:
        return None


def median_recency_score(repo: Path, paths: List[Path], half_life_days: float = 730.0) -> Optional[float]:
    days = [file_age_days(repo, p) for p in paths]
    vals = sorted(d for d in days if d is not None)
    if not vals:
        return None
    median_days = vals[len(vals) // 2]
    return exp_recency(median_days, half_life_days=half_life_days)


# =========================================================
# Data-driven weighting
# =========================================================

def data_driven_weights_from_blocks(blocks: Dict[str, Optional[float]]) -> Dict[str, float]:
    """
    Repository-level fallback. Observed blocks farther from the repository-level
    mean receive more weight because they discriminate the project more. If all
    observed blocks are identical, equal weights are used across observed blocks.
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
    """Corpus weights: variance × coverage × uniqueness."""
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
# Energy-domain pattern families
# =========================================================

MODEL_TYPE_FAMILIES = {
    "planning": [r"\bplanning\b", r"capacity expansion", r"investment", r"expansion planning", r"generation expansion"],
    "optimization": [r"\boptimization\b", r"\boptimisation\b", r"optimal power flow", r"\bOPF\b", r"linear programming", r"mixed[- ]integer", r"unit commitment"],
    "simulation": [r"\bsimulation\b", r"dynamic simulation", r"time[- ]series simulation", r"power flow simulation", r"load flow"],
    "control": [r"\bcontrol\b", r"controller", r"MPC\b", r"model predictive", r"dispatch control", r"frequency control", r"voltage control"],
    "forecasting": [r"\bforecast", r"prediction", r"load forecast", r"renewable forecast", r"weather forecast"],
}

DER_FAMILIES = {
    "solar": [r"\bsolar\b", r"\bphotovoltaic\b", r"\bPV\b"],
    "wind": [r"\bwind\b", r"wind farm", r"turbine"],
    "storage": [r"\bstorage\b", r"battery", r"BESS\b", r"hydrogen storage", r"thermal storage"],
    "ev": [r"electric vehicle", r"\bEV\b", r"charging station", r"vehicle[- ]to[- ]grid", r"V2G"],
    "demand_response": [r"demand response", r"demand[- ]side", r"load shifting", r"flexible demand"],
    "distributed_generation": [r"distributed generation", r"DER\b", r"prosumer", r"microgrid"],
    "heat_sector": [r"heat pump", r"district heating", r"CHP\b", r"combined heat"],
}

GRID_ANALYSIS_FAMILIES = {
    "power_flow": [r"power flow", r"load flow", r"AC power flow", r"DC power flow", r"Newton[- ]Raphson"],
    "opf": [r"optimal power flow", r"\bOPF\b", r"security[- ]constrained OPF", r"SCOPF"],
    "contingency": [r"contingency", r"N-1", r"outage analysis", r"security constrained"],
    "state_estimation": [r"state estimation", r"WLS\b", r"phasor", r"PMU\b"],
    "dispatch": [r"economic dispatch", r"dispatch", r"unit commitment", r"UC\b"],
    "short_circuit": [r"short circuit", r"fault analysis", r"protection study"],
    "stability": [r"stability", r"small[- ]signal", r"transient stability", r"frequency stability"],
}

VOLTAGE_FAMILIES = {
    "transmission": [r"transmission", r"HV\b", r"EHV\b", r"high voltage", r"\b400\s?kV\b", r"\b220\s?kV\b"],
    "distribution": [r"distribution", r"MV\b", r"LV\b", r"medium voltage", r"low voltage", r"feeder"],
    "microgrid": [r"microgrid", r"islanded", r"off[- ]grid"],
    "substation": [r"substation", r"busbar", r"breaker", r"transformer"],
    "multi_voltage": [r"multi[- ]voltage", r"voltage level", r"voltage levels", r"grid levels"],
}

HARDWARE_PROTOCOL_FAMILIES = {
    "scada": [r"SCADA", r"RTU\b", r"telemetry", r"supervisory control"],
    "iec61850": [r"IEC\s*61850", r"GOOSE\b", r"MMS\b", r"sampled values"],
    "dnp3": [r"DNP3", r"IEEE\s*1815"],
    "modbus": [r"Modbus", r"Modbus[- ]TCP"],
    "opcua": [r"OPC[- ]?UA", r"OPC UA"],
    "mqtt": [r"MQTT", r"message broker", r"pub/sub"],
    "pmu": [r"PMU\b", r"phasor measurement", r"synchrophasor", r"IEEE\s*C37\.118"],
    "inverter": [r"inverter", r"smart inverter", r"DER controller"],
    "hil": [r"hardware[- ]in[- ]the[- ]loop", r"HIL\b", r"real[- ]time simulator", r"OPAL[- ]RT", r"RTDS\b", r"Typhoon HIL"],
}

REAL_TIME_FAMILIES = {
    "hard_realtime": [r"hard real[- ]time", r"deterministic latency", r"deadline", r"bounded latency"],
    "soft_realtime": [r"soft real[- ]time", r"near real[- ]time", r"real[- ]time"],
    "timesteps": [r"time step", r"sampling rate", r"control cycle", r"\bHz\b", r"millisecond", r"latency"],
    "streaming": [r"streaming", r"online", r"event[- ]driven", r"message queue", r"Kafka", r"MQTT"],
    "cosimulation": [r"co[- ]simulation", r"FMI\b", r"FMU\b", r"HELICS", r"mosaik"],
}

CONTROL_ARCH_FAMILIES = {
    "centralized": [r"centralized control", r"centralised control", r"central controller"],
    "distributed": [r"distributed control", r"decentralized", r"decentralised", r"multi[- ]agent", r"agent[- ]based"],
    "hierarchical": [r"hierarchical control", r"primary control", r"secondary control", r"tertiary control"],
    "market_control": [r"market[- ]based control", r"transactive energy", r"auction", r"price signal"],
    "protection_control": [r"protection", r"relay", r"fault ride[- ]through", r"droop control"],
}

ENS_FAMILIES = {
    "ens": [r"\bENS\b", r"energy not supplied", r"energy not served", r"unserved energy"],
    "eens": [r"\bEENS\b", r"expected energy not supplied", r"expected energy not served"],
    "load_shedding": [r"load shedding", r"load curtailment", r"curtailed load", r"served load", r"demand not served"],
    "outage_cost": [r"outage cost", r"value of lost load", r"\bVOLL\b", r"interruption cost"],
}

RELIABILITY_FAMILIES = {
    "lole": [r"\bLOLE\b", r"loss of load expectation"],
    "lolp": [r"\bLOLP\b", r"loss of load probability"],
    "lolh": [r"\bLOLH\b", r"loss of load hours"],
    "saidi": [r"\bSAIDI\b", r"system average interruption duration"],
    "saifi": [r"\bSAIFI\b", r"system average interruption frequency"],
    "adequacy": [r"resource adequacy", r"generation adequacy", r"capacity adequacy", r"reliability assessment"],
    "availability": [r"availability", r"forced outage", r"outage rate", r"failure rate", r"MTTF", r"MTTR"],
}

STANDARD_CASE_FAMILIES = {
    "ieee_bus": [r"IEEE\s*(?:\d+)[- ]bus", r"IEEE\s*bus", r"case\d+", r"case14", r"case30", r"case57", r"case118", r"case300"],
    "matpower": [r"MATPOWER", r"pglib", r"Power Grid Lib", r"PEGASE", r"RTS[- ]GMLC"],
    "cigre": [r"CIGRE", r"European LV test feeder", r"benchmark grid"],
    "pandapower": [r"pandapower.*network", r"simple_four_bus", r"mv_oberrhein", r"create_cigre_network"],
    "opf_benchmark": [r"OPF benchmark", r"optimal power flow benchmark", r"benchmark case"],
}

SCENARIO_FAMILIES = {
    "der_scenarios": [r"DER scenario", r"renewable scenario", r"PV scenario", r"wind scenario", r"EV scenario"],
    "contingency_scenarios": [r"contingency scenario", r"N-1 scenario", r"outage scenario", r"fault scenario"],
    "planning_scenarios": [r"planning scenario", r"expansion scenario", r"decarboni[sz]ation scenario", r"policy scenario"],
    "operational_scenarios": [r"operational scenario", r"dispatch scenario", r"market scenario", r"real[- ]time scenario"],
    "stress_scenarios": [r"stress test", r"extreme event", r"resilience scenario", r"peak load scenario"],
}

OPERATIONAL_DATA_FAMILIES = {
    "load_timeseries": [r"load profile", r"load time series", r"demand profile", r"demand time series"],
    "renewable_timeseries": [r"renewable time series", r"wind time series", r"solar time series", r"capacity factor"],
    "weather": [r"weather data", r"ERA5", r"MERRA", r"meteorological", r"reanalysis"],
    "scada_measurements": [r"SCADA data", r"measurement data", r"telemetry", r"sensor data", r"PMU data"],
    "market_data": [r"market data", r"price time series", r"electricity price", r"ENTSO[- ]E", r"EIA data"],
    "field_data": [r"field data", r"real[- ]world data", r"historical data", r"operational dataset"],
}

STANDARDS_VALIDATION_FAMILIES = {
    "iec": [r"IEC\s*61850", r"IEC\s*61970", r"IEC\s*61968", r"IEC\s*62325", r"Common Information Model", r"\bCIM\b"],
    "ieee": [r"IEEE\s*\d+", r"IEEE\s*C37\.118", r"IEEE\s*1547", r"IEEE\s*2030"],
    "entsoe": [r"ENTSO[- ]E", r"CGMES", r"network code", r"grid code"],
    "validation_method": [r"validated against", r"compared against", r"benchmark against", r"reference result", r"accepted benchmark"],
    "domain_compliance": [r"standard compliance", r"standards compliance", r"grid code compliance", r"compliant with"],
}


# =========================================================
# EDF.1 ESCI — Energy System Capability Index
# =========================================================

def compute_model_type_coverage(repo_path: Path) -> Tuple[Optional[float], Dict[str, Any]]:
    texts = candidate_texts(repo_path, include_source=True)
    counts = family_counts(texts, MODEL_TYPE_FAMILIES)
    path_hits = count_path_hits(repo_path, [r"planning", r"optimization", r"optimisation", r"simulation", r"control", r"forecast"])
    ref = repo_hit_ref(*counts.values(), path_hits)
    mt_family_breadth = family_diversity_score(counts)
    mt_family_entropy = family_entropy_score(counts)
    mt_evidence_density = _log_repo_signal(sum(counts.values()), ref)
    mt_path_evidence = _log_repo_signal(path_hits, ref)
    block = mean([mt_family_breadth, mt_family_entropy, mt_evidence_density, mt_path_evidence])
    return block, {"available": True, "block": block, "sub": {
        "MT_family_breadth": mt_family_breadth,
        "MT_family_entropy": mt_family_entropy,
        "MT_evidence_density": mt_evidence_density,
        "MT_path_evidence": mt_path_evidence,
    }, "raw": {"family_counts": counts, "path_hits": path_hits, "hit_ref": ref}}


def compute_der_support(repo_path: Path) -> Tuple[Optional[float], Dict[str, Any]]:
    texts = candidate_texts(repo_path, include_source=True)
    counts = family_counts(texts, DER_FAMILIES)
    path_hits = count_path_hits(repo_path, [r"solar|photovoltaic|pv", r"wind", r"storage|battery", r"ev|charging", r"demand[-_ ]response", r"microgrid|der"])
    ref = repo_hit_ref(*counts.values(), path_hits)
    der_family_breadth = family_diversity_score(counts)
    der_family_entropy = family_entropy_score(counts)
    der_evidence_density = _log_repo_signal(sum(counts.values()), ref)
    der_path_evidence = _log_repo_signal(path_hits, ref)
    block = mean([der_family_breadth, der_family_entropy, der_evidence_density, der_path_evidence])
    return block, {"available": True, "block": block, "sub": {
        "DER_family_breadth": der_family_breadth,
        "DER_family_entropy": der_family_entropy,
        "DER_evidence_density": der_evidence_density,
        "DER_path_evidence": der_path_evidence,
    }, "raw": {"family_counts": counts, "path_hits": path_hits, "hit_ref": ref}}


def compute_grid_analysis_capability(repo_path: Path) -> Tuple[Optional[float], Dict[str, Any]]:
    texts = candidate_texts(repo_path, include_source=True)
    counts = family_counts(texts, GRID_ANALYSIS_FAMILIES)
    path_hits = count_path_hits(repo_path, [r"power[-_ ]?flow|load[-_ ]?flow|opf|contingency|dispatch|state[-_ ]?estimation|stability|fault"])
    ref = repo_hit_ref(*counts.values(), path_hits)
    ga_family_breadth = family_diversity_score(counts)
    ga_family_entropy = family_entropy_score(counts)
    ga_evidence_density = _log_repo_signal(sum(counts.values()), ref)
    ga_path_evidence = _log_repo_signal(path_hits, ref)
    block = mean([ga_family_breadth, ga_family_entropy, ga_evidence_density, ga_path_evidence])
    return block, {"available": True, "block": block, "sub": {
        "GA_family_breadth": ga_family_breadth,
        "GA_family_entropy": ga_family_entropy,
        "GA_evidence_density": ga_evidence_density,
        "GA_path_evidence": ga_path_evidence,
    }, "raw": {"family_counts": counts, "path_hits": path_hits, "hit_ref": ref}}


def compute_voltage_level_coverage(repo_path: Path) -> Tuple[Optional[float], Dict[str, Any]]:
    texts = candidate_texts(repo_path, include_source=True)
    counts = family_counts(texts, VOLTAGE_FAMILIES)
    path_hits = count_path_hits(repo_path, [r"transmission|distribution|microgrid|voltage|feeder|substation|bus"])
    ref = repo_hit_ref(*counts.values(), path_hits)
    vl_family_breadth = family_diversity_score(counts)
    vl_family_entropy = family_entropy_score(counts)
    vl_evidence_density = _log_repo_signal(sum(counts.values()), ref)
    vl_path_evidence = _log_repo_signal(path_hits, ref)
    block = mean([vl_family_breadth, vl_family_entropy, vl_evidence_density, vl_path_evidence])
    return block, {"available": True, "block": block, "sub": {
        "VL_family_breadth": vl_family_breadth,
        "VL_family_entropy": vl_family_entropy,
        "VL_evidence_density": vl_evidence_density,
        "VL_path_evidence": vl_path_evidence,
    }, "raw": {"family_counts": counts, "path_hits": path_hits, "hit_ref": ref}}


def compute_esci(repo_path: Path, repo_url: Optional[str] = None, github_token: Optional[str] = None,
                 learned_weights: Optional[Dict[str, float]] = None, calibration: Optional[Dict[str, Any]] = None,
                 aggregation: str = "geometric") -> Dict[str, Any]:
    n_mt, mt_d = compute_model_type_coverage(repo_path)
    n_der, der_d = compute_der_support(repo_path)
    n_ga, ga_d = compute_grid_analysis_capability(repo_path)
    n_vl, vl_d = compute_voltage_level_coverage(repo_path)
    blocks = {"model_type_coverage": n_mt, "der_support": n_der, "grid_analysis_capability": n_ga, "voltage_level_coverage": n_vl}
    score, weights, weights_source, agg_name = final_score_from_blocks(blocks, learned_weights, aggregation)
    return {"repository_path": str(repo_path), "repository_url": repo_url, "score": score, "band": band(score), "aggregation": agg_name,
            "details": {"ESCI": score, "N_model_type_coverage": n_mt, "N_der_support": n_der,
                        "N_grid_analysis_capability": n_ga, "N_voltage_level_coverage": n_vl,
                        "weights": weights, "weights_source": weights_source,
                        "blocks": {"model_type_coverage": mt_d, "der_support": der_d, "grid_analysis_capability": ga_d, "voltage_level_coverage": vl_d},
                        "method_note": "ESCI measures energy-system modelling breadth, not generic software quality or adoption."}}


# =========================================================
# EDF.2 CPRI — Cyber-Physical Readiness Index
# =========================================================

def compute_hardware_interface_score(repo_path: Path) -> Tuple[Optional[float], Dict[str, Any]]:
    texts = candidate_texts(repo_path, include_source=True)
    counts = family_counts(texts, HARDWARE_PROTOCOL_FAMILIES)
    path_hits = count_path_hits(repo_path, [r"hardware|scada|pmu|rtu|modbus|opc|iec61850|hil|inverter|telemetry"])
    interface_files = files_matching(repo_path, [r"hardware|scada|pmu|rtu|modbus|opc|iec61850|hil|inverter|telemetry"], suffixes=TEXT_SUFFIXES)
    ref = repo_hit_ref(*counts.values(), path_hits, len(interface_files))
    hi_protocol_breadth = family_diversity_score(counts)
    hi_protocol_entropy = family_entropy_score(counts)
    hi_docs_density = _log_repo_signal(sum(counts.values()), ref)
    hi_path_evidence = _log_repo_signal(path_hits + len(interface_files), ref)
    block = mean([hi_protocol_breadth, hi_protocol_entropy, hi_docs_density, hi_path_evidence])
    return block, {"available": True, "block": block, "sub": {
        "HI_protocol_breadth": hi_protocol_breadth,
        "HI_protocol_entropy": hi_protocol_entropy,
        "HI_docs_density": hi_docs_density,
        "HI_path_evidence": hi_path_evidence,
    }, "raw": {"family_counts": counts, "path_hits": path_hits, "interface_files_count": len(interface_files), "interface_files_sample": [str(p.relative_to(repo_path)) for p in interface_files[:30]], "hit_ref": ref}}


def compute_real_time_score(repo_path: Path) -> Tuple[Optional[float], Dict[str, Any]]:
    texts = candidate_texts(repo_path, include_source=True)
    counts = family_counts(texts, REAL_TIME_FAMILIES)
    path_hits = count_path_hits(repo_path, [r"real[-_ ]?time|realtime|stream|cosim|helics|fmi|latency|scheduler"])
    source = source_texts(repo_path)
    timer_hits = count_regex_hits_in_texts(source, [r"time\.sleep", r"asyncio", r"threading", r"scheduler", r"perf_counter", r"setInterval", r"setTimeout"])
    ref = repo_hit_ref(*counts.values(), path_hits, timer_hits)
    rt_family_breadth = family_diversity_score(counts)
    rt_family_entropy = family_entropy_score(counts)
    rt_docs_density = _log_repo_signal(sum(counts.values()), ref)
    if len(source) < MIN_SOURCE_FILES_FOR_IMPLEMENTATION_SIGNAL:
        rt_implementation_signal = None
    else:
        rt_implementation_signal = _log_repo_signal(path_hits + timer_hits, ref)
    block = mean([rt_family_breadth, rt_family_entropy, rt_docs_density, rt_implementation_signal])
    return block, {"available": True, "block": block, "sub": {
        "RT_family_breadth": rt_family_breadth,
        "RT_family_entropy": rt_family_entropy,
        "RT_docs_density": rt_docs_density,
        "RT_implementation_signal": rt_implementation_signal,
    }, "raw": {"family_counts": counts, "path_hits": path_hits, "timer_or_scheduler_hits": timer_hits, "hit_ref": ref}}


def compute_control_architecture_score(repo_path: Path) -> Tuple[Optional[float], Dict[str, Any]]:
    texts = candidate_texts(repo_path, include_source=True)
    counts = family_counts(texts, CONTROL_ARCH_FAMILIES)
    path_hits = count_path_hits(repo_path, [r"control|controller|agent|hierarchical|distributed|droop|protection|market"])
    ref = repo_hit_ref(*counts.values(), path_hits)
    ca_family_breadth = family_diversity_score(counts)
    ca_family_entropy = family_entropy_score(counts)
    ca_docs_density = _log_repo_signal(sum(counts.values()), ref)
    ca_path_evidence = _log_repo_signal(path_hits, ref)
    block = mean([ca_family_breadth, ca_family_entropy, ca_docs_density, ca_path_evidence])
    return block, {"available": True, "block": block, "sub": {
        "CA_family_breadth": ca_family_breadth,
        "CA_family_entropy": ca_family_entropy,
        "CA_docs_density": ca_docs_density,
        "CA_path_evidence": ca_path_evidence,
    }, "raw": {"family_counts": counts, "path_hits": path_hits, "hit_ref": ref}}


def compute_cpri(repo_path: Path, repo_url: Optional[str] = None, github_token: Optional[str] = None,
                 learned_weights: Optional[Dict[str, float]] = None, calibration: Optional[Dict[str, Any]] = None,
                 aggregation: str = "geometric") -> Dict[str, Any]:
    n_hi, hi_d = compute_hardware_interface_score(repo_path)
    n_rt, rt_d = compute_real_time_score(repo_path)
    n_ca, ca_d = compute_control_architecture_score(repo_path)
    blocks = {"hardware_interface_score": n_hi, "real_time_score": n_rt, "control_architecture_score": n_ca}
    score, weights, weights_source, agg_name = final_score_from_blocks(blocks, learned_weights, aggregation)
    return {"repository_path": str(repo_path), "repository_url": repo_url, "score": score, "band": band(score), "aggregation": agg_name,
            "details": {"CPRI": score, "N_hardware_interface_score": n_hi, "N_real_time_score": n_rt, "N_control_architecture_score": n_ca,
                        "weights": weights, "weights_source": weights_source,
                        "blocks": {"hardware_interface_score": hi_d, "real_time_score": rt_d, "control_architecture_score": ca_d},
                        "method_note": "CPRI measures cyber-physical suitability for power-system operation, not generic interoperability from DI."}}


# =========================================================
# EDF.3 ENSC — ENS & Reliability Capability
# =========================================================

def compute_ens_calculation_capability(repo_path: Path) -> Tuple[Optional[float], Dict[str, Any]]:
    texts = candidate_texts(repo_path, include_source=True)
    counts = family_counts(texts, ENS_FAMILIES)
    path_hits = count_path_hits(repo_path, [r"ens|eens|unserved|not[-_ ]served|load[-_ ]shedding|curtail|outage|voll"])
    test_hits = count_path_hits(repo_path, [r"test.*(?:ens|eens|reliability|load[-_ ]shedding|unserved)|(?:ens|eens|reliability).*test"])
    ref = repo_hit_ref(*counts.values(), path_hits, test_hits)
    ens_family_breadth = family_diversity_score(counts)
    ens_family_entropy = family_entropy_score(counts)
    ens_docs_density = _log_repo_signal(sum(counts.values()), ref)
    ens_implementation_evidence = _log_repo_signal(path_hits + test_hits, ref)
    block = mean([ens_family_breadth, ens_family_entropy, ens_docs_density, ens_implementation_evidence])
    return block, {"available": True, "block": block, "sub": {
        "ENS_family_breadth": ens_family_breadth,
        "ENS_family_entropy": ens_family_entropy,
        "ENS_docs_density": ens_docs_density,
        "ENS_implementation_evidence": ens_implementation_evidence,
    }, "raw": {"family_counts": counts, "path_hits": path_hits, "test_hits": test_hits, "hit_ref": ref}}


def compute_reliability_metrics(repo_path: Path) -> Tuple[Optional[float], Dict[str, Any]]:
    texts = candidate_texts(repo_path, include_source=True)
    counts = family_counts(texts, RELIABILITY_FAMILIES)
    path_hits = count_path_hits(repo_path, [r"reliability|lole|lolp|lolh|saidi|saifi|adequacy|availability|outage"])
    test_hits = count_path_hits(repo_path, [r"test.*(?:lole|lolp|saidi|saifi|adequacy|reliability)|(?:lole|lolp|saidi|saifi|adequacy|reliability).*test"])
    ref = repo_hit_ref(*counts.values(), path_hits, test_hits)
    rm_family_breadth = family_diversity_score(counts)
    rm_family_entropy = family_entropy_score(counts)
    rm_docs_density = _log_repo_signal(sum(counts.values()), ref)
    rm_implementation_evidence = _log_repo_signal(path_hits + test_hits, ref)
    block = mean([rm_family_breadth, rm_family_entropy, rm_docs_density, rm_implementation_evidence])
    return block, {"available": True, "block": block, "sub": {
        "RM_family_breadth": rm_family_breadth,
        "RM_family_entropy": rm_family_entropy,
        "RM_docs_density": rm_docs_density,
        "RM_implementation_evidence": rm_implementation_evidence,
    }, "raw": {"family_counts": counts, "path_hits": path_hits, "test_hits": test_hits, "hit_ref": ref}}


def compute_ensc(repo_path: Path, repo_url: Optional[str] = None, github_token: Optional[str] = None,
                 learned_weights: Optional[Dict[str, float]] = None, calibration: Optional[Dict[str, Any]] = None,
                 aggregation: str = "geometric") -> Dict[str, Any]:
    n_ens, ens_d = compute_ens_calculation_capability(repo_path)
    n_rel, rel_d = compute_reliability_metrics(repo_path)
    blocks = {"ens_calculation_capability": n_ens, "reliability_metrics": n_rel}
    score, weights, weights_source, agg_name = final_score_from_blocks(blocks, learned_weights, aggregation)
    return {"repository_path": str(repo_path), "repository_url": repo_url, "score": score, "band": band(score), "aggregation": agg_name,
            "details": {"ENSC": score, "N_ens_calculation_capability": n_ens, "N_reliability_metrics": n_rel,
                        "weights": weights, "weights_source": weights_source,
                        "blocks": {"ens_calculation_capability": ens_d, "reliability_metrics": rel_d},
                        "method_note": "ENSC measures explicit power-system reliability and adequacy capabilities, not generic software reliability from EQ/SRSW."}}


# =========================================================
# EDF.4 EDVS — Energy-Domain Validation Score
# =========================================================

def compute_standard_test_cases(repo_path: Path) -> Tuple[Optional[float], Dict[str, Any]]:
    texts = candidate_texts(repo_path, include_source=True)
    counts = family_counts(texts, STANDARD_CASE_FAMILIES)
    case_files = files_matching(repo_path, [r"case\d+|ieee|matpower|pglib|pegase|cigre|rts[-_ ]?gmlc|benchmark"], suffixes=TEXT_SUFFIXES)
    path_hits = count_path_hits(repo_path, [r"case\d+|ieee|matpower|pglib|pegase|cigre|rts[-_ ]?gmlc|benchmark"])
    recency = median_recency_score(repo_path, case_files, half_life_days=1095.0) if case_files else None
    ref = repo_hit_ref(*counts.values(), len(case_files), path_hits)
    stc_family_breadth = family_diversity_score(counts)
    stc_family_entropy = family_entropy_score(counts)
    stc_evidence_density = _log_repo_signal(sum(counts.values()), ref)
    stc_artifact_evidence = _log_repo_signal(len(case_files) + path_hits, ref)
    block = mean([stc_family_breadth, stc_family_entropy, stc_evidence_density, stc_artifact_evidence, recency])
    return block, {"available": True, "block": block, "sub": {
        "STC_family_breadth": stc_family_breadth,
        "STC_family_entropy": stc_family_entropy,
        "STC_evidence_density": stc_evidence_density,
        "STC_artifact_evidence": stc_artifact_evidence,
        "STC_recency": recency,
    }, "raw": {"family_counts": counts, "case_files_count": len(case_files), "case_files_sample": [str(p.relative_to(repo_path)) for p in case_files[:50]], "path_hits": path_hits, "hit_ref": ref}}


def compute_scenario_validation(repo_path: Path) -> Tuple[Optional[float], Dict[str, Any]]:
    texts = candidate_texts(repo_path, include_source=True)
    counts = family_counts(texts, SCENARIO_FAMILIES)
    scenario_files = files_matching(repo_path, [r"scenario|contingency|outage|stress|dispatch|planning|der|renewable|resilience"], suffixes=TEXT_SUFFIXES)
    validation_hits = count_regex_hits_in_texts(texts, [r"validation", r"validated", r"verification", r"reference result", r"expected output", r"baseline", r"golden"])
    ref = repo_hit_ref(*counts.values(), len(scenario_files), validation_hits)
    sv_family_breadth = family_diversity_score(counts)
    sv_family_entropy = family_entropy_score(counts)
    sv_docs_density = _log_repo_signal(sum(counts.values()) + validation_hits, ref)
    sv_artifact_evidence = _log_repo_signal(len(scenario_files), ref)
    block = mean([sv_family_breadth, sv_family_entropy, sv_docs_density, sv_artifact_evidence])
    return block, {"available": True, "block": block, "sub": {
        "SV_family_breadth": sv_family_breadth,
        "SV_family_entropy": sv_family_entropy,
        "SV_docs_density": sv_docs_density,
        "SV_artifact_evidence": sv_artifact_evidence,
    }, "raw": {"family_counts": counts, "scenario_files_count": len(scenario_files), "scenario_files_sample": [str(p.relative_to(repo_path)) for p in scenario_files[:50]], "validation_hits": validation_hits, "hit_ref": ref}}


def compute_operational_datasets(repo_path: Path) -> Tuple[Optional[float], Dict[str, Any]]:
    texts = candidate_texts(repo_path, include_source=True)
    counts = family_counts(texts, OPERATIONAL_DATA_FAMILIES)
    data_files = files_matching(repo_path, [r"data|dataset|timeseries|time[-_ ]series|load|demand|weather|scada|measurement|market|price|entsoe|eia"], suffixes={".csv", ".json", ".yaml", ".yml", ".xml", ".txt", ".md", ".rst", ".ipynb", ".nc", ".h5", ".hdf5", ".parquet"})
    data_readmes = files_matching(repo_path, [r"data.*readme|dataset.*readme|license.*data|data.*license"], suffixes=TEXT_SUFFIXES)
    ref = repo_hit_ref(*counts.values(), len(data_files), len(data_readmes))
    od_family_breadth = family_diversity_score(counts)
    od_family_entropy = family_entropy_score(counts)
    od_docs_density = _log_repo_signal(sum(counts.values()), ref)
    od_artifact_evidence = _log_repo_signal(len(data_files), ref)
    od_license_docs = sat_exp(len(data_readmes), scale=1.0)
    block = mean([od_family_breadth, od_family_entropy, od_docs_density, od_artifact_evidence, od_license_docs])
    return block, {"available": True, "block": block, "sub": {
        "OD_family_breadth": od_family_breadth,
        "OD_family_entropy": od_family_entropy,
        "OD_docs_density": od_docs_density,
        "OD_artifact_evidence": od_artifact_evidence,
        "OD_license_or_readme_docs": od_license_docs,
    }, "raw": {"family_counts": counts, "data_files_count": len(data_files), "data_files_sample": [str(p.relative_to(repo_path)) for p in data_files[:50]], "data_readme_or_license_count": len(data_readmes), "hit_ref": ref}}


def compute_standards_validation(repo_path: Path) -> Tuple[Optional[float], Dict[str, Any]]:
    texts = candidate_texts(repo_path, include_source=True)
    counts = family_counts(texts, STANDARDS_VALIDATION_FAMILIES)
    standard_files = files_matching(repo_path, [r"iec|ieee|cim|cgmes|entsoe|grid[-_ ]code|standard|compliance|validation|benchmark"], suffixes=TEXT_SUFFIXES)
    ref = repo_hit_ref(*counts.values(), len(standard_files))
    std_family_breadth = family_diversity_score(counts)
    std_family_entropy = family_entropy_score(counts)
    std_docs_density = _log_repo_signal(sum(counts.values()), ref)
    std_artifact_evidence = _log_repo_signal(len(standard_files), ref)
    block = mean([std_family_breadth, std_family_entropy, std_docs_density, std_artifact_evidence])
    return block, {"available": True, "block": block, "sub": {
        "STD_family_breadth": std_family_breadth,
        "STD_family_entropy": std_family_entropy,
        "STD_docs_density": std_docs_density,
        "STD_artifact_evidence": std_artifact_evidence,
    }, "raw": {"family_counts": counts, "standard_files_count": len(standard_files), "standard_files_sample": [str(p.relative_to(repo_path)) for p in standard_files[:50]], "hit_ref": ref}}


def compute_edvs(repo_path: Path, repo_url: Optional[str] = None, github_token: Optional[str] = None,
                 learned_weights: Optional[Dict[str, float]] = None, calibration: Optional[Dict[str, Any]] = None,
                 aggregation: str = "geometric") -> Dict[str, Any]:
    n_stc, stc_d = compute_standard_test_cases(repo_path)
    n_sv, sv_d = compute_scenario_validation(repo_path)
    n_od, od_d = compute_operational_datasets(repo_path)
    n_std, std_d = compute_standards_validation(repo_path)
    blocks = {"standard_test_cases": n_stc, "scenario_validation": n_sv, "operational_datasets": n_od, "standards_validation": n_std}
    score, weights, weights_source, agg_name = final_score_from_blocks(blocks, learned_weights, aggregation)
    return {"repository_path": str(repo_path), "repository_url": repo_url, "score": score, "band": band(score), "aggregation": agg_name,
            "details": {"EDVS": score, "N_standard_test_cases": n_stc, "N_scenario_validation": n_sv,
                        "N_operational_datasets": n_od, "N_standards_validation": n_std,
                        "weights": weights, "weights_source": weights_source,
                        "blocks": {"standard_test_cases": stc_d, "scenario_validation": sv_d, "operational_datasets": od_d, "standards_validation": std_d},
                        "method_note": "EDVS measures validation on recognized and realistic energy-domain artifacts, not generic test coverage from EQ/GC."}}


# =========================================================
# EDF category aggregate
# =========================================================

def compute_edf(repo_path: Path, repo_url: Optional[str] = None, github_token: Optional[str] = None,
                learned_weights: Optional[Dict[str, float]] = None, calibration: Optional[Dict[str, Any]] = None,
                aggregation: str = "geometric") -> Dict[str, Any]:
    esci = compute_esci(repo_path, repo_url, github_token, None, calibration, aggregation)
    cpri = compute_cpri(repo_path, repo_url, github_token, None, calibration, aggregation)
    ensc = compute_ensc(repo_path, repo_url, github_token, None, calibration, aggregation)
    edvs = compute_edvs(repo_path, repo_url, github_token, None, calibration, aggregation)
    blocks = {
        "ESCI": esci.get("score") / 100.0 if esci.get("score") is not None else None,
        "CPRI": cpri.get("score") / 100.0 if cpri.get("score") is not None else None,
        "ENSC": ensc.get("score") / 100.0 if ensc.get("score") is not None else None,
        "EDVS": edvs.get("score") / 100.0 if edvs.get("score") is not None else None,
    }
    score, weights, weights_source, agg_name = final_score_from_blocks(blocks, learned_weights, aggregation)
    return {"repository_path": str(repo_path), "repository_url": repo_url, "score": score, "band": band(score), "aggregation": agg_name,
            "details": {"EDF": score, "ESCI": esci.get("score"), "CPRI": cpri.get("score"), "ENSC": ensc.get("score"), "EDVS": edvs.get("score"),
                        "weights": weights, "weights_source": weights_source,
                        "scores": {"ESCI": esci, "CPRI": cpri, "ENSC": ensc, "EDVS": edvs},
                        "missingness_note": "Unavailable values remain None and are excluded with weight renormalization; observable absence is scored as 0.0.",
                        "method_note": "EDF evaluates power-sector fitness: energy-system capability, cyber-physical readiness, reliability/ENS capability, and domain validation."}}


# =========================================================
# Corpus weight-learning entry points
# =========================================================

def learn_esci_weights(dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = ["model_type_coverage", "der_support", "grid_analysis_capability", "voltage_level_coverage"]
    rows = [{k: row.get(k, row.get("N_" + k)) for k in keys} for row in dataset]
    return {"weights": norm_weights_from_rows(rows, keys), "n_repos": len(dataset)}


def learn_cpri_weights(dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = ["hardware_interface_score", "real_time_score", "control_architecture_score"]
    rows = [{k: row.get(k, row.get("N_" + k)) for k in keys} for row in dataset]
    return {"weights": norm_weights_from_rows(rows, keys), "n_repos": len(dataset)}


def learn_ensc_weights(dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = ["ens_calculation_capability", "reliability_metrics"]
    rows = [{k: row.get(k, row.get("N_" + k)) for k in keys} for row in dataset]
    return {"weights": norm_weights_from_rows(rows, keys), "n_repos": len(dataset)}


def learn_edvs_weights(dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = ["standard_test_cases", "scenario_validation", "operational_datasets", "standards_validation"]
    rows = [{k: row.get(k, row.get("N_" + k)) for k in keys} for row in dataset]
    return {"weights": norm_weights_from_rows(rows, keys), "n_repos": len(dataset)}


def learn_edf_weights(dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = ["ESCI", "CPRI", "ENSC", "EDVS"]
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
    parser = argparse.ArgumentParser(description="RECOPS Energy-Domain Fitness scores: ESCI, CPRI, ENSC, EDVS, EDF")
    parser.add_argument("repo_url", nargs="?", help="GitHub repository URL, e.g. https://github.com/owner/repo")
    parser.add_argument("--local", dest="local_repo", help="Analyze an already-local repository path instead of cloning")
    parser.add_argument("--token", dest="github_token", default=os.environ.get("GITHUB_TOKEN"), help="GitHub token; defaults to env GITHUB_TOKEN")
    parser.add_argument("--score", choices=["esci", "cpri", "ensc", "edvs", "edf", "all"], default="all", help="Which EDF score to compute")
    parser.add_argument("--weights-json", dest="weights_json", help="Optional learned weights JSON for the selected score")
    parser.add_argument("--calibration-json", dest="calibration_json", help="Optional corpus calibration JSON")
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
    if sc == "esci":
        result = compute_esci(repo_path, args.repo_url, args.github_token, split_weights_for_score("esci", weights_loaded), calibration, args.aggregation)
    elif sc == "cpri":
        result = compute_cpri(repo_path, args.repo_url, args.github_token, split_weights_for_score("cpri", weights_loaded), calibration, args.aggregation)
    elif sc == "ensc":
        result = compute_ensc(repo_path, args.repo_url, args.github_token, split_weights_for_score("ensc", weights_loaded), calibration, args.aggregation)
    elif sc == "edvs":
        result = compute_edvs(repo_path, args.repo_url, args.github_token, split_weights_for_score("edvs", weights_loaded), calibration, args.aggregation)
    elif sc == "edf":
        result = compute_edf(repo_path, args.repo_url, args.github_token, split_weights_for_score("edf", weights_loaded), calibration, args.aggregation)
    else:
        result = {"repository_path": str(repo_path), "repository_url": args.repo_url,
                  "scores": {"ESCI": compute_esci(repo_path, args.repo_url, args.github_token, None, calibration, args.aggregation),
                             "CPRI": compute_cpri(repo_path, args.repo_url, args.github_token, None, calibration, args.aggregation),
                             "ENSC": compute_ensc(repo_path, args.repo_url, args.github_token, None, calibration, args.aggregation),
                             "EDVS": compute_edvs(repo_path, args.repo_url, args.github_token, None, calibration, args.aggregation),
                             "EDF": compute_edf(repo_path, args.repo_url, args.github_token, None, calibration, args.aggregation)},
                  "note": "With --score all, --weights-json is ignored to avoid applying one weight file to multiple formulas."}

    txt = json.dumps(result, indent=args.indent, ensure_ascii=False, sort_keys=False)
    if args.out:
        Path(args.out).write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
