#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RECOPS Repository Scorer v2

"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import warnings
from collections import Counter
from pathlib import Path

try:
    import requests
except Exception:
    requests = None

try:
    import networkx as nx
except Exception:
    nx = None

try:
    from radon.metrics import mi_visit
    from radon.complexity import cc_visit
except Exception:
    mi_visit = None
    cc_visit = None


# =========================================================
# Utils
# =========================================================

def clamp01(x):
    return max(0.0, min(1.0, float(x)))


def mean(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def geometric(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    if any(v <= 0 for v in vals):
        return 0.0
    p = 1.0
    for v in vals:
        p *= v
    return p ** (1 / len(vals))




def weighted_geometric(pairs, eps=1e-6):
    """
    Weighted geometric mean with explicit missingness.

    - None values are ignored.
    - Remaining weights are renormalized.
    - A zero value remains near-zero through eps, preserving non-compensation
      without numerical log(0) failure.
    """
    vals = []
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
    # numerical collapse to near-zero in sparse evidence settings.
    positive = sorted([v for v, _ in vals if v > 0.0])
    if positive:
        floor = min(0.05, max(eps, 0.5 * positive[0]))
    else:
        floor = eps
    acc = 0.0
    for v, w in vals:
        acc += (w / wsum) * math.log(max(floor, v))
    return clamp01(math.exp(acc))


def weighted_sum(pairs):
    """Weighted arithmetic mean with None values removed and weights renormalized."""
    vals = []
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


def _data_driven_weights_from_blocks(blocks):
    """
    Repository-level fallback when corpus-learned weights are unavailable.

    Observed blocks farther from the repository cross-block mean receive more
    weight because they discriminate this repository more. If all observed blocks
    are identical, equal weights are used across observed blocks. Missing blocks
    receive zero weight.
    """
    observed = {k: float(v) for k, v in blocks.items() if v is not None}
    out = {k: 0.0 for k in blocks}
    if not observed:
        return out
    mu = sum(observed.values()) / len(observed)
    raw = {k: abs(v - mu) for k, v in observed.items()}
    z = sum(raw.values())
    if z <= 0:
        raw = {k: 1.0 for k in observed}
        z = float(len(observed))
    # Smooth toward uniform weights to reduce over-concentration from a single
    # outlier block when observed evidence is sparse.
    n = float(len(observed))
    smooth = 0.30
    uniform = 1.0 / n
    for k, v in raw.items():
        out[k] = (1.0 - smooth) * (v / z) + smooth * uniform
    return out


def _robust_variance(vals):
    vals = [float(v) for v in vals if v is not None]
    if len(vals) < 2:
        return 0.0
    mu = sum(vals) / len(vals)
    return sum((x - mu) ** 2 for x in vals) / len(vals)


def _pairwise_abs_corr(a, b):
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if len(pairs) < 3:
        return None
    xs = [float(x) for x, _ in pairs]
    ys = [float(y) for _, y in pairs]
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs) / len(xs)
    vy = sum((y - my) ** 2 for y in ys) / len(ys)
    if vx <= 0 or vy <= 0:
        return None
    cov = sum((x - mx) * (y - my) for x, y in pairs) / len(pairs)
    return abs(cov / math.sqrt(vx * vy))


def _norm_weights_from_rows(rows, keys):
    """
    Learn corpus weights using:
        raw_weight(k) = variance(k) * coverage(k) * uniqueness(k)

    variance rewards discriminative blocks; coverage rewards reliable availability;
    uniqueness penalizes redundant blocks.
    """
    vals = {k: [r.get(k) for r in rows] for k in keys}

    def coverage(arr):
        return sum(1 for x in arr if x is not None) / max(1, len(arr))

    info = {k: _robust_variance(vals[k]) for k in keys}
    covg = {k: coverage(vals[k]) for k in keys}
    uniq = {}
    for k in keys:
        corrs = [_pairwise_abs_corr(vals[k], vals[j]) for j in keys if j != k]
        c = mean(corrs)
        uniq[k] = clamp01(1.0 - c) if c is not None else 1.0
    raw = {k: max(1e-9, info[k] * covg[k] * uniq[k]) for k in keys}
    z = sum(raw.values()) or 1.0
    return {k: raw[k] / z for k in keys}


def _final_score_from_blocks(blocks, learned_weights=None, aggregation="geometric"):
    """
    Compute final indicator score with data-driven block weights.

    Priority:
    1) use corpus-learned weights when supplied;
    2) otherwise use repository-level dispersion fallback.
    """
    weights_source = "learned_from_corpus" if learned_weights is not None else "repo_data_driven_fallback"
    weights = learned_weights or _data_driven_weights_from_blocks(blocks)
    pairs = [(blocks[k], float(weights.get(k, 0.0))) for k in blocks]
    agg = (aggregation or "geometric").strip().lower()
    if agg == "sum":
        val = weighted_sum(pairs)
        agg_name = "weighted_sum"
    else:
        val = weighted_geometric(pairs)
        agg_name = "weighted_geometric"
    if val is None:
        return None, weights, weights_source, agg_name
    return 100.0 * val, weights, weights_source, agg_name


def learn_eq_weights(dataset):
    """
    Learn top-level block weights for each EQ indicator from a corpus.

    Each dataset row may contain raw block names or RECOPS detail keys.
    Returns a dictionary keyed by indicator: CQI, ARS, PER, SPS, SRSW.
    """
    specs = {
        "CQI": {
            "maintainability": ["maintainability", "N_maint"],
            "testing": ["testing", "N_test"],
            "documentation": ["documentation", "N_doc"],
            "practices": ["practices", "N_practice"],
        },
        "ARS": {
            "modularity": ["modularity", "N_mod"],
            "graph": ["graph", "N_graph"],
            "architecture": ["architecture", "N_arch"],
            "scalability": ["scalability", "N_scale"],
        },
        "PER": {
            "benchmark": ["benchmark", "N_bench"],
            "execution": ["execution", "N_exec"],
            "memory": ["memory", "N_memory"],
            "profile": ["profile", "N_profile"],
        },
        "SPS": {
            "inventory": ["inventory", "N_inventory"],
            "pinning": ["pinning", "N_pinning"],
            "automation": ["automation", "N_automation"],
            "vulnerability": ["vulnerability", "N_vuln"],
        },
        "SRSW": {
            "error_handling": ["error_handling", "N_err"],
            "recovery": ["recovery", "N_rec"],
            "release_stability": ["release_stability", "N_rel"],
            "observability": ["observability", "N_obs"],
        },
    }
    learned = {}
    for indicator, mapping in specs.items():
        rows = []
        for row in dataset:
            src = row.get(indicator, row)
            if isinstance(src, dict) and "details" in src and isinstance(src["details"], dict):
                src = src["details"]
            out = {}
            for key, aliases in mapping.items():
                val = None
                if isinstance(src, dict):
                    for a in aliases:
                        if a in src:
                            val = src.get(a)
                            break
                out[key] = val
            rows.append(out)
        learned[indicator] = _norm_weights_from_rows(rows, list(mapping.keys()))
    return {"weights": learned, "n_repos": len(dataset)}


def _load_learned_weights(path, indicator):
    if not path:
        return None
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(data, dict) and "weights" in data and isinstance(data["weights"], dict):
        data = data["weights"]
    if isinstance(data, dict) and indicator in data and isinstance(data[indicator], dict):
        return data[indicator]
    if isinstance(data, dict):
        return data
    return None


def run(cmd, cwd=None):
    try:
        env = os.environ.copy()
        env["GIT_TERMINAL_PROMPT"] = "0"
        r = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=45
        )
        return r.returncode, r.stdout, r.stderr
    except Exception:
        return 1, "", ""


# =========================================================
# GitHub download
# =========================================================

def github_to_zip(url: str):
    m = re.search(r"github\.com/([^/]+)/([^/]+)", url)
    if not m:
        raise ValueError("Invalid GitHub URL")
    owner = m.group(1)
    repo = m.group(2).replace(".git", "")
    return owner, repo, f"https://github.com/{owner}/{repo}/archive/refs/heads/main.zip"


def clone_or_download(url: str):
    temp = tempfile.mkdtemp(prefix="recops_")
    target = Path(temp)
    owner = None
    repo = None

    try:
        owner, repo, _ = github_to_zip(url)
    except Exception:
        owner, repo = None, None

    # Fast-fail for invalid/private/unreachable GitHub repos when requests is available.
    if requests is not None and owner and repo:
        try:
            meta = requests.get(f"https://api.github.com/repos/{owner}/{repo}", timeout=20)
            if meta.status_code == 404:
                raise RuntimeError(f"Repository not found: {owner}/{repo}")
            if meta.status_code in (401, 403):
                raise RuntimeError(
                    f"Repository metadata is not accessible (HTTP {meta.status_code}) for {owner}/{repo}. "
                    "It may be private or rate-limited."
                )
        except RuntimeError:
            raise
        except Exception:
            # Continue with git/download attempts if metadata check fails transiently.
            pass

    code, _, _ = run(["git", "--version"])
    if code == 0:
        # Keep commit/tag history for integrity of contributor and activity-derived signals.
        # Use partial clone to reduce payload while preserving history metadata.
        code, _, _ = run(["git", "clone", "--filter=blob:none", "--tags", url, str(target / "repo")])
        if code == 0:
            return target / "repo"

    if requests is None:
        raise RuntimeError("Need git or requests")

    owner, repo, _ = github_to_zip(url)
    zpath = target / "repo.zip"

    # Prefer the repository default branch from GitHub API, then fallback to
    # common branch names for older/newer repositories.
    candidate_branches = []
    try:
        meta = requests.get(f"https://api.github.com/repos/{owner}/{repo}", timeout=30)
        if meta.status_code == 200:
            default_branch = (meta.json() or {}).get("default_branch")
            if default_branch:
                candidate_branches.append(str(default_branch))
    except Exception:
        pass
    candidate_branches.extend(["main", "master"])

    seen = set()
    ordered_candidates = []
    for b in candidate_branches:
        if b not in seen:
            seen.add(b)
            ordered_candidates.append(b)

    r = None
    for branch in ordered_candidates:
        zip_url = f"https://github.com/{owner}/{repo}/archive/refs/heads/{branch}.zip"
        resp = requests.get(zip_url, timeout=30)
        if resp.status_code == 200:
            r = resp
            break
    if r is None:
        raise RuntimeError(
            f"Cannot download repository archive for {owner}/{repo} "
            f"(tried branches: {', '.join(ordered_candidates)})"
        )

    with open(zpath, "wb") as f:
        f.write(r.content)

    shutil.unpack_archive(str(zpath), str(target))
    dirs = [p for p in target.iterdir() if p.is_dir()]
    for d in dirs:
        if d.name != "repo":
            return d
    raise RuntimeError("Cannot extract repo")


# =========================================================
# File scan
# =========================================================

IGNORE_DIRS = {
    ".git", "__pycache__", ".pytest_cache",
    "node_modules", "dist", "build",
    ".venv", "venv", ".idea"
}


def iter_files(root):
    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        for f in files:
            yield Path(base) / f


def source_files(root):
    exts = {".py", ".js", ".ts", ".java", ".cpp", ".c", ".cs", ".jl"}
    return [p for p in iter_files(root) if p.suffix.lower() in exts]


def count_loc(path):
    total = 0
    for p in source_files(path):
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                total += sum(1 for _ in f)
        except Exception:
            pass
    return total


# =========================================================
# Better structural module detection
# =========================================================

def structural_module_name(root: Path, file: Path):
    rel = file.relative_to(root)
    parts = rel.parts

    if len(parts) == 1:
        return parts[0].replace(file.suffix, "")

    banned = {"tests", "test", "docs", "examples", "example", "benchmarks"}

    cleaned = [p for p in parts[:-1] if p.lower() not in banned]

    if not cleaned:
        return parts[0]

    # keep two levels if possible
    if len(cleaned) >= 2:
        return "/".join(cleaned[:2])

    return cleaned[0]


def detect_modules(root):
    c = Counter()
    for f in source_files(root):
        mod = structural_module_name(root, f)
        c[mod] += 1
    return c


# =========================================================
# Repo signals
# =========================================================

def exists_any(root, names):
    names = {n.lower() for n in names}
    for p in iter_files(root):
        if p.name.lower() in names:
            return 1.0
    return 0.0


def has_dir(root, names):
    names = {n.lower() for n in names}
    for p in root.iterdir():
        if p.is_dir() and p.name.lower() in names:
            return 1.0
    return 0.0


def count_test_files(root):
    n = 0
    for p in iter_files(root):
        low = p.name.lower()
        if "test" in low and p.suffix in {".py", ".js", ".ts"}:
            n += 1
    return n


def detect_ci(root):
    wf = root / ".github" / "workflows"
    return 1.0 if wf.exists() else 0.0


def detect_lint(root):
    names = [
        ".flake8", "pyproject.toml", "ruff.toml",
        ".eslintrc", ".pylintrc"
    ]
    return exists_any(root, names)


def detect_format(root):
    names = [
        "pyproject.toml", ".prettierrc",
        "black.toml"
    ]
    return exists_any(root, names)


def detect_precommit(root):
    return exists_any(root, [".pre-commit-config.yaml"])


def detect_arch_doc(root):
    return exists_any(root, [
        "architecture.md", "design.md", "system_design.md"
    ])


def detect_diagrams(root):
    for p in iter_files(root):
        if p.suffix.lower() in {".drawio", ".puml", ".svg"}:
            return 1.0
    return 0.0


def detect_examples(root):
    return has_dir(root, ["examples", "notebooks", "demo"])


def detect_api_docs(root):
    return exists_any(root, ["mkdocs.yml", "sphinx.conf", "conf.py"])


def detect_bench(root):
    return has_dir(root, ["benchmarks", "benchmark"])


# =========================================================
# Python metrics
# =========================================================

def python_files(root):
    return [p for p in source_files(root) if p.suffix == ".py"]


def python_metrics(root):
    files = python_files(root)

    if not files:
        return None, None, None

    mis = []
    ccs = []
    doc_total = 0
    doc_ok = 0

    for p in files:
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        if mi_visit:
            try:
                with warnings.catch_warnings():
                    # Some third-party repositories contain invalid escape sequences
                    # in string literals; suppress parser SyntaxWarning from radon.
                    warnings.filterwarnings(
                        "ignore",
                        message=r".*invalid escape sequence.*",
                        category=SyntaxWarning,
                    )
                    mis.append(mi_visit(txt, True))
            except Exception:
                pass

        if cc_visit:
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=r".*invalid escape sequence.*",
                        category=SyntaxWarning,
                    )
                    blocks = cc_visit(txt)
                for b in blocks:
                    ccs.append(b.complexity)
            except Exception:
                pass

        funcs = len(re.findall(r"^\s*def\s+", txt, flags=re.M))
        docs = len(re.findall(r'"""', txt)) // 2
        doc_total += funcs
        doc_ok += min(funcs, docs)

    mean_mi = mean(mis) if mis else None
    mean_cc = mean(ccs) if ccs else None
    doc_cov = (doc_ok / doc_total) if doc_total > 0 else None

    return mean_mi, mean_cc, doc_cov


# =========================================================
# Import graph (Python only)
# =========================================================

def build_python_graph(root):
    if nx is None:
        return None

    G = nx.DiGraph()
    files = python_files(root)

    names = {}
    for f in files:
        mod = structural_module_name(root, f)
        names[f] = mod
        G.add_node(mod)

    for f in files:
        try:
            txt = f.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        src = names[f]

        for m in re.findall(r"^\s*import\s+([a-zA-Z0-9_\.]+)", txt, flags=re.M):
            tgt = m.split(".")[0]
            for n in G.nodes:
                if n.startswith(tgt):
                    G.add_edge(src, n)

        for m in re.findall(r"^\s*from\s+([a-zA-Z0-9_\.]+)\s+import", txt, flags=re.M):
            tgt = m.split(".")[0]
            for n in G.nodes:
                if n.startswith(tgt):
                    G.add_edge(src, n)

    return G


# =========================================================
# CQI
# =========================================================

def compute_cqi(root, learned_weights=None, aggregation="geometric"):
    loc = count_loc(root)
    mods = detect_modules(root)
    structural_modules = len(mods)
    ci_signal = detect_ci(root)

    mean_mi, mean_cc, doc_cov = python_metrics(root)

    n_mi = clamp01(mean_mi / 100) if mean_mi is not None else None
    n_cc = clamp01(math.exp(-mean_cc / 10)) if mean_cc is not None else None
    n_struct = clamp01(structural_modules / max(1, math.sqrt(loc / 1000)))

    n_maint = mean([n_mi, n_cc, n_struct])

    test_files = count_test_files(root)
    n_ci = ci_signal
    test_density = clamp01(test_files / max(1, structural_modules))
    n_test = mean([n_ci, test_density])

    has_readme = exists_any(root, ["readme.md", "readme.rst"])
    n_docs = has_dir(root, ["docs"])
    n_api = detect_api_docs(root)

    documented_modules = max(0, structural_modules - 1)
    doc_density = clamp01(documented_modules / max(1, structural_modules))

    n_doc = mean([
        doc_cov,
        n_docs,
        n_api,
        doc_density,
        has_readme,
    ])

    n_practice = mean([
        detect_lint(root),
        detect_format(root),
        detect_precommit(root),
        ci_signal
    ])

    blocks = {
        "maintainability": n_maint,
        "testing": n_test,
        "documentation": n_doc,
        "practices": n_practice,
    }
    cqi, weights, weights_source, agg_name = _final_score_from_blocks(blocks, learned_weights, aggregation)

    return {
        "score": cqi,
        "band": band(cqi),
        "details": {
            "CQI": cqi,
            "weights": weights,
            "weights_source": weights_source,
            "aggregation": agg_name,
            "N_maint": n_maint,
            "N_test": n_test,
            "N_doc": n_doc,
            "N_practice": n_practice,
            "raw.loc": loc,
            "raw.structural_modules": structural_modules,
        }
    }


# =========================================================
# ARS
# =========================================================

def entropy(counter):
    vals = list(counter.values())
    s = sum(vals)
    if s == 0:
        return None
    if len(vals) <= 1:
        return 0.0
    probs = [v / s for v in vals]
    H = -sum(p * math.log(p) for p in probs if p > 0)
    return H / math.log(len(vals))


def compute_ars(root, learned_weights=None, aggregation="geometric"):
    loc = count_loc(root)
    mods = detect_modules(root)
    structural_modules = len(mods)

    n_module_count = clamp01(
        structural_modules / max(1, math.sqrt(loc / 1000))
    )
    n_balance = entropy(mods)
    n_mod = mean([n_module_count, n_balance])

    G = build_python_graph(root)
    if nx is None:
        density = None
        cycle_ratio = None
        n_graph = None
    elif G is not None and len(G.nodes) > 1:
        density = nx.density(G)
        n_density = clamp01(math.exp(-2 * density))

        scc = list(nx.strongly_connected_components(G))
        cyc_nodes = sum(len(c) for c in scc if len(c) > 1)
        cycle_ratio = cyc_nodes / len(G.nodes)
        n_cycles = clamp01(1 - cycle_ratio)

        n_graph = mean([n_density, n_cycles])
    else:
        density = None
        cycle_ratio = None
        n_graph = None

    doc_density = clamp01((structural_modules - 1) / max(1, structural_modules))

    n_arch = mean([
        detect_arch_doc(root),
        detect_diagrams(root),
        detect_examples(root),
        detect_api_docs(root),
        doc_density
    ])

    # Data-driven scalability block (no fixed constants).
    # Uses observable engineering signals already used elsewhere in the scorer.
    perf = _performance_engineering_signals(root)
    hit_ref = max(
        1.0,
        perf["parallel_hits"],
        perf["accel_hits"] + perf["compiled_ext_hits"],
        perf["sparse_hits"],
        perf["profiler_hits"],
        perf["perfdoc_hits"],
    )

    # Parallel-distribution evidence
    n_parallel = _log_repo_signal(perf["parallel_hits"], hit_ref)

    # Solver/compute scaling evidence (accelerators + compiled extensions)
    n_solver = _log_repo_signal(perf["accel_hits"] + perf["compiled_ext_hits"], hit_ref)

    # Sparse / large-scale memory handling evidence
    n_sparse = _log_repo_signal(perf["sparse_hits"], hit_ref)

    # Benchmark evidence from suite presence + CI usage + docs
    bench_presence = _detect_benchmark_files(root)
    bench_ci = _workflow_signal(
        root,
        [
            r"pytest.*benchmark",
            r"--benchmark-json",
            r"\basv\b",
            r"\bbenchmark\b",
            r"\bperf\b",
            r"\bperformance\b",
        ],
    )
    bench_doc = _log_repo_signal(perf["perfdoc_hits"], hit_ref)
    n_bench = mean([bench_presence, bench_ci, bench_doc])

    n_scale = mean([n_parallel, n_solver, n_sparse, n_bench])

    blocks = {
        "modularity": n_mod,
        "graph": n_graph,
        "architecture": n_arch,
        "scalability": n_scale,
    }
    ars, weights, weights_source, agg_name = _final_score_from_blocks(blocks, learned_weights, aggregation)

    return {
        "score": ars,
        "band": band(ars),
        "details": {
            "ARS": ars,
            "weights": weights,
            "weights_source": weights_source,
            "aggregation": agg_name,
            "N_mod": n_mod,
            "N_graph": n_graph,
            "N_arch": n_arch,
            "N_scale": n_scale,
            "raw.loc": loc,
            "raw.structural_modules": structural_modules,
            "raw.module_file_counts": dict(mods),
            "raw.import_graph_density": density,
            "raw.cycle_ratio": cycle_ratio,
            "N_scale.N_parallel": n_parallel,
            "N_scale.N_solver": n_solver,
            "N_scale.N_sparse": n_sparse,
            "N_scale.N_bench": n_bench,
            "N_scale.N_bench.bench_presence": bench_presence,
            "N_scale.N_bench.bench_ci": bench_ci,
            "N_scale.N_bench.bench_doc": bench_doc,
            "raw.scale.parallel_hits": perf["parallel_hits"],
            "raw.scale.accel_hits": perf["accel_hits"],
            "raw.scale.compiled_ext_hits": perf["compiled_ext_hits"],
            "raw.scale.sparse_hits": perf["sparse_hits"],
            "raw.scale.profiler_hits": perf["profiler_hits"],
            "raw.scale.perfdoc_hits": perf["perfdoc_hits"],
            "raw.scale.hit_ref": hit_ref
        }
    }


# =========================================================
# Band
# =========================================================

def band(score):
    if score is None:
        return "Unavailable"
    if score >= 80:
        return "Excellent"
    if score >= 60:
        return "Good"
    if score >= 40:
        return "Moderate"
    if score >= 20:
        return "Poor"
    return "Very Poor"

# =========================================================
# Shared helpers for PER and SPS
# =========================================================

def _read_text(path):
    try:
        return Path(path).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _safe_json_load(path):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return json.load(f)
    except Exception:
        return None


def _signal_hits(hits):
    """
    Continuous saturation used for small binary-like counts.
    """
    return clamp01(1.0 - math.exp(-max(0.0, float(hits))))


def _log_repo_signal(hits, ref):
    """
    Slower saturation than 1-exp(-h), normalized by repository-relative reference.
    Prevents very large hit counts from trivially saturating all performance blocks.
    """
    hits = max(0.0, float(hits))
    ref = max(0.0, float(ref))
    if hits <= 0 or ref <= 0:
        return 0.0
    return clamp01(math.log1p(hits) / math.log1p(ref))


def _repo_text_files(root, suffixes=None):
    suffixes = suffixes or {
        ".py", ".js", ".ts", ".java", ".jl", ".c", ".cpp", ".h", ".hpp",
        ".yml", ".yaml", ".toml", ".json", ".md", ".txt", ".rst", ".ini", ".cfg"
    }
    return [p for p in iter_files(root) if p.suffix.lower() in suffixes]


def _count_pattern_hits(root, patterns, suffixes=None):
    total = 0
    rx = [re.compile(p, flags=re.I | re.M) for p in patterns]
    for p in _repo_text_files(root, suffixes=suffixes):
        txt = _read_text(p)
        if not txt:
            continue
        for r in rx:
            total += len(r.findall(txt))
    return total


def _workflow_texts(root):
    wf = root / ".github" / "workflows"
    texts = []
    if not wf.exists():
        return texts
    for p in wf.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".yml", ".yaml"}:
            texts.append(_read_text(p))
    return texts


def _workflow_signal(root, patterns):
    texts = _workflow_texts(root)
    if not texts:
        # No workflow YAML present under .github/workflows — treat CI benchmark hooks as unknown, not absent.
        return None
    rx = [re.compile(p, flags=re.I | re.M) for p in patterns]
    score = 0
    for txt in texts:
        if any(r.search(txt) for r in rx):
            score += 1
    return _signal_hits(score)


def _is_python_repo(root):
    return len(python_files(root)) > 0


# =========================================================
# Benchmark and performance evidence
# =========================================================

def _performance_doc_hits(root):
    """
    Detect performance-related evidence in README/docs/examples/notebooks.
    This prevents N_bench from collapsing to zero when performance work exists
    but no formal benchmark suite is shipped in the repository.
    """
    hits = 0
    rx = [
        re.compile(r"\bperformance\b", flags=re.I),
        re.compile(r"\bbenchmark\b", flags=re.I),
        re.compile(r"\bruntime\b", flags=re.I),
        re.compile(r"\bspeed\b", flags=re.I),
        re.compile(r"\bscalab", flags=re.I),
        re.compile(r"\bprofil", flags=re.I),
        re.compile(r"\bmemory\b", flags=re.I),
        re.compile(r"\bsparse\b", flags=re.I),
        re.compile(r"\bparallel", flags=re.I),
    ]

    allowed_dirs = {"docs", "doc", "examples", "example", "notebooks"}

    for p in iter_files(root):
        if p.suffix.lower() not in {".md", ".rst", ".txt", ".ipynb", ".py", ".toml", ".yml", ".yaml"}:
            continue

        rel = p.relative_to(root)
        rel_parts = {x.lower() for x in rel.parts[:-1]}

        if p.name.lower().startswith("readme") or rel_parts.intersection(allowed_dirs):
            txt = _read_text(p)
            if not txt:
                continue
            for r in rx:
                hits += len(r.findall(txt))

    return hits


def _detect_benchmark_files(root):
    score = 0
    if has_dir(root, ["benchmarks", "benchmark"]) > 0:
        score += 1
    if exists_any(root, ["asv.conf.json", "asv.conf.toml"]) > 0:
        score += 1
    score += int(
        _count_pattern_hits(
            root,
            [r"pytest-benchmark", r"--benchmark-json", r"\basv\b", r"\bbenchmark\b"],
            suffixes={".toml", ".txt", ".cfg", ".ini", ".yml", ".yaml", ".md", ".json"},
        ) > 0
    )
    return _signal_hits(score)


def _parse_pytest_benchmark_json(root):
    """
    Returns a stability score based on coefficient of variation when benchmark JSON exists.
    If multiple benchmark entries are found, their CV-based scores are averaged.
    """
    scores = []
    for p in iter_files(root):
        if p.suffix.lower() != ".json":
            continue
        data = _safe_json_load(p)
        if not isinstance(data, dict):
            continue
        benches = data.get("benchmarks")
        if not isinstance(benches, list):
            continue
        for b in benches:
            stats = b.get("stats", {})
            mean_v = stats.get("mean")
            std_v = stats.get("stddev")
            if mean_v and mean_v > 0 and std_v is not None:
                cv = std_v / mean_v
                scores.append(math.exp(-cv))
    return mean(scores) if scores else None


def _parse_asv_results(root):
    """
    Returns an evidence score if ASV result artifacts are present.
    This does not compare absolute runtimes across machines; it only rewards
    the presence of structured benchmark results.
    """
    count = 0
    for p in iter_files(root):
        low = str(p).lower()
        if "asv" in low and p.suffix.lower() == ".json":
            count += 1
    return _signal_hits(count) if count > 0 else None


def _performance_engineering_signals(root):
    parallel_hits = _count_pattern_hits(
        root,
        [
            r"\bmultiprocessing\b",
            r"\bconcurrent\.futures\b",
            r"\bjoblib\b",
            r"\bdask\b",
            r"\bray\b",
            r"\bnumba\.prange\b",
            r"#pragma\s+omp",
            r"\bthreading\b",
            r"\bDistributed\b",
            r"\bThreads\.@threads\b",
        ],
    )

    accel_hits = _count_pattern_hits(
        root,
        [
            r"\bnumba\b",
            r"\bcython\b",
            r"\bpybind11\b",
            r"\bpyo3\b",
            r"\bcupy\b",
            r"\bjax\b",
            r"\btorch\.jit\b",
            r"\bnumpy\b",
            r"\bscipy\b",
            r"\bgurobipy\b",
            r"\bhighspy\b",
            r"\bcplex\b",
            r"\bipopt\b",
            r"\bcbc\b",
            r"\bglpk\b",
            r"\bSparseArrays\b",
        ],
    )

    sparse_hits = _count_pattern_hits(
        root,
        [
            r"\bscipy\.sparse\b",
            r"\bsparse\b",
            r"\bSparseArrays\b",
            r"\bcsr_matrix\b",
            r"\bcsc_matrix\b",
            r"\bcoo_matrix\b",
            r"\bmemmap\b",
            r"\bzarr\b",
            r"\bxarray\b",
            r"\bdask\.array\b",
            r"\bchunk",
            r"\bstream",
        ],
    )

    profiler_hits = _count_pattern_hits(
        root,
        [
            r"\bcProfile\b",
            r"\bprofile\b",
            r"\bline_profiler\b",
            r"\bpyinstrument\b",
            r"\bscalene\b",
            r"\bperf_counter\b",
            r"\btimeit\b",
        ],
    )

    compiled_ext_hits = 0
    for p in iter_files(root):
        if p.suffix.lower() in {".c", ".cc", ".cpp", ".cxx", ".pyx", ".pxd", ".so", ".pyd", ".jl", ".rs"}:
            compiled_ext_hits += 1

    perfdoc_hits = _performance_doc_hits(root)

    return {
        "parallel_hits": parallel_hits,
        "accel_hits": accel_hits,
        "sparse_hits": sparse_hits,
        "profiler_hits": profiler_hits,
        "compiled_ext_hits": compiled_ext_hits,
        "perfdoc_hits": perfdoc_hits,
    }


def compute_per(root, learned_weights=None, aggregation="geometric"):
    """
    Corrected Performance Efficiency Rating.

    Global formula unchanged:
    PER = 100 * (N_bench * N_exec * N_memory * N_profile)^(1/4)

    Internal blocks are computed additively and with repository-relative
    saturation to avoid artificial collapse or trivial saturation.
    """
    bench_presence = _detect_benchmark_files(root)
    bench_ci = _workflow_signal(
        root,
        [
            r"pytest.*benchmark",
            r"--benchmark-json",
            r"\basv\b",
            r"\bbenchmark\b",
            r"\bperf\b",
            r"\bperformance\b",
        ],
    )
    bench_json_score = _parse_pytest_benchmark_json(root)
    asv_score = _parse_asv_results(root)

    perf = _performance_engineering_signals(root)
    perfdoc_hits = perf["perfdoc_hits"]

    hit_ref = max(
        1.0,
        perf["parallel_hits"],
        perf["accel_hits"] + perf["compiled_ext_hits"],
        perf["sparse_hits"],
        perf["profiler_hits"],
        perfdoc_hits,
    )

    perfdoc_score = _log_repo_signal(perfdoc_hits, hit_ref)

    n_bench = mean([
        bench_presence,
        bench_ci,
        bench_json_score,
        asv_score,
        perfdoc_score,
    ])

    n_parallel = _log_repo_signal(perf["parallel_hits"], hit_ref)
    n_accel = _log_repo_signal(perf["accel_hits"] + perf["compiled_ext_hits"], hit_ref)
    n_exec = mean([n_parallel, n_accel])

    n_sparse = _log_repo_signal(perf["sparse_hits"], hit_ref)

    loc = count_loc(root)
    structural_modules = len(detect_modules(root))
    n_struct = clamp01(structural_modules / max(1, math.sqrt(max(1, loc) / 1000.0)))
    n_memory = mean([n_sparse, n_struct])

    n_profiler = _log_repo_signal(perf["profiler_hits"], hit_ref)
    n_profile = mean([n_profiler, bench_ci, perfdoc_score])

    blocks = {
        "benchmark": n_bench,
        "execution": n_exec,
        "memory": n_memory,
        "profile": n_profile,
    }
    per, weights, weights_source, agg_name = _final_score_from_blocks(blocks, learned_weights, aggregation)

    return {
        "score": per,
        "band": band(per),
        "details": {
            "PER": per,
            "weights": weights,
            "weights_source": weights_source,
            "aggregation": agg_name,
            "N_bench": n_bench,
            "N_exec": n_exec,
            "N_memory": n_memory,
            "N_profile": n_profile,
            "N_bench.bench_presence": bench_presence,
            "N_bench.bench_ci": bench_ci,
            "N_bench.pytest_benchmark_json": bench_json_score,
            "N_bench.asv_results": asv_score,
            "N_bench.perfdoc_score": perfdoc_score,
            "N_exec.N_parallel": n_parallel,
            "N_exec.N_accel": n_accel,
            "N_memory.N_sparse": n_sparse,
            "N_memory.N_struct": n_struct,
            "N_profile.N_profiler": n_profiler,
            "N_profile.N_bench_ci": bench_ci,
            "N_profile.N_perfdoc": perfdoc_score,
            "raw.parallel_hits": perf["parallel_hits"],
            "raw.accel_hits": perf["accel_hits"],
            "raw.sparse_hits": perf["sparse_hits"],
            "raw.profiler_hits": perf["profiler_hits"],
            "raw.compiled_ext_hits": perf["compiled_ext_hits"],
            "raw.perfdoc_hits": perfdoc_hits,
            "raw.hit_ref": hit_ref,
            "raw.loc": loc,
            "raw.structural_modules": structural_modules,
        },
    }


# =========================================================
# Dependency and supply-chain evidence for SPS
# =========================================================

def _dependency_files(root):
    """
    Return only runtime-relevant dependency manifests and lockfiles.
    Excludes docs/, tests/, examples/, benchmarks/, notebooks/ scoped files.
    """
    manifests = []
    lockfiles = []

    manifest_names = {
        "requirements.txt", "pyproject.toml", "setup.py", "setup.cfg",
        "package.json", "Pipfile", "environment.yml", "environment.yaml",
        "Project.toml", "Cargo.toml", "pom.xml"
    }
    lock_names = {
        "poetry.lock", "Pipfile.lock", "package-lock.json", "yarn.lock",
        "pnpm-lock.yaml", "conda-lock.yml", "conda-lock.yaml",
        "Cargo.lock", "Manifest.toml"
    }

    excluded_scopes = {"docs", "doc", "tests", "test", "examples", "example", "benchmarks", "benchmark", "notebooks"}

    for p in iter_files(root):
        rel = p.relative_to(root)
        rel_dirs = {x.lower() for x in rel.parts[:-1]}
        if rel_dirs.intersection(excluded_scopes):
            continue

        if p.name in manifest_names:
            manifests.append(p)
        if p.name in lock_names:
            lockfiles.append(p)

    return manifests, lockfiles


def _parse_requirements_stats(path):
    total = 0
    exact = 0
    bounded = 0

    txt = _read_text(path)
    for line in txt.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-r"):
            continue

        total += 1

        if "==" in line and ".*" not in line:
            exact += 1
            bounded += 1
        elif any(op in line for op in [">=", "<=", "~=", ">", "<"]):
            bounded += 1

    return total, exact, bounded


def _parse_package_json_stats(path):
    data = _safe_json_load(path)
    if not isinstance(data, dict):
        return 0, 0, 0

    total = 0
    exact = 0
    bounded = 0

    for key in ["dependencies", "devDependencies", "peerDependencies", "optionalDependencies"]:
        deps = data.get(key, {})
        if not isinstance(deps, dict):
            continue

        for _, ver in deps.items():
            total += 1
            if isinstance(ver, str):
                v = ver.strip()
                if re.match(r"^\d+(\.\d+)*([\-+].+)?$", v):
                    exact += 1
                    bounded += 1
                elif any(op in v for op in ["^", "~", ">", "<", ">=", "<="]):
                    bounded += 1

    return total, exact, bounded


def _parse_pyproject_stats(path):
    txt = _read_text(path)
    if not txt:
        return 0, 0, 0

    total = 0
    exact = 0
    bounded = 0

    for line in txt.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue

        # Poetry-style / generic TOML entries
        if "=" in s and "[" not in s and "]" not in s:
            left, right = s.split("=", 1)
            left = left.strip()
            right = right.strip().strip('"').strip("'")

            if left.lower() in {"python", "name", "version", "description", "readme", "license"}:
                continue

            if right:
                total += 1
                if re.match(r"^\d+(\.\d+)*([\-+].+)?$", right) or "==" in right:
                    exact += 1
                    bounded += 1
                elif any(op in right for op in [">=", "<=", "~=", ">", "<", "^"]):
                    bounded += 1

        # PEP 621 arrays like dependencies = ["x>=1", "y==2"]
        for dep in re.findall(r'"([^"]+)"', s):
            dep = dep.strip()
            if not dep:
                continue

            total += 1
            if "==" in dep:
                exact += 1
                bounded += 1
            elif any(op in dep for op in [">=", "<=", "~=", ">", "<", "^"]):
                bounded += 1

    return total, exact, bounded


def _dependency_pinning_stats(root):
    manifests, lockfiles = _dependency_files(root)

    total = 0
    exact = 0
    bounded = 0

    for p in manifests:
        if p.name.startswith("requirements"):
            t, e, b = _parse_requirements_stats(p)
        elif p.name == "package.json":
            t, e, b = _parse_package_json_stats(p)
        elif p.name == "pyproject.toml":
            t, e, b = _parse_pyproject_stats(p)
        else:
            t, e, b = 0, 0, 0

        total += t
        exact += e
        bounded += b

    has_manifest = 1.0 if manifests else 0.0
    has_lockfile = 1.0 if lockfiles else 0.0

    exact_ratio = (exact / total) if total > 0 else None
    bounded_ratio = (bounded / total) if total > 0 else None

    return {
        "has_manifest": has_manifest,
        "has_lockfile": has_lockfile,
        "dependency_total": total,
        "dependency_exact": exact,
        "dependency_bounded": bounded,
        "exact_ratio": exact_ratio,
        "bounded_ratio": bounded_ratio,
        "manifest_files": [str(p) for p in manifests],
        "lock_files": [str(p) for p in lockfiles],
    }


def _manifest_recency_score(root):
    manifests, lockfiles = _dependency_files(root)
    dep_files = manifests + lockfiles
    if not dep_files:
        return None, None

    import time
    now = time.time()
    days = []

    for p in dep_files:
        code, out, _ = run(["git", "log", "-1", "--format=%ct", "--", str(p)], cwd=root)
        if code == 0 and out.strip().isdigit():
            ts = int(out.strip())
            days.append((now - ts) / 86400.0)
        else:
            try:
                mtime = p.stat().st_mtime
                days.append((now - mtime) / 86400.0)
            except Exception:
                pass

    if not days:
        return None, None

    median_days = sorted(days)[len(days) // 2]
    score = math.exp(-median_days / 365.0)
    return clamp01(score), median_days


def _automation_signals(root):
    dependabot = 1.0 if (root / ".github" / "dependabot.yml").exists() else 0.0
    renovate = 1.0 if exists_any(root, ["renovate.json", "renovate.json5", ".renovaterc", ".renovaterc.json"]) > 0 else 0.0

    dep_scan_ci = _workflow_signal(
        root,
        [
            r"\bpip-audit\b",
            r"\bnpm audit\b",
            r"\byarn audit\b",
            r"\bpnpm audit\b",
            r"\bosv-scanner\b",
            r"\bsafety\b",
            r"\bdependabot\b",
        ],
    )

    return {
        "dependabot": dependabot,
        "renovate": renovate,
        "dep_scan_ci": dep_scan_ci,
    }


def _run_pip_audit_if_possible(root):
    """
    Optional Python-only vulnerability evidence.
    Returns (score, vuln_count, audited_deps).
    """
    if not _is_python_repo(root):
        return None, None, None

    manifests, _ = _dependency_files(root)
    reqs = [p for p in manifests if p.name.startswith("requirements")]
    if not reqs:
        return None, None, None

    audit_counts = []
    audited_deps = []

    for req in reqs:
        code, out, err = run(["pip-audit", "-r", str(req), "--format", "json"], cwd=root)
        if code != 0:
            continue

        try:
            data = json.loads(out)
        except Exception:
            continue

        deps = data if isinstance(data, list) else data.get("dependencies", [])
        vuln_count = 0
        dep_count = 0

        if isinstance(deps, list):
            for dep in deps:
                dep_count += 1
                vulns = dep.get("vulns", []) or dep.get("vulnerabilities", []) or []
                vuln_count += len(vulns)

        if dep_count > 0:
            audit_counts.append(vuln_count / dep_count)
            audited_deps.append(dep_count)

    if not audit_counts:
        return None, None, None

    vuln_density = mean(audit_counts)
    score = math.exp(-vuln_density)
    return clamp01(score), sum(int(x * y) for x, y in zip(audit_counts, audited_deps)), sum(audited_deps)


def compute_sps(root, learned_weights=None, aggregation="geometric"):
    dep = _dependency_pinning_stats(root)

    n_inventory = mean([
        dep["has_manifest"],
        dep["has_lockfile"],
    ])

    n_pinning = mean([
        dep["exact_ratio"],
        dep["bounded_ratio"],
        dep["has_lockfile"],
    ])

    recency_score, median_manifest_days = _manifest_recency_score(root)
    auto = _automation_signals(root)

    n_automation = mean([
        auto["dependabot"],
        auto["renovate"],
        auto["dep_scan_ci"],
        recency_score,
    ])

    pip_audit_score, pip_audit_vulns, pip_audit_audited = _run_pip_audit_if_possible(root)

    n_vuln = mean([
        pip_audit_score,
        auto["dependabot"],
        auto["dep_scan_ci"],
    ])

    blocks = {
        "inventory": n_inventory,
        "pinning": n_pinning,
        "automation": n_automation,
        "vulnerability": n_vuln,
    }
    sps, weights, weights_source, agg_name = _final_score_from_blocks(blocks, learned_weights, aggregation)

    return {
        "score": sps,
        "band": band(sps),
        "details": {
            "SPS": sps,
            "weights": weights,
            "weights_source": weights_source,
            "aggregation": agg_name,
            "N_inventory": n_inventory,
            "N_pinning": n_pinning,
            "N_automation": n_automation,
            "N_vuln": n_vuln,
            "N_inventory.has_manifest": dep["has_manifest"],
            "N_inventory.has_lockfile": dep["has_lockfile"],
            "N_pinning.exact_ratio": dep["exact_ratio"],
            "N_pinning.bounded_ratio": dep["bounded_ratio"],
            "N_pinning.has_lockfile": dep["has_lockfile"],
            "N_automation.dependabot": auto["dependabot"],
            "N_automation.renovate": auto["renovate"],
            "N_automation.dep_scan_ci": auto["dep_scan_ci"],
            "N_automation.recency_score": recency_score,
            "N_vuln.pip_audit_score": pip_audit_score,
            "raw.dependency_total": dep["dependency_total"],
            "raw.dependency_exact": dep["dependency_exact"],
            "raw.dependency_bounded": dep["dependency_bounded"],
            "raw.manifest_files": dep["manifest_files"],
            "raw.lock_files": dep["lock_files"],
            "raw.median_manifest_days": median_manifest_days,
            "raw.pip_audit_vulns": pip_audit_vulns,
            "raw.pip_audit_audited": pip_audit_audited,
        },
    }


# =========================================================
# Stability and reliability software workmanship (SRSW)
# =========================================================

def _doc_hits_with_keywords(root, patterns):
    """
    Count keyword evidence in README/docs/examples/notebooks.
    Returns None when no eligible documentation source is found.
    """
    hits = 0
    seen_doc_source = False
    rx = [re.compile(p, flags=re.I | re.M) for p in patterns]
    allowed_dirs = {"docs", "doc", "examples", "example", "notebooks"}

    for p in iter_files(root):
        if p.suffix.lower() not in {".md", ".rst", ".txt", ".ipynb", ".py", ".toml", ".yml", ".yaml"}:
            continue

        rel = p.relative_to(root)
        rel_parts = {x.lower() for x in rel.parts[:-1]}
        if not (p.name.lower().startswith("readme") or rel_parts.intersection(allowed_dirs)):
            continue

        seen_doc_source = True
        txt = _read_text(p)
        if not txt:
            continue
        for r in rx:
            hits += len(r.findall(txt))

    if not seen_doc_source:
        return None
    return hits


def compute_srsw(root, learned_weights=None, aggregation="geometric"):
    """
    Software Reliability & Stability Workmanship score.

    This indicator does NOT estimate production reliability (MTBF/failure rates).
    It estimates observable evidence of stability engineering practices inside
    the repository.
    """
    # Error handling signals
    e_validate_hits = _count_pattern_hits(
        root,
        [
            r"\bValueError\b",
            r"\bTypeError\b",
            r"\bAssertionError\b",
            r"\braise\b",
            r"\bvalidate\b",
            r"\bvalidator\b",
            r"\bpydantic\b",
            r"\bmarshmallow\b",
            r"\bcerberus\b",
            r"\bassert\b",
        ],
    )
    e_except_hits = _count_pattern_hits(
        root,
        [
            r"\btry\b",
            r"\bexcept\b",
            r"\bcatch\b",
            r"\bfinally\b",
            r"\bexception\b",
            r"\berror handler\b",
        ],
    )
    e_grace_hits = _count_pattern_hits(
        root,
        [
            r"\bfallback\b",
            r"\bdefault\b",
            r"\bgraceful\b",
            r"\bwarn(?:ing)?\b",
            r"\bignore(?:d)? error\b",
            r"\bon[_\s-]?error\b",
        ],
    )
    e_testerr_hits = _count_pattern_hits(
        root,
        [
            r"\bpytest\.raises\b",
            r"\bassertRaises\b",
            r"\btoThrow\b",
            r"\binvalid\b",
            r"\bexception\b",
            r"\berror\b",
            r"\bfail(?:ed|ure)?\b",
        ],
        suffixes={".py", ".js", ".ts", ".java", ".md"},
    )

    # Recovery signals
    r_retry_hits = _count_pattern_hits(
        root,
        [
            r"\bretry\b",
            r"\bbackoff\b",
            r"\btenacity\b",
            r"\bmax[_\s-]?retries\b",
            r"\bexponential backoff\b",
        ],
    )
    r_checkpoint_hits = _count_pattern_hits(
        root,
        [
            r"\bcheckpoint\b",
            r"\bsnapshot\b",
            r"\bsave[_\s-]?state\b",
            r"\bload[_\s-]?state\b",
            r"\bresume\b",
            r"\bpersist(?:ence|ent)?\b",
        ],
    )
    r_rollback_hits = _count_pattern_hits(
        root,
        [
            r"\brollback\b",
            r"\brevert\b",
            r"\bundo\b",
            r"\brestore\b",
            r"\btransaction\b",
        ],
    )
    r_recoverydoc_hits = _doc_hits_with_keywords(
        root,
        [
            r"\brecovery\b",
            r"\bcheckpoint\b",
            r"\brestart\b",
            r"\bresume\b",
            r"\bfault tolerance\b",
            r"\brollback\b",
        ],
    )

    # Release stability signals
    structural_modules = len(detect_modules(root))
    test_files = count_test_files(root)
    rl_tests = (
        clamp01(test_files / max(1, structural_modules)) if structural_modules > 0 else None
    )
    rl_ci = mean([
        detect_ci(root),
        _workflow_signal(
            root,
            [
                r"\bpytest\b",
                r"\bunittest\b",
                r"\bnpm test\b",
                r"\byarn test\b",
                r"\bpnpm test\b",
                r"\bmvn test\b",
                r"\bgradle test\b",
                r"\bbuild\b",
            ],
        ),
    ])
    semver_hits = _count_pattern_hits(
        root,
        [
            r"\bv\d+\.\d+\.\d+\b",
            r"\b\d+\.\d+\.\d+\b",
            r"\brelease\b",
            r"\bversion\b",
        ],
        suffixes={".md", ".rst", ".txt", ".toml", ".yml", ".yaml", ".json"},
    )
    rl_semver = mean([
        exists_any(root, ["CHANGELOG.md", "CHANGELOG.rst", "changelog.md", "changelog.rst"]),
        _signal_hits(semver_hits),
    ])
    rl_regression_hits = _count_pattern_hits(
        root,
        [
            r"\bregression\b",
            r"\bsnapshot\b",
            r"\bgolden\b",
            r"\bbaseline\b",
            r"\bexpected output\b",
            r"\bapproval test\b",
        ],
    )

    # Observability signals
    o_log_hits = _count_pattern_hits(
        root,
        [
            r"\blogging\b",
            r"\blogger\b",
            r"\bstructlog\b",
            r"\bloguru\b",
            r"\bslf4j\b",
            r"\bwinston\b",
        ],
    )
    o_metrics_hits = _count_pattern_hits(
        root,
        [
            r"\bprometheus\b",
            r"\bmetric(?:s)?\b",
            r"\bcounter\b",
            r"\bgauge\b",
            r"\bhistogram\b",
            r"\bopentelemetry\b",
            r"\bstatsd\b",
        ],
    )
    o_health_hits = _count_pattern_hits(
        root,
        [
            r"/health\b",
            r"\bhealth[_\s-]?check\b",
            r"\bheartbeat\b",
            r"\breadyz\b",
            r"\blivez\b",
            r"\bself[-_\s]?test\b",
        ],
    )
    o_alerts_hits = _count_pattern_hits(
        root,
        [
            r"\balert(?:s)?\b",
            r"\bsentry\b",
            r"\bpagerduty\b",
            r"\bnotify on failure\b",
            r"\bwebhook\b",
            r"\bincident\b",
        ],
    )

    # Repository-relative reference for hit-based signals
    major_raw_families = [
        e_validate_hits + e_except_hits + e_grace_hits + e_testerr_hits,
        r_retry_hits + r_checkpoint_hits + r_rollback_hits + (r_recoverydoc_hits or 0),
        semver_hits + rl_regression_hits,
        o_log_hits + o_metrics_hits + o_health_hits + o_alerts_hits,
    ]
    hit_ref = max(1.0, *major_raw_families)

    # Sub-signals (hit-based -> log normalized, bounded signals kept as-is)
    e_validate = _log_repo_signal(e_validate_hits, hit_ref)
    e_except = _log_repo_signal(e_except_hits, hit_ref)
    e_grace = _log_repo_signal(e_grace_hits, hit_ref)
    e_testerr = _log_repo_signal(e_testerr_hits, hit_ref)
    n_err = mean([e_validate, e_except, e_grace, e_testerr])

    r_retry = _log_repo_signal(r_retry_hits, hit_ref)
    r_checkpoint = _log_repo_signal(r_checkpoint_hits, hit_ref)
    r_rollback = _log_repo_signal(r_rollback_hits, hit_ref)
    r_recoverydoc = _log_repo_signal(r_recoverydoc_hits, hit_ref) if r_recoverydoc_hits is not None else None
    n_rec = mean([r_retry, r_checkpoint, r_rollback, r_recoverydoc])

    rl_regression = _log_repo_signal(rl_regression_hits, hit_ref)
    n_rel = mean([rl_tests, rl_ci, rl_semver, rl_regression])

    o_log = _log_repo_signal(o_log_hits, hit_ref)
    o_metrics = _log_repo_signal(o_metrics_hits, hit_ref)
    o_health = _log_repo_signal(o_health_hits, hit_ref)
    o_alerts = _log_repo_signal(o_alerts_hits, hit_ref)
    n_obs = mean([o_log, o_metrics, o_health, o_alerts])

    blocks = {
        "error_handling": n_err,
        "recovery": n_rec,
        "release_stability": n_rel,
        "observability": n_obs,
    }
    srsw, weights, weights_source, agg_name = _final_score_from_blocks(blocks, learned_weights, aggregation)

    return {
        "score": srsw,
        "band": band(srsw),
        "details": {
            "SRSW": srsw,
            "weights": weights,
            "weights_source": weights_source,
            "aggregation": agg_name,
            "N_err": n_err,
            "N_rec": n_rec,
            "N_rel": n_rel,
            "N_obs": n_obs,
            "N_err.E_validate": e_validate,
            "N_err.E_except": e_except,
            "N_err.E_grace": e_grace,
            "N_err.E_testerr": e_testerr,
            "N_rec.R_retry": r_retry,
            "N_rec.R_checkpoint": r_checkpoint,
            "N_rec.R_rollback": r_rollback,
            "N_rec.R_recoverydoc": r_recoverydoc,
            "N_rel.RL_tests": rl_tests,
            "N_rel.RL_ci": rl_ci,
            "N_rel.RL_semver": rl_semver,
            "N_rel.RL_regression": rl_regression,
            "N_obs.O_log": o_log,
            "N_obs.O_metrics": o_metrics,
            "N_obs.O_health": o_health,
            "N_obs.O_alerts": o_alerts,
            "raw.E_validate_hits": e_validate_hits,
            "raw.E_except_hits": e_except_hits,
            "raw.E_grace_hits": e_grace_hits,
            "raw.E_testerr_hits": e_testerr_hits,
            "raw.R_retry_hits": r_retry_hits,
            "raw.R_checkpoint_hits": r_checkpoint_hits,
            "raw.R_rollback_hits": r_rollback_hits,
            "raw.R_recoverydoc_hits": r_recoverydoc_hits,
            "raw.RL_regression_hits": rl_regression_hits,
            "raw.RL_semver_hits": semver_hits,
            "raw.O_log_hits": o_log_hits,
            "raw.O_metrics_hits": o_metrics_hits,
            "raw.O_health_hits": o_health_hits,
            "raw.O_alerts_hits": o_alerts_hits,
            "raw.hit_ref": hit_ref,
            "raw.structural_modules": structural_modules,
            "raw.test_files": test_files,
            "method_note": "SRSW measures observable stability-engineering evidence in the repository, not operational reliability in production.",
            "missingness_note": "Missing documentation evidence is represented as None when no eligible doc source exists; interpret missing signals cautiously.",
        },
    }


def learn_eq_family_weights(dataset):
    """Learn top-level EQ family weights over headline indicators (CQI..SRSW)."""
    keys = ["CQI", "ARS", "PER", "SPS", "SRSW"]
    rows = []
    for row in dataset:
        src = row.get("EQ", row)
        if isinstance(src, dict) and "scores" in src:
            src = src["scores"]
        out = {}
        for k in keys:
            blk = src.get(k) if isinstance(src, dict) else None
            val = None
            if isinstance(blk, dict):
                val = blk.get("score")
                if val is None and k in blk and isinstance(blk[k], (int, float)):
                    val = blk[k]
            if isinstance(val, (int, float)):
                out[k] = float(val) / 100.0 if float(val) > 1.0 else float(val)
            else:
                out[k] = None
        rows.append(out)
    return {"weights": _norm_weights_from_rows(rows, keys), "n_repos": len(dataset)}


def compute_eq(root, learned_weights=None, aggregation="geometric"):
    """
    Engineering Quality family composite over CQI, ARS, PER, SPS, SRSW.
    """
    cqi = compute_cqi(root, learned_weights=None, aggregation=aggregation)
    ars = compute_ars(root, learned_weights=None, aggregation=aggregation)
    per = compute_per(root, learned_weights=None, aggregation=aggregation)
    sps = compute_sps(root, learned_weights=None, aggregation=aggregation)
    srsw = compute_srsw(root, learned_weights=None, aggregation=aggregation)

    def _to_block(res):
        s = res.get("score") if isinstance(res, dict) else None
        if s is None:
            return None
        return float(s) / 100.0

    blocks = {
        "CQI": _to_block(cqi),
        "ARS": _to_block(ars),
        "PER": _to_block(per),
        "SPS": _to_block(sps),
        "SRSW": _to_block(srsw),
    }
    family_weights = None
    if isinstance(learned_weights, dict):
        family_weights = learned_weights.get("EQ") or learned_weights.get("eq")
        if family_weights is None and all(k in learned_weights for k in blocks):
            family_weights = learned_weights
    score, weights, weights_source, agg_name = _final_score_from_blocks(blocks, family_weights, aggregation)
    return {
        "score": score,
        "band": band(score),
        "aggregation": agg_name,
        "details": {
            "EQ": score,
            "CQI": cqi.get("score"),
            "ARS": ars.get("score"),
            "PER": per.get("score"),
            "SPS": sps.get("score"),
            "SRSW": srsw.get("score"),
            "weights": weights,
            "weights_source": weights_source,
            "scores": {"CQI": cqi, "ARS": ars, "PER": per, "SPS": sps, "SRSW": srsw},
            "missingness_note": "Unavailable headline indicators remain None and are excluded with weight renormalization.",
            "method_note": "EQ aggregates five engineering-quality headline indicators with the same aggregation options as each sub-indicator.",
        },
    }


def main():
    ap = argparse.ArgumentParser(description="RECOPS Engineering Quality scores: CQI, ARS, PER, SPS, SRSW, EQ")
    ap.add_argument("github_url", nargs="?", help="GitHub repository URL")
    ap.add_argument("--local", dest="local_repo", help="Analyze an already-local repository path instead of cloning")
    ap.add_argument("--output-json", default=None)
    ap.add_argument("--aggregation", choices=["geometric", "sum"], default="geometric")
    ap.add_argument(
        "--score",
        choices=["cqi", "ars", "per", "sps", "srsw", "eq", "all"],
        default="all",
        help="Which EQ score to compute",
    )
    ap.add_argument(
        "--weights-json",
        default=None,
        help="Optional corpus-learned EQ weights JSON. If absent, repository-level data-driven fallback weights are used.",
    )
    args = ap.parse_args()

    if not args.local_repo and not args.github_url:
        ap.error("Provide either a GitHub repo_url or --local PATH")

    if args.local_repo:
        repo = Path(args.local_repo).expanduser().resolve()
        if not repo.exists() or not repo.is_dir():
            print(json.dumps({"error": f"Local path not found: {repo}"}, indent=2))
            sys.exit(1)
    else:
        try:
            repo = clone_or_download(args.github_url)
        except Exception as e:
            err = {
                "repository": args.github_url,
                "error": str(e),
                "hint": "Verify the GitHub URL and repository visibility (public/private), then retry.",
            }
            print(json.dumps(err, indent=2))
            sys.exit(1)

    sc = args.score.lower()
    w_all = _load_learned_weights(args.weights_json, None) if args.weights_json else None

    if sc == "cqi":
        out = compute_cqi(repo, learned_weights=_load_learned_weights(args.weights_json, "CQI"), aggregation=args.aggregation)
    elif sc == "ars":
        out = compute_ars(repo, learned_weights=_load_learned_weights(args.weights_json, "ARS"), aggregation=args.aggregation)
    elif sc == "per":
        out = compute_per(repo, learned_weights=_load_learned_weights(args.weights_json, "PER"), aggregation=args.aggregation)
    elif sc == "sps":
        out = compute_sps(repo, learned_weights=_load_learned_weights(args.weights_json, "SPS"), aggregation=args.aggregation)
    elif sc == "srsw":
        out = compute_srsw(repo, learned_weights=_load_learned_weights(args.weights_json, "SRSW"), aggregation=args.aggregation)
    elif sc == "eq":
        out = compute_eq(repo, learned_weights=w_all, aggregation=args.aggregation)
    else:
        out = {
            "repository_path": str(repo),
            "repository_url": args.github_url,
            "scores": {
                "CQI": compute_cqi(repo, learned_weights=None, aggregation=args.aggregation),
                "ARS": compute_ars(repo, learned_weights=None, aggregation=args.aggregation),
                "PER": compute_per(repo, learned_weights=None, aggregation=args.aggregation),
                "SPS": compute_sps(repo, learned_weights=None, aggregation=args.aggregation),
                "SRSW": compute_srsw(repo, learned_weights=None, aggregation=args.aggregation),
                "EQ": compute_eq(repo, learned_weights=None, aggregation=args.aggregation),
            },
            "note": "With --score all, --weights-json is ignored to avoid applying one weight file to multiple formulas.",
        }

    print(json.dumps(out, indent=2))

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
