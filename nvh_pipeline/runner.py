"""
runner — executes the analysis stages and writes manifest.json.

The twelve EP_*.py scripts are grouped into six user-facing stages so the UI
and CLI expose one checkbox per deliverable, not one per script:

    segment     EP_segment                                    (must run first)
    vibration   EP_loudness, EP_loudness_plot, EP_loudness_by_sample
    efficiency  EP_torque, EP_efficiency
    order       EP_order_analysis
    envelope    EP_Envelope, EP_bandpass
    report      EP_Sample_summary, EP_actuation_plot, EP_characteristics

After segmentation the analysis groups are independent (they all read the
segmented parquets and write disjoint output folders), so ``run_all`` executes
them in PARALLEL worker processes — wall-clock time approaches the slowest
group instead of the sum.  Each stage's outputs (PNG/CSV/HTML/PDF produced
during its run) are collected into the manifest record so the app / FAMOS can
display exactly what a run created.

A stage failure never aborts the pipeline: the record carries status 'error'
plus the traceback, and the remaining stages still run.
"""

from __future__ import annotations

import importlib
import io
import json
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
from typing import Callable, Optional

from .config import PipelineConfig, set_active

# Ordered stage registry.  'label' is the user-facing name; 'modules' the
# EP_* scripts run in order; 'outputs' the folders (under output_root) whose
# fresh files become the stage's manifest entry.
STAGES: dict = {
    "segment": {
        "label": "Segmentation",
        "modules": ["EP_segment"],
        "outputs": ["output_seg"],
    },
    "vibration": {
        "label": "Vibration Level",
        "modules": ["EP_loudness", "EP_loudness_plot", "EP_loudness_by_sample"],
        "outputs": ["output_loudness", "output_loudness_samples"],
    },
    "efficiency": {
        "label": "Mechanical Efficiency",
        "modules": ["EP_torque", "EP_efficiency"],
        "outputs": ["output_torque", "output_efficiency"],
    },
    "order": {
        "label": "Order Analysis",
        "modules": ["EP_order_analysis"],
        "outputs": ["output_order"],
    },
    "envelope": {
        "label": "Envelope Analysis",
        "modules": ["EP_Envelope", "EP_bandpass"],
        "outputs": ["output_envelope"],
    },
    "report": {
        "label": "Handover Report",
        "modules": ["EP_Sample_summary", "EP_actuation_plot",
                    "EP_characteristics"],
        "outputs": ["output_handover", "output_actuation_plot"],
    },
}

# File types that count as run results in the manifest.
_RESULT_EXTS = (".png", ".csv", ".html", ".pdf", ".json")
# Internal bookkeeping files that must not appear as results.
_RESULT_SKIP = {"manifest.json", "_progress.json", "nvh_config.json"}


def _collect_outputs(base_dir: str, subdirs: list, since: float) -> list:
    """All result files under base_dir/<subdir> modified at/after ``since``."""
    found = []
    for sub in subdirs:
        root = base_dir if sub in (".", "") else os.path.join(base_dir, sub)
        if not os.path.isdir(root):
            continue
        for dirpath, _dirnames, filenames in os.walk(root):
            for fn in filenames:
                if fn in _RESULT_SKIP:
                    continue
                if os.path.splitext(fn)[1].lower() not in _RESULT_EXTS:
                    continue
                full = os.path.join(dirpath, fn)
                try:
                    if os.path.getmtime(full) >= since:
                        found.append(str(os.path.abspath(full)))
                except OSError:
                    continue
    return sorted(found)


def _run_modules(stage: str, cfg: PipelineConfig) -> dict:
    """Run one stage group in THIS process and return its manifest record."""
    spec = STAGES[stage]
    set_active(cfg)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.makedirs(cfg.output_root, exist_ok=True)

    t0 = time.time()
    since = t0 - 1.0            # tolerate coarse filesystem mtime granularity
    buf = io.StringIO()
    record = {"stage": stage, "label": spec["label"], "status": "ok",
              "files": [], "error": None, "log": "", "elapsed_s": 0.0}
    try:
        with redirect_stdout(buf), redirect_stderr(buf):
            for mod_name in spec["modules"]:
                mod = importlib.import_module(mod_name)
                mod.main()
    except Exception:
        record["status"] = "error"
        record["error"] = traceback.format_exc()
    record["log"] = buf.getvalue()[-20000:]
    record["files"] = _collect_outputs(cfg.output_root, spec["outputs"], since)
    record["elapsed_s"] = round(time.time() - t0, 2)
    return record


def run_stage(stage: str, cfg: PipelineConfig) -> dict:
    """Run one stage group serially; unknown stages return an error record."""
    if stage not in STAGES:
        return {"stage": stage, "label": stage, "status": "error",
                "files": [], "log": "",
                "error": f"unknown stage {stage!r} — valid: {list(STAGES)}",
                "elapsed_s": 0.0}
    return _run_modules(stage, cfg)


def _worker(stage: str, cfg_dict: dict, config_path: Optional[str]) -> dict:
    """ProcessPoolExecutor entry point (must be picklable, module-level).

    The worker rebuilds the config from the serialized dict; NVH_CONFIG is
    also exported so any stage that re-discovers the file finds the same one.
    """
    if config_path:
        os.environ["NVH_CONFIG"] = config_path
    os.environ["MPLBACKEND"] = "Agg"
    return _run_modules(stage, PipelineConfig.from_dict(cfg_dict))


def _canonical(stages) -> list:
    chosen = list(stages) if stages else list(STAGES)
    return [s for s in STAGES if s in chosen]


def run_all(cfg: PipelineConfig,
            stages=None,
            progress_cb: Optional[Callable[[dict], None]] = None,
            config_path: Optional[str] = None,
            parallel: bool = True) -> dict:
    """Run the requested stages and write ``<output_root>/manifest.json``.

    ``segment`` (when selected) always runs first and alone — everything else
    reads its parquets.  The remaining groups run in parallel worker processes
    (``parallel=False`` forces serial, e.g. for debugging).

    ``progress_cb`` receives event dicts::

        {"phase": "start"|"done", "stage": s, "label": L, "i": k, "n": N}
        {"phase": "parallel_start", "labels": [...], "i": k, "n": N}

    'done' events additionally carry ``status``.  The same protocol feeds the
    app's live progress file.
    """
    chosen = _canonical(stages)
    n = len(chosen)
    set_active(cfg)
    os.makedirs(cfg.output_root, exist_ok=True)

    def _emit(ev: dict) -> None:
        if progress_cb:
            try:
                progress_cb(ev)
            except Exception:
                pass

    t_start = time.time()
    records: dict[str, dict] = {}
    i = 0

    serial_head = [s for s in chosen if s == "segment"]
    parallel_pool = [s for s in chosen if s != "segment"]
    # 'report' aggregates the other stages' CSVs — run it after them.
    tail = [s for s in parallel_pool if s == "report"]
    parallel_pool = [s for s in parallel_pool if s != "report"]

    for stage in serial_head:
        i += 1
        _emit({"phase": "start", "stage": stage,
               "label": STAGES[stage]["label"], "i": i, "n": n})
        rec = run_stage(stage, cfg)
        records[stage] = rec
        _emit({"phase": "done", "stage": stage,
               "label": STAGES[stage]["label"], "i": i, "n": n,
               "status": rec["status"]})

    if parallel_pool:
        labels = [STAGES[s]["label"] for s in parallel_pool]
        if parallel and len(parallel_pool) > 1:
            _emit({"phase": "parallel_start", "labels": labels,
                   "i": i + 1, "n": n})
            try:
                cfg_dict = cfg.to_dict()
                with ProcessPoolExecutor(
                        max_workers=min(len(parallel_pool), 4)) as pool:
                    futs = {s: pool.submit(_worker, s, cfg_dict, config_path)
                            for s in parallel_pool}
                    for s, fut in futs.items():
                        i += 1
                        rec = fut.result()
                        records[s] = rec
                        _emit({"phase": "done", "stage": s,
                               "label": STAGES[s]["label"], "i": i, "n": n,
                               "status": rec["status"]})
            except Exception:
                # Pool failure (pickling, spawn limits, sandbox): run serially.
                for s in parallel_pool:
                    if s in records:
                        continue
                    i += 1
                    _emit({"phase": "start", "stage": s,
                           "label": STAGES[s]["label"], "i": i, "n": n})
                    rec = run_stage(s, cfg)
                    records[s] = rec
                    _emit({"phase": "done", "stage": s,
                           "label": STAGES[s]["label"], "i": i, "n": n,
                           "status": rec["status"]})
        else:
            for s in parallel_pool:
                i += 1
                _emit({"phase": "start", "stage": s,
                       "label": STAGES[s]["label"], "i": i, "n": n})
                rec = run_stage(s, cfg)
                records[s] = rec
                _emit({"phase": "done", "stage": s,
                       "label": STAGES[s]["label"], "i": i, "n": n,
                       "status": rec["status"]})

    for stage in tail:
        i += 1
        _emit({"phase": "start", "stage": stage,
               "label": STAGES[stage]["label"], "i": i, "n": n})
        rec = run_stage(stage, cfg)
        records[stage] = rec
        _emit({"phase": "done", "stage": stage,
               "label": STAGES[stage]["label"], "i": i, "n": n,
               "status": rec["status"]})

    ordered = [records[s] for s in chosen if s in records]
    manifest = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_dir": cfg.data_dir,
        "output_root": os.path.abspath(cfg.output_root),
        "timing": {
            "total_s": round(time.time() - t_start, 2),
            "stages": {r["stage"]: r["elapsed_s"] for r in ordered},
        },
        "stages": ordered,
    }
    manifest_path = os.path.join(cfg.output_root, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    n_ok = sum(1 for r in ordered if r["status"] == "ok")
    print(f"\nPIPELINE DONE — {n_ok}/{len(ordered)} stage(s) ok "
          f"-> {manifest_path}")
    for r in ordered:
        mark = "ok " if r["status"] == "ok" else "ERR"
        print(f"  [{mark}] {r['label']:<22} {r['elapsed_s']:>6.1f}s  "
              f"{len(r['files'])} file(s)")
        if r["status"] != "ok" and r["error"]:
            last = r["error"].strip().splitlines()[-1]
            print(f"        {last}")
    return manifest
