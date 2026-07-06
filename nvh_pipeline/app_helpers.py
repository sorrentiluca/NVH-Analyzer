"""
Pure helpers for the Streamlit front end — no Streamlit import, so they are
unit-testable and reusable from any UI.
"""

from __future__ import annotations

import datetime
import json
import os
from typing import Iterable, Optional


def csv_list(text: str) -> list:
    """'a, b , c' → ['a', 'b', 'c'] (empty tokens dropped)."""
    if not text:
        return []
    return [tok.strip() for tok in str(text).split(",") if tok.strip()]


def as_str_list(value) -> list:
    """Accept a list/tuple or a comma-separated string; return clean strings."""
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    return csv_list(value)


def generate_run_name(parts: Iterable, rpms: Iterable) -> str:
    """A readable, sortable run-folder name: ``YYYY-MM-DD_HHMM_<parts>_<rpms>``.

    Parts are capped at three (a trailing ``+`` marks truncation); speeds
    collapse to a range when there are more than three.  Sorting the speeds
    makes the name independent of selection order.
    """
    now = datetime.datetime.now()
    stamp = now.strftime("%Y-%m-%d_%H%M")

    plist = [str(p) for p in (parts or []) if str(p)]
    if not plist:
        parts_str = "noparts"
    elif len(plist) <= 3:
        parts_str = "-".join(plist)
    else:
        parts_str = "-".join(plist[:3]) + "+"

    try:
        rlist = sorted({int(float(r)) for r in (rpms or [])})
    except (TypeError, ValueError):
        rlist = []
    if not rlist:
        rpms_str = "norpm"
    elif len(rlist) <= 3:
        rpms_str = "-".join(str(r) for r in rlist) + "rpm"
    else:
        rpms_str = f"{rlist[0]}-{rlist[-1]}rpm"

    return f"{stamp}_{parts_str}_{rpms_str}"


def scan_past_runs(results_root: str) -> list:
    """Find completed runs (sub-folders holding a manifest.json) under a
    results root, newest first.

    Each entry: ``{name, manifest, generated, stages_ok, stages_total,
    parts, rpms}``.  parts/rpms come from the run's saved nvh_config.json
    (the filter the user actually chose, falling back to the campaign lists).
    """
    if not results_root or not os.path.isdir(results_root):
        return []

    runs = []
    for name in os.listdir(results_root):
        run_dir = os.path.join(results_root, name)
        manifest_path = os.path.join(run_dir, "manifest.json")
        if not os.path.isdir(run_dir) or not os.path.isfile(manifest_path):
            continue
        try:
            with open(manifest_path, "r", encoding="utf-8") as fh:
                manifest = json.load(fh)
        except Exception:
            continue

        stages = manifest.get("stages") or []
        parts, rpms = [], []
        cfg_path = os.path.join(run_dir, "nvh_config.json")
        if os.path.isfile(cfg_path):
            try:
                with open(cfg_path, "r", encoding="utf-8") as fh:
                    c = json.load(fh)
                parts = list(c.get("only_parts") or c.get("parts") or [])
                rpms = [int(r) for r in
                        (c.get("only_rpms") or c.get("rpms") or [])]
            except Exception:
                pass

        runs.append({
            "name": name,
            "manifest": manifest_path,
            "generated": manifest.get("generated", ""),
            "stages_ok": sum(1 for s in stages if s.get("status") == "ok"),
            "stages_total": len(stages),
            "parts": parts,
            "rpms": rpms,
        })

    runs.sort(key=lambda r: (r["generated"], r["name"]), reverse=True)
    return runs


def default_for(saved: str, options: list, *hints: str) -> Optional[str]:
    """Pick the best default for a column-role selector.

    Order of preference: the saved/configured value when it is still a valid
    option → the first option whose name contains a hint (case-insensitive,
    hints tried in order) → the first option.  With no options at all the
    saved value is returned unchanged (free-text mode).
    """
    if not options:
        return saved
    if saved in options:
        return saved
    for hint in hints:
        h = str(hint).lower()
        for opt in options:
            if h in str(opt).lower():
                return opt
    return options[0]
