"""
dataset — one shared DuckDB-over-parquet engine for every analysis stage.

EP_segment writes one ``*_segments.parquet`` per raw CSV.  Every downstream
stage reads through this module so:

  * column selection is pushed down into the parquet reader (a stage that
    needs 6 metadata columns never materialises the 100 Hz–100 kHz signal
    columns),
  * per-segment reductions (RMS, means, durations) run as ONE SQL GROUP BY
    over all files instead of a Python loop per segment, and
  * there is exactly one place to change if the storage layout evolves.

DuckDB is a single self-contained wheel (no server, no admin).  When it is not
installed the read paths transparently fall back to pandas/pyarrow;
``reduce_segments`` requires DuckDB (its callers pass raw SQL expressions).
"""

from __future__ import annotations

import glob
import os
import threading
from dataclasses import dataclass, field
from typing import Iterator, Optional

import numpy as np
import pandas as pd

SEG_GLOB = "*_segments.parquet"

# Metadata columns EP_segment stamps onto every row of a segment parquet.
META_COLS = ("part", "material_type", "rpm_category", "trial",
             "direction", "segment_id")

try:
    import duckdb as _duckdb
except ImportError:                                   # pragma: no cover
    _duckdb = None

# One connection per process, guarded for the ThreadPoolExecutor stages.
_CONN = None
_CONN_LOCK = threading.Lock()


def _connect():
    """The process-wide DuckDB connection (None if DuckDB is unavailable)."""
    global _CONN
    if _duckdb is None:
        return None
    with _CONN_LOCK:
        if _CONN is None:
            _CONN = _duckdb.connect(database=":memory:")
        return _CONN


def _sql(query: str, params: Optional[list] = None) -> pd.DataFrame:
    con = _connect()
    with _CONN_LOCK:
        return con.execute(query, params or []).df()


def _resolve(seg_dir: str, path: str) -> str:
    """Accept absolute paths or basenames relative to the segment folder."""
    return path if os.path.isabs(path) else os.path.join(seg_dir, path)


def _quote_path(p: str) -> str:
    return "'" + p.replace("\\", "/").replace("'", "''") + "'"


# ─────────────────────────────────────────────────────────────────────────────
#  Listing / schema
# ─────────────────────────────────────────────────────────────────────────────

def list_segment_files(seg_dir: str) -> list:
    """Sorted absolute paths of all segment parquets ([] if the folder is
    missing) — the canonical 'what data do we have' call."""
    if not seg_dir or not os.path.isdir(seg_dir):
        return []
    return sorted(os.path.abspath(p)
                  for p in glob.glob(os.path.join(seg_dir, SEG_GLOB)))


# Schema cache keyed by (path, mtime): stages ask for a file's columns two or
# three times per file per run; the footer read is cheap but not free, and the
# mtime key keeps the cache correct if a file is rewritten mid-session.
_SCHEMA_CACHE: dict = {}


def file_columns(seg_dir: str, path: str) -> list:
    """Column names of one parquet from its footer — no data pages read."""
    full = _resolve(seg_dir, path)
    try:
        key = (full, os.path.getmtime(full))
    except OSError:
        key = None
    if key is not None and key in _SCHEMA_CACHE:
        return list(_SCHEMA_CACHE[key])
    try:
        import pyarrow.parquet as pq
        cols = list(pq.read_schema(full).names)
    except Exception:
        try:
            cols = list(pd.read_parquet(full).columns)
        except Exception:
            cols = []
    if key is not None and cols:
        if len(_SCHEMA_CACHE) > 4096:          # bound the cache for long sessions
            _SCHEMA_CACHE.clear()
        _SCHEMA_CACHE[key] = cols
    return list(cols)


# ─────────────────────────────────────────────────────────────────────────────
#  Reading
# ─────────────────────────────────────────────────────────────────────────────

def read_file(seg_dir: str, path: str, columns: Optional[list] = None,
              where: Optional[str] = None) -> pd.DataFrame:
    """Read one segment parquet with column pushdown and an optional SQL
    predicate (e.g. ``where="segment_id = 3"``).

    Requested columns that don't exist in the file are silently dropped from
    the selection so callers can probe for optional channels.
    """
    full = _resolve(seg_dir, path)
    if columns is not None:
        avail = set(file_columns(seg_dir, full))
        columns = [c for c in dict.fromkeys(columns) if c in avail]
        if not columns:
            return pd.DataFrame()

    con = _connect()
    if con is not None:
        cols_sql = ("*" if columns is None
                    else ", ".join('"' + c.replace('"', '""') + '"'
                                   for c in columns))
        q = f"SELECT {cols_sql} FROM read_parquet({_quote_path(full)})"
        if where:
            q += f" WHERE {where}"
        return _sql(q)

    df = pd.read_parquet(full, columns=columns)
    if where:
        df = df.query(where.replace("=", "==") if "==" not in where else where)
    return df


def iter_file_segments(seg_dir: str, path: str,
                       columns: Optional[list] = None
                       ) -> Iterator[tuple[int, pd.DataFrame]]:
    """Yield ``(segment_id, segment_df)`` for one file, in segment order.

    The file is materialised once and split — peak memory stays at one file,
    which is the unit the stage loops already stream by.
    """
    df = read_file(seg_dir, path, columns=columns)
    if df.empty or "segment_id" not in df.columns:
        return
    for seg_id, seg in df.groupby("segment_id", sort=True):
        yield int(seg_id), seg


def segment_catalog(seg_dir: str) -> Optional[pd.DataFrame]:
    """One row per segment across the whole dataset:
    part, material_type, rpm_category, trial, direction, segment_id,
    source_file (basename).  Reads only metadata columns."""
    files = list_segment_files(seg_dir)
    if not files:
        return None

    con = _connect()
    if con is not None:
        lst = "[" + ", ".join(_quote_path(f) for f in files) + "]"
        try:
            df = _sql(
                "SELECT part, material_type, rpm_category, trial, direction, "
                "       segment_id, "
                "       regexp_extract(filename, '[^/\\\\]+$') AS source_file "
                f"FROM read_parquet({lst}, filename=true, union_by_name=true) "
                "GROUP BY ALL ORDER BY source_file, segment_id")
            return df
        except Exception:
            pass                              # fall through to pandas

    rows = []
    for f in files:
        try:
            df = pd.read_parquet(
                f, columns=[c for c in META_COLS
                            if c in file_columns(seg_dir, f)])
        except Exception:
            continue
        if "segment_id" not in df.columns:
            continue
        first = df.groupby("segment_id", sort=True).head(1).copy()
        first["source_file"] = os.path.basename(f)
        rows.append(first)
    if not rows:
        return None
    return pd.concat(rows, ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Reductions
# ─────────────────────────────────────────────────────────────────────────────

def reduce_segments(seg_dir: str, agg_exprs: dict,
                    where: Optional[str] = None) -> pd.DataFrame:
    """One SQL GROUP BY per segment over ALL parquet files at once.

    ``agg_exprs`` maps output column name → DuckDB aggregate expression, e.g.
    ``{'rms_x': 'SQRT(AVG("X Accel (m/s2)" * "X Accel (m/s2)"))'}``.

    Returns one row per (source_file, segment_id) carrying the metadata
    columns (part, material_type, rpm_category, trial, direction, segment_id,
    source_file) plus the requested aggregates.  DuckDB skips NULL/NaN inside
    AVG, matching the previous per-segment ``np.isfinite`` filtering.
    """
    if _duckdb is None:
        raise ImportError(
            "duckdb is required for reduce_segments() — pip install duckdb")
    files = list_segment_files(seg_dir)
    if not files:
        return pd.DataFrame()

    aggs = ", ".join(
        f'{expr} AS "{name}"' for name, expr in agg_exprs.items())
    lst = "[" + ", ".join(_quote_path(f) for f in files) + "]"
    q = (
        "SELECT part, material_type, rpm_category, trial, direction, "
        "       segment_id, "
        "       regexp_extract(filename, '[^/\\\\]+$') AS source_file, "
        f"       {aggs} "
        f"FROM read_parquet({lst}, filename=true, union_by_name=true) "
    )
    if where:
        q += f"WHERE {where} "
    q += ("GROUP BY part, material_type, rpm_category, trial, direction, "
          "segment_id, filename ORDER BY source_file, segment_id")
    return _sql(q)


# ─────────────────────────────────────────────────────────────────────────────
#  Data quality
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DataQuality:
    """Dataset-health summary consumed by the Review tab's Data Quality view."""
    n_files: int = 0
    n_segments: int = 0
    parts: list = field(default_factory=list)
    rpms: list = field(default_factory=list)
    counts: pd.DataFrame = field(default_factory=pd.DataFrame)
    missing_combos: list = field(default_factory=list)   # requested but absent
    imbalanced: list = field(default_factory=list)       # POS/NEG unequal
    flags: list = field(default_factory=list)            # human-readable notes


def data_quality(seg_dir: str,
                 requested_parts: tuple = (),
                 requested_rpms: tuple = ()) -> DataQuality:
    """Audit segment coverage against what the user asked for.

    Flags (part, rpm) combinations that were requested but produced no
    segments, and strokes whose POS/NEG actuation counts are unbalanced —
    both are acquisition problems that silently bias comparisons.
    """
    dq = DataQuality()
    cat = segment_catalog(seg_dir)
    if cat is None or cat.empty:
        dq.flags.append("No segmented data found.")
        return dq

    dq.n_files = int(cat["source_file"].nunique())
    dq.n_segments = int(len(cat))
    dq.parts = sorted(str(p) for p in cat["part"].dropna().unique())
    dq.rpms = sorted(int(r) for r in cat["rpm_category"].dropna().unique())

    counts = (cat.groupby(["part", "rpm_category"])
              .agg(n_segments=("segment_id", "count"),
                   n_pos=("direction", lambda d: int((d == "POS").sum())),
                   n_neg=("direction", lambda d: int((d == "NEG").sum())))
              .reset_index())
    dq.counts = counts

    have = {(str(r["part"]), int(r["rpm_category"]))
            for _, r in counts.iterrows()}
    for p in (requested_parts or ()):
        for r in (requested_rpms or ()):
            if (str(p), int(r)) not in have:
                dq.missing_combos.append((str(p), int(r)))
    if dq.missing_combos:
        dq.flags.append(
            f"{len(dq.missing_combos)} requested part×speed combination(s) "
            "have no data: "
            + ", ".join(f"{p}@{r}rpm" for p, r in dq.missing_combos[:8])
            + ("…" if len(dq.missing_combos) > 8 else ""))

    imb = counts[counts["n_pos"] != counts["n_neg"]]
    for _, r in imb.iterrows():
        dq.imbalanced.append((str(r["part"]), int(r["rpm_category"]),
                              int(r["n_pos"]), int(r["n_neg"])))
    if dq.imbalanced:
        dq.flags.append(
            f"{len(dq.imbalanced)} part×speed group(s) have unbalanced "
            "POS/NEG actuation counts — check the per-file metadata.")
    return dq
