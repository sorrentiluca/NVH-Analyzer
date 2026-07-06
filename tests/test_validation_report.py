"""
Opt-in gate for the visual validation report.

Marked ``report`` and excluded from the default ``pytest`` run (see pytest.ini
``addopts = -m "not report"``).  Run it explicitly with:

    pytest -m report

It executes ``tests/validation_report.py`` in a SUBPROCESS — a fresh interpreter
with real matplotlib, isolated from any matplotlib stubbing other test modules
install — and asserts the report built cleanly with every case PASS.
"""
from __future__ import annotations

import csv
import os
import subprocess
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_REPORT = os.path.join(_HERE, 'validation_report.py')


def _long(path: str) -> str:
    """Windows extended-length path so stat/open survive a >260-char repo root
    (this project lives deep under OneDrive — a plain os.path.isfile on the long
    path hits MAX_PATH and falsely reports the file missing)."""
    ap = os.path.abspath(path)
    if os.name == 'nt' and not ap.startswith('\\\\?\\'):
        return '\\\\?\\' + ap
    return ap


@pytest.mark.report
def test_validation_report_builds_and_all_pass():
    env = dict(os.environ, MPLBACKEND='Agg', PYTHONIOENCODING='utf-8')
    proc = subprocess.run([sys.executable, _REPORT],
                          capture_output=True, text=True, env=env, cwd=_ROOT)
    assert proc.returncode == 0, (
        f"validation_report.py reported {proc.returncode} failed case(s)\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")

    outdir = os.path.join(_ROOT, 'output_validation')
    pdf = _long(os.path.join(outdir, 'validation_report.pdf'))
    summary = _long(os.path.join(outdir, 'validation_summary.csv'))
    assert os.path.isfile(pdf) and os.path.getsize(pdf) > 0
    assert os.path.isfile(summary)

    with open(summary, newline='', encoding='utf-8') as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) >= 6, f"expected >= 6 validation cases, got {len(rows)}"
    failed = [r['case'] for r in rows if r['result'] != 'PASS']
    assert not failed, f"validation cases FAILED: {failed}"
