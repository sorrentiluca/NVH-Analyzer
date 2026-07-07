"""
═══════════════════════════════════════════════════════════════════════════════
 streamlit_app  —  point-and-click front end for the Ballscrew NVH pipeline
═══════════════════════════════════════════════════════════════════════════════

This is the same pipeline the CLI drives (nvh_pipeline.runner.run_all);
the only thing this file adds is a browser UI in front of it:

    • a form that edits the PipelineConfig fields,
    • a Run button that executes the chosen stages, and
    • a results view that reads back manifest.json and shows every PNG / CSV.

It is meant to be launched from the bundled, no-install Python runtime built by
``packaging/build_windows_app.ps1`` (double-click ``Run NVH Analyzer.bat``), but
it also runs in plain dev:

    pip install -r requirements.txt streamlit
    streamlit run streamlit_app.py

Nothing here is OS-specific; it only calls the public pipeline API.
"""

from __future__ import annotations

import datetime
import io
import json
import os
import subprocess
import sys
from contextlib import contextmanager, redirect_stdout, redirect_stderr

# Headless plotting — every stage draws with the Agg backend, no GUI windows.
os.environ.setdefault("MPLBACKEND", "Agg")

# Make the package + EP_*.py stage scripts importable no matter where Streamlit
# is launched from (the bundled launcher sets cwd to the app folder, but be safe).
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

_SETTINGS_FILE = os.path.join(_HERE, ".nvh_settings.json")


def _load_settings() -> dict:
    try:
        with open(_SETTINGS_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_settings(d: dict) -> None:
    try:
        with open(_SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2)
    except Exception:
        pass


def _remember_root(path: str) -> None:
    """Persist the active results root AND keep a recent-projects list, so
    switching between distinct projects/datasets is one click, not retyping."""
    s = _load_settings()
    recent = [r for r in s.get("recent_roots", []) if r and r != path]
    recent.insert(0, path)
    s["recent_roots"] = recent[:8]
    s["results_root"] = path
    _save_settings(s)


def _remember_data_dir(path: str) -> None:
    """Persist the active data folder so reopening the app lands exactly
    where the user left off."""
    s = _load_settings()
    s["data_dir"] = path
    _save_settings(s)


import streamlit as st

from nvh_pipeline.app_helpers import (
    generate_run_name as _generate_run_name,
    scan_past_runs as _scan_past_runs,
    default_for as _default_for,
    csv_list as _csv_list,
    as_str_list as _as_str_list,
)
from nvh_pipeline.common import discover_dataset, inspect_headers
from nvh_pipeline.config import PipelineConfig, set_active, seg_data_dir
from nvh_pipeline.runner import STAGES, run_all

# How many rows to show inline before suggesting the download for the full table.
_CSV_PREVIEW_ROWS = 200


st.set_page_config(page_title="NVH Analyzer", layout="wide")

# WealthSimple-inspired design tokens — one restrained accent, calm neutrals.
_ACCENT = "#1E4D3A"        # muted forest green — single action accent
_INK = "#1A1A1E"           # deep charcoal — primary text
_MUTED = "#6B6B76"         # neutral gray — labels, captions, helper text
_HAIRLINE = "#E5E5EA"      # soft divider / card border (no harsh black lines)
_SURFACE = "#F9F9FB"       # ultra-light off-white surface

# Kept for backward compatibility with any external references.
_SCHAEFFLER_GREEN = _ACCENT


def _inject_css() -> None:
    """Apply the WealthSimple visual language to the main content pane.

    One injected stylesheet handles: a pinned, accent-tinted primary tab bar;
    premium typography and generous whitespace; soft hairline dividers; bordered
    containers/expanders/tables rendered as calm cards with a micro-shadow
    instead of harsh 1px boxes; quieted metric labels with large values; and
    rounded, restrained buttons/inputs.

    Each rule is independent and selector-scoped, so a Streamlit version that
    renames a ``data-testid`` simply skips that rule and degrades gracefully.
    """
    st.markdown(
        f"""
        <style>
          /* ── Typography & rhythm ─────────────────────────────────────────── */
          html, body, [class*="css"] {{
              font-family: -apple-system, system-ui, "Inter", "Segoe UI",
                           sans-serif;
          }}
          section.main h1, section.main h2, section.main h3 {{
              color: {_INK};
              font-weight: 650;
              letter-spacing: -0.012em;
          }}
          /* Helper / caption text in calm neutral gray. */
          section.main [data-testid="stCaptionContainer"],
          section.main .stCaption, section.main small {{
              color: {_MUTED};
          }}

          /* ── Airiness — let the content breathe ──────────────────────────── */
          section.main .block-container {{
              padding-top: 3rem;
              padding-left: 2.5rem;
              padding-right: 2.5rem;
              max-width: 1750px;   /* three side-by-side dashboard panels */
          }}

          /* ── Soft dividers (no heavy contrast lines) ─────────────────────── */
          section.main hr, section.main [data-testid="stDivider"] {{
              border-color: {_HAIRLINE};
              background: {_HAIRLINE};
          }}

          /* ── Cards, not boxes — bordered containers & expanders ──────────── */
          section.main [data-testid="stVerticalBlockBorderWrapper"],
          section.main [data-testid="stExpander"] {{
              border: 1px solid {_HAIRLINE} !important;
              border-radius: 12px !important;
              box-shadow: 0 1px 4px rgba(0,0,0,0.02);
          }}
          section.main [data-testid="stExpander"] summary {{
              border-radius: 12px;
          }}

          /* ── Tables — soft frame + roomy cells, right-aligned numerics ───── */
          section.main [data-testid="stDataFrame"] {{
              border: 1px solid {_HAIRLINE};
              border-radius: 12px;
              overflow: hidden;
          }}
          section.main [data-testid="stTable"] td,
          section.main [data-testid="stTable"] th {{
              padding-top: 0.65rem;
              padding-bottom: 0.65rem;
          }}

          /* ── Metrics — quiet label, confident value ──────────────────────── */
          section.main [data-testid="stMetricLabel"] {{
              color: {_MUTED};
              font-size: 0.8rem;
              font-weight: 500;
          }}
          section.main [data-testid="stMetricValue"] {{
              color: {_INK};
              font-weight: 680;
              letter-spacing: -0.02em;
          }}

          /* ── Buttons — rounded, restrained ───────────────────────────────── */
          section.main .stButton > button,
          section.main [data-testid="stDownloadButton"] > button {{
              border-radius: 8px;
              border: 1px solid {_HAIRLINE};
              padding: 0.4rem 1rem;
              transition: border-color .15s ease, background .15s ease;
          }}
          section.main .stButton > button:hover {{
              border-color: {_ACCENT};
              color: {_ACCENT};
          }}
          /* Primary button keeps the themed accent fill from config.toml. */
          section.main .stButton > button[kind="primary"] {{
              border-color: {_ACCENT};
          }}

          /* ── Inputs — soft border, accent focus ring ─────────────────────── */
          section.main [data-baseweb="input"],
          section.main [data-baseweb="select"] > div {{
              border-radius: 8px;
          }}
          section.main [data-baseweb="input"]:focus-within,
          section.main [data-baseweb="select"] > div:focus-within {{
              border-color: {_ACCENT} !important;
              box-shadow: 0 0 0 1px {_ACCENT};
          }}

          /* ── Small "chip" pills for active filters ───────────────────────── */
          .ws-chip {{
              display: inline-block;
              padding: 0.15rem 0.6rem;
              margin: 0 0.3rem 0.3rem 0;
              border: 1px solid {_HAIRLINE};
              border-radius: 999px;
              font-size: 0.8rem;
              color: {_INK};
              background: {_SURFACE};
          }}

          /* ── Pin the primary Define / Run / Review tab bar to the top ────── */
          /* Streamlit's own header bar is fixed at the very top (~3.5rem). Make
             it opaque white so the pinned tab bar tucks flush beneath it rather
             than showing page content bleeding through. */
          [data-testid="stHeader"] {{
              background: #FFFFFF;
          }}
          /* Pin only the FIRST tab-list (the Define/Run/Review bar). The nested
             sub-tab ribbons sit deeper in the DOM and keep scrolling normally.
             Both the modern (stMain) and legacy (section.main) scroll parents
             are covered so the rule survives Streamlit version changes. */
          [data-testid="stMain"] div[data-baseweb="tab-list"]:first-of-type,
          section.main div[data-baseweb="tab-list"]:first-of-type {{
              position: sticky;
              top: 3.5rem;            /* sit just below Streamlit's fixed header */
              z-index: 100;
              background: #FFFFFF;
              padding-top: 0.4rem;
              margin-top: -0.4rem;
              border-bottom: 1px solid {_HAIRLINE};
              box-shadow: 0 2px 6px rgba(0,0,0,0.03);
          }}
          /* Accent on the active tab label + underline. */
          div[data-baseweb="tab-list"] button[aria-selected="true"] {{
              color: {_ACCENT};
          }}
          div[data-baseweb="tab-highlight"] {{
              background-color: {_ACCENT};
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  UI kit  —  reusable presentation primitives
# ─────────────────────────────────────────────────────────────────────────────
# Every screen (present and future) is composed from these helpers so the
# right-pane UX stays consistent and a new analysis/section inherits it for free.
# They are pure presentation: no pipeline state, no data assumptions.

def _screen_header(title: str, subtitle: str = None, *, actions=None) -> None:
    """Screen title (spatial awareness) + an optional right-aligned action belt.

    ``actions`` is a list of zero-argument callables; each renders its own widget
    (button, download_button, …) into a column on the right. Callers declare
    intent — "put these actions here" — without owning the layout, so global
    actions live in one consistent place instead of scattered across the page.
    """
    actions = [a for a in (actions or []) if a]
    if actions:
        left, right = st.columns([0.62, 0.38])
    else:
        left, right = st.container(), None
    with left:
        st.markdown(f"### {title}")
        if subtitle:
            st.caption(subtitle)
    if actions and right is not None:
        with right:
            cols = st.columns(len(actions))
            for col, render in zip(cols, actions):
                with col:
                    render()


def _breadcrumb(*parts) -> None:
    """A muted ``Section › Sub-section`` trail so location is always visible."""
    trail = "  ›  ".join(str(p) for p in parts if p)
    if trail:
        st.caption(trail)


def _kpi_row(items) -> None:
    """A scannable KPI strip. Each item: {label, value, help?, details?}.

    ``details`` (a list of markdown lines) is tucked behind a "Details" expander
    — high-level number first, granularity on demand (progressive disclosure).
    """
    items = [it for it in (items or []) if it]
    if not items:
        return
    cols = st.columns(len(items))
    for col, it in zip(cols, items):
        with col:
            st.metric(it["label"], it.get("value", "—"), help=it.get("help"))
            if it.get("details"):
                with st.expander("Details"):
                    for line in it["details"]:
                        st.markdown(line)


def _state(kind: str, message: str, *, cta=None) -> None:
    """A calm loading / empty / error block — the shared 4-state architecture.

    Rendered as a soft card (never a harsh red box) with a quiet heading, the
    message, and an optional call-to-action (``cta`` is a zero-arg callable that
    renders a button). The 'data' state is just normal body content, so callers
    only reach for this when there is nothing — or something wrong — to show.
    """
    heading = {"loading": "Working…",
               "empty": "Nothing here yet",
               "error": "Needs attention"}.get(kind, "")
    with st.container(border=True):
        if heading:
            st.markdown(f"**{heading}**")
        st.caption(message)
        if cta:
            cta()


@contextmanager
def _card(title: str = None):
    """A soft card (styled bordered container) for grouping related content."""
    with st.container(border=True):
        if title:
            st.markdown(f"**{title}**")
        yield


def _filter_bar(spec: dict, *, key: str) -> dict:
    """A unified Filters popover + dismissible chips for any list view.

    ``spec`` maps a field label to its list of options. Returns
    ``{field: [selected, …]}``. Selections persist in session_state under
    ``f"{key}_{field}"``; clicking a chip clears that one value. Feed any
    future filterable table the same ``{field: options}`` spec to reuse this.
    """
    for field in spec:
        st.session_state.setdefault(f"{key}_{field}", [])
    pop = (st.popover("Filters") if hasattr(st, "popover")
           else st.expander("Filters"))
    with pop:
        for field, options in spec.items():
            # Drop selections that are no longer valid options before rendering.
            st.session_state[f"{key}_{field}"] = [
                v for v in st.session_state.get(f"{key}_{field}", [])
                if v in options]
            st.multiselect(field, options, key=f"{key}_{field}")
    active = {f: list(st.session_state.get(f"{key}_{f}", [])) for f in spec}
    chips = [(f, v) for f, vals in active.items() for v in vals]
    if chips:
        cols = st.columns(min(6, len(chips)))
        for i, (f, v) in enumerate(chips):
            if cols[i % len(cols)].button(
                    f"{v}  ✕", key=f"{key}_chip_{f}_{v}",
                    help=f"Clear filter — {f}: {v}"):
                st.session_state[f"{key}_{f}"] = [
                    x for x in st.session_state.get(f"{key}_{f}", []) if x != v]
                st.rerun()
    return active


def _render_stepper(steps: list) -> None:
    """The linear workflow progress bar pinned under the title.

    ``steps`` is a list of ``(label, state)`` where state is 'done', 'active'
    or 'todo'.  Completed steps render as green checkmark nodes, the active
    step as an accent-ringed node, and the nodes are joined by connector lines
    that turn green once the step before them is complete — so the user always
    sees where they are between data import, processing and review.
    """
    seg = []
    for i, (label, state) in enumerate(steps):
        if i:
            prev_done = steps[i - 1][1] == "done"
            seg.append(
                f'<div class="ws-step-line{" done" if prev_done else ""}">'
                "</div>")
        icon = "✓" if state == "done" else str(i + 1)
        seg.append(
            f'<div class="ws-step {state}">'
            f'  <div class="ws-step-dot">{icon}</div>'
            f'  <div class="ws-step-label">{label}</div>'
            f"</div>")
    st.markdown(
        f"""
        <style>
          .ws-stepper {{
              display: flex; align-items: flex-start; gap: 0;
              margin: 0.2rem 0 1.0rem 0;
          }}
          .ws-step {{
              display: flex; flex-direction: column; align-items: center;
              gap: 0.35rem; min-width: 84px;
          }}
          .ws-step-dot {{
              width: 30px; height: 30px; border-radius: 50%;
              display: flex; align-items: center; justify-content: center;
              font-size: 0.85rem; font-weight: 650;
              border: 2px solid {_HAIRLINE};
              background: {_SURFACE}; color: {_MUTED};
              transition: background .2s ease, border-color .2s ease;
          }}
          .ws-step.done .ws-step-dot {{
              background: #12805C; border-color: #12805C; color: #FFFFFF;
          }}
          .ws-step.active .ws-step-dot {{
              border-color: {_ACCENT}; color: {_ACCENT};
              box-shadow: 0 0 0 3px rgba(30, 77, 58, 0.15);
          }}
          .ws-step-label {{
              font-size: 0.78rem; color: {_MUTED}; text-align: center;
              max-width: 110px; line-height: 1.25;
          }}
          .ws-step.done .ws-step-label {{ color: #12805C; font-weight: 600; }}
          .ws-step.active .ws-step-label {{ color: {_ACCENT}; font-weight: 650; }}
          .ws-step-line {{
              flex: 1 1 auto; height: 2px; background: {_HAIRLINE};
              margin-top: 14px; min-width: 24px;
              transition: background .2s ease;
          }}
          .ws-step-line.done {{ background: #12805C; }}
        </style>
        <div class="ws-stepper">{''.join(seg)}</div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────
# _csv_list, _as_str_list, _generate_run_name, _scan_past_runs, _default_for
# are imported from nvh_pipeline.app_helpers (pure, no Streamlit dependency).

def _int_list(text: str):
    out = []
    for tok in _csv_list(text):
        try:
            out.append(int(float(tok)))
        except ValueError:
            st.warning(f"Ignoring non-numeric RPM value: {tok!r}")
    return out


def _as_int_list(val):
    """Accept a list of ints/str or a comma string; warn on non-numeric tokens."""
    if isinstance(val, (list, tuple)):
        out = []
        for t in val:
            try:
                out.append(int(float(t)))
            except (ValueError, TypeError):
                st.warning(f"Ignoring non-numeric RPM value: {t!r}")
        return out
    return _int_list(val)


@st.cache_data(show_spinner=False)
def _discover(data_dir: str, mtime: float) -> dict:
    """Cached dataset scan. ``mtime`` is part of the cache key, so the scan
    re-runs when the folder changes but not on every keystroke."""
    return discover_dataset(data_dir)


@st.cache_data(show_spinner=False)
def _inspect(data_dir: str, mtime: float) -> dict:
    """Cached header-consistency scan (reads only each CSV's header row).
    ``mtime`` keys the cache to the folder's current state."""
    return inspect_headers(data_dir)


@st.cache_data(show_spinner=False)
def _single_file_columns(path: str, mtime: float) -> list:
    """Cached header read for a single CSV (continuous mode).  Returns the column
    names, or [] if the file can't be read.  ``mtime`` keys the cache."""
    try:
        import pandas as _pd
        return list(_pd.read_csv(path, nrows=0).columns)
    except Exception:
        return []


def _dir_mtime(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


def _runs_fingerprint(root: str) -> tuple:
    """Cheap change-detector for the run library: (run name, manifest mtime)
    pairs from one scandir pass — no JSON parsing.  Used as the cache key so
    a library of hundreds of runs re-parses manifests only when one changes."""
    try:
        out = []
        with os.scandir(root) as it:
            for e in it:
                if e.is_dir():
                    try:
                        out.append((e.name, os.path.getmtime(
                            os.path.join(e.path, "manifest.json"))))
                    except OSError:
                        pass
        return tuple(sorted(out))
    except OSError:
        return ()


@st.cache_data(show_spinner=False)
def _scan_runs_cached(root: str, fingerprint: tuple) -> list:
    """Cached run-library scan, keyed on the fingerprint — re-scans the
    moment a run is added, re-run or deleted, never on plain reruns."""
    return _scan_past_runs(root)


# PowerShell body for the folder picker. Opens the MODERN Windows folder dialog
# (COM IFileOpenDialog with FOS_PICKFOLDERS) instead of the dated tree-style
# FolderBrowserDialog, and parents it to a hidden, top-most owner window placed
# on the monitor under the cursor — so the dialog appears on the same screen as
# the browser and in front of other windows. Falls back to FolderBrowserDialog
# if the COM interop can't be built. ``$initial`` is prepended by Python.
_FOLDER_PICKER_PS = r'''
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

function New-OwnerForm {
    # A 1px, invisible, top-most form on the cursor's monitor. The modern dialog
    # centers on its owner's monitor, which fixes the wrong-screen complaint.
    $pt = [System.Windows.Forms.Cursor]::Position
    $wa = [System.Windows.Forms.Screen]::FromPoint($pt).WorkingArea
    $f = New-Object System.Windows.Forms.Form
    $f.StartPosition = 'Manual'
    $f.FormBorderStyle = 'None'
    $f.ShowInTaskbar = $false
    $f.Opacity = 0
    $f.Width = 1; $f.Height = 1
    $f.Left = [int]($wa.X + $wa.Width / 2)
    $f.Top  = [int]($wa.Y + $wa.Height / 2)
    $f.TopMost = $true
    [void]$f.Show()
    $f.Activate()
    return $f
}

$csharp = @"
using System;
using System.Runtime.InteropServices;
namespace NvhPicker {
  [ComImport, Guid("43826D1E-E718-42EE-BC55-A1E261C37BFE"),
   InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
  public interface IShellItem {
    void BindToHandler(IntPtr pbc, ref Guid bhid, ref Guid riid, out IntPtr ppv);
    void GetParent(out IShellItem ppsi);
    void GetDisplayName(uint sigdnName, out IntPtr ppszName);
    void GetAttributes(uint sfgaoMask, out uint psfgaoAttribs);
    void Compare(IShellItem psi, uint hint, out int piOrder);
  }
  [ComImport, Guid("42F85136-DB7E-439C-85F1-E4075D135FC8"),
   InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
  public interface IFileOpenDialog {
    [PreserveSig] int Show(IntPtr parent);
    void SetFileTypes(uint cFileTypes, IntPtr rgFilterSpec);
    void SetFileTypeIndex(uint iFileType);
    void GetFileTypeIndex(out uint piFileType);
    void Advise(IntPtr pfde, out uint pdwCookie);
    void Unadvise(uint dwCookie);
    void SetOptions(uint fos);
    void GetOptions(out uint pfos);
    void SetDefaultFolder(IShellItem psi);
    void SetFolder(IShellItem psi);
    void GetFolder(out IShellItem ppsi);
    void GetCurrentSelection(out IShellItem ppsi);
    void SetFileName([MarshalAs(UnmanagedType.LPWStr)] string pszName);
    void GetFileName([MarshalAs(UnmanagedType.LPWStr)] out string pszName);
    void SetTitle([MarshalAs(UnmanagedType.LPWStr)] string pszTitle);
    void SetOkButtonLabel([MarshalAs(UnmanagedType.LPWStr)] string pszText);
    void SetFileNameLabel([MarshalAs(UnmanagedType.LPWStr)] string pszLabel);
    void GetResult(out IShellItem ppsi);
    void AddPlace(IShellItem psi, int fdap);
    void SetDefaultExtension([MarshalAs(UnmanagedType.LPWStr)] string pszDefaultExtension);
    void Close(int hr);
    void SetClientGuid(ref Guid guid);
    void ClearClientData();
    void SetFilter(IntPtr pFilter);
    void GetResults(out IntPtr ppenum);
    void GetSelectedItems(out IntPtr ppsai);
  }
  [ComImport, Guid("DC1C5A9C-E88A-4DDE-A5A1-60F82A20AEF7")]
  public class FileOpenDialog { }
  public static class Picker {
    const uint FOS_PICKFOLDERS = 0x20;
    const uint FOS_FORCEFILESYSTEM = 0x40;
    const uint SIGDN_FILESYSPATH = 0x80058000;
    [DllImport("shell32.dll", CharSet = CharSet.Unicode, PreserveSig = false)]
    static extern void SHCreateItemFromParsingName(
        [MarshalAs(UnmanagedType.LPWStr)] string pszPath, IntPtr pbc,
        ref Guid riid, out IShellItem ppv);
    public static string Pick(IntPtr owner, string initialDir) {
      IFileOpenDialog dlg = (IFileOpenDialog)(new FileOpenDialog());
      uint opts; dlg.GetOptions(out opts);
      dlg.SetOptions(opts | FOS_PICKFOLDERS | FOS_FORCEFILESYSTEM);
      if (!string.IsNullOrEmpty(initialDir)) {
        try {
          Guid iid = new Guid("43826D1E-E718-42EE-BC55-A1E261C37BFE");
          IShellItem item;
          SHCreateItemFromParsingName(initialDir, IntPtr.Zero, ref iid, out item);
          if (item != null) dlg.SetFolder(item);
        } catch { }
      }
      int hr = dlg.Show(owner);
      if (hr != 0) return null;
      IShellItem result; dlg.GetResult(out result);
      IntPtr psz; result.GetDisplayName(SIGDN_FILESYSPATH, out psz);
      string path = Marshal.PtrToStringUni(psz);
      Marshal.FreeCoTaskMem(psz);
      return path;
    }
  }
}
"@

$result = $null
try {
    Add-Type -TypeDefinition $csharp | Out-Null
    $owner = New-OwnerForm
    try { $result = [NvhPicker.Picker]::Pick($owner.Handle, $initial) }
    finally { $owner.Close(); $owner.Dispose() }
} catch {
    # Fallback: classic dialog, still parented to an owner on the cursor monitor.
    $d = New-Object System.Windows.Forms.FolderBrowserDialog
    $d.Description = 'Select folder'
    $d.ShowNewFolderButton = $true
    if ($initial) { $d.SelectedPath = $initial }
    $owner2 = New-OwnerForm
    try {
        if ($d.ShowDialog($owner2) -eq [System.Windows.Forms.DialogResult]::OK) {
            $result = $d.SelectedPath
        }
    } finally { $owner2.Close(); $owner2.Dispose() }
}
if ($result) { Write-Output $result }
'''


def _native_folder_picker(initial_dir: str = "") -> str | None:
    """Open the modern Windows folder-picker dialog via PowerShell.

    Uses the COM IFileOpenDialog (modern, resizable, with address bar / quick
    access), opened on the monitor under the cursor and forced to the front.
    Blocks until the user selects a folder or cancels.  Returns the chosen
    absolute path, or None if cancelled / PowerShell is unavailable.

    Only works when the Streamlit server runs on the user's machine (local
    mode) — the intended deployment for this app.
    """
    initial = initial_dir if (initial_dir and os.path.isdir(initial_dir)) else ""
    # Single-quote the path as a PowerShell literal; escape embedded quotes.
    ps_script = f"$initial = '{initial.replace(chr(39), chr(39) * 2)}'\n" \
        + _FOLDER_PICKER_PS
    try:
        r = subprocess.run(
            ['powershell', '-NoProfile', '-Sta', '-WindowStyle', 'Hidden',
             '-NonInteractive', '-Command', ps_script],
            capture_output=True, text=True, timeout=300,
        )
        path = r.stdout.strip()
        return os.path.abspath(path) if path and os.path.isdir(path) else None
    except Exception:
        return None


def _native_file_picker(initial_dir: str = "") -> str | None:
    """Open a Windows Open-File dialog (CSV filter) via PowerShell WinForms.

    Continuous mode analyses a single file, so this complements the folder
    picker.  Returns the chosen absolute file path, or None if cancelled /
    PowerShell is unavailable.  Local-mode only, like the folder picker."""
    initial = initial_dir if (initial_dir and os.path.isdir(initial_dir)) else ""
    ps_script = (
        "$initial = '" + initial.replace(chr(39), chr(39) * 2) + "'\n"
        "Add-Type -AssemblyName System.Windows.Forms\n"
        "$d = New-Object System.Windows.Forms.OpenFileDialog\n"
        "$d.Filter = 'CSV files (*.csv)|*.csv|All files (*.*)|*.*'\n"
        "if ($initial) { $d.InitialDirectory = $initial }\n"
        "if ($d.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) "
        "{ Write-Output $d.FileName }\n"
    )
    try:
        r = subprocess.run(
            ['powershell', '-NoProfile', '-Sta', '-WindowStyle', 'Hidden',
             '-NonInteractive', '-Command', ps_script],
            capture_output=True, text=True, timeout=300,
        )
        path = r.stdout.strip()
        return os.path.abspath(path) if path and os.path.isfile(path) else None
    except Exception:
        return None


# ── Folder-browser callbacks ──────────────────────────────────────────────────
# Streamlit forbids assigning a widget-bound session key (text inputs) *after*
# the widget is instantiated — callbacks run before any widget is created, so
# they're safe to write widget-bound keys.

def _browse_file_native(browse_key: str, target_key: str) -> None:
    """on_click: open native Open-File dialog, write result to state."""
    current = st.session_state.get(target_key) or ""
    initial = os.path.dirname(current) if current else os.getcwd()
    path = _native_file_picker(initial_dir=initial)
    if path:
        st.session_state[target_key] = path
        st.session_state[browse_key] = path


def _browse_native(browse_key: str, target_key: str) -> None:
    """on_click: open native Windows folder picker, write result to state."""
    current = (st.session_state.get(target_key) or
               st.session_state.get(browse_key) or
               os.getcwd())
    if not os.path.isdir(str(current)):
        current = os.getcwd()
    path = _native_folder_picker(initial_dir=str(current))
    if path:
        st.session_state[target_key] = path
        st.session_state[browse_key] = path
        if target_key == "results_root_input":
            _remember_root(path)


def _auto_name_run(parts_key: str = "_detected_parts",
                   rpms_key: str = "_detected_rpms") -> None:
    """on_click: regenerate run_name_input from current parts/RPMs."""
    parts = st.session_state.get(parts_key) or []
    rpms = st.session_state.get(rpms_key) or []
    st.session_state["run_name_input"] = _generate_run_name(parts, rpms)


def _load_run_settings(manifest_path: str) -> None:
    """on_click: load a past run's nvh_config.json back into the workspace.

    Reads the resolved config saved next to the run and restores the setup
    dict + form fields, so a previous run can be reviewed, tweaked and re-run
    without retyping anything.
    """
    out_root = os.path.dirname(manifest_path)
    cfg_file = os.path.join(out_root, "nvh_config.json")
    if not os.path.isfile(cfg_file):
        st.toast("That run has no saved nvh_config.json to load.")
        return
    try:
        cfg = PipelineConfig.from_json(cfg_file)
    except Exception as exc:
        st.toast(f"Could not load settings: {exc}")
        return

    ss = st.session_state
    ss["single_file_input"] = getattr(cfg, "single_file", None) or ""
    ss["data_dir_input"] = cfg.data_dir or ""
    # output_root = <results_root>/<run_name>: split it back into the two fields.
    _out = (cfg.output_root or "").rstrip("\\/")
    ss["results_root_input"] = os.path.dirname(_out) or "results"
    ss["run_name_input"] = os.path.basename(_out) or ""
    ss["_edit_results_root"] = False

    # The whole setup travels in one persistent dict (see the Setup view).
    ss["_cfg"] = {
        "analysis_mode": getattr(cfg, "analysis_mode", "reciprocating"),
        "signal_label": getattr(cfg, "signal_label", "signal") or "signal",
        "time_col": cfg.time_col,
        "rpm_col": cfg.rpm_col,
        "angle_col": cfg.angle_col,
        "torque_col": cfg.torque_col,
        "accel_cols": {ax: c for ax, c in (cfg.accel_cols or {}).items()
                       if c},
        # only_parts/only_rpms is the filter the user actually chose.
        "parts": [str(p) for p in (cfg.only_parts or cfg.parts or ())],
        "rpms": [int(r) for r in (cfg.only_rpms or cfg.rpms or ())],
    }
    # Drop the Setup view's widget echoes so they re-seed from the new _cfg.
    for _k in ("analysis_mode_ui", "cont_signal_label", "map_time", "map_rpm",
               "map_angle", "map_torque", "setup_parts", "setup_rpms",
               "accel_on_X", "accel_on_Y", "accel_on_Z",
               "accel_col_X", "accel_col_Y", "accel_col_Z"):
        ss.pop(_k, None)

    # Advanced parameters.
    ss["adv_samples_per_rev"] = cfg.samples_per_rev
    ss["adv_ball_pass_order"] = cfg.ball_pass_order
    ss["adv_n_bpf_harmonics"] = cfg.n_bpf_harmonics
    ss["adv_bp_filter_order"] = cfg.bp_filter_order
    ss["adv_bp_min_hz"] = cfg.bp_min_hz
    ss["adv_bp_min_hz_abs"] = getattr(cfg, "bp_min_hz_abs", 50.0)
    ss["adv_bp_min_bw_hz"] = cfg.bp_min_bw_hz
    ss["adv_kurtogram_levels"] = cfg.kurtogram_levels
    ss["adv_min_revolutions"] = getattr(cfg, "min_revolutions", 3.0)
    ss["adv_plateau_gating"] = getattr(cfg, "plateau_gating", True)
    ss["adv_plateau_frac"] = getattr(cfg, "plateau_frac", 0.90)
    ss["adv_bp_max_hz"] = getattr(cfg, "bp_max_hz", 12000.0)
    ss["adv_campbell_window_rev"] = getattr(cfg, "campbell_window_rev", 1.5)
    ss["adv_campbell_rpm_bin"] = getattr(cfg, "campbell_rpm_bin", 50.0)

    # Suppress the folder-change reset (it clears the setup dict when the
    # data folder changes) so the mapping we just loaded survives the rerun.
    ss["_last_data_dir"] = cfg.data_dir or ""
    ss["view"] = "home"
    st.toast(f"Loaded settings from '{os.path.basename(_out)}' — "
             "review them in Setup.")


_COL_NONE = "— none —"


def _col_picker(label: str, options: list, default: str, key: str,
                help: str = None, disabled: bool = False,
                allow_none: bool = False) -> str:
    """A column-role selector. Uses a dropdown of detected columns when a data
    folder is set; falls back to free text otherwise.  Persists via ``key`` while
    guaranteeing the stored value stays within the available options.

    ``allow_none=True`` prepends a "— none —" choice so an optional channel
    (e.g. torque / angle in continuous mode) can be explicitly omitted; the
    picker then returns "" for that selection."""
    opts = list(dict.fromkeys([c for c in options if c]))
    if not opts:
        return st.text_input(label, value=st.session_state.get(key, default or ""),
                             key=key, help=help, disabled=disabled)
    if allow_none:
        opts = [_COL_NONE] + opts
    cur = st.session_state.get(key)
    if cur not in opts:
        st.session_state[key] = default if default in opts else opts[0]
    sel = st.selectbox(label, opts, key=key, help=help, disabled=disabled)
    return "" if sel == _COL_NONE else sel


def _config_from_form(f: dict) -> PipelineConfig:
    """Build a PipelineConfig from the form dict, starting from defaults.

    ``parts``/``rpms`` may be lists (from the auto-detect multiselects) or
    comma-separated strings (the manual-override fallback)."""
    cfg = PipelineConfig()
    cfg.analysis_mode = f.get("analysis_mode", "reciprocating")
    cfg.data_dir = f["data_dir"].strip()
    cfg.output_root = f["output_root"].strip() or "."
    cfg.parts = tuple(_as_str_list(f["parts"]))
    cfg.rpms = tuple(_as_int_list(f["rpms"]))
    cfg.time_col = f["time_col"]
    cfg.rpm_col = f["rpm_col"]
    cfg.angle_col = f["angle_col"]
    cfg.torque_col = f["torque_col"]

    if cfg.analysis_mode == "continuous":
        # One steady signal: bypass folder discovery / part+rpm naming.  A blank
        # nominal RPM means "auto-derive from the data" (handled in EP_segment).
        cfg.single_file = (f.get("single_file") or "").strip() or None
        cfg.signal_label = (f.get("signal_label") or "signal").strip() or "signal"
        _nom = f.get("nominal_rpm")
        cfg.nominal_rpm = float(_nom) if _nom else None
        cfg.parts = (cfg.signal_label,)
        cfg.rpms = (int(round(cfg.nominal_rpm)),) if cfg.nominal_rpm else (0,)
    # Signals to analyze: only the included, mapped axes — an excluded axis is
    # simply absent from accel_cols, which every stage tolerates (it iterates
    # accel_cols.items()).  Fall back to defaults if nothing was provided.
    accel = {ax: col for ax, col in (f.get("accel_cols") or {}).items() if col}
    if accel:
        cfg.accel_cols = accel
    cfg.samples_per_rev = int(f["samples_per_rev"])
    cfg.ball_pass_order = float(f["ball_pass_order"])
    cfg.n_bpf_harmonics = int(f["n_bpf_harmonics"])
    cfg.bp_min_hz = float(f["bp_min_hz"])
    cfg.bp_min_hz_abs = float(f.get("bp_min_hz_abs", 50.0))
    cfg.bp_min_bw_hz = float(f["bp_min_bw_hz"])
    cfg.kurtogram_levels = int(f["kurtogram_levels"])
    cfg.bp_filter_order = int(f["bp_filter_order"])
    if "min_revolutions" in f:
        cfg.min_revolutions = float(f["min_revolutions"])
    if "plateau_gating" in f:
        cfg.plateau_gating = bool(f["plateau_gating"])
    if "plateau_frac" in f:
        cfg.plateau_frac = float(f["plateau_frac"])
    if "bp_max_hz" in f:
        cfg.bp_max_hz = float(f["bp_max_hz"])
    if "campbell_window_rev" in f:
        cfg.campbell_window_rev = float(f["campbell_window_rev"])
    if "campbell_rpm_bin" in f:
        cfg.campbell_rpm_bin = float(f["campbell_rpm_bin"])
    only_parts = _csv_list(f["only_parts"])
    only_rpms = _int_list(f["only_rpms"])
    # When no manual override is given, the multiselect selection IS the filter —
    # propagate it to only_parts/only_rpms so downstream stages and the viz layer
    # both respect what the user chose.
    cfg.only_parts = tuple(only_parts) if only_parts else tuple(cfg.parts) or None
    cfg.only_rpms = tuple(only_rpms) if only_rpms else tuple(cfg.rpms) or None
    if cfg.analysis_mode == "continuous":
        # One signal, one whole-signal segment — no part/RPM filtering, and the
        # actual rpm_category is auto-derived at load time (so a placeholder
        # rpms filter here would wrongly exclude it).
        cfg.only_parts = None
        cfg.only_rpms = None
    cfg.show_plots = False
    cfg.render_plots = bool(f.get("render_plots", True))
    return cfg


def _requested_from_config(out_root: str):
    """Pull the parts/RPMs filter from nvh_config.json written next to results.

    Returns (parts, rpms) using only_parts/only_rpms when present (the filter
    the user actually chose), falling back to parts/rpms (what was segmented).
    Returns empty tuples if the file isn't there.
    """
    cfg_file = os.path.join(out_root, "nvh_config.json")
    if not os.path.isfile(cfg_file):
        return (), ()
    try:
        with open(cfg_file, "r", encoding="utf-8") as fh:
            c = json.load(fh)
        # Continuous mode has no user-requested part/RPM filter — the single
        # rpm_category is auto-derived from the recording, and cfg.rpms holds a
        # placeholder (0).  Return empties so the data-quality view reports the
        # actual analyzed speed instead of flagging the placeholder as "missing".
        if c.get("analysis_mode") == "continuous":
            return (), ()
        parts = c.get("only_parts") or c.get("parts") or []
        rpms = c.get("only_rpms") or c.get("rpms") or []
        return tuple(parts), tuple(rpms)
    except Exception:
        return (), ()


def _render_table_file(f: str, key: str):
    """Inline-preview a CSV (row-capped) or render an HTML table, + a download."""
    name = os.path.basename(f)
    if f.lower().endswith(".csv"):
        st.markdown(f"**{name}**")
        try:
            import pandas as pd
            df = pd.read_csv(f)
            st.dataframe(df.head(_CSV_PREVIEW_ROWS), width='stretch',
                         hide_index=True)
            if len(df) > _CSV_PREVIEW_ROWS:
                st.caption(f"Showing first {_CSV_PREVIEW_ROWS:,} of {len(df):,} "
                           "rows — download for the full table.")
        except Exception:
            st.caption("(Inline preview unavailable — use the download.)")
    elif f.lower().endswith(".html"):
        st.markdown(f"**{name}**")
        try:
            with open(f, "r", encoding="utf-8") as fh:
                # These HTML tables are generated by our own stages (trusted, simple
                # <table> markup), so render inline rather than in a deprecated iframe.
                st.markdown(fh.read(), unsafe_allow_html=True)
        except Exception:
            st.caption("(Inline render unavailable — use the download.)")
    with open(f, "rb") as fh:
        st.download_button(f"Download {name}", fh.read(), file_name=name, key=key)


def _render_files_expander(view, key_prefix: str):
    """The per-tab 'Figures & data files' expander: PNG/PDF + CSV/HTML downloads."""
    figs = [f for f in view.figure_files if os.path.isfile(f)]
    tbls = [f for f in view.table_files if os.path.isfile(f)]
    if not figs and not tbls:
        return
    with st.expander(f"Figures & data files ({len(figs) + len(tbls)})"):
        if tbls:
            for j, f in enumerate(tbls):
                _render_table_file(f, key=f"{key_prefix}_tbl_{j}")
        if figs:
            st.caption("Publication-quality figures (matplotlib):")
            cols = st.columns(2)
            for j, png in enumerate(figs):
                with cols[j % 2]:
                    if png.lower().endswith(".png"):
                        st.image(png, caption=os.path.basename(png),
                                 width='stretch')
                    with open(png, "rb") as fh:
                        st.download_button(f"Download {os.path.basename(png)}",
                                           fh.read(),
                                           file_name=os.path.basename(png),
                                           key=f"{key_prefix}_fig_{j}")


def _render_view(view, key_prefix: str, *, error: str = None):
    """Render one results tab: purpose · takeaways · interactive charts · files.

    Routes the empty/error cases through the shared ``_state`` block so every
    view's "nothing yet" / "something failed" surface looks the same and calm.
    """
    if view.purpose:
        st.caption(view.purpose)

    # ── 4-state architecture ──────────────────────────────────────────────────
    if error:
        _state("error",
               f"This analysis did not finish — {error}. Check the run log, "
               "then run it again.")
        _render_files_expander(view, key_prefix)
        return
    if not view.available:
        _state("empty",
               "This analysis hasn't been run yet. Choose its stage in the "
               "Run tab and start an analysis to populate this view.")
        return

    # ── Data state ────────────────────────────────────────────────────────────
    _kpi_row(view.takeaways)

    for caption, chart in view.charts:
        if caption:
            st.markdown(f"**{caption}**")
        st.altair_chart(chart, width='stretch')

    # Interactive split groups: a selector picks which pre-built variant to show.
    for gi, (caption, variants) in enumerate(getattr(view, "chart_variants", [])):
        if caption:
            st.markdown(f"**{caption}**")
        labels = list(variants.keys())
        sel_key = f"{key_prefix}_split_{gi}"
        if hasattr(st, "segmented_control"):
            choice = st.segmented_control(
                "Split by", labels, default=labels[0],
                key=sel_key, label_visibility="collapsed")
        else:
            choice = st.radio("Split by", labels, horizontal=True,
                              key=sel_key, label_visibility="collapsed")
        if choice not in variants:
            choice = labels[0]
        st.altair_chart(variants[choice], width='stretch')

    for note in view.notes:
        # viz.py prefixes warnings/ok with an emoji; classify on it, then strip it
        # so the rendered note carries no emoji (Streamlit supplies its own icon).
        if note.startswith("⚠"):
            st.warning(note.lstrip("⚠").strip())
        elif note.startswith("✅"):
            st.success(note.lstrip("✅").strip())
        else:
            st.caption(note)

    _render_files_expander(view, key_prefix)


@st.cache_data(show_spinner="Loading results…")
def _build_views(manifest_path: str, mtime: float):
    """Cache viz.build_views() keyed on manifest path + mtime.

    ``mtime`` keys the cache, so a re-run that rewrites the manifest is
    picked up immediately.  Altair
    chart objects and file-path strings are fully picklable so Streamlit's
    pickle-based cache works without issue.
    """
    from nvh_pipeline import viz as _viz

    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    out_root  = manifest.get("output_root", "")
    seg_dir   = os.path.join(out_root, "output_seg", "segmented_data")
    req_parts, req_rpms = _requested_from_config(out_root)
    return _viz.build_views(manifest, out_root, seg_dir,
                            requested_parts=req_parts, requested_rpms=req_rpms)


# Maps each StageView.key to the manifest stage key(s) that feed it, so a failed
# stage can be flagged inside the view it affects (mirrors viz.build_views specs).
_VIEW_STAGE_KEYS = {
    "segment":      ("segment",),
    "loudness":     ("vibration", "loudness", "loudness_plot", "loudness_by_sample"),
    "order":        ("order",),
    "envelope":     ("envelope", "bandpass"),
    "torque":       ("efficiency", "torque"),
    "summary":      ("report", "sample_summary", "characteristics", "actuation_plot"),
    "data_quality": (),
}


def _render_results(manifest_path: str, key_prefix: str = "run", *, on_close=None):
    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    records = manifest.get("stages", [])
    ok = sum(1 for r in records if r["status"] == "ok")
    out_root = manifest.get("output_root", "")
    run_name = os.path.basename(os.path.dirname(manifest_path)) or "Run"
    failed = {r["stage"]: r.get("error", "")
              for r in records if r["status"] != "ok"}

    # ── Header utility belt — title + per-run global actions in one row ───────
    cfg_file = os.path.join(out_root, "nvh_config.json")
    actions = []
    if os.path.isfile(cfg_file):
        def _load(_mp=manifest_path, _k=key_prefix):
            # Pre-fill Define & Setup with this run's settings (re-run / tweak).
            st.button("Load settings", key=f"{_k}_load", width='stretch',
                      disabled=_busy, on_click=_load_run_settings, args=(_mp,),
                      help="Fill Define & Setup with this run's settings.")
        actions.append(_load)

        def _dl_cfg(_cf=cfg_file, _k=key_prefix):
            with open(_cf, "rb") as fh:
                st.download_button("Download config", fh.read(),
                                   file_name="nvh_config.json",
                                   key=f"{_k}_dl_cfg", width='stretch')
        actions.append(_dl_cfg)
    if on_close:
        def _close(_cb=on_close, _k=key_prefix):
            # Closing a saved run is read-only — allowed even during a run.
            st.button("Close", key=f"{_k}_close", width='stretch', on_click=_cb)
        actions.append(_close)

    _screen_header(run_name, f"Folder: {out_root}", actions=actions)
    _breadcrumb("Review", "Open runs", run_name)

    # Failures surface calmly up top; each affected view is flagged again inline.
    if failed:
        _state("error",
               "Some stages did not finish: "
               + "; ".join(f"{s} ({e})" for s, e in failed.items())
               + ". Affected analyses are flagged below.")

    views = _build_views(manifest_path, os.path.getmtime(manifest_path))

    # ── Executive scorecard via the shared KPI strip ──────────────────────────
    dq = next((v for v in views if v.key == "data_quality"), None)
    vib = next((v for v in views if v.key == "loudness"), None)
    def _metric_val(view, label, default="—"):
        if not view:
            return default
        for t in view.takeaways:
            if t["label"].startswith(label):
                return t["value"]
        return default

    _kpi_row([
        {"label": "Stages ok", "value": f"{ok}/{len(records)}"},
        {"label": "Actuations", "value": _metric_val(dq, "Actuations")},
        {"label": "Parts × RPMs", "value": _metric_val(dq, "Parts")},
        {"label": "Plastic / metal", "value": _metric_val(vib, "Plastic / metal")},
        {"label": "Data flags", "value": _metric_val(dq, "Flags")},
    ])
    st.caption(f"Generated {manifest.get('generated','')}")

    # ── One tab per module ───────────────────────────────────────────────────
    labels = [("• " if not v.available else "") + v.title for v in views]
    tabs = st.tabs(labels)
    for i, (tab, view) in enumerate(zip(tabs, views)):
        with tab:
            err = next((failed[k] for k in _VIEW_STAGE_KEYS.get(view.key, ())
                        if k in failed), None)
            _render_view(view, f"{key_prefix}_{i}", error=err)


# ─────────────────────────────────────────────────────────────────────────────
#  Background-run management  —  subprocess spawn, poll, abort
# ─────────────────────────────────────────────────────────────────────────────
# The pipeline runs as a child process (python -m nvh_pipeline … --progress-file)
# so it can be hard-aborted and so the UI stays responsive while it works.

def _read_progress(pf: str) -> dict:
    if not pf or not os.path.isfile(pf):
        return {}
    try:
        with open(pf, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def _pid_alive(pid: int) -> bool:
    """True if a process with this PID is still running (Windows tasklist)."""
    if not pid:
        return False
    try:
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
            capture_output=True, text=True, timeout=10).stdout
        return str(pid) in out
    except Exception:
        return True   # if we can't tell, assume alive (don't false-finish a run)


def _abort_run(pid: int) -> None:
    """Hard-kill the child and its worker pool (whole tree) on Windows."""
    if not pid:
        return
    try:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                       capture_output=True, text=True, timeout=15)
    except Exception:
        pass


def _tail_file(path: str, n_bytes: int = 8000) -> str:
    if not path or not os.path.isfile(path):
        return ""
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - n_bytes))
            data = fh.read().decode("utf-8", errors="replace")
        return data if size <= n_bytes else "…\n" + data
    except Exception:
        return ""


def _spawn_run(cfg_path: str, chosen_stages: list, out_root: str) -> dict:
    """Launch the pipeline as a child process; return the run-job descriptor."""
    pf = os.path.join(out_root, "_progress.json")
    log = os.path.join(out_root, "_run.log")
    for stale in (pf, log):
        try:
            os.remove(stale)
        except OSError:
            pass
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    # Force UTF-8 in the child so its console output (box-drawing glyphs etc.)
    # never trips the Windows cp1252 codec when redirected to the log file.
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["MPLBACKEND"] = "Agg"
    logf = open(log, "w", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, "-m", "nvh_pipeline",
         "--config", cfg_path, "--stages", ",".join(chosen_stages),
         "--progress-file", pf],
        stdout=logf, stderr=subprocess.STDOUT, cwd=_HERE,
        creationflags=flags, env=env)
    logf.close()   # child keeps its own inherited handle
    return {"pid": proc.pid, "pf": pf, "log": log, "out_root": out_root,
            "manifest": os.path.join(out_root, "manifest.json"),
            "stages": list(chosen_stages)}


_STAGE_STATE_LABEL = {"pending": "•", "running": "▶", "ok": "✓", "FAILED": "✗"}


def _render_progress_body(job: dict) -> None:
    """One render of the live progress: bar, per-stage checklist, log, abort.

    Returns nothing; flips session_state and triggers a full rerun on
    completion/abort. Designed to be called from a `run_every` fragment.
    """
    prog = _read_progress(job.get("pf"))
    events = prog.get("events", [])
    done = bool(prog.get("done"))
    pid = job.get("pid")

    # Derive per-stage status from the event stream.
    status: dict = {s: "pending" for s in job.get("stages", [])}
    cur_label, frac = "Starting…", 0.0
    for ev in events:
        n = ev.get("n") or 1
        i = ev.get("i") or 0
        ph = ev.get("phase")
        if ph in ("start", "parallel_start"):
            cur_label = (", ".join(ev.get("labels") or [])
                         or ev.get("label") or "")
            frac = max(frac, (i - 1) / n if n else 0.0)
            if ph == "start" and ev.get("stage") in status:
                status[ev["stage"]] = "running"
            for lab in (ev.get("labels") or []):
                for s, sp in STAGES.items():
                    if sp["label"] == lab and s in status:
                        status[s] = "running"
        elif ph in ("done", "parallel_done"):
            st_ok = "ok" if ev.get("status") == "ok" else "FAILED"
            if ev.get("stage") in status:
                status[ev["stage"]] = st_ok
            frac = max(frac, i / n if n else 0.0)

    st.progress(min(1.0, frac), text=f"{cur_label}")

    # Per-stage checklist (no emojis in labels themselves; markers are glyphs).
    lines = []
    for s in job.get("stages", []):
        mark = _STAGE_STATE_LABEL.get(status.get(s, "pending"), "•")
        lines.append(f"{mark} {STAGES.get(s, {}).get('label', s)}")
    st.markdown("  \n".join(lines))

    if st.button("Abort run", type="secondary", key="abort_run_btn"):
        _abort_run(pid)
        st.session_state["running"] = False
        st.session_state["last_run_msg"] = ("warning", "Run aborted.")
        st.rerun()

    with st.expander("Run log"):
        st.text(_tail_file(job.get("log")))

    # Completion: trust the progress 'done' flag, fall back to PID liveness.
    if done or not _pid_alive(pid):
        st.session_state["running"] = False
        manifest = prog.get("manifest") or job.get("manifest")
        if manifest and os.path.isfile(manifest):
            st.session_state["report_manifest"] = manifest
            st.session_state["_run_completed"] = True
            st.session_state["last_run_msg"] = (
                "success", "Run complete — the report is ready.")
        else:
            err = prog.get("error", "see the run log above")
            st.session_state["last_run_msg"] = (
                "error", f"Run did not finish ({err}).")
        st.rerun()


if hasattr(st, "fragment"):
    _render_progress = st.fragment(run_every=0.8)(_render_progress_body)
else:  # very old Streamlit — degrade to a single render per rerun
    _render_progress = _render_progress_body


# ─────────────────────────────────────────────────────────────────────────────
#  Page  —  title, workspace settings, and the three-step workflow
# ─────────────────────────────────────────────────────────────────────────────
_inject_css()  # calm card visual language + accent

d = PipelineConfig()  # defaults used to prefill the form

# One-time session-state initialisation. The workspace (results root + data
# folder) is restored from settings, so reopening the app lands where the
# user left off.
_settings = _load_settings()
st.session_state.setdefault("data_dir_input", _settings.get("data_dir", ""))
st.session_state.setdefault("results_root_input",
                            _settings.get("results_root", "results"))
st.session_state.setdefault("run_name_input", _generate_run_name([], []))

# True while a pipeline subprocess is running — used to grey out every action
# button so nothing can be changed or re-submitted mid-run.
_busy = bool(st.session_state.get("running"))

# ── Navigation state — one overview page, deep dives behind it ────────────────
# The whole app is a single dashboard ("home") with two deep-dive views the
# panels open: "setup" (column mapping, groups & speeds) and "report" (the
# full results of one run).  Buttons switch views; nothing is more than one
# click from the overview.
_view = st.session_state.setdefault("view", "home")


def _go(view: str) -> None:
    st.session_state["view"] = view


def _open_report(manifest_path: str) -> None:
    st.session_state["report_manifest"] = manifest_path
    st.session_state["view"] = "report"
    # Set in the callback (pre-script) so the stepper's "Review results"
    # checkmark turns green on the very same rerun.
    st.session_state["_report_viewed"] = True


# ═══════════════════════════════════════════════════════════════════════════════
#  Demo workspace  —  one-click starter campaign with known answers
# ═══════════════════════════════════════════════════════════════════════════════

def _launch_demo_workspace() -> None:
    """Generate the synthetic teaching campaign, point the workspace at it,
    and start a full analysis — one click from empty app to live results."""
    from nvh_pipeline.demo import create_demo_workspace

    ws = create_demo_workspace(os.path.join(_HERE, "demo_workspace"))
    st.session_state["data_dir_input"] = ws["data_dir"]
    st.session_state["results_root_input"] = ws["results_dir"]
    st.session_state["_last_data_dir"] = ws["data_dir"]
    st.session_state.pop("_cfg", None)
    st.session_state["run_name_input"] = "demo_walkthrough"
    st.session_state["_demo_guide"] = ws["guide"]
    _remember_root(ws["results_dir"])
    _remember_data_dir(ws["data_dir"])

    scan = discover_dataset(ws["data_dir"])
    form = dict(
        analysis_mode="reciprocating", single_file=None, nominal_rpm=None,
        signal_label="signal", data_dir=ws["data_dir"],
        output_root=os.path.join(ws["results_dir"], "demo_walkthrough"),
        parts=scan["parts"], rpms=scan["rpms"],
        time_col=d.time_col, rpm_col=d.rpm_col, angle_col=d.angle_col,
        torque_col=d.torque_col,
        accel_cols=dict(d.accel_cols),
        samples_per_rev=d.samples_per_rev, ball_pass_order=d.ball_pass_order,
        n_bpf_harmonics=d.n_bpf_harmonics, bp_min_hz=d.bp_min_hz,
        bp_min_hz_abs=d.bp_min_hz_abs, bp_min_bw_hz=d.bp_min_bw_hz,
        kurtogram_levels=d.kurtogram_levels, bp_filter_order=d.bp_filter_order,
        min_revolutions=d.min_revolutions,
        only_parts="", only_rpms="", render_plots=True,
    )
    cfg = _config_from_form(form)
    set_active(cfg)
    os.makedirs(cfg.output_root, exist_ok=True)
    cfg_path = os.path.join(cfg.output_root, "nvh_config.json")
    cfg.to_json(cfg_path)
    os.environ["NVH_CONFIG"] = os.path.abspath(cfg_path)
    st.session_state["run_job"] = _spawn_run(
        os.path.abspath(cfg_path), list(STAGES), cfg.output_root)
    st.session_state["running"] = True
    st.session_state["view"] = "home"



# ── Sidebar — workspace settings; the workflow itself lives on the dashboard ──
with st.sidebar:
    st.markdown("### NVH Analyzer")
    st.caption("Vibration analysis for time-series CSV exports.")

    nav_map = [("home", "Overview"), ("setup", "Setup"), ("report", "Report")]
    for _vk, _vl in nav_map:
        st.button(("● " if _view == _vk else "") + _vl,
                  key=f"nav_{_vk}", width='stretch',
                  on_click=_go, args=(_vk,))
    st.divider()
    st.subheader("Workspace")

    # The Results root presents as a settled value; the editor (input + Browse +
    # Save) is tucked behind a Change button so the sidebar never looks like it
    # is demanding input. The value persists in session_state even while the
    # text widget isn't rendered, so every downstream read keeps working.
    _rr = st.session_state.get("results_root_input", "")
    st.session_state.setdefault("_edit_results_root", not _rr)

    if st.session_state["_edit_results_root"] or not _rr:
        st.text_input(
            "Results folder",
            key="results_root_input", disabled=_busy,
            placeholder=r"e.g.  D:\NVH_Results",
            help="Parent folder. Every run is saved in its own sub-folder here.")
        _b1, _b2 = st.columns(2)
        _b1.button("Browse", key="browse_results_root_native",
                   width='stretch', on_click=_browse_native,
                   args=("browser_results_root", "results_root_input"),
                   disabled=_busy,
                   help="Open the Windows folder picker.")
        if _b2.button("Save", width='stretch', disabled=_busy,
                      help="Save and use this results folder."):
            _remember_root(st.session_state.get("results_root_input", ""))
            st.session_state["_edit_results_root"] = False
            st.toast("Results folder saved.")
            st.rerun()
    else:
        _leaf = os.path.basename(_rr.rstrip("\\/")) or _rr
        _disp = _rr if len(_rr) <= 48 else _rr[:20] + "…" + _rr[-26:]
        st.caption("Results saved to")
        st.markdown(f"**{_leaf}**")
        st.caption(_disp)
        if st.button("Change", key="edit_results_root_btn", disabled=_busy,
                     help="Choose a different results folder."):
            st.session_state["_edit_results_root"] = True
            st.rerun()

        # One-click switching between distinct projects / datasets: every
        # results root ever saved is remembered, so an engineer juggling
        # several test campaigns hops between them without retyping paths.
        _recent = [r for r in _settings.get("recent_roots", [])
                   if r and r != _rr and os.path.isdir(r)]
        if _recent:
            def _switch_project() -> None:
                sel = st.session_state.get("project_switcher")
                if sel and sel != st.session_state.get("results_root_input"):
                    st.session_state["results_root_input"] = sel
                    st.session_state.pop("report_manifest", None)
                    _remember_root(sel)

            st.selectbox(
                "Switch project", [_rr] + _recent, key="project_switcher",
                disabled=_busy, on_change=_switch_project,
                format_func=lambda p: os.path.basename(p.rstrip("\\/")) or p,
                help="Jump to another results folder you've used before. "
                     "Each project keeps its own run history.")

    results_root = st.session_state.get("results_root_input", "")

    # On deep-dive views the live poller moves here so progress + abort stay
    # visible; on the overview it renders in the Status panel instead.
    if _busy and _view != "home":
        st.divider()
        st.subheader("Run in progress")
        _job = st.session_state.get("run_job") or {}
        st.caption(f"Saving to {os.path.basename(_job.get('out_root', '')) or '…'}")
        _render_progress(_job)

    st.divider()
    st.caption("Reads time-series CSV exports, splits each actuation, and "
               "reports vibration level, order content, bearing envelope, "
               "torque and efficiency.")
    st.button("🎓 Demo workspace", key="sidebar_demo", disabled=_busy,
              on_click=_launch_demo_workspace,
              help="Rebuild the synthetic teaching campaign (known answers, "
                   "guided walkthrough) and run a full analysis on it.")

# ── Shared pre-compute (runs before any view renders) ─────────────────────────
# All user setup lives in one persistent dict, st.session_state["_cfg"], so it
# survives moving between the overview and the deep-dive views (Streamlit
# forgets widget state for widgets that are not on the current view).
_cfg_saved: dict = st.session_state.get("_cfg") or {}
_continuous = _cfg_saved.get("analysis_mode", "reciprocating") == "continuous"

data_dir = st.session_state.get("data_dir_input", "").strip()
_dd = data_dir
detected = _discover(_dd, _dir_mtime(_dd)) if os.path.isdir(_dd) else None
headers = _inspect(_dd, _dir_mtime(_dd)) if os.path.isdir(_dd) else None
past_runs = _scan_runs_cached(results_root.strip(),
                              _runs_fingerprint(results_root.strip()))

# Continuous mode: the single CSV drives column detection (no folder scan).
single_file_path = st.session_state.get("single_file_input", "").strip()
_single_valid = bool(single_file_path) and os.path.isfile(single_file_path)
if _continuous and _single_valid:
    _cols_avail = _single_file_columns(single_file_path,
                                       os.path.getmtime(single_file_path))
else:
    _cols_avail = headers["all_columns"] if headers else []

# A change of data folder invalidates the saved setup (column roles, groups):
# hint-matching must run fresh against the new folder's column names.
if _dd != st.session_state.get("_last_data_dir", ""):
    st.session_state.pop("_cfg", None)
    _cfg_saved = {}
    st.session_state["_last_data_dir"] = _dd

_ACCEL_HINTS = {
    "X": ("x accel", "accel x", "acc_x", "ax ", "ch1", " x", "X Accel", "accel",
          "vib"),
    "Y": ("y accel", "accel y", "acc_y", "ay ", "ch2", " y", "Y Accel", "accel"),
    "Z": ("z accel", "accel z", "acc_z", "az ", "ch3", " z", "Z Accel", "accel"),
}


def _auto_signal_map(cols: list) -> dict:
    """Best-guess column roles from the detected headers — the zero-config
    path.  The Setup view refines these; until then the guesses are used."""
    m = {
        "time_col": _default_for(d.time_col, cols,
                                 "time", "t ", "ts", "timestamp"),
        "rpm_col": _default_for(d.rpm_col, cols,
                                "rpm", "frequency", "freq", "speed",
                                "rot", "rev"),
        "angle_col": _default_for(d.angle_col, cols,
                                  "angle", "deg", "encoder", "enc", "pos"),
        "torque_col": _default_for(d.torque_col, cols,
                                   "torque", "nm", "moment", "trq"),
    }
    if cols:
        m["accel_cols"] = {ax: _default_for(d.accel_cols.get(ax, ""), cols,
                                            *_ACCEL_HINTS[ax])
                           for ax in ("X", "Y", "Z")}
        # In continuous single-channel data the same column can win several
        # axes — keep each detected column once.
        seen: set = set()
        m["accel_cols"] = {ax: c for ax, c in m["accel_cols"].items()
                           if c and not (c in seen or seen.add(c))}
    else:
        m["accel_cols"] = dict(d.accel_cols)
    return m


# Effective setup = auto-detected defaults overlaid with whatever the Setup
# view saved.  Every consumer (blockers, run builder, stepper) reads this.
_setup = _auto_signal_map(_cols_avail)
for _k in ("time_col", "rpm_col", "angle_col", "torque_col", "accel_cols",
           "signal_label"):
    if _cfg_saved.get(_k) is not None:
        _setup[_k] = _cfg_saved[_k]
_setup.setdefault("signal_label", d.signal_label)

_detected_parts = list(detected["parts"]) if detected else []
_detected_rpms = list(detected["rpms"]) if detected else []
parts = list(_cfg_saved.get("parts") or _detected_parts)
rpms = list(_cfg_saved.get("rpms") or _detected_rpms)

run_name = st.session_state.get("run_name_input", "").strip()

# ── Workflow stepper — the linear guide across the whole journey ──────────────
# States are derived from real readiness (not which view is open), so the nodes
# turn into green checkmarks exactly when a step is genuinely complete.
_step1_done = bool(
    (_continuous and _single_valid and _cols_avail)
    or (not _continuous and detected and detected.get("matched", 0) > 0))
_step2_done = bool(_step1_done and _cols_avail)
_step3_done = bool((st.session_state.get("_run_completed")
                    or st.session_state.get("_report_viewed")) and not _busy)
_step4_done = bool(st.session_state.get("_report_viewed") and not _busy)

_states = []
for _done in (_step1_done, _step2_done, _step3_done, _step4_done):
    _states.append("done" if _done else "todo")
if _busy:
    _states[2] = "active"
else:
    for _i, _s in enumerate(_states):
        if _s == "todo":
            _states[_i] = "active"
            break

st.markdown("## NVH Analyzer")
_render_stepper([
    ("Import data", _states[0]),
    ("Map signals", _states[1]),
    ("Run analysis", _states[2]),
    ("Review results", _states[3]),
])


# ═══════════════════════════════════════════════════════════════════════════════
#  Cached probes for the Data Source summary + run comparison
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def _probe_fs(path: str, mtime: float):
    """Sampling rate estimated from the first ~400 rows of one CSV."""
    try:
        import pandas as _pd
        import numpy as _np
        df = _pd.read_csv(path, nrows=400)
        tc = next((c for c in df.columns if "time" in str(c).lower()),
                  df.columns[0])
        t = _pd.to_numeric(df[tc], errors="coerce").dropna().to_numpy()
        if len(t) < 3:
            return None
        dt = float(_np.median(_np.diff(t)))
        return (1.0 / dt) if dt > 0 else None
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def _run_quick_metrics(manifest_path: str, mtime: float) -> dict:
    """Key numbers for one finished run — feeds the Compare-runs table."""
    row: dict = {}
    try:
        with open(manifest_path, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
    except Exception:
        return row
    recs = manifest.get("stages", [])
    row["Analyses ok"] = (f"{sum(1 for r in recs if r['status'] == 'ok')}"
                          f"/{len(recs)}")
    out_root = manifest.get("output_root") or os.path.dirname(manifest_path)
    try:
        import pandas as _pd
        rms = _pd.read_csv(os.path.join(out_root, "output_loudness",
                                        "rms_per_segment.csv"))
        row["Actuations"] = len(rms)
        by = rms.groupby("material")["rms_combined"].mean()
        for mat, val in by.items():
            row[f"RMS {mat} (m/s²)"] = round(float(val), 4)
        if {"metal", "plastic"} <= set(by.index) and by["metal"] > 0:
            row["Plastic / metal"] = f"{by['plastic'] / by['metal']:.2f}×"
    except Exception:
        pass
    try:
        with open(os.path.join(out_root, "output_order",
                               "run_metadata.json"), encoding="utf-8") as fh:
            om = json.load(fh)
        row["Mean revolutions"] = om.get("mean_revolutions")
    except Exception:
        pass
    return row


# ═══════════════════════════════════════════════════════════════════════════════
#  VIEW: SETUP  —  deep dive behind the Data Source panel
# ═══════════════════════════════════════════════════════════════════════════════

def _page_setup() -> None:
    _screen_header("Setup",
                   "Confirm how the recordings are read: which column is "
                   "which signal, and which specimens and speeds to include.",
                   actions=[lambda: st.button("← Back to overview",
                                              width='stretch',
                                              on_click=_go, args=("home",))])
    _breadcrumb("Overview", "Setup")

    # ── Analysis mode ─────────────────────────────────────────────────────────
    _MODE_LABELS = {
        "reciprocating": "Test rig  ·  a folder of recordings, split into strokes",
        "continuous":    "Single continuous signal  ·  one file, analysed whole",
    }
    st.session_state.setdefault("analysis_mode_ui",
                                _cfg_saved.get("analysis_mode",
                                               "reciprocating"))
    mode = st.radio(
        "Data type",
        options=list(_MODE_LABELS.keys()),
        format_func=lambda k: _MODE_LABELS[k],
        key="analysis_mode_ui", horizontal=True, disabled=_busy,
        help="Test rig: a folder of files named <group>-<speed>rpm-<trial>.csv, "
             "each split into individual strokes. Single continuous signal: "
             "one steady recording (e.g. a fan or gearbox) analysed as a "
             "whole — no naming rules, and torque / angle may be left out.")
    _cont = mode == "continuous"

    signal_label = _setup.get("signal_label", "signal")
    if _cont:
        _dflt_label = (os.path.splitext(os.path.basename(single_file_path))[0]
                       if single_file_path else d.signal_label)
        st.session_state.setdefault("cont_signal_label",
                                    _cfg_saved.get("signal_label",
                                                   _dflt_label))
        signal_label = st.text_input(
            "Signal name", key="cont_signal_label", disabled=_busy,
            help="A name for this signal, used in plot titles and result "
                 "tables.").strip() or "signal"

    st.divider()

    # ── Signals & columns ─────────────────────────────────────────────────────
    cols_avail = _cols_avail
    st.markdown("**Vibration signals to analyze**")
    st.caption("Include one or more accelerometer axes. Excluded axes are "
               "skipped entirely.")
    if not cols_avail:
        st.caption(("Choose a valid CSV file on the overview" if _cont
                    else "Set a valid data folder on the overview")
                   + " to map columns. Defaults are used until then.")
    accel_map: dict = {}
    _saved_accel = _setup.get("accel_cols") or {}
    for ax in ("X", "Y", "Z"):
        c_on, c_sel = st.columns([1, 4])
        st.session_state.setdefault(f"accel_on_{ax}",
                                    bool(_saved_accel.get(ax))
                                    or not _saved_accel)
        on = c_on.checkbox(f"{ax} axis", key=f"accel_on_{ax}", disabled=_busy)
        dflt = _saved_accel.get(ax) or _default_for(
            d.accel_cols.get(ax, ""), cols_avail, *_ACCEL_HINTS[ax])
        with c_sel:
            sel = _col_picker(f"{ax} column", cols_avail, dflt,
                              key=f"accel_col_{ax}", disabled=_busy)
        if on and sel:
            accel_map[ax] = sel
    if not accel_map:
        st.warning("No signals selected — include at least one axis to run.")

    st.markdown("**Other columns**")
    mc1, mc2 = st.columns(2)
    with mc1:
        time_sel = _col_picker(
            "Time", cols_avail, _setup["time_col"], key="map_time",
            disabled=_busy,
            help="The timestamp of each sample, in seconds.")
        rpm_sel = _col_picker(
            "Speed (RPM)", cols_avail, _setup["rpm_col"], key="map_rpm",
            disabled=_busy, allow_none=_cont,
            help="The rotation speed at each sample — used to split strokes "
                 "and to track orders.")
    with mc2:
        angle_sel = _col_picker(
            "Angle", cols_avail, _setup["angle_col"], key="map_angle",
            disabled=_busy, allow_none=_cont,
            help="The shaft angle from the encoder. The most accurate basis "
                 "for order tracking when present.")
        torque_sel = _col_picker(
            "Torque", cols_avail, _setup["torque_col"], key="map_torque",
            disabled=_busy, allow_none=_cont,
            help="Drive torque — needed for the Torque & Efficiency analysis "
                 "only.")
    if _cont:
        st.caption("Continuous mode: **Time** and at least one of "
                   "**RPM / Angle** are required; **Torque** and the unused "
                   "speed channel may be set to “— none —”.")

    st.divider()

    # ── Groups & speeds (test-rig mode only) ─────────────────────────────────
    sel_parts, sel_rpms = list(parts), list(rpms)
    if not _cont:
        st.markdown("**Specimens & speeds to include**")
        if detected and detected["parts"]:
            _opts_p, _opts_r = detected["parts"], detected["rpms"]
            st.session_state.setdefault(
                "setup_parts", [p for p in parts if p in _opts_p] or
                list(_opts_p))
            st.session_state.setdefault(
                "setup_rpms", [r for r in rpms if r in _opts_r] or
                list(_opts_r))
            st.session_state["setup_parts"] = [
                p for p in st.session_state["setup_parts"] if p in _opts_p]
            st.session_state["setup_rpms"] = [
                r for r in st.session_state["setup_rpms"] if r in _opts_r]
            sel_parts = st.multiselect(
                "Specimens", options=_opts_p, key="setup_parts",
                disabled=_busy,
                help="Detected from the file names. Deselect to exclude.")
            sel_rpms = st.multiselect(
                "Speed classes", options=_opts_r, key="setup_rpms",
                disabled=_busy, format_func=str)
            st.caption(f"{detected['matched']} file(s) · "
                       f"{len(detected['parts'])} specimen(s) · "
                       f"{len(detected['rpms'])} speed class(es).")
            if detected["unmatched"]:
                with st.expander(f"Assign groups to "
                                 f"{len(detected['unmatched'])} file(s) "
                                 "not auto-recognized"):
                    st.caption("Files without a group are not processed. "
                               "Rename them `<group>-<speed>rpm-<trial>.csv` "
                               "for full support, or assign here.")
                    _manual_parts: list = []
                    _manual_rpms: list = []
                    for _uf in detected["unmatched"][:50]:
                        _uc1, _uc2, _uc3 = st.columns([3, 2, 2])
                        _uc1.write(_uf)
                        _ug = _uc2.text_input(
                            "Group", key=f"manual_group_{_uf}", disabled=_busy,
                            label_visibility="collapsed", placeholder="Group")
                        _ur = _uc3.text_input(
                            "RPM", key=f"manual_rpm_{_uf}", disabled=_busy,
                            label_visibility="collapsed", placeholder="RPM")
                        if _ug and _ug not in _manual_parts:
                            _manual_parts.append(_ug)
                        if _ur:
                            try:
                                _urv = int(float(_ur))
                                if _urv not in _manual_rpms:
                                    _manual_rpms.append(_urv)
                            except ValueError:
                                pass
                    sel_parts = list(sel_parts) + [
                        p for p in _manual_parts if p not in sel_parts]
                    sel_rpms = list(sel_rpms) + [
                        r for r in _manual_rpms if r not in sel_rpms]
        else:
            st.info("Set a data folder on the overview, or define specimens "
                    "and speeds manually.")
            _po = st.text_input("Specimens (comma-separated)",
                                value=", ".join(str(p) for p in parts),
                                disabled=_busy)
            _ro = st.text_input("Speed classes (comma-separated)",
                                value=", ".join(str(r) for r in rpms),
                                disabled=_busy)
            sel_parts = _csv_list(_po)
            sel_rpms = _as_int_list(_ro)

        st.divider()

        # ── Column consistency check ─────────────────────────────────────────
        with st.expander("Column consistency across files"):
            if not headers or not headers["n_files"]:
                st.info("Set a valid data folder to inspect columns.")
            else:
                st.caption("Columns are matched by **name**, so a different "
                           "column order between files is handled "
                           "automatically. Only missing or unexpected "
                           "columns need attention.")
                _nf = headers["n_files"]
                _nmis = headers["n_mismatched"]
                _nreord = headers.get("n_reordered", 0)
                _summary = (f"{_nf - _nmis - _nreord} identical · "
                            f"{_nreord} reordered (handled) · "
                            f"{_nmis} need review")
                if _nmis == 0:
                    st.success(f"All {_nf} file(s) usable — {_summary}.")
                else:
                    st.warning(f"{_nmis} of {_nf} file(s) have missing or "
                               f"unexpected columns — {_summary}.")
                _status_label = {
                    "identical":  "Identical",
                    "reordered":  "Reordered (handled)",
                    "mismatch":   "Missing/extra columns",
                    "empty":      "Empty file",
                }
                import pandas as _pd
                rows = [{
                    "File": fr["name"],
                    "Status": _status_label.get(fr.get("status"), "Review"),
                    "Missing columns": ", ".join(fr["missing"]) or "—",
                    "Unexpected columns": ", ".join(fr["extra"]) or "—",
                } for fr in headers["files"]]
                st.dataframe(_pd.DataFrame(rows), width='stretch',
                             hide_index=True)
                if headers["unreadable"]:
                    st.error("Could not read: "
                             + "; ".join(f"{n} ({e})"
                                         for n, e in headers["unreadable"]))

    # ── Persist — the overview and the run builder read this dict ────────────
    st.session_state["_cfg"] = {
        "analysis_mode": mode,
        "signal_label": signal_label,
        "time_col": time_sel,
        "rpm_col": rpm_sel,
        "angle_col": angle_sel,
        "torque_col": torque_sel,
        "accel_cols": accel_map,
        "parts": list(sel_parts),
        "rpms": [int(r) for r in sel_rpms] if sel_rpms else [],
    }

    st.button("← Back to overview", on_click=_go, args=("home",),
              type="primary")


# ═══════════════════════════════════════════════════════════════════════════════
#  VIEW: REPORT  —  deep dive behind the Status panel
# ═══════════════════════════════════════════════════════════════════════════════

def _page_report() -> None:
    if not past_runs:
        _screen_header("Report", "Full results for one analysis run.")
        _state("empty",
               "No completed runs in this workspace yet. Run an analysis "
               "from the overview to create one.",
               cta=lambda: st.button("← Back to overview",
                                     on_click=_go, args=("home",)))
        return

    names = {pr["manifest"]: pr["name"] for pr in past_runs}
    options = list(names)
    current = st.session_state.get("report_manifest")
    if current not in options:
        current = options[0]
    sel = st.selectbox("Run", options,
                       index=options.index(current),
                       format_func=lambda m: names[m],
                       help="Every completed run in this workspace.")
    st.session_state["report_manifest"] = sel
    st.session_state["_report_viewed"] = True
    _render_results(sel, key_prefix=f"rep{options.index(sel)}",
                    on_close=lambda: _go("home"))


# ═══════════════════════════════════════════════════════════════════════════════
#  VIEW: HOME  —  the three-panel overview (Data · Analyze · Status)
# ═══════════════════════════════════════════════════════════════════════════════

def _panel_data_source() -> None:
    st.markdown("#### Data source")

    if _continuous:
        f_path, f_btn = st.columns([3, 1])
        with f_path:
            st.text_input(
                "CSV file", key="single_file_input", disabled=_busy,
                placeholder=r"e.g.  D:\tests\fan_rig.csv",
                help="One continuous recording: vibration, a speed "
                     "reference and time.")
        with f_btn:
            st.write("")
            st.button("Browse", key="browse_file_native", width='stretch',
                      on_click=_browse_file_native,
                      args=("browser_file", "single_file_input"),
                      disabled=_busy)
        if not single_file_path:
            st.info("Choose a CSV file to begin.")
        elif not _single_valid:
            st.error("That path is not a file.")
        elif not _cols_avail:
            st.warning("Could not read any columns — check it is a valid CSV.")
        else:
            st.success(f"{len(_cols_avail)} columns detected.")
    else:
        c_path, c_btn = st.columns([3, 1])
        with c_path:
            st.text_input(
                "Data folder", key="data_dir_input", disabled=_busy,
                placeholder=r"e.g.  D:\tests\campaign1",
                help="Folder of CSV recordings — one file per trial, named "
                     "<group>-<speed>rpm-<trial>.csv.")
        with c_btn:
            st.write("")
            st.button("Browse", key="browse_data_native", width='stretch',
                      on_click=_browse_native,
                      args=("browser_data", "data_dir_input"),
                      disabled=_busy)

        # First-visit onboarding: an empty app offers the demo campaign.
        if not _dd and not past_runs:
            with st.container(border=True):
                st.markdown("**New here?**")
                st.caption("Create the demo workspace: a small synthetic "
                           "test campaign with known answers. It runs a "
                           "full analysis immediately, so you can explore "
                           "every screen with data whose correct results "
                           "are documented.")
                st.button("Create demo workspace", type="primary",
                          disabled=_busy, on_click=_launch_demo_workspace)

        if _dd and not os.path.isdir(_dd):
            st.error("That path is not a folder.")
        elif detected:
            n_total = detected.get("n_files", detected["matched"])
            n_matched = detected["matched"]
            if n_total == 0:
                st.warning("No CSV files found here.")
            elif n_matched == n_total:
                st.success(f"{n_total} file(s) — all recognized.")
            elif n_matched > 0:
                st.warning(f"{n_total} file(s) — {n_matched} recognized, "
                           f"{n_total - n_matched} need a group "
                           "(assign in Setup).")
            else:
                st.warning(f"{n_total} file(s) found, none match the "
                           "`<group>-<speed>rpm-<trial>.csv` naming — "
                           "assign groups in Setup.")

        # Drag & drop straight into the data folder (files stay local; the
        # app server runs on this machine).
        with st.expander("Add files — drag & drop"):
            ups = st.file_uploader(
                "Drop CSV file(s) here", type=["csv"],
                accept_multiple_files=True, key="dropzone", disabled=_busy,
                help="Dropped files are copied into the data folder above.")
            if ups:
                target = _dd
                if not target:
                    target = os.path.join(results_root.strip() or ".",
                                          "incoming_data")
                saved_ids = st.session_state.setdefault("_dropped_ids", set())
                n_new = 0
                for up in ups:
                    fid = (up.name, getattr(up, "size", len(up.getvalue())))
                    if fid in saved_ids:
                        continue
                    os.makedirs(target, exist_ok=True)
                    dest = os.path.join(target, os.path.basename(up.name))
                    with open(dest, "wb") as fh:
                        fh.write(up.getvalue())
                    saved_ids.add(fid)
                    n_new += 1
                if n_new:
                    if not _dd:
                        st.session_state["data_dir_input"] = target
                    st.toast(f"Added {n_new} file(s) to "
                             f"{os.path.basename(target)}.")
                    st.rerun()

        # File explorer: what was recognized, grouped.
        if detected and detected["counts"]:
            with st.expander(
                    f"File explorer — {detected['matched']} recognized"):
                import pandas as _pd
                st.dataframe(_pd.DataFrame(detected["counts"]),
                             width='stretch', hide_index=True)
                if detected["unmatched"]:
                    st.caption("Not recognized: "
                               + ", ".join(detected["unmatched"][:10]))

    # Workspace guide (present in the demo workspace, or any workspace that
    # ships one next to its data folder).
    _guide = st.session_state.get("_demo_guide")
    if not _guide and _dd:
        _cand = os.path.join(os.path.dirname(_dd.rstrip("\\/")),
                             "WORKSPACE_GUIDE.md")
        _guide = _cand if os.path.isfile(_cand) else None
    if _guide and os.path.isfile(_guide):
        with st.expander("📖 Workspace guide — what to look for"):
            try:
                with open(_guide, "r", encoding="utf-8") as fh:
                    st.markdown(fh.read())
            except Exception:
                st.caption("(Guide could not be read.)")

    # ── Summary strip ─────────────────────────────────────────────────────────
    st.divider()
    _first_csv = None
    if _continuous and _single_valid:
        _first_csv = single_file_path
    elif detected and detected["counts"] and _dd:
        for _n in sorted(os.listdir(_dd)):
            if _n.lower().endswith(".csv"):
                _first_csv = os.path.join(_dd, _n)
                break
    _fs = (_probe_fs(_first_csv, _dir_mtime(_first_csv))
           if _first_csv else None)
    _n_files = (1 if (_continuous and _single_valid)
                else (detected.get("n_files", 0) if detected else 0))
    s1, s2, s3 = st.columns(3)
    s1.metric("Files", str(_n_files) if _n_files else "—")
    s2.metric("Channels", str(len(_cols_avail)) if _cols_avail else "—",
              help="Columns detected in the recordings.")
    s3.metric("Sampling rate",
              f"{_fs / 1000:.1f} kHz" if _fs and _fs >= 1000
              else (f"{_fs:.0f} Hz" if _fs else "—"),
              help="Estimated from the time column of the first file.")
    if not _continuous and detected and detected["parts"]:
        st.caption(f"{len(parts)}/{len(_detected_parts)} specimen(s) and "
                   f"{len(rpms)}/{len(_detected_rpms)} speed class(es) "
                   "selected — change in Setup.")

    st.button("Define setup →", width='stretch', disabled=_busy,
              on_click=_go, args=("setup",),
              help="Column mapping, specimens & speeds, consistency checks.")


def _panel_analyze() -> tuple:
    """Center panel. Returns (chosen_stages, run_options dict)."""
    st.markdown("#### Analyze")

    _STAGE_TILES = {
        "segment":    ("🧩", "Split recordings into individual strokes. "
                             "Runs first."),
        "vibration":  ("📈", "Overall shake level (RMS) per stroke."),
        "efficiency": ("⚙️", "Drive torque + efficiency per speed."),
        "order":      ("🎯", "Vibration per shaft revolution — finds "
                             "ball-pass content."),
        "envelope":   ("🔔", "Bearing-impact detection via the resonance "
                             "band."),
        "report":     ("📋", "Deliverable summary tables."),
    }
    stage_labels = {k: v["label"] for k, v in STAGES.items()}
    _stage_opts = [k for k in STAGES
                   if not (_continuous and k == "efficiency")]
    _stage_default = [k for k in STAGES if k != "report"
                      and not (_continuous and k == "efficiency")]
    chosen = []
    _tile_cols = st.columns(2)
    for _ti, _sk in enumerate(_stage_opts):
        _icon, _desc = _STAGE_TILES.get(_sk, ("•", ""))
        with _tile_cols[_ti % 2]:
            with st.container(border=True):
                if st.checkbox(f"{_icon}  **{stage_labels[_sk]}**",
                               value=_sk in _stage_default,
                               key=f"stage_tile_{_sk}", disabled=_busy):
                    chosen.append(_sk)
                st.caption(_desc)
    if _continuous:
        st.caption("Efficiency & torque are omitted in continuous mode "
                   "(they need torque / angle channels).")

    nc1, nc2 = st.columns([3, 1])
    with nc1:
        st.text_input("Run name", key="run_name_input", disabled=_busy,
                      help="The sub-folder this run is saved under.")
    with nc2:
        st.write("")
        st.button("Auto-name", width='stretch', disabled=_busy,
                  on_click=_auto_name_run,
                  help="Generate a name from the date and your selections.")
    st.session_state["_detected_parts"] = parts
    st.session_state["_detected_rpms"] = rpms

    auto_include_segment = st.checkbox(
        "Prepare data automatically when needed", value=True, disabled=_busy,
        help="If later analyses are selected but the data hasn't been "
             "segmented yet, run segmentation first.")
    st.session_state.setdefault("render_plots", True)
    render_plots = st.checkbox(
        "Generate figures", key="render_plots", disabled=_busy,
        help="Uncheck for a fast numbers-only run. Tables and interactive "
             "charts are always produced; static PNG figures are skipped.")

    with st.expander("Advanced options"):
        st.caption("Order / envelope tuning. The defaults suit the standard "
                   "setup — hover any field for a plain-English explanation.")
        for _k, _v in (("adv_samples_per_rev", d.samples_per_rev),
                       ("adv_ball_pass_order", d.ball_pass_order),
                       ("adv_n_bpf_harmonics", d.n_bpf_harmonics),
                       ("adv_bp_filter_order", d.bp_filter_order),
                       ("adv_bp_min_hz", d.bp_min_hz),
                       ("adv_bp_min_hz_abs", d.bp_min_hz_abs),
                       ("adv_bp_min_bw_hz", d.bp_min_bw_hz),
                       ("adv_kurtogram_levels", d.kurtogram_levels),
                       ("adv_min_revolutions", d.min_revolutions),
                       ("adv_plateau_gating", d.plateau_gating),
                       ("adv_plateau_frac", d.plateau_frac),
                       ("adv_bp_max_hz", d.bp_max_hz),
                       ("adv_campbell_window_rev", d.campbell_window_rev),
                       ("adv_campbell_rpm_bin", d.campbell_rpm_bin)):
            st.session_state.setdefault(_k, _v)
        ac1, ac2 = st.columns(2)
        with ac1:
            st.number_input(
                "samples_per_rev", step=1, key="adv_samples_per_rev",
                disabled=_busy,
                help="How finely each shaft revolution is resampled. The "
                     "highest readable order is half this value (128 → "
                     "order 64). Raising it multiplies compute time.")
            st.number_input(
                "ball_pass_order", step=0.01, format="%.2f",
                key="adv_ball_pass_order", disabled=_busy,
                help="How many balls pass a fixed point per shaft "
                     "revolution — the fingerprint of a recirculation "
                     "defect. Comes from the screw geometry; change it only "
                     "for different hardware.")
            st.number_input(
                "n_bpf_harmonics", step=1, key="adv_n_bpf_harmonics",
                disabled=_busy,
                help="How many multiples of the ball-pass order to mark and "
                     "integrate. A real defect usually shows several "
                     "multiples; noise shows one.")
            st.number_input(
                "bp_filter_order", step=1, key="adv_bp_filter_order",
                disabled=_busy,
                help="Steepness of the band-pass filter in the "
                     "single-stroke extraction plot. 4 is a good default.")
            st.number_input(
                "min_revolutions", step=0.5, format="%.1f",
                key="adv_min_revolutions", disabled=_busy,
                help="Minimum shaft rotations per stroke to include it in "
                     "order/envelope analysis. Raise for finer order "
                     "resolution at the cost of rejecting short strokes.")
        with ac2:
            st.number_input(
                "bp_min_hz", step=50.0, key="adv_bp_min_hz", disabled=_busy,
                help="Impact-band search floor at max speed. Scaled down "
                     "automatically per stroke.")
            st.number_input(
                "bp_min_hz_abs", step=10.0, key="adv_bp_min_hz_abs",
                disabled=_busy,
                help="Absolute minimum search frequency (Hz), regardless "
                     "of speed.")
            st.number_input(
                "bp_min_bw_hz", step=50.0, key="adv_bp_min_bw_hz",
                disabled=_busy,
                help="Narrowest impact band the search may pick (Hz). Too "
                     "narrow drops the sidebands the envelope needs — keep "
                     "≥ 200 Hz unless the resonance is very sharp.")
            st.number_input(
                "kurtogram_levels", step=1, key="adv_kurtogram_levels",
                disabled=_busy,
                help="How finely the impact-band search divides the "
                     "frequency axis (each level halves the band width). "
                     "Higher = finer but slower.")
            st.number_input(
                "bp_max_hz", step=500.0, key="adv_bp_max_hz", disabled=_busy,
                help="Upper limit (Hz) for the impact band — keep it inside "
                     "the accelerometer's calibrated range so the search "
                     "can't latch onto sensor noise.")
        st.divider()
        pg1, pg2 = st.columns(2)
        with pg1:
            st.checkbox(
                "Plateau gating", key="adv_plateau_gating", disabled=_busy,
                help="On: steady-state numbers use only the constant-speed "
                     "part of each stroke, so ramps don't bias torque, "
                     "efficiency or the order ceiling. Off: full stroke.")
        with pg2:
            st.number_input(
                "plateau_frac", min_value=0.1, max_value=1.0, step=0.05,
                format="%.2f", key="adv_plateau_frac", disabled=_busy,
                help="'Constant speed' means at least this fraction of the "
                     "nominal speed. Lower it if strokes are short and too "
                     "many fall back to the full stroke.")
        cb1, cb2 = st.columns(2)
        with cb1:
            st.number_input(
                "campbell_window_rev", min_value=0.5, max_value=10.0,
                step=0.25, format="%.2f", key="adv_campbell_window_rev",
                disabled=_busy,
                help="Speed-map window length in revolutions. Order "
                     "resolution ≈ 1/this; shorter = finer speed axis but "
                     "coarser orders.")
        with cb2:
            st.number_input(
                "campbell_rpm_bin", min_value=5.0, max_value=500.0, step=5.0,
                format="%.0f", key="adv_campbell_rpm_bin", disabled=_busy,
                help="Speed-axis bin width for the speed map (smaller = "
                     "more bins).")

    return chosen, {"auto_include_segment": auto_include_segment,
                    "render_plots": render_plots}


def _run_analysis(chosen: list, opts: dict) -> None:
    """Validate, persist the config next to the results, and spawn the run."""
    _rr = results_root.strip() or "results"
    _rn = run_name or _generate_run_name(parts, rpms)
    form = dict(
        analysis_mode=("continuous" if _continuous else "reciprocating"),
        single_file=single_file_path, nominal_rpm=None,
        signal_label=_setup.get("signal_label", "signal"),
        data_dir=data_dir, output_root=os.path.join(_rr, _rn),
        parts=parts, rpms=rpms,
        time_col=_setup["time_col"], rpm_col=_setup["rpm_col"],
        angle_col=_setup["angle_col"], torque_col=_setup["torque_col"],
        accel_cols=_setup["accel_cols"],
        samples_per_rev=st.session_state["adv_samples_per_rev"],
        ball_pass_order=st.session_state["adv_ball_pass_order"],
        n_bpf_harmonics=st.session_state["adv_n_bpf_harmonics"],
        bp_min_hz=st.session_state["adv_bp_min_hz"],
        bp_min_hz_abs=st.session_state["adv_bp_min_hz_abs"],
        bp_min_bw_hz=st.session_state["adv_bp_min_bw_hz"],
        kurtogram_levels=st.session_state["adv_kurtogram_levels"],
        bp_filter_order=st.session_state["adv_bp_filter_order"],
        min_revolutions=st.session_state["adv_min_revolutions"],
        bp_max_hz=st.session_state["adv_bp_max_hz"],
        plateau_gating=st.session_state["adv_plateau_gating"],
        plateau_frac=st.session_state["adv_plateau_frac"],
        campbell_window_rev=st.session_state["adv_campbell_window_rev"],
        campbell_rpm_bin=st.session_state["adv_campbell_rpm_bin"],
        only_parts="", only_rpms="",
        render_plots=opts["render_plots"],
    )
    cfg = _config_from_form(form)
    set_active(cfg)

    if "segment" in chosen and not _continuous:
        scan = _discover(cfg.data_dir, _dir_mtime(cfg.data_dir))
        if scan["matched"] == 0:
            st.error("No recognized CSV files were found in:\n\n"
                     f"{cfg.data_dir}\n\nNothing to process.")
            st.stop()

    chosen_stages = list(chosen)
    if ("segment" not in chosen_stages
            and not os.path.isdir(seg_data_dir(cfg))):
        if opts["auto_include_segment"]:
            chosen_stages = ["segment"] + chosen_stages
            st.info("Preparing data (segmentation) first so the selected "
                    "analyses have input.")
        else:
            st.error("This data hasn't been prepared yet. Enable "
                     "\"Prepare data automatically\", or select the "
                     "Segmentation tile.")
            st.stop()
    chosen_stages = [s for s in STAGES if s in chosen_stages]

    os.makedirs(cfg.output_root, exist_ok=True)
    cfg_path = os.path.join(cfg.output_root, "nvh_config.json")
    cfg.to_json(cfg_path)
    os.environ["NVH_CONFIG"] = os.path.abspath(cfg_path)
    _remember_root(_rr)
    if cfg.data_dir:
        _remember_data_dir(cfg.data_dir)

    st.session_state["run_job"] = _spawn_run(
        os.path.abspath(cfg_path), chosen_stages, cfg.output_root)
    st.session_state["running"] = True
    st.rerun()


def _panel_status() -> None:
    st.markdown("#### Investigate & status")

    # ── Current processing ────────────────────────────────────────────────────
    if _busy:
        _job = st.session_state.get("run_job") or {}
        with st.container(border=True):
            st.markdown("**Current processing**")
            st.caption("Active run: "
                       f"{os.path.basename(_job.get('out_root', '')) or '…'}")
            _render_progress(_job)
    else:
        _msg = st.session_state.pop("last_run_msg", None)
        if _msg:
            {"success": st.success, "warning": st.warning,
             "error": st.error}.get(_msg[0], st.info)(_msg[1])
            if (_msg[0] == "success"
                    and st.session_state.get("report_manifest")):
                st.button("Open the report →", type="primary",
                          on_click=_open_report,
                          args=(st.session_state["report_manifest"],))

    # ── Run history ───────────────────────────────────────────────────────────
    st.markdown("**Run history**")
    if not past_runs:
        _state("empty", "No completed runs yet. Set a data source and press "
                        "Run analysis — results appear here.")
        return

    for pr in past_runs[:6]:
        _ok = (pr["stages_ok"] == pr["stages_total"] and pr["stages_total"])
        with st.container(border=True):
            hc1, hc2 = st.columns([3, 1])
            with hc1:
                st.markdown(f"**{pr['name']}**")
                st.caption(f"{pr['generated'] or 'unknown date'} · "
                           + ("✅ all ok" if _ok else
                              f"⚠ {pr['stages_ok']}/{pr['stages_total']} ok")
                           + (f" · {len(pr['parts'])} specimen(s)"
                              if pr["parts"] else ""))
            with hc2:
                st.button("View report", key=f"open_{pr['name']}",
                          width='stretch',
                          on_click=_open_report, args=(pr["manifest"],))
                st.button("Load settings", key=f"load_{pr['name']}",
                          width='stretch', disabled=_busy,
                          on_click=_load_run_settings,
                          args=(pr["manifest"],),
                          help="Re-fill the setup with this run's settings.")
    if len(past_runs) > 6:
        st.caption(f"…and {len(past_runs) - 6} more — open any from the "
                   "Report view's run selector.")

    # ── Compare runs ──────────────────────────────────────────────────────────
    if len(past_runs) >= 2:
        with st.expander("Compare runs"):
            st.caption("Key numbers side by side. Pick two or more runs.")
            _by_name = {pr["name"]: pr for pr in past_runs}
            picks = st.multiselect("Runs to compare", list(_by_name),
                                   max_selections=6, key="compare_runs")
            if len(picks) >= 2:
                import pandas as _pd
                rows = []
                for name in picks:
                    m = _by_name[name]["manifest"]
                    row = {"Run": name,
                           **_run_quick_metrics(m, _dir_mtime(m))}
                    rows.append(row)
                st.dataframe(_pd.DataFrame(rows).set_index("Run").T,
                             width='stretch')


def _page_home() -> None:
    col_data, col_run, col_status = st.columns([1.0, 1.05, 1.0], gap="medium")

    with col_data:
        with st.container(border=True):
            _panel_data_source()

    with col_run:
        with st.container(border=True):
            chosen, opts = _panel_analyze()

            # ── Readiness gating — a clear reason, never a dead button ───────
            blockers = []
            if _continuous:
                if not single_file_path:
                    blockers.append("choose a CSV file (Data source)")
                elif not _single_valid:
                    blockers.append("the CSV file path is not valid")
                if not (_setup["rpm_col"] or _setup["angle_col"]):
                    blockers.append("map a speed reference — RPM or angle "
                                    "(Setup)")
            else:
                if not data_dir:
                    blockers.append("set a data folder (Data source)")
                elif not os.path.isdir(data_dir):
                    blockers.append("the data folder path is not valid")
                elif detected and detected.get("matched", 0) == 0:
                    blockers.append("no recognized recordings in the folder")
            if not _setup["accel_cols"]:
                blockers.append("include at least one vibration signal "
                                "(Setup)")
            if not run_name:
                blockers.append("name the run")
            if not chosen:
                blockers.append("select at least one analysis tile")

            if _busy:
                _state("loading",
                       "An analysis is running — progress is in the "
                       "Investigate & status panel.")
            else:
                if blockers:
                    st.warning("Before running, please "
                               + "; ".join(blockers) + ".")
                if st.button("RUN ANALYSIS", type="primary",
                             width='stretch', disabled=bool(blockers)):
                    _run_analysis(chosen, opts)

    with col_status:
        with st.container(border=True):
            _panel_status()


# ── Router ─────────────────────────────────────────────────────────────────────
if _view == "setup":
    _page_setup()
elif _view == "report":
    _page_report()
else:
    _page_home()
