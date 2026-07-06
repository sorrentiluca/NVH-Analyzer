"""
═══════════════════════════════════════════════════════════════════════════════
 streamlit_app  —  point-and-click front end for the Ballscrew NVH pipeline
═══════════════════════════════════════════════════════════════════════════════

This is the same pipeline the CLI and FAMOS drive (nvh_pipeline.runner.run_all);
the only thing this file adds is a browser UI in front of it:

    • a form that edits the PipelineConfig fields,
    • a Run button that executes the chosen stages, and
    • a results view that reads back manifest.json and shows every PNG / CSV.

It is meant to be launched from the bundled, no-install Python runtime built by
``packaging/build_windows_app.ps1`` (double-click ``Run NVH Analyzer.bat``), but
it also runs in plain dev:

    pip install -r requirements.txt streamlit
    streamlit run streamlit_app.py

Nothing here is FAMOS- or OS-specific; it only calls the public pipeline API.
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
              padding-left: 3.5rem;
              padding-right: 3.5rem;
              max-width: 1400px;
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
def _discover(data_dir: str, _mtime: float) -> dict:
    """Cached dataset scan. ``_mtime`` busts the cache when the folder changes so
    per-keystroke reruns don't rescan a large raw-data share."""
    return discover_dataset(data_dir)


@st.cache_data(show_spinner=False)
def _inspect(data_dir: str, _mtime: float) -> dict:
    """Cached header-consistency scan (reads only each CSV's header row).
    ``_mtime`` busts the cache when the folder changes."""
    return inspect_headers(data_dir)


@st.cache_data(show_spinner=False)
def _single_file_columns(path: str, _mtime: float) -> list:
    """Cached header read for a single CSV (continuous mode).  Returns the column
    names, or [] if the file can't be read.  ``_mtime`` busts the cache on edit."""
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
            _save_settings({"results_root": path})


def _auto_name_run(parts_key: str = "_detected_parts",
                   rpms_key: str = "_detected_rpms") -> None:
    """on_click: regenerate run_name_input from current parts/RPMs."""
    parts = st.session_state.get(parts_key) or []
    rpms = st.session_state.get(rpms_key) or []
    st.session_state["run_name_input"] = _generate_run_name(parts, rpms)


def _open_run(manifest_path: str) -> None:
    """on_click: add a run to the Review tab set (most-recent first), no reprocessing."""
    runs = st.session_state.setdefault("opened_runs", [])
    if manifest_path in runs:
        runs.remove(manifest_path)
    runs.insert(0, manifest_path)


def _close_run(manifest_path: str) -> None:
    """on_click: remove a run from the Review tab set."""
    runs = st.session_state.get("opened_runs", [])
    if manifest_path in runs:
        runs.remove(manifest_path)


def _load_run_settings(manifest_path: str) -> None:
    """on_click: load a past run's nvh_config.json back into the Define form.

    Reads the resolved config saved next to the run and writes the form's
    widget-bound session keys, so a previous run can be reviewed, tweaked and
    re-run without retyping anything (replaces the old "Reproduce this run").
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
    # Analysis mode + continuous-only fields.
    ss["analysis_mode_ui"] = getattr(cfg, "analysis_mode", "reciprocating")
    ss["single_file_input"] = getattr(cfg, "single_file", None) or ""
    ss["cont_signal_label"] = getattr(cfg, "signal_label", "signal") or "signal"
    ss["data_dir_input"] = cfg.data_dir or ""
    # output_root = <results_root>/<run_name>: split it back into the two fields.
    _out = (cfg.output_root or "").rstrip("\\/")
    ss["results_root_input"] = os.path.dirname(_out) or "results"
    ss["run_name_input"] = os.path.basename(_out) or ""
    ss["_edit_results_root"] = False

    # Column mapping (only set an axis column when that axis was included).
    ss["map_time"] = cfg.time_col
    ss["map_rpm"] = cfg.rpm_col
    ss["map_angle"] = cfg.angle_col
    ss["map_torque"] = cfg.torque_col
    for ax in ("X", "Y", "Z"):
        col = (cfg.accel_cols or {}).get(ax, "")
        ss[f"accel_on_{ax}"] = bool(col)
        if col:
            ss[f"accel_col_{ax}"] = col

    # Groups / speeds — only_parts/only_rpms is the filter the user actually chose.
    ss["setup_parts"] = [str(p) for p in (cfg.only_parts or cfg.parts or ())]
    ss["setup_rpms"] = [int(r) for r in (cfg.only_rpms or cfg.rpms or ())]

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

    # Suppress the folder-change reset (it clears map_*/accel_*/setup_* when the
    # data folder changes) so the mapping we just loaded survives the next rerun.
    ss["_last_data_dir"] = cfg.data_dir or ""
    st.toast(f"Loaded settings from '{os.path.basename(_out)}' — "
             "review in Define & Setup.")


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
def _build_views(manifest_path: str, _mtime: float):
    """Cache viz.build_views() keyed on manifest path + mtime.

    ``_mtime`` busts the cache after a re-run rewrites the manifest.  Altair
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
            _open_run(manifest)
            st.session_state["last_run_msg"] = (
                "success", "Run complete. Open the Review tab to explore results.")
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
_inject_css()  # sticky main tabs + Schaeffler-green accent

d = PipelineConfig()  # defaults used to prefill the form

# One-time session-state initialisation. results_root is restored from settings.
_settings = _load_settings()
st.session_state.setdefault("data_dir_input", "")
st.session_state.setdefault("results_root_input",
                            _settings.get("results_root", "results"))
st.session_state.setdefault("run_name_input", _generate_run_name([], []))
st.session_state.setdefault("opened_runs", [])

# True while a pipeline subprocess is running — used to grey out every action
# button so nothing can be changed or re-submitted mid-run.
_busy = bool(st.session_state.get("running"))

# ── Sidebar — workspace settings only; the workflow itself lives in the tabs ───
with st.sidebar:
    st.markdown("### NVH Analyzer")
    st.caption("Vibration analysis for DEWESOFT / FAMOS CSV exports.")
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
            "Results root",
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
            _save_settings(
                {"results_root": st.session_state.get("results_root_input", "")})
            st.session_state["_edit_results_root"] = False
            st.toast("Results root saved.")
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

    results_root = st.session_state.get("results_root_input", "")

    # Global run status — the single live poller lives here so progress + abort
    # are visible from every tab while the rest of the app stays usable.
    if _busy:
        st.divider()
        st.subheader("Run in progress")
        _job = st.session_state.get("run_job") or {}
        st.caption(f"Saving to {os.path.basename(_job.get('out_root', '')) or '…'}")
        _render_progress(_job)

    st.divider()
    st.caption("Reads DEWESOFT-style CSV exports, segments each actuation, and "
               "reports vibration level, order content, bearing envelope, torque "
               "and efficiency.")

# ── Pre-compute state needed by all tabs (runs before any tab renders) ─────────
# Analysis mode: 'reciprocating' (ballscrew rig, a folder of files) or
# 'continuous' (one steady rotating-machine signal).  The radio that sets this
# renders inside the Define tab, but its value is read here (from session state)
# because the pre-compute block runs first on every rerun.
st.session_state.setdefault("analysis_mode_ui", "reciprocating")
_continuous = st.session_state.get("analysis_mode_ui") == "continuous"

data_dir = st.session_state.get("data_dir_input", "").strip()
_dd = data_dir
detected = _discover(_dd, _dir_mtime(_dd)) if os.path.isdir(_dd) else None
headers = _inspect(_dd, _dir_mtime(_dd)) if os.path.isdir(_dd) else None
past_runs = _scan_past_runs(results_root.strip())

# Continuous mode: the single CSV drives column detection (no folder scan).
single_file_path = st.session_state.get("single_file_input", "").strip()
_single_valid = bool(single_file_path) and os.path.isfile(single_file_path)
if _continuous and _single_valid:
    _cols_avail = _single_file_columns(single_file_path,
                                       os.path.getmtime(single_file_path))
else:
    _cols_avail = headers["all_columns"] if headers else []

# Clear column-role session keys when the data folder changes so hint-matching
# runs fresh against the new folder's column names.
_COL_ROLE_KEYS = (
    "map_time", "map_rpm", "map_angle", "map_torque",
    "accel_col_X", "accel_col_Y", "accel_col_Z",
    "accel_on_X", "accel_on_Y", "accel_on_Z",
    # Groups/speeds selections also re-seed to "all detected" on a folder change.
    "setup_parts", "setup_rpms",
)
if _dd != st.session_state.get("_last_data_dir", ""):
    for _k in _COL_ROLE_KEYS:
        st.session_state.pop(_k, None)
    st.session_state["_last_data_dir"] = _dd

# Safe defaults — overwritten by widget values on every rerun once the user
# visits the Define & Setup tab (all tab code executes on each rerun).
accel_cols_map: dict = {}
time_col: str = d.time_col
rpm_col: str = d.rpm_col
angle_col: str = d.angle_col
torque_col: str = d.torque_col
parts = list(d.parts)
rpms = list(d.rpms)
run_name: str = ""
# Continuous-mode form values (overwritten by widgets when that mode is active).
signal_label_val: str = d.signal_label
nominal_rpm_val = None

# ── Primary navigation — immediately below the title ──────────────────────────
tab_define, tab_run, tab_review = st.tabs(
    ["Define & Setup", "Run", "Review"])


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — Define & Setup  (sub-ribbon: Data · Columns · Signals · Run setup)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_define:
    _screen_header("Define & Setup",
                   "Point at your raw CSVs, confirm the column mapping, and name "
                   "the run.")
    _breadcrumb("Define & Setup")

    # ── Analysis mode ──────────────────────────────────────────────────────────
    # Ballscrew rig = a folder of <part>-<rpm>rpm-<trial>.csv, split into strokes.
    # Continuous = one steady rotating-machine signal, analysed whole (no naming).
    _MODE_LABELS = {
        "reciprocating": "Ballscrew rig  ·  multi-file, stroke segmentation",
        "continuous":    "Single continuous signal  ·  one file, whole-signal",
    }
    st.radio(
        "Analysis mode",
        options=list(_MODE_LABELS.keys()),
        format_func=lambda k: _MODE_LABELS[k],
        key="analysis_mode_ui", horizontal=True, disabled=_busy,
        help="Ballscrew rig: a folder of named CSVs, each split into alternating "
             "actuations. Single continuous signal: one steady recording (e.g. a "
             "helicopter vibration file) analysed as a whole — no part/speed "
             "naming, and torque / angle may be omitted.")
    if _continuous:
        st.caption("Continuous mode: point at one CSV, map its columns "
                   "(Time + RPM or angle required), and run — the whole signal "
                   "is treated as a single segment.")

    # Per-tab readiness metrics
    _d1, _d2 = st.columns(2)
    with _d1:
        if detected and detected.get("n_files", detected["matched"]):
            _n_total = detected.get("n_files", detected["matched"])
            st.metric("Data", f"{_n_total} CSV files",
                      help=f"{detected['matched']} auto-grouped, "
                           f"{len(detected['parts'])} group(s), "
                           f"{len(detected['rpms'])} speed class(es).")
        elif _dd:
            st.metric("Data", "No files found")
        else:
            st.metric("Data", "No folder set")
    with _d2:
        if headers and headers["n_files"]:
            if headers["n_mismatched"] == 0:
                st.metric("Columns", "All consistent")
            else:
                st.metric("Columns",
                          f"{headers['n_mismatched']} of {headers['n_files']} differ")
        else:
            st.metric("Columns", "—")

    sub_data, sub_cols, sub_signals, sub_setup = st.tabs(
        ["Data source", "Column check", "Signals & columns", "Run setup"])

    # ── Sub-tab: Data source ───────────────────────────────────────────────────
    with sub_data:
      if _continuous:
        # One file drives the whole analysis.
        f_path, f_btn = st.columns([5, 1])
        with f_path:
            single_file_path = st.text_input(
                "CSV file", key="single_file_input", disabled=_busy,
                placeholder=r"e.g.  D:\tests\helicopter_vibration.csv",
                help="A single continuous recording (vibration + a speed "
                     "reference + time).").strip()
        with f_btn:
            st.write("")
            st.button("Browse", key="browse_file_native",
                      width='stretch', on_click=_browse_file_native,
                      args=("browser_file", "single_file_input"),
                      disabled=_busy)
        if not single_file_path:
            st.info("Choose a CSV file above to begin.")
        elif not os.path.isfile(single_file_path):
            st.error("That path is not a file.")
        elif not _cols_avail:
            st.warning("Could not read any columns from that file — check it is "
                       "a valid CSV.")
        else:
            st.success(f"{len(_cols_avail)} columns detected — "
                       "map them in the **Signals & columns** tab.")

        # No operating-speed input: order tracking follows the instantaneous RPM,
        # so a single "nominal RPM" is neither needed nor meaningful for a
        # variable-speed signal. The RPM range is reported in the run summary.
        _dflt_label = (os.path.splitext(os.path.basename(single_file_path))[0]
                       if single_file_path else d.signal_label)
        st.session_state.setdefault("cont_signal_label", _dflt_label)
        signal_label_val = st.text_input(
            "Signal label", key="cont_signal_label", disabled=_busy,
            help="A name for this signal, used in plot titles / result tables "
                 "in place of the ballscrew part id.").strip() or "signal"
        nominal_rpm_val = None   # always auto-derived (median of the RPM channel)
        # Keep the folder-mode vars sane so downstream reads don't break.
        data_dir = ""
      else:
        c_path, c_btn = st.columns([5, 1])
        with c_path:
            data_dir = st.text_input(
                "Raw CSV folder", key="data_dir_input", disabled=_busy,
                placeholder=r"e.g.  D:\tests\Plastic",
                help="Folder of CSV exports — one file per trial.").strip()
        with c_btn:
            st.write("")
            st.button("Browse", key="browse_data_native",
                      width='stretch', on_click=_browse_native,
                      args=("browser_data", "data_dir_input"),
                      disabled=_busy)
        if not _dd:
            st.info("Set a data folder above to begin.")
        elif not os.path.isdir(_dd):
            st.error("That path is not a folder.")
        elif detected:
            n_total = detected.get("n_files", detected["matched"])
            n_matched = detected["matched"]
            if n_total == 0:
                st.warning("No CSV files found here. Check the folder.")
            elif n_matched == n_total:
                st.success(f"{n_total} CSV file(s) found — "
                           f"{len(detected['parts'])} group(s), "
                           f"{len(detected['rpms'])} speed class(es). "
                           "All files match the auto-grouping naming convention.")
            elif n_matched > 0:
                st.success(f"{n_total} CSV file(s) found — {n_matched} auto-grouped, "
                           f"{n_total - n_matched} need manual assignment (see Run setup).")
            else:
                st.warning(f"{n_total} CSV file(s) found. "
                           "None match the auto-grouping naming format — "
                           "assign groups manually in Run setup.")
            if headers and headers.get("all_columns"):
                st.info(f"{len(headers['all_columns'])} columns detected — "
                        "review mapping in the **Signals & columns** tab.")

    # ── Sub-tab: Column check ──────────────────────────────────────────────────
    with sub_cols:
        if not headers or not headers["n_files"]:
            st.info("Set a valid data folder (Data source tab) to inspect columns.")
        else:
            st.caption("Columns are matched by **name**, so a different column "
                       "order between files is handled automatically. Only "
                       "missing or unexpected columns need attention.")
            _nf = headers["n_files"]
            _nmis = headers["n_mismatched"]
            _nreord = headers.get("n_reordered", 0)
            _nident = _nf - _nmis - _nreord
            _summary = (f"{_nident} identical · {_nreord} reordered (handled) · "
                        f"{_nmis} need review")
            if _nmis == 0:
                st.success(f"All {_nf} file(s) usable — {_summary}. "
                           f"Reference layout has "
                           f"{len(headers['reference_columns'])} columns.")
            else:
                st.warning(f"{_nmis} of {_nf} file(s) have missing or unexpected "
                           f"columns and may not process correctly — {_summary}.")

            _status_label = {
                "identical":  "Identical",
                "reordered":  "Reordered (handled)",
                "mismatch":   "Missing/extra columns",
                "empty":      "Empty file",
            }
            import pandas as _pd
            rows = [{
                "File": fr["name"],
                "Status": _status_label.get(fr.get("status"),
                                            "OK" if fr["matches_reference"]
                                            else "Review"),
                "Missing columns": ", ".join(fr["missing"]) or "—",
                "Unexpected columns": ", ".join(fr["extra"]) or "—",
            } for fr in headers["files"]]
            st.dataframe(_pd.DataFrame(rows), width='stretch',
                         hide_index=True)

            # Legend so each Status value is self-explanatory.
            st.caption(
                "Status — **Identical**: matches the reference exactly · "
                "**Reordered (handled)**: same columns, different order (fine) · "
                "**Missing/extra columns**: differs from the reference and may "
                "not process · **Empty file**: no header row.")

            # The reference layout every file is compared against.
            with st.expander(f"Reference layout — "
                             f"{len(headers['reference_columns'])} columns"):
                st.caption("The majority column signature; other files are "
                           "compared against it by name.")
                st.code(", ".join(headers["reference_columns"]) or "(none)")

            # Per-file header preview — answers exactly what is wrong with a file.
            _files = headers["files"]
            if _files:
                _names = [fr["name"] for fr in _files]
                _flagged = [fr["name"] for fr in _files
                            if fr.get("status") in ("mismatch", "empty")]
                _pick = st.selectbox(
                    "Preview a file's headers", _names,
                    index=(_names.index(_flagged[0]) if _flagged else 0),
                    help="Inspect any file's exact columns and how they differ "
                         "from the reference.")
                _fr = next((f for f in _files if f["name"] == _pick), None)
                if _fr is not None:
                    st.caption(
                        f"Status: **{_status_label.get(_fr.get('status'), '—')}**"
                        f"  ·  {len(_fr['columns'])} columns")
                    if _fr.get("missing"):
                        st.markdown("**Missing** (in reference, absent here): "
                                    + ", ".join(_fr["missing"]))
                    if _fr.get("extra"):
                        st.markdown("**Unexpected** (here, not in reference): "
                                    + ", ".join(_fr["extra"]))
                    if not _fr.get("missing") and not _fr.get("extra"):
                        st.caption("Columns match the reference"
                                   + (" (different order)."
                                      if _fr.get("status") == "reordered"
                                      else "."))
                    st.code(", ".join(_fr["columns"]) or "(empty)")

            if headers["unreadable"]:
                st.error("Could not read: "
                         + "; ".join(f"{n} ({e})"
                                     for n, e in headers["unreadable"]))

    # ── Sub-tab: Signals & columns ─────────────────────────────────────────────
    with sub_signals:
        # Continuous mode reads columns from the single file; reciprocating from
        # the folder's header scan.
        cols_avail = _cols_avail
        if not cols_avail:
            st.caption(("Choose a valid CSV file" if _continuous
                        else "Set a valid data folder")
                       + " to map columns. Defaults are used until then.")
        st.markdown("**Vibration signals to analyze**")
        st.caption("Include one or more accelerometer axes. Excluded axes are "
                   "skipped entirely.")
        _accel_hints = {
            "X": ("x accel", "accel x", "acc_x", "ax ", "ch1", " x", "X Accel", "accel"),
            "Y": ("y accel", "accel y", "acc_y", "ay ", "ch2", " y", "Y Accel", "accel"),
            "Z": ("z accel", "accel z", "acc_z", "az ", "ch3", " z", "Z Accel", "accel"),
        }
        accel_cols_map = {}
        for ax in ("X", "Y", "Z"):
            c_on, c_sel = st.columns([1, 4])
            on = c_on.checkbox(f"{ax} axis", value=True, key=f"accel_on_{ax}",
                               disabled=_busy)
            dflt = _default_for(d.accel_cols.get(ax, ""), cols_avail,
                                *_accel_hints[ax])
            with c_sel:
                sel = _col_picker(f"{ax} column", cols_avail, dflt,
                                  key=f"accel_col_{ax}", disabled=_busy)
            if on and sel:
                accel_cols_map[ax] = sel
        if not accel_cols_map:
            st.warning("No signals selected — include at least one axis to run.")

        st.markdown("**Other columns**")
        mc1, mc2 = st.columns(2)
        with mc1:
            time_col = _col_picker(
                "Time", cols_avail,
                _default_for(d.time_col, cols_avail,
                             "time", "t ", "ts", "timestamp"),
                key="map_time", disabled=_busy)
            rpm_col = _col_picker(
                "Speed (RPM)", cols_avail,
                _default_for(d.rpm_col, cols_avail,
                             "rpm", "frequency", "freq", "speed", "rot", "rev"),
                key="map_rpm", disabled=_busy, allow_none=_continuous)
        with mc2:
            # In continuous mode torque and angle are optional (choose "— none —"
            # to omit); only Time + one speed reference (RPM or angle) is needed.
            angle_col = _col_picker(
                "Angle", cols_avail,
                _default_for(d.angle_col, cols_avail,
                             "angle", "deg", "encoder", "enc", "pos"),
                key="map_angle", disabled=_busy, allow_none=_continuous)
            torque_col = _col_picker(
                "Torque", cols_avail,
                _default_for(d.torque_col, cols_avail,
                             "torque", "nm", "moment", "trq"),
                key="map_torque", disabled=_busy, allow_none=_continuous)
        if _continuous:
            st.caption("Continuous mode: **Time** and at least one of "
                       "**RPM / Angle** are required; **Torque** and the unused "
                       "speed channel may be set to “— none —”.")

    # ── Sub-tab: Run setup (groups, speeds, run name) ─────────────────────────
    with sub_setup:
      if _continuous:
        st.caption("Continuous mode analyses the whole signal — no groups or "
                   "speed classes to select. Set the operating speed and label "
                   "in the **Data source** tab.")
      else:
        st.markdown("**Groups & speeds to include**")
        if detected and detected["parts"]:
            # Keyed selections persist across reruns and can be pre-filled by
            # "Load settings"; seed to all detected, and keep them valid against
            # the current folder's options (a folder change re-seeds via the
            # reset block below).
            _opts_p, _opts_r = detected["parts"], detected["rpms"]
            st.session_state.setdefault("setup_parts", list(_opts_p))
            st.session_state.setdefault("setup_rpms", list(_opts_r))
            st.session_state["setup_parts"] = [
                p for p in st.session_state["setup_parts"] if p in _opts_p]
            st.session_state["setup_rpms"] = [
                r for r in st.session_state["setup_rpms"] if r in _opts_r]
            parts = st.multiselect(
                "Groups / specimens", options=_opts_p, key="setup_parts",
                disabled=_busy,
                help="Detected from filenames. Deselect to exclude.")
            rpms = st.multiselect(
                "Speed classes", options=_opts_r, key="setup_rpms",
                disabled=_busy, format_func=str)
            st.session_state["_detected_parts"] = parts
            st.session_state["_detected_rpms"] = rpms
            st.caption(f"{detected['matched']} file(s) · {len(detected['parts'])} "
                       f"group(s) · {len(detected['rpms'])} speed class(es).")
            if detected["counts"]:
                st.dataframe(detected["counts"], width='stretch',
                             hide_index=True)
            if detected["unmatched"]:
                with st.expander(f"Assign groups to {len(detected['unmatched'])} "
                                 "file(s) not auto-recognized"):
                    st.caption("Files without a group assignment will not be "
                               "processed by segmentation. Rename them as "
                               "`<group>-<speed>rpm-<trial>.csv` for full pipeline "
                               "support.")
                    _manual_parts: list = []
                    _manual_rpms: list = []
                    _hc1, _hc2, _hc3 = st.columns([3, 2, 2])
                    _hc1.caption("File")
                    _hc2.caption("Group")
                    _hc3.caption("RPM")
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
                    for _mp in _manual_parts:
                        if _mp not in parts:
                            parts = list(parts) + [_mp]
                    for _mr in _manual_rpms:
                        if _mr not in rpms:
                            rpms = list(rpms) + [_mr]
            with st.expander("Define groups / speeds manually"):
                st.caption("Comma-separated. Leave blank to use the selections "
                           "above.")
                _po = st.text_input("Groups override", "", disabled=_busy)
                _ro = st.text_input("Speeds override", "", disabled=_busy)
                if _csv_list(_po):
                    parts = _po
                if _csv_list(_ro):
                    rpms = _ro
        else:
            if _dd and os.path.isdir(_dd):
                st.warning("No recognized `<group>-<rpm>rpm-<trial>.csv` files "
                           "here. Enter groups and speeds manually.")
            else:
                st.info("Set a data folder, or define groups and speeds manually.")
            parts = st.text_input("Groups (comma-separated)",
                                  value=", ".join(d.parts), disabled=_busy)
            rpms = st.text_input("Speed classes (comma-separated)",
                                 value=", ".join(str(r) for r in d.rpms),
                                 disabled=_busy)

      # Run naming applies to both modes.
      st.divider()
      st.markdown("**Name this run**")
      nc1, nc2 = st.columns([5, 1])
      with nc1:
          run_name = st.text_input(
              "Run name", key="run_name_input", disabled=_busy,
              help="The sub-folder this run is saved under.").strip()
      with nc2:
          st.write("")
          st.button("Auto-name", width='stretch',
                    on_click=_auto_name_run, disabled=_busy,
                    help="Generate a name from the date and your selections.")
      _full_out = os.path.join(results_root.strip() or "results",
                               run_name or "run")
      st.caption(f"Will be saved to:  `{_full_out}`")
      st.caption("Next: open the **Run** tab to choose analyses and start.")


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — Run
# ═══════════════════════════════════════════════════════════════════════════════
with tab_run:
    _screen_header("Run",
                   "Choose which analyses to run and start the pipeline.")
    _breadcrumb("Run")
    _primary_stages = [k for k in STAGES if k != "report"]
    st.metric("Analyses available", str(len(_primary_stages)),
              help="Segmentation + 4 analysis modules. "
                   "Add Handover Report for deliverable tables.")

    with st.container(border=True):
        st.subheader("Analyses")
        stage_labels = {k: v["label"] for k, v in STAGES.items()}
        # Continuous mode has no torque/angle, so efficiency is not offered.
        _stage_opts = [k for k in STAGES if not (_continuous and k == "efficiency")]
        _stage_default = [k for k in _primary_stages
                          if not (_continuous and k == "efficiency")]
        chosen = st.multiselect(
            "Analyses to run", options=_stage_opts,
            default=_stage_default, disabled=_busy,
            format_func=lambda k: stage_labels[k],
            help=("Vibration level, order and envelope run in parallel after "
                  "segmentation." if _continuous else
                  "The 4 analysis stages run in parallel after segmentation."))
        if _continuous:
            st.caption("Efficiency & torque are omitted in continuous mode "
                       "(they require torque / angle channels).")
        auto_include_segment = st.checkbox(
            "Prepare data automatically when needed", value=True, disabled=_busy,
            help="If later analyses are selected but the data hasn't been "
                 "segmented yet, run segmentation first.")

    with st.expander("Advanced parameters"):
        st.caption("Order / envelope tuning. The defaults suit the standard setup.")
        # Keyed (seeded via setdefault, no value=) so values persist across reruns
        # and can be pre-filled by "Load settings"; avoids the value=/state warning.
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
            samples_per_rev = st.number_input("samples_per_rev", step=1,
                                              key="adv_samples_per_rev",
                                              disabled=_busy)
            ball_pass_order = st.number_input("ball_pass_order", step=0.01,
                                              format="%.2f",
                                              key="adv_ball_pass_order",
                                              disabled=_busy)
            n_bpf_harmonics = st.number_input("n_bpf_harmonics", step=1,
                                              key="adv_n_bpf_harmonics",
                                              disabled=_busy)
            bp_filter_order = st.number_input("bp_filter_order", step=1,
                                              key="adv_bp_filter_order",
                                              disabled=_busy)
            min_revolutions = st.number_input(
                "min_revolutions", step=0.5, format="%.1f",
                key="adv_min_revolutions", disabled=_busy,
                help="Minimum ballscrew rotations per actuation segment to "
                     "include in order/envelope analysis. Typical actuations "
                     "cover 3–5 revolutions; raise for finer order resolution "
                     "at the cost of rejecting short strokes.")
        with ac2:
            bp_min_hz = st.number_input("bp_min_hz", step=50.0,
                                        key="adv_bp_min_hz", disabled=_busy,
                                        help="Envelope search floor at max RPM "
                                             "(2300 RPM → 1000 Hz). Scaled down "
                                             "automatically per actuation.")
            bp_min_hz_abs = st.number_input(
                "bp_min_hz_abs", step=10.0,
                key="adv_bp_min_hz_abs", disabled=_busy,
                help="Absolute minimum search frequency (Hz). "
                     "Never searches below this value regardless of RPM. "
                     "Raise to ~1000 for high-RPM-only datasets.")
            bp_min_bw_hz = st.number_input("bp_min_bw_hz", step=50.0,
                                           key="adv_bp_min_bw_hz", disabled=_busy)
            kurtogram_levels = st.number_input("kurtogram_levels", step=1,
                                               key="adv_kurtogram_levels",
                                               disabled=_busy)
            bp_max_hz = st.number_input(
                "bp_max_hz", step=500.0, key="adv_bp_max_hz", disabled=_busy,
                help="Upper bound (Hz) on the envelope resonance band — keep it "
                     "inside the accelerometer's calibrated range. Sensor here is "
                     "flat to 9 kHz (±5%) / 12 kHz (±10%); above ~12 kHz "
                     "the kurtogram latches onto sensor noise. Also capped at "
                     "0.8×Nyquist automatically.")

        st.divider()
        st.caption("Steady-state gating — analyse only the constant-speed "
                   "plateau of each actuation (excludes ramp-up/ramp-down). "
                   "Affects order, torque, efficiency and envelope.")
        pg1, pg2 = st.columns(2)
        with pg1:
            plateau_gating = st.checkbox(
                "Plateau gating", key="adv_plateau_gating", disabled=_busy,
                help="On: steady-state metrics use only the plateau, so ramps "
                     "don't bias torque/efficiency and don't collapse the "
                     "high-RPM order ceiling. Off: use the full stroke (legacy).")
        with pg2:
            plateau_frac = st.number_input(
                "plateau_frac", min_value=0.1, max_value=1.0, step=0.05,
                format="%.2f", key="adv_plateau_frac", disabled=_busy,
                help="Plateau = |RPM| ≥ this fraction of the nominal speed. "
                     "0.90 keeps the constant-speed portion; lower it if strokes "
                     "are short and too many fall back to full-stroke.")

        st.divider()
        st.caption("Campbell waterfall — order × measured-RPM map built from the "
                   "ramp sweep of each stroke (uses the ramp data the plateau "
                   "gate excludes elsewhere).")
        cb1, cb2 = st.columns(2)
        with cb1:
            campbell_window_rev = st.number_input(
                "campbell_window_rev", min_value=0.5, max_value=10.0, step=0.25,
                format="%.2f", key="adv_campbell_window_rev", disabled=_busy,
                help="Window length in revolutions. Order resolution ≈ 1/this; "
                     "shorter = finer speed axis but coarser orders.")
        with cb2:
            campbell_rpm_bin = st.number_input(
                "campbell_rpm_bin", min_value=5.0, max_value=500.0, step=5.0,
                format="%.0f", key="adv_campbell_rpm_bin", disabled=_busy,
                help="RPM-axis bin width for the Campbell map (smaller = more "
                     "speed bins).")

    st.session_state.setdefault("render_plots", True)
    render_plots = st.checkbox(
        "Generate plots", key="render_plots", disabled=_busy,
        help="Uncheck for a fast numbers-only run. "
             "CSV / parquet results are always written; "
             "PNG figures are skipped and can be generated later from the Review tab.")

    # The category multiselects already act as the analysis filter; no separate
    # only_parts / only_rpms inputs are needed.
    only_parts = ""
    only_rpms = ""

    # ── Readiness gating — disable the button with a clear reason ───────────────
    blockers = []
    if _continuous:
        if not single_file_path:
            blockers.append("choose a CSV file (Define & Setup > Data source)")
        elif not os.path.isfile(single_file_path):
            blockers.append("the CSV file path is not valid")
        if not (rpm_col or angle_col):
            blockers.append("map a speed reference — RPM or angle "
                            "(Define & Setup > Signals & columns)")
    else:
        if not data_dir:
            blockers.append("set a data folder (Define & Setup > Data source)")
        elif not os.path.isdir(data_dir):
            blockers.append("the data folder path is not valid")
    if not accel_cols_map:
        blockers.append("include at least one vibration signal "
                        "(Define & Setup > Signals & columns)")
    if not str(run_name).strip():
        blockers.append("name the run (Define & Setup > Run setup)")
    if not chosen:
        blockers.append("select at least one analysis above")

    # Surface the outcome of the previous run (set when the poller finished).
    _msg = st.session_state.pop("last_run_msg", None)
    if _msg:
        {"success": st.success, "warning": st.warning,
         "error": st.error}.get(_msg[0], st.info)(_msg[1])

    if _busy:
        # A run is in progress. The live poller (bar, checklist, abort) lives in
        # the sidebar so it stays visible on every tab — show a calm pointer here
        # rather than a second poller racing the same completion rerun.
        _job = st.session_state.get("run_job") or {}
        _state("loading",
               "An analysis is running. Live progress and the Abort button are "
               f"in the sidebar. Saving to {_job.get('out_root', '')}.")
    else:
        if blockers:
            st.warning("Before running, please " + "; ".join(blockers) + ".")
        run_clicked = st.button("Run analysis", type="primary",
                                width='stretch', disabled=bool(blockers))

        if run_clicked:
            _rr = results_root.strip() or "results"
            _rn = run_name or _generate_run_name(_as_str_list(parts),
                                                 _as_int_list(rpms))
            _computed_output_root = os.path.join(_rr, _rn)

            form = dict(
                analysis_mode=("continuous" if _continuous else "reciprocating"),
                single_file=single_file_path,
                nominal_rpm=nominal_rpm_val, signal_label=signal_label_val,
                data_dir=data_dir, output_root=_computed_output_root,
                parts=parts, rpms=rpms,
                time_col=time_col, rpm_col=rpm_col, angle_col=angle_col,
                torque_col=torque_col, accel_cols=accel_cols_map,
                samples_per_rev=samples_per_rev, ball_pass_order=ball_pass_order,
                n_bpf_harmonics=n_bpf_harmonics, bp_min_hz=bp_min_hz,
                bp_min_hz_abs=bp_min_hz_abs,
                bp_min_bw_hz=bp_min_bw_hz, kurtogram_levels=kurtogram_levels,
                bp_filter_order=bp_filter_order, min_revolutions=min_revolutions,
                bp_max_hz=bp_max_hz, plateau_gating=plateau_gating,
                plateau_frac=plateau_frac,
                campbell_window_rev=campbell_window_rev,
                campbell_rpm_bin=campbell_rpm_bin,
                only_parts=only_parts, only_rpms=only_rpms,
                render_plots=render_plots,
            )
            # Build cfg up-front so path helpers (seg_data_dir) work before the run.
            cfg = _config_from_form(form)
            set_active(cfg)

            # A 'segment' run with zero matching raw CSVs is a silent no-op.
            # (Continuous mode reads one explicit file, so this folder scan is
            # skipped — validity was already checked by the readiness gating.)
            if "segment" in chosen and not _continuous:
                scan = _discover(cfg.data_dir, _dir_mtime(cfg.data_dir))
                if scan["matched"] == 0:
                    st.error("No recognized CSV files were found in:\n\n"
                             f"{cfg.data_dir}\n\nNothing to process.")
                    st.stop()

            # Segment-first safety: every later stage reads segmented_data.
            chosen_stages = list(chosen)
            if ("segment" not in chosen_stages
                    and not os.path.isdir(seg_data_dir(cfg))):
                if auto_include_segment:
                    chosen_stages = ["segment"] + chosen_stages
                    st.info("Preparing data (segmentation) first so the selected "
                            "analyses have input.")
                else:
                    st.error("This data hasn't been prepared yet. Enable "
                             "\"Prepare data automatically\", or add the "
                             "segmentation analysis.")
                    st.stop()

            # Keep canonical order so segmentation runs first in the child.
            chosen_stages = [s for s in STAGES if s in chosen_stages]

            # Persist the resolved config next to results (reproducible run).
            os.makedirs(cfg.output_root, exist_ok=True)
            cfg_path = os.path.join(cfg.output_root, "nvh_config.json")
            cfg.to_json(cfg_path)
            os.environ["NVH_CONFIG"] = os.path.abspath(cfg_path)

            # Spawn the pipeline as a child process and switch to the poller.
            st.session_state["run_job"] = _spawn_run(
                os.path.abspath(cfg_path), chosen_stages, cfg.output_root)
            st.session_state["running"] = True
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — Review  (sub-ribbon: Run library · Open runs)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_review:
    _screen_header("Review",
                   "Browse saved runs and explore their results.")
    _breadcrumb("Review")
    _opened_count = len([m for m in st.session_state.get("opened_runs", [])
                         if os.path.isfile(m)])
    _kpi_row([
        {"label": "Run library", "value": f"{len(past_runs)} saved"},
        {"label": "Open runs", "value": str(_opened_count)},
    ])

    sub_lib, sub_open = st.tabs(["Run library", "Open runs"])

    with sub_lib:
        if not past_runs:
            _state("empty",
                   "No saved runs yet. Complete an analysis in the Run tab to "
                   "get started.")
        else:
            # Unified Filters popover + dismissible chips over groups / speeds.
            _all_groups = sorted({str(g) for pr in past_runs
                                  for g in pr.get("parts", [])})
            _all_speeds = sorted({str(s) for pr in past_runs
                                  for s in pr.get("rpms", [])}, key=str)
            _flt = ({} if not (_all_groups or _all_speeds)
                    else _filter_bar({"Group": _all_groups, "Speed": _all_speeds},
                                     key="lib_filter"))

            def _matches(pr):
                gsel, ssel = _flt.get("Group", []), _flt.get("Speed", [])
                if gsel and not ({str(g) for g in pr.get("parts", [])} & set(gsel)):
                    return False
                if ssel and not ({str(s) for s in pr.get("rpms", [])} & set(ssel)):
                    return False
                return True

            _visible = [pr for pr in past_runs if _matches(pr)]
            if not _visible:
                _state("empty",
                       "No saved runs match the active filters. Clear a chip "
                       "above to widen the search.")
            for pr in _visible[:25]:
                lc, mc, rc, rc2 = st.columns([5, 3, 1, 1])
                lc.markdown(f"**{pr['name']}**  \n"
                            f"<small>{pr['generated'] or 'unknown date'}</small>",
                            unsafe_allow_html=True)
                mc.markdown(f"{pr['stages_ok']}/{pr['stages_total']} analyses ok  \n"
                            f"<small>{len(pr['parts'])} group(s) · "
                            f"{len(pr['rpms'])} speed(s)</small>",
                            unsafe_allow_html=True)
                # Open is read-only (safe mid-run); Load settings overwrites the
                # locked form, so it stays disabled while a run is in progress.
                rc.button("Open", key=f"open_{pr['name']}",
                          width='stretch',
                          on_click=_open_run, args=(pr["manifest"],))
                rc2.button("Load settings", key=f"load_{pr['name']}",
                           width='stretch', disabled=_busy,
                           on_click=_load_run_settings, args=(pr["manifest"],),
                           help="Fill Define & Setup with this run's settings to "
                                "re-run or tweak.")
            st.caption("Opened runs appear in the Open runs tab. "
                       "Load settings pre-fills Define & Setup.")

    with sub_open:
        # Drop any manifests that have been deleted since last rerun.
        opened = [m for m in st.session_state.get("opened_runs", [])
                  if os.path.isfile(m)]
        st.session_state["opened_runs"] = opened

        if not opened:
            _state("empty",
                   "Open a run from the Run library tab to view its results here.")
        else:
            def _run_label(m):
                return os.path.basename(os.path.dirname(m)) or m
            inner = st.tabs([_run_label(m) for m in opened])
            for ti, (itab, m) in enumerate(zip(inner, opened)):
                with itab:
                    # The per-run title, Download config and Close all live in
                    # the screen-header utility belt inside _render_results.
                    _render_results(m, key_prefix=f"open{ti}",
                                    on_close=(lambda mm=m: _close_run(mm)))
