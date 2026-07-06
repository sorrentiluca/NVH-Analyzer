"""
famos_entry — the one function imc FAMOS calls.

FAMOS's Python Kit executes a script in-process; this module gives it a
single, stable entry point that runs the pipeline from a config file and
returns the manifest path (which the FAMOS panel then reads to load the
produced PNG/CSV files).  See famos/SETUP_FAMOS.md for the panel/sequence
templates.

Usage from a FAMOS sequence (PyCodeExecute / exec_python_file):

    from nvh_pipeline.famos_entry import run_pipeline
    manifest_path = run_pipeline(r"C:/path/to/nvh_config.json")
"""

from __future__ import annotations

import os
from typing import Optional

from .config import PipelineConfig, set_active
from .runner import run_all


def run_pipeline(config_path: Optional[str] = None,
                 stages: Optional[list] = None,
                 data_dir: Optional[str] = None) -> str:
    """Run the pipeline and return the absolute manifest.json path.

    ``config_path`` defaults to $NVH_CONFIG / ./nvh_config.json / built-in
    defaults; ``stages`` defaults to the config's own list; ``data_dir``
    overrides the raw-data folder without editing the file.
    """
    os.environ.setdefault("MPLBACKEND", "Agg")   # never open GUI windows
    if config_path:
        config_path = os.path.abspath(config_path)
        os.environ["NVH_CONFIG"] = config_path
        cfg = PipelineConfig.from_json(config_path)
    else:
        from .config import active
        cfg = active()

    if data_dir:
        cfg.data_dir = data_dir
    cfg.show_plots = False                       # headless inside FAMOS
    set_active(cfg)

    manifest = run_all(cfg, stages=stages or list(cfg.stages),
                       config_path=config_path)
    return os.path.join(manifest["output_root"], "manifest.json")
