"""
CLI:  python -m nvh_pipeline [--config nvh_config.json] [--stages a,b,c]
                             [--data-dir DIR] [--progress-file PATH]
      python -m nvh_pipeline --write-example nvh_config.json

Exit code = number of failed stages (0 = clean run), so it doubles as a CI
success check.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .config import PipelineConfig, set_active, write_example
from .runner import STAGES, run_all


def _progress_writer(path: str):
    """Append-events JSON progress file the Streamlit app polls."""
    state = {"events": [], "done": False}

    def _flush() -> None:
        tmp = path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(state, fh)
            os.replace(tmp, path)
        except OSError:
            pass

    def cb(event: dict) -> None:
        state["events"].append(event)
        _flush()

    return state, _flush, cb


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m nvh_pipeline",
        description="Ballscrew recirculation NVH analysis pipeline.")
    ap.add_argument("--config", default=None,
                    help="path to nvh_config.json (default: $NVH_CONFIG, "
                         "then ./nvh_config.json, then built-in defaults)")
    ap.add_argument("--stages", default=None,
                    help=f"comma-separated subset of {', '.join(STAGES)} "
                         "(default: the config's 'stages' list)")
    ap.add_argument("--data-dir", default=None,
                    help="override the config's raw-data folder")
    ap.add_argument("--output-root", default=None,
                    help="override the config's output root")
    ap.add_argument("--progress-file", default=None,
                    help="write live progress events to this JSON file")
    ap.add_argument("--serial", action="store_true",
                    help="disable the parallel stage pool (debugging)")
    ap.add_argument("--write-example", metavar="PATH", default=None,
                    help="write a starter config file and exit")
    args = ap.parse_args(argv)

    if args.write_example:
        write_example(args.write_example)
        print(f"Example config written -> {os.path.abspath(args.write_example)}")
        return 0

    config_path = None
    if args.config:
        config_path = os.path.abspath(args.config)
        cfg = PipelineConfig.from_json(config_path)
        os.environ["NVH_CONFIG"] = config_path
    else:
        env = os.environ.get("NVH_CONFIG")
        if env and os.path.isfile(env):
            config_path = os.path.abspath(env)
            cfg = PipelineConfig.from_json(config_path)
        elif os.path.isfile("nvh_config.json"):
            config_path = os.path.abspath("nvh_config.json")
            cfg = PipelineConfig.from_json(config_path)
            os.environ["NVH_CONFIG"] = config_path
        else:
            cfg = PipelineConfig()

    if args.data_dir:
        cfg.data_dir = args.data_dir
    if args.output_root:
        cfg.output_root = args.output_root
    set_active(cfg)

    stages = ([s.strip() for s in args.stages.split(",") if s.strip()]
              if args.stages else list(cfg.stages))
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        ap.error(f"unknown stage(s) {unknown} — valid: {list(STAGES)}")

    state = flush = cb = None
    if args.progress_file:
        state, flush, cb = _progress_writer(args.progress_file)

    try:
        manifest = run_all(cfg, stages=stages, progress_cb=cb,
                           config_path=config_path,
                           parallel=not args.serial)
    except Exception as exc:
        if state is not None:
            state["done"] = True
            state["error"] = f"{type(exc).__name__}: {exc}"
            flush()
        raise

    if state is not None:
        state["done"] = True
        state["manifest"] = os.path.join(manifest["output_root"],
                                         "manifest.json")
        flush()

    return sum(1 for r in manifest["stages"] if r["status"] != "ok")


if __name__ == "__main__":
    sys.exit(main())
