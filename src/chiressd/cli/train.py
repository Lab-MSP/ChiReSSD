"""``chiressd-train`` — fine-tune StyleTTS2 on child disordered speech.

Renders the selected recipe into a concrete config (upstream cannot interpolate
``${...}``), then hands it to upstream's ``train_finetune.py``. The rendered
config is kept beside the run so the exact paths and hyperparameters used are
recoverable afterwards.
"""

from __future__ import annotations

import argparse
import runpy
import sys
from datetime import datetime
from pathlib import Path

from ..config import (
    ConfigError,
    available_train_configs,
    load_train_config,
    render_train_config,
)
from ..paths import PathsError, load_paths
from ..vendor import UpstreamError, activate, require_upstream


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chiressd-train",
        description="Fine-tune StyleTTS2 on child disordered speech.",
    )
    parser.add_argument(
        "--config",
        default="config_ft_F0_v5",
        help="Training recipe name or path (default: config_ft_F0_v5, the released recipe).",
    )
    parser.add_argument("--paths", type=Path, default=None, help="Path to a paths.yaml.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Where to write the rendered config (default: <work_root>/run-<timestamp>).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Render and print the config, then stop without training.",
    )
    parser.add_argument(
        "--list", action="store_true", help="List available training recipes and exit."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list:
        for name in available_train_configs():
            print(name)
        return 0

    try:
        paths = load_paths(args.paths)
        upstream = require_upstream(paths.upstream)
        config = load_train_config(args.config, paths, upstream=upstream)
    except (PathsError, ConfigError, UpstreamError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    run_dir = args.run_dir or paths.work_root / f"run-{datetime.now():%Y%m%d-%H%M%S}"
    try:
        rendered = render_train_config(config, Path(run_dir) / "config.yml")
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"paths:    {paths.source}")
    print(f"upstream: {upstream}")
    print(f"config:   {rendered}")
    print(f"log_dir:  {config.get('log_dir')}")
    print(
        "recipe:   "
        f"{config.get('epochs')} epochs, batch {config.get('batch_size')}, "
        f"lambda_F0 {config.get('loss_params', {}).get('lambda_F0')}, "
        f"diff_epoch {config.get('loss_params', {}).get('diff_epoch')}, "
        f"joint_epoch {config.get('loss_params', {}).get('joint_epoch')}"
    )

    if args.dry_run:
        print("\n--dry-run: rendered config written, not training.")
        return 0

    missing = [
        key
        for key in ("pretrained_model", "data_params.train_data", "data_params.root_path")
        for value in [_dig(config, key)]
        if value is None or not Path(value).exists()
    ]
    if missing:
        print(
            "error: these inputs do not exist on disk:\n  "
            + "\n  ".join(f"{k} -> {_dig(config, k)}" for k in missing)
            + "\nEdit paths.yaml, and see DATA.md for how to obtain the corpora.",
            file=sys.stderr,
        )
        return 1

    # Upstream exposes training only as a script, so run it as one. Two things
    # are needed for that and neither is automatic:
    #
    #   sys.path -- `python script.py` puts the script's directory on sys.path,
    #     but runpy.run_path() does not, so upstream's `from meldataset import
    #     ...` would fail. activate() inserts the checkout.
    #   cwd -- upstream resolves its frozen helper models relative to the
    #     working directory, so chdir into it for the duration of the call.
    trainer = upstream / "train_finetune.py"
    if not trainer.is_file():
        print(f"error: {trainer} not found; run chiressd-setup.", file=sys.stderr)
        return 1

    import os

    activate(upstream)

    previous = Path.cwd()
    os.chdir(upstream)
    argv_backup = sys.argv[:]
    sys.argv = [str(trainer), "--config_path", str(rendered)]
    try:
        runpy.run_path(str(trainer), run_name="__main__")
    finally:
        sys.argv = argv_backup
        os.chdir(previous)
    return 0


def _dig(config: dict, dotted: str):
    node = config
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
