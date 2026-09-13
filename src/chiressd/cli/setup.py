"""``chiressd-setup`` — prepare the upstream StyleTTS2 checkout.

Run this once after installing. It clones StyleTTS2 at the pinned commit,
applies the ChiReSSD patches, and checks that the frozen helper weights came
with it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..paths import PathsError, load_paths
from ..vendor import UpstreamError, ensure_upstream, patch_files, verify_upstream


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chiressd-setup",
        description="Clone and patch the upstream StyleTTS2 checkout that ChiReSSD builds on.",
    )
    parser.add_argument("--paths", type=Path, default=None, help="Path to a paths.yaml.")
    parser.add_argument(
        "--upstream",
        type=Path,
        default=None,
        help="Where to put the checkout (default: the 'upstream' key in paths.yaml).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report status and exit non-zero if setup is needed; change nothing.",
    )
    parser.add_argument(
        "--offline", action="store_true", help="Never clone; only verify and patch."
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-verify and re-patch an existing checkout."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    dest = args.upstream
    if dest is None:
        try:
            dest = load_paths(args.paths).upstream
        except PathsError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    if args.check:
        status = verify_upstream(dest)
        print(status.explain())
        return 0 if status.ok else 1

    print(f"Preparing StyleTTS2 in {dest}")
    print(f"Patches: {', '.join(p.name for p in patch_files()) or 'none found'}")
    try:
        status = ensure_upstream(dest, offline=args.offline, force=args.force)
    except (UpstreamError, PathsError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(status.explain())
    if status.missing_assets:
        print(
            "\nwarning: pretrained helper weights are missing from the checkout.\n"
            "They ship inside the upstream repository; a partial or filtered clone is "
            "the usual cause. Re-clone without filters.",
            file=sys.stderr,
        )
        return 1

    print("\nNext: copy configs/paths.example.yaml to paths.yaml and set data_root.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
