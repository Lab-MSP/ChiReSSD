"""``chiressd-eval`` — compare reconstructions against their originals.

    chiressd-eval speaker --hypothesis <recon_dir> --reference <orig_dir>
    chiressd-eval pitch   --hypothesis <recon_dir> --reference <orig_dir>

``speaker`` is cosine similarity of Resemblyzer embeddings: did the voice
survive? ``pitch`` is the F0 difference in percent and semitones: no learned
representation, so no bias against child or disordered voices. Read them
together.

Files are matched by key — ``NC_<original>.wav`` with ``<original>.wav`` —
never by directory-listing order, which is filesystem-dependent.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..metrics.pairing import PairingError, pair_by_prefix
from ..paths import PathsError, load_paths

# --- shared argument groups ----------------------------------------------


def _add_output_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--paths", type=Path, default=None)
    parser.add_argument("--name", default=None, help="Label for the output files.")


def _add_pair_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--hypothesis", type=Path, required=True, help="Directory of reconstructed audio."
    )
    parser.add_argument(
        "--reference", type=Path, required=True, help="Directory of original recordings."
    )
    parser.add_argument(
        "--lenient",
        action="store_true",
        help="Skip reconstructions with no matching original instead of failing.",
    )
    _add_output_args(parser)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chiressd-eval",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="metric", required=True)

    speaker = sub.add_parser("speaker", help="Resemblyzer speaker similarity.")
    _add_pair_args(speaker)
    speaker.add_argument("--device", default=None)

    pitch = sub.add_parser("pitch", help="F0, semitone and percent difference.")
    _add_pair_args(pitch)
    pitch.add_argument("--fmin", type=float, default=25.0)
    pitch.add_argument("--fmax", type=float, default=800.0)
    pitch.add_argument(
        "--voicing",
        choices=["none", "pyin"],
        default="none",
        help="'none' (default) averages over every frame: yin never returns NaN, "
        "so silence is included. 'pyin' keeps only voiced frames.",
    )

    return parser


# --- helpers --------------------------------------------------------------


def _resolve_out_dir(args) -> Path:
    if getattr(args, "out_dir", None):
        return args.out_dir
    try:
        return load_paths(getattr(args, "paths", None))["outputs.metrics"]
    except PathsError:
        return Path.cwd()


def _print_summary(label: str, summary: dict) -> None:
    print(f"\n{label}:")
    for key, value in summary.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")


def _write(out_dir: Path, metric: str, name: str, table, summary: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{metric}_{name}.csv"
    json_path = out_dir / f"{metric}_{name}.json"
    table.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nper-row:  {csv_path}\nsummary:  {json_path}")


# --- metric handlers ------------------------------------------------------


def _run_pairwise(args, out_dir: Path, name: str) -> int:
    try:
        pairs = pair_by_prefix(args.hypothesis, args.reference, strict=not args.lenient)
    except PairingError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not pairs:
        print("error: no reconstruction/original pairs found.", file=sys.stderr)
        return 1

    print(f"{len(pairs)} pair(s) from {args.hypothesis}")

    if args.metric == "speaker":
        from ..metrics.speaker import similarities, summarize

        table = similarities(pairs, device=args.device)
        summary = summarize(table)
    else:
        from ..metrics.pitch import pitch_table, summarize

        table = pitch_table(pairs, fmin=args.fmin, fmax=args.fmax, voicing=args.voicing)
        summary = summarize(table)

    _print_summary(f"{args.metric} summary for {name}", summary)
    _write(out_dir, args.metric, name, table, summary)
    return 0


# --- entry point ----------------------------------------------------------


#: resemblyzer is an optional extra, so name it rather than letting an
#: ImportError traceback reach the user.
_EXTRA_FOR_METRIC = {"speaker": "eval-speaker"}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = _resolve_out_dir(args)

    try:
        if args.metric in ("speaker", "pitch"):
            return _run_pairwise(args, out_dir, args.name or args.hypothesis.name)
    except ImportError as exc:
        extra = _EXTRA_FOR_METRIC.get(args.metric)
        hint = f'\n  pip install "chiressd[{extra}]"' if extra else ""
        print(f"error: {exc}{hint}", file=sys.stderr)
        return 2

    raise SystemExit(f"unknown metric {args.metric}")  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
