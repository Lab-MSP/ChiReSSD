"""``chiressd-style`` — precompute and cache style vectors for a reference directory.

Style extraction runs two encoders over a mel-spectrogram. Doing it once per
speaker rather than once per utterance is the difference between a cheap
synthesis run and an expensive one.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..paths import PathsError, load_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chiressd-style",
        description="Precompute style vectors for every recording in a directory.",
    )
    parser.add_argument("reference_dir", type=Path, help="Directory of reference recordings.")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--paths", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--suffix", default=".wav")
    parser.add_argument("--refresh", action="store_true", help="Recompute cached vectors.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        paths = load_paths(args.paths)
    except PathsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    references = sorted(p for p in args.reference_dir.glob(f"*{args.suffix}"))
    if not references:
        print(f"error: no {args.suffix} files in {args.reference_dir}", file=sys.stderr)
        return 1

    cache_dir = args.cache_dir or paths["outputs.styles"]

    from ..model import ChiReSSD
    from ..styles import get_or_compute_style

    model = ChiReSSD.from_pretrained(
        checkpoint=args.checkpoint,
        config=args.config,
        upstream=paths.upstream,
        device=args.device,
    )

    for reference in references:
        get_or_compute_style(model, reference, cache_dir, refresh=args.refresh)
        print(f"  {reference.name}")

    print(f"\ncached {len(references)} style vector(s) in {cache_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
