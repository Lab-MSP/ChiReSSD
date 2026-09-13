"""``chiressd-prepare`` — turn a corpus into StyleTTS2 training lists.

Takes ``filename|text[|speaker_id]`` metadata, phonemizes each transcript, and
writes the ``filename|phonemes|speaker_id`` lists the trainer reads.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..data.prepare_lists import (
    PrepareError,
    build_entries,
    read_metadata,
    split_entries,
    write_list,
)
from ..paths import PathsError, load_paths
from ..text import DEFAULT_LANGUAGE, TextError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chiressd-prepare", description=__doc__)
    parser.add_argument("metadata", type=Path, help="filename|text[|speaker_id] per line.")
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=None,
        help="Check audio exists (default: datasets.train.audio).",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--language",
        default=DEFAULT_LANGUAGE,
        help=f"Phonemizer language (default: {DEFAULT_LANGUAGE}).",
    )
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--paths", type=Path, default=None)
    parser.add_argument(
        "--keep-missing",
        action="store_true",
        help="Fail instead of skipping rows whose audio is absent.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    audio_dir, out_dir = args.audio_dir, args.out_dir
    if audio_dir is None or out_dir is None:
        try:
            paths = load_paths(args.paths)
            audio_dir = audio_dir or paths["datasets.train.audio"]
            out_dir = out_dir or paths["datasets.train.train_list"].parent
        except PathsError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    try:
        rows = read_metadata(args.metadata)
        print(f"{len(rows)} row(s) from {args.metadata}")
        entries = build_entries(
            rows,
            audio_dir=audio_dir,
            language=args.language,
            skip_missing=not args.keep_missing,
            progress=True,
        )
        train, val = split_entries(entries, val_fraction=args.val_fraction, seed=args.seed)
        train_path = write_list(Path(out_dir) / "train_list.txt", train)
        val_path = write_list(Path(out_dir) / "val_list.txt", val)
    except (PrepareError, TextError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"\ntrain: {len(train)} -> {train_path}")
    print(f"val:   {len(val)} -> {val_path}")
    if entries:
        print(f"\nexample: {entries[0].to_line()[:100]}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
