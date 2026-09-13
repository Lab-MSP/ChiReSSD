"""``chiressd-synth`` — batch reconstruction from a manifest."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from ..audio import SAMPLE_RATE, write_audio
from ..config import ConfigError, available_presets, load_preset
from ..manifest import ManifestError, group_by_reference, read_manifest
from ..paths import PathsError, load_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chiressd-synth",
        description="Reconstruct utterances listed in a manifest.",
    )
    parser.add_argument("manifest", nargs="?", type=Path, help="TSV/CSV of key,text,reference.")
    parser.add_argument(
        "--preset",
        default="default",
        help="Inference preset (default: 'default', the released operating point).",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Local path or hf:// reference (default: the released checkpoint).",
    )
    parser.add_argument("--config", default=None, help="Training config for the checkpoint.")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--prefix", default="NC_", help="Prefix for output filenames; evaluation pairs on it."
    )
    parser.add_argument("--paths", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed the sampler. Synthesis is stochastic without it.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--list-presets", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_presets:
        for name in available_presets():
            print(load_preset(name))
        return 0

    if args.manifest is None:
        print("error: a manifest is required (or use --list-presets).", file=sys.stderr)
        return 2

    try:
        paths = load_paths(args.paths)
        preset = load_preset(args.preset)
        jobs = read_manifest(args.manifest)
    except (PathsError, ConfigError, ManifestError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    out_dir = args.out_dir or paths["outputs.synth"] / preset.name
    out_dir.mkdir(parents=True, exist_ok=True)
    style_cache = paths["outputs.styles"] / preset.name

    print(f"preset:  {preset}")
    print(f"jobs:    {len(jobs)} utterance(s), {len(group_by_reference(jobs))} reference(s)")
    print(f"output:  {out_dir}")
    if args.seed is None:
        print("note:    no --seed given; synthesis is stochastic and will not repeat exactly.")

    from ..model import ChiReSSD
    from ..styles import get_or_compute_style

    model = ChiReSSD.from_pretrained(
        checkpoint=args.checkpoint,
        config=args.config,
        upstream=paths.upstream,
        device=args.device,
    )

    written, skipped = 0, 0
    # Reference on the outside: style extraction is the expensive part.
    for reference, reference_jobs in group_by_reference(jobs).items():
        if not Path(reference).is_file():
            print(f"  missing reference, skipping {len(reference_jobs)} job(s): {reference}")
            skipped += len(reference_jobs)
            continue

        ref_s = get_or_compute_style(model, reference, style_cache)
        for job in reference_jobs:
            destination = out_dir / f"{args.prefix}{job.key}"
            if destination.suffix != ".wav":
                destination = destination.with_suffix(".wav")
            if destination.exists() and not args.overwrite:
                skipped += 1
                continue
            wav = model.synthesize(job.target_text, ref_s, seed=args.seed, **preset.as_kwargs())
            write_audio(destination, wav, SAMPLE_RATE)
            written += 1

    record = {
        "preset": preset.name,
        "preset_values": preset.as_kwargs(),
        "seed": args.seed,
        "checkpoint": str(model.checkpoint),
        "manifest": str(args.manifest),
        "written": written,
        "skipped": skipped,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out_dir / "run.json").write_text(json.dumps(record, indent=2), encoding="utf-8")

    print(f"\nwrote {written}, skipped {skipped}. Provenance: {out_dir / 'run.json'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
