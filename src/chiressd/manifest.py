"""The synthesis manifest: what to say, in whose voice, written where.

One row per utterance, as TSV or CSV:

    key         a stable identifier; also the output filename stem
    text        the target transcript to synthesize
    reference   path to the style reference recording
    repeat      optional; repeat the text N times (the three-productions
                protocol used by the clinical prompts)

Keeping this in a file rather than in code is what lets the same corpus be
re-synthesized under several presets without editing anything.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

REQUIRED_COLUMNS = ("key", "text", "reference")


class ManifestError(RuntimeError):
    pass


@dataclass(frozen=True)
class Job:
    key: str
    text: str
    reference: Path
    repeat: int = 1

    @property
    def target_text(self) -> str:
        return " ".join([self.text] * self.repeat) if self.repeat > 1 else self.text


def _dialect(path: Path):
    return csv.excel_tab if path.suffix.lower() in (".tsv", ".tab") else csv.excel


def read_manifest(path: str | Path, *, root: str | Path | None = None) -> list[Job]:
    """Read a manifest; relative reference paths resolve against ``root``."""
    path = Path(path)
    if not path.is_file():
        raise ManifestError(f"No manifest at {path}")

    root = Path(root) if root else path.parent
    jobs: list[Job] = []

    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, dialect=_dialect(path))
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ManifestError(
                f"{path} is missing column(s): {', '.join(missing)}. "
                f"Required: {', '.join(REQUIRED_COLUMNS)}"
            )

        seen: set[str] = set()
        for line_no, row in enumerate(reader, start=2):
            key = (row.get("key") or "").strip()
            text = (row.get("text") or "").strip()
            reference = (row.get("reference") or "").strip()
            if not key or not text or not reference:
                raise ManifestError(f"{path}:{line_no}: key, text and reference are all required.")
            if key in seen:
                raise ManifestError(f"{path}:{line_no}: duplicate key {key!r}.")
            seen.add(key)

            ref_path = Path(reference)
            jobs.append(
                Job(
                    key=key,
                    text=text,
                    reference=ref_path if ref_path.is_absolute() else root / ref_path,
                    repeat=int(row.get("repeat") or 1),
                )
            )
    return jobs


def write_manifest(path: str | Path, jobs: Iterable[Job]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, dialect=_dialect(path))
        writer.writerow([*REQUIRED_COLUMNS, "repeat"])
        for job in jobs:
            writer.writerow([job.key, job.text, str(job.reference), job.repeat])
    return path


def group_by_reference(jobs: Sequence[Job]) -> dict[Path, list[Job]]:
    """Group jobs by style reference.

    Style extraction is the expensive part and depends only on the reference,
    so synthesis loops over references on the outside and utterances inside.
    """
    grouped: dict[Path, list[Job]] = {}
    for job in jobs:
        grouped.setdefault(job.reference, []).append(job)
    return grouped
