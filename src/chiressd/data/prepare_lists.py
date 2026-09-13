"""Building the training lists StyleTTS2 expects.

The trainer reads pipe-delimited lines::

    filename.wav|ɪt hɐz ðə fˈiːl|0

The middle field is **phonemes, not text**. Getting this wrong is quiet rather
than loud: training proceeds and simply learns from mis-specified input, so the
conversion happens here, once, rather than being left to each caller.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from ..text import DEFAULT_LANGUAGE, phonemize

#: StyleTTS2's list format is pipe-delimited, so the fields cannot contain it.
SEPARATOR = "|"


class PrepareError(RuntimeError):
    pass


@dataclass(frozen=True)
class Entry:
    filename: str
    phonemes: str
    speaker_id: int = 0

    def to_line(self) -> str:
        return f"{self.filename}{SEPARATOR}{self.phonemes}{SEPARATOR}{self.speaker_id}"


def read_metadata(path: str | Path) -> list[tuple[str, str, int]]:
    """Read ``filename|text|speaker_id`` rows.

    The speaker id is optional and defaults to 0, which is what a single-speaker
    or speaker-agnostic corpus wants.
    """
    path = Path(path)
    if not path.is_file():
        raise PrepareError(f"No metadata file at {path}")

    rows: list[tuple[str, str, int]] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(SEPARATOR)
        if len(parts) < 2:
            raise PrepareError(
                f"{path}:{line_no}: expected 'filename|text[|speaker_id]', got {line!r}"
            )
        filename, text = parts[0].strip(), parts[1].strip()
        speaker = int(parts[2]) if len(parts) > 2 and parts[2].strip() else 0
        rows.append((filename, text, speaker))
    return rows


def build_entries(
    rows: Iterable[tuple[str, str, int]],
    *,
    audio_dir: str | Path | None = None,
    language: str = DEFAULT_LANGUAGE,
    with_stress: bool = False,
    tokenize: bool = False,
    skip_missing: bool = True,
    progress: bool = False,
) -> list[Entry]:
    """Phonemize each transcript, optionally dropping rows with no audio.

    ``with_stress=False`` and ``tokenize=False`` are the defaults: that is the
    convention the training lists use. It differs from the synthesis path, which
    phonemizes *with* stress -- a train/inference asymmetry inherited from the
    original pipeline. Set both to True to phonemize the way synthesis does.
    """
    audio_dir = Path(audio_dir) if audio_dir else None
    entries: list[Entry] = []
    missing = 0

    for filename, text, speaker in rows:
        if audio_dir is not None and not (audio_dir / filename).is_file():
            missing += 1
            if skip_missing:
                continue
            raise PrepareError(f"No audio for {filename} in {audio_dir}")

        phonemes = phonemize(text, language, with_stress=with_stress, tokenize=tokenize).strip()
        if SEPARATOR in phonemes:
            # Would silently corrupt the three-field format.
            raise PrepareError(f"Phonemes for {filename!r} contain {SEPARATOR!r}.")
        if not phonemes:
            continue
        entries.append(Entry(filename=filename, phonemes=phonemes, speaker_id=speaker))
        if progress and len(entries) % 200 == 0:
            print(f"  phonemized {len(entries)}")

    if missing:
        print(f"  skipped {missing} row(s) with no audio file")
    return entries


def split_entries(
    entries: Sequence[Entry], *, val_fraction: float = 0.15, seed: int = 0
) -> tuple[list[Entry], list[Entry]]:
    """Shuffle and split into train and validation.

    Shuffling is seeded so the split is reproducible; an unshuffled split would
    follow filename order, which in these corpora groups by speaker and session.
    """
    if not 0.0 <= val_fraction < 1.0:
        raise PrepareError("val_fraction must be in [0, 1).")

    shuffled = list(entries)
    random.Random(seed).shuffle(shuffled)
    cut = int(len(shuffled) * val_fraction)
    return shuffled[cut:], shuffled[:cut]


def write_list(path: str | Path, entries: Iterable[Entry]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(entry.to_line() for entry in entries) + "\n", encoding="utf-8")
    return path
