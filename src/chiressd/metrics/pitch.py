"""Pitch comparison between a reconstruction and its original.

Complements the learned speaker embedding with something that has no training
data behind it and so cannot be biased against child or disordered voices: how
close the reconstruction's fundamental frequency sits to the original's,
reported in percent and in semitones.

Roughly three semitones is the range where listeners start to hear a pitch
difference as a different voice, which is the scale to read these against.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import log2
from pathlib import Path
from typing import Literal

import numpy as np

from .pairing import Pair

#: Wide enough for children's voices at the top and adult male at the bottom.
DEFAULT_FMIN = 25.0
DEFAULT_FMAX = 800.0
FRAME_LENGTH = 2048
HOP_LENGTH = 256


@dataclass(frozen=True)
class F0Stats:
    geom_mean: float
    median: float
    mean: float
    n_frames: int
    n_used: int


def f0_stats(
    path: str | Path,
    *,
    fmin: float = DEFAULT_FMIN,
    fmax: float = DEFAULT_FMAX,
    frame_length: int = FRAME_LENGTH,
    hop_length: int = HOP_LENGTH,
    voicing: Literal["none", "pyin"] = "none",
) -> F0Stats | None:
    """Summarise a file's F0 contour, defaulting to the geometric mean.

    The geometric mean is the right average for pitch because pitch is
    perceived multiplicatively -- it is the arithmetic mean in the log domain.

    ``voicing`` controls which frames count. The default, ``"none"``, uses every
    frame: ``librosa.yin`` returns an estimate for *every* frame and never NaN,
    so silence and unvoiced segments are included in the average. ``"pyin"``
    applies librosa's voicing decision and keeps only voiced frames, which
    tracks the speaking voice more closely but is slower and changes the
    numbers, so switch deliberately rather than by default.
    """
    import librosa

    y, sr = librosa.load(str(path), sr=None)
    if y.size == 0:
        return None

    if voicing == "pyin":
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=fmin, fmax=fmax, sr=sr, frame_length=frame_length, hop_length=hop_length
        )
        f0 = np.asarray(f0, dtype=float)
        used = f0[np.asarray(voiced_flag, dtype=bool) & ~np.isnan(f0)]
    else:
        f0 = np.asarray(
            librosa.yin(
                y,
                fmin=fmin,
                fmax=fmax,
                sr=sr,
                frame_length=frame_length,
                hop_length=hop_length,
            ),
            dtype=float,
        )
        used = f0[~np.isnan(f0)]

    used = used[used > 0]
    if used.size == 0:
        return None

    return F0Stats(
        geom_mean=float(np.exp(np.mean(np.log(used)))),
        median=float(np.median(used)),
        mean=float(np.mean(used)),
        n_frames=int(f0.size),
        n_used=int(used.size),
    )


def semitone_difference(f_ref: float | None, f_rec: float | None) -> float | None:
    """Signed difference in semitones; positive means the reconstruction is higher."""
    if f_ref is None or f_rec is None or f_ref <= 0 or f_rec <= 0:
        return None
    return 12.0 * log2(f_rec / f_ref)


def percent_difference(f_ref: float | None, f_rec: float | None) -> float | None:
    """Absolute difference as a percentage of the mean of the two.

    Symmetric, so it does not depend on which recording is called the
    reference.
    """
    if f_ref is None or f_rec is None or (f_ref + f_rec) == 0:
        return None
    return abs(f_rec - f_ref) / ((f_ref + f_rec) / 2.0) * 100.0


def pitch_table(pairs: Sequence[Pair], **kwargs):
    import pandas as pd

    rows = []
    for pair in pairs:
        ref = f0_stats(pair.reference, **kwargs)
        rec = f0_stats(pair.hypothesis, **kwargs)
        f_ref = ref.geom_mean if ref else None
        f_rec = rec.geom_mean if rec else None
        semitones = semitone_difference(f_ref, f_rec)
        rows.append(
            {
                "key": pair.key,
                "f0_reference": f_ref,
                "f0_reconstruction": f_rec,
                "semitone_difference": semitones,
                "abs_semitone_difference": abs(semitones) if semitones is not None else None,
                "percent_difference": percent_difference(f_ref, f_rec),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "key",
            "f0_reference",
            "f0_reconstruction",
            "semitone_difference",
            "abs_semitone_difference",
            "percent_difference",
        ],
    )


def summarize(df) -> dict:
    valid = df.dropna(subset=["abs_semitone_difference"])
    if valid.empty:
        return {"n": 0}
    semitones = valid["abs_semitone_difference"]
    return {
        "n": int(len(valid)),
        "mean_percent_difference": float(valid["percent_difference"].mean()),
        "mean_abs_semitone_difference": float(semitones.mean()),
        "percent_over_half_semitone": float((semitones > 0.5).mean() * 100),
        "percent_over_one_semitone": float((semitones > 1.0).mean() * 100),
    }
