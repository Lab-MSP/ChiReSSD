"""Speaker similarity between a reconstruction and the original recording.

Cosine similarity of Resemblyzer utterance embeddings. Resemblyzer is trained on
typical adult speech, so its embeddings are less well matched to child and
disordered voices; the F0 measures in :mod:`chiressd.metrics.pitch` involve no
learned representation and so carry no such bias.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .pairing import Pair


def load_encoder(device: str | None = None):
    """Load the Resemblyzer voice encoder.

    Cheap to keep around and expensive to rebuild, so callers that loop over
    conditions should load once and pass it in.
    """
    try:
        from resemblyzer import VoiceEncoder
    except ImportError as exc:  # pragma: no cover
        raise ImportError("resemblyzer is required: pip install 'chiressd[eval-speaker]'") from exc
    return VoiceEncoder(device) if device else VoiceEncoder()


def embed(path: str | Path, encoder) -> np.ndarray:
    from resemblyzer import preprocess_wav

    return encoder.embed_utterance(preprocess_wav(str(path)))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator == 0:
        return float("nan")
    return float(np.dot(a, b) / denominator)


def similarities(pairs: Sequence[Pair], encoder=None, *, device: str | None = None):
    """Per-pair speaker similarity.

    Returns a DataFrame with ``key`` and ``similarity``, in the order given --
    which, for pairs built by :func:`~chiressd.metrics.pairing.pair_by_prefix`,
    is sorted by key and therefore aligned across conditions.
    """
    import pandas as pd

    encoder = encoder or load_encoder(device)
    rows = []
    # Originals repeat across conditions; embedding one twice is pure waste.
    cache: dict[Path, np.ndarray] = {}

    for pair in pairs:
        if pair.reference not in cache:
            cache[pair.reference] = embed(pair.reference, encoder)
        rows.append(
            {
                "key": pair.key,
                "similarity": cosine(cache[pair.reference], embed(pair.hypothesis, encoder)),
            }
        )
    return pd.DataFrame(rows, columns=["key", "similarity"])


def summarize(df) -> dict:
    values = df["similarity"].dropna()
    return {
        "n": int(len(values)),
        "mean": float(values.mean()) if len(values) else float("nan"),
        "std": float(values.std()) if len(values) else float("nan"),
        "median": float(values.median()) if len(values) else float("nan"),
    }
