"""Comparing a reconstruction against the recording it came from.

* :mod:`speaker` — is it still the same voice? Cosine similarity of Resemblyzer
  embeddings.
* :mod:`pitch` — is the pitch still the speaker's? F0 difference in percent and
  semitones, using no learned representation.

Both start from :mod:`pairing`, which matches files by key rather than by
directory-listing position.
"""

from __future__ import annotations

from .pairing import Pair, PairingError, align_conditions, keys_of, pair_by_prefix
from .stats import TTestResult, paired_ttest

__all__ = [
    "Pair",
    "PairingError",
    "TTestResult",
    "align_conditions",
    "keys_of",
    "pair_by_prefix",
    "paired_ttest",
]
