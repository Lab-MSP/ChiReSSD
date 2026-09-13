"""ChiReSSD — speaker-preserving reconstruction of disordered speech.

Synthesizes a target transcript with canonical pronunciation while keeping the
speaker's voice and prosody: pronunciation comes from the text pathway,
identity and prosody from the style pathway.

See https://github.com/Lab-MSP/ChiReSSD.
"""

from __future__ import annotations

from ._refs import GITHUB_URL, HF_REPO, UPSTREAM_COMMIT, UPSTREAM_URL
from .paths import Paths, PathsError, load_paths

__version__ = "0.1.0"

__all__ = [
    "GITHUB_URL",
    "HF_REPO",
    "UPSTREAM_COMMIT",
    "UPSTREAM_URL",
    "Paths",
    "PathsError",
    "load_paths",
    "__version__",
]
