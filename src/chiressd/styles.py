"""Caching of reference style vectors.

Style extraction runs two encoders over a mel-spectrogram, so recomputing it for
every utterance of the same speaker is pure waste: in the original notebooks the
same eight-line load-or-compute-and-save block was pasted into eight cells.

A style vector is a 256-d float array derived from a recording. It is not audio
and cannot be inverted to audio, but it is speaker-specific, so the cache
belongs in working storage rather than anywhere that gets published.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def style_cache_path(cache_dir: str | Path, key: str) -> Path:
    return Path(cache_dir) / f"{key}.npy"


def get_or_compute_style(
    model,
    reference: str | Path,
    cache_dir: str | Path | None = None,
    *,
    key: str | None = None,
    refresh: bool = False,
):
    """Return the style vector for ``reference``, caching it on disk.

    ``key`` defaults to the reference filename without its extension.
    """
    import torch

    reference = Path(reference)
    if cache_dir is None:
        return model.compute_style(reference)

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = style_cache_path(cache_dir, key or reference.stem)

    if path.is_file() and not refresh:
        return torch.from_numpy(np.load(path, allow_pickle=False)).to(model.device)

    style = model.compute_style(reference)
    np.save(path, style.cpu().numpy(), allow_pickle=False)
    return style


def load_style(path: str | Path, device=None):
    """Load a cached style vector as a tensor."""
    import torch

    array = np.load(Path(path), allow_pickle=False)
    tensor = torch.from_numpy(array)
    return tensor.to(device) if device is not None else tensor
