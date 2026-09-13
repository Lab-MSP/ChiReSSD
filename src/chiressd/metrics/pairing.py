"""Matching reconstructed files to the originals they came from.

Every metric compares a reconstruction against its original, so every metric
needs this. In the research notebooks it was open-coded per cell as two
independent ``os.listdir`` calls fed to ``zip()``:

    org_list = ["_".join(f.split('_')[1:]) for f in recon_list_child]
    ...
    sims_zs = compute_similarities(org_dir, recon_dir_zs, org_list, recon_list_zs)

``os.listdir`` returns directory order, not sorted order, so pairing two
listings positionally is only correct when both happen to enumerate in the same
sequence. Worse, the originals list above is derived from the *child* listing
and then zipped against a *different* condition's listing.

Pairing here is by key, never by position, and mismatches are reported rather
than silently truncated by ``zip``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

#: Reconstructions are written with a condition prefix before the original
#: filename -- "NC_", "NCC_", "B_" in the STAR runs.
DEFAULT_AUDIO_SUFFIX = ".wav"


class PairingError(RuntimeError):
    pass


@dataclass(frozen=True)
class Pair:
    """One original recording and the reconstruction derived from it."""

    key: str
    reference: Path
    hypothesis: Path


def strip_condition_prefix(filename: str) -> str:
    """Recover the original filename from a prefixed reconstruction.

    Drops everything up to the first underscore, matching the original
    ``"_".join(name.split("_")[1:])``. Names without an underscore are returned
    unchanged rather than becoming empty strings.
    """
    head, sep, tail = filename.partition("_")
    return tail if sep else filename


def pair_by_prefix(
    hypothesis_dir: str | Path,
    reference_dir: str | Path,
    *,
    suffix: str = DEFAULT_AUDIO_SUFFIX,
    strict: bool = True,
) -> list[Pair]:
    """Pair every reconstruction in ``hypothesis_dir`` with its original.

    Results are sorted by key, so downstream lists from different conditions
    line up. With ``strict``, a reconstruction whose original is missing is an
    error; otherwise it is skipped.
    """
    hypothesis_dir, reference_dir = Path(hypothesis_dir), Path(reference_dir)
    if not hypothesis_dir.is_dir():
        raise PairingError(f"Not a directory: {hypothesis_dir}")
    if not reference_dir.is_dir():
        raise PairingError(f"Not a directory: {reference_dir}")

    pairs: list[Pair] = []
    missing: list[str] = []

    for hypothesis in sorted(hypothesis_dir.iterdir()):
        if not hypothesis.name.endswith(suffix):
            continue
        key = strip_condition_prefix(hypothesis.name)
        reference = reference_dir / key
        if not reference.is_file():
            missing.append(hypothesis.name)
            continue
        pairs.append(Pair(key=key, reference=reference, hypothesis=hypothesis))

    if missing and strict:
        shown = ", ".join(missing[:5])
        more = f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""
        raise PairingError(
            f"{len(missing)} reconstruction(s) in {hypothesis_dir} have no original in "
            f"{reference_dir}: {shown}{more}. Pass strict=False to skip them."
        )
    return pairs


def align_conditions(conditions: Mapping[str, Sequence[Pair]]) -> dict[str, list[Pair]]:
    """Restrict several conditions to the keys they all share, in one order.

    This is what makes a paired statistical test meaningful: element *i* of
    every returned list refers to the same utterance.
    """
    if not conditions:
        return {}

    shared: set[str] | None = None
    for pairs in conditions.values():
        keys = {p.key for p in pairs}
        shared = keys if shared is None else (shared & keys)

    order = sorted(shared or ())
    aligned: dict[str, list[Pair]] = {}
    for name, pairs in conditions.items():
        index = {p.key: p for p in pairs}
        aligned[name] = [index[k] for k in order]
    return aligned


def keys_of(pairs: Sequence[Pair]) -> list[str]:
    return [p.key for p in pairs]
