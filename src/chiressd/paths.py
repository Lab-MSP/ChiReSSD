"""Resolution of every filesystem location this project needs.

The research code this package was extracted from hardcoded roughly two hundred
absolute paths across its configs and notebooks. They all collapse into one YAML
file, resolved here, so that porting the pipeline to another machine is a
single edit rather than a search-and-replace across notebooks.

Values may reference one another with ``${dotted.key}``; ``~`` is expanded and
every result is made absolute.

Resolution order, first hit wins:

1. an explicit path passed to :func:`load_paths`
2. ``$CHIRESSD_PATHS``
3. ``./paths.yaml``
4. the packaged ``configs/paths.example.yaml``

Individual roots can then be overridden by environment variable
(``CHIRESSD_WORK_ROOT``, ``CHIRESSD_DATA_ROOT``, ``CHIRESSD_UPSTREAM``,
``CHIRESSD_CACHE_ROOT``), which is what makes the pipeline usable inside a
cluster job that cannot edit files.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_REF = re.compile(r"\$\{([A-Za-z0-9_.]+)\}")
_MAX_PASSES = 10

#: Root keys overridable by environment variable.
_ENV_OVERRIDES = {
    "work_root": "CHIRESSD_WORK_ROOT",
    "data_root": "CHIRESSD_DATA_ROOT",
    "upstream": "CHIRESSD_UPSTREAM",
    "cache_root": "CHIRESSD_CACHE_ROOT",
}

PACKAGED_EXAMPLE = Path(__file__).resolve().parents[2] / "configs" / "paths.example.yaml"


class PathsError(RuntimeError):
    """Raised when the paths file is missing, cyclic, or references an unknown key."""


def _flatten(node: Any, prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    if isinstance(node, dict):
        for key, value in node.items():
            flat.update(_flatten(value, f"{prefix}.{key}" if prefix else str(key)))
    else:
        flat[prefix] = node
    return flat


def _unflatten(flat: dict[str, Path]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for dotted, value in flat.items():
        *parents, leaf = dotted.split(".")
        node = out
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = value
    return out


def _interpolate(flat: dict[str, Any]) -> dict[str, str]:
    """Expand ``${dotted.key}`` references until the mapping stops changing."""
    values = {k: ("" if v is None else str(v)) for k, v in flat.items()}

    for _ in range(_MAX_PASSES):
        changed = False
        for key, value in values.items():
            if "${" not in value:
                continue

            # `key` is bound as a default so the closure cannot pick up a later
            # iteration's value; it is only used to name the offender in errors.
            def replace(match: re.Match[str], key: str = key) -> str:
                ref = match.group(1)
                if ref not in values:
                    raise PathsError(
                        f"{key!r} references unknown key ${{{ref}}}. "
                        f"Known keys: {', '.join(sorted(values))}"
                    )
                return values[ref]

            new = _REF.sub(replace, value)
            if new != value:
                values[key] = new
                changed = True
        if not changed:
            break
    else:
        unresolved = sorted(k for k, v in values.items() if "${" in v)
        raise PathsError(f"Cyclic or unresolvable ${{...}} reference in: {unresolved}")

    return values


@dataclass(frozen=True)
class Paths:
    """Resolved absolute locations, addressable by dotted key.

    >>> paths["datasets.train.audio"]     # doctest: +SKIP
    PosixPath('/data/audio')
    """

    tree: dict[str, Any]
    flat: dict[str, Path]
    source: Path

    def __getitem__(self, dotted: str) -> Path:
        try:
            return self.flat[dotted]
        except KeyError:
            raise PathsError(
                f"No path {dotted!r}. Known keys: {', '.join(sorted(self.flat))}"
            ) from None

    def get(self, dotted: str, default: Path | None = None) -> Path | None:
        return self.flat.get(dotted, default)

    def __contains__(self, dotted: object) -> bool:
        return dotted in self.flat

    # Convenience accessors for the four roots, which nearly every caller needs.
    @property
    def work_root(self) -> Path:
        return self["work_root"]

    @property
    def data_root(self) -> Path:
        return self["data_root"]

    @property
    def upstream(self) -> Path:
        return self["upstream"]

    @property
    def cache_root(self) -> Path:
        return self["cache_root"]


def _locate(source: str | Path | None) -> Path:
    if source is not None:
        path = Path(source).expanduser()
        if not path.is_file():
            raise PathsError(f"Paths file not found: {path}")
        return path

    from_env = os.environ.get("CHIRESSD_PATHS")
    if from_env:
        path = Path(from_env).expanduser()
        if not path.is_file():
            raise PathsError(f"$CHIRESSD_PATHS points at a missing file: {path}")
        return path

    local = Path.cwd() / "paths.yaml"
    if local.is_file():
        return local

    if PACKAGED_EXAMPLE.is_file():
        return PACKAGED_EXAMPLE

    raise PathsError(
        "No paths file found. Copy configs/paths.example.yaml to paths.yaml and edit it, "
        "or set $CHIRESSD_PATHS."
    )


def load_paths(source: str | Path | None = None) -> Paths:
    """Load, interpolate, and absolutise the paths configuration."""
    path = _locate(source)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise PathsError(f"{path} must contain a mapping at the top level.")

    flat_raw = _flatten(raw)
    for key, env_var in _ENV_OVERRIDES.items():
        override = os.environ.get(env_var)
        if override:
            flat_raw[key] = override

    resolved = _interpolate(flat_raw)
    flat = {k: Path(v).expanduser().resolve() for k, v in resolved.items() if v}

    return Paths(tree=_unflatten(flat), flat=flat, source=path)
