"""Loading of training configs and inference presets.

Training configs ship with ``${dotted.key}`` references into ``paths.yaml``.
Upstream's ``train_finetune.py`` reads plain YAML and cannot interpolate, so a
config is *rendered* to a concrete file before training starts. The rendered
file is written beside the run and is the provenance record for it: it names
every path and hyperparameter actually used.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import yaml

from .paths import Paths, PathsError

_REF = re.compile(r"\$\{([A-Za-z0-9_.]+)\}")

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs"
TRAIN_CONFIG_DIR = CONFIG_ROOT / "train"
PRESET_DIR = CONFIG_ROOT / "inference"

#: Relative to the upstream checkout, not to the user's data.
_UPSTREAM_RELATIVE_KEYS = ("F0_path", "ASR_config", "ASR_path", "PLBERT_dir")


class ConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class InferencePreset:
    """A named synthesis operating point.

    ``alpha`` weights the acoustic/timbre style and ``beta`` the prosodic style;
    1.0 is fully diffusion-sampled, 0.0 fully reference-driven.
    """

    name: str
    alpha: float
    beta: float
    diffusion_steps: int = 10
    embedding_scale: float = 1.0
    noise: Literal["zeros", "random"] = "zeros"
    source: Path | None = field(default=None, compare=False)

    def as_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for :meth:`chiressd.model.ChiReSSD.synthesize`."""
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "diffusion_steps": self.diffusion_steps,
            "embedding_scale": self.embedding_scale,
            "noise": self.noise,
        }

    def __str__(self) -> str:
        return (
            f"{self.name}(alpha={self.alpha}, beta={self.beta}, "
            f"steps={self.diffusion_steps}, scale={self.embedding_scale}, "
            f"noise={self.noise})"
        )


def available_presets() -> list[str]:
    if not PRESET_DIR.is_dir():
        return []
    return sorted(p.stem for p in PRESET_DIR.glob("*.yaml"))


def load_preset(preset: str | Path) -> InferencePreset:
    """Load an inference preset by name or by path."""
    path = Path(preset)
    if not path.is_file():
        path = PRESET_DIR / f"{preset}.yaml"
    if not path.is_file():
        raise ConfigError(
            f"Unknown inference preset {preset!r}. Available: {', '.join(available_presets())}"
        )

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    missing = {"alpha", "beta"} - data.keys()
    if missing:
        # alpha and beta control whether disordered articulation is reproduced,
        # so a preset that omits them is never silently defaulted.
        raise ConfigError(f"Preset {path} must set {', '.join(sorted(missing))}.")

    noise = data.get("noise", "zeros")
    if noise not in ("zeros", "random"):
        raise ConfigError(f"Preset {path}: noise must be 'zeros' or 'random', got {noise!r}.")

    return InferencePreset(
        name=data.get("name", path.stem),
        alpha=float(data["alpha"]),
        beta=float(data["beta"]),
        diffusion_steps=int(data.get("diffusion_steps", 10)),
        embedding_scale=float(data.get("embedding_scale", 1.0)),
        noise=noise,
        source=path,
    )


def available_train_configs() -> list[str]:
    if not TRAIN_CONFIG_DIR.is_dir():
        return []
    return sorted(p.stem for p in TRAIN_CONFIG_DIR.glob("*.yml"))


def _resolve_refs(node: Any, paths: Paths, *, where: str = "") -> Any:
    if isinstance(node, dict):
        return {
            k: _resolve_refs(v, paths, where=f"{where}.{k}" if where else k)
            for k, v in node.items()
        }
    if isinstance(node, list):
        return [_resolve_refs(v, paths, where=where) for v in node]
    if isinstance(node, str) and "${" in node:

        def replace(match: re.Match[str]) -> str:
            key = match.group(1)
            try:
                return str(paths[key])
            except PathsError as exc:
                raise ConfigError(f"{where}: {exc}") from None

        return _REF.sub(replace, node)
    return node


def load_train_config(config: str | Path, paths: Paths, *, upstream: Path | None = None) -> dict:
    """Load a training config and resolve its ``${...}`` references.

    ``upstream`` makes the frozen helper-model paths absolute. Upstream's
    trainer resolves them relative to its own directory; absolutising them here
    is what lets training run without chdir-ing into the checkout.
    """
    path = Path(config)
    if not path.is_file():
        path = TRAIN_CONFIG_DIR / (
            config if str(config).endswith((".yml", ".yaml")) else f"{config}.yml"
        )
    if not path.is_file():
        raise ConfigError(
            f"Unknown training config {config!r}. Available: {', '.join(available_train_configs())}"
        )

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    resolved = _resolve_refs(raw, paths)

    if upstream is not None:
        upstream = Path(upstream)
        for key in _UPSTREAM_RELATIVE_KEYS:
            value = resolved.get(key)
            if isinstance(value, str) and not Path(value).is_absolute():
                resolved[key] = str(upstream / value)
        ood = resolved.get("data_params", {}).get("OOD_data")
        if isinstance(ood, str) and not Path(ood).is_absolute():
            resolved["data_params"]["OOD_data"] = str(upstream / ood)

    resolved["_chiressd"] = {
        "source_config": str(path),
        "paths_file": str(paths.source),
        "rendered_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return resolved


def unresolved_refs(config: dict) -> list[str]:
    """Dotted keys whose values still contain a ``${...}`` reference."""
    found: list[str] = []

    def walk(node: Any, where: str = "") -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{where}.{k}" if where else str(k))
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{where}[{i}]")
        elif isinstance(node, str) and "${" in node:
            found.append(where)

    walk(config)
    return found


def render_train_config(config: dict, destination: Path) -> Path:
    """Write a fully resolved config, refusing to emit one with dangling refs."""
    dangling = unresolved_refs(config)
    if dangling:
        raise ConfigError(
            "Refusing to write a config with unresolved ${...} references: " + ", ".join(dangling)
        )
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(config, sort_keys=False, default_flow_style=False), encoding="utf-8"
    )
    return destination
