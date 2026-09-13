"""Locating model checkpoints, locally or on the Hugging Face Hub.

The released checkpoint is ~766 MB and is distributed through the Hub rather
than git. A reference can be:

* a local path — ``/models/epoch_2nd_00003.pth``
* a Hub reference — ``hf://Lab-MSP/ChiReSSD``, optionally ``@revision``
  and ``:filename``
* ``None`` — the released checkpoint
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from ._refs import HF_CHECKPOINT_FILE, HF_MANIFEST_FILE, HF_REPO

_HF_REF = re.compile(r"^hf://(?P<repo>[^@:]+)(?:@(?P<revision>[^:]+))?(?::(?P<filename>.+))?$")

#: Training configs are written next to their checkpoints by the trainer.
_CONFIG_CANDIDATES = ("config.yml", "config.yaml")


class CheckpointError(RuntimeError):
    pass


def sha256(path: str | Path, *, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(repo: str, filename: str, *, revision: str | None, cache_dir: Path | None) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover
        raise CheckpointError(
            "huggingface-hub is required to fetch checkpoints: pip install huggingface-hub"
        ) from exc

    return Path(
        hf_hub_download(
            repo_id=repo,
            filename=filename,
            revision=revision,
            cache_dir=str(cache_dir) if cache_dir else None,
        )
    )


def _verify_against_manifest(
    local: Path, repo: str, *, revision: str | None, cache_dir: Path | None
) -> None:
    """Check the download against the published manifest, if there is one.

    A truncated or corrupted checkpoint otherwise fails much later, as strange
    audio rather than as an error.
    """
    try:
        manifest_path = _download(repo, HF_MANIFEST_FILE, revision=revision, cache_dir=cache_dir)
    except Exception:
        return  # No manifest published; nothing to check against.

    try:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return

    expected = manifest.get("sha256")
    if expected and sha256(local) != expected:
        raise CheckpointError(
            f"Checksum mismatch for {local}: manifest says {expected}. "
            "Delete the cached file and re-download."
        )


def resolve_checkpoint(
    ref: str | Path | None = None,
    *,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    verify: bool = True,
) -> Path:
    """Return a local path to the requested checkpoint, downloading if needed."""
    if cache_dir is None:
        try:
            from .paths import load_paths

            cache_dir = load_paths().cache_root
        except Exception:
            cache_dir = None
    cache_dir = Path(cache_dir) if cache_dir else None

    if ref is None:
        ref = f"hf://{HF_REPO}"

    text = str(ref)
    match = _HF_REF.match(text)
    if match is None:
        path = Path(ref).expanduser()
        if not path.is_file():
            raise CheckpointError(
                f"No checkpoint at {path}. Pass a local path, an hf:// reference, "
                "or None for the released checkpoint."
            )
        return path.resolve()

    repo = match.group("repo")
    filename = match.group("filename") or HF_CHECKPOINT_FILE
    revision = match.group("revision") or revision

    try:
        local = _download(repo, filename, revision=revision, cache_dir=cache_dir)
    except Exception as exc:
        raise CheckpointError(
            f"Could not fetch {filename} from {repo}: {exc}\n"
            "The model repository may be private; run `huggingface-cli login`."
        ) from exc

    if verify:
        _verify_against_manifest(local, repo, revision=revision, cache_dir=cache_dir)
    return local


def resolve_config(
    config: str | Path | None,
    checkpoint: str | Path,
    *,
    cache_dir: str | Path | None = None,
) -> Path:
    """Find the training config that describes ``checkpoint``.

    Explicit wins; otherwise look beside the checkpoint, which is where the
    trainer writes it; otherwise fetch it from the Hub alongside the weights.
    """
    if config is not None:
        path = Path(config).expanduser()
        if not path.is_file():
            from .config import TRAIN_CONFIG_DIR

            candidate = TRAIN_CONFIG_DIR / (
                str(config) if str(config).endswith((".yml", ".yaml")) else f"{config}.yml"
            )
            if not candidate.is_file():
                raise CheckpointError(f"No config at {path}.")
            path = candidate
        return path.resolve()

    directory = Path(checkpoint).parent
    for name in _CONFIG_CANDIDATES:
        if (directory / name).is_file():
            return (directory / name).resolve()
    # The trainer copies its recipe in under its original name.
    for candidate in sorted(directory.glob("config*.y*ml")):
        return candidate.resolve()

    try:
        return resolve_checkpoint(f"hf://{HF_REPO}:config.yml", cache_dir=cache_dir, verify=False)
    except CheckpointError as exc:
        raise CheckpointError(
            f"No config found next to {checkpoint} and none could be fetched. "
            "Pass config= explicitly."
        ) from exc
