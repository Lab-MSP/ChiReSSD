"""Bootstrap the upstream StyleTTS2 checkout that ChiReSSD builds on.

ChiReSSD does not vendor StyleTTS2. It clones it at a pinned commit and applies
the patches in ``_upstream_patches``.

"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from ._refs import UPSTREAM_COMMIT, UPSTREAM_URL

PATCH_DIR = Path(__file__).resolve().parent / "_upstream_patches"

#: Frozen helper weights that ship inside the upstream repository. Without them
#: the model cannot be built, and a partial clone is the usual cause.
REQUIRED_ASSETS = (
    Path("Utils/ASR/epoch_00080.pth"),
    Path("Utils/ASR/config.yml"),
    Path("Utils/JDC/bst.t7"),
    Path("Utils/PLBERT/step_1000000.t7"),
    Path("Utils/PLBERT/config.yml"),
)

#: Modules imported from the upstream tree once it is on sys.path.
REQUIRED_MODULES = ("models.py", "meldataset.py", "utils.py", "text_utils.py", "losses.py")


class UpstreamError(RuntimeError):
    """Raised when the upstream checkout is missing, at the wrong commit, or unpatched."""


@dataclass(frozen=True)
class UpstreamStatus:
    path: Path
    exists: bool
    commit: str | None
    patched: bool
    missing_assets: tuple[Path, ...]

    @property
    def ok(self) -> bool:
        return (
            self.exists
            and self.commit == UPSTREAM_COMMIT
            and self.patched
            and not self.missing_assets
        )

    def explain(self) -> str:
        if not self.exists:
            return f"No StyleTTS2 checkout at {self.path}."
        if self.commit != UPSTREAM_COMMIT:
            return (
                f"StyleTTS2 at {self.path} is at commit {self.commit}, expected {UPSTREAM_COMMIT}."
            )
        if not self.patched:
            return f"StyleTTS2 at {self.path} is missing the ChiReSSD patches."
        if self.missing_assets:
            names = ", ".join(str(p) for p in self.missing_assets)
            return f"StyleTTS2 at {self.path} is missing pretrained assets: {names}."
        return f"StyleTTS2 at {self.path} is ready."


def patch_files() -> tuple[Path, ...]:
    return tuple(sorted(PATCH_DIR.glob("*.patch")))


def _git(*args: str, cwd: Path) -> str:
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise UpstreamError(f"git {' '.join(args)} failed in {cwd}:\n{result.stderr.strip()}")
    return result.stdout.strip()


def _is_patched(dest: Path) -> bool:
    """True when the patches are already present."""
    for patch in patch_files():
        result = subprocess.run(
            ["git", "apply", "--reverse", "--check", "-p1", str(patch)],
            cwd=dest,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return False
    return True


def verify_upstream(dest: Path) -> UpstreamStatus:
    """Report whether ``dest`` is a correctly pinned, patched, complete checkout."""
    dest = Path(dest)
    if not (dest / ".git").exists():
        return UpstreamStatus(dest, False, None, False, ())

    try:
        commit = _git("rev-parse", "HEAD", cwd=dest)
    except UpstreamError:
        commit = None

    missing = tuple(asset for asset in REQUIRED_ASSETS if not (dest / asset).is_file())
    return UpstreamStatus(dest, True, commit, _is_patched(dest), missing)


def ensure_upstream(dest: Path, *, offline: bool = False, force: bool = False) -> UpstreamStatus:
    """Clone StyleTTS2 at the pinned commit and apply the ChiReSSD patches."""
    dest = Path(dest).expanduser()
    status = verify_upstream(dest)

    if status.ok and not force:
        return status

    if not status.exists:
        if offline:
            raise UpstreamError(
                f"No checkout at {dest} and --offline was requested. "
                f"Clone {UPSTREAM_URL} there first."
            )
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Full clone, not --depth 1: we check out a specific commit, which a
        # shallow clone of the default branch may not contain.
        subprocess.run(["git", "clone", UPSTREAM_URL, str(dest)], check=True)
        status = verify_upstream(dest)

    if status.commit != UPSTREAM_COMMIT:
        _git("checkout", "--quiet", UPSTREAM_COMMIT, cwd=dest)

    if not _is_patched(dest):
        for patch in patch_files():
            check = subprocess.run(
                ["git", "apply", "--check", "-p1", str(patch)],
                cwd=dest,
                capture_output=True,
                text=True,
                check=False,
            )
            if check.returncode != 0:
                raise UpstreamError(
                    f"Patch {patch.name} does not apply to {dest}:\n{check.stderr.strip()}\n"
                    "The checkout may be modified; delete it and re-run chiressd-setup."
                )
            subprocess.run(["git", "apply", "-p1", str(patch)], cwd=dest, check=True)

    status = verify_upstream(dest)
    if not status.exists or status.commit != UPSTREAM_COMMIT or not status.patched:
        raise UpstreamError(status.explain())
    return status


def require_upstream(dest: Path) -> Path:
    """Return ``dest`` if it is a usable checkout, else raise with the fix."""
    status = verify_upstream(dest)
    if not status.ok:
        raise UpstreamError(f"{status.explain()}\nRun: chiressd-setup")
    return status.path


def activate(dest: Path) -> Path:
    """Put the upstream checkout on ``sys.path`` so its modules are importable.

    This replaces the ``%cd /path/to/StyleTTS2`` that the original notebooks
    relied on. Nothing here changes the process working directory, so the
    package stays importable from anywhere and relative user paths keep
    meaning what the caller expects.
    """
    import sys

    path = require_upstream(Path(dest))
    entry = str(path)
    if entry in sys.path:
        sys.path.remove(entry)
    sys.path.insert(0, entry)
    return path
