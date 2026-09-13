"""Canonical locations for this project. Kept in one module so a fork needs one edit."""

from __future__ import annotations

GITHUB_REPO = "Lab-MSP/ChiReSSD"
GITHUB_URL = f"https://github.com/{GITHUB_REPO}"

#: Hugging Face repo holding the released checkpoint.
HF_REPO = "Lab-MSP/ChiReSSD"
HF_CHECKPOINT_FILE = "model.pth"
HF_MANIFEST_FILE = "manifest.json"

#: Upstream StyleTTS2, cloned and patched at setup time rather than vendored.
UPSTREAM_URL = "https://github.com/yl4579/StyleTTS2"
UPSTREAM_COMMIT = "5cedc71c333f8d8b8551ca59378bdcc7af4c9529"

PAPER_TITLE = (
    "Generative Reconstruction of Pediatric Disordered Speech for Automated Clinical Evaluation"
)
