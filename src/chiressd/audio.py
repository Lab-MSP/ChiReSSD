"""Audio loading and mel-spectrogram preprocessing.

The constants here are not tunable knobs: they must match what the model was
trained with, so they are defined once and shared by every caller rather than
being retyped per script.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

#: Model sample rate. StyleTTS2 operates at 24 kHz throughout.
SAMPLE_RATE = 24_000

#: Mel front end, matching the training configuration.
N_MELS = 80
N_FFT = 2048
WIN_LENGTH = 1200
HOP_LENGTH = 300

#: Log-mel normalisation constants used during training.
MEL_MEAN = -4.0
MEL_STD = 4.0

#: Leading/trailing silence below this many dB is trimmed from style references.
TRIM_TOP_DB = 30


@lru_cache(maxsize=1)
def _mel_transform():
    import torchaudio

    return torchaudio.transforms.MelSpectrogram(
        n_mels=N_MELS, n_fft=N_FFT, win_length=WIN_LENGTH, hop_length=HOP_LENGTH
    )


def load_audio(path: str | Path, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load a waveform as mono float at ``sr``."""
    import librosa

    wave, _ = librosa.load(str(path), sr=sr)
    return wave


def trim_silence(wave: np.ndarray, top_db: int = TRIM_TOP_DB) -> np.ndarray:
    import librosa

    trimmed, _ = librosa.effects.trim(wave, top_db=top_db)
    return trimmed


def resample(wave: np.ndarray, orig_sr: int, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    if orig_sr == target_sr:
        return wave
    import librosa

    return librosa.resample(wave, orig_sr=orig_sr, target_sr=target_sr)


def to_mel(wave: np.ndarray):
    """Waveform -> normalised log-mel tensor, as the style encoders expect."""
    import torch

    wave_tensor = torch.from_numpy(wave).float()
    mel = _mel_transform()(wave_tensor)
    return (torch.log(1e-5 + mel.unsqueeze(0)) - MEL_MEAN) / MEL_STD


def write_audio(path: str | Path, wave: np.ndarray, sr: int = SAMPLE_RATE) -> Path:
    import soundfile as sf

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), wave, sr)
    return path


def concatenate(
    waves: list[np.ndarray], silence_s: float = 0.0, sr: int = SAMPLE_RATE
) -> np.ndarray:
    """Join waveforms, optionally padding a gap between them.

    Used to build a longer style reference from several short clinical prompts:
    a four-second reference gives the style encoders far more to work with than
    a single one-second word.
    """
    if not waves:
        raise ValueError("nothing to concatenate")
    if silence_s <= 0:
        return np.concatenate(waves)
    gap = np.zeros(int(silence_s * sr), dtype=waves[0].dtype)
    joined: list[np.ndarray] = []
    for i, wave in enumerate(waves):
        if i:
            joined.append(gap)
        joined.append(wave)
    return np.concatenate(joined)
