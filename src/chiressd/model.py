"""The ChiReSSD model wrapper.

Consolidates the synthesis code that the research notebooks carried as a
~260-line block copied verbatim into five of them.

The method in one paragraph: a reference recording of the child yields two
128-dimensional style vectors (acoustic and prosodic); the target transcript
yields phonemes. A diffusion sampler proposes a style from the text, and that
proposal is interpolated with the reference style — ``alpha`` for timbre,
``beta`` for prosody. Pronunciation therefore comes from the phonemes while
identity and prosody come from the reference, which is what lets the output
correct the mispronunciation while still sounding like the child.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from . import audio as audio_utils
from . import text as text_utils
from .vendor import activate

#: Style vectors are 128-d each, concatenated into one 256-d reference vector:
#: [:128] is acoustic/timbre, [128:] is prosodic.
STYLE_DIM = 128


class ModelError(RuntimeError):
    pass


#: Keys that only a training recipe carries. A config with these was not the one
#: saved alongside a checkpoint.
_TRAINING_ONLY_KEYS = ("loss_params", "optimizer_params", "slmadv_params")


def _warn_if_training_config(cfg: dict, path) -> None:
    """Warn when a training recipe is used to load a trained checkpoint.

    The trainer writes the *estimated* ``sigma_data`` back into the config it
    saves beside each checkpoint, because ``estimate_sigma_data`` computes it
    from the data. A training recipe still carries the placeholder it started
    from -- 0.2 in the released recipe, against roughly 0.27 estimated.

    That value scales the diffusion denoiser's preconditioning (``c_skip``,
    ``c_out``, ``c_in``) at every sampling step, so loading trained weights
    under the placeholder produces different audio, silently and plausibly.
    Hence a warning rather than a note in the docs.
    """
    import warnings

    present = [k for k in _TRAINING_ONLY_KEYS if k in cfg]
    if not present:
        return

    dist = cfg.get("model_params", {}).get("diffusion", {}).get("dist", {})
    detail = ""
    if dist.get("estimate_sigma_data"):
        detail = (
            f" Its sigma_data ({dist.get('sigma_data')}) is the pre-training placeholder, "
            "not the value estimated during training."
        )

    warnings.warn(
        f"{path} looks like a training recipe (it defines {', '.join(present)}), "
        f"not the config saved with a checkpoint.{detail} For inference, pass "
        "config=None so the config published alongside the checkpoint is used.",
        UserWarning,
        stacklevel=3,
    )


def _strip_data_parallel(state: dict) -> dict:
    """Remove the ``module.`` prefix that DataParallel adds to checkpoint keys.

    Training ran under DataParallel, so ten of the thirteen submodules carry the
    prefix and three do not. The released checkpoint is published already
    stripped; this keeps raw training checkpoints loadable too.
    """
    if not any(k.startswith("module.") for k in state):
        return state
    return {k[len("module.") :] if k.startswith("module.") else k: v for k, v in state.items()}


@dataclass
class ChiReSSD:
    """A loaded ChiReSSD model, ready to synthesize."""

    model: Any
    sampler: Any
    model_params: Any
    device: Any
    language: str = text_utils.DEFAULT_LANGUAGE
    checkpoint: Path | None = None

    # -- construction ------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | Path | None = None,
        config: str | Path | None = None,
        *,
        upstream: str | Path | None = None,
        device: str | None = None,
        language: str = text_utils.DEFAULT_LANGUAGE,
        verbose: bool = False,
    ) -> ChiReSSD:
        """Build the model and load weights.

        ``checkpoint`` may be a local path, an ``hf://`` reference, or ``None``
        for the released checkpoint. ``config`` defaults to the training config
        stored alongside the checkpoint.
        """
        import torch
        import yaml

        from .checkpoints import resolve_checkpoint, resolve_config

        if upstream is None:
            from .paths import load_paths

            upstream = load_paths().upstream
        activate(Path(upstream))

        # Imported only after the upstream checkout is on sys.path.
        from models import build_model, load_ASR_models, load_F0_models  # type: ignore
        from utils import recursive_munch  # type: ignore
        from Utils.PLBERT.util import load_plbert  # type: ignore

        checkpoint_path = resolve_checkpoint(checkpoint)
        config_path = resolve_config(config, checkpoint_path)
        cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        _warn_if_training_config(cfg, config_path)

        upstream_path = Path(upstream)

        def _asset(key: str) -> str:
            """Resolve a helper-model path, which may be relative to upstream."""
            value = cfg.get(key)
            if not value:
                raise ModelError(f"Config {config_path} does not set {key!r}.")
            path = Path(value)
            return str(path if path.is_absolute() else upstream_path / path)

        text_aligner = load_ASR_models(_asset("ASR_path"), _asset("ASR_config"))
        pitch_extractor = load_F0_models(_asset("F0_path"))
        plbert = load_plbert(_asset("PLBERT_dir"))

        model_params = recursive_munch(cfg["model_params"])
        model = build_model(model_params, text_aligner, pitch_extractor, plbert)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_device = torch.device(device)

        for key in model:
            model[key].eval()
            model[key].to(torch_device)

        params = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        params = params.get("net", params)

        loaded, skipped = [], []
        for key in model:
            if key not in params:
                skipped.append(key)
                continue
            state = _strip_data_parallel(params[key])
            try:
                model[key].load_state_dict(state)
            except RuntimeError:
                # Mirrors the original loader: tolerate a partial match rather
                # than refusing to load. Reported, never silent.
                model[key].load_state_dict(state, strict=False)
                skipped.append(f"{key} (partial)")
                loaded.append(key)
            else:
                loaded.append(key)

        if verbose or skipped:
            print(f"loaded {len(loaded)}/{len(list(model))} modules from {checkpoint_path.name}")
            if skipped:
                print(f"  not fully loaded: {', '.join(skipped)}")

        for key in model:
            model[key].eval()

        from Modules.diffusion.sampler import (  # type: ignore
            ADPM2Sampler,
            DiffusionSampler,
            KarrasSchedule,
        )

        sampler = DiffusionSampler(
            model.diffusion.diffusion,
            sampler=ADPM2Sampler(),
            sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
            clamp=False,
        )

        return cls(
            model=model,
            sampler=sampler,
            model_params=model_params,
            device=torch_device,
            language=language,
            checkpoint=Path(checkpoint_path),
        )

    # -- style -------------------------------------------------------------

    def compute_style(
        self,
        source: str | Path | np.ndarray,
        sr: int | None = None,
        *,
        trim_top_db: int = audio_utils.TRIM_TOP_DB,
    ):
        """Extract the 256-d reference style vector from a recording.

        Accepts a path or an in-memory waveform; the original code had two
        near-identical functions for those two cases.
        """
        import torch

        if isinstance(source, (str, Path)):
            wave = audio_utils.load_audio(source, sr=audio_utils.SAMPLE_RATE)
            sr = audio_utils.SAMPLE_RATE
        else:
            wave = np.asarray(source)
            if sr is None:
                raise ModelError("sr is required when passing a waveform array.")

        wave = audio_utils.trim_silence(wave, top_db=trim_top_db)
        wave = audio_utils.resample(wave, sr, audio_utils.SAMPLE_RATE)

        duration = len(wave) / audio_utils.SAMPLE_RATE
        if len(wave) == 0:
            raise ModelError(f"{source} is silent after trimming, so there is no style to extract.")

        mel = audio_utils.to_mel(wave).to(self.device)
        try:
            with torch.no_grad():
                ref_s = self.model.style_encoder(mel.unsqueeze(1))
                ref_p = self.model.predictor_encoder(mel.unsqueeze(1))
        except RuntimeError as exc:
            # The style encoders downsample repeatedly, so a short reference
            # runs out of frames partway through and torch reports it as a
            # kernel-larger-than-input convolution error, naming neither the
            # file nor the real cause.
            if "Kernel size" not in str(exc):
                raise
            raise ModelError(
                f"Reference is too short to extract a style from: {duration:.2f}s of audio "
                f"after silence trimming ({mel.shape[-1]} mel frames)"
                + (f", from {source}" if isinstance(source, (str, Path)) else "")
                + ".\nThe style encoders downsample several times, so very short clips run "
                "out of frames. Around four seconds is the documented minimum; concatenating "
                "several short prompts from the same speaker is what the original pipeline "
                "did."
            ) from exc
        return torch.cat([ref_s, ref_p], dim=1)

    # -- synthesis ---------------------------------------------------------

    def _encode_text(self, text: str):
        import torch

        phonemes = text_utils.phonemize(text, self.language)
        tokens = text_utils.tokenize(phonemes)
        return torch.LongTensor(tokens).to(self.device).unsqueeze(0)

    def _style_noise(self, noise: str):
        import torch

        if noise == "zeros":
            return torch.zeros((1, 2 * STYLE_DIM)).unsqueeze(1).to(self.device)
        if noise == "random":
            return torch.randn((1, 2 * STYLE_DIM)).unsqueeze(1).to(self.device)
        raise ModelError(f"noise must be 'zeros' or 'random', got {noise!r}")

    def synthesize(
        self,
        text: str,
        ref_s,
        *,
        alpha: float,
        beta: float,
        diffusion_steps: int = 10,
        embedding_scale: float = 1.0,
        noise: Literal["zeros", "random"] = "zeros",
        seed: int | None = None,
    ) -> np.ndarray:
        """Synthesize ``text`` in the voice described by ``ref_s``.

        ``alpha`` and ``beta`` are required, deliberately. They interpolate the
        diffusion-sampled style with the reference style — 1.0 fully sampled,
        0.0 fully reference-driven — and a low ``alpha`` pulls in the reference
        acoustics, which is exactly where the disordered articulation lives.
        Inheriting a default here would silently reproduce the mispronunciation
        the model exists to correct, so the caller must choose.

        Synthesis is **stochastic**. ``noise`` only sets the initial latent;
        the ADPM2 sampler is ancestral and adds fresh noise at every step, so
        repeated calls differ — in duration as well as in waveform — unless
        ``seed`` is given. Pass ``seed`` for reproducible output.
        """
        import torch

        if seed is not None:
            torch.manual_seed(seed)

        tokens = self._encode_text(text)

        with torch.no_grad():
            input_lengths = torch.LongTensor([tokens.shape[-1]]).to(self.device)
            text_mask = _length_to_mask(input_lengths).to(self.device)

            t_en = self.model.text_encoder(tokens, input_lengths, text_mask)
            bert_dur = self.model.bert(tokens, attention_mask=(~text_mask).int())
            d_en = self.model.bert_encoder(bert_dur).transpose(-1, -2)

            s_pred = self.sampler(
                noise=self._style_noise(noise),
                embedding=bert_dur,
                embedding_scale=embedding_scale,
                features=ref_s,
                num_steps=diffusion_steps,
            ).squeeze(1)

            ref = s_pred[:, :STYLE_DIM]  # acoustic / timbre
            s = s_pred[:, STYLE_DIM:]  # prosodic

            ref = alpha * ref + (1 - alpha) * ref_s[:, :STYLE_DIM]
            s = beta * s + (1 - beta) * ref_s[:, STYLE_DIM:]

            return self._decode(t_en, d_en, s, ref, input_lengths, text_mask)

    def synthesize_long(
        self,
        text: str,
        ref_s,
        *,
        alpha: float,
        beta: float,
        s_prev=None,
        t: float = 0.7,
        diffusion_steps: int = 10,
        embedding_scale: float = 1.0,
        noise: Literal["zeros", "random"] = "random",
        seed: int | None = None,
    ) -> tuple[np.ndarray, Any]:
        """Synthesize one segment of a longer passage, carrying style forward.

        Returns the waveform and the style vector to pass back as ``s_prev``,
        which keeps voice and pacing continuous across sentence boundaries.
        """
        import torch

        if seed is not None:
            torch.manual_seed(seed)

        tokens = self._encode_text(text)

        with torch.no_grad():
            input_lengths = torch.LongTensor([tokens.shape[-1]]).to(self.device)
            text_mask = _length_to_mask(input_lengths).to(self.device)

            t_en = self.model.text_encoder(tokens, input_lengths, text_mask)
            bert_dur = self.model.bert(tokens, attention_mask=(~text_mask).int())
            d_en = self.model.bert_encoder(bert_dur).transpose(-1, -2)

            s_pred = self.sampler(
                noise=self._style_noise(noise),
                embedding=bert_dur,
                embedding_scale=embedding_scale,
                features=ref_s,
                num_steps=diffusion_steps,
            ).squeeze(1)

            if s_prev is not None:
                s_pred = t * s_prev + (1 - t) * s_pred

            ref = s_pred[:, :STYLE_DIM]
            s = s_pred[:, STYLE_DIM:]

            ref = alpha * ref + (1 - alpha) * ref_s[:, :STYLE_DIM]
            s = beta * s + (1 - beta) * ref_s[:, STYLE_DIM:]

            carried = torch.cat([ref, s], dim=-1)
            wav = self._decode(t_en, d_en, s, ref, input_lengths, text_mask)
            return wav, carried

    # -- shared decoding path ---------------------------------------------

    def _decode(self, t_en, d_en, s, ref, input_lengths, text_mask) -> np.ndarray:
        """Duration prediction, alignment, prosody, vocoding."""
        import torch

        d = self.model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = self.model.predictor.lstm(d)
        duration = self.model.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1)
        pred_dur = torch.round(duration.squeeze()).clamp(min=1)

        # Expand per-phoneme durations into a frame-level alignment matrix.
        n_tokens = int(input_lengths.item())
        n_frames = int(pred_dur.sum().item())
        pred_aln_trg = torch.zeros(n_tokens, n_frames)
        c_frame = 0
        for i in range(n_tokens):
            step = int(pred_dur[i].item())
            pred_aln_trg[i, c_frame : c_frame + step] = 1
            c_frame += step
        pred_aln_trg = pred_aln_trg.unsqueeze(0).to(self.device)

        en = d.transpose(-1, -2) @ pred_aln_trg
        en = _shift_for_hifigan(en, self.model_params)

        F0_pred, N_pred = self.model.predictor.F0Ntrain(en, s)

        asr = t_en @ pred_aln_trg
        asr = _shift_for_hifigan(asr, self.model_params)

        out = self.model.decoder(asr, F0_pred, N_pred, ref.squeeze().unsqueeze(0))
        return out.squeeze().cpu().numpy()


def _length_to_mask(lengths):
    import torch

    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    return torch.gt(mask + 1, lengths.unsqueeze(1))


def _shift_for_hifigan(x, model_params):
    """Delay the sequence by one frame, as the HiFi-GAN decoder expects.

    The iSTFTNet decoder does not want this, so it is conditional.
    """
    import torch

    if model_params.decoder.type != "hifigan":
        return x
    shifted = torch.zeros_like(x)
    shifted[:, :, 0] = x[:, :, 0]
    shifted[:, :, 1:] = x[:, :, 0:-1]
    return shifted
