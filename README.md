# ChiReSSD

Speaker-preserving reconstruction of disordered speech, built on
[StyleTTS2](https://github.com/yl4579/StyleTTS2).

[Demo](https://lab-msp.github.io/ChiReSSD/) · [Model](https://huggingface.co/Lab-MSP/ChiReSSD) ·
[Data](DATA.md)

Give it a target transcript and a short reference recording of a speaker, and it synthesizes
that transcript with canonical pronunciation while keeping the speaker's voice and prosody.

The problem it solves is specific. Ordinary style-preserving TTS treats disordered
articulation as part of the speaker's style, so it faithfully reproduces the mispronunciation
it was meant to correct. ChiReSSD separates the two pathways: pronunciation comes from the
text, identity and prosody come from the reference. It works on unseen speakers from a
few seconds of audio, needs no paired typical/atypical recordings, and trains no per-speaker
model.

This repository runs inference with the released checkpoint, and fine-tunes
your own on your own data.

## Install

```bash
conda env create -f environment.yml     # python 3.10 + ffmpeg + espeak-ng
conda activate chiressd
pip install -e .

chiressd-setup                          # fetch and patch upstream StyleTTS2
cp configs/paths.example.yaml paths.yaml
```

Then edit two lines in `paths.yaml`: `work_root` for scratch, and `data_root` if you intend to
fine-tune. 

`espeak-ng` and `ffmpeg` are system packages, not pip ones — without them, phonemization and
audio decoding fail. `chiressd-setup` clones StyleTTS2 at a pinned commit and applies the
ChiReSSD patches; see [below](#relationship-to-styletts2).

## Synthesize

```python
from chiressd.model import ChiReSSD
from chiressd.config import load_preset

model = ChiReSSD.from_pretrained()  # downloads the checkpoint
style = model.compute_style("speaker_reference.wav")  # 4+ seconds works well

wav = model.synthesize(
    "butterfly butterfly butterfly",
    style,
    seed=1234,
    **load_preset("default").as_kwargs(),
)
```

In batch, from a manifest of `key,text,reference` rows:

```bash
chiressd-synth manifest.tsv --preset default --seed 1234
```

**Pass a seed if you want the same audio twice.** The diffusion sampler is ancestral and adds
fresh noise at every step, so unseeded calls differ — in duration as well as waveform.

`alpha` and `beta` interpolate the sampled style with the reference style, 1.0 fully sampled
and 0.0 fully reference-driven, for timbre and prosody. They are required arguments with no
default: a low `alpha` leans on the reference acoustics, which is where disordered
articulation lives, so an inherited default could reproduce the very mispronunciation you are
correcting. Two presets ship — `default` (α 0.8, β 0.5, 10 steps), the released operating
point, and `torgo` (α 1.0, β 0.5, 15 steps) for adult dysarthric speech.

## Fine-tune on your own data

```bash
chiressd-prepare metadata.txt                     # filename|text -> filename|phonemes|speaker
chiressd-train --config config_ft_F0_v5 --dry-run # inspect the rendered config
chiressd-train --config config_ft_F0_v5
```

`metadata.txt` is one `filename|text|speaker_id` row per utterance; `chiressd-prepare`
phonemizes it into the form the trainer reads. See [DATA.md](DATA.md).

`config_ft_F0_v5` is the recipe behind the released checkpoint: 4 epochs, batch 4,
`lambda_F0` 20, style diffusion from epoch 2, the adversarial objective from epoch 3. The
pitch loss is weighted 20× upstream's because the pitch extractor was pretrained on adult
voices and children's F0 is higher and more variable. Four epochs, not more, because longer
schedules start fitting the disordered articulation itself.

**It needs more than 48 GB of GPU memory as shipped** — the original run spread batch 4 across
two 48 GB cards. On a single smaller card, lower `batch_size` and `max_len` in a copy of the
config.

`--dry-run` renders the config and stops. Every real run writes its rendered config beside the
output, which is the record of what that run actually used.

## Evaluate

If you have the original recordings to compare against:

```bash
chiressd-eval speaker --hypothesis <recon_dir> --reference <orig_dir>
chiressd-eval pitch   --hypothesis <recon_dir> --reference <orig_dir>
```

`speaker` is cosine similarity of Resemblyzer embeddings (`pip install -e ".[eval]"`) — did
the voice survive. `pitch` is the F0 difference in percent and semitones; it uses no learned
representation, so it carries no bias against child or disordered voices. Reconstructions are
paired with originals by key, never by directory-listing order.

## Layout

```
src/chiressd/     the package: model, metrics, data prep, CLI
configs/          the training recipe, inference presets, paths.example.yaml
notebooks/        walkthroughs for data preparation and synthesis
docs/             the demo page
```

Commands: `chiressd-setup`, `chiressd-prepare`, `chiressd-train`, `chiressd-style`,
`chiressd-synth`, `chiressd-eval`.

Every filesystem location lives in one `paths.yaml`; nothing is hardcoded. Individual roots
can be overridden per run with `CHIRESSD_WORK_ROOT`, `CHIRESSD_DATA_ROOT`,
`CHIRESSD_UPSTREAM`, `CHIRESSD_CACHE_ROOT`.

## Relationship to StyleTTS2

ChiReSSD builds upon StyleTTS2, with its modifications published as patch files in
[`src/chiressd/_upstream_patches/`](src/chiressd/_upstream_patches). `chiressd-setup` clones
upstream at a pinned commit and applies them.

Upstream is not vendored: it tracks ~134 MB of pretrained binaries that would otherwise live
in this repository's history forever, and keeping the changes as patches keeps them legible.
Nor is it a submodule, since a submodule cannot carry local modifications —
`git submodule update` would silently revert them, and one of the patches alters training
dynamics.

## Scope

No corpora ship here. See [DATA.md](DATA.md). The nine clips in `docs/audio/` back the
demo page and are the only audio in the repository.

## Intended use and limitations

Research on speech reconstruction and on automated clinical evaluation of speech sound
disorders.

- **Not a medical device.** No diagnostic or treatment decision should rest on its output, and
  it does not replace assessment by a licensed speech-language pathologist.
- **Accurate voice reproduction is the point, and therefore the risk.** The property that lets
  feedback be delivered in a speaker's own voice makes this a voice-cloning tool. Use it only
  with appropriate consent and ethical oversight.

## Citation

```bibtex
@inproceedings{rosero2026chiressd,
  title     = {Generative Reconstruction of Pediatric Disordered Speech
               for Automated Clinical Evaluation},
  author    = {Rosero, Karen and Yeo, Eunjung and Mortensen, David R. and
               Van't Slot, Cortney and Hallac, Rami R. and Busso, Carlos},
  booktitle = {Proceedings of the IEEE Spoken Language Technology Workshop
               2026 (SLT '26)},
  address   = {Palermo, Italy},
  year      = {2026},
}
```

See [CITATION.cff](CITATION.cff).

## License

MIT — see [LICENSE](LICENSE). Third-party components are acknowledged in [NOTICE](NOTICE).
