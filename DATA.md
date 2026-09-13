# Data

**No speech corpus is distributed with this repository.** The recordings this model was built
on are identifiable clinical speech, most of it from children, held under agreements that do
not permit redistribution.

You need data of your own only to fine-tune. Inference with the released checkpoint needs
nothing but a reference recording of whoever you want to synthesize.

The nine clips in [`docs/audio/`](docs/audio) back the demo page and are the only audio here.

## Fine-tuning on your own data

### What you need

Audio files, and one metadata row per utterance:

```
filename.wav|the spoken text|speaker_id
```

`speaker_id` is optional and defaults to 0. Audio should be mono; it is resampled to 24 kHz.
An hour or so is workable — the released checkpoint was fine-tuned on about 1,700 utterances.

Point `paths.yaml` at it:

```yaml
data_root: /path/to/your/data

datasets:
  train:
    audio: ${data_root}/audio
    train_list: ${data_root}/train_list.txt
    val_list: ${data_root}/val_list.txt
```

### Building the lists

```bash
chiressd-prepare metadata.txt
```

This phonemizes each transcript and writes `train_list.txt` and `val_list.txt` in the form the
trainer reads:

```
filename.wav|fɔː faɪv sɪks sɛvən|38
```

The middle field is **phonemes, not text**. Getting that wrong fails quietly: training runs
and simply learns from mis-specified input.

Two defaults worth knowing:

- `--language en-gb`. The original speakers had a Central Scottish accent, which British
  English approximates far better than American. Change it to match your speakers.
- Training lists carry **no stress marks**, although synthesis phonemizes *with* them. That
  asymmetry is inherited from the original pipeline. Pass `with_stress=True, tokenize=True` to
  `build_entries()` if you would rather the two agree.

### A public corpus to start from

If you want child disordered speech and have none, the UltraSuite benchmark is a reasonable
starting point:

<https://huggingface.co/datasets/changelinglab/ultrasuite-benchmark>

It carries its own licence and access conditions.

## The base checkpoint

Fine-tuning starts from the StyleTTS2 LibriTTS second-stage checkpoint. It is **not**
redistributed here — download it from the [upstream release](https://github.com/yl4579/StyleTTS2)
and point `models.libritts_base` at it.

The frozen helper models (ASR text aligner, JDCNet pitch extractor, PL-BERT) ship inside the
upstream repository and arrive with the checkout `chiressd-setup` creates. Nothing to fetch
separately.

## If you are working with clinical recordings

Two things are easy to get wrong:

- **Style caches are speaker-derived.** `outputs.styles` holds 256-dimensional vectors
  extracted from reference recordings. They are not audio and cannot be inverted to audio, but
  they are speaker-specific. Keep them in working storage, not in a repository.
- **Notebook outputs leak.** Executed cells can embed audio and transcripts of it. The
  `nbstripout` pre-commit hook strips them; that is a privacy control, not housekeeping.
