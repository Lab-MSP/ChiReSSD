"""Grapheme-to-phoneme conversion and tokenization.

Pronunciation reaches the model only through this path. The style pathway
carries identity and prosody; the phonemes carry *what* is said. That split is
the whole method, so the phonemizer's configuration is part of the method and
not an incidental detail.

The dialect default is ``en-gb``, not upstream's ``en-us``: the target speakers
are children with a Central Scottish accent, which British English approximates
considerably better than American English.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

DEFAULT_LANGUAGE = "en-gb"


class TextError(RuntimeError):
    pass


@lru_cache(maxsize=8)
def get_phonemizer(language: str = DEFAULT_LANGUAGE, with_stress: bool = True) -> Any:
    """An espeak-ng backend, cached per (language, stress) pair.

    Construction spawns espeak-ng, so it is cached: rebuilding it per utterance
    dominates runtime on a corpus of short clinical prompts.
    """
    try:
        import phonemizer
    except ImportError as exc:  # pragma: no cover
        raise TextError("phonemizer is required: pip install phonemizer") from exc

    try:
        return phonemizer.backend.EspeakBackend(
            language=language, preserve_punctuation=True, with_stress=with_stress
        )
    except RuntimeError as exc:  # pragma: no cover
        raise TextError(
            f"Could not start espeak-ng for language {language!r}. "
            "Install the system package (e.g. `apt install espeak-ng`)."
        ) from exc


def _word_tokenize(text: str) -> list[str]:
    from nltk.tokenize import word_tokenize

    try:
        return word_tokenize(text)
    except LookupError as exc:  # pragma: no cover
        raise TextError(
            "NLTK's punkt tokenizer data is missing. Run:\n"
            "  python -c \"import nltk; nltk.download('punkt_tab')\""
        ) from exc


def phonemize(
    text: str,
    language: str = DEFAULT_LANGUAGE,
    *,
    with_stress: bool = True,
    tokenize: bool = True,
) -> str:
    """Convert text to a spaced IPA phoneme string.

    The defaults are the **synthesis** convention: stress marks kept, and the
    phonemized string word-tokenized then rejoined on single spaces. Tokenizing
    *after* phonemization is what separates punctuation from adjacent phonemes
    so the text cleaner sees them as their own symbols.

    Training lists use the opposite convention — ``with_stress=False`` and no
    tokenization (see :mod:`chiressd.data.prepare_lists`). Both are selectable
    here, so set them to match whichever side you are feeding.
    """
    phonemized = get_phonemizer(language, with_stress).phonemize([text.strip()])[0]
    return " ".join(_word_tokenize(phonemized)) if tokenize else phonemized


def tokenize(phonemes: str) -> list[int]:
    """Map a phoneme string to model token ids, with the leading pad token.

    Uses upstream's ``TextCleaner``, so :func:`chiressd.vendor.activate` must
    have run first.
    """
    try:
        from text_utils import TextCleaner
    except ImportError as exc:
        raise TextError(
            "Upstream StyleTTS2 is not importable. Run chiressd-setup, or call "
            "chiressd.vendor.activate(<checkout>) first."
        ) from exc

    tokens = TextCleaner()(phonemes)
    # Leading 0 is the pad/BOS token the model is trained to expect.
    tokens.insert(0, 0)
    return tokens


def repeat_word(word: str, times: int = 3) -> str:
    """Repeat a stimulus word, matching the "say the word three times" protocol.

    The STAR prompts elicit each target word three times, so both the reference
    audio and the target text cover three productions.
    """
    if times < 1:
        raise ValueError("times must be >= 1")
    return " ".join([word.strip()] * times)
