"""Turning a corpus of recordings and transcripts into training lists."""

from __future__ import annotations

from .prepare_lists import Entry, build_entries, read_metadata, split_entries, write_list

__all__ = ["Entry", "build_entries", "read_metadata", "split_entries", "write_list"]
