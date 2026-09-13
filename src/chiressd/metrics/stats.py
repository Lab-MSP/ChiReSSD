"""Significance testing.

A paired test is only valid when element *i* of both samples refers to the same
utterance. :func:`paired_ttest` therefore requires the keys and refuses to run
if they disagree, so a mismatch surfaces as an error rather than as a p-value
that looks fine.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class TTestResult:
    label: str
    statistic: float
    pvalue: float
    n: int

    def significant(self, alpha: float = 0.05) -> bool:
        return self.pvalue < alpha

    def __str__(self) -> str:
        return f"{self.label}: t = {self.statistic:.4f}, p = {self.pvalue:.3e}, n = {self.n}"


def paired_ttest(
    a: Sequence[float],
    b: Sequence[float],
    *,
    keys_a: Sequence[str],
    keys_b: Sequence[str],
    label: str = "",
) -> TTestResult:
    """Two-sided paired t-test over two aligned samples.

    ``keys_a`` and ``keys_b`` identify what each observation refers to and must
    match element-for-element. Build them with
    :func:`chiressd.metrics.pairing.align_conditions`.
    """
    from scipy.stats import ttest_rel

    if len(a) != len(keys_a) or len(b) != len(keys_b):
        raise ValueError("Each sample must have one key per observation.")

    if list(keys_a) != list(keys_b):
        mismatched = [(x, y) for x, y in zip(keys_a, keys_b, strict=False) if x != y]
        raise ValueError(
            f"Samples are not aligned: {len(mismatched)} position(s) differ "
            f"(first: {mismatched[0] if mismatched else 'length mismatch'}). "
            "A paired test over unaligned samples is meaningless -- align them with "
            "pairing.align_conditions() first."
        )

    if len(a) < 2:
        raise ValueError("A paired t-test needs at least two observations.")

    statistic, pvalue = ttest_rel(list(a), list(b))
    return TTestResult(label=label, statistic=float(statistic), pvalue=float(pvalue), n=len(a))
