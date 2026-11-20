"""Exact reachable-bin computation for blending splitters (pure).

A recipe assigns each ingredient channel a rational coefficient and the
batch a rational bias.  Channels naming the same sample carry the *same*
unknown original, so coefficients are netted per sample first; only then
is each certified rounding interval scaled by its net coefficient and
summed.  Different samples vary independently, so the total range is the
Minkowski sum of the scaled intervals plus the bias.

Cut points are rational and each cut point belongs to the bin on its
right: with cuts ``c1 < ... < cn`` the bins are
``(-inf, c1), [c1, c2), ..., [cn, +inf)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable

from .canonical import frac_str


@dataclass(frozen=True)
class Range:
    """A rational interval with per-endpoint inclusion flags."""

    low: Fraction
    low_included: bool
    high: Fraction
    high_included: bool

    def canonical(self) -> dict:
        return {
            "low": frac_str(self.low),
            "low_included": self.low_included,
            "high": frac_str(self.high),
            "high_included": self.high_included,
        }


def net_coefficients(channels: Iterable[dict]) -> dict[str, Fraction]:
    """Sum channel coefficients per sample (same sample = same unknown)."""
    net: dict[str, Fraction] = {}
    for channel in channels:
        sample = channel["sample_id"]
        net[sample] = net.get(sample, Fraction(0)) + Fraction(channel["coefficient"])
    return net


def scale_range(interval: Range, coefficient: Fraction) -> Range:
    """Scale by a non-zero coefficient, flipping the ends when negative."""
    if coefficient > 0:
        return Range(
            coefficient * interval.low,
            interval.low_included,
            coefficient * interval.high,
            interval.high_included,
        )
    if coefficient < 0:
        return Range(
            coefficient * interval.high,
            interval.high_included,
            coefficient * interval.low,
            interval.low_included,
        )
    raise ValueError("a zero net coefficient contributes an exact zero, not a range")


def sum_ranges(ranges: Iterable[Range], bias: Fraction) -> Range:
    """Minkowski sum plus bias; an end stays included iff every end is."""
    low = high = bias
    low_included = high_included = True
    for item in ranges:
        low += item.low
        high += item.high
        low_included = low_included and item.low_included
        high_included = high_included and item.high_included
    return Range(low, low_included, high, high_included)


def reachable_bins(total: Range, cut_points: Iterable[Fraction]) -> list[int]:
    """0-based indices of every bin the total range can reach.

    Bin ``i`` is ``[cuts[i-1], cuts[i])`` with infinities at the ends; the
    lower cut belongs to the bin, the upper cut does not.
    """
    cuts = list(cut_points)
    reachable = []
    for index in range(len(cuts) + 1):
        if index < len(cuts) and not total.low < cuts[index]:
            continue  # the whole range starts at or right of the upper cut
        if index > 0:
            cut = cuts[index - 1]
            if not (total.high > cut or (total.high == cut and total.high_included)):
                continue  # the whole range ends at or left of the lower cut
        reachable.append(index)
    return reachable


def endpoint_hits(total: Range, cut_points: Iterable[Fraction]) -> list[dict]:
    """Cut points that coincide with a range endpoint, with attainability."""
    hits = []
    for cut in cut_points:
        if total.low == cut:
            hits.append({"cut": cut, "end": "low", "attainable": total.low_included})
        if total.high == cut:
            hits.append({"cut": cut, "end": "high", "attainable": total.high_included})
    return hits
