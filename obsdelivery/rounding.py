"""Rounding-interval semantics for metered binary32 readings (pure).

The meter rounds the true quantity to the nearest binary32 value, ties to
even, and reports the resulting bit pattern.  Conclusions for the blending
splitter must hold for *every* original rational that could have produced
the reported pattern, so each certifiable pattern is assigned the exact
interval between the midpoints to its neighbours, with per-endpoint
attainability.

A tie at a midpoint rounds to the neighbour with the even mantissa, so a
midpoint is attainable (included in the interval) iff the reported
pattern's own mantissa is even.  Adjacent values may cross an exponent
boundary — the left neighbour of ``3f800000`` (1.0) is ``3f7fffff``
(0.99999994, exponent 126) — which halves the step; midpoints are computed
from the exact neighbour values, never from a fixed ulp.

Instrument contract ``rn-te-normal-neighbours-v1`` certifies only normal
finite non-zero values whose immediate neighbours on both sides are also
normal finite values.  Every other pattern is reported with an explicit
non-certifiable reason and the original bits are preserved.

No host floating point is involved: endpoints are exact ``Fraction``s.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from . import semantics
from .canonical import frac_str

CONTRACT_ID = "rn-te-normal-neighbours-v1"

#: The instrument contract this library certifies rounding intervals under.
CONTRACT = {
    "id": CONTRACT_ID,
    "rounding": "nearest, ties-to-even",
    "certifies": (
        "normal finite non-zero values whose immediate neighbours "
        "on both sides are normal finite values"
    ),
}

REASON_MISSING = "missing_observation"
REASON_NONFINITE = "nonfinite"
REASON_ZERO = "signed_zero_not_normal"
REASON_SUBNORMAL = "subnormal_not_normal"
REASON_LOWER_NEIGHBOUR = "lower_neighbour_not_normal_finite"
REASON_UPPER_NEIGHBOUR = "upper_neighbour_not_normal_finite"

_REASON_DETAILS = {
    REASON_MISSING: "no observation has been admitted for this sample",
    REASON_NONFINITE: "NaN and infinities have no finite rounding interval",
    REASON_ZERO: "signed zero is not a normal non-zero value",
    REASON_SUBNORMAL: "subnormal values are not normal numbers",
    REASON_LOWER_NEIGHBOUR: (
        "the immediate neighbour closer to zero is not a normal finite value"
    ),
    REASON_UPPER_NEIGHBOUR: (
        "the immediate neighbour farther from zero is not a normal finite value"
    ),
}


@dataclass(frozen=True)
class RoundingInterval:
    """The exact set of originals that round to one reported pattern."""

    bits: str | None
    kind: str
    sign: int | None
    certifiable: bool
    reason: str | None
    value: Fraction | None
    low: Fraction | None = None
    low_included: bool | None = None
    high: Fraction | None = None
    high_included: bool | None = None
    lower_neighbour_bits: str | None = None
    upper_neighbour_bits: str | None = None

    @property
    def reason_detail(self) -> str | None:
        return _REASON_DETAILS.get(self.reason)

    def canonical(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "bits": self.bits,
            "kind": self.kind,
            "sign": self.sign,
            "certifiable": self.certifiable,
            "reason": self.reason,
            "reason_detail": self.reason_detail,
            "value": None if self.value is None else frac_str(self.value),
            "interval": None,
            "endpoints": None,
        }
        if self.certifiable:
            body["interval"] = {
                "low": frac_str(self.low),
                "low_included": self.low_included,
                "high": frac_str(self.high),
                "high_included": self.high_included,
            }
            body["endpoints"] = {
                "lower": {
                    "midpoint": frac_str(self.low),
                    "included": self.low_included,
                    "between": [self.lower_neighbour_bits, self.bits],
                },
                "upper": {
                    "midpoint": frac_str(self.high),
                    "included": self.high_included,
                    "between": [self.bits, self.upper_neighbour_bits],
                },
            }
        return body


def _reject(decoded: semantics.Decoded, reason: str) -> RoundingInterval:
    return RoundingInterval(
        bits=decoded.bits,
        kind=decoded.kind,
        sign=decoded.sign,
        certifiable=False,
        reason=reason,
        value=decoded.value,
    )


def certify_missing() -> RoundingInterval:
    """No observation was ever admitted for the sample."""
    return RoundingInterval(
        bits=None,
        kind=semantics.MISSING,
        sign=None,
        certifiable=False,
        reason=REASON_MISSING,
        value=None,
    )


def certify_bits(bits: Any) -> RoundingInterval:
    return certify_decoded(semantics.decode(semantics.parse_bits(bits)))


def certify_decoded(decoded: semantics.Decoded) -> RoundingInterval:
    """Classify one decoded pattern and compute its rounding interval."""
    if decoded.kind in semantics.NONFINITE_KINDS:
        return _reject(decoded, REASON_NONFINITE)
    if decoded.kind == semantics.ZERO:
        return _reject(decoded, REASON_ZERO)
    if decoded.exponent == 0:
        return _reject(decoded, REASON_SUBNORMAL)
    magnitude = decoded.raw & 0x7FFFFFFF
    predecessor = magnitude - 1  # neighbour closer to zero
    successor = magnitude + 1  # neighbour farther from zero
    if predecessor >> 23 < 1:
        return _reject(decoded, REASON_LOWER_NEIGHBOUR)
    if successor >> 23 > 254:
        return _reject(decoded, REASON_UPPER_NEIGHBOUR)

    value = abs(decoded.value)
    low_mid = (abs(semantics.decode(predecessor).value) + value) / 2
    high_mid = (value + abs(semantics.decode(successor).value)) / 2
    # A midpoint tie rounds to the even mantissa; both midpoints are
    # attainable exactly when this pattern's own mantissa is even.
    attainable = decoded.fraction % 2 == 0
    if decoded.sign:
        low, high = -high_mid, -low_mid
        lower_bits = semantics.format_bits(0x80000000 | successor)
        upper_bits = semantics.format_bits(0x80000000 | predecessor)
    else:
        low, high = low_mid, high_mid
        lower_bits = semantics.format_bits(predecessor)
        upper_bits = semantics.format_bits(successor)
    return RoundingInterval(
        bits=decoded.bits,
        kind=decoded.kind,
        sign=decoded.sign,
        certifiable=True,
        reason=None,
        value=decoded.value,
        low=low,
        low_included=attainable,
        high=high,
        high_included=attainable,
        lower_neighbour_bits=lower_bits,
        upper_neighbour_bits=upper_bits,
    )
