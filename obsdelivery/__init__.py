"""Independent observation-semantics service and decimal delivery library.

Two responsibilities with a real boundary, plus a third built on top:

* ``obsdelivery.semantics`` / ``obsdelivery.service`` interpret original
  binary32 bit patterns exactly and judge them against references.  This works
  with no delivery library present.
* ``obsdelivery.delivery_algo`` / ``obsdelivery.delivery`` take a *published*
  semantic product and an independent reader contract and decide a concrete
  decimal delivery, fail-closed and all-or-nothing.
* ``obsdelivery.rounding`` / ``obsdelivery.intervals`` publish immutable
  rounding-interval products ("which originals could have produced this
  pattern"), and ``obsdelivery.splitter_algo`` / ``obsdelivery.splitter``
  evaluate versioned blending recipes against those intervals, fail-closed
  and all-or-nothing.
"""

from .contracts import ReaderContract
from .delivery import DeliveryService
from .delivery_algo import ScaleAttempt, Selection, round_half_even, select_representation
from .errors import (
    BatchRejected,
    DeliveryIdConflict,
    DeliveryNotFound,
    DeliveryRejected,
    IntegrityError,
    ProductConflict,
    ProductNotFound,
    RecipeConflict,
    RecipeNotFound,
    ReferenceConflict,
    ReferenceNotFound,
    ReportConflict,
    ReportNotFound,
    SplitterRejected,
    SplitterTaskConflict,
    SplitterTaskNotFound,
    ValidationError,
)
from .intervals import IntervalService
from .rounding import CONTRACT as INSTRUMENT_CONTRACT
from .rounding import RoundingInterval, certify_bits, certify_decoded
from .semantics import decode, interpret, judge, parse_bits
from .service import SemanticsService
from .splitter import SplitterService
from .splitter_algo import (
    Range,
    endpoint_hits,
    net_coefficients,
    reachable_bins,
    scale_range,
    sum_ranges,
)
from .storage import Store

__all__ = [
    "ReaderContract",
    "DeliveryService",
    "ScaleAttempt",
    "Selection",
    "round_half_even",
    "select_representation",
    "BatchRejected",
    "DeliveryIdConflict",
    "DeliveryNotFound",
    "DeliveryRejected",
    "IntegrityError",
    "ProductConflict",
    "ProductNotFound",
    "RecipeConflict",
    "RecipeNotFound",
    "ReferenceConflict",
    "ReferenceNotFound",
    "ReportConflict",
    "ReportNotFound",
    "SplitterRejected",
    "SplitterTaskConflict",
    "SplitterTaskNotFound",
    "ValidationError",
    "IntervalService",
    "INSTRUMENT_CONTRACT",
    "RoundingInterval",
    "certify_bits",
    "certify_decoded",
    "decode",
    "interpret",
    "judge",
    "parse_bits",
    "SemanticsService",
    "SplitterService",
    "Range",
    "endpoint_hits",
    "net_coefficients",
    "reachable_bins",
    "scale_range",
    "sum_ranges",
    "Store",
]
