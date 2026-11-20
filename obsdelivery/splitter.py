"""Blending splitters: versioned recipes and reachable-bin tasks.

Each splitter owns a recipe: a rational coefficient per ingredient channel
and a rational bias, evaluated against the *rounding intervals* of an
immutable interval product — never against the stored point value, which
would misclassify originals near a cut.  Channels naming the same sample
carry the same unknown original, so coefficients are netted per sample
before the exact interval arithmetic; different samples vary independently.

A task binds one interval product and one recipe version.  It is immutable
and replayable; the same id with a different request conflicts.  A
non-certifiable channel with a non-zero net coefficient rejects the whole
task (audited in ``splitter_attempts``, no task row), and the same task id
may be retried once the underlying product is corrected.  A channel whose
net coefficient is exactly zero cancels the unknown exactly and therefore
needs no certification.
"""

from __future__ import annotations

import json
from collections import Counter
from fractions import Fraction
from typing import Any, Iterable

from .canonical import canonical_json, digest, frac_str, parse_fraction
from .errors import (
    IntegrityError,
    RecipeConflict,
    RecipeNotFound,
    SplitterRejected,
    SplitterTaskConflict,
    SplitterTaskNotFound,
    ValidationError,
)
from .intervals import IntervalService
from .splitter_algo import (
    Range,
    endpoint_hits,
    net_coefficients,
    reachable_bins,
    scale_range,
    sum_ranges,
)
from .storage import Store, utcnow

DECODER = (
    "total = bias + sum(net_coefficient[sample] * original[sample]); "
    "bins are split at cut_points and each cut point belongs to the right bin"
)


def _require_rational(value: Any, field: str) -> Fraction:
    try:
        return parse_fraction(value, field=field)
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc


def _normalise_channels(channels: Any) -> list[dict]:
    if isinstance(channels, (str, bytes)) or not isinstance(channels, Iterable):
        raise ValidationError("channels must be an iterable of objects")
    normalised = []
    for channel in channels:
        if not isinstance(channel, dict):
            raise ValidationError(f"channel is not an object: {channel!r}")
        sample = channel.get("sample_id")
        if not isinstance(sample, str) or not sample.strip():
            raise ValidationError(
                f"channel sample_id must be a non-empty string, got {sample!r}"
            )
        coefficient = _require_rational(channel.get("coefficient"), "coefficient")
        normalised.append({"sample_id": sample, "coefficient": coefficient})
    # Channel order carries no meaning; canonicalise for a stable identity.
    normalised.sort(key=lambda entry: (entry["sample_id"], entry["coefficient"]))
    return normalised


def _normalise_cuts(cut_points: Any) -> list[Fraction]:
    if isinstance(cut_points, (str, bytes)) or not isinstance(cut_points, Iterable):
        raise ValidationError("cut_points must be an iterable of rationals")
    cuts = [_require_rational(cut, "cut_points") for cut in cut_points]
    if len(set(cuts)) != len(cuts):
        raise ValidationError("cut_points contains duplicates")
    return sorted(cuts)


class SplitterService:
    """Third responsibility: recipe versions and reachable-bin tasks."""

    def __init__(self, store: Store, intervals: IntervalService | None = None):
        self.store = store
        self.intervals = intervals or IntervalService(store)

    # -- recipes -------------------------------------------------------------
    def publish_recipe(
        self,
        recipe_id: str,
        channels: Iterable[dict],
        bias: Any = "0",
        cut_points: Iterable[Any] = (),
    ) -> dict:
        """Publish (or replay) an immutable splitter recipe version."""
        if not isinstance(recipe_id, str) or not recipe_id:
            raise ValidationError("recipe_id must be a non-empty string")
        normalised = _normalise_channels(channels)
        bias_value = _require_rational(bias, "bias")
        cuts = _normalise_cuts(cut_points)
        body = {
            "kind": "splitter_recipe",
            "recipe_id": recipe_id,
            "channels": [
                {"sample_id": entry["sample_id"], "coefficient": frac_str(entry["coefficient"])}
                for entry in normalised
            ],
            "bias": frac_str(bias_value),
            "cut_points": [frac_str(cut) for cut in cuts],
            "decoder": DECODER,
        }
        content_digest = digest(body)

        existing = self.store.conn.execute(
            "SELECT * FROM splitter_recipes WHERE recipe_id=?", (recipe_id,)
        ).fetchone()
        if existing is not None:
            if existing["content_digest"] != content_digest:
                raise RecipeConflict(
                    f"splitter recipe {recipe_id!r} exists with different content"
                )
            return self._recipe_view(existing, created=False)

        with self.store.transaction() as conn:
            conn.execute(
                "INSERT INTO splitter_recipes"
                "(recipe_id, channels_json, bias, cut_points_json, content_digest, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (
                    recipe_id,
                    canonical_json(body["channels"]),
                    body["bias"],
                    canonical_json(body["cut_points"]),
                    content_digest,
                    utcnow(),
                ),
            )
        row = self.store.conn.execute(
            "SELECT * FROM splitter_recipes WHERE recipe_id=?", (recipe_id,)
        ).fetchone()
        return self._recipe_view(row, created=True)

    def get_recipe(self, recipe_id: str) -> dict:
        row = self.store.conn.execute(
            "SELECT * FROM splitter_recipes WHERE recipe_id=?", (recipe_id,)
        ).fetchone()
        if row is None:
            raise RecipeNotFound(recipe_id)
        return self._recipe_view(row, created=False)

    def _recipe_view(self, row, created: bool) -> dict:
        body = {
            "kind": "splitter_recipe",
            "recipe_id": row["recipe_id"],
            "channels": json.loads(row["channels_json"]),
            "bias": row["bias"],
            "cut_points": json.loads(row["cut_points_json"]),
            "decoder": DECODER,
        }
        if digest(body) != row["content_digest"]:
            raise IntegrityError(f"splitter_recipes:{row['recipe_id']} digest mismatch")
        result = dict(body)
        result["content_digest"] = row["content_digest"]
        result["created"] = created
        return result

    # -- tasks ---------------------------------------------------------------
    def submit(self, task_id: str, product_id: str, recipe_id: str) -> dict:
        """All-or-nothing reachable-bin task.  Raises ``SplitterRejected``."""
        if not isinstance(task_id, str) or not task_id:
            raise ValidationError("task_id must be a non-empty string")
        product = self.intervals.get_interval_product(product_id)
        recipe = self.get_recipe(recipe_id)
        request_body = {
            "product_id": product_id,
            "product_digest": product["content_digest"],
            "recipe_id": recipe_id,
            "recipe_digest": recipe["content_digest"],
        }
        request_digest = digest(request_body)

        existing = self.store.conn.execute(
            "SELECT * FROM splitter_tasks WHERE task_id=?", (task_id,)
        ).fetchone()
        if existing is not None:
            if existing["request_digest"] != request_digest:
                raise SplitterTaskConflict(
                    f"splitter task {task_id!r} exists for a different product/recipe"
                )
            return self._task_view(existing, created=False)

        items = {item["sample_id"]: item for item in product["items"]}
        channels = recipe["channels"]
        net = net_coefficients(channels)
        channel_counts = Counter(channel["sample_id"] for channel in channels)
        unknown = [sample for sample in sorted(net) if sample not in items]
        if unknown:
            raise ValidationError(
                f"recipe names sample(s) {unknown} absent from interval product "
                f"{product_id!r}"
            )

        contributions: list[dict] = []
        problems: list[dict] = []
        ranges: list[Range] = []
        for sample in sorted(net):
            coefficient = net[sample]
            item = items[sample]
            entry = {
                "sample_id": sample,
                "channels": channel_counts[sample],
                "net_coefficient": frac_str(coefficient),
                "certifiable": item["certifiable"],
                "bits": item["bits"],
            }
            if coefficient == 0:
                entry["contribution"] = {
                    "kind": "exact_zero",
                    "note": (
                        "net coefficient is zero: the same unknown feeds every "
                        "channel of this sample and cancels exactly, so "
                        "certifiability is irrelevant"
                    ),
                    "range": None,
                }
            elif not item["certifiable"]:
                entry["contribution"] = None
                problems.append(
                    {
                        "sample_id": sample,
                        "reason": "uncertifiable_net_contribution",
                        "net_coefficient": frac_str(coefficient),
                        "bits": item["bits"],
                        "certifiable_reason": item["reason"],
                        "detail": (
                            "a non-certifiable channel with a non-zero net "
                            "coefficient makes the total unbounded; the batch "
                            "cannot be sorted automatically"
                        ),
                    }
                )
            else:
                interval = item["interval"]
                scaled = scale_range(
                    Range(
                        Fraction(interval["low"]),
                        interval["low_included"],
                        Fraction(interval["high"]),
                        interval["high_included"],
                    ),
                    coefficient,
                )
                ranges.append(scaled)
                entry["contribution"] = {"kind": "interval", "range": scaled.canonical()}
            contributions.append(entry)

        if problems:
            attempt_id = self._record_attempt(
                task_id, product_id, recipe_id, request_digest, "rejected", problems
            )
            raise SplitterRejected(task_id, problems, attempt_id=attempt_id)

        bias = Fraction(recipe["bias"])
        cuts = [Fraction(cut) for cut in recipe["cut_points"]]
        total = sum_ranges(ranges, bias)
        reachable = reachable_bins(total, cuts)
        hits = [
            {"cut": frac_str(hit["cut"]), "end": hit["end"], "attainable": hit["attainable"]}
            for hit in endpoint_hits(total, cuts)
        ]
        bins = [
            {
                "index": index,
                "lower_cut": None if index == 0 else frac_str(cuts[index - 1]),
                "upper_cut": None if index == len(cuts) else frac_str(cuts[index]),
                # Each cut point belongs to the bin on its right.
                "lower_included": index > 0,
                "upper_included": False,
                "reachable": index in reachable,
            }
            for index in range(len(cuts) + 1)
        ]
        result_body = {
            "kind": "splitter_task",
            "task_id": task_id,
            "product_id": product_id,
            "product_digest": product["content_digest"],
            "instrument_contract": product["instrument_contract"],
            "recipe_id": recipe_id,
            "recipe_digest": recipe["content_digest"],
            "recipe": {
                "channels": recipe["channels"],
                "bias": recipe["bias"],
                "cut_points": recipe["cut_points"],
                "decoder": DECODER,
            },
            "contributions": contributions,
            "total": total.canonical(),
            "bins": bins,
            "reachable_bins": reachable,
            "endpoint_hits": hits,
            "auto_sortable": len(reachable) == 1,
            "ambiguous": len(reachable) > 1,
        }
        content_digest = digest(result_body)
        with self.store.transaction() as conn:
            conn.execute(
                "INSERT INTO splitter_tasks"
                "(task_id, product_id, recipe_id, request_digest, result_json,"
                " content_digest, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    task_id,
                    product_id,
                    recipe_id,
                    request_digest,
                    canonical_json(result_body),
                    content_digest,
                    utcnow(),
                ),
            )
            conn.execute(
                "INSERT INTO splitter_attempts"
                "(task_id, product_id, recipe_id, request_digest, status,"
                " reasons_json, created_at)"
                " VALUES (?, ?, ?, ?, 'published', ?, ?)",
                (task_id, product_id, recipe_id, request_digest, canonical_json([]), utcnow()),
            )
        row = self.store.conn.execute(
            "SELECT * FROM splitter_tasks WHERE task_id=?", (task_id,)
        ).fetchone()
        return self._task_view(row, created=True)

    def _record_attempt(
        self,
        task_id: str,
        product_id: str,
        recipe_id: str,
        request_digest: str,
        status: str,
        reasons: list,
    ) -> int:
        with self.store.transaction() as conn:
            cursor = conn.execute(
                "INSERT INTO splitter_attempts"
                "(task_id, product_id, recipe_id, request_digest, status,"
                " reasons_json, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    task_id,
                    product_id,
                    recipe_id,
                    request_digest,
                    status,
                    canonical_json(reasons),
                    utcnow(),
                ),
            )
            return cursor.lastrowid

    def get_task(self, task_id: str) -> dict:
        row = self.store.conn.execute(
            "SELECT * FROM splitter_tasks WHERE task_id=?", (task_id,)
        ).fetchone()
        if row is None:
            raise SplitterTaskNotFound(task_id)
        return self._task_view(row, created=False)

    def _task_view(self, row, created: bool) -> dict:
        """Independently recompute the stored conclusions before returning."""
        result = json.loads(row["result_json"])
        if digest(result) != row["content_digest"]:
            raise IntegrityError(f"splitter_tasks:{row['task_id']} digest mismatch")

        recipe = result["recipe"]
        net = net_coefficients(recipe["channels"])
        expected_net = {sample: frac_str(coefficient) for sample, coefficient in net.items()}
        stored_net = {
            entry["sample_id"]: entry["net_coefficient"]
            for entry in result["contributions"]
        }
        if stored_net != expected_net:
            raise IntegrityError(
                f"splitter_tasks:{row['task_id']} net coefficients disagree with the recipe"
            )

        ranges = []
        for entry in result["contributions"]:
            contribution = entry["contribution"]
            if contribution is None:
                raise IntegrityError(
                    f"splitter_tasks:{row['task_id']} holds an uncertified contribution"
                )
            if contribution["kind"] == "exact_zero":
                continue
            interval = contribution["range"]
            ranges.append(
                Range(
                    Fraction(interval["low"]),
                    interval["low_included"],
                    Fraction(interval["high"]),
                    interval["high_included"],
                )
            )
        total = sum_ranges(ranges, Fraction(recipe["bias"]))
        if total.canonical() != result["total"]:
            raise IntegrityError(f"splitter_tasks:{row['task_id']} total range mismatch")

        cuts = [Fraction(cut) for cut in recipe["cut_points"]]
        if reachable_bins(total, cuts) != result["reachable_bins"]:
            raise IntegrityError(f"splitter_tasks:{row['task_id']} reachable bins mismatch")
        expected_hits = [
            {"cut": frac_str(hit["cut"]), "end": hit["end"], "attainable": hit["attainable"]}
            for hit in endpoint_hits(total, cuts)
        ]
        if expected_hits != result["endpoint_hits"]:
            raise IntegrityError(f"splitter_tasks:{row['task_id']} endpoint hits mismatch")

        view = dict(result)
        view["content_digest"] = row["content_digest"]
        view["created"] = created
        return view

    def attempts(self, task_id: str) -> list[dict]:
        rows = self.store.conn.execute(
            "SELECT attempt_id, task_id, product_id, recipe_id, request_digest,"
            " status, reasons_json, created_at FROM splitter_attempts"
            " WHERE task_id=? ORDER BY attempt_id",
            (task_id,),
        ).fetchall()
        return [
            {
                "attempt_id": row["attempt_id"],
                "task_id": row["task_id"],
                "product_id": row["product_id"],
                "recipe_id": row["recipe_id"],
                "request_digest": row["request_digest"],
                "status": row["status"],
                "reasons": json.loads(row["reasons_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]
