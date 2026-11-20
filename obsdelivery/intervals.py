"""Immutable rounding-interval products over the observation library.

An interval product freezes, per sample, the exact set of rationals that
could have produced the metered bit pattern (nearest, ties to even), the
attainability of both endpoints and the neighbour patterns each endpoint
is derived from.  Products are immutable: correcting an observation leads
to a *new* product and never rewrites an issued one, so a reopened library
always reads an issued product back with the same content digest.
"""

from __future__ import annotations

import json
from typing import Any, Sequence

from .canonical import canonical_json, digest
from .errors import IntegrityError, ProductConflict, ProductNotFound, ValidationError
from .rounding import CONTRACT as INSTRUMENT_CONTRACT
from .rounding import certify_decoded, certify_missing
from .service import SemanticsService
from .storage import Store, utcnow


def _require_sample_id(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"sample_id must be a non-empty string, got {value!r}")
    return value


class IntervalService:
    """Publishes and re-opens immutable rounding-interval products."""

    def __init__(self, store: Store, semantics: SemanticsService | None = None):
        self.store = store
        self.semantics = semantics or SemanticsService(store)

    def publish_interval_product(
        self, product_id: str, sample_ids: Sequence[str]
    ) -> dict:
        """Publish (or replay) an immutable rounding-interval product."""
        if not isinstance(product_id, str) or not product_id:
            raise ValidationError("product_id must be a non-empty string")
        if isinstance(sample_ids, (str, bytes)) or not isinstance(sample_ids, Sequence):
            raise ValidationError("sample_ids must be a sequence")
        samples = [_require_sample_id(sample) for sample in sample_ids]
        if len(set(samples)) != len(samples):
            raise ValidationError("sample_ids contains duplicates")
        samples = sorted(samples)

        items = []
        for sample in samples:
            effective = self.semantics.effective_observation(sample)
            if effective is None:
                revision, interval = None, certify_missing()
            else:
                revision, decoded = effective
                interval = certify_decoded(decoded)
            items.append(
                {"sample_id": sample, "received_revision": revision, **interval.canonical()}
            )

        body = {
            "kind": "rounding_interval_product",
            "id": product_id,
            "instrument_contract": INSTRUMENT_CONTRACT,
            "items": items,
        }
        content_digest = digest(body)

        existing = self.store.conn.execute(
            "SELECT * FROM interval_products WHERE product_id=?", (product_id,)
        ).fetchone()
        if existing is not None:
            if existing["content_digest"] != content_digest:
                raise ProductConflict(
                    f"interval product {product_id!r} exists with different content"
                )
            return self._view(existing, created=False)

        with self.store.transaction() as conn:
            conn.execute(
                "INSERT INTO interval_products"
                "(product_id, instrument_contract, sample_ids_json, content_digest, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (
                    product_id,
                    canonical_json(INSTRUMENT_CONTRACT),
                    canonical_json(samples),
                    content_digest,
                    utcnow(),
                ),
            )
            for item in items:
                conn.execute(
                    "INSERT INTO interval_product_item"
                    "(product_id, sample_id, received_revision, original_bits, "
                    " certifiable, reason, canonical_json)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        product_id,
                        item["sample_id"],
                        item["received_revision"],
                        item["bits"],
                        1 if item["certifiable"] else 0,
                        item["reason"],
                        canonical_json(item),
                    ),
                )
        row = self.store.conn.execute(
            "SELECT * FROM interval_products WHERE product_id=?", (product_id,)
        ).fetchone()
        return self._view(row, created=True)

    def get_interval_product(self, product_id: str) -> dict:
        """Read an issued product back, re-verifying its stored content."""
        row = self.store.conn.execute(
            "SELECT * FROM interval_products WHERE product_id=?", (product_id,)
        ).fetchone()
        if row is None:
            raise ProductNotFound(product_id)
        return self._view(row, created=False)

    def _view(self, row, created: bool) -> dict:
        items = [
            json.loads(record["canonical_json"])
            for record in self.store.conn.execute(
                "SELECT canonical_json FROM interval_product_item "
                "WHERE product_id=? ORDER BY sample_id",
                (row["product_id"],),
            ).fetchall()
        ]
        body = {
            "kind": "rounding_interval_product",
            "id": row["product_id"],
            "instrument_contract": json.loads(row["instrument_contract"]),
            "items": items,
        }
        if digest(body) != row["content_digest"]:
            raise IntegrityError(
                f"interval_products:{row['product_id']} digest mismatch"
            )
        result = dict(body)
        result["sample_ids"] = json.loads(row["sample_ids_json"])
        result["content_digest"] = row["content_digest"]
        result["created"] = created
        return result
