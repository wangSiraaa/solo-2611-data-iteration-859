"""Public artefact rendering and verification.

* ``public_semantic_product`` — immutable canonical observation product.
* ``public_delivery`` — standalone delivery document with enough information
  for an independent reader to re-check the actual decimal representation
  without consulting the current observation state.
* ``public_splitter_task`` — standalone splitter document carrying the
  interval product, the recipe and the reachable-bin conclusions, so an
  independent reader can re-derive every interval from the original bits
  and recompute the total range, the reachable bins and the endpoint hits.

All builders re-verify the stored content digests before rendering.
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

from .canonical import digest, frac_str
from .delivery import DeliveryService
from .rounding import certify_bits
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


def public_semantic_product(semantics: SemanticsService, product_id: str) -> dict:
    product = semantics.get_semantic_product(product_id)
    return {
        "kind": "semantic_product",
        "product_id": product["id"],
        "reference_version": product["reference_version"],
        "reference_digest": product["reference_digest"],
        "content_digest": product["content_digest"],
        "sample_ids": product["sample_ids"],
        "records": [
            {
                "sample_id": item["sample_id"],
                "received_revision": item["received_revision"],
                "original_bits": item["bits"],
                "kind": item["kind"],
                "sign": item["sign"],
                "payload": item["payload"],
                "exact_value": item["value"],
                "negated_value": item["negated_value"],
                "expected": item["expected"],
                "tolerance": item["tolerance"],
                "status": item["status"],
                "difference": item["difference"],
                "abs_difference": item["abs_difference"],
            }
            for item in product["items"]
        ],
    }


def public_delivery(delivery: DeliveryService, delivery_id: str) -> dict:
    view = delivery.get_delivery(delivery_id)
    return {
        "kind": "delivery",
        "delivery_id": view["delivery_id"],
        "canonical_product": {
            "product_id": view["product_id"],
            "content_digest": view["product_digest"],
        },
        "reference": {
            "version": view["reference_version"],
            "content_digest": view["reference_digest"],
        },
        "reader_contract": view["contract"],
        "contract_id": view["contract_id"],
        "content_digest": view["content_digest"],
        "lines": [
            {
                "sample_id": line["sample_id"],
                "original_bits": line["original_bits"],
                "kind": line["kind"],
                "sign": line["sign"],
                "original_exact_value": line["exact_value"],
                "original_negated_value": line["negated_value"],
                "semantic_status": line["status"],
                "delivered_status": line["delivered_status"],
                "coefficient": line["coefficient"],
                "scale": line["scale"],
                "reader_value": line["reader_value"],
                "exact_error": line["error"],
                "decoder": "coefficient / 10**scale",
                "evidence": line["evidence"],
            }
            for line in view["lines"]
        ],
    }


def verify_public_delivery(document: dict) -> list[str]:
    """Re-derive all checks from a standalone public document.

    Returns a list of textual problems (empty means the document is
    internally consistent).  It does not need a database: it recomputes the
    coefficient bound, error budget, scale legality and reader value.
    """
    problems: list[str] = []
    contract = document["reader_contract"]
    allowed = set(contract["allowed_scales"])
    bound = contract["max_abs_coefficient"]
    budget = Fraction(contract["max_abs_error"])
    required = {
        "sample_id", "original_bits", "kind", "sign", "original_exact_value",
        "semantic_status", "delivered_status", "coefficient", "scale",
        "reader_value", "exact_error",
    }
    for line in document["lines"]:
        missing = required - line.keys()
        if missing:
            problems.append(f"{line.get('sample_id')}: missing {sorted(missing)}")
            continue
        sample = line["sample_id"]
        scale = line["scale"]
        coefficient = line["coefficient"]
        if scale not in allowed:
            problems.append(f"{sample}: scale {scale} not allowed")
        if abs(coefficient) > bound:
            problems.append(f"{sample}: coefficient {coefficient} exceeds {bound}")
        if line["semantic_status"] != line["delivered_status"]:
            problems.append(f"{sample}: semantic status changed on delivery")
        value = Fraction(line["original_exact_value"])
        delivered = Fraction(coefficient, 10 ** scale)
        error = delivered - value
        if str(Fraction(line["exact_error"])) != str(error):
            problems.append(f"{sample}: stored error {line['exact_error']} != {error}")
        if abs(error) > budget:
            problems.append(f"{sample}: error {error} exceeds budget {budget}")
        if str(Fraction(line["reader_value"])) != str(delivered):
            problems.append(f"{sample}: reader value mismatch")
    return problems


def write_json(path: str | Path, document: dict) -> Path:
    target = Path(path)
    target.write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")
    return target


def public_splitter_task(splitter: SplitterService, task_id: str) -> dict:
    """Standalone splitter document: endpoint hits and ambiguity included."""
    view = splitter.get_task(task_id)
    product = splitter.intervals.get_interval_product(view["product_id"])
    recipe = splitter.get_recipe(view["recipe_id"])
    return {
        "kind": "public_splitter_task",
        "task_id": view["task_id"],
        "interval_product": {
            "product_id": product["id"],
            "instrument_contract": product["instrument_contract"],
            "items": product["items"],
            "content_digest": product["content_digest"],
        },
        "recipe": {
            "recipe_id": recipe["recipe_id"],
            "channels": recipe["channels"],
            "bias": recipe["bias"],
            "cut_points": recipe["cut_points"],
            "decoder": recipe["decoder"],
            "content_digest": recipe["content_digest"],
        },
        "result": {
            key: value
            for key, value in view.items()
            if key not in ("content_digest", "created")
        },
        "content_digest": view["content_digest"],
    }


def verify_public_splitter(document: dict) -> list[str]:
    """Re-derive all checks from a standalone public splitter document.

    Returns a list of textual problems (empty means the document is
    internally consistent).  It does not need a database: it recomputes the
    content digests, re-derives every rounding interval from the original
    bits, nets the recipe channels, and recomputes the total range, the
    reachable bins and the endpoint hits.
    """
    problems: list[str] = []
    product = document["interval_product"]
    recipe = document["recipe"]
    result = document["result"]

    product_body = {
        "kind": "rounding_interval_product",
        "id": product["product_id"],
        "instrument_contract": product["instrument_contract"],
        "items": product["items"],
    }
    if digest(product_body) != product["content_digest"]:
        problems.append("interval product digest mismatch")
    recipe_body = {
        "kind": "splitter_recipe",
        "recipe_id": recipe["recipe_id"],
        "channels": recipe["channels"],
        "bias": recipe["bias"],
        "cut_points": recipe["cut_points"],
        "decoder": recipe["decoder"],
    }
    if digest(recipe_body) != recipe["content_digest"]:
        problems.append("recipe digest mismatch")
    if digest(result) != document["content_digest"]:
        problems.append("task result digest mismatch")
    if result["product_digest"] != product["content_digest"]:
        problems.append("task is bound to a different product digest")
    if result["recipe_digest"] != recipe["content_digest"]:
        problems.append("task is bound to a different recipe digest")

    items = {item["sample_id"]: item for item in product["items"]}
    for sample, item in items.items():
        if item["bits"] is None:
            if item["reason"] != "missing_observation":
                problems.append(f"{sample}: no original bits and not missing")
            continue
        derived = certify_bits(item["bits"]).canonical()
        for key in ("kind", "sign", "certifiable", "reason", "value", "interval", "endpoints"):
            if derived[key] != item[key]:
                problems.append(
                    f"{sample}: product {key} disagrees with the original bits"
                )

    net = net_coefficients(recipe["channels"])
    contributions = {entry["sample_id"]: entry for entry in result["contributions"]}
    if set(contributions) != set(net):
        problems.append("contributions do not match the recipe channels")
    ranges = []
    for sample, coefficient in net.items():
        entry = contributions.get(sample)
        if entry is None:
            continue
        if entry["net_coefficient"] != frac_str(coefficient):
            problems.append(
                f"{sample}: net coefficient {entry['net_coefficient']} "
                f"!= {frac_str(coefficient)}"
            )
        item = items.get(sample)
        if coefficient == 0:
            if entry["contribution"] is None or entry["contribution"]["kind"] != "exact_zero":
                problems.append(f"{sample}: zero net coefficient must cancel exactly")
            continue
        if item is None or not item["certifiable"]:
            problems.append(
                f"{sample}: non-certifiable channel delivered with a non-zero "
                "net coefficient"
            )
            continue
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
        if entry["contribution"] is None or entry["contribution"].get("range") != scaled.canonical():
            problems.append(f"{sample}: contribution range mismatch")
        ranges.append(scaled)

    total = sum_ranges(ranges, Fraction(recipe["bias"]))
    if total.canonical() != result["total"]:
        problems.append("total range mismatch")
    cuts = [Fraction(cut) for cut in recipe["cut_points"]]
    if reachable_bins(total, cuts) != result["reachable_bins"]:
        problems.append("reachable bins mismatch")
    expected_hits = [
        {"cut": frac_str(hit["cut"]), "end": hit["end"], "attainable": hit["attainable"]}
        for hit in endpoint_hits(total, cuts)
    ]
    if expected_hits != result["endpoint_hits"]:
        problems.append("endpoint hits mismatch")
    if result["auto_sortable"] != (len(result["reachable_bins"]) == 1):
        problems.append("auto_sortable flag inconsistent with reachable bins")
    if result["ambiguous"] != (len(result["reachable_bins"]) > 1):
        problems.append("ambiguity flag inconsistent with reachable bins")
    return problems
