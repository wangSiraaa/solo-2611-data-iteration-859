"""Run the rounding-interval and blending-splitter demonstration.

All arithmetic is exact (``Fraction``); no host floating point is involved.

Scenario
--------
* ``m1`` is metered as ``3f800000`` (1.0).  Its rounding interval is
  ``[1-2**-25, 1+2**-24]`` and both midpoints are attainable, so a recipe
  with the single channel ``m1 * 1`` and a cut at ``1+2**-24`` reaches two
  bins.
* Correcting ``m1`` to ``3f800001`` moves the whole interval to the right
  of that cut, and the cut itself is no longer attainable.
* Channels ``m1 * +1`` and ``m1 * -1`` carry the same unknown and cancel
  exactly: with the only cut at 0 the batch reaches only the right bin.
* ``n1`` starts as ``00000001`` (subnormal, not certifiable); a recipe
  giving it a non-zero net coefficient rejects the task.  After the
  correction to ``3f000000`` (0.5) the same task id is retried and
  published — one failed retry, both attempts audited.

Usage::

    python3 -m obsdelivery.splitter_demo [--db PATH] [--json]

The store is a real SQLite file (a fresh temporary one unless ``--db`` is
given); the demonstration closes and re-opens it to show that issued
interval products and splitter tasks survive corrections and restarts.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from .errors import ProductConflict, SplitterRejected, SplitterTaskConflict
from .intervals import IntervalService
from .public import public_splitter_task, verify_public_splitter
from .service import SemanticsService
from .splitter import SplitterService
from .storage import Store

#: 1 + 2**-24, the right rounding midpoint of 0x3f800000 (1.0).
RIGHT_MIDPOINT_1F = "16777217/16777216"


def _task_summary(task: dict) -> dict:
    return {
        "task_id": task["task_id"],
        "product_id": task["product_id"],
        "recipe_id": task["recipe_id"],
        "created": task["created"],
        "total": task["total"],
        "reachable_bins": task["reachable_bins"],
        "endpoint_hits": task["endpoint_hits"],
        "auto_sortable": task["auto_sortable"],
        "ambiguous": task["ambiguous"],
        "content_digest": task["content_digest"],
    }


def run(db_path: str | None = None) -> dict:
    if db_path is None:
        db_path = str(
            Path(tempfile.mkdtemp(prefix="obsdelivery-splitter-")) / "splitter.sqlite"
        )
    store = Store(db_path)
    semantics = SemanticsService(store)
    intervals = IntervalService(store, semantics)
    splitter = SplitterService(store, intervals)
    log: dict = {"db": db_path, "steps": []}

    def admit(sample_id, revision, bits):
        result = semantics.admit_observations(
            [{"sample_id": sample_id, "revision": revision, "bits": bits}]
        )
        log["steps"].append(
            {
                "op": "admit",
                "sample_id": sample_id,
                "revision": revision,
                "bits": bits,
                "outcome": result["outcome"],
            }
        )

    def publish_product(product_id, sample_ids):
        try:
            product = intervals.publish_interval_product(product_id, sample_ids)
            reopened = False
        except ProductConflict:
            # A re-run over the same store: the corrections are already in
            # effect, so the historical product is re-opened read-only
            # instead of being recomputed against current observations.
            product = intervals.get_interval_product(product_id)
            reopened = True
        log["steps"].append(
            {
                "op": "publish_interval_product",
                "product_id": product_id,
                "content_digest": product["content_digest"],
                "reopened_existing": reopened,
                "items": product["items"],
            }
        )
        return product

    def publish_recipe(recipe_id, channels, bias, cut_points):
        recipe = splitter.publish_recipe(
            recipe_id, channels, bias=bias, cut_points=cut_points
        )
        log["steps"].append(
            {
                "op": "publish_recipe",
                "recipe_id": recipe_id,
                "channels": recipe["channels"],
                "bias": recipe["bias"],
                "cut_points": recipe["cut_points"],
                "content_digest": recipe["content_digest"],
            }
        )

    def submit_task(task_id, product_id, recipe_id):
        task = splitter.submit(task_id, product_id, recipe_id)
        log["steps"].append({"op": "task", **_task_summary(task)})
        return task

    # 1. Initial meter readings.
    admit("m1", 1, "3f800000")  # 1.0
    admit("n1", 1, "00000001")  # smallest subnormal: not certifiable

    # 2. First interval product: [1-2**-25, 1+2**-24] for m1, both ends
    #    attainable; n1 keeps its bits and a non-certifiable reason.
    ip1 = publish_product("ip-v1", ["m1", "n1"])

    # 3. Immutable recipe versions.
    publish_recipe(
        "recipe-boundary-v1",
        [{"sample_id": "m1", "coefficient": "1"}],
        "0",
        [RIGHT_MIDPOINT_1F],
    )
    publish_recipe(
        "recipe-cancel-v1",
        [
            {"sample_id": "m1", "coefficient": "1"},
            {"sample_id": "m1", "coefficient": "-1"},
        ],
        "0",
        ["0"],
    )
    publish_recipe(
        "recipe-mix-v1",
        [
            {"sample_id": "m1", "coefficient": "1"},
            {"sample_id": "n1", "coefficient": "2"},
        ],
        "0",
        ["1"],
    )

    # 4. Boundary at the right midpoint: two reachable bins, endpoint hit.
    t_boundary = submit_task("T-boundary", "ip-v1", "recipe-boundary-v1")

    # 5. Same unknown on channels +1 and -1: exact cancellation, right bin only.
    submit_task("T-cancel", "ip-v1", "recipe-cancel-v1")

    # 6. n1 carries a non-zero net coefficient but is not certifiable: reject.
    try:
        splitter.submit("T-mix", "ip-v1", "recipe-mix-v1")
        log["steps"].append(
            {"op": "task_rejected", "task_id": "T-mix", "unexpected_success": True}
        )
    except SplitterRejected as exc:
        log["steps"].append(
            {
                "op": "task_rejected",
                "task_id": exc.task_id,
                "attempt_id": exc.attempt_id,
                "reasons": exc.reasons,
            }
        )
    except SplitterTaskConflict:
        # A re-run over the same store: the task id was already settled by
        # the successful retry below, so the failed request now conflicts
        # with the published identity instead of being audited again.
        log["steps"].append(
            {"op": "task_rejected", "task_id": "T-mix", "already_settled": True}
        )

    # 7. Correct n1 to 0.5, issue a new product, retry the same task id.
    admit("n1", 2, "3f000000")
    publish_product("ip-v2", ["m1", "n1"])
    submit_task("T-mix", "ip-v2", "recipe-mix-v1")
    log["steps"].append(
        {
            "op": "task_attempts",
            "task_id": "T-mix",
            "attempts": [
                {"attempt_id": a["attempt_id"], "status": a["status"]}
                for a in splitter.attempts("T-mix")
            ],
        }
    )

    # 8. Correct m1 to 3f800001: the old boundary is no longer attainable and
    #    the whole new interval lies to its right.
    admit("m1", 2, "3f800001")
    publish_product("ip-v3", ["m1"])
    submit_task("T-boundary-after", "ip-v3", "recipe-boundary-v1")

    # 9. Identity rules: replay returns the stored task; a different request
    #    under the same id is refused.
    replay = splitter.submit("T-boundary", "ip-v1", "recipe-boundary-v1")
    log["steps"].append(
        {
            "op": "task_replay",
            "task_id": "T-boundary",
            "created": replay["created"],
            "same_content_digest": replay["content_digest"] == t_boundary["content_digest"],
        }
    )
    try:
        splitter.submit("T-boundary", "ip-v1", "recipe-cancel-v1")
        conflict = False
    except SplitterTaskConflict:
        conflict = True
    log["steps"].append(
        {"op": "task_id_conflict", "task_id": "T-boundary", "refused": conflict}
    )

    # 10. Close and re-open the store: issued intervals and tasks are
    #     unchanged by the later corrections, and the public document of the
    #     boundary task still verifies standalone.
    store.close()
    store2 = Store(db_path)
    intervals2 = IntervalService(store2, SemanticsService(store2))
    splitter2 = SplitterService(store2, intervals2)
    ip1_reopened = intervals2.get_interval_product("ip-v1")
    t_reopened = splitter2.get_task("T-boundary")
    document = public_splitter_task(splitter2, "T-boundary")
    reader_problems = verify_public_splitter(document)
    log["reopen"] = {
        "interval_product": {
            "product_id": "ip-v1",
            "content_digest": ip1_reopened["content_digest"],
            "unchanged_by_corrections": ip1_reopened["content_digest"]
            == ip1["content_digest"],
            "items": ip1_reopened["items"],
        },
        "task": {
            "task_id": "T-boundary",
            "content_digest": t_reopened["content_digest"],
            "unchanged": t_reopened["content_digest"] == t_boundary["content_digest"],
        },
        "public_document": {
            "endpoint_hits": document["result"]["endpoint_hits"],
            "reachable_bins": document["result"]["reachable_bins"],
            "ambiguous": document["result"]["ambiguous"],
            "auto_sortable": document["result"]["auto_sortable"],
            "reader_problems": reader_problems,
        },
    }
    store2.close()
    return log


def _bracket(interval: dict) -> str:
    left = "[" if interval["low_included"] else "("
    right = "]" if interval["high_included"] else ")"
    return f"{left}{interval['low']}, {interval['high']}{right}"


def _print_item(item: dict) -> None:
    if item["certifiable"]:
        print(f"  {item['sample_id']}: bits={item['bits']} value={item['value']}")
        print(f"    interval {_bracket(item['interval'])}")
        for end in ("lower", "upper"):
            endpoint = item["endpoints"][end]
            print(
                f"    {end} midpoint {endpoint['midpoint']} "
                f"from {endpoint['between'][0]}..{endpoint['between'][1]} "
                f"attainable={endpoint['included']}"
            )
    else:
        print(
            f"  {item['sample_id']}: bits={item['bits']} kind={item['kind']} "
            f"NOT certifiable: {item['reason']} ({item['reason_detail']})"
        )


def _print_text(log: dict) -> None:
    print(f"store: {log['db']}")
    for step in log["steps"]:
        op = step["op"]
        if op == "admit":
            print(
                f"\n[admit] {step['sample_id']} r{step['revision']} "
                f"bits={step['bits']} -> {step['outcome']}"
            )
        elif op == "publish_interval_product":
            tag = "reopen interval product" if step["reopened_existing"] else "interval product"
            print(f"\n[{tag} {step['product_id']}] digest={step['content_digest'][:12]}")
            for item in step["items"]:
                _print_item(item)
        elif op == "publish_recipe":
            print(f"\n[recipe {step['recipe_id']}] digest={step['content_digest'][:12]}")
            for channel in step["channels"]:
                print(f"  channel {channel['sample_id']} x {channel['coefficient']}")
            print(f"  bias={step['bias']} cut_points={step['cut_points']}")
        elif op == "task":
            print(
                f"\n[task {step['task_id']}] product={step['product_id']} "
                f"recipe={step['recipe_id']} created={step['created']}"
            )
            print(f"  total range {_bracket(step['total'])}")
            print(
                f"  reachable_bins={step['reachable_bins']} "
                f"auto_sortable={step['auto_sortable']} ambiguous={step['ambiguous']}"
            )
            for hit in step["endpoint_hits"]:
                print(
                    f"  endpoint hit: cut={hit['cut']} end={hit['end']} "
                    f"attainable={hit['attainable']}"
                )
        elif op == "task_rejected":
            if step.get("already_settled"):
                print(
                    f"\n[task {step['task_id']}] failed request conflicts with the "
                    "already-settled task (re-run over the same store)"
                )
                continue
            print(f"\n[task {step['task_id']}] REJECTED (attempt {step.get('attempt_id')})")
            for reason in step.get("reasons", []):
                print(
                    f"  {reason['sample_id']}: {reason['reason']} "
                    f"net={reason['net_coefficient']} bits={reason['bits']} "
                    f"({reason['certifiable_reason']})"
                )
        elif op == "task_attempts":
            attempts = ", ".join(
                f"#{attempt['attempt_id']} {attempt['status']}" for attempt in step["attempts"]
            )
            print(f"\n[attempts {step['task_id']}] {attempts}")
        elif op == "task_replay":
            print(
                f"\n[task replay {step['task_id']}] created={step['created']} "
                f"same_digest={step['same_content_digest']}"
            )
        elif op == "task_id_conflict":
            print(f"\n[task id reuse {step['task_id']}] refused={step['refused']}")
    reopened = log["reopen"]
    product = reopened["interval_product"]
    print(
        f"\n[reopen {product['product_id']}] digest={product['content_digest'][:12]} "
        f"unchanged_by_corrections={product['unchanged_by_corrections']}"
    )
    for item in product["items"]:
        _print_item(item)
    task = reopened["task"]
    print(
        f"[reopen {task['task_id']}] digest={task['content_digest'][:12]} "
        f"unchanged={task['unchanged']}"
    )
    public = reopened["public_document"]
    print(
        f"[public {task['task_id']}] reachable={public['reachable_bins']} "
        f"ambiguous={public['ambiguous']} auto_sortable={public['auto_sortable']}"
    )
    for hit in public["endpoint_hits"]:
        print(
            f"  endpoint hit: cut={hit['cut']} end={hit['end']} "
            f"attainable={hit['attainable']}"
        )
    print(f"  independent reader problems: {public['reader_problems'] or 'none'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", help="SQLite path (default: a fresh temporary file)")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    args = parser.parse_args(argv)
    result = run(args.db)
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    _print_text(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
