"""Exact tests for rounding-interval products and blending-splitter tasks."""

import json
from fractions import Fraction

import pytest

from obsdelivery.errors import (
    IntegrityError,
    ProductConflict,
    RecipeConflict,
    SplitterRejected,
    SplitterTaskConflict,
    SplitterTaskNotFound,
    ValidationError,
)
from obsdelivery.intervals import IntervalService
from obsdelivery.public import public_splitter_task, verify_public_splitter
from obsdelivery.rounding import (
    REASON_LOWER_NEIGHBOUR,
    REASON_NONFINITE,
    REASON_SUBNORMAL,
    REASON_UPPER_NEIGHBOUR,
    REASON_ZERO,
    certify_bits,
)
from obsdelivery.service import SemanticsService
from obsdelivery.splitter import SplitterService
from obsdelivery.splitter_algo import Range, endpoint_hits, reachable_bins
from obsdelivery.storage import Store

LOW_1 = Fraction(33554431, 33554432)  # 1 - 2**-25
HIGH_1 = Fraction(16777217, 16777216)  # 1 + 2**-24
HIGH_1B = Fraction(16777219, 16777216)  # 1 + 3*2**-24


@pytest.fixture()
def env(tmp_path):
    store = Store(str(tmp_path / "rounding.sqlite"))
    semantics = SemanticsService(store)
    intervals = IntervalService(store, semantics)
    splitter = SplitterService(store, intervals)
    yield store, semantics, intervals, splitter
    store.close()


def admit(semantics, sample_id, revision, bits):
    return semantics.admit_observations(
        [{"sample_id": sample_id, "revision": revision, "bits": bits}]
    )


# ---------------------------------------------------------------------------
# pure rounding intervals
# ---------------------------------------------------------------------------


def test_interval_midpoints_1f():
    interval = certify_bits("3f800000")
    assert interval.certifiable and interval.reason is None
    assert interval.value == 1
    assert interval.low == LOW_1 and interval.high == HIGH_1
    # Both midpoints are attainable (1.0 has an even mantissa).
    assert interval.low_included and interval.high_included
    # The left neighbour crosses an exponent boundary (exponent 126).
    assert interval.lower_neighbour_bits == "3f7fffff"
    assert interval.upper_neighbour_bits == "3f800001"


def test_interval_odd_mantissa_is_open():
    interval = certify_bits("3f800001")
    assert interval.certifiable
    assert interval.low == HIGH_1 and interval.high == HIGH_1B
    assert not interval.low_included and not interval.high_included


def test_interval_negative_mirroring():
    interval = certify_bits("bf800000")  # -1.0
    assert interval.certifiable
    assert interval.low == -HIGH_1 and interval.high == -LOW_1
    assert interval.low_included and interval.high_included
    assert interval.lower_neighbour_bits == "bf800001"
    assert interval.upper_neighbour_bits == "bf7fffff"


def test_interval_crosses_exponent_boundary_both_ways():
    # 0x3f7fffff is the largest exponent-126 value; its right neighbour is 1.0.
    interval = certify_bits("3f7fffff")
    assert interval.certifiable
    assert interval.high == Fraction(16777215, 16777216) + Fraction(1, 33554432)
    assert interval.high == 1 - Fraction(1, 33554432)
    # Odd mantissa: neither midpoint attainable.
    assert not interval.low_included and not interval.high_included


def test_non_certifiable_patterns_keep_bits():
    cases = {
        "00000000": REASON_ZERO,  # +0
        "80000000": REASON_ZERO,  # -0
        "00000001": REASON_SUBNORMAL,
        "007fffff": REASON_SUBNORMAL,  # largest subnormal
        "00800000": REASON_LOWER_NEIGHBOUR,  # smallest normal
        "80800000": REASON_LOWER_NEIGHBOUR,  # negative smallest normal
        "7f7fffff": REASON_UPPER_NEIGHBOUR,  # max finite
        "ff7fffff": REASON_UPPER_NEIGHBOUR,  # negative max finite
        "7f800000": REASON_NONFINITE,  # +inf
        "ff800000": REASON_NONFINITE,  # -inf
        "7fc00001": REASON_NONFINITE,  # NaN
    }
    for bits, reason in cases.items():
        interval = certify_bits(bits)
        assert not interval.certifiable
        assert interval.reason == reason
        assert interval.bits == bits
        assert interval.low is None and interval.high is None


def test_certifiable_edge_normals():
    # One step above the smallest normal and one below the largest finite.
    assert certify_bits("00800001").certifiable
    assert certify_bits("7f7ffffe").certifiable


def test_reachable_bins_and_endpoint_hits():
    cuts = [Fraction(0)]
    # A point range exactly on the cut reaches only the right bin.
    point = Range(Fraction(0), True, Fraction(0), True)
    assert reachable_bins(point, cuts) == [1]
    assert endpoint_hits(point, cuts) == [
        {"cut": 0, "end": "low", "attainable": True},
        {"cut": 0, "end": "high", "attainable": True},
    ]
    # A closed interval ending on the cut reaches both bins.
    assert reachable_bins(Range(Fraction(-1), True, Fraction(0), True), cuts) == [0, 1]
    # An open end at the cut leaves the cut to the right bin only.
    assert reachable_bins(Range(Fraction(-1), True, Fraction(0), False), cuts) == [0]
    # A range strictly right of the cut reaches only the right bin.
    assert reachable_bins(Range(Fraction(1), False, Fraction(2), False), cuts) == [1]
    # No cuts at all: a single always-reachable bin.
    assert reachable_bins(Range(Fraction(5), True, Fraction(6), True), []) == [0]


# ---------------------------------------------------------------------------
# interval products
# ---------------------------------------------------------------------------


def test_interval_product_immutable_and_sourced(env):
    _, semantics, intervals, _ = env
    admit(semantics, "m1", 1, "3f800000")
    product = intervals.publish_interval_product("ip-v1", ["m1"])
    item = product["items"][0]
    assert item["interval"] == {
        "low": "33554431/33554432",
        "low_included": True,
        "high": "16777217/16777216",
        "high_included": True,
    }
    # Endpoint sources are queryable from the product.
    assert item["endpoints"]["lower"]["between"] == ["3f7fffff", "3f800000"]
    assert item["endpoints"]["upper"]["between"] == ["3f800000", "3f800001"]
    assert product["instrument_contract"]["id"] == "rn-te-normal-neighbours-v1"

    # A correction issues a NEW product; the issued one is never rewritten.
    admit(semantics, "m1", 2, "3f800001")
    product2 = intervals.publish_interval_product("ip-v2", ["m1"])
    assert product2["items"][0]["interval"]["low_included"] is False
    reopened = intervals.get_interval_product("ip-v1")
    assert reopened["content_digest"] == product["content_digest"]
    assert reopened["items"][0]["bits"] == "3f800000"
    assert reopened["items"][0]["interval"]["high"] == "16777217/16777216"

    # Same id, different content -> conflict.
    with pytest.raises(ProductConflict):
        intervals.publish_interval_product("ip-v1", ["m1", "n1"])
    # Identical replay returns the stored row.
    again = intervals.publish_interval_product("ip-v2", ["m1"])
    assert again["created"] is False
    assert again["content_digest"] == product2["content_digest"]


def test_interval_product_keeps_uncertifiable_originals(env):
    _, semantics, intervals, _ = env
    admit(semantics, "n1", 1, "00000001")
    product = intervals.publish_interval_product("ip", ["n1", "ghost"])
    items = {item["sample_id"]: item for item in product["items"]}
    assert items["n1"]["certifiable"] is False
    assert items["n1"]["reason"] == "subnormal_not_normal"
    assert items["n1"]["bits"] == "00000001"  # original preserved
    assert items["ghost"]["reason"] == "missing_observation"
    assert items["ghost"]["bits"] is None


# ---------------------------------------------------------------------------
# splitter tasks
# ---------------------------------------------------------------------------


def test_boundary_cut_gives_two_reachable_bins(env):
    _, semantics, intervals, splitter = env
    admit(semantics, "m1", 1, "3f800000")
    intervals.publish_interval_product("ip-v1", ["m1"])
    splitter.publish_recipe(
        "r-boundary",
        [{"sample_id": "m1", "coefficient": "1"}],
        bias="0",
        cut_points=["16777217/16777216"],
    )
    task = splitter.submit("T", "ip-v1", "r-boundary")
    assert task["total"] == {
        "low": "33554431/33554432",
        "low_included": True,
        "high": "16777217/16777216",
        "high_included": True,
    }
    # The cut sits exactly on the attainable right midpoint: two bins.
    assert task["reachable_bins"] == [0, 1]
    assert task["ambiguous"] and not task["auto_sortable"]
    assert task["endpoint_hits"] == [
        {"cut": "16777217/16777216", "end": "high", "attainable": True}
    ]
    assert [bin["reachable"] for bin in task["bins"]] == [True, True]
    assert task["bins"][1]["lower_included"] is True  # cut belongs to the right bin

    # Replay returns the stored task; a different recipe under the id conflicts.
    again = splitter.submit("T", "ip-v1", "r-boundary")
    assert again["created"] is False
    assert again["content_digest"] == task["content_digest"]
    splitter.publish_recipe(
        "r-other",
        [{"sample_id": "m1", "coefficient": "2"}],
        bias="0",
        cut_points=["16777217/16777216"],
    )
    with pytest.raises(SplitterTaskConflict):
        splitter.submit("T", "ip-v1", "r-other")


def test_corrected_pattern_moves_whole_interval_right(env):
    _, semantics, intervals, splitter = env
    admit(semantics, "m1", 1, "3f800001")
    intervals.publish_interval_product("ip", ["m1"])
    splitter.publish_recipe(
        "r",
        [{"sample_id": "m1", "coefficient": "1"}],
        bias="0",
        cut_points=["16777217/16777216"],
    )
    task = splitter.submit("T", "ip", "r")
    # The old boundary is the left endpoint but is NOT attainable, and the
    # whole interval lies to its right: only the right bin is reachable.
    assert task["total"]["low"] == "16777217/16777216"
    assert task["total"]["low_included"] is False
    assert task["reachable_bins"] == [1]
    assert task["auto_sortable"] and not task["ambiguous"]
    assert task["endpoint_hits"] == [
        {"cut": "16777217/16777216", "end": "low", "attainable": False}
    ]


def test_same_sample_plus_minus_cancels_exactly(env):
    _, semantics, intervals, splitter = env
    admit(semantics, "m1", 1, "3f800000")
    intervals.publish_interval_product("ip", ["m1"])
    splitter.publish_recipe(
        "r",
        [
            {"sample_id": "m1", "coefficient": "1"},
            {"sample_id": "m1", "coefficient": "-1"},
        ],
        bias="0",
        cut_points=["0"],
    )
    task = splitter.submit("T", "ip", "r")
    assert task["total"] == {
        "low": "0",
        "low_included": True,
        "high": "0",
        "high_included": True,
    }
    # The only cut is 0 and the total is exactly 0: only the right bin.
    assert task["reachable_bins"] == [1]
    assert task["auto_sortable"]
    entry = task["contributions"][0]
    assert entry["channels"] == 2
    assert entry["net_coefficient"] == "0"
    assert entry["contribution"]["kind"] == "exact_zero"


def test_different_samples_vary_independently(env):
    _, semantics, intervals, splitter = env
    admit(semantics, "m1", 1, "3f800000")
    admit(semantics, "m2", 1, "3f800000")
    intervals.publish_interval_product("ip", ["m1", "m2"])
    splitter.publish_recipe(
        "r",
        [
            {"sample_id": "m1", "coefficient": "1"},
            {"sample_id": "m2", "coefficient": "-1"},
        ],
        bias="0",
        cut_points=["0"],
    )
    task = splitter.submit("T", "ip", "r")
    # m1 - m2 spans a real interval around 0, reaching both bins.
    assert task["total"]["low"] == "-3/33554432"
    assert task["total"]["high"] == "3/33554432"
    assert task["reachable_bins"] == [0, 1]


def test_uncertifiable_nonzero_net_rejected_then_retried(env):
    _, semantics, intervals, splitter = env
    admit(semantics, "m1", 1, "3f800000")
    admit(semantics, "n1", 1, "00000001")  # subnormal: not certifiable
    intervals.publish_interval_product("ip-v1", ["m1", "n1"])
    splitter.publish_recipe(
        "r-mix",
        [
            {"sample_id": "m1", "coefficient": "1"},
            {"sample_id": "n1", "coefficient": "2"},
        ],
        bias="0",
        cut_points=["1"],
    )
    with pytest.raises(SplitterRejected) as exc:
        splitter.submit("T-mix", "ip-v1", "r-mix")
    reason = exc.value.reasons[0]
    assert reason["reason"] == "uncertifiable_net_contribution"
    assert reason["sample_id"] == "n1"
    assert reason["net_coefficient"] == "2"
    assert reason["bits"] == "00000001"
    assert reason["certifiable_reason"] == "subnormal_not_normal"
    # No task row, only an audited attempt; the id stays usable.
    with pytest.raises(SplitterTaskNotFound):
        splitter.get_task("T-mix")
    assert [a["status"] for a in splitter.attempts("T-mix")] == ["rejected"]

    # Correct the observation, issue a new product, retry the same task id.
    admit(semantics, "n1", 2, "3f000000")  # 0.5, certifiable
    intervals.publish_interval_product("ip-v2", ["m1", "n1"])
    task = splitter.submit("T-mix", "ip-v2", "r-mix")
    assert task["created"] is True
    # total = 1*m1 + 2*n1 in [2-2**-24, 2+2**-23]: right of the cut at 1.
    assert task["total"] == {
        "low": "33554431/16777216",
        "low_included": True,
        "high": "16777217/8388608",
        "high_included": True,
    }
    assert task["reachable_bins"] == [1]
    assert task["auto_sortable"]
    assert [a["status"] for a in splitter.attempts("T-mix")] == ["rejected", "published"]
    # The failed attempt did not consume the successful identity.
    again = splitter.submit("T-mix", "ip-v2", "r-mix")
    assert again["created"] is False
    assert again["content_digest"] == task["content_digest"]


def test_zero_net_needs_no_certification(env):
    _, semantics, intervals, splitter = env
    admit(semantics, "n1", 1, "00000001")  # subnormal: not certifiable
    intervals.publish_interval_product("ip", ["n1"])
    splitter.publish_recipe(
        "r",
        [
            {"sample_id": "n1", "coefficient": "5"},
            {"sample_id": "n1", "coefficient": "-5"},
        ],
        bias="0",
        cut_points=["0"],
    )
    task = splitter.submit("T", "ip", "r")
    assert task["reachable_bins"] == [1]
    entry = task["contributions"][0]
    assert entry["certifiable"] is False
    assert entry["contribution"]["kind"] == "exact_zero"


def test_recipe_validation_and_immutability(env):
    _, _, _, splitter = env
    for bad_channels, bad_cuts in [
        ([{"sample_id": "m1"}], []),  # missing coefficient
        ([{"sample_id": "m1", "coefficient": 0.5}], []),  # float coefficient
        ([{"sample_id": "m1", "coefficient": "1"}], ["1", "1"]),  # duplicate cuts
        ([{"sample_id": "m1", "coefficient": "1"}], ["1/0"]),  # bad rational
    ]:
        with pytest.raises(ValidationError):
            splitter.publish_recipe("r-bad", bad_channels, bias="0", cut_points=bad_cuts)
    recipe = splitter.publish_recipe(
        "r", [{"sample_id": "m1", "coefficient": "1"}], bias="0", cut_points=["0"]
    )
    assert recipe["created"] is True
    again = splitter.publish_recipe(
        "r", [{"sample_id": "m1", "coefficient": "1"}], bias="0", cut_points=["0"]
    )
    assert again["created"] is False
    with pytest.raises(RecipeConflict):
        splitter.publish_recipe(
            "r", [{"sample_id": "m1", "coefficient": "2"}], bias="0", cut_points=["0"]
        )


def test_recipe_naming_unknown_sample_rejected(env):
    _, semantics, intervals, splitter = env
    admit(semantics, "m1", 1, "3f800000")
    intervals.publish_interval_product("ip", ["m1"])
    splitter.publish_recipe(
        "r", [{"sample_id": "ghost", "coefficient": "1"}], bias="0", cut_points=["0"]
    )
    with pytest.raises(ValidationError):
        splitter.submit("T", "ip", "r")


def test_task_integrity_detects_tampering(env):
    store, semantics, intervals, splitter = env
    admit(semantics, "m1", 1, "3f800000")
    intervals.publish_interval_product("ip", ["m1"])
    splitter.publish_recipe(
        "r", [{"sample_id": "m1", "coefficient": "1"}], bias="0", cut_points=["1"]
    )
    splitter.submit("T", "ip", "r")
    row = store.conn.execute(
        "SELECT result_json FROM splitter_tasks WHERE task_id='T'"
    ).fetchone()
    tampered = json.loads(row["result_json"])
    tampered["reachable_bins"] = [0]
    store.conn.execute(
        "UPDATE splitter_tasks SET result_json=? WHERE task_id='T'",
        (json.dumps(tampered),),
    )
    with pytest.raises(IntegrityError):
        splitter.get_task("T")


# ---------------------------------------------------------------------------
# public document and reopening
# ---------------------------------------------------------------------------


def test_public_document_verifies_and_detects_tampering(env):
    _, semantics, intervals, splitter = env
    admit(semantics, "m1", 1, "3f800000")
    intervals.publish_interval_product("ip", ["m1"])
    splitter.publish_recipe(
        "r",
        [{"sample_id": "m1", "coefficient": "1"}],
        bias="0",
        cut_points=["16777217/16777216"],
    )
    splitter.submit("T", "ip", "r")
    document = public_splitter_task(splitter, "T")
    # Endpoint hits and ambiguity are visible from the public delivery.
    assert document["result"]["endpoint_hits"] == [
        {"cut": "16777217/16777216", "end": "high", "attainable": True}
    ]
    assert document["result"]["ambiguous"] is True
    assert verify_public_splitter(document) == []

    def tampered(**changes):
        document_copy = json.loads(json.dumps(document))
        for key, value in changes.items():
            document_copy[key] = value
        return document_copy

    bad = tampered()
    bad["result"]["reachable_bins"] = [1]
    assert verify_public_splitter(bad)
    bad2 = tampered()
    bad2["result"]["endpoint_hits"][0]["attainable"] = False
    assert verify_public_splitter(bad2)
    bad3 = tampered()
    bad3["interval_product"]["items"][0]["interval"]["high_included"] = False
    assert verify_public_splitter(bad3)
    bad4 = tampered()
    bad4["interval_product"]["items"][0]["bits"] = "3f800001"
    assert verify_public_splitter(bad4)


def test_reopen_store_reads_products_and_tasks(tmp_path):
    path = str(tmp_path / "reopen.sqlite")
    store = Store(path)
    semantics = SemanticsService(store)
    intervals = IntervalService(store, semantics)
    splitter = SplitterService(store, intervals)
    admit(semantics, "m1", 1, "3f800000")
    product = intervals.publish_interval_product("ip", ["m1"])
    splitter.publish_recipe(
        "r",
        [{"sample_id": "m1", "coefficient": "1"}],
        bias="0",
        cut_points=["16777217/16777216"],
    )
    task = splitter.submit("T", "ip", "r")
    store.close()

    store2 = Store(path)
    intervals2 = IntervalService(store2, SemanticsService(store2))
    splitter2 = SplitterService(store2, intervals2)
    product2 = intervals2.get_interval_product("ip")
    assert product2["content_digest"] == product["content_digest"]
    assert product2["items"][0]["endpoints"]["lower"]["between"] == [
        "3f7fffff",
        "3f800000",
    ]
    task2 = splitter2.get_task("T")
    assert task2["content_digest"] == task["content_digest"]
    assert task2["reachable_bins"] == [0, 1]
    store2.close()
