"""Unit tests for VAD reject-routing classification (Change C).

Exercises ONLY the pure `is_rejected` classification helper used by
export_vad to route dropped segments out of the delivery JSON. No Azure /
Label Studio access — we feed minimal fake Label Studio `result` lists.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from export_handler import is_rejected


def _result(from_name, value):
    return {"from_name": from_name, "value": value}


def test_rejected_task_is_classified_as_rejected():
    # A task carrying the "reject" Choices control -> rejected with reason.
    result = [
        _result("speaker", {"choices": ["Customer"]}),
        _result("transcript", {"text": ["kuch samajh nahi aaya"]}),
        _result("reject", {"choices": ["Drop - unintelligible"]}),
    ]
    assert is_rejected(result) == "Drop - unintelligible"


def test_non_rejected_task_is_not_rejected():
    # A normal accepted task -> not rejected.
    result = [
        _result("speaker", {"choices": ["Customer"]}),
        _result("transcript", {"text": ["namaste"]}),
    ]
    assert is_rejected(result) is None


def test_reject_wins_even_with_transcript_present():
    # Reject must drop the segment even when a transcript exists (routing rule).
    result = [
        _result("transcript", {"text": ["has text but still dropped"]}),
        _result("reject", {"choices": ["Drop - noise"]}),
    ]
    assert is_rejected(result) == "Drop - noise"


def test_empty_reject_choices_is_not_rejected():
    # Reject control present but with no choice selected -> not rejected.
    result = [
        _result("transcript", {"text": ["ok"]}),
        _result("reject", {"choices": []}),
    ]
    assert is_rejected(result) is None


def test_empty_and_none_result_is_not_rejected():
    assert is_rejected([]) is None
    assert is_rejected(None) is None


def test_routing_excludes_rejected_from_delivery_list():
    """End-to-end shape of the parse-loop routing: rejected go to a separate
    list, accepted stay in the delivery list. Mirrors export_vad's decision."""
    tasks = [
        {"segment_id": 1, "result": [
            _result("speaker", {"choices": ["Customer"]}),
            _result("transcript", {"text": ["accepted one"]}),
        ]},
        {"segment_id": 2, "result": [
            _result("transcript", {"text": ["rejected one"]}),
            _result("reject", {"choices": ["Drop - unintelligible"]}),
        ]},
    ]

    segments = []
    rejected_segments = []
    for t in tasks:
        reject = is_rejected(t["result"])
        if reject:
            rejected_segments.append({"segment_id": t["segment_id"], "reason": reject})
            continue
        segments.append({"segment_id": t["segment_id"]})

    delivered_ids = [s["segment_id"] for s in segments]
    rejected_ids = [s["segment_id"] for s in rejected_segments]

    assert delivered_ids == [1]           # accepted only
    assert rejected_ids == [2]            # rejected routed out
    assert 2 not in delivered_ids         # rejected excluded from delivery
    assert rejected_segments[0]["reason"] == "Drop - unintelligible"


if __name__ == "__main__":
    import traceback

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
