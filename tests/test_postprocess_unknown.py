"""
Test for VAD_EMIT_UNKNOWN gating in vad_processor.postprocess().

Client rule change: <UNKNOWN> must NOT be auto-emitted by default. Annotators
drop unintelligible segments instead. The old behaviour is preserved only when
the env flag VAD_EMIT_UNKNOWN="1" is set.

Runnable with either:
    python -m pytest tests/test_postprocess_unknown.py
    python tests/test_postprocess_unknown.py
"""

import os
import sys

# Make the parent dir importable when run directly (python tests/...).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from vad_processor import postprocess
except Exception as exc:  # heavy/optional deps missing → skip rather than error
    postprocess = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


# Low-confidence inputs: avg_logprob well below threshold.
LOW_CONF_TEXT = "कुछ"
LOW_CONF_LOGPROB = -5.0
THRESHOLD = -2.0


def test_unknown_not_emitted_by_default():
    """With VAD_EMIT_UNKNOWN unset/"0", low-confidence text flows through as
    normal processed text — never the literal <UNKNOWN>."""
    os.environ.pop("VAD_EMIT_UNKNOWN", None)  # ensure unset (default OFF)
    result = postprocess(LOW_CONF_TEXT, LOW_CONF_LOGPROB, THRESHOLD)
    assert result != "<UNKNOWN>", f"expected processed text, got {result!r}"
    assert result, "expected non-empty processed text"

    # Explicit "0" behaves the same as unset.
    os.environ["VAD_EMIT_UNKNOWN"] = "0"
    try:
        result0 = postprocess(LOW_CONF_TEXT, LOW_CONF_LOGPROB, THRESHOLD)
        assert result0 != "<UNKNOWN>", f"expected processed text, got {result0!r}"
    finally:
        os.environ.pop("VAD_EMIT_UNKNOWN", None)


def test_unknown_emitted_when_flag_on():
    """With VAD_EMIT_UNKNOWN="1", the legacy behaviour is preserved: a
    low-confidence segment returns the literal <UNKNOWN> tag."""
    os.environ["VAD_EMIT_UNKNOWN"] = "1"
    try:
        result = postprocess(LOW_CONF_TEXT, LOW_CONF_LOGPROB, THRESHOLD)
        assert result == "<UNKNOWN>", f"expected <UNKNOWN>, got {result!r}"
    finally:
        os.environ.pop("VAD_EMIT_UNKNOWN", None)


def _run():
    if postprocess is None:
        print(f"SKIP: could not import postprocess from vad_processor: {_IMPORT_ERROR}")
        return 0  # treat missing heavy deps as a skip, not a failure
    test_unknown_not_emitted_by_default()
    test_unknown_emitted_when_flag_on()
    print("OK: both assertions passed")
    return 0


if __name__ == "__main__":
    sys.exit(_run())
