from site_audit.gap_thresholds import (
    CLUSTER_SIMILARITY,
    COVERED,
    META_MAX_CHARS,
    OFF_INTENT,
    PARTIAL,
    PREVALENCE_CRITICAL,
    PREVALENCE_HIGH,
    TITLE_MAX_CHARS,
    band,
    similarity_band_prompt_text,
)


def test_canonical_threshold_values_are_pinned() -> None:
    # These literals define the verification loop's before/after comparability
    # and the agent prompt contract; changing any of them is a behavior change.
    assert COVERED == 0.78
    assert PARTIAL == 0.62
    assert OFF_INTENT == 0.52
    assert PREVALENCE_CRITICAL == 0.8
    assert PREVALENCE_HIGH == 0.6
    assert CLUSTER_SIMILARITY == 0.78
    assert TITLE_MAX_CHARS == 65
    assert META_MAX_CHARS == 165


def test_similarity_band_boundaries() -> None:
    assert band(0.78) == "covered"
    assert band(0.7799) == "partial"
    assert band(0.62) == "partial"
    assert band(0.6199) == "weak"


def test_prompt_and_title_limits_use_canonical_values() -> None:
    text = similarity_band_prompt_text()
    assert ">= 0.78 covered" in text
    assert "0.62-0.78 partial" in text
    assert "< 0.62 weak" in text
