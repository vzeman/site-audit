from site_audit.ctr_curve import estimate_clicks_gain, expected_ctr


def test_expected_ctr_is_monotonically_non_increasing() -> None:
    values = [expected_ctr(pos) for pos in range(1, 101)]

    assert all(left >= right for left, right in zip(values, values[1:]))


def test_expected_ctr_clamps_bounds() -> None:
    assert expected_ctr(-10) == expected_ctr(1)
    assert expected_ctr(250) == expected_ctr(100)


def test_expected_ctr_interpolates_between_integer_positions() -> None:
    midpoint = expected_ctr(1.5)

    assert expected_ctr(1) > midpoint > expected_ctr(2)
    assert midpoint == (expected_ctr(1) + expected_ctr(2)) / 2


def test_estimate_clicks_gain_zero_when_target_is_not_better() -> None:
    assert estimate_clicks_gain(1000, 5, 5) == 0
    assert estimate_clicks_gain(1000, 5, 6) == 0
    assert estimate_clicks_gain(1000, 5, 3) > 0
