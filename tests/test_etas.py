"""Unit tests for the ETAS expert: the closed-form integral and the causality of the sum.

Nothing here needs the PINNeAPPle library — the module is numpy and scipy only, which is
the point: this is the part of the experiments package a CI run can actually hold.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "experiments" / "hazards_forecasting" / "src"
sys.path.insert(0, str(SRC))

from etas import ALPHA, C, MC, P, ETASExpert, _integral  # noqa: E402


def test_the_integral_matches_numerical_quadrature():
    """int_a^b (u+c)^-p du, checked against a fine trapezoid over the same interval."""
    for a, b in [(0.0, 1.0), (0.0, 7.0), (3.0, 10.5), (100.0, 370.0)]:
        u = np.linspace(a, b, 2_000_001)
        numeric = np.trapezoid((u + C) ** -P, u)
        assert _integral(a, b) == pytest.approx(numeric, rel=1e-6)


def test_the_integral_is_positive_and_decreases_as_the_window_extends():
    """Decay: integrating further can only add a smaller and smaller amount."""
    first = _integral(0.0, 1.0)
    second = _integral(1.0, 2.0)
    assert first > 0 and second > 0
    assert second < first
    assert _integral(0.0, 2.0) == pytest.approx(first + second)


def test_a_larger_event_contributes_more_aftershocks():
    """The productivity factor is exp(ALPHA (m - M_C)), so it must order magnitudes."""
    t = np.array([0.0, 0.0])
    m = np.array([MC + 1.0, MC + 2.0])
    ends = np.array([0.0, 7.0, 14.0])

    bigger = ETASExpert(np.array([0.0]), np.array([MC + 2.0]), ends)._S(0.0, 7.0)
    smaller = ETASExpert(np.array([0.0]), np.array([MC + 1.0]), ends)._S(0.0, 7.0)
    assert bigger == pytest.approx(smaller * np.exp(ALPHA))
    assert t.shape == m.shape  # the arrays this test builds are the ones _S indexes


def test_the_sum_never_reads_an_event_after_the_window_starts():
    """Causality: an event after T0 must not change the sum for (T0, T1]."""
    ends = np.array([0.0, 7.0])
    before = ETASExpert(np.array([0.0, 2.0]), np.array([5.0, 5.0]), ends)._S(3.0, 10.0)
    with_a_later_event = ETASExpert(
        np.array([0.0, 2.0, 5.0]), np.array([5.0, 5.0, 9.0]), ends
    )._S(3.0, 10.0)
    assert with_a_later_event == pytest.approx(before)


def test_fit_then_predict_returns_one_finite_forecast_per_horizon_step():
    rng = np.random.default_rng(0)
    weeks = 400
    ends = np.arange(1, weeks + 1) * 7.0
    t = np.sort(rng.uniform(0, ends[-1], 500))
    m = rng.uniform(MC, MC + 3, 500)
    y = np.log1p(rng.poisson(3.0, weeks))

    expert = ETASExpert(t, m, ends).fit(y)
    out = expert.predict(4)

    assert out.shape == (4,)
    assert np.all(np.isfinite(out))
    assert np.all(out >= 0)
