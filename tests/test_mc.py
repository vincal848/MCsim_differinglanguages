"""Monte Carlo pricer in mc.py against Black-Scholes, against itself under
reparametrization, and against the invariants that must hold regardless of the
random draws: put-call parity pathwise, E[S_T] at the risk-free forward, SE
shrinking like 1/sqrt(N), and antithetic variates reducing variance.
"""

import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pytest

import black_scholes as bs
import mc

PARAMS = dict(S0=100.0, r=0.03, sigma=0.2, T=1.0, n_steps=252)
STRIKES = (95.0, 100.0, 105.0)


def test_mc_price_matches_black_scholes_within_3_se():
    """Large n_paths, same seed every strike (the table shares one sample)."""
    rows = mc.price_european(PARAMS["S0"], STRIKES, PARAMS["r"], PARAMS["sigma"], PARAMS["T"],
                             PARAMS["n_steps"], 200_000, seed=123)
    for row in rows:
        call_ref = bs.price(PARAMS["S0"], row["K"], PARAMS["r"], PARAMS["sigma"],
                            PARAMS["T"], "call")
        put_ref = bs.price(PARAMS["S0"], row["K"], PARAMS["r"], PARAMS["sigma"],
                           PARAMS["T"], "put")
        assert abs(row["call"] - call_ref) <= 3 * row["call_se"], (
            "K=%s call %.6f vs BS %.6f, SE %.6f" % (row["K"], row["call"], call_ref, row["call_se"]))
        assert abs(row["put"] - put_ref) <= 3 * row["put_se"], (
            "K=%s put %.6f vs BS %.6f, SE %.6f" % (row["K"], row["put"], put_ref, row["put_se"]))


def test_put_call_parity_holds_pathwise_exactly():
    """call_payoff - put_payoff = S_T - K for every path, not just on average, so
    the two Monte Carlo estimates disagree with each other by exactly the
    discounted forward minus strike -- no statistical slack at all.
    """
    ST = mc.simulate_terminal(PARAMS["S0"], PARAMS["r"], PARAMS["sigma"], PARAMS["T"],
                              PARAMS["n_steps"], 5000, seed=123)
    discount = math.exp(-PARAMS["r"] * PARAMS["T"])
    rows = mc.price_european(PARAMS["S0"], STRIKES, PARAMS["r"], PARAMS["sigma"], PARAMS["T"],
                             PARAMS["n_steps"], 5000, seed=123)
    for row in rows:
        expected = discount * (ST.mean() - row["K"])
        assert row["call"] - row["put"] == pytest.approx(expected, abs=1e-9)


def test_terminal_expectation_within_3_se_of_the_risk_neutral_forward():
    """E[S_T] = S0 e^{rT} is what makes the measure risk-neutral in the first
    place -- the original scripts simulated under mu = 0.05 and would fail this.
    """
    ST = mc.simulate_terminal(PARAMS["S0"], PARAMS["r"], PARAMS["sigma"], PARAMS["T"],
                              PARAMS["n_steps"], 200_000, seed=123)
    se = ST.std(ddof=1) / math.sqrt(len(ST))
    forward = PARAMS["S0"] * math.exp(PARAMS["r"] * PARAMS["T"])
    assert abs(ST.mean() - forward) <= 3 * se


def test_seed_reproducibility():
    rows_a = mc.price_european(PARAMS["S0"], STRIKES, PARAMS["r"], PARAMS["sigma"],
                               PARAMS["T"], PARAMS["n_steps"], 1000, seed=42)
    rows_b = mc.price_european(PARAMS["S0"], STRIKES, PARAMS["r"], PARAMS["sigma"],
                               PARAMS["T"], PARAMS["n_steps"], 1000, seed=42)
    assert rows_a == rows_b


def test_different_seeds_give_different_prices():
    rows_a = mc.price_european(PARAMS["S0"], STRIKES, PARAMS["r"], PARAMS["sigma"],
                               PARAMS["T"], PARAMS["n_steps"], 1000, seed=1)
    rows_b = mc.price_european(PARAMS["S0"], STRIKES, PARAMS["r"], PARAMS["sigma"],
                               PARAMS["T"], PARAMS["n_steps"], 1000, seed=2)
    assert rows_a != rows_b


def test_standard_error_shrinks_like_1_over_sqrt_n():
    """SE(4N) should be about half of SE(N); allow generous slack since this is
    one random draw at each size, not an average over many.
    """
    small = mc.price_european(PARAMS["S0"], [100.0], PARAMS["r"], PARAMS["sigma"],
                              PARAMS["T"], PARAMS["n_steps"], 5000, seed=123)[0]
    big = mc.price_european(PARAMS["S0"], [100.0], PARAMS["r"], PARAMS["sigma"],
                            PARAMS["T"], PARAMS["n_steps"], 20_000, seed=123)[0]
    ratio = small["call_se"] / big["call_se"]
    assert 1.5 <= ratio <= 2.7, "expected close to sqrt(20000/5000) = 2.0, got %.3f" % ratio


def test_antithetic_reduces_variance():
    plain = mc.price_european(PARAMS["S0"], [100.0], PARAMS["r"], PARAMS["sigma"],
                               PARAMS["T"], PARAMS["n_steps"], 10_000, seed=123,
                               antithetic=False)[0]
    anti = mc.price_european(PARAMS["S0"], [100.0], PARAMS["r"], PARAMS["sigma"],
                              PARAMS["T"], PARAMS["n_steps"], 10_000, seed=123,
                              antithetic=True)[0]
    assert anti["call_se"] < plain["call_se"]
    assert anti["put_se"] < plain["put_se"]


def test_antithetic_price_also_matches_black_scholes_within_3_se():
    rows = mc.price_european(PARAMS["S0"], STRIKES, PARAMS["r"], PARAMS["sigma"],
                             PARAMS["T"], PARAMS["n_steps"], 50_000, seed=123,
                             antithetic=True)
    for row in rows:
        call_ref = bs.price(PARAMS["S0"], row["K"], PARAMS["r"], PARAMS["sigma"],
                            PARAMS["T"], "call")
        assert abs(row["call"] - call_ref) <= 3 * row["call_se"]


def test_simulate_paths_shape_and_initial_value():
    paths = mc.simulate_paths(PARAMS["S0"], PARAMS["r"], PARAMS["sigma"], PARAMS["T"],
                              PARAMS["n_steps"], n_paths=10, seed=1)
    assert paths.shape == (10, PARAMS["n_steps"] + 1)
    assert np.all(paths[:, 0] == PARAMS["S0"])


def test_invalid_inputs_raise():
    with pytest.raises(ValueError, match="S0"):
        mc.simulate_terminal(-1, 0.03, 0.2, 1.0, 252, 1000)
    with pytest.raises(ValueError, match="sigma"):
        mc.simulate_terminal(100, 0.03, 0.0, 1.0, 252, 1000)
    with pytest.raises(ValueError, match="n_steps"):
        mc.simulate_terminal(100, 0.03, 0.2, 1.0, 0, 1000)
    with pytest.raises(ValueError, match="n_paths"):
        mc.simulate_terminal(100, 0.03, 0.2, 1.0, 252, 0)
    with pytest.raises(ValueError, match="antithetic"):
        mc.simulate_terminal(100, 0.03, 0.2, 1.0, 252, 101, antithetic=True)
    with pytest.raises(ValueError, match="non-empty"):
        mc.price_european(100, [], 0.03, 0.2, 1.0, 252, 1000)
    with pytest.raises(ValueError, match="K"):
        mc.price_european(100, [-5.0], 0.03, 0.2, 1.0, 252, 1000)
