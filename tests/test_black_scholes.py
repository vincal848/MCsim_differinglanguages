"""Closed-form Black-Scholes against a published value and the identities that
must hold regardless of the parameters.
"""

import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pytest

import black_scholes as bs


def test_atm_reference_values():
    """S=K=100, T=1, r=5%, sigma=20% -- the textbook case, Hull ch. 15."""
    a = dict(S0=100.0, K=100.0, r=0.05, sigma=0.20, T=1.0)
    assert bs.price(**a, kind="call") == pytest.approx(10.450584, abs=1e-6)
    assert bs.price(**a, kind="put") == pytest.approx(5.573526, abs=1e-6)


def test_call_strike_105_at_the_mc_default_parameters():
    """S0=100, K=105, r=3%, sigma=20%, T=1 -- the case this repository's tables
    are built around.
    """
    assert bs.price(100.0, 105.0, 0.03, 0.2, 1.0, "call") == pytest.approx(7.128065, abs=1e-6)


@pytest.mark.parametrize("S0,K,r,sigma,T", [
    (100, 100, 0.05, 0.20, 1.0),
    (100, 95, 0.03, 0.20, 1.0),
    (100, 105, 0.03, 0.20, 1.0),
    (42, 40, 0.10, 0.20, 0.5),
])
def test_put_call_parity(S0, K, r, sigma, T):
    """C - P = S0 - K e^{-rT}, exactly, for any parameters (no dividend here)."""
    c = bs.price(S0, K, r, sigma, T, "call")
    p = bs.price(S0, K, r, sigma, T, "put")
    assert c - p == pytest.approx(S0 - K * math.exp(-r * T), abs=1e-10)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError, match="call.*or.*put"):
        bs.price(100, 100, 0.05, 0.2, 1.0, "straddle")
    with pytest.raises(ValueError, match="sigma"):
        bs.price(100, 100, 0.05, 0.0, 1.0, "call")
    with pytest.raises(ValueError, match="positive"):
        bs.price(-1, 100, 0.05, 0.2, 1.0, "call")
    with pytest.raises(ValueError, match="T"):
        bs.price(100, 100, 0.05, 0.2, -1.0, "call")
