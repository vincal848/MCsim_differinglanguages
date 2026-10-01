"""Closed-form Black-Scholes price, the reference mc.py, mc.R and mc.cpp are
checked against.

Standard formula (Hull ch. 15), constant r and sigma, no dividend -- that is the
only case the Monte Carlo pricer in this repository simulates, so there is no
dividend yield parameter here.

Uses only the standard library (math.erf for the normal CDF) so that importing
this module does not pull in scipy just to run the tests.
"""

import math


def check_kind(kind):
    """Normalize and validate the option type."""
    k = str(kind).lower()
    if k not in ("call", "put"):
        raise ValueError("kind must be 'call' or 'put', got %r" % (kind,))
    return k


def _ncdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def price(S0, K, r, sigma, T, kind="call"):
    """European option price.

    >>> round(price(100, 100, 0.05, 0.20, 1.0, "call"), 6)
    10.450584
    >>> round(price(100, 100, 0.05, 0.20, 1.0, "put"), 6)
    5.573526
    """
    k = check_kind(kind)
    if S0 <= 0 or K <= 0:
        raise ValueError("S0 and K must be strictly positive")
    if sigma <= 0:
        raise ValueError("sigma must be strictly positive")
    if T <= 0:
        raise ValueError("T must be strictly positive")

    vol = sigma * math.sqrt(T)
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / vol
    d2 = d1 - vol

    if k == "call":
        return S0 * _ncdf(d1) - K * math.exp(-r * T) * _ncdf(d2)
    return K * math.exp(-r * T) * _ncdf(-d2) - S0 * _ncdf(-d1)
