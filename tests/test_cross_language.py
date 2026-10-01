"""Runs mc.R under Rscript and mc.cpp compiled with g++, and checks both agree
with Black-Scholes within 3 standard errors -- the same check test_mc.py applies
to the Python implementation. Skips a language whose toolchain is not found
locally; CI sets up both (r-lib/actions/setup-r, g++ is the Ubuntu default) so
neither is skipped there.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pytest

import black_scholes as bs
import run

PARAMS = dict(s0=100.0, r=0.03, sigma=0.2, T=1.0, n_steps=252, n_paths=200_000,
             strikes="95,100,105", seed=123, antithetic=False)


def _assert_matches_black_scholes(rows):
    assert rows, "no rows parsed from the program's stdout"
    for row in rows:
        call_ref = bs.price(PARAMS["s0"], row["K"], PARAMS["r"], PARAMS["sigma"],
                            PARAMS["T"], "call")
        put_ref = bs.price(PARAMS["s0"], row["K"], PARAMS["r"], PARAMS["sigma"],
                           PARAMS["T"], "put")
        assert abs(row["call"] - call_ref) <= 3 * row["call_se"], (
            "K=%s call %.6f vs BS %.6f, SE %.6f" % (row["K"], row["call"], call_ref, row["call_se"]))
        assert abs(row["put"] - put_ref) <= 3 * row["put_se"], (
            "K=%s put %.6f vs BS %.6f, SE %.6f" % (row["K"], row["put"], put_ref, row["put_se"]))


def test_r_implementation_matches_black_scholes_within_3_se():
    if not run.find_rscript():
        pytest.skip("Rscript not found (set RSCRIPT or install R)")
    rows, err = run.run_r(PARAMS)
    assert rows is not None, err
    _assert_matches_black_scholes(rows)


def test_cpp_implementation_matches_black_scholes_within_3_se():
    if not run.find_gxx():
        pytest.skip("g++ not found (set CXX or install one)")
    rows, err = run.run_cpp(PARAMS)
    assert rows is not None, err
    _assert_matches_black_scholes(rows)
