'''Boost stats tests.'''

import numpy as np
from scipy.stats import (
    bernoulli as scipy_bernoulli,
    beta as scipy_beta,
    nbinom as scipy_nbinom,
    binom as scipy_binom,
    ncx2 as scipy_ncx2,
)
from scipy.stats.boost import (
    bernoulli as boost_bernoulli,
    beta as boost_beta,
    nbinom as boost_nbinom,
    binom as boost_binom,
    ncx2 as boost_ncx2,
)


def test_issue_11026():
    p = 0.5
    assert boost_bernoulli(p).median() == 0
    assert boost_bernoulli(p).interval(1) == scipy_bernoulli(p).interval(1)
    assert boost_bernoulli(p).interval(0.999) == scipy_bernoulli(p).interval(0.999)
    assert boost_bernoulli(p).interval(1-1e-14) == scipy_bernoulli(p).interval(1-1e-14)

def test_issue_12635():
    p = 0.9999999999997369
    a, b = 76.0, 66334470.0
    tol = 1e-2
    assert not scipy_beta.ppf(p, a-1, b) - scipy_beta.ppf(p, a, b) < tol
    assert boost_beta.ppf(p, a-1, b) - boost_beta.ppf(p, a, b) < tol

def test_issue_12794():
    count_list = [10, 100, 1000]
    p = 1e-11
    sci = [scipy_beta.isf(p, count + 1, 100000 - count) for count in count_list]
    boo = [boost_beta.isf(p, count + 1, 100000 - count) for count in count_list]
    assert not all(np.diff(sci) > 0)
    assert all(np.diff(boo) > 0)

def test_issue_10317():
    alpha, n, p = 0.9, 10, 1
    assert scipy_nbinom.interval(alpha=alpha, n=n, p=p) != (0, 0)
    assert boost_nbinom.interval(alpha=alpha, n=n, p=p) == (0, 0)

def test_issue_11134():
    alpha, n, p = 0.95, 10, 0
    assert scipy_binom.interval(alpha=alpha, n=n, p=p) != (0, 0)
    assert boost_binom.interval(alpha=alpha, n=n, p=p) == (0, 0)

def test_issue_11777():
    df, nc = 6700, 5300
    n = 100
    s = scipy_ncx2(df, nc)
    b = boost_ncx2(df, nc)
    assert not all(s.pdf(np.linspace(s.ppf(0.001), s.ppf(0.999), num=n)) > 0)
    assert all(b.pdf(np.linspace(b.ppf(0.001), b.ppf(0.999), num=n)) > 0)

def test_issue_12796():
    q, a, b = (
        0.999995,
        np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
                  19, 20]),
        np.array([99999, 99998, 99997, 99996, 99995, 99994, 99993, 99992, 99991,
                  99990, 99989, 99988, 99987, 99986, 99985, 99984, 99983, 99982,
                  99981]))
    assert not all(np.diff(scipy_beta(a, b).ppf(q)) > 0)
    assert all(np.diff(boost_beta(a, b).ppf(q)) > 0)
