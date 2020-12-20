'''Boost stats tests.'''

import numpy as np
from scipy.stats.boost import (
    beta as boost_beta,
    nbinom as boost_nbinom,
    binom as boost_binom,
    ncx2 as boost_ncx2,
)

def test_issue_12635():
    # Confirm that Boost's beta distribution resolves gh-12635. Check against R:
    # options(digits=16)
    # p = 0.9999999999997369
    # a = 75.0
    # b = 66334470.0
    # print(qbeta(p, a, b))
    p, a, b = 0.9999999999997369, 75.0, 66334470.0
    assert np.allclose(boost_beta.ppf(p, a, b), 2.343620802982393e-06)

def test_issue_12794():
    # Confirm that Boost's beta distribution resolves gh-12794. Check against R.
    # options(digits=16)
    # p = 1e-11
    # count_list = c(10,100,1000)
    # print(qbeta(1-p, count_list + 1, 100000 - count_list))
    inv_R = np.array([0.0004944464889611935,
                      0.0018360586912635726,
                      0.0122663919942518351])
    count_list = np.array([10, 100, 1000])
    p = 1e-11
    inv = boost_beta.isf(p, count_list + 1, 100000 - count_list)
    assert np.allclose(inv, inv_R)
    res = boost_beta.sf(inv, count_list + 1, 100000 - count_list)
    assert np.allclose(res, p)

def test_issue_12796():
    #Confirm that Boost's beta distribution succeeds in the case of gh-12796
    alpha_2 = 5e-6
    count_ = np.arange(1, 20)
    nobs = 100000
    q, a, b = 1 - alpha_2, count_ + 1, nobs - count_
    inv = boost_beta.ppf(q, a, b)
    res = boost_beta.cdf(inv, a, b)
    assert np.allclose(res, 1 - alpha_2)

def test_issue_10317():
    alpha, n, p = 0.9, 10, 1
    assert boost_nbinom.interval(alpha=alpha, n=n, p=p) == (0, 0)

def test_issue_11134():
    alpha, n, p = 0.95, 10, 0
    assert boost_binom.interval(alpha=alpha, n=n, p=p) == (-1, 0)

def test_issue_7406():
    np.random.seed(0)
    assert np.all(boost_binom.ppf(np.random.rand(10), 0, 0.5) == 0)

def test_binom_ppf_endpoints():
    assert boost_binom.ppf(0, 0, 0.5) == -1
    assert boost_binom.ppf(1, 0, 0.5) == 0

def test_issue_11777():
    df, nc = 6700, 5300
    n = 100
    b = boost_ncx2(df, nc)
    assert all(b.pdf(np.linspace(b.ppf(0.001), b.ppf(0.999), num=n)) > 0)
