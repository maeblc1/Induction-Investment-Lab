"""Run from repo root: python -m pytest -q"""
from dataclasses import replace
import math
import numpy as np
import pytest
from model import (Project,npv,_A,_B,gradient,hessian,break_even_savings,
    break_even_tariff,make_sigma,monte_carlo,monte_carlo_extended,validate,
    timing_break_even_k,replace_now_vs_wait,elasticity)

P=Project(I0=25000.,k=18000.,p0=.25,T=20,M=100,delta=.01,share=.8)
g,r=.02,.035

def test_closed_form():
    expected=-P.I0+P.k*P.p0*P.share*_B(P,g,r)-P.M*_A(P,r)
    assert npv(P,g,r)==pytest.approx(expected,abs=1e-8)

def test_break_even():
    assert npv(replace(P,k=break_even_savings(P,g,r)),g,r)==pytest.approx(0,abs=1e-7)
    assert npv(replace(P,p0=break_even_tariff(P,g,r)),g,r)==pytest.approx(0,abs=1e-7)

def test_gradient():
    e=1e-5
    diff=np.array([(npv(P,g+e,r)-npv(P,g-e,r))/(2*e),
                  (npv(P,g,r+e)-npv(P,g,r-e))/(2*e)])
    assert np.allclose(gradient(P,g,r),diff,atol=.05)

def test_hessian():
    e=1e-4
    center=npv(P,g,r)
    hgg=(npv(P,g+e,r)-2*center+npv(P,g-e,r))/e**2
    hrr=(npv(P,g,r+e)-2*center+npv(P,g,r-e))/e**2
    hgr=(npv(P,g+e,r+e)-npv(P,g+e,r-e)-npv(P,g-e,r+e)+npv(P,g-e,r-e))/(4*e**2)
    assert np.allclose(hessian(P,g,r),[[hgg,hgr],[hgr,hrr]],atol=5.)

def test_zero_uncertainty():
    sigma=make_sigma(0,0,0)
    assert monte_carlo(P,g,r,sigma,n=20)['mean']==pytest.approx(npv(P,g,r),abs=1e-7)
    result=monte_carlo_extended(P,g,r,sigma,n=20,sd_k_frac=0,sd_cost_frac=0,sd_share=0)
    assert result['mean']==pytest.approx(npv(P,g,r),abs=1e-7)

def test_invalid_inputs():
    with pytest.raises(ValueError,match='Lifetime'):validate(replace(P,T=0),g,r)
    with pytest.raises(ValueError,match='Correlation'):make_sigma(.02,.01,1.2)


def test_simulation_confidence_interval():
    z=monte_carlo_extended(P,g,r,make_sigma(.02,.01,0),n=500,sd_k_frac=.2,sd_cost_frac=.2)
    assert 0<=z['ci_low']<=z['p_positive']<=z['ci_high']<=1
    assert len(z['samples'])==500


def test_timing_root():
    n=5
    cutoff=timing_break_even_k(P,g,r,n)
    if cutoff is not None and cutoff>0:
        diff=replace_now_vs_wait(replace(P,k=cutoff),g,r,n)['now_minus_wait']
        assert abs(diff)<.001


def test_elasticity():
    assert isinstance(elasticity(P,g,r),dict)
