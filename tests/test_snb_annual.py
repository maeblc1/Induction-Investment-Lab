"""Checks for the team-supplied annual 10-year Confederation yield summary."""
from pathlib import Path
import pandas as pd
import numpy as np
from model import calibrated_sds

DATA=Path(__file__).resolve().parents[1]/'data'

def test_yield_schema_and_years():
    df=pd.read_csv(DATA/'yields_10y_annual.csv')
    assert {'year','mean','last','min','max','n_obs'}.issubset(df.columns)
    assert list(df.year)==list(range(2020,2027))
    assert (df.n_obs>0).all()
    assert (df['min']<=df['mean']).all()
    assert (df['mean']<=df['max']).all()
    assert (df['min']<=df['last']).all()
    assert (df['last']<=df['max']).all()

def test_bundled_last_and_volatility_units():
    df=pd.read_csv(DATA/'yields_10y_annual.csv')
    assert np.isclose(float(df.iloc[-1]['last']),0.505)
    cal=calibrated_sds(yield_series=df['last'].to_numpy(),horizon=10)
    assert cal['yield_observations']==7
    assert np.isclose(cal['sd_r'],df['last'].std(ddof=1)/100)
    assert 0<cal['sd_r']<0.1
