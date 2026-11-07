"""Checks of the bundled, team-supplied C3 tariff data and national aggregation."""
from pathlib import Path
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / 'data'


def test_tariff_schema_and_coverage():
    df = pd.read_csv(DATA / 'tariffs_C3_clean.csv')
    assert {'year','gemeindeNummer','gemeindeName','netzbetreiber','variable_tariff'}.issubset(df.columns)
    assert len(df) == 18277
    assert set(df['year'].unique()) == set(range(2020, 2028))
    assert (df.variable_tariff > 0).all()


def test_annual_summary_matches_municipality_level_medians():
    df = pd.read_csv(DATA / 'tariffs_C3_clean.csv')
    summary = pd.read_csv(DATA / 'tariffs_C3_summary_by_year.csv').set_index('year')
    medians = (df.groupby(['year','gemeindeNummer'])['variable_tariff'].median()
               .groupby('year').median())
    assert (medians.round(2) - summary.median_variable_tariff).abs().max() <= 0.01001
    assert (df.groupby('year').size() == summary.n_rows).all()
