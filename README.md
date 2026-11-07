# Induction Investment Lab

An interactive financial decision tool for a **Swiss restaurant replacing a conventional electric hob with induction**. It does **not** model gas hob replacement. Designed for a mathematics-for-economics course project.

## Deploy

Upload all project files and the `data/` and `tests/` folders to the **root of your existing GitHub repository**. In Streamlit Community Cloud set main file to `app.py`. Existing deployment will redeploy from the repository after commits. To run locally: `python -m pip install -r requirements.txt && streamlit run app.py`.

## What changed

- Decision-first interface and high-contrast result cards, restrained editorial styling, no decorative emojis.
- Year-0 discounted cash-flow chart with zero line and payback marker.
- Preset-based scenario tornado and a complementary dimensionless elasticity ranking (not an empirical confidence ranking).
- Baseline vs alternative in session state; export differences.
- Vectorised 2,000–100,000-draw extended Monte Carlo: correlated normal growth/discount-rate shocks and independent lognormal savings and cost shocks; clipped normal avoidable share. Wilson confidence interval and convergence chart, comparison for correlations -0.6, 0 and +0.6, bisection stress threshold for P(NPV>0) approximately 90%.
- Break-even annual kWh for **replace now vs wait** via bisection (full-cost scenario only).
- Live verification panel and `tests/test_model.py` with 9 tests.
- Official-source references and transparent origin tracing; download an AI comparison prompt.

## Dataset availability and source status

The app bundles two C3 tariff files supplied by the project team on 2026-11-07:

- `data/tariffs_C3_clean.csv` — 18,277 municipality/operator rows (2020–2027), containing the variable tariff in Rp/kWh and tariff components.
- `data/tariffs_C3_summary_by_year.csv` — annual national summaries including municipality-level median tariff, quantiles and reported log changes.
- `data/snb_verified_observation.csv` — one documented 10-year Confederation yield snapshot (0.450%, 2026-11-06), **not** a historical series.
- `data/PROVENANCE.md` — unit, scope, calculation, limitations and missing original-download metadata.

**Source distinction:** these C3 CSVs were supplied by the project team. Their original retrieval date, raw ElCom export and cleaning procedure are not available for independent verification. The app labels their provenance accordingly. ElCom source: https://www.elcom.admin.ch/en/tariff-data-and-visualisations .

National medians first collapse multiple operator rows into one median per municipality, then calculate the national median of municipalities. This explains differences versus the raw-row median. Default year is 2026; 2027 is offered explicitly as a future tariff year. Users may override with an actual invoice or a separate uploaded CSV. Note that C3 is not a verified restaurant-specific marginal avoided price.

A team-supplied 2020–2026 annual SNB-style yield summary is now bundled and used for descriptive discount-rate volatility calibration. The underlying raw daily source and extraction date are still needed for independent authentication; annual variation is not a forward-looking risk distribution.

## Mathematical interpretation and constraints

Nominal tariff growth `g` is paired with nominal rate `r` = risk-free rate + assumed risk spread. **The 2.5 percentage point spread is unverified** and should be justified by an appropriate source or presented as a scenario. For restaurants, the fraction of the published average tariff that is *avoidable* may be below 100%; the interface starts at 80%, which is also a scenario assumption.

The SFOE 18,000 kWh annual reduction is a monitored average for a kitchen case, **not** an installation-specific energy audit. CHF 25,000 purchase cost and the implied incremental cost are assumptions in the cited SFOE model, **not supplier quotations**. The tool assumes the user includes site-specific installation costs, subsidies (confirmed only), maintenance and other material cash flows in their inputs.

**Monte Carlo:** the probability is conditional on the selected distributions and standard deviations, not a calibrated probability unless real data and realistic distributions are established. A zero-failure run must not be described as “100% certain”; the Wilson interval is reported. The delta-method mean/variance are based on *g/r only* and should not be compared as if they included uncertain k, I0 and share. Correlation scenarios are illustrative, and cost / savings / share independent shocks are simplifying assumptions.

**Tornado:** ranges for equipment cost, lifetime, maintenance, kWh and tariff fraction are from `model.PRESETS`, where each range's empirical/assumed origin is stated. Tariff defaults to an explicitly illustrative ±25% range unless sufficient real historical data have been uploaded. The g/r stress widths remain assumptions. Elasticities can explode near NPV=0, so they are suppressed there.

**Comparison:** a saved baseline is held for the duration of the Streamlit session. The full-price replacement-timing comparison uses a common horizon, an assumed residual value and no old appliance salvage price. For an already-needed replacement, use **incremental** cost; for timing, use **full** cost.

## Validation

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Nine repeatable checks cover NPV identity, analytical vs numerical derivatives, break-even equalities, zero-uncertainty simulation, invalid life/correlation, MC confidence bounds, and the timing root.

## AI comparison

Export the app's *AI-comparison prompt*, submit the same inputs to an AI model, then report its model name/version, date, full response, NPV difference, sourcing gaps and sensitivity omissions. The app **does not fabricate an AI answer or imply that an AI comparison has already taken place**.


## Bundled ElCom data (added 2026-11-07)
The application now defaults to `data/tariffs_C3_clean.csv`, and provides year/municipality selection, operator-level tracing, 2020–2027 annual history and `data/tariffs_C3_summary_by_year.csv`. Both files were supplied by the project team; the independent ElCom download date and extraction method are not documented. Annual national medians are calculated after aggregating multiple operators per municipality. Year 2027 is forward-looking. The automatically selected year is 2026 to avoid choosing a future tariff silently. CSV uploads can still override the bundled data. The C3 tariff is **not necessarily equal to avoidable savings**.

## Annual ten-year Confederation bond yield data (added 2026-11-07)

The app now bundles `data/yields_10y_annual.csv` alongside the independent one-day observation. The 2026 `last` value (0.505%, based on 217 underlying observations) is used as the default selected yield, with a year/mean/last selector and a traceable source table. Historical SD is available from the seven annual values, but the dataset was supplied by the project team; original extraction metadata and independent daily-data reconciliation remain outstanding. The one-day 6 November observation is 0.450%, which differs from the supplied annual `last` value.
