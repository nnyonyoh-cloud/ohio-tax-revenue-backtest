# Ohio State Tax Revenue Back-Test

Code and data for the working paper *Interest Rates, Inflation and Ohio State Tax
Revenue: An Out-of-Sample Back-Test of Forecasting and Scenario Models*
(Nicholas Nyonyoh, draft, October 2026).

## What it does

- Forecasts annual growth in Ohio state tax collections (total, individual income,
  general sales) for FY2000–FY2025 with an expanding window, using only data
  available before each forecast year.
- Compares 10 approaches: no change, full-history drift, 10-year drift, AR(1),
  ARDL with macro variables, VAR(1), ridge, random forest, gradient boosting and
  an equal-weight combination. Horizons: 1 year and 2 years (biennium).
- Estimates a scenario model that maps fiscal-year inflation, federal funds rate
  change, Ohio unemployment change and U.S. real GDP growth into revenue growth,
  evaluates it out of sample, and produces illustrative scenarios.

## Run

```
pip install numpy pandas scipy scikit-learn
python src/backtest.py       # ~90 seconds; writes output/*.csv and meta.json
python src/key_numbers.py    # headline numbers used in the paper
python src/verify_data.py ~/Downloads   # optional: compare your own FRED downloads
```

Tested with Python 3.12, numpy 2.5, pandas 3.0, scipy 1.18, scikit-learn 1.9.
Random seeds are fixed.

## Data (`data/`, annual, from FRED, retrieved 20 October 2026)

| File | Series |
|---|---|
| OHTOTLTAX.csv | Census STC: Ohio total state taxes (thousands $) |
| OHINCTAX.csv | Census STC: Ohio individual income taxes |
| OHSALESTAX.csv | Census STC: Ohio general sales and gross receipts taxes |
| RIFSPFFNA.csv | Effective federal funds rate (%) |
| RIFLGFCY10NA.csv | 10-year Treasury constant maturity (%) |
| CPALTT01USA661S.csv | CPI, United States (OECD, 2015=100) |
| LAUST390000000000003A.csv | Ohio unemployment rate (%) |
| OHPCPI.csv | Ohio per-capita personal income ($) |
| GDPCA.csv | U.S. real GDP (chained 2017 $) |

Census STC years are state fiscal years; Ohio's FY t runs July t-1 to June t.

**Before citing these results, download each series yourself from
https://fred.stlouisfed.org/series/<ID> (Download > CSV), put the files in one
folder, and run `python src/verify_data.py <folder>`.** The script reports any
value that differs from the files in `data/`. If FRED has revised a series,
replace the file in `data/` and re-run `backtest.py`.

## Outputs (`output/`)

| File | Contents |
|---|---|
| accuracy.csv | RMSE, MAE, bias, relative RMSE, Diebold-Mariano tests by tax, horizon, model |
| forecasts.csv | Every out-of-sample forecast and error |
| interval_coverage.csv | Coverage of 80% prediction intervals (AR, ARDL) |
| scenario_sensitivities.csv | Scenario-model coefficients with Newey-West SEs |
| conditional_forecasts.csv / conditional_accuracy.csv | Scenario model out of sample |
| scenarios.csv | Illustrative scenario effects with 80% intervals |
| key_numbers.json | Figures quoted in the paper |
