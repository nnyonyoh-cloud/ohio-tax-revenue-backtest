"""
Ohio state tax revenue back-test.

Out-of-sample (expanding-window) evaluation of models that forecast Ohio state
government tax collections from interest-rate, inflation and activity conditions,
plus a conditional "scenario" model that maps macro paths into revenue growth.

Data: FRED annual series (see README). Census Annual Survey of State Government
Tax Collections reports state fiscal years; Ohio's fiscal year t runs July 1 of
calendar year t-1 to June 30 of calendar year t.

Run:  python src/backtest.py           (reads data/*.csv, writes output/*)
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
DATA, OUT = ROOT / "data", ROOT / "output"
OUT.mkdir(exist_ok=True)
SEED = 20261020

TAXES = {"total": "OHTOTLTAX", "income": "OHINCTAX", "sales": "OHSALESTAX"}
TAX_LABEL = {"total": "Total taxes", "income": "Individual income tax",
             "sales": "General sales tax"}
FIRST_TARGET = 1979      # first fiscal year used in estimation
FIRST_FORECAST = 2000    # first fiscal year forecast out of sample
Z80 = stats.norm.ppf(0.90)


# --------------------------------------------------------------------------- data
def read_fred(series_id):
    """Read a FRED CSV (either 'DATE' or 'observation_date' header)."""
    df = pd.read_csv(DATA / f"{series_id}.csv")
    df.columns = ["date", "value"]
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["year"] = df["date"].astype(str).str[:4].astype(int)
    return df.dropna().set_index("year")["value"]


def build_panel():
    """Annual panel indexed by calendar year (macro) / fiscal year (revenue)."""
    m = pd.DataFrame({
        "ff": read_fred("RIFSPFFNA"),             # fed funds, %
        "r10": read_fred("RIFLGFCY10NA"),         # 10y Treasury, %
        "cpi": read_fred("CPALTT01USA661S"),      # CPI index
        "ur": read_fred("LAUST390000000000003A"), # Ohio unemployment, %
        "pcpi": read_fred("OHPCPI"),              # Ohio per-capita personal income, $
        "gdp": read_fred("GDPCA"),                # US real GDP
    })
    x = pd.DataFrame(index=m.index)
    x["infl"] = 100 * np.log(m["cpi"]).diff()
    x["ff"] = m["ff"]
    x["dff"] = m["ff"].diff()
    x["term"] = m["r10"] - m["ff"]
    x["dur"] = m["ur"].diff()
    x["pinc"] = 100 * np.log(m["pcpi"]).diff()
    x["gdp"] = 100 * np.log(m["gdp"]).diff()
    rev = pd.DataFrame({k: read_fred(v) for k, v in TAXES.items()})
    return m, x, rev


# ------------------------------------------------------------------------ helpers
def ols(X, y):
    X1 = np.column_stack([np.ones(len(X)), X])
    b, *_ = np.linalg.lstsq(X1, y, rcond=None)
    resid = y - X1 @ b
    return b, resid


def newey_west(X, resid, lags=1):
    X1 = np.column_stack([np.ones(len(X)), X])
    n, k = X1.shape
    u = X1 * resid[:, None]
    S = u.T @ u
    for l in range(1, lags + 1):
        w = 1 - l / (lags + 1)
        G = u[l:].T @ u[:-l]
        S += w * (G + G.T)
    XtXi = np.linalg.inv(X1.T @ X1)
    V = XtXi @ S @ XtXi * n / (n - k)
    return V


def dm_test(e1, e2, h=1):
    """Diebold-Mariano with Harvey-Leybourne-Newbold correction, squared loss.
    Negative stat => model 1 more accurate. Two-sided p-value."""
    d = e1 ** 2 - e2 ** 2
    n = len(d)
    dbar = d.mean()
    gamma = [np.sum((d[l:] - dbar) * (d[:n - l] - dbar)) / n for l in range(h)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / n
    if var <= 0:
        return np.nan, np.nan
    stat = dbar / np.sqrt(var)
    stat *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    p = 2 * stats.t.sf(abs(stat), df=n - 1)
    return stat, p


# --------------------------------------------------------------- design matrices
MACRO = ["infl", "dff", "ff", "term", "dur", "gdp", "pinc"]
ARDL_VARS = ["infl", "dff", "dur", "gdp"]           # parsimonious direct model
VAR_VARS = ["infl", "dff", "dur", "gdp"]            # system for iterated VAR


def unconditional_frame(y, x):
    """Rows indexed by fiscal year t: info set at start of FY t (July t-1).
    Uses y_{t-1} and full calendar-year macro for t-2 (conservative timing)."""
    f = pd.DataFrame(index=y.index)
    f["y"] = y
    f["y_lag"] = y.shift(1)
    for v in MACRO:
        f[v] = x[v].reindex(f.index - 2).values
    f["y2"] = y + y.shift(-1)          # cumulative growth FY t and t+1 (biennium)
    return f


def var_frame(y, x):
    """State vector Z_t = [y_t, macro_{t-1}] so that Z_{t-1} = [y_{t-1}, macro_{t-2}]
    matches the information set used above."""
    z = pd.DataFrame(index=y.index)
    z["y"] = y
    for v in VAR_VARS:
        z[v] = x[v].reindex(z.index - 1).values
    return z


# --------------------------------------------------------------------- models
def fit_predict(name, train, test_row, target, feats):
    Xtr, ytr = train[feats].values, train[target].values
    Xte = test_row[feats].values.reshape(1, -1)
    if name == "Ridge":
        mdl = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 40)))
    elif name == "Random forest":
        mdl = RandomForestRegressor(n_estimators=500, min_samples_leaf=3,
                                    max_features=0.5, random_state=SEED)
    elif name == "Gradient boosting":
        mdl = GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=0.05,
                                        subsample=0.8, random_state=SEED)
    else:
        b, resid = ols(Xtr, ytr)
        pred = b[0] + Xte @ b[1:]
        se = np.sqrt(resid @ resid / (len(ytr) - len(b)))
        return float(pred[0]), float(se)
    mdl.fit(Xtr, ytr)
    return float(mdl.predict(Xte)[0]), np.nan


def var_forecast(zt, origin, steps):
    """Estimate VAR(1) by OLS on rows < origin; iterate from row origin-1."""
    tr = zt.loc[:origin - 1].dropna()
    Y, X = tr.values[1:], tr.values[:-1]
    X1 = np.column_stack([np.ones(len(X)), X])
    B, *_ = np.linalg.lstsq(X1, Y, rcond=None)
    z = zt.loc[origin - 1].values
    path = []
    for _ in range(steps):
        z = np.concatenate([[1.0], z]) @ B
        path.append(z[0])
    return path


MODELS = ["No change", "Drift", "Drift (10-yr)", "AR(1)", "ARDL-macro", "VAR(1)", "Ridge",
          "Random forest", "Gradient boosting", "Combination"]
COMBINE = ["AR(1)", "ARDL-macro", "VAR(1)", "Ridge", "Random forest", "Gradient boosting"]


def run_unconditional(tax, y, x):
    f = unconditional_frame(y, x).loc[FIRST_TARGET:]
    zt = var_frame(y, x).loc[FIRST_TARGET - 1:]
    all_feats = ["y_lag"] + MACRO
    rows = []
    last = int(y.index.max())
    for h, target in [(1, "y"), (2, "y2")]:
        for t in range(FIRST_FORECAST, last + 1 - (h - 1)):
            # train only on rows whose target is fully observed before July t-1
            train = f.loc[:t - h].dropna(subset=[target] + all_feats)
            if h == 2:
                train = train.loc[:t - 2]
            test = f.loc[t]
            actual = test[target]
            if np.isnan(actual):
                continue
            hist = y.loc[FIRST_TARGET:t - 1]
            p = {"No change": 0.0, "Drift": h * hist.mean(),
                 "Drift (10-yr)": h * hist.iloc[-10:].mean()}
            se = {}
            p["AR(1)"], se["AR(1)"] = fit_predict("OLS", train, test, target, ["y_lag"])
            p["ARDL-macro"], se["ARDL-macro"] = fit_predict("OLS", train, test, target,
                                                            ["y_lag"] + ARDL_VARS)
            path = var_forecast(zt, t, h)
            p["VAR(1)"] = float(sum(path))
            for mname in ["Ridge", "Random forest", "Gradient boosting"]:
                p[mname], _ = fit_predict(mname, train, test, target, all_feats)
            p["Combination"] = float(np.mean([p[m] for m in COMBINE]))
            for m in MODELS:
                rows.append({"tax": tax, "h": h, "fy": t, "model": m, "forecast": p[m],
                             "actual": actual, "error": actual - p[m],
                             "se": se.get(m, np.nan), "n_train": len(train)})
    return pd.DataFrame(rows)


def summarize(fc):
    out = []
    for (tax, h), g in fc.groupby(["tax", "h"]):
        bench = g[g.model == "Drift"].set_index("fy")["error"]
        for m, gm in g.groupby("model", sort=False):
            e = gm.set_index("fy")["error"]
            rmse = np.sqrt((e ** 2).mean())
            dm, pv = dm_test(e.values, bench.loc[e.index].values, h) if m != "Drift" \
                else (np.nan, np.nan)
            b10 = g[g.model == "Drift (10-yr)"].set_index("fy")["error"]
            dm10, pv10 = dm_test(e.values, b10.loc[e.index].values, h) \
                if m != "Drift (10-yr)" else (np.nan, np.nan)
            # downturn years: actual decline
            dec = gm[gm.actual < 0]
            row = {"MAE_decline_years": dec.error.abs().mean() if len(dec) else np.nan,
                   "n_decline": len(dec),
"tax": tax, "h": h, "model": m, "n": len(e),
                   "RMSE": rmse, "MAE": e.abs().mean(), "Bias": e.mean(),
                   "RMSE_vs_drift": rmse / np.sqrt((bench ** 2).mean()),
                   "RMSE_vs_nochange": rmse / np.sqrt((g[g.model == "No change"].error ** 2).mean()),
                   "RMSE_vs_drift10": rmse / np.sqrt((g[g.model == "Drift (10-yr)"].error ** 2).mean()),
                   "DM_stat": dm, "DM_p": pv,
                   "DM_stat_vs_drift10": dm10, "DM_p_vs_drift10": pv10,
                   "RMSE_ex_2021": np.sqrt((e.drop([2021, 2022], errors="ignore") ** 2).mean())}
            out.append(row)
    return pd.DataFrame(out)


def interval_coverage(fc):
    rows = []
    for (tax, h, m), g in fc[fc.model.isin(["AR(1)", "ARDL-macro"])].groupby(["tax", "h", "model"]):
        inside = (g.error.abs() <= Z80 * g.se)
        rows.append({"tax": tax, "h": h, "model": m, "nominal": 0.80,
                     "coverage": inside.mean(), "n": len(g)})
    return pd.DataFrame(rows)


# ------------------------------------------------------- conditional scenario model
SCEN_VARS = ["infl", "dff", "dur", "gdp"]


def conditional_frame(y, x):
    """FY-aligned macro: average of calendar t-1 and t (FY t spans both)."""
    f = pd.DataFrame(index=y.index)
    f["y"] = y
    f["y_lag"] = y.shift(1)
    for v in SCEN_VARS:
        f[v] = 0.5 * (x[v].reindex(f.index - 1).values + x[v].reindex(f.index).values)
    return f.loc[FIRST_TARGET:].dropna()


def run_conditional(tax, y, x):
    f = conditional_frame(y, x)
    feats = ["y_lag"] + SCEN_VARS
    # full-sample sensitivities with Newey-West SEs
    b, resid = ols(f[feats].values, f["y"].values)
    V = newey_west(f[feats].values, resid, lags=1)
    se = np.sqrt(np.diag(V))
    coefs = pd.DataFrame({"tax": tax, "term": ["const"] + feats, "coef": b, "se": se,
                          "t": b / se, "p": 2 * stats.t.sf(np.abs(b / se), len(f) - len(b))})
    r2 = 1 - resid @ resid / np.sum((f.y - f.y.mean()) ** 2)
    # pseudo out-of-sample with realized (known) scenario inputs
    errs = []
    for t in range(FIRST_FORECAST, int(f.index.max()) + 1):
        tr = f.loc[:t - 1]
        bb, _ = ols(tr[feats].values, tr["y"].values)
        pred = bb[0] + f.loc[t, feats].values @ bb[1:]
        drift = y.loc[FIRST_TARGET:t - 1].mean()
        drift10 = y.loc[:t - 1].iloc[-10:].mean()
        errs.append({"tax": tax, "fy": t, "actual": f.loc[t, "y"], "scenario_fc": pred,
                     "drift_fc": drift, "drift10_fc": drift10})
    e = pd.DataFrame(errs)
    meta = {"tax": tax, "n": len(f), "first_fy": int(f.index.min()), "last_fy": int(f.index.max()),
            "r2": r2, "resid_sd": float(np.sqrt(resid @ resid / (len(f) - len(b))))}
    return coefs, e, meta, (b, V, feats)


# Illustrative scenarios for a fiscal year (inputs, not predictions).
SCENARIOS = {
    "Baseline":                {"infl": 2.5, "dff": 0.0,  "dur": 0.0, "gdp": 2.0},
    "Higher for longer":       {"infl": 3.5, "dff": 1.0,  "dur": 0.5, "gdp": 1.0},
    "Inflation resurgence":    {"infl": 5.5, "dff": 2.0,  "dur": 0.5, "gdp": 1.0},
    "Recession with rate cuts": {"infl": 1.5, "dff": -2.0, "dur": 2.0, "gdp": -1.5},
}


def scenario_table(tax, b, V, feats):
    rows = []
    base = SCENARIOS["Baseline"]
    for name, s in SCENARIOS.items():
        d = np.zeros(len(b))
        for i, v in enumerate(feats):
            if v in s:
                d[i + 1] = s[v] - base[v]
        eff = d @ b
        se = np.sqrt(d @ V @ d)
        rows.append({"tax": tax, "scenario": name, **{f"in_{k}": v for k, v in s.items()},
                     "diff_vs_baseline_pp": eff, "lo80": eff - Z80 * se, "hi80": eff + Z80 * se})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------- main
def main():
    m, x, rev = build_panel()
    growth = 100 * np.log(rev).diff()
    fcs, condc, conde, condm, scen = [], [], [], [], []
    for tax in TAXES:
        y = growth[tax].dropna()
        fcs.append(run_unconditional(tax, y, x))
        c, e, meta, (b, V, feats) = run_conditional(tax, y, x)
        condc.append(c); conde.append(e); condm.append(meta)
        scen.append(scenario_table(tax, b, V, feats))
    fc = pd.concat(fcs)
    summ = summarize(fc)
    cov = interval_coverage(fc)
    coefs = pd.concat(condc)
    ce = pd.concat(conde)
    cond_acc = ce.groupby("tax").apply(lambda g: pd.Series({
        "n": len(g),
        "RMSE_scenario": np.sqrt(((g.actual - g.scenario_fc) ** 2).mean()),
        "RMSE_drift": np.sqrt(((g.actual - g.drift_fc) ** 2).mean()),
        "RMSE_drift10": np.sqrt(((g.actual - g.drift10_fc) ** 2).mean()),
        "DM_p": dm_test((g.actual - g.scenario_fc).values, (g.actual - g.drift_fc).values)[1],
        "DM_p_drift10": dm_test((g.actual - g.scenario_fc).values,
                                (g.actual - g.drift10_fc).values)[1],
        "RMSE_scenario_ex_2021_22": np.sqrt(((g.actual - g.scenario_fc)[~g.fy.isin([2021, 2022])] ** 2).mean()),
        "RMSE_drift10_ex_2021_22": np.sqrt(((g.actual - g.drift10_fc)[~g.fy.isin([2021, 2022])] ** 2).mean()),
    })).reset_index()
    cond_acc["ratio_drift10"] = cond_acc.RMSE_scenario / cond_acc.RMSE_drift10
    cond_acc["ratio"] = cond_acc.RMSE_scenario / cond_acc.RMSE_drift
    sc = pd.concat(scen)

    fc.to_csv(OUT / "forecasts.csv", index=False)
    summ.to_csv(OUT / "accuracy.csv", index=False)
    cov.to_csv(OUT / "interval_coverage.csv", index=False)
    coefs.to_csv(OUT / "scenario_sensitivities.csv", index=False)
    ce.to_csv(OUT / "conditional_forecasts.csv", index=False)
    cond_acc.to_csv(OUT / "conditional_accuracy.csv", index=False)
    sc.to_csv(OUT / "scenarios.csv", index=False)
    growth.loc[FIRST_TARGET:].round(3).to_csv(OUT / "revenue_growth.csv")
    meta = {"conditional": condm, "last_level_thousands": rev.iloc[-1].to_dict(),
            "last_fy": int(rev.index.max())}
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2, default=float))

    pd.set_option("display.width", 200, "display.max_columns", 20)
    print(summ.round(3).to_string())
    print(cov.round(2).to_string())
    print(coefs.round(3).to_string())
    print(cond_acc.round(3).to_string())
    print(sc.round(2).to_string())
    print(json.dumps(meta, indent=1, default=float))


if __name__ == "__main__":
    main()
