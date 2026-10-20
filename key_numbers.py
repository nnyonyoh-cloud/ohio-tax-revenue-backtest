"""Collect headline numbers cited in the working paper into output/key_numbers.json."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "output"
acc = pd.read_csv(OUT / "accuracy.csv")
ca = pd.read_csv(OUT / "conditional_accuracy.csv")
cf = pd.read_csv(OUT / "conditional_forecasts.csv")
fc = pd.read_csv(OUT / "forecasts.csv")
sens = pd.read_csv(OUT / "scenario_sensitivities.csv")
meta = json.loads((OUT / "meta.json").read_text())

k = {}
lvl = meta["last_level_thousands"]
k["fy2025_level_billion"] = {t: v / 1e6 for t, v in lvl.items()}
k["one_pp_million_fy2025"] = {t: v / 1e5 for t, v in lvl.items()}  # 1% of level, $ millions
# downturn years for the scenario model vs 10-yr trend
down = cf[cf.fy.isin([2009, 2010, 2020])].copy()
down["abs_err_scenario"] = (down.actual - down.scenario_fc).abs()
down["abs_err_drift10"] = (down.actual - down.drift10_fc).abs()
k["downturn_table"] = down.round(2).to_dict(orient="records")
k["downturn_mae"] = down.groupby("tax")[["abs_err_scenario", "abs_err_drift10"]].mean().round(2).to_dict()
# best unconditional model per tax/h by RMSE
best = acc.loc[acc.groupby(["tax", "h"]).RMSE.idxmin(), ["tax", "h", "model", "RMSE"]]
k["best_unconditional"] = best.round(3).to_dict(orient="records")
k["conditional_accuracy"] = ca.round(3).to_dict(orient="records")
k["dollar_rmse_total_million"] = {
    "scenario": float(ca.loc[ca.tax == "total", "RMSE_scenario"].iloc[0] * lvl["total"] / 1e5),
    "drift10": float(ca.loc[ca.tax == "total", "RMSE_drift10"].iloc[0] * lvl["total"] / 1e5),
}
k["sensitivities"] = sens.round(3).to_dict(orient="records")
(OUT / "key_numbers.json").write_text(json.dumps(k, indent=1, default=float))
print(json.dumps(k, indent=1, default=float)[:4000])
