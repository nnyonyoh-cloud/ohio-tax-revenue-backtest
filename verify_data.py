"""
Compare FRED CSV files you downloaded yourself against the files in data/.

Usage:  python src/verify_data.py <folder with your downloaded CSVs>

FRED downloads are named <SERIES_ID>.csv with a header like
'observation_date,<SERIES_ID>'. Any year whose value differs by more than a
rounding tolerance is listed.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    df = pd.read_csv(path)
    df.columns = ["date", "value"]
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["year"] = df["date"].astype(str).str[:4].astype(int)
    return df.dropna().set_index("year")["value"]


def main(folder):
    folder = Path(folder).expanduser()
    problems = 0
    for ours in sorted((ROOT / "data").glob("*.csv")):
        theirs = folder / ours.name
        if not theirs.exists():
            print(f"{ours.stem:28s} not found in {folder}")
            problems += 1
            continue
        a, b = load(ours), load(theirs)
        common = a.index.intersection(b.index)
        diff = (a[common] - b[common]).abs()
        tol = 0.001  # FRED values are exact; allow only rounding noise
        bad = diff[diff > tol]
        if len(bad):
            problems += len(bad)
            print(f"{ours.stem:28s} {len(bad)} differing years:")
            for y in bad.index:
                print(f"    {y}: data/={a[y]}  yours={b[y]}")
        else:
            print(f"{ours.stem:28s} OK ({len(common)} years match)")
    print("\nAll series match." if problems == 0 else f"\n{problems} issue(s) found.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
