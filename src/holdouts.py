"""Score models on the two holdouts.

A: cutoff 2017-07-30, targets 2017-07-31..08-15, all stores.
B: cutoff 2016-08-15, targets 2016-08-16..31, stores 18 and 25 excluded (they closed
   during the window, which no model can foresee).

Usage: python src/holdouts.py lgb            cache LightGBM predictions for A and B
       python src/holdouts.py linear W ALPHA TREND  score the linear model and blends
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from pipeline import HORIZON, CONFIG, lgb_forecast, linear_forecast, prepare, zero_rule  # noqa: E402

HOLDOUTS = {"A": ("2017-07-30", []), "B": ("2016-08-15", [18, 25])}
# C is a diagnostic only (not used for decisions): its window contains the largest
# store-wide promotion spikes in the training data (2016-01-20, 01-27, 02-03).
# The LightGBM lag-16 model needs lag16_start="2015-01-01" for this cutoff.
DIAG = {"C": ("2016-01-19", [])}


def setup(fb, name):
    cut, drop = {**HOLDOUTS, **DIAG}[name]
    c = int(np.where(fb.dates == pd.Timestamp(cut))[0][0])
    ser = np.tile(np.arange(fb.Y.shape[0]), HORIZON)
    y = fb.Y[ser, c + np.repeat(np.arange(1, HORIZON + 1), fb.Y.shape[0])]
    keep = ~np.isin(fb.keys.store_nbr.values[ser], drop) & ~np.isnan(y)
    dead = zero_rule(fb.Y, c, CONFIG["zero_days"])[ser]
    return c, y, keep, dead


def score(p, y, keep, dead):
    p = np.where(dead, 0, np.clip(p, 0, None))
    return float(np.sqrt(np.mean((p[keep] - y[keep]) ** 2)))


if __name__ == "__main__":
    fb, _ = prepare("data")
    os.makedirs("outputs", exist_ok=True)
    if sys.argv[1] == "lgb":
        for name in HOLDOUTS:
            c, y, keep, dead = setup(fb, name)
            _, p = lgb_forecast(fb, c)
            np.save(f"outputs/hold_{name}_lgb.npy", p)
            print(f"{name} LightGBM {score(p, y, keep, dead):.5f}", flush=True)
    elif sys.argv[1] == "linear":
        window, alpha, trend = int(sys.argv[2]), float(sys.argv[3]), sys.argv[4] == "1"
        for name in HOLDOUTS:
            c, y, keep, dead = setup(fb, name)
            pl = linear_forecast(fb, c, window, alpha, trend)
            np.save(f"outputs/hold_{name}_lin_{window}_{alpha}_{int(trend)}.npy", pl)
            pg = np.load(f"outputs/hold_{name}_lgb.npy")
            row = [f"{name} lin {score(pl, y, keep, dead):.5f} lgb {score(pg, y, keep, dead):.5f} blend"]
            row += [f"{w}:{score((1 - w) * pg + w * pl, y, keep, dead):.5f}" for w in (0.1, 0.2, 0.3, 0.4)]
            print(" ".join(row), flush=True)
