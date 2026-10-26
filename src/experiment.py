"""Holdout experiments: cutoff 2017-07-30, targets 2017-07-31 to 2017-08-15.

Usage: python src/experiment.py NAME START STRIDE MODE ROUNDS
MODE: single (one model, h as a feature), perh (one model per horizon), lag16.
Saves holdout predictions to outputs/val_NAME.npy for blending.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from pipeline import HORIZON, LGB_PARAMS, fit_predict, make_lag16, make_train, prepare, zero_rule  # noqa: E402

name, start, stride, per_h, rounds = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], int(sys.argv[5])
t0 = time.time()
fb, te = prepare("data")
cut0 = int(np.where(fb.dates == pd.Timestamp("2017-07-30"))[0][0])
if per_h == "lag16":
    train = make_lag16(fb, pd.Timestamp(start) + pd.Timedelta(days=16), fb.dates[cut0], start=start)
    valid = make_lag16(fb, pd.Timestamp("2017-07-31"), pd.Timestamp("2017-08-15"))
    valid["h"] = valid.cutoff.values + 16 - cut0  # report by forecast day
else:
    train = make_train(fb, cut0, pd.Timestamp(start), stride)
    valid = fb.build(cut0)
print(f"train rows {len(train):,}  features {train.shape[1] - 3}  build {time.time() - t0:.0f}s", flush=True)

params = dict(LGB_PARAMS, **json.loads(os.environ.get("LGB_OVERRIDE", "{}")))
pred, model, logs = fit_predict(train, valid.assign(h=16) if per_h == "lag16" else valid, rounds, params=params,
                                per_horizon=per_h == "perh", valid=valid.assign(h=16) if per_h == "lag16" else valid)
# log-space RMSE by round (holdout has no closed days, so this is RMSLE before clipping)
curve = np.sqrt(np.mean(np.stack(logs), 0)) if per_h == "perh" else np.sqrt(logs[0])
for r in [200, 400, 600, 800, 1000, 1200, 1500, 2000]:
    if r <= len(curve):
        print(f"  round {r}: {curve[r - 1]:.5f}")

y = valid.target.values
p = np.clip(pred, 0, None)
raw = np.sqrt(np.mean((p - y) ** 2))
zr = {}
for days in (21, 56, 180, 365):
    dead = zero_rule(fb.Y, cut0, days)[valid.series.values]
    zr[days] = np.sqrt(np.mean((np.where(dead, 0, p) - y) ** 2))
p0 = p
byh = [np.sqrt(np.mean((p0[valid.h.values == h] - y[valid.h.values == h]) ** 2)) for h in range(1, HORIZON + 1)]
print(f"{name}: RMSLE {raw:.5f}  zero rule by window " + " ".join(f"{k}d:{v:.5f}" for k, v in zr.items())
      + f"  ({time.time() - t0:.0f}s)")
print("by horizon", " ".join(f"{v:.3f}" for v in byh))
np.save(f"outputs/val_{name}.npy", p0)
np.save("outputs/val_target.npy", y)

if per_h != "perh":
    imp = pd.Series(model.feature_importance("gain"), index=model.feature_name()).sort_values(ascending=False)
    print((imp / imp.sum()).round(4).head(25).to_string())
