"""Baselines: zero rule for dead series and a day-of-week weighted mean of recent weeks."""
import sys
import numpy as np
import pandas as pd

DATA = sys.argv[1] if len(sys.argv) > 1 else "data"

tr = pd.read_csv(f"{DATA}/train.csv", parse_dates=["date"])
te = pd.read_csv(f"{DATA}/test.csv", parse_dates=["date"])
w = tr.pivot_table(index=["store_nbr", "family"], columns="date", values="sales")


def rmsle(y, p):
    return float(np.sqrt(np.mean((np.log1p(np.clip(p, 0, None)) - np.log1p(y)) ** 2)))


def dow_forecast(hist, dates, n_weeks=8, decay=0.8, zero_days=21):
    """Weighted mean of log1p sales on the same weekday over the last n_weeks.
    More recent weeks get higher weight (decay per week). Series with no sales
    in the last zero_days are forecast as zero."""
    lh = np.log1p(hist)
    last = hist.columns.max()
    out = {}
    for d in dates:
        same = [last - pd.Timedelta(days=k) for k in range(1, 7 * n_weeks + 1)
                if (last - pd.Timedelta(days=k)).dayofweek == d.dayofweek
                and (last - pd.Timedelta(days=k)) in lh.columns]
        same = sorted(same, reverse=True)[:n_weeks]
        wts = np.array([decay ** i for i in range(len(same))])
        out[d] = np.expm1((lh[same].values * wts).sum(axis=1) / wts.sum())
    f = pd.DataFrame(out, index=hist.index)
    dead = hist.iloc[:, -zero_days:].sum(axis=1) == 0
    f.loc[dead] = 0
    return f


cut = pd.Timestamp("2017-07-30")
hist = w.loc[:, :cut]
vdates = pd.date_range("2017-07-31", "2017-08-15")
truth = w[vdates]

dead = hist.iloc[:, -21:].sum(axis=1) == 0
print("dead series at cutoff:", int(dead.sum()), "of", len(dead))

# (a) zero rule plus last-14-day mean for everything else
m14 = np.expm1(np.log1p(hist.iloc[:, -14:]).mean(axis=1))
pa = pd.DataFrame(np.repeat(m14.values[:, None], 16, 1), index=w.index, columns=vdates)
pa.loc[dead] = 0
print("zero rule + 14d mean   ", round(rmsle(truth.values, pa.values), 5))

best = None
for nw in [4, 8, 12]:
    for dc in [0.6, 0.8, 1.0]:
        s = rmsle(truth.values, dow_forecast(hist, vdates, nw, dc).values)
        print(f"dow mean weeks={nw} decay={dc}  {s:.5f}")
        if best is None or s < best[0]:
            best = (s, nw, dc)
print("best", best)

# Refit on all data with the best setting and write the test submission
_, nw, dc = best
f = dow_forecast(w, pd.date_range("2017-08-16", "2017-08-31"), nw, dc)
long = f.stack().rename("sales").reset_index().rename(columns={"level_2": "date"})
sub = te.merge(long, on=["date", "store_nbr", "family"], how="left")[["id", "sales"]]
assert sub.sales.notna().all() and len(sub) == 28512
sub.to_csv("submissions/baseline_dow.csv", index=False)
print(sub.describe())
