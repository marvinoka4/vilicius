"""Audit the test-time forecast: feature freshness, promo join, id order, family totals."""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from pipeline import HORIZON, prepare  # noqa: E402

SUB = sys.argv[1]
fb, te = prepare("data")
tr = pd.read_csv("data/train.csv", parse_dates=["date"])
oil = pd.read_csv("data/oil.csv", parse_dates=["date"])
c = int(np.where(fb.dates == pd.Timestamp("2017-08-15"))[0][0])
X = fb.build(c)
k = fb.keys
ok = lambda name, cond: print(f"{'PASS' if cond else 'FAIL'}  {name}")

# 1. features at the test cutoff use data through 2017-08-15
w = tr.pivot(index=["store_nbr", "family"], columns="date", values="sales")
w = w.reindex(pd.MultiIndex.from_frame(k[["store_nbr", "family"]]))
L = np.log1p(w)
h1 = X[X.h == 1]
ok("last = log1p(sales on 2017-08-15)", np.allclose(h1["last"], L["2017-08-15"].values))
ok("mean_7 = mean log1p over 2017-08-09..15",
   np.allclose(h1.mean_7, L.loc[:, "2017-08-09":"2017-08-15"].mean(axis=1).values))
ok("mean_112 ends on 2017-08-15",
   np.allclose(h1.mean_112, L.loc[:, "2017-04-26":"2017-08-15"].mean(axis=1).values, equal_nan=True))
pw = tr.pivot(index=["store_nbr", "family"], columns="date", values="onpromotion").reindex(w.index)
ok("promo_sum_14 = promos 2017-08-02..15", np.allclose(h1.promo_sum_14, pw.loc[:, "2017-08-02":"2017-08-15"].sum(axis=1).values))
print("      days_since_sale range:", h1.days_since_sale.min(), h1.days_since_sale.max())

# 2. test onpromotion joins on (date, store_nbr, family)
d = fb.dates[X.cutoff.values + X.h.values.astype(int)]
chk = pd.DataFrame({"date": d, "store_nbr": k.store_nbr.values[X.series.values],
                    "family": k.family.values[X.series.values], "promo_feat": X.promo.values})
m = te.merge(chk, on=["date", "store_nbr", "family"], how="left")
ok("every test row has a feature row", m.promo_feat.notna().all() and len(m) == len(te))
ok("promo feature == test.onpromotion", np.allclose(m.promo_feat, m.onpromotion))
ok("promo_future_sum == test promo total per series",
   np.allclose(h1.promo_future_sum.values,
               te.groupby(["store_nbr", "family"]).onpromotion.sum().reindex(w.index).values))

# oil on target days equals the filled oil series
o = oil.set_index("date").dcoilwtico.reindex(pd.date_range("2013-01-01", "2017-08-31")).ffill().bfill()
ok("oil on target day == filled oil.csv", np.allclose(X.oil.values, o.reindex(d).values))
print("      oil.csv last date:", oil.date.max().date(), " missing in test window:",
      int(oil.set_index("date").dcoilwtico.reindex(pd.date_range("2017-08-16", "2017-08-31")).isna().sum()))

# 3. submission ids
sub = pd.read_csv(SUB)
ok("submission ids == test ids in the same order", (sub.id.values == te.id.values).all())
ok("no missing / negative sales", sub.sales.notna().all() and (sub.sales >= 0).all())
samp = pd.read_csv("data/sample_submission.csv")
ok("ids match sample_submission order", (sub.id.values == samp.id.values).all())

# 4. per-family totals: forecast vs last 16 days of actuals, and vs same window last year
s = te.merge(sub, on="id")
f_now = s.groupby("family").sales.sum()
a_last = tr[tr.date > "2017-07-30"].groupby("family").sales.sum()
ly = tr[(tr.date >= "2016-08-16") & (tr.date <= "2016-08-31")].groupby("family").sales.sum()
ly_prev = tr[(tr.date >= "2016-07-31") & (tr.date <= "2016-08-15")].groupby("family").sales.sum()
t = pd.DataFrame({"forecast_vs_last16": f_now / a_last, "last_year_same_change": ly / ly_prev}).round(2)
t["flag"] = np.where((t.forecast_vs_last16 - t.last_year_same_change).abs() > 0.25, "CHECK", "")
print(t.sort_values("forecast_vs_last16").to_string())
