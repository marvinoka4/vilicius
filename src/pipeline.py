# %% [markdown]
# # Store Sales: direct multi-horizon LightGBM
#
# One global LightGBM model on log1p(sales) for all 1782 store-family series.
# Each training row is (series, cutoff date, horizon h in 1..16). Features use only
# data up to the cutoff, plus things known in advance for the target day
# (calendar, holidays, promotions, oil). The test set is forecast from the cutoff
# 2017-08-15, so nothing inside the forecast window is ever needed.

# %%
import glob
import os
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)


def find_data_dir():
    """Kaggle mounts competition data under /kaggle/input; locally we use data/."""
    hits = glob.glob("/kaggle/input/**/store-sales-time-series-forecasting/train.csv", recursive=True)
    hits += glob.glob("/kaggle/input/**/train.csv", recursive=True)
    for h in hits:
        d = os.path.dirname(h)
        if os.path.exists(os.path.join(d, "holidays_events.csv")):
            return d
    return "data"


QUAKE = (pd.Timestamp("2016-04-16"), pd.Timestamp("2016-05-31"))
HORIZON = 16

# %% [markdown]
# ## Load data and build the panel
#
# Sales go into a wide matrix: one row per series, one column per day from
# 2013-01-01 to 2017-08-31. Values are log1p(sales). Days with no real trading are
# set to NaN: Dec 25 (missing from train, stores closed) and Jan 1 (almost all
# stores closed). Window means skip NaNs, and NaN targets are never trained on.

# %%
def load_data(data_dir):
    tr = pd.read_csv(f"{data_dir}/train.csv", parse_dates=["date"])
    te = pd.read_csv(f"{data_dir}/test.csv", parse_dates=["date"])
    stores = pd.read_csv(f"{data_dir}/stores.csv")
    oil = pd.read_csv(f"{data_dir}/oil.csv", parse_dates=["date"])
    hol = pd.read_csv(f"{data_dir}/holidays_events.csv", parse_dates=["date"])
    return tr, te, stores, oil, hol


def build_panel(tr, te, stores):
    dates = pd.date_range(tr.date.min(), te.date.max())
    both = pd.concat([tr, te], ignore_index=True)
    sales = tr.pivot(index=["store_nbr", "family"], columns="date", values="sales")
    keys = sales.index.to_frame(index=False)
    sales = sales.reindex(columns=dates)
    promo = (both.pivot(index=["store_nbr", "family"], columns="date", values="onpromotion")
             .reindex(index=sales.index, columns=dates).fillna(0))

    Y = np.log1p(sales.to_numpy(dtype=np.float64))
    closed = (dates.month == 1) & (dates.day == 1) | (dates.month == 12) & (dates.day == 25)
    Y[:, closed] = np.nan
    P = promo.to_numpy(dtype=np.float32)

    keys = keys.merge(stores, on="store_nbr", how="left")
    for c in ["family", "city", "state", "type"]:
        keys[c + "_code"] = keys[c].astype("category").cat.codes
    return dates, keys, Y, P


# %% [markdown]
# ## Holidays resolved per store
#
# Rules from the data description:
# - A Holiday with `transferred=True` is a normal working day. The day it moved to
#   appears as a separate row of type Transfer, which we count as a holiday.
# - Bridge and Additional days are extra days off, counted as holidays (with their
#   own flags too). Work Day is a normal weekend day turned into a working day.
# - National rows apply to every store, Regional rows to stores in that state,
#   Local rows to stores in that city.

# %%
def holiday_matrices(hol, stores, dates):
    n_st = stores.store_nbr.max() + 1
    names = ["hol_nat", "hol_reg", "hol_loc", "bridge", "additional", "workday", "event"]
    M = {k: np.zeros((n_st, len(dates)), dtype=np.int8) for k in names}
    didx = pd.Series(np.arange(len(dates)), index=dates)
    hol = hol[~hol.transferred & hol.date.isin(dates)]
    for r in hol.itertuples():
        if r.locale == "National":
            st = stores.store_nbr.values
        elif r.locale == "Regional":
            st = stores.store_nbr[stores.state == r.locale_name].values
        else:
            st = stores.store_nbr[stores.city == r.locale_name].values
        j = didx[r.date]
        if r.type in ("Holiday", "Transfer", "Additional", "Bridge"):
            key = {"National": "hol_nat", "Regional": "hol_reg", "Local": "hol_loc"}[r.locale]
            M[key][st, j] = 1
        if r.type == "Bridge":
            M["bridge"][st, j] = 1
        if r.type == "Additional":
            M["additional"][st, j] = 1
        if r.type == "Work Day":
            M["workday"][st, j] = 1
        if r.type == "Event":
            M["event"][st, j] = 1
    M["hol_any"] = np.maximum.reduce([M["hol_nat"], M["hol_reg"], M["hol_loc"]])
    return M


def oil_series(oil, dates):
    """Daily oil price with weekends and gaps filled: forward fill, then back fill
    for the first day which has no earlier value."""
    s = oil.set_index("date").dcoilwtico.reindex(dates)
    return s.ffill().bfill().to_numpy(dtype=np.float32)


# %% [markdown]
# ## Features for one cutoff
#
# For a cutoff index `c` (the last day with known sales), we build 16 rows per
# series, one per horizon. Window statistics come from cumulative sums so each
# cutoff costs only a few vectorised operations.

# %%
class FeatureBuilder:
    WINDOWS = [3, 7, 14, 28, 56, 112]

    def __init__(self, dates, keys, Y, P, H, oil):
        self.dates, self.keys, self.Y, self.P, self.H, self.oil = dates, keys, Y, P, H, oil
        n, T = Y.shape
        valid = ~np.isnan(Y)
        Y0 = np.where(valid, Y, 0.0)
        # cumulative sums with a leading zero column so window sum = cs[:, c+1] - cs[:, c+1-w]
        self.cs = np.concatenate([np.zeros((n, 1)), Y0.cumsum(1)], 1)
        self.cs2 = np.concatenate([np.zeros((n, 1)), (Y0 ** 2).cumsum(1)], 1)
        self.cn = np.concatenate([np.zeros((n, 1)), valid.cumsum(1)], 1)
        self.cz = np.concatenate([np.zeros((n, 1)), (valid & (Y0 == 0)).cumsum(1)], 1)
        self.cp = np.concatenate([np.zeros((n, 1)), P.cumsum(1)], 1)
        pos = np.where(Y0 > 0, np.arange(T)[None, :], -10_000)
        self.lastpos = np.maximum.accumulate(pos, axis=1)
        self.firstpos = np.where((Y0 > 0).any(1), (Y0 > 0).argmax(1), T + 1)
        self.store_idx = keys.store_nbr.values
        # pad promo on the right so c+h+1 never runs past the end
        self.Ppad = np.concatenate([P, np.zeros((n, 2), dtype=P.dtype)], 1)
        self.static = {
            "store_nbr": keys.store_nbr.values, "family": keys.family_code.values,
            "city": keys.city_code.values, "state": keys.state_code.values,
            "stype": keys.type_code.values, "cluster": keys.cluster.values,
        }

    def wmean(self, c, w):
        a, b = c + 1 - w, c + 1
        return (self.cs[:, b] - self.cs[:, a]) / (self.cn[:, b] - self.cn[:, a])

    def wstd(self, c, w):
        a, b = c + 1 - w, c + 1
        n = self.cn[:, b] - self.cn[:, a]
        m = (self.cs[:, b] - self.cs[:, a]) / n
        return np.sqrt(np.maximum((self.cs2[:, b] - self.cs2[:, a]) / n - m ** 2, 0))

    def series_features(self, c):
        Y, f = self.Y, {}
        f["last"] = Y[:, c]
        for w in self.WINDOWS:
            f[f"mean_{w}"] = self.wmean(c, w)
        for w in [7, 28, 112]:
            f[f"std_{w}"] = self.wstd(c, w)
        f["trend_7_28"] = f["mean_7"] - f["mean_28"]
        f["trend_28_112"] = f["mean_28"] - f["mean_112"]
        for w in [14, 56]:
            f[f"zeros_{w}"] = self.cz[:, c + 1] - self.cz[:, c + 1 - w]
        f["days_since_sale"] = np.minimum(c - self.lastpos[:, c], 400)
        f["age"] = np.minimum(c - self.firstpos, 2000)
        for w in [14, 56]:
            f[f"promo_sum_{w}"] = self.cp[:, c + 1] - self.cp[:, c + 1 - w]
        f["promo_future_sum"] = self.cp[:, c + 1 + HORIZON] - self.cp[:, c + 1]
        # same period last year: level change between last year and now,
        # and how sales moved over the matching 16 days last year
        ly = c - 364
        f["ly_mean_28"] = self.wmean(ly, 28)
        f["ly_window_vs_base"] = self.wmean(ly + HORIZON, HORIZON) - f["ly_mean_28"]
        f["yoy_level"] = f["mean_28"] - f["ly_mean_28"]
        # store and family aggregates of recent level
        k = self.keys
        f["store_mean_14"] = pd.Series(f["mean_14"]).groupby(k.store_nbr.values).transform("mean").values
        f["fam_mean_14"] = pd.Series(f["mean_14"]).groupby(k.family_code.values).transform("mean").values
        f["store_trend"] = (pd.Series(f["trend_7_28"]).groupby(k.store_nbr.values).transform("mean").values)
        # day-of-week means: dow[r][n] = mean of the days c-r, c-r-7, ... (n weeks)
        dow = {}
        for r in range(7):
            for nw in (4, 12, 52):
                idx = c - r - 7 * np.arange(nw)
                dow[(r, nw)] = np.nanmean(Y[:, idx], axis=1)
        return f, dow

    def build(self, c, horizons=range(1, HORIZON + 1)):
        f, dow = self.series_features(c)
        n = self.Y.shape[0]
        parts = []
        for h in horizons:
            t = c + h
            d = self.dates[t]
            r = (7 - h % 7) % 7
            g = dict(f)
            g.update(self.static)
            g["h"] = np.full(n, h)
            for nw in (4, 12, 52):
                g[f"dow_mean_{nw}w"] = dow[(r, nw)]
                g[f"dow_ratio_{nw}w"] = dow[(r, nw)] - f["mean_28"] if nw != 52 else dow[(r, nw)] - f["mean_112"]
            g["ly_same_day"] = self.Y[:, t - 364]
            g["ly_week"] = self.wmean(t - 364 + 3, 7)
            g["dow"] = np.full(n, d.dayofweek)
            g["dom"] = np.full(n, d.day)
            g["month"] = np.full(n, d.month)
            g["doy"] = np.full(n, d.dayofyear)
            g["payday"] = np.full(n, int(d.day == 15 or d.is_month_end))
            g["after_payday"] = np.full(n, int(d.day in (1, 16)))
            g["promo"] = self.Ppad[:, t]
            g["promo_prev"] = self.Ppad[:, t - 1]
            g["promo_next"] = self.Ppad[:, t + 1]
            g["promo_vs_hist"] = np.log1p(self.Ppad[:, t]) - np.log1p(f["promo_sum_56"] / 56)
            for k, M in self.H.items():
                g[k] = M[self.store_idx, t]
            g["hol_any_prev"] = self.H["hol_any"][self.store_idx, t - 1]
            g["hol_any_next"] = self.H["hol_any"][self.store_idx, min(t + 1, len(self.dates) - 1)]
            g["oil"] = np.full(n, self.oil[t])
            g["oil_chg_28"] = np.full(n, self.oil[c] - self.oil[c - 28])
            g["target"] = self.Y[:, t]
            g["series"] = np.arange(n)
            g["cutoff"] = np.full(n, c)
            parts.append(pd.DataFrame(g))
        out = pd.concat(parts, ignore_index=True)
        fcols = [x for x in out.columns if x not in ("target", "series", "cutoff")]
        out[fcols] = out[fcols].astype(np.float32)
        return out


CAT_FEATURES = ["store_nbr", "family", "city", "state", "stype", "cluster"]

# %% [markdown]
# ## Training set
#
# Cutoffs step back from the forecast cutoff in multiples of `stride` days, so the
# training cutoffs fall on the same weekday as the real one. A cutoff close to the
# forecast cutoff contributes only the horizons whose targets are already known.
#
# Earthquake: the 16 April 2016 quake caused weeks of unusual sales. Rows whose
# target date or cutoff falls between 2016-04-16 and 2016-05-31 are dropped.
# Rows for series that had never sold anything by the cutoff are dropped too.

# %%
def make_train(fb, cut0, start, stride=7):
    dates = fb.dates
    rows = []
    k = 1
    while True:
        c = cut0 - k * stride
        if dates[c] < start:
            break
        k += 1
        if QUAKE[0] <= dates[c] <= QUAKE[1]:
            continue
        hs = [h for h in range(1, HORIZON + 1) if c + h <= cut0]
        df = fb.build(c, hs)
        td = dates[df.cutoff.values + df.h.values.astype(int)]
        keep = df.target.notna().values & ~((td >= QUAKE[0]) & (td <= QUAKE[1])) & (df.age.values >= 0)
        rows.append(df[keep])
    return pd.concat(rows, ignore_index=True)


LGB_PARAMS = dict(
    objective="regression", learning_rate=0.05, num_leaves=255, min_data_in_leaf=100,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
    max_bin=255, verbose=-1, num_threads=os.cpu_count(),
)


def fit_predict(train, pred, n_rounds, params=LGB_PARAMS, per_horizon=False, valid=None, seed=0):
    """Train on `train` and predict `pred`. With per_horizon=True one model is
    trained per horizon h (direct strategy); otherwise one model with h as a feature."""
    fcols = [x for x in train.columns if x not in ("target", "series", "cutoff")]
    p = dict(params, seed=seed)
    out = np.zeros(len(pred))
    groups = [(None, np.ones(len(train), bool), np.ones(len(pred), bool))]
    if per_horizon:
        groups = [(h, train.h.values == h, pred.h.values == h) for h in range(1, HORIZON + 1)]
    logs = []
    for h, mt, mp in groups:
        cols = [x for x in fcols if not (per_horizon and x == "h")]
        dtr = lgb.Dataset(train.loc[mt, cols], train.target.values[mt], categorical_feature=CAT_FEATURES,
                          free_raw_data=True)
        evals, cb = [], []
        if valid is not None:
            mv = valid.h.values == h if per_horizon else np.ones(len(valid), bool)
            dv = lgb.Dataset(valid.loc[mv, cols], valid.target.values[mv], reference=dtr)
            evals, res = [dv], {}
            cb = [lgb.record_evaluation(res)]
        m = lgb.train(p, dtr, n_rounds, valid_sets=evals, callbacks=cb)
        if valid is not None:
            logs.append(np.array(res["valid_0"]["l2"]))
        out[mp] = m.predict(pred.loc[mp, cols])
    return out, m, logs


def zero_rule(Y, c, days=21):
    """Series with no sales in the last `days` days before the cutoff."""
    w = np.nan_to_num(Y[:, c + 1 - days:c + 1])
    return w.sum(1) == 0


# %%
def prepare(data_dir):
    tr, te, stores, oil, hol = load_data(data_dir)
    dates, keys, Y, P = build_panel(tr, te, stores)
    H = holiday_matrices(hol, stores, dates)
    fb = FeatureBuilder(dates, keys, Y, P, H, oil_series(oil, dates))
    return fb, te


def make_lag16(fb, first_target, last_target, start=None):
    """Rows for the lag>=16 variant: every target day t uses the cutoff t-16,
    so all features are at least 16 days old. One row per series per target day."""
    t0 = int(np.where(fb.dates == first_target)[0][0])
    t1 = int(np.where(fb.dates == last_target)[0][0])
    rows = []
    for t in range(t0, t1 + 1):
        d = fb.dates[t]
        c = t - HORIZON
        if start is not None and (QUAKE[0] <= d <= QUAKE[1] or QUAKE[0] <= fb.dates[c] <= QUAKE[1]):
            continue
        df = fb.build(c, [HORIZON])
        if start is not None:
            df = df[df.target.notna().values & (df.age.values >= 0)]
        rows.append(df)
    return pd.concat(rows, ignore_index=True)
