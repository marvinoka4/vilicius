# %% [markdown]
# # Backtesting store sales: why one holdout is not enough
#
# The usual way to validate a model for this competition is to hide the last 16 days
# of train (2017-07-31 to 2017-08-15), score that window, and trust the number. I did
# exactly that. A LightGBM pipeline scored **0.383** RMSLE on that holdout and
# **0.420** on the leaderboard. The leaderboard here scores the entire test set (there
# is no private split), so the gap is real, not sampling noise.
#
# This notebook backtests a compact model over many cutoffs and looks at what a
# single holdout hides:
#
# 1. How much the score moves from one 16-day window to the next.
# 2. A window a year earlier where two stores closed, and what that does to a score.
# 3. A modelling choice (the "zero rule" for dead series) that flips depending on the
#    window you test it on.
# 4. What was different about the test window: an early, very large back-to-school
#    season and a promotion spike on the first test day.
#
# Everything here uses only the competition files and runs on Kaggle CPU in about
# 10 minutes.

# %%
import glob
import os
import time
import warnings

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

hits = [h for h in glob.glob("/kaggle/input/**/train.csv", recursive=True)
        if os.path.exists(os.path.join(os.path.dirname(h), "holidays_events.csv"))]
DATA = os.path.dirname(hits[0]) if hits else "data"

BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
plt.rcParams.update({
    "figure.figsize": (12, 4), "figure.dpi": 110, "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb", "axes.edgecolor": "#c9c8c3", "axes.labelcolor": "#52514e",
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#e8e7e3", "grid.linewidth": 0.8, "xtick.color": "#52514e",
    "ytick.color": "#52514e", "text.color": "#0b0b0b", "lines.linewidth": 1.6,
    "legend.frameon": False,
})
H = 16

# %% [markdown]
# ## Data as a matrix
#
# One row per (store, family) series, one column per day, values log1p(sales). RMSLE
# is the RMSE of log1p values, so everything below works in that space. Dec 25
# (missing) and Jan 1 (stores closed) are set to NaN so they never act as targets or
# drag down averages.

# %%
train = pd.read_csv(f"{DATA}/train.csv", parse_dates=["date"])
test = pd.read_csv(f"{DATA}/test.csv", parse_dates=["date"])
stores = pd.read_csv(f"{DATA}/stores.csv")

dates = pd.date_range(train.date.min(), test.date.max())
sales = train.pivot(index=["store_nbr", "family"], columns="date", values="sales").reindex(columns=dates)
promo = (pd.concat([train, test]).pivot(index=["store_nbr", "family"], columns="date", values="onpromotion")
         .reindex(index=sales.index, columns=dates).fillna(0).to_numpy(np.float32))
keys = sales.index.to_frame(index=False)
Y = np.log1p(sales.to_numpy(np.float64))
Y[:, ((dates.month == 1) & (dates.day == 1)) | ((dates.month == 12) & (dates.day == 25))] = np.nan
N = len(keys)
fam_code = keys.family.astype("category").cat.codes.values
didx = {d: i for i, d in enumerate(dates)}
print(Y.shape)

# %% [markdown]
# ## A compact forecaster
#
# For backtesting I want something quick enough to refit many times, so this is a
# trimmed version of a direct multi-horizon LightGBM model:
#
# - For a cutoff day `c`, each series gets features computed from data up to `c`:
#   recent means, same-weekday means, last year's change over the same 16 days, and
#   the promotion count on the target day relative to the series' usual level.
# - One row per (series, horizon h = 1..16). Training uses earlier cutoffs every 7
#   days over the previous year, so the model never sees anything after `c`.
#
# The forecast for any window therefore only uses information that would have been
# available on the cutoff day. That is what makes the backtest honest.

# %%
def win_mean(c, w):
    return np.nanmean(Y[:, c - w + 1:c + 1], axis=1)


def features(c, horizons=range(1, H + 1)):
    base = {f"mean_{w}": win_mean(c, w) for w in (3, 7, 14, 28, 56)}
    base["ly_change"] = win_mean(c - 364 + H, H) - win_mean(c - 364, 28)
    promo_hist = promo[:, c - 55:c + 1].mean(1)
    rows = []
    for h in horizons:
        t = c + h
        r = (7 - h % 7) % 7
        g = dict(base)
        g["dow_4w"] = np.nanmean(Y[:, c - r - 7 * np.arange(4)], axis=1)
        g["dow_12w"] = np.nanmean(Y[:, c - r - 7 * np.arange(12)], axis=1)
        g["ly_day"] = Y[:, t - 364]
        g["promo"] = promo[:, t]
        g["promo_rel"] = np.log1p(promo[:, t]) - np.log1p(promo_hist)
        g["h"] = np.full(N, h)
        g["dow"] = np.full(N, dates[t].dayofweek)
        g["dom"] = np.full(N, dates[t].day)
        g["store"] = keys.store_nbr.values
        g["family"] = fam_code
        g["target"] = Y[:, t] if t < Y.shape[1] else np.nan
        g["series"] = np.arange(N)
        rows.append(pd.DataFrame(g))
    return pd.concat(rows, ignore_index=True)


QUAKE = (pd.Timestamp("2016-04-16"), pd.Timestamp("2016-05-31"))
PARAMS = dict(objective="regression", learning_rate=0.08, num_leaves=127, min_data_in_leaf=100,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=0)


def forecast(cutoff, n_weeks=52, rounds=300):
    """Train on cutoffs every 7 days in the year before `cutoff`, then forecast the
    16 days after it. Returns log1p predictions in (horizon, series) order."""
    c = didx[pd.Timestamp(cutoff)]
    parts = []
    for k in range(1, n_weeks + 1):
        ck = c - 7 * k
        if QUAKE[0] - pd.Timedelta(days=H) <= dates[ck] <= QUAKE[1]:
            continue
        df = features(ck, [h for h in range(1, H + 1) if ck + h <= c])
        parts.append(df[df.target.notna()])
    tr = pd.concat(parts, ignore_index=True)
    cols = [x for x in tr.columns if x not in ("target", "series")]
    model = lgb.train(PARAMS, lgb.Dataset(tr[cols], tr.target, categorical_feature=["store", "family"]), rounds)
    return np.clip(model.predict(features(c)[cols]), 0, None)


def baseline(cutoff):
    """Mean of log1p sales on the same weekday over the last 12 weeks."""
    c = didx[pd.Timestamp(cutoff)]
    return np.concatenate([np.nanmean(Y[:, c + h - 7 * np.arange(1, 13)], axis=1) for h in range(1, H + 1)])


def truth(cutoff):
    c = didx[pd.Timestamp(cutoff)]
    return np.concatenate([Y[:, c + h] for h in range(1, H + 1)])


def zero_rule(cutoff, days):
    """True for series with no sales in the `days` days up to the cutoff."""
    c = didx[pd.Timestamp(cutoff)]
    return np.tile(np.nansum(np.expm1(Y[:, c - days + 1:c + 1]), axis=1) == 0, H)


def rmsle(p, y, mask=None):
    m = ~np.isnan(y) if mask is None else (mask & ~np.isnan(y))
    return float(np.sqrt(np.mean((p[m] - y[m]) ** 2)))


# %% [markdown]
# ## 1. Many windows instead of one
#
# Ten 16-day windows: one every four weeks through 2017 up to the usual holdout, plus
# the late-August window from 2016 (the same calendar window as the test set) and a
# window in early 2016. For each one the model is refit using only data before the
# window starts.

# %%
CUTOFFS = ["2016-01-19", "2016-08-15", "2017-01-23", "2017-02-20", "2017-03-20",
           "2017-04-17", "2017-05-15", "2017-06-12", "2017-07-10", "2017-07-30"]
results, preds = [], {}
for cut in CUTOFFS:
    t0 = time.time()
    p = forecast(cut)
    y = truth(cut)
    preds[cut] = p
    results.append(dict(cutoff=cut, window=f"{(pd.Timestamp(cut) + pd.Timedelta(days=1)).date()} to "
                                         f"{(pd.Timestamp(cut) + pd.Timedelta(days=H)).date()}",
                        model=rmsle(p, y), baseline=rmsle(baseline(cut), y)))
    print(f"{cut}: model {results[-1]['model']:.4f}  baseline {results[-1]['baseline']:.4f}  ({time.time() - t0:.0f}s)")
res = pd.DataFrame(results)

# %%
fig, ax = plt.subplots(figsize=(12, 4.5))
x = np.arange(len(res))
ax.bar(x - 0.2, res.baseline, width=0.38, color=GRAY, label="12-week same-weekday mean")
ax.bar(x + 0.2, res.model, width=0.38, color=BLUE, label="LightGBM")
ax.set_xticks(x, [w.replace(" to ", "\nto ") for w in res.window], fontsize=8)
ax.set_ylabel("RMSLE")
ax.set_title("The same model, refit and scored on ten different 16-day windows")
ax.legend(loc="upper left")
plt.show()
print(f"model RMSLE across windows: min {res.model.min():.3f}, max {res.model.max():.3f}, "
      f"std {res.model.std():.3f}")

# %% [markdown]
# Two things stand out.
#
# - The score moves a lot from one window to the next, even leaving aside the 2016
#   window (explained below). A single 16-day holdout is one draw from that spread, so
#   a gap of a few hundredths between a holdout and the leaderboard is not surprising
#   on its own.
# - The model beats the baseline in nine windows out of ten. The exception is the
#   window after 2017-02-20. The breakdown below shows what happened there.
#
# The practical rule: compare models on the *same* set of windows and look at the
# per-window differences, not at one absolute number.

# %%
# Which families drive the error in the three worst 2017 windows?
fam_all = np.tile(keys.family.values, H)
worst = res[res.cutoff.str.startswith("2017")].sort_values("model").cutoff.tail(3)
tab = {}
for cut in worst:
    y = truth(cut)
    m = ~np.isnan(y)
    se = pd.Series(((preds[cut] - y) ** 2)[m]).groupby(fam_all[m]).sum()
    tab[cut] = (se / se.sum()).sort_values(ascending=False).head(4).round(2)
for cut, t in tab.items():
    print(f"window after {cut}: share of squared error by family")
    print(t.to_string(), "\n")

# %% [markdown]
# Liquor, wine and beer dominate the two worst early-2017 windows. Ecuador bans
# alcohol sales around elections, and 2017 had two: the first round on 19 February
# and the runoff on 2 April.

# %%
lq = train[(train.family == "LIQUOR,WINE,BEER") & (train.date >= "2017-02-01") & (train.date <= "2017-04-15")]
lq = lq.groupby("date").sales.sum()
fig, ax = plt.subplots()
ax.bar(lq.index, lq.values / 1e3, color=BLUE, width=0.8)
ban = lq[lq == 0].index
ax.scatter(ban, np.zeros(len(ban)) + 0.3, color=ORANGE, marker="v", s=60, zorder=3, label="zero sales in every store")
ax.legend(loc="upper right")
for cut in ("2017-02-20", "2017-03-20"):
    a = pd.Timestamp(cut) + pd.Timedelta(days=1)
    ax.axvspan(a, a + pd.Timedelta(days=H - 1), color=GRAY, alpha=0.12)
ax.set_title("Liquor, wine and beer, all stores (thousands of units; shaded = backtest windows)")
plt.show()

# %% [markdown]
# On 18 and 19 February and on 1 and 2 April 2017, liquor sales are exactly zero in
# all 54 stores. The March window contains the runoff ban, which no feature in the
# model knows about. The February window starts the day after the first ban, so the
# model's most recent inputs (the last 3 and 7 days) include two days of zeros, and
# then Carnaval (27 and 28 February) pushes sales well above normal. The model has no
# holiday features, and its last-year features do not line up because Carnaval 2016
# was almost three weeks earlier.
#
# Neither event says much about model quality, but each moves a 16-day score by
# several hundredths. One window would never have revealed either.

# %% [markdown]
# ## 2. The 2016 window and two closed stores
#
# The 2016-08-16 to 08-31 window scores far worse than its neighbours. The reason is
# not the model.

# %%
st = train[(train.date >= "2016-07-20") & (train.date <= "2016-09-10")]
st = st.groupby(["date", "store_nbr"]).sales.sum().unstack()
fig, ax = plt.subplots()
for s_, col in ((18, BLUE), (25, ORANGE)):
    ax.plot(st.index, st[s_] / 1e3, color=col, label=f"store {s_}")
ax.axvspan(pd.Timestamp("2016-08-16"), pd.Timestamp("2016-08-31"), color=GRAY, alpha=0.12)
ax.set_title("Daily sales of stores 18 and 25 around the 2016 window (thousands of units)")
ax.legend()
plt.show()

# %%
cut = "2016-08-15"
y = truth(cut)
store_of = np.tile(keys.store_nbr.values, H)
open_mask = ~np.isin(store_of, [18, 25])
print(f"2016 window, all stores:        model {rmsle(preds[cut], y):.4f}   baseline {rmsle(baseline(cut), y):.4f}")
print(f"2016 window, without 18 and 25: model {rmsle(preds[cut], y, open_mask):.4f}   "
      f"baseline {rmsle(baseline(cut), y, open_mask):.4f}")
share = ((preds[cut] - y) ** 2)[~open_mask & ~np.isnan(y)].sum() / ((preds[cut] - y) ** 2)[~np.isnan(y)].sum()
print(f"share of squared error from the two closed stores: {share:.0%}")

# %% [markdown]
# Store 18 stopped selling on 15 August 2016 and store 25 on 22 August. Two stores out
# of 54 produce three quarters of the window's squared error, because predicting about
# log 5 for a day that turns out to be 0 costs 25 units of squared error per series.
#
# The same thing happens in reverse when a store opens: store 52 had never sold
# anything until 20 April 2017, inside the window after 2017-04-17. Its 33 series go
# from 0 to full sales, and the model, which has no history for them, forecasts about 0.
#
# No model can forecast a closure from sales history. Excluding them is the right way
# to compare models on this window, but the raw number is a useful reminder: the
# leaderboard window can contain events like this too, and a holdout without them will
# look better than the leaderboard.

# %% [markdown]
# ## 3. A choice that flips with the window: the zero rule
#
# Some series sell nothing for weeks. A popular trick is to forecast exactly zero for
# any series with no sales in the last N days. On the standard holdout, which N is
# right?

# %%
rows = []
for cut in CUTOFFS:
    y = truth(cut)
    m = None if cut != "2016-08-15" else open_mask
    r = {"cutoff": cut, "no rule": rmsle(preds[cut], y, m)}
    for days in (21, 56, 365):
        r[f"{days} days"] = rmsle(np.where(zero_rule(cut, days), 0, preds[cut]), y, m)
    rows.append(r)
z = pd.DataFrame(rows).set_index("cutoff")
diff = z[["21 days", "56 days", "365 days"]].sub(z["no rule"], axis=0)
print("Change in RMSLE from applying the rule (negative = better):")
print(diff.round(4).to_string())

# %%
fig, ax = plt.subplots(figsize=(12, 4))
xx = np.arange(len(diff))
TOP = 25
for i, (col, color) in enumerate(zip(diff.columns, (ORANGE, AQUA, BLUE))):
    v = diff[col].values * 1000
    ax.bar(xx + (i - 1) * 0.27, np.minimum(v, TOP), width=0.25, color=color, label=f"zero if no sales in {col}")
for j in np.where((diff.values * 1000).max(axis=1) > TOP)[0]:
    ax.text(xx[j], TOP + 0.8, f"up to +{(diff.values[j] * 1000).max():.0f}", ha="center", fontsize=8, color="#52514e")
ax.set_ylim(-3, TOP + 4)
ax.axhline(0, color=GRAY, lw=1)
ax.set_xticks(xx, diff.index, fontsize=8)
ax.set_ylabel("change in RMSLE x 1000")
ax.set_title("Zero rule: effect per window (below 0 helps; bars above 25 are cut off)")
ax.legend(loc="upper right", fontsize=9)
plt.show()

# %% [markdown]
# With a 21-day or 56-day window, the rule helps by a hair in a few windows and costs
# clearly in others (2017-03-20 and the standard 2017-07-30 holdout): a series that was
# quiet for three weeks often starts selling again when a product is restocked. With a
# 365-day window it only zeroes series that have been dead for a year, and it is
# neutral everywhere except one window, where it is a disaster: store 52, which opened
# on 20 April 2017, had no sales in the previous year, so the rule forced all of its
# opening-week forecasts to zero.
#
# So the right answer depends on the window. Tuned on the 2016-01-19 window alone, the
# 21-day rule looks as good as any; on 2017-04-17 every version looks terrible. The
# safe version is the 365-day rule plus a check that no whole store is being zeroed.
# At the real test cutoff (2017-08-15) no store is entirely dead: the 365-day rule only
# touches 65 single families, such as books in stores that never stocked them.

# %% [markdown]
# ## 4. What was different about the test window
#
# Before blaming the model, I checked the pipeline: features at the final cutoff end
# on 2017-08-15, test promotions join on (date, store, family), and submission ids
# match test.csv row for row. All of that was correct. Two things about the test
# window itself stood out.
#
# ### 4a. Back to school came early and big in 2017
#
# School and office supplies spike before the school year starts in the Sierra region.

# %%
sc = train[train.family == "SCHOOL AND OFFICE SUPPLIES"]
fig, ax = plt.subplots()
for yr, col in ((2015, GRAY), (2016, AQUA), (2017, BLUE)):
    s_ = sc[(sc.date >= f"{yr}-07-01") & (sc.date <= f"{yr}-09-30")].groupby("date").sales.sum()
    ax.plot(s_.index.dayofyear, s_.rolling(7, min_periods=1).mean(), color=col, label=str(yr))
ax.axvline(pd.Timestamp("2017-08-15").dayofyear, color=ORANGE, ls=":", lw=1.5)
ax.text(pd.Timestamp("2017-08-15").dayofyear + 1, ax.get_ylim()[1] * 0.9, "end of train", color="#52514e")
ax.set_xticks([pd.Timestamp(f"2017-{m:02d}-01").dayofyear for m in (7, 8, 9)], ["1 Jul", "1 Aug", "1 Sep"])
ax.set_title("School and office supplies, all stores, 7-day mean of daily units")
ax.legend()
plt.show()

# %% [markdown]
# In 2015 and 2016 the season built through late August and peaked in early
# September. In 2017 it started about a month earlier and peaked at roughly twice the
# 2016 level; in the first half of August it was 5 to 10 times higher than in the same
# weeks of 2016. The standard holdout (31 July to 15 August) already sits on the peak. A model that has
# learned "late August is higher than early August" from previous years will push the
# test forecast up further, while 2017's season may have been past its peak. The
# holdout cannot reveal this, because the holdout is the part of 2017 that went up.

# %% [markdown]
# ### 4b. A promotion spike on the first test day

# %%
pr = pd.concat([train[train.date >= "2017-06-01"], test]).groupby("date").onpromotion.sum()
fig, ax = plt.subplots()
ax.bar(pr.index, pr.values / 1e3, color=[ORANGE if d >= pd.Timestamp("2017-08-16") else BLUE for d in pr.index],
       width=0.8)
ax.set_title("Items on promotion per day, all stores (thousands; orange = test)")
plt.show()
print(f"2017-08-16: {pr['2017-08-16']:,.0f} items on promotion, "
      f"median of the 8 weeks before: {pr['2017-06-20':'2017-08-15'].median():,.0f}")

# %%
# How did sales respond to large family-level promotion spikes in the past?
g = train[train.date >= "2015-06-01"].assign(l=lambda d: np.log1p(d.sales))
g = g.groupby(["family", "date"]).agg(promo=("onpromotion", "sum"), ml=("l", "mean")).reset_index()
g = g.sort_values(["family", "date"])
for col in ("promo", "ml"):
    s_ = g.groupby("family")[col]
    g[col + "_ref"] = pd.concat([s_.shift(k) for k in (7, 14, -7, -14)], axis=1).mean(axis=1)
g = g[g.promo_ref >= 20].copy()
g["spike"] = pd.cut(g.promo / g.promo_ref, [0, 0.8, 1.25, 2, 3, 100],
                    labels=["< 0.8x", "0.8 to 1.25x", "1.25 to 2x", "2 to 3x", "> 3x"])
lift = g.assign(d=g.ml - g.ml_ref).groupby("spike", observed=True).d.agg(["mean", "median", "count"])
print("Mean log1p sales of a family vs. the same weekday 1 and 2 weeks around it,")
print("by how many items were on promotion relative to those days:")
print(lift.round(3).to_string())

# %% [markdown]
# On 2017-08-16 three times the usual number of items were on promotion. Historically,
# even when a family's promotions tripled, its mean log sales rose by only about 0.1 to
# 0.15 on average (median close to zero). A model that reads promotion counts at
# face value can extrapolate far beyond that for series whose count jumps well past
# anything in their recent history. Let's see what this compact model does.

# %%
p_test = forecast("2017-08-15")
X_test = features(didx[pd.Timestamp("2017-08-15")])
day = np.repeat(np.arange(1, H + 1), N)
fam_names = np.tile(keys.family.values, H)
d1 = pd.DataFrame({"family": fam_names, "day": day, "pred": p_test,
                   "promo_rel": X_test.promo_rel.values})
cmp_ = d1[d1.day.isin([1, 8])].pivot_table(index="family", columns="day", values=["pred", "promo_rel"])
cmp_["pred_jump"] = cmp_[("pred", 1)] - cmp_[("pred", 8)]
cmp_["promo_jump"] = cmp_[("promo_rel", 1)] - cmp_[("promo_rel", 8)]
out = cmp_[["pred_jump", "promo_jump"]].droplevel(1, axis=1).sort_values("pred_jump")
print("Forecast on the first test day (Wed 16 Aug) minus the following Wednesday (23 Aug):")
print(out.tail(8).round(2).to_string())

# %% [markdown]
# Families whose promotions jump the most on day 1 get the largest forecast jumps, far
# above the historical response. The holdout cannot catch this because nothing like
# this spike happens in it.
#
# ## What I changed, and what the leaderboard said
#
# The full pipeline has its own notebook. After this audit it got two changes, both
# judged only on two holdouts (the standard one and the 2016 late-August window
# without stores 18 and 25):
#
# | Version | Holdout A (2017-07-31 to 08-15) | Holdout B (2016-08-16 to 31, without stores 18 and 25) | Leaderboard |
# |---|---|---|---|
# | Same-weekday 12-week mean | 0.521 | 0.458 | 0.499 |
# | LightGBM blend | 0.383 | 0.381 | 0.420 |
# | LightGBM blend + 20% per-family linear model | 0.381 | 0.377 | 0.410 |
#
# The linear model (one ridge regression per family on the last 224 days: store level,
# store-by-weekday, paydays, holidays, promotions) never looks at last year, and it
# responds to promotions in a damped, linear way. Its gain on the leaderboard (0.010)
# was larger than on either holdout, which fits the two risks above. I also tried
# capping future promotion counts at each series' recent maximum; it made holdout A
# much worse (promotions were rising for real in late July 2017), so it was dropped
# even though it would have softened the day-1 spike.
#
# About 0.03 of the gap remains. Some of it is ordinary window-to-window variation
# (section 1) and some may be events that no sales history predicts (section 2).
#
# ## Takeaways
#
# - Backtest on several windows, and compare models window by window.
# - Look at the windows that score worst. Here the worst one was explained by two store
#   closures, not by the model.
# - Re-check simple rules (like zeroing quiet series) on every window before trusting
#   them.
# - Before submitting, compare the test forecast with recent history family by family.
#   Seasonal shifts and unusual inputs (like a promotion spike) show up there even when
#   every holdout looks fine.
