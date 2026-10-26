# %% [markdown]
# # Store Sales EDA: what drives Favorita sales
#
# This notebook looks at the data for the
# [Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)
# competition with one question in mind: what should a forecasting model know about?
#
# Sections:
# 1. The data at a glance
# 2. Long-term trend and the 2016 earthquake
# 3. Weekly and monthly patterns (including paydays)
# 4. Product families and stores
# 5. Promotions
# 6. Holidays and closures
# 7. Oil price
# 8. Zero-sales series
# 9. What this means for a model
#
# The metric is RMSLE, so most charts use log1p(sales), which is the scale the
# metric cares about. A series means one (store, product family) pair. There are
# 54 x 33 = 1782 of them.

# %%
import glob
import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

hits = [h for h in glob.glob("/kaggle/input/**/train.csv", recursive=True)
        if os.path.exists(os.path.join(os.path.dirname(h), "holidays_events.csv"))]
DATA = os.path.dirname(hits[0]) if hits else "data"

# Palette: fixed order, colorblind-checked. Text stays in neutral inks.
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

train = pd.read_csv(f"{DATA}/train.csv", parse_dates=["date"])
test = pd.read_csv(f"{DATA}/test.csv", parse_dates=["date"])
stores = pd.read_csv(f"{DATA}/stores.csv")
oil = pd.read_csv(f"{DATA}/oil.csv", parse_dates=["date"])
hol = pd.read_csv(f"{DATA}/holidays_events.csv", parse_dates=["date"])
trans = pd.read_csv(f"{DATA}/transactions.csv", parse_dates=["date"])
train["log_sales"] = np.log1p(train.sales)

# %% [markdown]
# ## 1. The data at a glance

# %%
print(f"train: {len(train):,} rows, {train.date.min().date()} to {train.date.max().date()}")
print(f"test:  {len(test):,} rows, {test.date.min().date()} to {test.date.max().date()}")
print(f"stores: {train.store_nbr.nunique()}, families: {train.family.nunique()}, "
      f"series: {train.groupby(['store_nbr', 'family']).ngroups}")
missing = pd.date_range(train.date.min(), train.date.max()).difference(train.date.unique())
print("dates missing from train:", [d.date().isoformat() for d in missing])
print(f"rows with zero sales: {(train.sales == 0).mean():.1%}")
stores.head()

# %% [markdown]
# Every Dec 25 is missing: stores are closed. Almost a third of all rows are zero,
# so zeros deserve their own section later.

# %% [markdown]
# ## 2. Long-term trend and the 2016 earthquake

# %%
daily = train.groupby("date").sales.sum()
fig, ax = plt.subplots()
ax.plot(daily.index, daily / 1e3, color=GRAY, lw=0.6, alpha=0.6, label="daily total")
ax.plot(daily.index, daily.rolling(28, center=True).mean() / 1e3, color=BLUE, lw=2, label="28-day mean")
ax.axvspan(pd.Timestamp("2016-04-16"), pd.Timestamp("2016-05-31"), color=ORANGE, alpha=0.15)
ax.text(pd.Timestamp("2016-04-20"), ax.get_ylim()[1] * 0.95, "earthquake\n16 Apr 2016", color="#52514e", va="top")
ax.set_title("Total daily sales, all stores (thousands of units)")
ax.legend(loc="upper left")
plt.show()

# %% [markdown]
# Sales grow steadily over four and a half years. The dips to near zero are Jan 1
# each year. After the 7.8 earthquake on 16 April 2016, sales jump for a few weeks
# as people bought water and supplies for relief. That period is unusual, so a model
# should drop or downweight it rather than learn from it.

# %%
# Family lift in the month after the quake vs. the 4 weeks before, minus the same
# lift one year earlier. Subtracting 2015 removes normal April seasonality (the
# school year starts in April on the coast, for example).
def lift_for(year):
    a = train[(train.date >= f"{year}-03-19") & (train.date <= f"{year}-04-15")]
    b = train[(train.date >= f"{year}-04-16") & (train.date <= f"{year}-05-15")]
    return np.log(b.groupby("family").sales.mean() / a.groupby("family").sales.mean())


excess = (lift_for(2016) - lift_for(2015)).dropna().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(12, 4))
top = excess.head(8)
ax.barh(top.index[::-1], top.values[::-1], color=BLUE, height=0.6)
for y, v in enumerate(top.values[::-1]):
    ax.text(v + 0.01, y, f"{np.expm1(v):+.0%}", va="center", color="#52514e")
ax.set_title("Families with the largest extra lift after the earthquake (vs. the same weeks in 2015)")
ax.set_xlabel("log change in mean sales, 2016 minus 2015")
plt.show()

# %% [markdown]
# ## 3. Weekly and monthly patterns

# %%
recent = train[train.date >= "2016-08-01"].copy()
recent["dow"] = recent.date.dt.dayofweek
by_dow = recent.groupby(["date", "dow"]).sales.sum().groupby("dow").mean()
by_dow = by_dow / by_dow.mean()
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
ax = axes[0]
ax.bar(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], by_dow.values, color=BLUE, width=0.6)
ax.axhline(1, color=GRAY, lw=1, ls="--")
ax.set_title("Day of week (1.0 = average day)")
recent["dom"] = recent.date.dt.day
by_dom = recent.groupby(["date", "dom"]).sales.sum().groupby("dom").mean()
ax = axes[1]
ax.plot(by_dom.index, by_dom / by_dom.mean(), color=BLUE, marker="o", ms=4)
for d in (1, 15, 16):
    ax.axvline(d, color=ORANGE, lw=1, ls=":")
ax.set_title("Day of month (1.0 = average day)")
ax.set_xlabel("day of month; dotted lines at 1, 15 and 16")
plt.tight_layout()
plt.show()

# %% [markdown]
# The weekly cycle is the strongest short-term pattern: weekends sell about 20 to
# 30 percent more than midweek. Public sector wages are paid on the 15th and the
# last day of the month, and sales are higher in the first days of the month and
# just after the 15th. Day-of-week and payday flags are cheap features with a
# clear effect.

# %%
fam_dow = recent.groupby(["family", "dow"]).sales.mean().unstack()
fam_dow = fam_dow.div(fam_dow.mean(axis=1), axis=0)
fam_dow = fam_dow.loc[fam_dow.max(axis=1).sort_values().index]
fig, ax = plt.subplots(figsize=(8, 10))
im = ax.imshow(fam_dow.values, cmap="Blues", aspect="auto", vmin=0.6, vmax=1.6)
ax.set_xticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
ax.set_yticks(range(len(fam_dow)), fam_dow.index, fontsize=8)
ax.grid(False)
fig.colorbar(im, ax=ax, shrink=0.5, label="sales vs. family average")
ax.set_title("Weekly profile differs by family")
plt.show()

# %% [markdown]
# The weekly shape is not the same everywhere. Liquor and some home goods peak much
# harder on weekends than staples do. A global model needs family (and store) as
# features so it can learn a different weekly profile per series, or it needs
# per-series day-of-week averages as inputs.

# %% [markdown]
# ## 4. Product families and stores

# %%
share = train[train.date >= "2017-01-01"].groupby("family").sales.sum().sort_values()
share = share / share.sum() * 100
fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(share.index, share.values, color=BLUE, height=0.65)
for y, v in enumerate(share.values):
    if v >= 1:
        ax.text(v + 0.3, y, f"{v:.1f}%", va="center", color="#52514e", fontsize=8)
ax.set_title("Share of 2017 unit sales by family")
ax.tick_params(axis="y", labelsize=8)
plt.show()

# %% [markdown]
# Five families (grocery, beverages, produce, cleaning, dairy) make up about three
# quarters of unit sales. RMSLE is computed on log1p, though, so every series
# counts roughly equally regardless of volume. Getting the small families right
# (books, baby care, hardware) matters as much as getting grocery right.

# %%
st = train[train.date >= "2017-01-01"].groupby("store_nbr").sales.sum().rename("sales").reset_index()
st = st.merge(stores, on="store_nbr")
order = st.groupby("type").sales.median().sort_values().index
fig, ax = plt.subplots(figsize=(10, 4))
for i, t in enumerate(order):
    v = st.loc[st.type == t, "sales"] / 1e6
    ax.scatter(np.full(len(v), i) + np.random.default_rng(0).uniform(-0.12, 0.12, len(v)), v,
               color=BLUE, s=36, alpha=0.8, edgecolor="#fcfcfb", linewidth=1)
    ax.plot([i - 0.25, i + 0.25], [v.median()] * 2, color="#0b0b0b", lw=2)
ax.set_xticks(range(len(order)), [f"type {t}" for t in order])
ax.set_title("2017 sales per store by store type (millions of units; bar = median)")
plt.show()

# %% [markdown]
# Store type separates big and small stores (type A is largest), but there is a lot
# of spread inside each type. Store number itself is a more precise feature.

# %%
first = train[train.sales > 0].groupby("store_nbr").date.min().sort_values()
print("Stores whose first sale is after 2013-01-02:")
print(first[first > "2013-01-02"].dt.date.to_string())

# %% [markdown]
# Several stores open part way through the history, and before that they show
# zero sales. Those zeros are "not open yet", not "no demand", and should not be
# treated as real observations.

# %% [markdown]
# ## 5. Promotions

# %%
pm = train.groupby("date").onpromotion.sum()
fig, ax = plt.subplots()
ax.plot(pm.index, pm.rolling(7).mean() / 1e3, color=BLUE)
ax.set_title("Items on promotion per day, all stores (thousands, 7-day mean)")
plt.show()

# %% [markdown]
# Promotion data is all zero until spring 2014 and grows a lot afterwards. Any
# model trained on the early years sees "no promotions" that really means "not
# recorded".

# %%
# Compare promoted and non-promoted days within the same series, so big stores
# (which run more promotions) do not inflate the effect. Only days with sales > 0
# count, so store closures and not-yet-stocked families do not either.
r = train[(train.date >= "2016-06-01") & (train.sales > 0)]
within = (r.assign(promo=r.onpromotion > 0)
          .groupby(["store_nbr", "family", "promo"]).log_sales.mean().unstack().dropna())
within = (within[True] - within[False]).groupby("family").median().sort_values()
fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(within.index, within.values, color=[BLUE if v >= 0 else ORANGE for v in within.values], height=0.65)
ax.axvline(0, color=GRAY, lw=1)
ax.set_title("Promotion lift within a series: promoted minus non-promoted days")
ax.set_xlabel("median across stores of the difference in mean log1p(sales); 0.69 means roughly double")
ax.tick_params(axis="y", labelsize=8)
plt.show()

# %% [markdown]
# Even within one series, days with at least one item on promotion sell more in
# almost every family. This is still not a clean causal effect (promotions are
# planned around busy periods), but the direction is consistent. Still, the promotion count for the target day is
# known in advance (it is in test.csv), which makes it one of the few features
# that tells the model something about the future itself. Comparing it with the
# series' usual promotion level is more useful than the raw count.

# %%
s = train[(train.date >= "2016-06-01") & (train.store_nbr == 44) & (train.family == "BEVERAGES")]
fig, ax = plt.subplots()
ax.scatter(s.onpromotion, s.sales, color=BLUE, s=14, alpha=0.6)
ax.set_xlabel("items on promotion")
ax.set_ylabel("sales")
ax.set_title("One series up close: store 44, beverages, since June 2016")
plt.show()

# %% [markdown]
# ## 6. Holidays and closures
#
# `holidays_events.csv` needs care:
# - **Transferred** holidays (`transferred=True`) were moved to another day. On the
#   original date it was a normal working day. The day it was moved to appears as
#   a separate row with type **Transfer**.
# - **Bridge** days are extra days off added to make a long weekend. **Work Day**
#   rows are weekend days that become working days to pay back a bridge.
# - **Additional** days extend a holiday, for example the days around Christmas.
# - Locale matters: **National** holidays apply to all stores, **Regional** ones to
#   stores in that state, **Local** ones to stores in that city.

# %%
print(hol.type.value_counts().to_string())
print()
print(hol.locale.value_counts().to_string())
hol[hol.transferred | (hol.type == "Transfer")].sort_values("date").head(8)

# %%
# Resolve to (store, date): which stores actually had a holiday on each day
rows = []
for r_ in hol[~hol.transferred & hol.type.isin(["Holiday", "Transfer", "Additional", "Bridge"])].itertuples():
    if r_.locale == "National":
        st_ = stores.store_nbr
    elif r_.locale == "Regional":
        st_ = stores.store_nbr[stores.state == r_.locale_name]
    else:
        st_ = stores.store_nbr[stores.city == r_.locale_name]
    rows += [(r_.date, s_, r_.locale) for s_ in st_]
store_hol = pd.DataFrame(rows, columns=["date", "store_nbr", "locale"]).drop_duplicates(["date", "store_nbr"])

# Effect: store sales on a holiday vs. the same store's mean on the same weekday in
# the 4 weeks before and after (excluding other holidays)
ss = train[train.date >= "2014-01-01"].groupby(["store_nbr", "date"]).sales.sum().reset_index()
ss = ss.merge(store_hol, on=["date", "store_nbr"], how="left")
ss["dow"] = ss.date.dt.dayofweek
ss = ss.sort_values(["store_nbr", "dow", "date"])
ref = ss.assign(v=ss.sales.where(ss.locale.isna()))
ref["base"] = (ref.groupby(["store_nbr", "dow"]).v
               .transform(lambda x: x.shift(1).rolling(4, min_periods=2).mean() / 2
                          + x.shift(-1)[::-1].rolling(4, min_periods=2).mean()[::-1] / 2))
eff = ref[ref.locale.notna() & (ref.base > 0)].assign(ratio=lambda d: np.log(d.sales.clip(lower=1) / d.base))
eff = eff[~((eff.date.dt.month == 1) & (eff.date.dt.day == 1))]
res = eff.groupby("locale").ratio.agg(["median", "count"])
print("Median log ratio of store sales on holidays vs. normal same-weekday sales "
      "(Jan 1 excluded):")
print(res.round(3).to_string())

# %%
nat = hol[(hol.locale == "National") & ~hol.transferred & (hol.type != "Event")].drop_duplicates("date")
top_h = (eff[eff.locale == "National"].merge(nat[["date", "description"]], on="date")
         .groupby("description").ratio.median().sort_values())
fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(top_h.index, top_h.values, color=[BLUE if v >= 0 else ORANGE for v in top_h.values], height=0.65)
ax.axvline(0, color=GRAY, lw=1)
ax.set_title("National holidays: median log ratio of store sales vs. normal days")
ax.tick_params(axis="y", labelsize=8)
plt.show()

# %% [markdown]
# Holidays do not have one direction. Days before Christmas sell far above normal;
# Jan 1 (excluded above) is close to zero because nearly every store is closed;
# some single-day holidays look like a busy weekend. A model should get separate
# flags (national, regional, local, bridge, additional, work day) rather than one
# "holiday" column, and it should get them per store after resolving locale.

# %%
ny = train[(train.date.dt.month == 1) & (train.date.dt.day == 1)]
open_ = ny.groupby(["date", "store_nbr"]).sales.sum().gt(0).groupby("date").sum()
print("Stores with any sales on Jan 1:")
print(open_.to_string())

# %% [markdown]
# Jan 1 is effectively a closure day and Dec 25 is missing entirely. Both are best
# removed from training targets and from rolling averages, since the test window
# (16 to 31 August) contains neither.

# %% [markdown]
# ## 7. Oil price
#
# Ecuador's economy depends on oil, so the data includes the daily WTI oil price. It
# has gaps on weekends and some weekdays; forward fill is the natural fix.

# %%
o = oil.set_index("date").dcoilwtico.reindex(pd.date_range(oil.date.min(), oil.date.max()))
print(f"oil: {o.isna().sum()} missing days out of {len(o)} before filling")
o = o.ffill().bfill()
monthly_sales = daily.resample("MS").mean()
monthly_oil = o.resample("MS").mean().reindex(monthly_sales.index)
fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
axes[0].plot(monthly_oil.index, monthly_oil, color=BLUE)
axes[0].set_title("Oil price, monthly mean (USD per barrel)")
axes[1].plot(monthly_sales.index, monthly_sales / 1e3, color=BLUE)
axes[1].set_title("Total daily sales, monthly mean (thousands of units)")
plt.tight_layout()
plt.show()

# %% [markdown]
# Oil fell from about 100 to about 40 dollars during late 2014 and 2015, and sales
# growth slowed around the same time. The relationship is slow and indirect. For a
# 16-day forecast, oil price adds little beyond what recent sales already say, which
# matches what tree models usually report as low feature importance.

# %% [markdown]
# ## 8. Zero-sales series

# %%
last = train[train.date > train.date.max() - pd.Timedelta(days=21)]
dead = last.groupby(["store_nbr", "family"]).sales.sum().eq(0)
dead_year = (train[train.date > train.date.max() - pd.Timedelta(days=365)]
             .groupby(["store_nbr", "family"]).sales.sum().eq(0))
print(f"series with no sales in the last 21 days:  {dead.sum()}")
print(f"series with no sales in the last 365 days: {dead_year.sum()}")
print()
print("Families with the most series that sold nothing in the last 21 days:")
print(dead[dead].reset_index().family.value_counts().head(8).to_string())

# %%
z = train.groupby("date").sales.apply(lambda x: (x == 0).mean())
fig, ax = plt.subplots()
ax.plot(z.index, z.rolling(7).mean() * 100, color=BLUE)
ax.set_title("Share of series with zero sales (%, 7-day mean)")
plt.show()

# %% [markdown]
# The share of zero rows falls from about half in 2013 to around 15 percent in 2017,
# as stores open and families like books, baby care and celebration get stocked.
# The square jumps in 2014 are whole families switching between recorded and
# all-zero for weeks at a time, which looks like a data recording issue rather than
# real demand. Another reason to start training in 2015 or later.
#
# A common trick is to forecast zero for every series that sold nothing recently.
# On a 16-day holdout ending 2017-08-15 this only helped when "recently" meant a full
# year. With a 21-day window some of the zeroed series came back with sales and the
# score got worse. Check this on your own validation split instead of assuming it.

# %% [markdown]
# ## 9. What this means for a model
#
# - Work on log1p(sales) and use squared error: that is exactly what RMSLE rewards.
# - Recent level (last 1 to 4 weeks) and per-series day-of-week averages carry most
#   of the signal for a 16-day horizon.
# - Promotions on the target day are known in advance; use them, relative to the
#   series' normal promotion level.
# - Resolve holidays per store by locale and keep the types separate. Treat
#   transferred holidays as normal days.
# - Remove Jan 1 and Dec 25 from targets and rolling statistics, and drop or
#   downweight the weeks after the April 2016 earthquake.
# - Ignore the zeros before a store opens.
# - Promotion data only starts in 2014, so training from 2015 or later loses little.
