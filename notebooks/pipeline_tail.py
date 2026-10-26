fb, test = prepare(find_data_dir())

# %% [markdown]
# ## Validation on two holdouts
#
# Both holdouts mirror the test setup exactly: a cutoff day, models trained only on
# targets up to that day, and a 16-day forecast after it.
#
# - **A**: 2017-07-31 to 2017-08-15, the last 16 days of train, all stores.
# - **B**: 2016-08-16 to 2016-08-31, the same calendar window as the test set one
#   year earlier. Stores 18 and 25 are left out of the score because they closed
#   during the window, which no model can foresee from sales history.
#
# Every choice (training start, model types, blend weights, zero rule) was made on
# these holdouts, never on the leaderboard. A change is kept only if it improves
# both.
#
# | Setup | A | B |
# |---|---|---|
# | Same-weekday mean of the last 12 weeks (baseline) | 0.5208 | 0.4577 |
# | Direct LightGBM, training from 2016-06 | 0.3962 | |
# | Direct LightGBM, training from 2015-01 | 0.3838 | 0.3864 |
# | LightGBM blend (0.8 direct + 0.2 lag-16) | 0.3829 | 0.3809 |
# | Linear model alone (224 days, ridge 30, no trend) | 0.4163 | 0.4059 |
# | **0.8 LightGBM blend + 0.2 linear (this notebook)** | **0.3814** | **0.3765** |
#
# Run-to-run differences (seed, LightGBM version, thread count) move these scores by
# about 0.001, so smaller gains are not meaningful.
#
# Tried and rejected: zeroing series with no sales in the last 21 or 56 days (hurts:
# many come back), per-horizon models, smaller trees, and capping future promotion
# counts at the series' recent maximum (holdout A got much worse, because promotions
# were rising for real in late July 2017).
#
# Set `RUN_VALIDATION = False` to skip this step and only produce the submission.

# %%
RUN_VALIDATION = True


def holdout_score(cutoff, drop_stores=()):
    c = int(np.where(fb.dates == pd.Timestamp(cutoff))[0][0])
    X, p = forecast(fb, c)
    y = X.target.values
    keep = ~np.isin(fb.keys.store_nbr.values[X.series.values], drop_stores)
    return float(np.sqrt(np.mean((p[keep] - y[keep]) ** 2)))


if RUN_VALIDATION:
    print(f"holdout A RMSLE: {holdout_score('2017-07-30'):.5f}")
    print(f"holdout B RMSLE: {holdout_score('2016-08-15', (18, 25)):.5f}")

# %% [markdown]
# ## Final fit and submission
#
# Same procedure with the cutoff moved to 2017-08-15, the last day of train. The
# models now also learn from the 16 holdout days.

# %%
cut_test = int(np.where(fb.dates == pd.Timestamp("2017-08-15"))[0][0])
Xt, pt = forecast(fb, cut_test)

pred = pd.DataFrame({
    "date": fb.dates[Xt.cutoff.values + Xt.h.values.astype(int)],
    "store_nbr": fb.keys.store_nbr.values[Xt.series.values],
    "family": fb.keys.family.values[Xt.series.values],
    "sales": np.expm1(pt),
})
sub = test.merge(pred, on=["date", "store_nbr", "family"], how="left")[["id", "sales"]]
assert len(sub) == 28512 and sub.sales.notna().all()
sub.to_csv("submission.csv", index=False)
sub.describe()

# %% [markdown]
# ## Known risks in the test window
#
# Two things about 2017-08-16 to 08-31 are not covered by either holdout:
#
# - The 2017 back-to-school season (school and office supplies, Sierra region) started
#   about a month earlier than in 2015 and 2016 and peaked at about twice the 2016
#   level. Holdout A
#   sits on its peak. Last-year features expect late August to be higher still, which
#   may not hold in 2017.
# - On 2017-08-16 three times the usual number of items are on promotion. Historically,
#   even a tripling of a family's promotions lifted its mean log sales by only about
#   0.1 to 0.15, but LightGBM forecasts much larger jumps for families whose count goes
#   far past their recent range. Capping promotions did not survive validation, so the
#   forecast is left as is.
#
# The leaderboard scores the whole test set (there is no private split). Public
# leaderboard RMSLE: 0.41968 for the LightGBM blend alone, 0.41008 with the linear
# model added (this notebook). The linear model gained more on the leaderboard than on
# either holdout, which fits the two risks above: it does not use last year's numbers
# and it reacts to promotions in a damped, linear way.
