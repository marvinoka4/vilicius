# %% [markdown]
# ## Forecast function
#
# The final forecast blends two LightGBM models:
# - **Direct model**: trained on cutoffs every 7 days since `MAIN_START`, all 16
#   horizons in one model with `h` as a feature. Every forecast day sees the most
#   recent data.
# - **Lag-16 model**: each target day uses features from 16 days before it, so all
#   lags are at least 16 days old. Weaker on its own, but its errors differ from the
#   direct model's, which makes the blend better than either.
#
# After blending, predictions are clipped at 0 and series that sold nothing in the
# 365 days before the cutoff are set to 0.

# %%
CONFIG = dict(
    main_start="2015-01-01", main_stride=7, main_rounds=800,
    lag16_start="2016-06-01", lag16_rounds=800,
    lag16_weight=0.2, zero_days=365,
)


def forecast(fb, cut0, cfg, seed=0):
    """Forecast the 16 days after cutoff index `cut0`. Returns log1p predictions in
    the row order of fb.build(cut0): horizon blocks of all series."""
    t = time.time()
    train = make_train(fb, cut0, pd.Timestamp(cfg["main_start"]), cfg["main_stride"])
    X = fb.build(cut0)
    print(f"direct model: {len(train):,} training rows", flush=True)
    p_main = fit_predict(train, X, cfg["main_rounds"], seed=seed)[0]
    del train

    start2 = pd.Timestamp(cfg["lag16_start"])
    train = make_lag16(fb, start2 + pd.Timedelta(days=HORIZON), fb.dates[cut0], start=start2)
    X2 = make_lag16(fb, fb.dates[cut0 + 1], fb.dates[cut0 + HORIZON]).assign(h=HORIZON)
    print(f"lag-16 model: {len(train):,} training rows", flush=True)
    p_lag = fit_predict(train, X2, cfg["lag16_rounds"], seed=seed)[0]
    del train

    w = cfg["lag16_weight"]
    p = np.clip((1 - w) * p_main + w * p_lag, 0, None)
    dead = zero_rule(fb.Y, cut0, cfg["zero_days"])[X.series.values]
    p[dead] = 0
    print(f"forecast done in {time.time() - t:.0f}s, {dead.sum() // HORIZON} series set to zero")
    return X, p


fb, test = prepare(find_data_dir())

# %% [markdown]
# ## Validation
#
# The holdout mirrors the test setup exactly: the last 16 days of train
# (2017-07-31 to 2017-08-15) are hidden, the cutoff is 2017-07-30, and the models
# are trained only on targets up to that cutoff. All choices in this notebook
# (training start, model types, blend weight, zero rule) were made on this holdout,
# never on the leaderboard.
#
# Holdout results that led here (RMSLE):
#
# | Setup | Holdout |
# |---|---|
# | Same-weekday mean of the last 12 weeks (baseline) | 0.5208 |
# | Direct model, training from 2017-01 | 0.4050 |
# | Direct model, training from 2016-06 | 0.3962 |
# | Lag-16 model, training from 2016-06 | 0.3944 |
# | Direct model, training from 2015-01 | 0.3838 |
# | Blend: 0.8 direct (2015) + 0.2 lag-16 (2016-06) | 0.3829 |
#
# Set `RUN_VALIDATION = False` to skip this step and only produce the submission.

# %%
RUN_VALIDATION = True

if RUN_VALIDATION:
    cut_val = int(np.where(fb.dates == pd.Timestamp("2017-07-30"))[0][0])
    Xv, pv = forecast(fb, cut_val, CONFIG)
    yv = Xv.target.values
    print(f"holdout RMSLE: {np.sqrt(np.mean((pv - yv) ** 2)):.5f}")
    by_h = [np.sqrt(np.mean((pv[Xv.h.values == h] - yv[Xv.h.values == h]) ** 2)) for h in range(1, HORIZON + 1)]
    print("by horizon:", " ".join(f"{v:.3f}" for v in by_h))

# %% [markdown]
# ## Final fit and submission
#
# Same procedure with the cutoff moved to 2017-08-15, the last day of train. The
# models now also learn from the 16 holdout days.

# %%
cut_test = int(np.where(fb.dates == pd.Timestamp("2017-08-15"))[0][0])
Xt, pt = forecast(fb, cut_test, CONFIG)

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
