# Store Sales - Time Series Forecasting (Kaggle)

Competition: `store-sales-time-series-forecasting` (Getting Started, rolling leaderboard).
Metric: RMSLE. Target: `sales` per (date, store_nbr, family), 2017-08-16 to 2017-08-31,
54 stores x 33 families = 1782 series, 28512 test rows.

## Goals

1. A strong leaderboard score.
2. Useful, readable Kaggle notebooks (EDA and final pipeline) that can earn upvotes.

## House rules

- No em dashes, no emoji headings, no filler words. Plain English everywhere.
- No AI attribution in commits, notebooks, or Kaggle titles/descriptions.
- Work on branch `work/store-sales`; never merge to main.
- `data/` and large outputs are git-ignored.
- Everything pushed to Kaggle stays private until Marvin reviews it.
- Never tune on the leaderboard. Decisions come from the holdout below.

## Competition constraints (from the rules and data pages, checked 2026-10-26)

- Submission limit: 5 per day.
- Leaderboard (from the Overview page, which governs this Getting Started competition): the whole test set is scored on the public leaderboard and there is no private leaderboard, so final submission picks do not matter. Rolling window of three months: a submission older than three months stops counting, and a team with no submission in three months drops off. The generic rules text still says "two months" and "up to 2 final submissions"; the Overview page is the specific, current statement.
- Because the public leaderboard scores all 16 test days, the LB number is the full test score, not a noisy sample.
- Rules acceptance is not required for this competition. Submitting by CSV upload is allowed (not kernels-only).
- Data license: "Subject to Competition Rules". Section 7.A allows non-commercial use only. Section 7.B: "You agree not to transmit, duplicate, publish, redistribute or otherwise provide or make available the Competition Data to any party not participating in the Competition."
- Consequence: no public Kaggle dataset built from the competition files. A per-store holiday table and gap-filled oil series are direct reshapes of `holidays_events.csv` and `oil.csv`, so publishing them would be redistribution. Notebooks that compute these features from the competition data are fine (public code sharing on Kaggle is allowed under 8.B).
- External data is allowed if publicly and freely available to all participants (7.C). Not used so far.

## Plan

1. Download data to `data/`.
2. Baselines: zero-forecast for dead series, day-of-week weighted mean of recent weeks. One LB submission.
3. LightGBM on log1p(sales), global model, features below. No feature needs data inside the forecast window.
4. Variants: direct per-horizon vs single model with lags >= 16; training start dates. Blend only if holdout improves.
5. Refit on all train, predict test, clip at 0, zero out dead series.
6. Kaggle notebook (private) that runs end to end and writes `submission.csv`; submit from it.
7. EDA notebook (private). Final pipeline notebook (private).

## Validation

- Holdout: 2017-07-31 to 2017-08-15 (16 days, same length as test). Train on data up to 2017-07-30.
- Score: RMSLE over all 1782 series x 16 days, predictions clipped at 0.
- Features for the holdout are built as if the cutoff were 2017-07-30, so the holdout mirrors the test setup exactly.

## Features (main model)

- Static: store_nbr, family, store type, cluster, city, state.
- Calendar: day of week, day of month, month, payday flags (15th, last day of month, and the day after).
- Promo: onpromotion on the day, promo aggregates over the forecast window and recent history.
- Lags/rolling: computed only from data at or before the cutoff date.
- Oil: daily price, forward/back filled across weekends and gaps.
- Holidays resolved per store: national to all stores, regional by state, local by city. Transferred rows are treated as normal days; Transfer, Bridge, Additional rows are holidays; Work Day is a separate flag.
- Earthquake (2016-04-16 to 2016-05-15): excluded or downweighted. Jan 1 and Dec 25: stores closed, rows dropped from training.

## Experiment log

| Date | Change | Holdout RMSLE | LB RMSLE |
|---|---|---|---|
| 2026-10-26 | Zero rule (no sales in last 21 days) + 14-day mean of log1p | 0.53612 | |
| 2026-10-26 | Zero rule + same-weekday mean of log1p over last 12 weeks (decay 1.0) | 0.52084 | 0.49918 (CSV upload, sub 57664521) |
| 2026-10-26 | LightGBM direct model (h as feature), cutoffs every 7 days from 2016-06, 255 leaves, 800 rounds, no zero rule | 0.39713 | |
| 2026-10-26 | Same, zero rule tested by window: 21d 0.40351, 56d 0.40155, 180d 0.40164, 365d 0.39622. Short windows hurt (series come back); keep 365d | 0.39622 | |
| 2026-10-26 | Direct model, training from 2015-01 (365d zero rule) | 0.38380 | |
| 2026-10-26 | Direct model, training from 2017-01 | 0.40504 | |
| 2026-10-26 | Per-horizon models (16), from 2016-06, 600 rounds | 0.42079 | |
| 2026-10-26 | Lag-16 single model, from 2016-06 | 0.39438 | |
| 2026-10-26 | Lag-16 single model, from 2015-01 | 0.39613 | |
| 2026-10-26 | Direct from 2015, num_leaves 63, min_data 300, 1500 rounds | 0.38640 | |
| 2026-10-26 | Direct from 2015, cutoff stride 3 (7.6M rows) | 0.38352 | |
| 2026-10-26 | Blend 0.8 direct 2015 + 0.2 lag-16 2016-06 (weights 0.2 to 0.5 tried; 0.2 best) | 0.38287 | |
| 2026-10-26 | Final pipeline on Kaggle (notebook v1/v2). Same code, Kaggle's LightGBM build gives a slightly different holdout; run-to-run noise is about 0.001 | 0.38414 (Kaggle) | 0.41968 (notebook v2 output, sub 57667115) |
| 2026-10-26 | Extra backtest, cutoff 2016-08-15 (same calendar window as test, one year earlier), direct model from 2015. Stores 18 and 25 closed during the window (unforecastable). Excluding stores 18, 25, 52: model 0.3864 vs baseline 0.4577; school supplies 0.604 vs 1.359. Not used for tuning, only as a sanity check | 0.3864 (open stores) | |

Holdouts from here on: A = cutoff 2017-07-30 (all stores), B = cutoff 2016-08-15 excluding stores 18 and 25 (both closed during that window). A change is kept only if it improves both. Scores below are "A / B".

| Date | Change | Holdout A / B | LB RMSLE |
|---|---|---|---|
| 2026-10-26 | Gap audit (src/audit.py): features at the test cutoff end on 2017-08-15 (last, mean_7, mean_112, promo_sum_14 checked against raw data); test onpromotion joins on (date, store_nbr, family) and promo_future_sum matches; oil on target days matches the filled oil.csv; submission ids equal test.csv and sample_submission row for row. No pipeline bug found | | |
| 2026-10-26 | Reference: LightGBM blend (0.8 direct + 0.2 lag-16), 365d zero rule | 0.38287 / 0.38094 | 0.41968 |
| 2026-10-26 | Per-family ridge on log1p, 112 days, store level + trend + store x dow + payday + holidays + centred log promo, ridge also on store levels (bug: shrinks levels toward 0) | 0.52528 / 0.56474 | |
| 2026-10-26 | Same, store levels unpenalised. Windows 112/168/224/280/364, alpha 3/30/100, trend on/off tried; 224 days, alpha 30, no trend is best overall | alone 0.41631 / 0.40588 | |
| 2026-10-26 | Blend 0.8 LightGBM + 0.2 linear (224, 30, no trend). Weight 0.1 to 0.4 tried; 0.2 best on A, 0.3 slightly better on B; kept 0.2 | 0.38137 / 0.37650 | |
| 2026-10-26 | Same blend run on Kaggle (pipeline notebook v4): A 0.38227 (Kaggle's v2 reference was 0.38414), B 0.37650 | 0.38227 / 0.37650 (Kaggle) | 0.41008 (notebook v4 output, sub 57671205) |
| 2026-10-26 | Linear weight 0.3 instead of 0.2: A 0.38221, B 0.37607. Worse than 0.2 on A, so not submitted | 0.38221 / 0.37607 | |
| 2026-10-26 | Promo cap: target-day promo features capped at 1.5x the series' max over the last 56 days. Diagnostic holdout C (cutoff 2016-01-19, contains the largest promo spikes) 0.39053 -> 0.38946 | 0.41015 / 0.38136 (rejected) | |

Notes on the leaderboard gap (holdout 0.383 vs LB 0.420; the LB scores the whole test set):
- No pipeline bug (see audit row).
- The 2017 back-to-school season (school and office supplies, Sierra) started about a month earlier than in 2015 and 2016 and peaked at about twice the 2016 level. In early August it was 5 to 10 times the same weeks of 2016 (about 23k units/week in the Sierra vs 1 to 2k). Holdout A sits on the peak; last-year features push the test forecast for late August up further, which may not hold.
- 2017-08-16 (test day 1) has 30,539 items on promotion vs a median of about 12k. Historically a family whose promotions tripled gained only about +0.15 mean log1p (median about 0). The model forecasts day 1 about +0.2 above the following Wednesday overall and +0.4 to +1.8 for families whose promotions jump 4 to 5x (lingerie, lawn and garden, school supplies). The promo cap that would address this failed holdout A, so it is documented, not shipped.
- In the 2016 late-August window, store closures (18, 25) made up 75 percent of the squared error. The test window can contain events like that too; nothing in the training data can predict them.
- Backtesting a compact model over ten 16-day windows (Notebook C): RMSLE ranges from 0.39 to 0.58 across 2017 windows, so a single holdout is a noisy estimate. The bad windows are explained by events: store closures (2016-08), store 52 opening (2017-04), election alcohol bans on 2017-02-18/19 and 2017-04-01/02 plus Carnaval.
- Adding the linear model gained 0.010 on the LB but only 0.0015 / 0.0044 on the holdouts. Consistent with the test window punishing LightGBM's last-year and promo extrapolation, which the linear model does not do. Remaining gap about 0.03.

## Submissions

| Date | Source | Public LB |
|---|---|---|
| 2026-10-26 | CSV upload, baseline (12-week same-weekday mean) | 0.49918 |
| 2026-10-26 | Kernel output, marvinoka4/store-sales-direct-multi-horizon-lightgbm v2 | 0.41968 |
| 2026-10-26 | Kernel output, marvinoka4/store-sales-direct-multi-horizon-lightgbm v4 (LightGBM blend + 0.2 linear) | 0.41008 |

## Kaggle assets (all private)

- EDA notebook: https://www.kaggle.com/code/marvinoka4/store-sales-eda-what-drives-favorita-sales
- Pipeline notebook: https://www.kaggle.com/code/marvinoka4/store-sales-direct-multi-horizon-lightgbm
- Backtesting notebook: https://www.kaggle.com/code/marvinoka4/backtesting-store-sales-one-holdout-is-not-enough (Kaggle caps titles at 50 characters, so the Kaggle title is "Backtesting Store Sales: One Holdout Is Not Enough"; the notebook heading keeps the full title)
- Dataset: not created. The rules (section 7.B) forbid redistributing the Competition Data to non-participants, and a per-store holiday table or gap-filled oil series is a direct reshape of competition files. The pipeline notebook computes these features from the competition data instead.
