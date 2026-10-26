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

- Submission limit: 5 per day. Up to 2 final selections. Submissions older than two months drop off the leaderboard.
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
