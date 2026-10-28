# vilicius
 
Forecasting daily grocery sales for Kaggle's [Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting) competition, using data from Corporación Favorita, a large Ecuadorian grocery retailer.
 
The task is to predict unit sales for 54 stores and 33 product families over the 16 days from 2017-08-16 to 2017-08-31. Submissions are scored with RMSLE (root mean squared logarithmic error).
 
## Results
 
| | RMSLE |
|---|---|
| Holdout A (2017-07-31 to 2017-08-15) | 0.382 |
| Holdout B (2016-08-16 to 2016-08-31, stores 18 and 25 excluded) | 0.377 |
| Public leaderboard | **0.41008** |
| Day-of-week baseline, public leaderboard | 0.49918 |
 
The leaderboard scores the full test set, so 0.41008 is the final score for this model.
 
## Approach
 
1. **Target.** The model predicts `log1p(sales)`, so ordinary RMSE in log space matches the competition metric.
2. **Main model.** A global LightGBM model forecasts all 16 days directly from the last known day, trained on data from 2015 onward. Features cover store attributes, calendar effects (weekday, payday on the 15th and month end), promotions, lagged and rolling sales, filled-in oil prices, and holidays matched to each store by locale (national, regional, local), including transferred and bridge days.
3. **Second LightGBM model.** Uses only information at least 16 days old, blended 80/20 with the main model.
4. **Linear model.** One ridge regression per product family on the last 224 days, with store level, store-by-weekday, payday, holiday and promotion terms. It makes up 20% of the final blend and reacts to promotions far less than the tree models do.
5. **Zero rule.** Series with no sales in the past 365 days are forecast as zero. A 21-day version was tested and made validation worse, because some quiet series start selling again.
## Validation
 
Every change had to improve **both** holdouts before it was kept, and nothing was tuned on the leaderboard.
 
- **Holdout A** is the last 16 days of training data, which mirrors the test window.
- **Holdout B** is the same calendar window one year earlier. Stores 18 and 25 closed during that window, so they are excluded from scoring.
Other variants tried and rejected include per-horizon models, later training start dates, smaller trees, denser training dates, and a cap on future promotion counts.
 
## Why the leaderboard score sits above the holdouts
 
An audit of the final pipeline found no bugs: the feature cutoff, promotion join, oil fill and submission ids all check out. The gap comes from two features of the test window that neither holdout contains:
 
- **Early back-to-school season.** School and office supply sales in the Sierra region peaked about a month earlier in 2017 than in previous years. The model's year-on-year features expect late August to keep rising.
- **Promotion spike on test day 1.** 2017-08-16 has about three times the usual number of items on promotion. The tree models react more strongly than history suggests they should for a few families. Capping promotion features made validation worse, so this is documented rather than patched.
## Kaggle notebooks
 
- [Store Sales EDA: What Drives Favorita Sales](https://www.kaggle.com/code/marvinoka4/store-sales-eda-what-drives-favorita-sales)
- [Store Sales: Direct Multi-Horizon LightGBM](https://www.kaggle.com/code/marvinoka4/store-sales-direct-multi-horizon-lightgbm), the full pipeline, which runs end to end on Kaggle CPU and writes `submission.csv`
- [Backtesting Store Sales: One Holdout Is Not Enough](https://www.kaggle.com/code/marvinoka4/backtesting-store-sales-one-holdout-is-not-enough), which scores a model on ten 16-day windows and traces the worst ones to store openings, closures, election-period alcohol bans and Carnaval
## Repository layout
 
```
src/pipeline.py   Feature building, LightGBM and ridge models, blending
src/audit.py      Checks on the final submission (feature cutoff, joins, ids)
data/             Competition data (not tracked; download it yourself)
```
 
## Running it
 
The simplest way to reproduce the submission is to fork the pipeline notebook on Kaggle and run it.
 
To run locally:
 
```bash
pip install kaggle lightgbm pandas numpy scikit-learn
kaggle competitions download -c store-sales-time-series-forecasting -p data
unzip data/store-sales-time-series-forecasting.zip -d data
```
 
You need a Kaggle account, an API token, and to have accepted the competition rules.
 
## Data
 
The competition data is not included in this repository. The competition rules do not allow redistributing it, so download it from Kaggle directly.
 
