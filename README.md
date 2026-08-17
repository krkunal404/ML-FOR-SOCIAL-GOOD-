# 🌍 Agricultural Risk Early-Warning

Classifying low-yield risk from soil, weather, and farming data — an ML for Social Good project.

**Course:** MCA 521-4 — Machine Learning · **Assessment:** CIA-III — ML for Social Good Ensemble Challenge

## Problem

Small farmers across India plan an entire season — what to plant, how much fertilizer to buy, how much land to commit — largely on experience and hope, with no early-warning signal about whether that season's soil and weather conditions resemble past seasons that ended in a significantly below-normal harvest.

This project builds a **low-yield-risk early-warning classifier**: given a crop, season, state, cultivated area, expected weather, soil nutrient levels, and planned fertilizer/pesticide use, it predicts whether that combination is at risk of falling into the **bottom 20% of yield outcomes for that specific crop** (a per-crop threshold, since "low yield" means very different things for sugarcane vs. a pulse crop).

**Beneficiaries:** small/marginal farmers, agricultural extension workers, and state agriculture departments / food-security planners.

## Prediction target

Binary classification, `low_yield_risk`:
- `1` — yield at or below the crop's own 20th percentile (at-risk)
- `0` — yield above that threshold (normal/good)

## Data

| File | Description |
|---|---|
| `crop_yield.csv` | State/crop/season/year-level Indian agricultural records (area, production, fertilizer, pesticide, yield), 1997–2020 |
| `state_soil_data.csv` | State-level soil N/P/K/pH reference values |
| `state_weather_data_1997_2020.csv` | State/year-level weather averages (temperature, rainfall, humidity) |

Unit of analysis: one row = one **crop + season + state + year** combination.

> These CSVs are not included in this repository. Place them alongside the notebook before running it (the notebook falls back to a Colab file-upload prompt if they're missing).

## Approach

1. **Data wrangling** — merge the three sources, fix inconsistent text formatting, validate merge keys, and confirm no missing values/duplicates.
2. **Leakage check** — `yield` and `production` are excluded from model inputs since the target is derived from `yield`.
3. **Feature engineering** — fertilizer/pesticide normalized to per-hectare intensity; outliers capped at the 99th percentile of the training set only.
4. **Preprocessing** — one-hot encoding for categoricals, median imputation + standard scaling for numerics, SMOTE oversampling applied only to the training split (stratified 70/15/15 train/val/test).
5. **Modeling** — Decision Tree baseline, Random Forest (bagging), XGBoost and LightGBM (boosting), and Voting + Stacking ensembles, each tuned with `GridSearchCV` and compared on an untouched test set (Accuracy, Precision, Recall, F1, ROC-AUC).
6. **Explainability** — global and local SHAP explanations on the best boosting model, plus a written ethics discussion (bias, privacy, uncertainty, false-positive/false-negative costs, human oversight).
7. **Demo** — a reusable `predict_and_explain()` function runs the saved pipeline on synthetic farmer scenarios.

## Repository contents

- [2547231_KUNNAL_ML_cia3.ipynb](2547231_KUNNAL_ML_cia3.ipynb) — full notebook (analysis, models, plots, explanations, ethics statement)
- [2547231_kunnal_ml_cia3.py](2547231_kunnal_ml_cia3.py) — script export of the same notebook

## Running it

```bash
pip install shap lightgbm imbalanced-learn xgboost scikit-learn pandas numpy matplotlib seaborn joblib
```

Place `crop_yield.csv`, `state_soil_data.csv`, and `state_weather_data_1997_2020.csv` in the working directory, then run the notebook top to bottom (or `python 2547231_kunnal_ml_cia3.py`). The trained pipeline is saved to `models/best_pipeline.joblib`.

## Limitations

Soil and weather values are state-level averages, not farm-level measurements; the dataset lacks irrigation, crop-variety, and real-time weather information; and patterns learned from 1997–2020 may shift as climate and farming practices change. This is a **decision-support tool**, not a replacement for agricultural experts or a guarantee of any farmer's actual outcome.
