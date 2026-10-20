# Seoul Apartment Price Prediction

**MSc thesis project — Sungkyunkwan University, Big Data (2022)**

Predicting apartment transaction prices in Seoul from **1,090,441 records (2006–2021)**, with a focus on whether green-building certifications are reflected in market prices.

---

## Research question

Korean buildings carry two public environmental certifications — **BEEC** (Building Energy Efficiency Certification) and **G-SEED** (Green Standard for Energy and Environmental Design). Both are costly for developers to obtain. The question this project set out to answer:

> Do these certifications show up in the price people actually pay per square metre, once building and location characteristics are controlled for?

To answer it, I built a dataset joining five years of transaction records with building registry, official assessed values and certification data, then compared classical statistical models against gradient-boosted trees.

---

## Data

No single dataset existed for this, so the first half of the project was building one from public APIs.

| Source | What it provided |
|---|---|
| Seoul Open Data Plaza | Apartment transaction records, 2006–2021 (~2.4M rows collected, paginated) |
| Ministry of Land official assessed values | Government valuation per unit/floor/year (35M rows, aggregated) |
| BEEC certification registry | Energy efficiency grade, energy saving rate, EPI score |
| Building registry | Floor area ratio, building coverage ratio, parking spaces, households |

Records were joined on **parcel codes (지번코드)**, which required constructing a composite key from district, dong, block and lot numbers — the single messiest part of the pipeline.

**Final modelling dataset:** 1,090,441 rows × 17 features

Features include unit area, floor, construction year, building age, official assessed value, parking ratio, household count, district group (5 Seoul regions, one-hot encoded), and the two certification variables. Target: **price per m²**.

---

## Pipeline

| Notebook | What it does |
|---|---|
| `1_price_collection` | Collects transaction records from the Seoul Open API (XML, paginated) |
| `2_data_collection` | Collects official assessed values, certification and building registry data |
| `3_data_join` | Joins sources on parcel codes; derives price per m² and parking ratio |
| `4_outlier_detection` | Distribution checks and outlier removal with **Isolation Forest** |
| `5_linear_regression` | Correlation analysis, **VIF** multicollinearity checks, **OLS** regression, ANOVA |
| `6_hyperparameter_tuning` | `RandomizedSearchCV` over XGBoost parameters (5-fold CV, ~83 min wall time) |
| `7_model_comparison` | Trains and compares four tree-based regressors |

The classical statistics come before the machine learning deliberately: correlation and VIF were used to decide which features to keep, and OLS/ANOVA gave an interpretable baseline before moving to a model that predicts better but explains less.

---

## Results

All models trained on the same features and split (70/30, stratified on certification group), target standardised.

| Model | R² | RMSE |
|---|---|---|
| **XGBoost** (tuned) | **0.947** | **0.229** |
| Decision Tree | 0.901 | 0.314 |
| Gradient Boosting | 0.874 | 0.355 |
| AdaBoost | 0.686 | 0.561 |

Tuned XGBoost parameters found by random search:

```
eta=0.1, n_estimators=500, max_depth=10,
min_child_weight=0, gamma=0.3, subsample=0.5, booster='gbtree'
```

---

## Limitations, and what I would do differently now

Writing this up again in 2026, three things stand out that I would change:

**1. `OfficialValue` is close to leakage.** The government assessed value is itself derived with reference to market transactions, so it is structurally correlated with the target. A meaningful share of the 0.947 R² likely comes from this one feature. The honest version of this project runs the comparison with and without it and reports both.

**2. Scaling was fit before the train/test split.** `StandardScaler` was fitted on the full dataset, which leaks test-set distribution information into training. With over a million rows the practical effect is small, but it is the wrong order — the scaler belongs inside the pipeline, fitted on training data only.

**3. Prediction accuracy is not the research question.** The thesis asked whether certification is *reflected* in price. A strong predictive model does not establish that. The follow-up this deserves is a causal design — comparing certified and non-certified buildings that are otherwise similar — rather than a better R².

I have left the analysis as it was submitted rather than quietly fixing it. The limitations are part of the record.

---

## A note on the code

This was written in 2022, before AI coding assistants were part of everyday work. Every API parser, pagination loop and join key here was written by hand, and the collection notebooks in particular show it — repeated blocks, trial-and-error cells, Korean comments written for an audience of one.

It is not how I would structure a pipeline today. I have kept it close to its original form because it is an accurate record of the work, and because the reasoning behind each step still holds up even where the code does not.

---

## Running this

The notebooks were written in Google Colab and read from Google Drive paths (`/content/drive/MyDrive/Colab Notebooks/apt/...`). To run them you will need to change those paths to your own.

API keys for the public data sources have been removed — replace `YOUR_API_KEY` with your own key from the [Seoul Open Data Plaza](https://data.seoul.go.kr/).

```
pandas · numpy · scikit-learn · xgboost · statsmodels · beautifulsoup4 · matplotlib · seaborn
```

---

## Author

**Chunjin Kim** — Data Scientist, London
[LinkedIn](https://linkedin.com/in/chunjinkim) · [GitHub](https://github.com/churnjean2)
