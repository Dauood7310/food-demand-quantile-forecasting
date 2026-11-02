# Machine Learning for Demand Forecasting in Perishable Food Production Planning

MSc project (COM748, Ulster University) - Mohammad Dauood, supervisor Dr Serif Salem.

The project forecasts weekly meal demand 1 to 10 weeks ahead for a meal delivery company, so the kitchens can
order perishable ingredients with less waste and fewer shortages. It compares simple baselines, tree-based models
(Random Forest, XGBoost, LightGBM) and quantile regression models, using time-based (rolling-origin) validation,
and then simulates the waste, shortage and cost of the order quantities. The work follows CRISP-DM.

Research questions:
1. Do tree-based models forecast weekly meal demand better than simple baselines?
2. Can quantile models give useful forecasts of different demand levels (not just one number)?
3. Does ordering at the right quantile reduce the cost of waste and shortage?

## Dataset

Genpact / Analytics Vidhya "Food Demand Forecasting", Kaggle:
https://www.kaggle.com/datasets/kannanaikkal/food-demand-forecasting

456,548 rows: weekly orders of 51 meals in 77 fulfilment centres over 145 weeks, with price and promotion
information. Used only for academic research, there is no personal data in it.

Files in `data/raw`:

- `train.csv` - weekly orders per centre and meal, weeks 1-145 (target = `num_orders`)
- `fulfilment_center_info.csv` - centre city, region, type and operating area
- `meal_info.csv` - meal category and cuisine
- `test.csv`, `sample_submission.csv` - from the original hackathon, no target, not used
- `CHECKSUMS.md5` - md5 of each file, to check the data has not changed

## Folders

```
data/raw/          original Kaggle files (don't edit)
data/processed/    cleaned data and features
notebooks/         one notebook per step
src/               python files used by the notebooks
results/figures/   figures for the report (png)
results/tables/    results tables (csv)
```

## Setup

Python 3.10 or newer (I used 3.13 on a Mac).

```bash
python3 -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m ipykernel install --user --name food-demand --display-name "Python 3 (food-demand)"
```

If LightGBM or XGBoost give an error on Mac, run `brew install libomp`.

### Kaggle key (only needed to download the data again)

The data is already in `data/raw`. To download it again with notebook 01 you need a Kaggle API key:

1. kaggle.com -> your profile -> Settings -> API -> Create New Token (downloads `kaggle.json`)
2. move it:

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

Never commit `kaggle.json` (it is in `.gitignore`).

## Running

```bash
jupyter lab
```

Choose the "Python 3 (food-demand)" kernel and run the notebooks in order.

| Notebook | What it does |
|---|---|
| 01_download_dataset | downloads the data to `data/raw`, saves md5 checksums, quick check of rows / weeks / centres / meals |
| 02_data_understanding | EDA plots and notes (missing weeks, price, promotions, autocorrelation, volatility), figures fig01-fig19 |
