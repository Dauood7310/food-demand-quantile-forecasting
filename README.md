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

The data files are already included in `data/raw`, so nothing needs to be downloaded.

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

The steps below are for a Windows machine. Python 3.10 or newer (tested with Python 3.13).

Install Python from https://www.python.org/downloads/ and tick **"Add python.exe to PATH"** during the install.
Then open Command Prompt or PowerShell in the project folder:

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m ipykernel install --user --name food-demand --display-name "Python 3 (food-demand)"
```

- If PowerShell says "running scripts is disabled on this system", run this once and try again:
  `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`
- If LightGBM or XGBoost give a DLL error, install the Microsoft Visual C++ Redistributable (x64):
  https://aka.ms/vs/17/release/vc_redist.x64.exe

### Optional: Mac

On a Mac the same steps are:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name food-demand --display-name "Python 3 (food-demand)"
```

If LightGBM or XGBoost give an error on a Mac, run `brew install libomp`.

## Running

Activate the virtual environment (`.venv\Scripts\activate`, or `source .venv/bin/activate` on a Mac), then:

```bat
jupyter lab
```

Choose the "Python 3 (food-demand)" kernel and run the notebooks in order.

### The app

A small Streamlit app shows the results in simple words and has an order planner.
The saved results are already in this repository, so the app works right after the setup
(the notebooks do not need to be run first).

Pages in the side menu: Home, Order planner, The data.
Three more pages (Accuracy results, Quantile results, Cost results) are kept in `app/hidden_pages`.
To show one again, move its file back into `app/pages` and restart the app.

#### How to run the app on a Windows machine

**Step 1 - install Python (only once).**
Download Python 3.10 or newer from https://www.python.org/downloads/ and tick
**"Add python.exe to PATH"** during the install.

**Step 2 - get the code (only once).**
Open **Command Prompt** (press the Windows key, type `cmd`, press Enter) and run:

```bat
git clone https://github.com/Dauood7310/food-demand-quantile-forecasting.git
cd food-demand-quantile-forecasting
```

If you already have the folder, go into it and run `git pull` to get the latest version.
(No Git? On the GitHub page click **Code > Download ZIP**, unzip it, and open Command Prompt in that folder.)

**Step 3 - install the packages (only once, takes a few minutes).**

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**Step 4 - start the app (every time).**

```bat
.venv\Scripts\activate
streamlit run app/Home.py
```

The app opens in the browser at http://localhost:8501. If it does not open, copy that address into the browser.
To stop the app, press `Ctrl + C` in the Command Prompt window.

**If something goes wrong**

| Problem | What to do |
|---|---|
| Streamlit asks for an email the first time | Just press Enter |
| `python` is not recognized | Install Python again and tick "Add python.exe to PATH", then open a new Command Prompt |
| PowerShell says "running scripts is disabled on this system" | Use Command Prompt, or run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once |
| LightGBM or XGBoost give a DLL error | Install https://aka.ms/vs/17/release/vc_redist.x64.exe |
| `streamlit` is not recognized | Run `.venv\Scripts\activate` first, then try again |
| Port 8501 is already in use | Run `streamlit run app/Home.py --server.port 8502` and open http://localhost:8502 |

On a Mac the same steps work with `python3 -m venv .venv` and `source .venv/bin/activate`.

### More about the research

- [ABOUT.md](ABOUT.md) - the project, the words and all results explained in easy English
- [RESEARCH_QA.md](RESEARCH_QA.md) - expected questions and short answers

| Notebook | What it does |
|---|---|
| 01_download_dataset | where the data comes from, md5 checksums and a quick check of rows / weeks / centres / meals (the data is already in `data/raw`, no need to run it) |
| 02_data_understanding | EDA plots and notes (missing weeks, price, promotions, autocorrelation, volatility), figures fig01-fig19 |
| 03_data_preparation | cleaning, leak-free features (`src/features.py`), 4 leakage tests, saves `clean.parquet` and `history_features.parquet` to `data/processed` |
| 04_baselines | rolling-origin folds (`src/folds.py`), metrics (`src/metrics.py`), naive / 26-week moving average / Linear Regression baselines, results in `results/tables`, figure fig22 |
| 05_tree_models | Random Forest, XGBoost and LightGBM (one global model, horizon as a feature), results in `results/tables`, Diebold-Mariano tests, figures fig23-fig25 |
| 06_quantile_models | linear quantile regression, quantile XGBoost and quantile LightGBM for tau 0.3-0.9, pinball loss and coverage, figures fig26-fig28 |
| 07_simulation | turns the forecasts into orders and simulates waste, shortage and cost for cost ratios 0.5-5 (newsvendor level), Diebold-Mariano tests, figures fig29-fig31 |
| 08_robustness | safety-stock baseline, corrected Diebold-Mariano test and bootstrap confidence intervals for the savings, figures fig32-fig33 |
