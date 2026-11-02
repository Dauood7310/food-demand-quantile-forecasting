# Feature engineering for the weekly meal demand forecast.
#
# The idea (how a real planner would use the model):
#   - at the end of week t (the "origin") the planner orders ingredients for the next weeks
#   - we forecast the orders for week t + h, where h = 1 ... 10 weeks ahead
#   - so the model is only allowed to use orders from week t and before
#
# Two kinds of features:
#   1. history features  -> calculated from orders up to the origin week only
#   2. target week info  -> price and promotion for the week we forecast.
#                           These are planned by the company in advance, so it's fair to use them.
#
# Doing it this way means there is no data leakage: nothing from after the origin week is used.

import numpy as np
import pandas as pd

N_WEEKS = 145
HORIZONS = range(1, 11)     # forecast 1 to 10 weeks ahead


# ---------------------------------------------------------------------------
# step 1: load and clean
# ---------------------------------------------------------------------------

def load_data(raw_dir):
    """Read the three csv files and merge them into one table."""
    train = pd.read_csv(raw_dir + '/train.csv')
    centers = pd.read_csv(raw_dir + '/fulfilment_center_info.csv')
    meals = pd.read_csv(raw_dir + '/meal_info.csv')

    df = train.merge(centers, on='center_id', how='left')
    df = df.merge(meals, on='meal_id', how='left')
    return df


def clean_data(df):
    """Add the columns we need and make the text columns into categories."""
    df = df.copy()

    # one id per centre-meal pair, e.g. centre 55 + meal 1885 -> 551885
    df['series_id'] = df['center_id'] * 10000 + df['meal_id']

    # target on log scale because num_orders is very skewed (see notebook 02)
    df['log_orders'] = np.log1p(df['num_orders'])

    # discount in %, can be negative because checkout price includes taxes / delivery
    df['discount'] = (df['base_price'] - df['checkout_price']) / df['base_price'] * 100

    # text columns -> category type (LightGBM / XGBoost can use these directly)
    for col in ['center_type', 'category', 'cuisine']:
        df[col] = df[col].astype('category')

    df = df.drop(columns=['id'])
    df = df.sort_values(['series_id', 'week']).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# step 2: history features (only use weeks up to the origin)
# ---------------------------------------------------------------------------

def make_panel(df, col):
    """Table with one row per week (1-145) and one column per series.
    Weeks where the series has no row stay empty (NaN) - we don't fill them with 0."""
    panel = df.pivot_table(index='week', columns='series_id', values=col)
    panel = panel.reindex(range(1, N_WEEKS + 1))
    return panel


def history_features(df):
    """For every series and every week t, calculate features using weeks <= t only.

    All pandas functions used here (rolling, expanding, ffill, cumsum) only look
    backwards in time, so the value for week t never uses week t+1 or later.
    """
    orders = make_panel(df, 'log_orders')
    price = make_panel(df, 'checkout_price')
    seen = orders.notna()          # True if the series had a row in that week

    f = {}

    # orders in the origin week itself (empty if the meal wasn't offered that week)
    f['orders_last_week'] = orders

    # last orders we know about, even if the origin week itself is missing
    f['orders_last_known'] = orders.ffill()

    # how many weeks ago the series was last seen
    week_seen = seen.mul(orders.index, axis=0).where(seen)   # week number where seen, else NaN
    f['weeks_since_seen'] = week_seen.ffill().rsub(orders.index, axis=0)

    # average level over the last 4, 8 and 26 weeks (missing weeks are ignored)
    f['orders_mean_4'] = orders.rolling(4, min_periods=1).mean()
    f['orders_mean_8'] = orders.rolling(8, min_periods=1).mean()
    f['orders_mean_26'] = orders.rolling(26, min_periods=1).mean()

    # how much the orders move around recently
    f['orders_std_8'] = orders.rolling(8, min_periods=2).std()

    # average of all weeks so far, and how many weeks of history the series has
    f['orders_mean_all'] = orders.expanding().mean()
    f['weeks_of_history'] = seen.cumsum()

    # price history: needed to see if the target week price is higher / lower than usual
    f['price_mean_all'] = price.expanding().mean()
    f['price_last_known'] = price.ffill()

    # turn every week x series table into a long table: one row per (series, origin week)
    n_weeks, n_series = orders.shape
    hist = pd.DataFrame({
        'series_id': np.tile(orders.columns.values, n_weeks),
        'origin_week': np.repeat(orders.index.values, n_series),
    })
    for name, table in f.items():
        hist[name] = table.values.ravel().astype('float32')

    return hist


# ---------------------------------------------------------------------------
# step 3: put everything together into the modelling table
# ---------------------------------------------------------------------------

# columns that describe the week we forecast (known in advance)
TARGET_WEEK_COLS = ['checkout_price', 'base_price', 'discount', 'emailer_for_promotion', 'homepage_featured']

# columns that never change for a series
STATIC_COLS = ['center_id', 'meal_id', 'city_code', 'region_code', 'center_type', 'op_area', 'category', 'cuisine']


def build_dataset(df, hist, horizons=HORIZONS):
    """One row for every (series, target week, horizon h).

    Example: target week 120 with h = 3 -> origin week 117,
    so the history features come from week 117 and before.
    """
    keep = ['series_id', 'week', 'num_orders', 'log_orders'] + TARGET_WEEK_COLS + STATIC_COLS

    parts = []
    for h in horizons:
        part = df[keep].copy()
        part['h'] = h
        part['origin_week'] = part['week'] - h
        parts.append(part)
    data = pd.concat(parts, ignore_index=True)

    # need at least week 1 as origin
    data = data[data['origin_week'] >= 1]

    # add the history features of the origin week
    data = data.merge(hist, on=['series_id', 'origin_week'], how='left')

    # price compared to what the meal usually cost (strong effect in notebook 02, fig 14)
    data['price_vs_mean'] = data['checkout_price'] / data['price_mean_all']
    data['price_vs_last'] = data['checkout_price'] / data['price_last_known']

    data = data.sort_values(['week', 'series_id', 'h']).reset_index(drop=True)
    return data


HISTORY_COLS = ['orders_last_week', 'orders_last_known', 'weeks_since_seen',
                'orders_mean_4', 'orders_mean_8', 'orders_mean_26', 'orders_std_8',
                'orders_mean_all', 'weeks_of_history']

# the final list of model inputs
FEATURES = (['h'] + HISTORY_COLS + TARGET_WEEK_COLS + ['price_vs_mean', 'price_vs_last'] + STATIC_COLS)

TARGET = 'log_orders'
