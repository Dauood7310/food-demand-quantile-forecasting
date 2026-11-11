# Things every page of the app needs: where the files are, and loading them.
# The app only READS the results made by the notebooks, it does not train any model.

import os
import sys

import pandas as pd
import streamlit as st

CODE_DIR = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(CODE_DIR)          # so the pages can use src/simulation.py

DATA_DIR = os.path.join(CODE_DIR, 'data', 'processed')
TABLE_DIR = os.path.join(CODE_DIR, 'results', 'tables')
FIGURE_DIR = os.path.join(CODE_DIR, 'results', 'figures')


@st.cache_data
def load_table(name):
    return pd.read_csv(os.path.join(TABLE_DIR, name))


@st.cache_data
def load_orders():
    return pd.read_parquet(os.path.join(DATA_DIR, 'clean.parquet'))


@st.cache_data
def load_forecasts():
    """Forecasts for the last test period (made at week 135 for weeks 136-145)."""
    keys = ['series_id', 'week', 'origin']
    quantiles = pd.read_parquet(os.path.join(DATA_DIR, 'predictions_quantiles.parquet'))
    point = pd.read_parquet(os.path.join(DATA_DIR, 'predictions_trees.parquet'))[keys + ['LightGBM']]
    forecasts = quantiles.merge(point, on=keys)
    return forecasts[forecasts['origin'] == 135]


@st.cache_data
def meal_names():
    """Readable name for every meal number, e.g. 1062 -> 'Italian Beverages (meal 1062)'.
    The data has no real meal names, only the category and the cuisine."""
    meals = load_orders()[['meal_id', 'category', 'cuisine']].drop_duplicates()
    return {row.meal_id: f'{row.cuisine} {row.category} (meal {row.meal_id})' for row in meals.itertuples()}


@st.cache_data
def centre_names():
    """Readable name for every centre number, e.g. 10 -> 'Centre 10 (Type B, city code 590, region code 56)'.
    The data is anonymous: it has no real centre or city names, only the type and the city and region CODES."""
    centres = load_orders()[['center_id', 'center_type', 'city_code', 'region_code']].drop_duplicates()
    return {row.center_id: f'Centre {row.center_id} (Type {str(row.center_type)[-1]}, city code {row.city_code}, '
                           f'region code {row.region_code})' for row in centres.itertuples()}


def show_figure(name, caption):
    st.image(os.path.join(FIGURE_DIR, name + '.png'), caption=caption, width='stretch')
