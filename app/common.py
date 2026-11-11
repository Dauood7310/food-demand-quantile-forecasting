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


def show_figure(name, caption):
    st.image(os.path.join(FIGURE_DIR, name + '.png'), caption=caption, width='stretch')
