import streamlit as st

from common import load_table, show_figure

st.set_page_config(page_title='Accuracy results', page_icon='🎯', layout='wide')
st.title('Accuracy results')
st.subheader('Question 1: are machine learning models more accurate than simple methods?')

st.write(
    'Every model forecast the orders of all meals 1 to 10 weeks ahead. We tested them **three times** on weeks they '
    'had never seen (weeks 116-125, 126-135 and 136-145) and took the average.'
)
st.info('**Error (WAPE)** = how far the forecasts are from the real orders, in %. **Lower is better.**')

results = load_table('05_results_mean.csv')
groups = {
    'Naive': 'Simple: copy last week',
    'Moving average (26 wk)': 'Simple: average of last 26 weeks',
    'Linear Regression': 'Simple: straight-line model',
    'Random Forest': 'Machine learning (trees)',
    'XGBoost': 'Machine learning (trees)',
    'LightGBM': 'Machine learning (trees)',
}
table = results[['model', 'WAPE']].copy()
table['type'] = table['model'].map(groups)
table['WAPE'] = table['WAPE'].round(1)
table.columns = ['Model', 'Error (%)', 'Type']

col1, col2 = st.columns([1, 1])
col1.dataframe(table[['Model', 'Type', 'Error (%)']], hide_index=True, width='stretch',
               column_config={'Error (%)': st.column_config.NumberColumn(format='%.1f')})
col2.bar_chart(table, x='Model', y='Error (%)', horizontal=True, sort='-Error (%)')

st.success(
    '**Answer: yes.** LightGBM (28.6%) and XGBoost (28.7%) have the lowest error. That is about **10% better** than the '
    'best simple method, Linear Regression (32.0%). A statistical test (Diebold-Mariano) shows this difference is real, '
    'not luck.'
)

st.subheader('Error for each week ahead')
col1, col2 = st.columns([1, 1])
with col1:
    show_figure('fig23_tree_wape_by_horizon', 'Error for 1 to 10 weeks ahead')
with col2:
    st.write('')
    st.write(
        'Forecasting further ahead is harder, so the error grows from about **27% for next week** to about '
        '**32% for ten weeks ahead**.'
    )
    st.write('The machine learning models are better than Linear Regression **for every week ahead**.')

st.subheader('What does the best model look at?')
col1, col2 = st.columns([1, 1])
with col1:
    show_figure('fig24_lightgbm_importance', 'Most useful information for LightGBM')
with col2:
    st.write('')
    st.write(
        'The model mostly uses the **average orders of the last 26 weeks** and the **average of all past weeks**. '
        'After that come the **price** compared with the usual price and the **email promotion**.'
    )
    st.write('Last week\'s orders alone are not very useful, because orders jump up and down a lot.')

st.warning(
    'All models still forecast **a bit too low** on average. If the kitchen orders these numbers, it will run out '
    'more often than it wastes food. This is why we also built quantile models (next page).'
)
