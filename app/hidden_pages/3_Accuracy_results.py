import streamlit as st

from common import load_table, show_figure

st.set_page_config(page_title='Accuracy results', page_icon='🎯', layout='wide')
st.title('Accuracy results')
st.subheader('Question 1: are machine learning models more accurate than simple methods?')

st.write(
    'Every model forecast the orders of all meals 1 to 10 weeks ahead. We tested them **three times** on weeks they '
    'had never seen (weeks 116-125, 126-135 and 136-145) and took the average.'
)
st.info(
    '**Error (WAPE)** = how far the forecasts are from the real orders, in %. **Lower is better.**  \n'
    'Example: an error of 30% means that for every 100 meals customers ordered, the forecast was wrong by 30 meals '
    '(too high or too low).'
)

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
col1.dataframe(
    table[['Model', 'Type', 'Error (%)']], hide_index=True, width='stretch',
    column_config={
        'Model': st.column_config.TextColumn('Model', help='The name of the forecasting method.'),
        'Type': st.column_config.TextColumn('Type', help='Simple method, or machine learning with decision trees.'),
        'Error (%)': st.column_config.NumberColumn(
            'Error (%)', format='%.1f',
            help='WAPE: for every 100 meals ordered, how many meals the forecast was wrong by. Lower is better.'),
    })
col2.bar_chart(table, x='Model', y='Error (%)', horizontal=True, sort='-Error (%)')

best = table.loc[table['Error (%)'].idxmin()]
worst = table.loc[table['Error (%)'].idxmax()]
st.markdown(
    f'''
**What the table and the chart show**

- **Model** - the forecasting method. The three simple ones are:
  **Naive** (next week = the same as last week), **Moving average** (the average of the last 26 weeks) and
  **Linear Regression** (a simple formula using past orders, price and promotions).
  The three machine learning ones (**Random Forest, XGBoost, LightGBM**) learn many yes/no rules from the data,
  for example "if there is a promotion and it is a beverage, expect more orders".
- **Type** - if the method is simple or machine learning.
- **Error (%)** - how wrong the method is. A shorter bar is better.
  *Example: {best["Model"]} has {best["Error (%)"]:.1f}%. So for every 100 meals ordered, it is wrong by about
  {best["Error (%)"]:.0f} meals. {worst["Model"]} has {worst["Error (%)"]:.1f}%, so it is wrong by about
  {worst["Error (%)"]:.0f} meals in every 100.*
'''
)

st.success(
    '**Answer: yes.** LightGBM (28.6%) and XGBoost (28.7%) have the lowest error. That is about **10% better** than the '
    'best simple method, Linear Regression (32.0%). A statistical test (Diebold-Mariano) shows this difference is real, '
    'not luck.'
)

st.subheader('Error for each week ahead',
             help='"Weeks ahead" = how far into the future the forecast is. 1 = next week, 10 = ten weeks from now.')
col1, col2 = st.columns([1, 1])
with col1:
    show_figure('fig23_tree_wape_by_horizon', 'Error for 1 to 10 weeks ahead')
with col2:
    st.write('')
    st.write(
        '**How to read it:** the bottom axis is how many weeks ahead we forecast (1 = next week, 10 = ten weeks '
        'from now). The side axis is the error. Each line is one model, and a lower line is better.'
    )
    st.write(
        'Forecasting further ahead is harder, so the error grows from about **27% for next week** to about '
        '**32% for ten weeks ahead**. *Example: for next week the best model is wrong by about 27 meals in every 100, '
        'for ten weeks ahead by about 32.*'
    )
    st.write('The machine learning models are better than Linear Regression **for every week ahead**.')

st.subheader('What does the best model look at?',
             help='Which pieces of information LightGBM uses most when it makes a forecast.')
col1, col2 = st.columns([1, 1])
with col1:
    show_figure('fig24_lightgbm_importance', 'Most useful information for LightGBM')
with col2:
    st.write('')
    st.write(
        '**How to read it:** each bar is one piece of information the model uses. A longer bar means the model '
        'uses it more. The numbers are shares of 100%.'
    )
    st.write(
        'The model mostly uses the **average orders of the last 26 weeks** and the **average of all past weeks** '
        '(together about 80%). After that come the **price** compared with the usual price and the **email promotion**. '
        '*Example: "this meal usually sells 150, it is cheaper than normal this week and it is promoted by email, '
        'so expect more than 150".*'
    )
    st.write('Last week\'s orders alone are not very useful, because orders jump up and down a lot.')

st.warning(
    'All models still forecast **a bit too low** on average: the forecast is above the real orders only about 46 times '
    'in 100. If the kitchen orders these numbers, it will run out more often than it wastes food. '
    'This is why we also built quantile models (next page).'
)
