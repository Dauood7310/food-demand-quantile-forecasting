import pandas as pd
import streamlit as st

from common import load_forecasts, load_orders
from src.simulation import TAUS, COST_RATIOS, critical_ratio, quantity_at_tau, simulate

st.set_page_config(page_title='Order planner', page_icon='🧾', layout='wide')
st.title('Order planner')

st.write(
    'This page shows how a planner could use the model. Pick a centre, a meal and your costs. '
    'The page gives a **suggested order for each of the next 10 weeks**.'
)
st.info(
    'The forecasts were made at **week 135**, using only the data up to that week, for **weeks 136-145**. '
    'Because these weeks are in the data, we can also show the real orders and check the suggestion.'
)

forecasts = load_forecasts()
orders = load_orders()

# 1. choices
col1, col2, col3 = st.columns(3)
centre = col1.selectbox('Centre', sorted(forecasts['center_id'].unique()), index=0)
meals = sorted(forecasts.loc[forecasts['center_id'] == centre, 'meal_id'].unique())
meal = col2.selectbox('Meal', meals)
r = col3.select_slider('A missed order costs ... times a wasted meal', options=COST_RATIOS, value=3)

tau = critical_ratio(r)
info = orders[orders['meal_id'] == meal].iloc[0]
st.write(f'Meal **{meal}** ({info["category"]}, {info["cuisine"]} cuisine). '
         f'With cost ratio **{r}**, the best order is the **{tau:.2f} quantile** forecast.')

# 2. forecast and suggested order for each week
rows = forecasts[(forecasts['center_id'] == centre) & (forecasts['meal_id'] == meal)].sort_values('week')
quantiles = rows[[f'LightGBM q{t}' for t in TAUS]].values
plan = pd.DataFrame({
    'Week': rows['week'].values,
    'Normal forecast': rows['LightGBM'].round().astype(int).values,
    'Suggested order': quantity_at_tau(quantiles, tau).round().astype(int),
    'Real orders': rows['num_orders'].astype(int).values,
})

st.subheader('Plan for the next 10 weeks')
col1, col2 = st.columns([1, 1.4])
col1.dataframe(plan, hide_index=True, width='stretch')
col2.line_chart(plan, x='Week', y=['Normal forecast', 'Suggested order', 'Real orders'],
                color=['#1f77b4', '#d62728', '#000000'], y_label='Orders')
st.caption('Weeks with no real orders (meal not on offer) are not shown.')

# 3. what would have happened
normal = simulate(plan['Real orders'], plan['Normal forecast'], c_o=1, c_u=r)
suggested = simulate(plan['Real orders'], plan['Suggested order'], c_o=1, c_u=r)

st.subheader('What would have happened?')
result = pd.DataFrame({
    'Order the normal forecast': [normal['Waste'], normal['Shortage'], normal['Cost']],
    'Order the suggested amount': [suggested['Waste'], suggested['Shortage'], suggested['Cost']],
}, index=['Wasted meals', 'Missed orders', f'Cost (1 per wasted meal, {r} per missed order)'])
st.dataframe(result.round(0).astype(int), width='stretch')

if suggested['Cost'] < normal['Cost']:
    saving = (normal['Cost'] - suggested['Cost']) / normal['Cost'] * 100
    st.success(f'For this meal, the suggested order is **{saving:.0f}% cheaper** than ordering the normal forecast.')
elif suggested['Cost'] > normal['Cost']:
    st.warning('For this meal the normal forecast happened to be cheaper. Over all meals and centres, the suggested '
               'order is cheaper on average (see the Cost results page).')
else:
    st.info('For this meal both orders give the same cost.')
