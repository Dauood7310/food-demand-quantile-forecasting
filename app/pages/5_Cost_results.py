import streamlit as st

from common import load_table, show_figure
from src.simulation import COST_RATIOS, critical_ratio

st.set_page_config(page_title='Cost results', page_icon='💷', layout='wide')
st.title('Cost results')
st.subheader('Question 3: does ordering the right quantile reduce waste and missed orders?')

st.write(
    'We used each forecast as the **order** for every meal, centre and week in the test periods, '
    'and counted what would have happened:'
)
st.markdown(
    '''
- **Waste** - we ordered more than the real orders, so the extra food is thrown away
- **Missed orders** - we ordered less, so some customers could not get their meal
- **Cost** = cost of the waste + cost of the missed orders
'''
)
st.info(
    'The real costs are not in the data, so we tried different cost ratios. A ratio of **3** means: '
    'one missed order costs **3 times** as much as one wasted meal.'
)

r = st.select_slider('How many times more does a missed order cost than a wasted meal?',
                     options=COST_RATIOS, value=3)
tau = critical_ratio(r)
st.write(f'For this ratio, theory (the newsvendor rule) says the best order is the **{tau:.2f} quantile** forecast.')

sim = load_table('07_simulation_mean.csv')
sim = sim[sim['r'] == r].set_index('policy')
normal = sim.loc['Point LightGBM']
quantile = sim.loc['Quantile LightGBM']
saving = (normal['Cost'] - quantile['Cost']) / normal['Cost'] * 100

st.subheader('Normal forecast vs quantile forecast (LightGBM)')
col1, col2, col3, col4 = st.columns(4)
col1.metric('Cost saving', f'{saving:.1f}%')
col2.metric('Waste (% of orders)', f'{quantile["Waste %"]:.1f}%', f'{quantile["Waste %"] - normal["Waste %"]:+.1f}', delta_color='inverse')
col3.metric('Missed orders (% of orders)', f'{quantile["Shortage %"]:.1f}%', f'{quantile["Shortage %"] - normal["Shortage %"]:+.1f}', delta_color='inverse')
col4.metric('Weeks with no missed orders', f'{quantile["Service level %"]:.0f}%', f'{quantile["Service level %"] - normal["Service level %"]:+.0f}')
st.caption('The small numbers show the change compared with ordering the normal LightGBM forecast.')

st.subheader('All order methods for this cost ratio')
names = {
    'Point Linear Regression': 'Normal forecast - Linear Regression',
    'Point LightGBM': 'Normal forecast - LightGBM',
    'Linear QR': 'Quantile - linear quantile regression',
    'Quantile XGBoost': 'Quantile - XGBoost',
    'Quantile LightGBM': 'Quantile - LightGBM',
}
table = sim.loc[list(names.keys()), ['Cost', 'Waste %', 'Shortage %', 'Service level %']].copy()
table['Cost'] = table['Cost'] / normal['Cost'] * 100
table.index = [names[p] for p in table.index]
table.columns = ['Cost (normal LightGBM = 100)', 'Waste %', 'Missed orders %', 'Weeks with no missed orders %']
st.dataframe(table.round(1), width='stretch')

st.subheader('A fairer test: normal forecast + safety stock')
st.write(
    'Real planners usually order the normal forecast **plus a safety stock** (an extra margin learned from past errors). '
    'We tested this too, to be fair.'
)
safety = load_table('08_saving_confidence.csv')
safety = safety[(safety['r'] == r) & (safety['compared with'] != 'Point LightGBM')].iloc[0]
st.metric('Extra saving of quantile LightGBM compared with normal forecast + safety stock',
          f'{safety["saving %"]:.1f}%')
st.caption(f'95% confidence range: {safety["95% CI low"]:.1f}% to {safety["95% CI high"]:.1f}%. '
           'The safety stock adds the same margin to every meal; the quantile model learns a different margin for each '
           'meal and week, so it wastes less food for almost the same number of missed orders.')

col1, col2 = st.columns([1, 1])
with col1:
    show_figure('fig29_cost_by_ratio', 'Cost of each method for every cost ratio')
with col2:
    show_figure('fig30_waste_shortage_by_ratio', 'Waste and missed orders for every cost ratio')

st.success(
    '**Answer: yes.** Ordering the quantile forecast is cheaper than ordering the normal forecast **for every cost ratio**. '
    'The saving is small when waste and missed orders cost the same (2%), and grows to **25% at ratio 3** and '
    '**40% at ratio 5**.'
)
st.warning(
    'It is a trade-off: to miss fewer orders, the kitchen orders more and so wastes more food. '
    'The right cost ratio is a business decision.'
)
