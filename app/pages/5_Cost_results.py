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
- **Waste** - we ordered more than the real orders, so the extra food is thrown away.
  *Example: make 100 meals, only 80 are ordered: 20 are wasted.*
- **Missed orders** - we ordered less, so some customers could not get their meal.
  *Example: make 100 meals, 120 are wanted: 20 orders are missed.*
- **Cost** = cost of the waste + cost of the missed orders. It is a score, not real money.
'''
)
st.info(
    'The real costs are not in the data, so we tried different cost ratios. A ratio of **3** means: '
    'one missed order costs **3 times** as much as one wasted meal.  \n'
    'Example: a wasted meal costs 3 of food, and a missed order loses a sale of 9. Then the ratio is 3.'
)

r = st.select_slider(
    'How many times more does a missed order cost than a wasted meal?', options=COST_RATIOS, value=3,
    help='The cost ratio. 1 = both mistakes cost the same. 3 = a missed order is 3 times worse than a wasted meal. '
         '0.5 = a wasted meal is the bigger problem.')
tau = critical_ratio(r)
st.write(
    f'For this ratio, theory (the newsvendor rule) says the best order is the **{tau:.2f} quantile** forecast. '
    f'In simple words: order the amount that is enough in about **{tau * 100:.0f} out of 100 weeks**.'
)

sim = load_table('07_simulation_mean.csv')
sim = sim[sim['r'] == r].set_index('policy')
normal = sim.loc['Point LightGBM']
quantile = sim.loc['Quantile LightGBM']
saving = (normal['Cost'] - quantile['Cost']) / normal['Cost'] * 100

st.subheader('Normal forecast vs quantile forecast (LightGBM)',
             help='What changes when the kitchen orders the quantile forecast instead of the normal forecast.')
col1, col2, col3, col4 = st.columns(4)
col1.metric('Cost saving', f'{saving:.1f}%',
            help='How much lower the total cost is with the quantile forecast than with the normal forecast.')
col2.metric('Waste (% of orders)', f'{quantile["Waste %"]:.1f}%', f'{quantile["Waste %"] - normal["Waste %"]:+.1f}',
            delta_color='inverse',
            help='Meals thrown away, for every 100 meals that customers ordered.')
col3.metric('Missed orders (% of orders)', f'{quantile["Shortage %"]:.1f}%',
            f'{quantile["Shortage %"] - normal["Shortage %"]:+.1f}', delta_color='inverse',
            help='Meals customers wanted but could not get, for every 100 meals ordered.')
col4.metric('Weeks with no missed orders', f'{quantile["Service level %"]:.0f}%',
            f'{quantile["Service level %"] - normal["Service level %"]:+.0f}',
            help='In how many cases out of 100 nobody went without a meal (the order was enough).')
st.caption('The big number is the result with the quantile forecast. The small number under it is the change '
           'compared with ordering the normal LightGBM forecast.')


def change_text(new, old):
    # "up from 10.1" / "down from 18.6"
    return f'{"up" if new > old else "down"} from {old:.1f}'


st.markdown(
    f'''
**What these four numbers mean** (for the cost ratio you picked: {r})

- **Cost saving: {saving:.1f}%** - how much cheaper the quantile forecast is than the normal forecast.
  *Example: if the normal forecast gives a cost of 100, the quantile forecast gives about {100 - saving:.0f}.*
- **Waste: {quantile["Waste %"]:.1f}%** - for every 100 meals customers ordered, about {quantile["Waste %"]:.0f} extra
  meals were made and thrown away ({change_text(quantile["Waste %"], normal["Waste %"])} with the normal forecast).
- **Missed orders: {quantile["Shortage %"]:.1f}%** - for every 100 meals customers wanted, about
  {quantile["Shortage %"]:.0f} could not be served ({change_text(quantile["Shortage %"], normal["Shortage %"])}
  with the normal forecast).
- **Weeks with no missed orders: {quantile["Service level %"]:.0f}%** - in about {quantile["Service level %"]:.0f}
  of 100 cases, every customer got their meal ({change_text(quantile["Service level %"], normal["Service level %"])}
  with the normal forecast).

A green small number means the change is good, a red one means it is worse. Waste and missed orders pull in opposite
directions: to miss fewer orders, the kitchen must make more food and so waste more.
'''
)

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
st.dataframe(
    table.round(1), width='stretch',
    column_config={
        'Cost (normal LightGBM = 100)': st.column_config.NumberColumn(
            help='Total cost compared with ordering the normal LightGBM forecast, which is set to 100. '
                 'Below 100 = cheaper. Example: 75 means 25% cheaper.'),
        'Waste %': st.column_config.NumberColumn(help='Meals thrown away for every 100 meals ordered.'),
        'Missed orders %': st.column_config.NumberColumn(help='Meals that could not be served for every 100 meals wanted.'),
        'Weeks with no missed orders %': st.column_config.NumberColumn(
            help='In how many cases out of 100 the order was enough for every customer.'),
    })

cheapest = table['Cost (normal LightGBM = 100)'].idxmin()
st.markdown(
    f'''
**How to read this table**

- Each **row** is one way of ordering. "Normal forecast" rows order the one usual forecast.
  "Quantile" rows order the safe-level forecast for your cost ratio.
- **Cost (normal LightGBM = 100)** - the total cost of that way of ordering. The normal LightGBM forecast is set
  to 100, so the other rows are easy to compare. Below 100 is cheaper, above 100 is more expensive.
  *Example: the cheapest row here is "{cheapest}" with {table.loc[cheapest, "Cost (normal LightGBM = 100)"]:.1f},
  so it is {100 - table.loc[cheapest, "Cost (normal LightGBM = 100)"]:.1f}% cheaper than the normal LightGBM forecast.*
- **Waste %**, **Missed orders %** and **Weeks with no missed orders %** - the same three measures as in the boxes
  above, for every way of ordering.
'''
)

st.subheader('A fairer test: normal forecast + safety stock')
st.write(
    'Real planners usually order the normal forecast **plus a safety stock** (an extra margin learned from past errors). '
    '*Example: the forecast says 100 and the model was usually about 15% too low, so the planner orders 115.* '
    'We tested this too, to be fair.'
)
safety = load_table('08_saving_confidence.csv')
safety = safety[(safety['r'] == r) & (safety['compared with'] != 'Point LightGBM')].iloc[0]
st.metric('Extra saving of quantile LightGBM compared with normal forecast + safety stock',
          f'{safety["saving %"]:.1f}%',
          help='How much cheaper the quantile forecast still is, even against the safety-stock method.')
st.markdown(
    f'''
**What this number means:** even compared with the safety-stock method, the quantile forecast is still
**{safety["saving %"]:.1f}% cheaper**. We are fairly sure the true saving is between
**{safety["95% CI low"]:.1f}% and {safety["95% CI high"]:.1f}%** (this range comes from repeating the calculation
1,000 times with different groups of meals). The whole range is above zero, so the saving is real.

*Why:* the safety stock adds the same extra share to every meal. The quantile model learns a different extra amount
for each meal and week, so it wastes less food for almost the same number of missed orders.
'''
)

col1, col2 = st.columns([1, 1])
with col1:
    show_figure('fig29_cost_by_ratio', 'Cost of each method for every cost ratio')
    st.write(
        '**How to read it:** the bottom axis is the cost ratio (0.5 to 5). The side axis is the cost, with the normal '
        'LightGBM forecast as 100. Each line is one way of ordering, and a lower line is cheaper. '
        'The quantile lines fall further below 100 as missed orders get more expensive.'
    )
with col2:
    show_figure('fig30_waste_shortage_by_ratio', 'Waste and missed orders for every cost ratio')
    st.write(
        '**How to read it:** for each cost ratio, how much food is wasted and how many orders are missed. '
        'With the normal forecast both stay the same. With the quantile forecast, waste goes up and missed orders '
        'go down as the cost ratio grows.'
    )

st.success(
    '**Answer: yes.** Ordering the quantile forecast is cheaper than ordering the normal forecast **for every cost ratio**. '
    'The saving is small when waste and missed orders cost the same (2%), and grows to **25% at ratio 3** and '
    '**40% at ratio 5**.'
)
st.warning(
    'It is a trade-off: to miss fewer orders, the kitchen orders more and so wastes more food. '
    'The right cost ratio is a business decision.'
)
