import pandas as pd
import streamlit as st

from common import load_forecasts, load_orders, load_table, meal_names, centre_names
from src.simulation import TAUS, COST_RATIOS, critical_ratio, quantity_at_tau

st.set_page_config(page_title='Order planner', page_icon='🧾', layout='wide')
st.title('Order planner')

st.write(
    'This page shows how a planner could use the model. Pick a centre, a meal and your costs. '
    'The page gives a **suggested order for each of the next 10 weeks**.'
)
st.info(
    'The planner stands at the end of **week 135**. It uses only the data up to that week and plans '
    'the next 10 weeks (**weeks 136-145**). The forecast and the suggested order do not use the real orders of '
    'those weeks. The real orders are shown in the table only so you can check the plan afterwards.'
)
st.caption('Tip: move the mouse over a small (?) icon to see what that item means.')

forecasts = load_forecasts()
orders = load_orders()
meal_name = meal_names()
centre_name = centre_names()

# 1. choices (every input has a help text, shown when the mouse is over the (?) icon)
st.subheader('1. Your choices', help='Pick the kitchen, the meal and how costly a missed order is for you.')
col1, col2, col3 = st.columns(3)
centre = col1.selectbox(
    'Centre', sorted(forecasts['center_id'].unique()), index=0, format_func=lambda c: centre_name[c],
    help='The kitchen (fulfilment centre) that cooks and sends out the meals. There are 77 centres. '
         'The company made the data anonymous, so there are no real centre or city names. '
         'A centre is shown by its number, its type (A, B or C: the company\'s own label for the kind of centre) '
         'and the code numbers of its city and region. Example: "city code 590" stands for one real city, '
         'but the data does not say which one.')
meals = sorted(forecasts.loc[forecasts['center_id'] == centre, 'meal_id'].unique(), key=lambda m: meal_name[m])
meal = col2.selectbox(
    'Meal', meals, format_func=lambda m: meal_name[m],
    help='The meal you want to plan for. The data has no real meal names, so a meal is shown by its cuisine '
         'and type, with its number (some meals have the same cuisine and type). '
         'Only the meals this centre sold in weeks 136-145 are in the list.')
r = col3.select_slider(
    'A missed order costs ... times a wasted meal', options=COST_RATIOS, value=3,
    help='How much worse is it to run out than to throw food away? '
         'Example: a wasted meal costs you 3 of food, and a missed order loses a sale of 9. '
         'Then a missed order costs 3 times more, so choose 3. '
         'Choose 1 if both cost the same. A higher number gives a bigger, safer order.')

tau = critical_ratio(r)
info = orders[orders['meal_id'] == meal].iloc[0]

col1, col2, col3 = st.columns(3)
col1.metric(
    'Meal type', f'{info["category"]}',
    help=f'The meal you picked: number {meal}, type "{info["category"]}", cuisine {info["cuisine"]}. '
         f'It is planned for {centre_name[centre]}.')
col2.metric(
    'Cost ratio', f'{r} to 1',
    help='Your choice above: one missed order costs this many times one wasted meal.')
col3.metric(
    'Safety level', f'{tau * 100:.0f}%',
    help=f'How safe the suggested order is. {tau * 100:.0f}% means the order should be enough in about '
         f'{tau * 100:.0f} out of 100 weeks. It comes from a stock rule (the newsvendor rule): '
         'the more a missed order costs, the higher the safety level. '
         'In the report this is called the quantile level.')

st.markdown(
    f'''
**What your choices mean**

- **Centre** - the kitchen that cooks the meals: {centre_name[centre]}. The data has no real names, only numbers and codes.
- **Meal** - the meal you plan for: {meal_name[meal]}.
- **Cost ratio: {r} to 1** - you said one missed order is {r} times as costly as one wasted meal.
  *Example: if a wasted meal costs you 3, a missed order costs you {3 * r:g}.*
- **Safety level: {tau * 100:.0f}%** - for this cost ratio, the best order is the amount that is enough in about
  {tau * 100:.0f} out of 100 weeks. A higher cost ratio gives a higher safety level, and so a bigger order.
'''
)

with st.expander('What is the cost ratio, and what is the unit of the cost?'):
    st.markdown(
        f'''
**Cost ratio** - how many times worse a missed order is than a wasted meal.
To find your own ratio, divide what you lose on a missed order by what you lose on a wasted meal.
*Example: a wasted meal loses 3 of food and a missed order loses a sale of 9, so the ratio is 9 / 3 = 3.*

| Cost ratio | Meaning |
|---|---|
| 0.5 | A missed order is only half as bad as a wasted meal |
| 1 | Both mistakes are equally bad |
| 3 | A missed order is 3 times worse than a wasted meal |
| 5 | A missed order is 5 times worse than a wasted meal |

**Unit of the cost** - the cost is **not money** (no pounds or dollars), because the data has no real prices.
It is counted in **"wasted meals"**: each wasted meal counts as 1, and each missed order counts as the cost ratio.
*Example with your ratio of {r}: 20 wasted meals and 10 missed orders give a cost of 20 x 1 + 10 x {r} = {20 + 10 * r:g}.
That means "as bad as throwing away {20 + 10 * r:g} meals".*

A company can turn it into money with its own prices.
*Example: if one wasted meal really costs 3, a cost of {20 + 10 * r:g} is {(20 + 10 * r) * 3:g} in money.*
'''
    )

# 2. forecast and suggested order for each week
rows = forecasts[(forecasts['center_id'] == centre) & (forecasts['meal_id'] == meal)].sort_values('week')
quantiles = rows[[f'LightGBM q{t}' for t in TAUS]].values
plan = pd.DataFrame({
    'Week': rows['week'].values,
    'Normal forecast': rows['LightGBM'].round().astype(int).values,
    'Suggested order': quantity_at_tau(quantiles, tau).round().astype(int),
})
plan['Difference'] = plan['Suggested order'] - plan['Normal forecast']
plan['Real orders'] = rows['num_orders'].astype(int).values      # only to check the plan, not used by it

st.subheader('2. Plan for the next 10 weeks',
             help='For each week: the normal forecast, the order this page suggests for your costs, '
                  'and (only to check) what customers really ordered.')
col1, col2 = st.columns([1.5, 1])
col1.dataframe(
    plan, hide_index=True, width='stretch',
    column_config={
        'Week': st.column_config.NumberColumn(
            'Week', help='The week we plan for. Weeks 136-145 are the 10 weeks after week 135.'),
        'Normal forecast': st.column_config.NumberColumn(
            'Normal forecast',
            help='The one best guess of the LightGBM model: the most likely number of meals. '
                 'It is too low about half of the time, and it does not look at your costs.'),
        'Suggested order': st.column_config.NumberColumn(
            'Suggested order',
            help='How many meals to make, for your costs. It is the forecast at your safety level, '
                 'so it is higher than the normal forecast when a missed order costs more than a wasted meal.'),
        'Difference': st.column_config.NumberColumn(
            'Difference', format='%+d',
            help='Suggested order minus normal forecast. A plus means: make this many meals more than the '
                 'normal forecast, to be safe. A minus means: make fewer, because waste is the bigger cost.'),
        'Real orders': st.column_config.NumberColumn(
            'Real orders',
            help='How many meals customers really ordered in that week. A real planner would not know this yet. '
                 'It is shown only to check the plan: it is not used to make the forecast or the suggested order.'),
    })
col2.line_chart(plan, x='Week', y=['Normal forecast', 'Suggested order'],
                color=['#1f77b4', '#d62728'], y_label='Meals')
st.caption('Blue = normal forecast, red = suggested order. The real orders are only in the table, not in the chart. '
           'Weeks when the meal is not on the menu are not shown.')

total_normal = plan['Normal forecast'].sum()
total_suggested = plan['Suggested order'].sum()
total_difference = total_suggested - total_normal
col1, col2, col3 = st.columns(3)
col1.metric('Normal forecast, all weeks', f'{total_normal:,}',
            help='The normal forecasts of all weeks in the table, added up.')
col2.metric('Suggested order, all weeks', f'{total_suggested:,}',
            help='The suggested orders of all weeks in the table, added up. This is how many meals to prepare in total.')
col3.metric('Difference', f'{total_difference:+,}',
            help='Suggested order minus normal forecast, for all weeks together.')

first = plan.iloc[0]
more_or_fewer = 'more' if first['Difference'] >= 0 else 'fewer'

# one sentence that checks the first week against the real orders
gap = first['Suggested order'] - first['Real orders']
if gap >= 0:
    check_text = (f'in week {first["Week"]} the suggested order ({first["Suggested order"]:,}) was enough, '
                  f'with {gap:,} meals left over.')
else:
    check_text = (f'in week {first["Week"]} the suggested order ({first["Suggested order"]:,}) was not enough: '
                  f'{abs(gap):,} customers would have gone without a meal.')
st.markdown(
    f'''
**What the columns mean** (the examples use week {first["Week"]}, the first row)

- **Week** - the week we plan for. Weeks 136 to 145 are the 10 weeks after week 135.
- **Normal forecast** - the model's one best guess of how many meals customers will order.
  *Example: {first["Normal forecast"]:,} meals.* It does not look at your costs, and it is too low about half of the time.
- **Suggested order** - how many meals this page tells the kitchen to make. It is the forecast at your safety level
  ({tau * 100:.0f}%), so it should be enough in about {tau * 100:.0f} out of 100 weeks. *Example: {first["Suggested order"]:,} meals.*
- **Difference** - suggested order minus normal forecast. *Example: {first["Difference"]:+,}, so the kitchen makes
  {abs(first["Difference"]):,} {more_or_fewer} meals than the normal forecast.*
- **Real orders** - how many meals customers really ordered in that week. *Example: {first["Real orders"]:,} meals.*
  A real planner would not know this number yet. It is shown only so you can check the plan:
  {check_text}

**The three boxes** add up all the weeks in the table: the normal forecast gives **{total_normal:,}** meals in total,
the suggested order gives **{total_suggested:,}**, a difference of **{total_difference:+,}** meals.

In the chart, the blue line is the normal forecast and the red line is the suggested order for each week.
'''
)

# 3. what to expect: the average result of this way of ordering in the tests (all meals and centres)
sim = load_table('07_simulation_mean.csv')
sim = sim[sim['r'] == r].set_index('policy')
normal = sim.loc['Point LightGBM']
suggested = sim.loc['Quantile LightGBM']
saving = (normal['Cost'] - suggested['Cost']) / normal['Cost'] * 100

st.subheader('3. What to expect with this plan',
             help='Nobody knows the real orders of the next weeks. So this part shows what happened on average '
                  'when the same way of ordering was tested on 30 past weeks, for all meals and centres.')
st.write(
    'Nobody knows the future orders. But the same way of ordering was **tested on 30 past weeks, for all meals and '
    f'all centres**. This is what happened on average, for your cost ratio of **{r}**:'
)
col1, col2, col3, col4 = st.columns(4)
col1.metric('Cost saving', f'{saving:.1f}%',
            help='How much lower the total cost was with the suggested order than with the normal forecast.')
col2.metric('Wasted meals', f'{suggested["Waste %"]:.1f}%', f'{suggested["Waste %"] - normal["Waste %"]:+.1f}',
            delta_color='inverse', help='Meals made but not ordered, for every 100 meals customers ordered.')
col3.metric('Missed orders', f'{suggested["Shortage %"]:.1f}%', f'{suggested["Shortage %"] - normal["Shortage %"]:+.1f}',
            delta_color='inverse', help='Meals customers wanted but could not get, for every 100 meals ordered.')
col4.metric('Weeks with no missed orders', f'{suggested["Service level %"]:.0f}%',
            f'{suggested["Service level %"] - normal["Service level %"]:+.0f}',
            help='In how many cases out of 100 the order was enough for every customer.')
st.caption('The big number is the result with the suggested order. The small number is the change compared with '
           'ordering the normal forecast (green = better, red = worse).')

st.markdown(
    f'''
**What these four numbers mean**

- **Cost saving: {saving:.1f}%** - the suggested order was this much cheaper than the normal forecast.
  *Example: if ordering the normal forecast costs 100, ordering the suggested amount costs about {100 - saving:.0f}.*
  The cost counts each wasted meal as 1 and each missed order as {r}.
- **Wasted meals: {suggested["Waste %"]:.1f}%** - for every 100 meals customers ordered, about
  {suggested["Waste %"]:.0f} extra meals were made and thrown away
  (with the normal forecast: {normal["Waste %"]:.1f}).
- **Missed orders: {suggested["Shortage %"]:.1f}%** - for every 100 meals customers wanted, about
  {suggested["Shortage %"]:.0f} could not be served (with the normal forecast: {normal["Shortage %"]:.1f}).
- **Weeks with no missed orders: {suggested["Service level %"]:.0f}%** - in about {suggested["Service level %"]:.0f}
  of 100 cases, every customer got their meal (with the normal forecast: {normal["Service level %"]:.0f}).
'''
)

if saving > 0:
    st.success(
        f'**In simple words:** when a missed order costs {r} times a wasted meal, ordering the suggested amount was '
        f'**{saving:.0f}% cheaper** on average than ordering the normal forecast.'
    )
st.info(
    'These are averages over all meals and centres. For one single meal the result can be better or worse, '
    'because a forecast is a good guess, not a promise. The costs are simulated, not real money.'
)
