import pandas as pd
import streamlit as st

from common import load_forecasts, load_orders, meal_names, centre_names
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

# 2. forecast and suggested order for each week
rows = forecasts[(forecasts['center_id'] == centre) & (forecasts['meal_id'] == meal)].sort_values('week')
quantiles = rows[[f'LightGBM q{t}' for t in TAUS]].values
plan = pd.DataFrame({
    'Week': rows['week'].values,
    'Normal forecast': rows['LightGBM'].round().astype(int).values,
    'Suggested order': quantity_at_tau(quantiles, tau).round().astype(int),
    'Real orders': rows['num_orders'].astype(int).values,
})

st.subheader('2. Plan for the next 10 weeks',
             help='For each week: the normal forecast, the order this page suggests, and what customers really ordered.')
col1, col2 = st.columns([1, 1.4])
col1.dataframe(
    plan, hide_index=True, width='stretch',
    column_config={
        'Week': st.column_config.NumberColumn(
            'Week', help='The week number in the data. Weeks 136-145 are the 10 weeks after the forecast was made.'),
        'Normal forecast': st.column_config.NumberColumn(
            'Normal forecast',
            help='The one best guess of the LightGBM model: the most likely number of meals. '
                 'It is too low about half of the time, and it does not look at your costs.'),
        'Suggested order': st.column_config.NumberColumn(
            'Suggested order',
            help='How many meals to make, for your costs. It is the forecast at your safety level, '
                 'so it is higher than the normal forecast when a missed order costs more than a wasted meal.'),
        'Real orders': st.column_config.NumberColumn(
            'Real orders',
            help='How many meals customers really ordered in that week. '
                 'A real planner would not know this yet. It is shown here only to check the suggestion.'),
    })
col2.line_chart(plan, x='Week', y=['Normal forecast', 'Suggested order', 'Real orders'],
                color=['#1f77b4', '#d62728', '#000000'], y_label='Orders')
st.caption('Blue = normal forecast, red = suggested order, black = real orders. '
           'Weeks with no real orders (meal not on offer) are not shown.')
first = plan.iloc[0]
difference = first['Suggested order'] - first['Normal forecast']
st.markdown(
    f'''
**What the columns mean** (the examples use week {first["Week"]}, the first row)

- **Week** - the week we plan for. Weeks 136 to 145 are the 10 weeks after the forecast was made.
- **Normal forecast** - the model's one best guess of how many meals will be ordered.
  *Example: {first["Normal forecast"]:,} meals.* It does not look at your costs.
- **Suggested order** - how many meals this page tells the kitchen to make. It is the forecast at your safety level
  ({tau * 100:.0f}%). *Example: {first["Suggested order"]:,} meals, which is {abs(difference):,}
  {"more" if difference >= 0 else "fewer"} than the normal forecast.*
- **Real orders** - how many meals customers really ordered. *Example: {first["Real orders"]:,} meals.*
  A real planner would not know this number yet. It is shown only to check how good the suggestion was.

In the chart, the same three numbers are drawn as lines for all 10 weeks.
'''
)
with st.expander('Why are the real orders sometimes far from the forecast?'):
    st.write(
        'A forecast is a good guess, not a promise. On average the model is wrong by about 29 meals in every 100. '
        'For most meals the real orders stay close to the forecast. But for some meals the real orders jump '
        'for a reason the data cannot show (for example a local event), and then the black line is far above '
        'or below the other lines. In these 10 weeks, the real orders were more than 1.5 times the normal '
        'forecast for about 8 of every 100 centre-meal pairs.'
    )
    st.write(
        'This is why the suggested order uses a safety level: it is made higher on purpose, '
        'so that surprises hurt less.'
    )

# 3. what would have happened
normal = simulate(plan['Real orders'], plan['Normal forecast'], c_o=1, c_u=r)
suggested = simulate(plan['Real orders'], plan['Suggested order'], c_o=1, c_u=r)

st.subheader('3. What would have happened?',
             help='We compare the two ways of ordering with the real orders of these weeks. '
                  'If we made more than was ordered, the extra meals are wasted. '
                  'If we made less, the missing meals are missed orders.')

waste_help = ('Meals that were made but not ordered, added up over the weeks. They are thrown away. '
              'Example: make 100, only 80 ordered -> 20 wasted.')
missed_help = ('Meals that customers wanted but the kitchen did not have, added up over the weeks. '
               'Example: make 100, 120 wanted -> 20 missed orders.')
cost_help = (f'Wasted meals x 1 + missed orders x {r}. Lower is better. '
             'This is not real money: it counts in "wasted meals", because the data has no real prices.')

st.write(
    'We look back at these 10 weeks and ask: **if the kitchen had made this many meals, what would have happened?** '
    'We check two ways of ordering against what customers really ordered.'
)
st.markdown(
    f'''
- **Wasted meals** = meals that were made but nobody ordered. They are thrown away.
- **Missed orders** = meals that customers wanted, but the kitchen had none left.
- **Cost** = a score, not real money. Each wasted meal counts as **1** and each missed order counts as **{r}**
  (your choice above). **Lower is better.**
'''
)


def show_result(title, result):
    # one row of three numbers, and the cost calculation in words below it
    waste, missed, cost = result['Waste'], result['Shortage'], result['Cost']
    st.write(f'**{title}**')
    col1, col2, col3 = st.columns(3)
    col1.metric('Wasted meals', f'{waste:,.0f}', help=waste_help)
    col2.metric('Missed orders', f'{missed:,.0f}', help=missed_help)
    col3.metric('Cost', f'{cost:,.0f}', help=cost_help)
    st.caption(f'Cost = {waste:,.0f} wasted meals x 1  +  {missed:,.0f} missed orders x {r}  =  {cost:,.0f}')


show_result('Way 1: make the normal forecast', normal)
show_result('Way 2: make the suggested amount', suggested)

def more_or_fewer(change):
    # +30 -> "30 more", -30 -> "30 fewer"
    return f'{abs(change):,.0f} more' if change > 0 else f'{abs(change):,.0f} fewer'


waste_change = suggested['Waste'] - normal['Waste']
missed_change = suggested['Shortage'] - normal['Shortage']

if suggested['Cost'] < normal['Cost']:
    saving = (normal['Cost'] - suggested['Cost']) / normal['Cost'] * 100
    st.success(
        f'**In simple words:** with the suggested amount, {more_or_fewer(waste_change)} meals are wasted and '
        f'{more_or_fewer(missed_change)} customers go without a meal. Because a missed order counts {r} times '
        f'a wasted meal, the total cost is **{saving:.0f}% lower** '
        f'({suggested["Cost"]:,.0f} instead of {normal["Cost"]:,.0f}).'
    )
elif suggested['Cost'] > normal['Cost']:
    st.warning(
        f'**In simple words:** for this meal the normal forecast happened to be cheaper '
        f'({normal["Cost"]:,.0f} against {suggested["Cost"]:,.0f}). This can happen for one meal. '
        'Over all meals and centres, the suggested order is cheaper on average (see the Cost results page).'
    )
else:
    st.info('For this meal both ways give the same cost.')
