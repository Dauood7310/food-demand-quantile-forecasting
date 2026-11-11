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

st.write('**If we order the normal forecast**')
col1, col2, col3 = st.columns(3)
col1.metric('Wasted meals', f'{normal["Waste"]:.0f}', help=waste_help)
col2.metric('Missed orders', f'{normal["Shortage"]:.0f}', help=missed_help)
col3.metric('Cost', f'{normal["Cost"]:.0f}', help=cost_help)

st.write('**If we order the suggested amount**')
col1, col2, col3 = st.columns(3)
col1.metric('Wasted meals', f'{suggested["Waste"]:.0f}', help=waste_help)
col2.metric('Missed orders', f'{suggested["Shortage"]:.0f}', help=missed_help)
col3.metric('Cost', f'{suggested["Cost"]:.0f}', help=cost_help)

if suggested['Cost'] < normal['Cost']:
    saving = (normal['Cost'] - suggested['Cost']) / normal['Cost'] * 100
    st.success(f'For this meal, the suggested order is **{saving:.0f}% cheaper** than ordering the normal forecast.')
elif suggested['Cost'] > normal['Cost']:
    st.warning('For this meal the normal forecast happened to be cheaper. Over all meals and centres, the suggested '
               'order is cheaper on average (see the Cost results page).')
else:
    st.info('For this meal both orders give the same cost.')
