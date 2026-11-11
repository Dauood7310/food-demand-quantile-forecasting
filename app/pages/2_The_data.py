import streamlit as st

from common import load_orders, show_figure, meal_names, centre_names

st.set_page_config(page_title='The data', page_icon='📊', layout='wide')
st.title('The data')

orders = load_orders()
meal_name = meal_names()
centre_name = centre_names()

st.write(
    'The data comes from a real meal delivery company (Genpact, shared on Kaggle). '
    'Each row is **one meal at one fulfilment centre in one week**, with the number of orders, the price '
    'and whether the meal was promoted.'
)

n_rows = len(orders)
n_meals = orders['meal_id'].nunique()
n_centres = orders['center_id'].nunique()
n_weeks = orders['week'].nunique()

col1, col2, col3, col4 = st.columns(4)
col1.metric('Rows', f'{n_rows:,}', help='How many lines the data has. One line = one meal at one centre in one week.')
col2.metric('Meals', n_meals, help='How many different meals the company sells.')
col3.metric('Centres', n_centres, help='How many kitchens (fulfilment centres) cook and send out the meals.')
col4.metric('Weeks', n_weeks, help='How many weeks of history the data has. That is almost 3 years.')

st.markdown(
    f'''
**What these four numbers mean**

- **Rows ({n_rows:,})** - the number of lines in the data. Each line says how many of one meal one centre sold in
  one week. *Example: "in week 1, centre 55 sold 177 of meal 1885" is one row.*
- **Meals ({n_meals})** - the number of different meals on the menu, for example a Thai beverage or an Italian pizza.
- **Centres ({n_centres})** - the number of kitchens that cook and deliver the meals. Each one serves a different area.
- **Weeks ({n_weeks})** - how long the history is. Week 1 is the first week and week {n_weeks} is the last
  (almost 3 years). The data does not give real dates.
'''
)

st.subheader('A few rows of the data')
preview = orders[['week', 'center_id', 'meal_id', 'category', 'cuisine', 'checkout_price',
                  'emailer_for_promotion', 'homepage_featured', 'num_orders']].head(10)
preview.columns = ['Week', 'Centre', 'Meal', 'Category', 'Cuisine', 'Price',
                   'Email promotion', 'On homepage', 'Orders']
st.dataframe(preview, hide_index=True, width='stretch')

first = preview.iloc[0]
st.markdown(
    f'''
**What each column means** (the examples use the first row of the table)

- **Week** - the week number, from 1 to {n_weeks}. *Example: {first["Week"]} means week {first["Week"]} of the data.*
- **Centre** - the number of the kitchen. The data has no real names. *Example: centre {first["Centre"]}.*
- **Meal** - the number of the meal. *Example: meal {first["Meal"]}.*
- **Category** - the type of meal. *Example: {first["Category"]}.*
- **Cuisine** - the cooking style. *Example: {first["Cuisine"]}.*
- **Price** - what the customer paid for one meal in that week. *Example: {first["Price"]:.2f}.*
- **Email promotion** - 1 if the meal was advertised by email that week, 0 if not. *Example: {first["Email promotion"]}.*
- **On homepage** - 1 if the meal was shown on the home page of the website that week, 0 if not.
  *Example: {first["On homepage"]}.*
- **Orders** - how many of this meal the centre sold in that week. **This is the number the models forecast.**
  *Example: {first["Orders"]:,} meals.*
'''
)

st.subheader('Orders of one meal')
st.write('Pick a centre and a meal to see its weekly orders.')
col1, col2 = st.columns(2)
centre = col1.selectbox(
    'Centre', sorted(orders['center_id'].unique()), index=0, format_func=lambda c: centre_name[c],
    help='The kitchen. The data is anonymous, so a centre is shown by its number, its type (A, B or C) '
         'and the code numbers of its city and region.')
meals = sorted(orders.loc[orders['center_id'] == centre, 'meal_id'].unique(), key=lambda m: meal_name[m])
meal = col2.selectbox(
    'Meal', meals, format_func=lambda m: meal_name[m],
    help='The meal. The data has no real meal names, so a meal is shown by its cuisine and type, with its number.')
one = orders[(orders['center_id'] == centre) & (orders['meal_id'] == meal)]
st.line_chart(one.set_index('week')['num_orders'], x_label='Week', y_label='Orders')
st.markdown(
    f'''
**How to read this chart:** the line shows how many of **{meal_name[meal]}** were sold at **{centre_name[centre]}**
in each week. A high point is a week with many orders (often a week with a promotion or a lower price).
This meal has orders in **{len(one)} of {n_weeks} weeks**. In the other weeks it was not on the menu, so there is no point.
The smallest week had **{one["num_orders"].min():,}** orders and the biggest had **{one["num_orders"].max():,}**.
'''
)

st.subheader('What we learned from the data')
col1, col2 = st.columns(2)
with col1:
    show_figure('fig03_weekly_total', 'Total orders per week')
    st.write('**What it shows:** all orders of all centres added up, week by week. '
             'The total goes up and down, but there is **no clear yearly pattern**. '
             '*Example: a busy week has about 1.1 million orders, a quiet week about 0.6 million.*')
with col2:
    show_figure('fig15_promotion_boxplot', 'Orders with and without promotion')
    st.write('**What it shows:** how many orders a meal gets with and without a promotion. '
             'Promoted meals sell **almost 3 times more**, so promotions are important for the forecast. '
             '*Example: about 120 orders in a normal week, but about 230-260 with an email or home-page promotion.*')

show_figure('fig10_orders_category_cuisine', 'Orders by meal category and cuisine')
st.write('**What it shows:** the total orders of each meal type and each cuisine over all weeks. '
         'Beverages are by far the most ordered type, so the type of meal also matters for the forecast.')
