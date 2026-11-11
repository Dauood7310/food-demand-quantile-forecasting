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

col1, col2, col3, col4 = st.columns(4)
col1.metric('Rows', f'{len(orders):,}')
col2.metric('Meals', orders['meal_id'].nunique())
col3.metric('Centres', orders['center_id'].nunique())
col4.metric('Weeks', orders['week'].nunique())

st.subheader('A few rows of the data')
preview = orders[['week', 'center_id', 'meal_id', 'category', 'cuisine', 'checkout_price',
                  'emailer_for_promotion', 'homepage_featured', 'num_orders']].head(10)
preview.columns = ['Week', 'Centre', 'Meal', 'Category', 'Cuisine', 'Price',
                   'Email promotion', 'On homepage', 'Orders']
st.dataframe(preview, hide_index=True, width='stretch')

st.subheader('Orders of one meal')
st.write('Pick a centre and a meal to see its weekly orders.')
col1, col2 = st.columns(2)
centre = col1.selectbox('Centre', sorted(orders['center_id'].unique()), index=0, format_func=lambda c: centre_name[c])
meals = sorted(orders.loc[orders['center_id'] == centre, 'meal_id'].unique(), key=lambda m: meal_name[m])
meal = col2.selectbox('Meal', meals, format_func=lambda m: meal_name[m])
one = orders[(orders['center_id'] == centre) & (orders['meal_id'] == meal)]
info = one.iloc[0]
st.caption(f'{meal_name[meal]} at {centre_name[centre]}. '
           f'{len(one)} of 145 weeks have orders (missing weeks = the meal was not on offer). '
           'The data has no real names, so meals and centres are shown by their type and number.')
st.line_chart(one.set_index('week')['num_orders'], x_label='Week', y_label='Orders')

st.subheader('What we learned from the data')
col1, col2 = st.columns(2)
with col1:
    show_figure('fig03_weekly_total', 'Total orders per week')
    st.write('Total orders go up and down from week to week, but there is **no clear yearly pattern**.')
with col2:
    show_figure('fig15_promotion_boxplot', 'Orders with and without promotion')
    st.write('Promoted meals sell **almost 3 times more**, so promotions are important for the forecast.')

show_figure('fig10_orders_category_cuisine', 'Orders by meal category and cuisine')
st.write('Beverages are by far the most ordered category, so the type of meal also matters.')
