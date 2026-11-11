# Home page of the app. Run from the code folder:  streamlit run app/Home.py

import streamlit as st

st.set_page_config(page_title='Meal Demand Forecasting', page_icon='🍲', layout='wide')

st.title('Meal Demand Forecasting for Fresh Food')
st.caption('MSc project (COM748) - Mohammad Dauood - Supervisor: Dr Serif Salem')

st.header('The problem')
st.write(
    'A meal delivery company must order fresh ingredients every week, **before** it knows how many '
    'meals customers will order.'
)
col1, col2 = st.columns(2)
col1.error('**Order too much** - the extra fresh food goes to waste.')
col2.warning('**Order too little** - meals run out and customers are let down.')
st.write(
    'This project uses machine learning to forecast how many meals will be ordered 1 to 10 weeks ahead, '
    'and then checks which forecast leads to the **lowest cost of waste and missed orders**.'
)

st.header('Three questions and the answers')

st.subheader('1. Are machine learning models more accurate than simple methods?')
st.write(
    '**Yes.** The best models (LightGBM and XGBoost) have an error of about **28.6%**, compared with **32.0%** for the '
    'best simple method (Linear Regression). That is about **10% lower error**.'
)

st.subheader('2. Can quantile models forecast a "safe" amount, not only the most likely amount?')
st.write(
    '**Yes.** A quantile forecast of 0.8 means: in about 8 out of 10 weeks the real orders stay below it. '
    'The models hit these targets closely (for 0.8, real orders were below the forecast in 77-80% of weeks).'
)

st.subheader('3. Does ordering the right quantile save money?')
st.write(
    '**Yes.** When a missed order costs 3 times more than a wasted meal, ordering the right quantile '
    'is **25% cheaper** than ordering the normal forecast. When it costs 5 times more, it is **40% cheaper**.'
)

st.header('How to use this app')
st.markdown(
    '''
Use the menu on the left:

- **Order planner** - pick a centre, a meal and your costs, and get a suggested order for each week
- **The data** - what the dataset looks like, and the orders of any meal
- **Accuracy results** - which model forecasts best (question 1)
- **Quantile results** - forecasting a safe amount (question 2)
- **Cost results** - waste, missed orders and cost (question 3)
'''
)
st.info('All costs are simulated. The dataset has no real cost data, so the results are estimates, not real savings.')
