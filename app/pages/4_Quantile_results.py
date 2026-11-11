import streamlit as st

from common import load_table, show_figure

st.set_page_config(page_title='Quantile results', page_icon='📈', layout='wide')
st.title('Quantile results')
st.subheader('Question 2: can we forecast a "safe" amount, not only the most likely amount?')

st.write(
    'A normal forecast gives **one number**: the most likely orders. A **quantile forecast** gives the amount '
    'that the real orders will stay below **most of the time**.'
)
st.info(
    'Example: a quantile forecast of **0.8** is the amount that the real orders stay below in **about 8 out of 10 weeks**. '
    'Ordering this amount means running out only in about 2 out of 10 weeks.'
)
st.write('We trained three quantile models for the levels 0.3, 0.4, ... 0.9: '
         '**Linear quantile regression**, **quantile XGBoost** and **quantile LightGBM**.')

st.subheader('Do the forecasts hit their target?')
coverage = load_table('06_coverage_mean.csv')
tau = st.select_slider('Pick a quantile level', options=coverage['tau'].tolist(), value=0.8)
row = coverage[coverage['tau'] == tau].iloc[0]

st.write(f'Target: real orders should be below the forecast in **{tau * 100:.0f}%** of weeks.')
col1, col2, col3, col4 = st.columns(4)
col1.metric('Linear quantile regression', f'{row["Linear QR"]:.1f}%')
col2.metric('Quantile XGBoost', f'{row["XGBoost"]:.1f}%')
col3.metric('Quantile LightGBM', f'{row["LightGBM"]:.1f}%')
col4.metric('Normal LightGBM forecast', f'{row["LightGBM point"]:.1f}%')
st.caption('The normal forecast is always the same number, so it covers only 46% of weeks, whatever level you need.')

col1, col2 = st.columns([1, 1])
with col1:
    show_figure('fig27_quantile_coverage', 'Target level vs real result (close to the line = good)')
with col2:
    show_figure('fig26_pinball_by_tau', 'Quantile error (pinball loss), lower is better')
    st.write(
        'The quantile models have a much **lower quantile error** than the normal forecast, '
        'especially for the high levels (0.8 and 0.9) that a kitchen needs for a safe order.'
    )

st.subheader('Example: one meal at one centre')
show_figure('fig28_example_quantiles', 'Forecasts made at week 135 for weeks 136-145')
st.write(
    'The blue band shows the range from level 0.3 to 0.9. The blue line is the middle (0.5) and the red dashed line is '
    'the "safe" 0.8 forecast. The black line is what really happened.'
)

st.success(
    '**Answer: yes.** The quantile models hit their targets closely, and the tree models (LightGBM and XGBoost) are the '
    'most accurate. Their quantile error is about **20% lower** on average than using the normal forecast.'
)
