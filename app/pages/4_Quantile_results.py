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
    'Ordering this amount means running out only in about 2 out of 10 weeks.  \n'
    'With numbers: the normal forecast says "100 meals". The 0.8 forecast may say "130 meals", '
    'and the 0.9 forecast "160 meals". A higher level is a bigger, safer order.'
)
st.write('We trained three quantile models for the levels 0.3, 0.4, ... 0.9: '
         '**Linear quantile regression**, **quantile XGBoost** and **quantile LightGBM**.')

st.subheader('Do the forecasts hit their target?',
             help='A level of 0.8 promises to be enough in 80% of weeks. Here we check if that promise is kept.')
coverage = load_table('06_coverage_mean.csv')
tau = st.select_slider(
    'Pick a quantile level', options=coverage['tau'].tolist(), value=0.8,
    help='The safety level you want to check. 0.5 = the middle (enough in half of the weeks). '
         '0.8 = enough in about 8 of 10 weeks. 0.9 = enough in about 9 of 10 weeks.')
row = coverage[coverage['tau'] == tau].iloc[0]
target = tau * 100

st.write(f'Target: real orders should be below the forecast in **{target:.0f}%** of weeks.')
tile_help = ('Coverage: in how many weeks out of 100 the real orders were below this forecast, '
             'so the forecast was enough. It should be close to the target.')
col1, col2, col3, col4 = st.columns(4)
col1.metric('Linear quantile regression', f'{row["Linear QR"]:.1f}%', help=tile_help)
col2.metric('Quantile XGBoost', f'{row["XGBoost"]:.1f}%', help=tile_help)
col3.metric('Quantile LightGBM', f'{row["LightGBM"]:.1f}%', help=tile_help)
col4.metric('Normal LightGBM forecast', f'{row["LightGBM point"]:.1f}%',
            help='The normal forecast gives one number only, so its coverage is the same for every level.')


def gap_text(value):
    # how far a model is from the target, in words
    gap = value - target
    if abs(gap) < 0.5:
        return 'right on the target'
    return f'{abs(gap):.1f} points {"above" if gap > 0 else "below"} the target'


st.markdown(
    f'''
**What these four numbers mean**

Each number is the **coverage**: in how many weeks out of 100 the real orders were below the forecast
(the forecast was enough). You picked level **{tau}**, so a good model should be close to **{target:.0f}%**.

- **Linear quantile regression: {row["Linear QR"]:.1f}%** - the simple quantile model. Its forecast was enough in
  about {row["Linear QR"]:.0f} of 100 weeks. That is {gap_text(row["Linear QR"])}.
- **Quantile XGBoost: {row["XGBoost"]:.1f}%** - a tree model. Enough in about {row["XGBoost"]:.0f} of 100 weeks,
  {gap_text(row["XGBoost"])}.
- **Quantile LightGBM: {row["LightGBM"]:.1f}%** - a tree model, the main model of this project. Enough in about
  {row["LightGBM"]:.0f} of 100 weeks, {gap_text(row["LightGBM"])}.
- **Normal LightGBM forecast: {row["LightGBM point"]:.1f}%** - the one normal forecast from the Accuracy page.
  It is always the same number, so it is enough in only about {row["LightGBM point"]:.0f} of 100 weeks,
  whatever level you need. It cannot be made "safer".

*Example: you want an order that is enough in {target:.0f} of 100 weeks. Quantile LightGBM gives a number that was
enough in {row["LightGBM"]:.0f} of 100 weeks. The normal forecast was enough in only {row["LightGBM point"]:.0f}.*
'''
)

col1, col2 = st.columns([1, 1])
with col1:
    show_figure('fig27_quantile_coverage', 'Target level vs real result (close to the line = good)')
    st.write(
        '**How to read it:** the bottom axis is the level we asked for (0.3 to 0.9). The side axis is the coverage we '
        'really got. The grey line is a perfect model. All three models stay close to the line, so their levels '
        'can be trusted.'
    )
with col2:
    show_figure('fig26_pinball_by_tau', 'Quantile error (pinball loss), lower is better')
    st.write(
        '**How to read it:** the **pinball loss** is the error score of a quantile forecast. Lower is better. '
        'For a safe level it punishes a forecast that is too low more than one that is too high. '
        '*Example: at level 0.8, being 10 meals too low costs 8, but being 10 meals too high costs only 2.*'
    )
    st.write(
        'The quantile models have a much **lower quantile error** than the normal forecast (black dotted line), '
        'especially for the high levels (0.8 and 0.9) that a kitchen needs for a safe order.'
    )

st.subheader('Example: one meal at one centre')
show_figure('fig28_example_quantiles', 'Forecasts made at week 135 for weeks 136-145')
st.write(
    '**How to read it:** the black line is what customers really ordered each week. Everything right of the dashed '
    'orange line (week 135) is the forecast. The blue band shows the range from level 0.3 (low) to 0.9 (high). '
    'The blue line is the middle (0.5) and the red dashed line is the "safe" 0.8 forecast. '
    'If the black line stays inside the band, the forecast range was right.'
)

st.success(
    '**Answer: yes.** The quantile models hit their targets closely, and the tree models (LightGBM and XGBoost) are the '
    'most accurate. Their quantile error is about **20% lower** on average than using the normal forecast.'
)
