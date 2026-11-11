# draws the method overview figure (CRISP-DM steps of this project) for the methodology section
# run from the code folder:  python -m src.framework_figure

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from src.plot_settings import set_ieee_style, save_fig, ONE_COL

set_ieee_style()

# (step name, what happens in it)
steps = [
    ('1. Business understanding', 'weekly ingredient order, forecast 1-10 weeks ahead'),
    ('2. Data understanding', 'Genpact data, 456,548 rows, EDA'),
    ('3. Data preparation', 'cleaning, leak-free features, log target'),
    ('4. Modelling', 'baselines, RF / XGBoost / LightGBM, quantile models'),
    ('5. Evaluation', 'rolling origin (3 folds), RMSE, MAE, WAPE, pinball'),
    ('6. Simulation', 'waste, shortage, cost, cost-ratio sensitivity'),
    ('7. Deployment', 'Streamlit prototype for planners'),
]

fig, ax = plt.subplots(figsize=(ONE_COL, 3.6))
ax.set_xlim(0, 1)
ax.set_ylim(0, len(steps))
ax.axis('off')

box_h = 0.68
for i, (name, detail) in enumerate(steps):
    y = len(steps) - i - 1 + (1 - box_h) / 2       # first step at the top
    color = '#cde2fb' if i not in (3, 5) else '#fde0cc'   # modelling and simulation in orange (the main contribution)
    ax.add_patch(FancyBboxPatch((0.05, y), 0.9, box_h, boxstyle='square,pad=0',
                                facecolor=color, edgecolor='#333333', linewidth=0.6))
    ax.text(0.5, y + box_h * 0.66, name, ha='center', va='center', fontsize=7.5, fontweight='bold')
    ax.text(0.5, y + box_h * 0.28, detail, ha='center', va='center', fontsize=6.5)

    # arrow down to the next step
    if i < len(steps) - 1:
        ax.annotate('', xy=(0.5, y - (1 - box_h) + 0.02), xytext=(0.5, y - 0.02),
                    arrowprops=dict(arrowstyle='-|>', color='#333333', lw=0.7, mutation_scale=7))

ax.set_title('Research Framework (CRISP-DM)')
save_fig(fig, 'fig21_research_framework')
print('saved fig21_research_framework')
