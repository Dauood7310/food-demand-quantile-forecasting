# matplotlib settings for the report figures (IEEE two column template)
# one column is 3.5 inch wide and the full page width is 7.16 inch
# figures are made at that size so I don't have to resize them in word

import os
import matplotlib.pyplot as plt

ONE_COL = 3.5
TWO_COL = 7.16

FIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'figures')


def set_ieee_style():
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.size'] = 8
    plt.rcParams['axes.labelsize'] = 8
    plt.rcParams['axes.titlesize'] = 8
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['figure.titlesize'] = 9
    plt.rcParams['figure.titleweight'] = 'bold'
    plt.rcParams['xtick.labelsize'] = 7
    plt.rcParams['ytick.labelsize'] = 7
    plt.rcParams['legend.fontsize'] = 7
    plt.rcParams['legend.frameon'] = False

    plt.rcParams['lines.linewidth'] = 1
    plt.rcParams['axes.linewidth'] = 0.6
    plt.rcParams['xtick.major.width'] = 0.6
    plt.rcParams['ytick.major.width'] = 0.6
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.grid'] = True
    plt.rcParams['axes.axisbelow'] = True
    plt.rcParams['grid.color'] = '#e0e0e0'
    plt.rcParams['grid.linewidth'] = 0.4

    plt.rcParams['figure.dpi'] = 150   # only for showing in the notebook
    plt.rcParams['figure.constrained_layout.use'] = True


def save_fig(fig, name):
    # 600 dpi png, good enough for print
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, name + '.png'), dpi=600)
