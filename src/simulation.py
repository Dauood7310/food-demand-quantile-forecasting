# Order quantity and cost simulation (used in notebook 07 and the Streamlit app).
#
# The kitchen orders q portions for a week and the real demand is y:
#   waste    = q - y  if we ordered too much  (each portion costs c_o)
#   shortage = y - q  if we ordered too little (each portion costs c_u)
#   cost     = c_o * waste + c_u * shortage
#
# Newsvendor rule: the cheapest order is the tau* quantile of demand, tau* = c_u / (c_u + c_o).

import numpy as np

TAUS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
COST_RATIOS = [0.5, 1, 1.5, 2, 3, 4, 5]     # r = c_u / c_o, with c_o = 1


def critical_ratio(r):
    """tau* for cost ratio r = c_u / c_o."""
    return r / (r + 1)


def quantity_at_tau(quantiles, tau_star, taus=TAUS):
    """Order quantity at tau* from the forecast quantiles.

    quantiles: array (rows x len(taus)), one column per tau, already sorted.
    tau* is usually between two of our levels (e.g. 0.75 is between 0.7 and 0.8),
    so we take the straight line between the two neighbouring quantiles.
    """
    quantiles = np.asarray(quantiles, dtype=float)
    tau_star = min(max(tau_star, taus[0]), taus[-1])     # stay inside 0.3 - 0.9

    i = int(np.searchsorted(taus, tau_star))
    if np.isclose(taus[i], tau_star):                    # tau* is one of our levels
        return quantiles[:, i]

    low, high = taus[i - 1], taus[i]
    weight = (tau_star - low) / (high - low)             # e.g. 0.75 -> half way between 0.7 and 0.8
    return quantiles[:, i - 1] + weight * (quantiles[:, i] - quantiles[:, i - 1])


def simulate(y, q, c_o=1.0, c_u=1.0):
    """Waste, shortage and cost of ordering q when the demand is y (whole portions)."""
    y = np.asarray(y, dtype=float)
    q = np.round(np.asarray(q, dtype=float))       # the kitchen orders whole portions
    waste = np.maximum(q - y, 0)
    shortage = np.maximum(y - q, 0)
    return {
        'Waste': waste.sum(),
        'Shortage': shortage.sum(),
        'Cost': c_o * waste.sum() + c_u * shortage.sum(),
        'Waste %': waste.sum() / y.sum() * 100,            # wasted portions as % of demand
        'Shortage %': shortage.sum() / y.sum() * 100,      # missed portions as % of demand
        'Service level %': np.mean(shortage == 0) * 100,   # share of weeks with no shortage
    }
