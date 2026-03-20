import numpy as np
import pandas as pd
from scipy.optimize import minimize
from yield_curve import spot_rate
from assets import bond_price, bond_macaulay_duration, bond_convexity
from liabilities import liability_pv, macaulay_duration


# ── LDI Duration Matching ─────────────────────────────────────────────────────

def duration_match_weights(
    bonds: list,
    cashflows,
    curve_params: dict,
    target_budget: float = None,
) -> dict:
    """
    Find bond portfolio weights that minimise the duration gap:
        |D_A(w) - D_L|

    subject to:
        - weights sum to 1 (fully invested)
        - weights >= 0 (long only)
        - optional: total MV ≤ target_budget

    Parameters
    ----------
    bonds          : list of bond dicts
    cashflows      : DataFrame of liability cash flows
    curve_params   : dict — Nelson-Siegel params
    target_budget  : float — if None, uses current total MV

    Returns
    -------
    dict with weights, resulting asset duration, liability duration, duration gap
    """
    n = len(bonds)

    # Pre-compute individual bond prices and durations
    prices = np.array([bond_price(b['face'], b['coupon'], b['maturity'], curve_params) for b in bonds])
    durs   = np.array([bond_macaulay_duration(b['face'], b['coupon'], b['maturity'], curve_params) for b in bonds])

    d_liab = macaulay_duration(cashflows, curve_params)

    def portfolio_dur_w(w):
        """Weighted average duration given weight vector."""
        return float(np.dot(w, durs))

    def objective(w):
        return (portfolio_dur_w(w) - d_liab) ** 2

    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(0.0, 1.0)] * n
    w0 = np.ones(n) / n

    res = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=constraints)
    w_opt = res.x

    # Compute dollar allocations
    if target_budget is None:
        target_budget = prices.sum()  # keep existing total MV

    allocations = w_opt * target_budget

    return {
        'weights':           w_opt,
        'allocations ($)':   allocations,
        'asset_duration':    portfolio_dur_w(w_opt),
        'liability_duration': d_liab,
        'duration_gap':      portfolio_dur_w(w_opt) - d_liab,
        'converged':         res.success,
    }


def ldi_summary_table(bonds: list, cashflows, curve_params: dict) -> pd.DataFrame:
    """
    Compare current portfolio vs duration-matched portfolio.
    """
    from assets import portfolio_duration, portfolio_mv

    result = duration_match_weights(bonds, cashflows, curve_params)
    curr_dur = portfolio_duration(bonds, curve_params)
    d_liab   = macaulay_duration(cashflows, curve_params)
    curr_gap = curr_dur - d_liab

    rows = [
        {
            'Portfolio':        'Current',
            'Asset Duration':   round(curr_dur, 2),
            'Liability Duration': round(d_liab, 2),
            'Duration Gap':     round(curr_gap, 2),
        },
        {
            'Portfolio':        'Duration Matched',
            'Asset Duration':   round(result['asset_duration'], 2),
            'Liability Duration': round(d_liab, 2),
            'Duration Gap':     round(result['duration_gap'], 4),
        },
    ]
    return pd.DataFrame(rows).set_index('Portfolio')


def ldi_bond_allocations(bonds: list, cashflows, curve_params: dict,
                         total_budget: float = None) -> pd.DataFrame:
    """
    Return bond-level LDI allocation for duration matching.
    """
    from assets import portfolio_mv

    if total_budget is None:
        total_budget = portfolio_mv(bonds, curve_params)

    result = duration_match_weights(bonds, cashflows, curve_params, total_budget)
    weights = result['weights']
    allocs  = result['allocations ($)']

    rows = []
    for i, b in enumerate(bonds):
        rows.append({
            'Bond':           b['name'],
            'Maturity (yr)':  b['maturity'],
            'Weight (%)':     round(weights[i] * 100, 2),
            'Allocation ($M)': round(allocs[i] / 1e6, 3),
        })
    return pd.DataFrame(rows).set_index('Bond')
