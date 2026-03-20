import numpy as np
import pandas as pd
import copy
from yield_curve import apply_scenario, fit_nelson_siegel, SCENARIOS
from liabilities import liability_pv, macaulay_duration, convexity_liability, dv01_liability
from assets import portfolio_mv, portfolio_duration, portfolio_convexity


# ── Surplus & funding ratio ───────────────────────────────────────────────────

def surplus(asset_mv: float, liability_pv: float) -> float:
    """Surplus = Assets - PV(Liabilities). Positive = overfunded."""
    return asset_mv - liability_pv


def funding_ratio(asset_mv: float, liability_pv_val: float) -> float:
    """Funding ratio = Assets / PV(Liabilities). 1.0 = fully funded."""
    if liability_pv_val == 0:
        return np.nan
    return asset_mv / liability_pv_val


# ── Duration gap ──────────────────────────────────────────────────────────────

def duration_gap(asset_dur: float, liability_dur: float,
                 asset_mv: float, liability_pv_val: float) -> float:
    """
    Duration gap = D_A - (PV_L / MV_A) * D_L

    A positive gap means assets are more rate-sensitive than liabilities.
    A gap of 0 means rate changes affect both sides equally (immunised).
    """
    if asset_mv == 0:
        return np.nan
    return asset_dur - (liability_pv_val / asset_mv) * liability_dur


# ── Rate sensitivity (DV01 of surplus) ───────────────────────────────────────

def surplus_dv01(bonds: list, cashflows, curve_params: dict,
                 bump: float = 0.0001) -> float:
    """
    Change in surplus for a 1 bp parallel rate increase.

    dSurplus/dr ≈ dMV_A/dr - dPV_L/dr
    Both computed via finite difference (parallel bump).
    """
    params_up = copy.deepcopy(curve_params)
    params_dn = copy.deepcopy(curve_params)
    params_up['beta0'] += bump
    params_dn['beta0'] -= bump

    mv_up = portfolio_mv(bonds, params_up)
    mv_dn = portfolio_mv(bonds, params_dn)
    pv_up = liability_pv(cashflows, params_up)
    pv_dn = liability_pv(cashflows, params_dn)

    surplus_up = mv_up - pv_up
    surplus_dn = mv_dn - pv_dn
    return (surplus_up - surplus_dn) / 2  # positive = surplus rises when rates rise


# ── ALM scenario dashboard ────────────────────────────────────────────────────

def alm_scenario_table(bonds: list, cashflows, base_params: dict) -> pd.DataFrame:
    """
    Full ALM dashboard across all rate scenarios:
    Asset MV, Liability PV, Surplus, Funding Ratio.
    """
    rows = []
    mats_bonds = np.array([b['maturity'] for b in bonds])
    mats_liab  = cashflows['maturity'].values

    for name in SCENARIOS:
        # Shocked rates for assets
        shocked_a = apply_scenario(mats_bonds, base_params, name)
        params_a  = fit_nelson_siegel(mats_bonds, shocked_a)

        # Shocked rates for liabilities
        shocked_l = apply_scenario(mats_liab, base_params, name)
        params_l  = fit_nelson_siegel(mats_liab, shocked_l)

        mv   = portfolio_mv(bonds, params_a)
        pv_l = liability_pv(cashflows, params_l)
        sur  = surplus(mv, pv_l)
        fr   = funding_ratio(mv, pv_l)

        rows.append({
            'Scenario':       name,
            'Asset MV ($M)':  round(mv / 1e6, 3),
            'Liability PV ($M)': round(pv_l / 1e6, 3),
            'Surplus ($M)':   round(sur / 1e6, 3),
            'Funding Ratio':  round(fr, 4),
        })

    return pd.DataFrame(rows).set_index('Scenario')


def alm_summary(bonds: list, cashflows, curve_params: dict) -> dict:
    """High-level ALM metrics under the base curve."""
    mv   = portfolio_mv(bonds, curve_params)
    pv_l = liability_pv(cashflows, curve_params)
    d_a  = portfolio_duration(bonds, curve_params)
    d_l  = macaulay_duration(cashflows, curve_params)
    c_a  = portfolio_convexity(bonds, curve_params)
    c_l  = convexity_liability(cashflows, curve_params)
    gap  = duration_gap(d_a, d_l, mv, pv_l)
    sdv01 = surplus_dv01(bonds, cashflows, curve_params)

    return {
        'Asset MV ($M)':        round(mv / 1e6, 3),
        'Liability PV ($M)':    round(pv_l / 1e6, 3),
        'Surplus ($M)':         round((mv - pv_l) / 1e6, 3),
        'Funding Ratio':        round(funding_ratio(mv, pv_l), 4),
        'Asset Duration':       round(d_a, 2),
        'Liability Duration':   round(d_l, 2),
        'Duration Gap':         round(gap, 2),
        'Asset Convexity':      round(c_a, 2),
        'Liability Convexity':  round(c_l, 2),
        'Surplus DV01 ($)':     round(sdv01, 0),
    }
