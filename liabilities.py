import numpy as np
import pandas as pd
from yield_curve import spot_rate, nelson_siegel_spot


# ── Cash flow generation ──────────────────────────────────────────────────────

def generate_pension_cashflows(
    annual_payment: float = 1_000_000,
    horizon: int = 35,
    cpi_indexation: float = 0.0,
    start_year: int = 1,
) -> pd.DataFrame:
    """
    Generate a schedule of annual pension benefit cash flows.

    Parameters
    ----------
    annual_payment : float — base annual payment (Year 1)
    horizon        : int   — years of payments (e.g. 35 for 30–40 yr horizon)
    cpi_indexation : float — annual CPI escalation rate (e.g. 0.02 for 2%)
    start_year     : int   — first payment year (default 1 = end of year 1)

    Returns
    -------
    DataFrame with columns: year, maturity, cash_flow
    """
    years = np.arange(start_year, start_year + horizon)
    cfs = annual_payment * (1 + cpi_indexation) ** (years - start_year)
    return pd.DataFrame({
        'year':      years,
        'maturity':  years.astype(float),
        'cash_flow': cfs,
    })


# ── Present value & duration ──────────────────────────────────────────────────

def pv_cashflows(cashflows: pd.DataFrame, curve_params: dict) -> pd.DataFrame:
    """
    Discount each cash flow using the continuously-compounded spot curve.

    PV_t = CF_t * exp(-r(t) * t)

    Adds columns: spot_rate, discount_factor, pv
    """
    df = cashflows.copy()
    df['spot_rate'] = df['maturity'].apply(lambda t: spot_rate(t, curve_params))
    df['discount_factor'] = np.exp(-df['spot_rate'] * df['maturity'])
    df['pv'] = df['cash_flow'] * df['discount_factor']
    return df


def liability_pv(cashflows: pd.DataFrame, curve_params: dict) -> float:
    """Total present value of liability cash flows."""
    return pv_cashflows(cashflows, curve_params)['pv'].sum()


def macaulay_duration(cashflows: pd.DataFrame, curve_params: dict) -> float:
    """
    Macaulay duration of the liability stream (continuously compounded).

    D_mac = sum(t * PV_t) / PV_total
    """
    df = pv_cashflows(cashflows, curve_params)
    total_pv = df['pv'].sum()
    if total_pv == 0:
        return 0.0
    return float((df['maturity'] * df['pv']).sum() / total_pv)


def modified_duration(cashflows: pd.DataFrame, curve_params: dict) -> float:
    """
    Modified duration = Macaulay duration (for continuous compounding D_mod = D_mac).
    For continuous compounding: dPV/dr = -D * PV, so D_mod = D_mac.
    """
    return macaulay_duration(cashflows, curve_params)


def dv01_liability(cashflows: pd.DataFrame, curve_params: dict,
                   bump: float = 0.0001) -> float:
    """
    DV01: change in PV for a 1 bp parallel shift in the yield curve.
    Computed via finite difference.
    """
    from yield_curve import SCENARIOS
    import copy

    params_up = copy.deepcopy(curve_params)
    params_dn = copy.deepcopy(curve_params)
    params_up['beta0'] += bump
    params_dn['beta0'] -= bump

    pv_up = liability_pv(cashflows, params_up)
    pv_dn = liability_pv(cashflows, params_dn)
    return (pv_dn - pv_up) / 2  # DV01 is positive when rates rise → PV falls


def convexity_liability(cashflows: pd.DataFrame, curve_params: dict) -> float:
    """
    Convexity of liability cash flows (continuous compounding).

    C = sum(t^2 * PV_t) / PV_total
    """
    df = pv_cashflows(cashflows, curve_params)
    total_pv = df['pv'].sum()
    if total_pv == 0:
        return 0.0
    return float(((df['maturity'] ** 2) * df['pv']).sum() / total_pv)


# ── Inflation sensitivity ─────────────────────────────────────────────────────

def inflation_sensitivity(
    annual_payment: float,
    horizon: int,
    curve_params: dict,
    cpi_base: float = 0.02,
    cpi_shocked: float = 0.03,
) -> dict:
    """
    Compare PV of liabilities under two CPI assumptions.
    Returns base PV, shocked PV, and change.
    """
    cf_base    = generate_pension_cashflows(annual_payment, horizon, cpi_base)
    cf_shocked = generate_pension_cashflows(annual_payment, horizon, cpi_shocked)
    pv_base    = liability_pv(cf_base, curve_params)
    pv_shocked = liability_pv(cf_shocked, curve_params)
    return {
        'PV (base CPI)':    pv_base,
        'PV (shocked CPI)': pv_shocked,
        'Change ($)':       pv_shocked - pv_base,
        'Change (%)':       (pv_shocked - pv_base) / pv_base * 100,
    }


# ── Rate scenario analysis ────────────────────────────────────────────────────

def scenario_pv_table(cashflows: pd.DataFrame, base_params: dict) -> pd.DataFrame:
    """
    PV of liabilities under each standard yield curve scenario.
    """
    from yield_curve import SCENARIOS, apply_scenario
    import copy

    rows = []
    base_pv = liability_pv(cashflows, base_params)

    for name in SCENARIOS:
        shocked_rates = apply_scenario(cashflows['maturity'].values, base_params, name)
        # Build shocked param set via direct rate overrides per maturity
        # (apply scenario adjusts NS params — reuse spot_rate via NS)
        from yield_curve import fit_nelson_siegel
        # Re-fit NS to scenario curve to get shocked params
        shocked_params = fit_nelson_siegel(cashflows['maturity'].values, shocked_rates)
        pv = liability_pv(cashflows, shocked_params)
        rows.append({
            'Scenario':   name,
            'PV ($M)':    round(pv / 1e6, 3),
            'Change ($M)': round((pv - base_pv) / 1e6, 3),
            'Change (%)': round((pv - base_pv) / base_pv * 100, 2),
        })

    return pd.DataFrame(rows).set_index('Scenario')
