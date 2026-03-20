import numpy as np
import pandas as pd
from yield_curve import nelson_siegel_spot, default_goc_params
from liabilities import liability_pv, generate_pension_cashflows
from assets import portfolio_mv


# ── Yield curve simulation (Vasicek / parallel random walk) ──────────────────

def simulate_yield_curves(
    base_params: dict,
    n_sims: int = 1000,
    horizon: float = 1.0,
    rate_vol: float = 0.010,
    mean_reversion: float = 0.15,
    seed: int = 42,
) -> list:
    """
    Simulate yield curve scenarios using a simplified 1-factor model applied
    to the Nelson-Siegel level parameter (beta0) — captures parallel shifts.

    Vasicek-style: Δbeta0 ~ N(κ*(θ - beta0)*dt, σ*sqrt(dt))

    Returns a list of `n_sims` NS param dicts.
    """
    rng = np.random.default_rng(seed)
    b0 = base_params['beta0']
    theta = b0  # long-run mean = current level
    dt = horizon

    sim_b0 = b0 + mean_reversion * (theta - b0) * dt + rate_vol * np.sqrt(dt) * rng.standard_normal(n_sims)
    sim_b0 = np.clip(sim_b0, 0.001, 0.20)

    sim_params = []
    for s in sim_b0:
        p = base_params.copy()
        p['beta0'] = s
        sim_params.append(p)
    return sim_params


def simulate_funding_ratios(
    bonds: list,
    annual_payment: float,
    horizon_liab: int,
    cpi_indexation: float,
    base_params: dict,
    n_sims: int = 1000,
    rate_vol: float = 0.010,
    seed: int = 42,
) -> np.ndarray:
    """
    Monte Carlo simulation of the funding ratio distribution.

    For each simulated yield curve:
        FR = MV_Assets(curve) / PV_Liabilities(curve)

    Returns array of shape (n_sims,).
    """
    cashflows = generate_pension_cashflows(annual_payment, horizon_liab, cpi_indexation)
    sim_params = simulate_yield_curves(base_params, n_sims, rate_vol=rate_vol, seed=seed)

    frs = np.zeros(n_sims)
    for i, p in enumerate(sim_params):
        mv   = portfolio_mv(bonds, p)
        pv_l = liability_pv(cashflows, p)
        frs[i] = mv / pv_l if pv_l > 0 else np.nan

    return frs


def funding_ratio_stats(frs: np.ndarray, confidence: float = 0.95) -> dict:
    """
    Summary statistics of the funding ratio distribution.
    """
    frs_clean = frs[~np.isnan(frs)]
    var_level = np.percentile(frs_clean, (1 - confidence) * 100)

    return {
        'Mean FR':           round(float(np.mean(frs_clean)), 4),
        'Median FR':         round(float(np.median(frs_clean)), 4),
        'Std Dev':           round(float(np.std(frs_clean)), 4),
        f'VaR FR ({int(confidence*100)}%)': round(float(var_level), 4),
        'P(FR < 1.0)':       round(float((frs_clean < 1.0).mean()), 4),
        'P(FR < 0.9)':       round(float((frs_clean < 0.9).mean()), 4),
        'Min FR':            round(float(frs_clean.min()), 4),
        'Max FR':            round(float(frs_clean.max()), 4),
    }


def surplus_var(
    bonds: list,
    annual_payment: float,
    horizon_liab: int,
    cpi_indexation: float,
    base_params: dict,
    n_sims: int = 1000,
    confidence: float = 0.95,
    rate_vol: float = 0.010,
    seed: int = 42,
) -> dict:
    """
    Surplus VaR: worst surplus at given confidence level.
    """
    cashflows = generate_pension_cashflows(annual_payment, horizon_liab, cpi_indexation)
    sim_params = simulate_yield_curves(base_params, n_sims, rate_vol=rate_vol, seed=seed)

    surpluses = np.zeros(n_sims)
    for i, p in enumerate(sim_params):
        mv   = portfolio_mv(bonds, p)
        pv_l = liability_pv(cashflows, p)
        surpluses[i] = mv - pv_l

    base_mv   = portfolio_mv(bonds, base_params)
    base_pv_l = liability_pv(cashflows, base_params)
    base_surplus = base_mv - base_pv_l

    var_surplus = np.percentile(surpluses, (1 - confidence) * 100)

    return {
        'Base Surplus ($M)':      round(base_surplus / 1e6, 3),
        f'Surplus VaR {int(confidence*100)}% ($M)': round(var_surplus / 1e6, 3),
        'Expected Surplus ($M)':  round(float(surpluses.mean()) / 1e6, 3),
        'Surplus Std Dev ($M)':   round(float(surpluses.std()) / 1e6, 3),
    }
