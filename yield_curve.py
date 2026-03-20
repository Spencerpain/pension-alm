import numpy as np
import pandas as pd
from scipy.optimize import minimize


# ── Nelson-Siegel model ───────────────────────────────────────────────────────

def nelson_siegel_spot(maturities: np.ndarray, beta0: float, beta1: float,
                       beta2: float, tau: float) -> np.ndarray:
    """
    Nelson-Siegel spot rate curve (continuously compounded, decimal).

    r(t) = beta0
           + beta1 * (1 - exp(-t/tau)) / (t/tau)
           + beta2 * [(1 - exp(-t/tau)) / (t/tau) - exp(-t/tau)]
    """
    t = np.asarray(maturities, dtype=float)
    lam = t / tau
    factor1 = (1 - np.exp(-lam)) / lam
    factor2 = factor1 - np.exp(-lam)
    return beta0 + beta1 * factor1 + beta2 * factor2


def fit_nelson_siegel(maturities: np.ndarray, yields: np.ndarray) -> dict:
    """
    Fit Nelson-Siegel to observed par/spot yields (decimal, continuously compounded).
    Returns dict with beta0, beta1, beta2, tau, and fitted curve.
    """
    def objective(params):
        b0, b1, b2, tau = params
        if tau <= 0:
            return 1e10
        fitted = nelson_siegel_spot(maturities, b0, b1, b2, tau)
        return float(np.sum((fitted - yields) ** 2))

    x0 = [0.04, -0.02, 0.01, 2.0]
    bounds = [(0.001, 0.20), (-0.15, 0.15), (-0.15, 0.15), (0.1, 30.0)]
    res = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
    b0, b1, b2, tau = res.x

    return {
        'beta0': b0,
        'beta1': b1,
        'beta2': b2,
        'tau':   tau,
        'rmse':  float(np.sqrt(res.fun / len(maturities))),
    }


def spot_rate(maturity: float, params: dict) -> float:
    """Spot rate for a single maturity given NS params dict."""
    return float(nelson_siegel_spot(
        np.array([maturity]),
        params['beta0'], params['beta1'], params['beta2'], params['tau']
    )[0])


# ── Government of Canada representative yield curve ───────────────────────────
# Approximate GoC nominal yield curve as of early 2024 (illustrative defaults).
# Users can override by providing observed maturities and yields.

GOC_MATURITIES = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0])
GOC_YIELDS = np.array([0.0490, 0.0488, 0.0468, 0.0430, 0.0415, 0.0390,
                        0.0385, 0.0375, 0.0370, 0.0368])  # continuously compounded


def default_goc_params() -> dict:
    """Fit Nelson-Siegel to the representative GoC curve."""
    return fit_nelson_siegel(GOC_MATURITIES, GOC_YIELDS)


# ── Scenario shocks ───────────────────────────────────────────────────────────

SCENARIOS = {
    'Base':             {'parallel': 0.000, 'slope': 0.000, 'curvature': 0.000},
    '+100 bp parallel': {'parallel': 0.010, 'slope': 0.000, 'curvature': 0.000},
    '-100 bp parallel': {'parallel':-0.010, 'slope': 0.000, 'curvature': 0.000},
    '+200 bp parallel': {'parallel': 0.020, 'slope': 0.000, 'curvature': 0.000},
    '-200 bp parallel': {'parallel':-0.020, 'slope': 0.000, 'curvature': 0.000},
    'Steepener':        {'parallel': 0.000, 'slope': 0.010, 'curvature': 0.000},
    'Flattener':        {'parallel': 0.000, 'slope':-0.010, 'curvature': 0.000},
    'Humped':           {'parallel': 0.000, 'slope': 0.000, 'curvature': 0.010},
}


def apply_scenario(maturities: np.ndarray, base_params: dict, scenario: str) -> np.ndarray:
    """
    Apply a named rate scenario to the base Nelson-Siegel curve.

    Parallel: shifts beta0 (long-run level).
    Slope:    shifts beta1 (short-end component).
    Curvature: shifts beta2 (medium-term hump).
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario '{scenario}'. Choose from: {list(SCENARIOS)}")
    shock = SCENARIOS[scenario]
    params = {
        'beta0': base_params['beta0'] + shock['parallel'],
        'beta1': base_params['beta1'] + shock['slope'],
        'beta2': base_params['beta2'] + shock['curvature'],
        'tau':   base_params['tau'],
    }
    return nelson_siegel_spot(maturities, **{k: params[k] for k in ['beta0','beta1','beta2','tau']})


def curve_dataframe(maturities: np.ndarray, base_params: dict,
                    scenarios: list = None) -> pd.DataFrame:
    """
    Return a DataFrame with spot rates for each scenario across maturities.
    Columns = scenario names, index = maturities.
    """
    if scenarios is None:
        scenarios = list(SCENARIOS)
    data = {}
    for sc in scenarios:
        data[sc] = apply_scenario(maturities, base_params, sc)
    return pd.DataFrame(data, index=maturities)
