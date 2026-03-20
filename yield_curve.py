import numpy as np
import pandas as pd
from scipy.optimize import minimize
import requests


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


# ── Government of Canada live yield curve ────────────────────────────────────
# Live data from Bank of Canada Valet API.
# Yields are quoted as annual % with semi-annual compounding (standard bond convention).
# Converted to continuously compounded decimal for Nelson-Siegel fitting.

_BOC_SERIES = {
    'TB.CDN.3MTH.DQ.YLD':  0.25,
    'TB.CDN.6MTH.DQ.YLD':  0.50,
    'TB.CDN.1YR.DQ.YLD':   1.0,
    'BD.CDN.2YR.DQ.YLD':   2.0,
    'BD.CDN.3YR.DQ.YLD':   3.0,
    'BD.CDN.5YR.DQ.YLD':   5.0,
    'BD.CDN.7YR.DQ.YLD':   7.0,
    'BD.CDN.10YR.DQ.YLD':  10.0,
    'BD.CDN.LONG.DQ.YLD':  30.0,
}

# Fallback hardcoded curve (approximate GoC yields, early 2024)
_FALLBACK_MATURITIES = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 30.0])
_FALLBACK_YIELDS     = np.array([0.0490, 0.0488, 0.0468, 0.0430, 0.0415, 0.0390,
                                  0.0385, 0.0375, 0.0368])

GOC_MATURITIES = _FALLBACK_MATURITIES
GOC_YIELDS     = _FALLBACK_YIELDS


def _semi_annual_to_cc(rate_pct: float) -> float:
    """Convert annual yield quoted with semi-annual compounding (%) to continuously compounded decimal."""
    return 2.0 * np.log(1.0 + rate_pct / 200.0)


def fetch_goc_yields() -> tuple[np.ndarray, np.ndarray, str]:
    """
    Fetch live Government of Canada benchmark yields from the Bank of Canada Valet API.

    Returns
    -------
    maturities : np.ndarray — years to maturity
    yields_cc  : np.ndarray — continuously compounded decimal yields
    as_of      : str        — date of the data
    """
    series_str = ','.join(_BOC_SERIES.keys())
    url = f'https://www.bankofcanada.ca/valet/observations/{series_str}/json?recent=5'

    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    # Walk back through recent observations to find the latest with all series populated
    observations = data.get('observations', [])
    for obs in reversed(observations):
        maturities, yields_cc = [], []
        for series, mat in _BOC_SERIES.items():
            val = obs.get(series, {}).get('v')
            if val is not None and val != '':
                maturities.append(mat)
                yields_cc.append(_semi_annual_to_cc(float(val)))
        if len(maturities) >= 6:  # enough points to fit NS
            as_of = obs.get('d', 'unknown')
            return np.array(maturities), np.array(yields_cc), as_of

    raise ValueError('No complete GoC yield observation found in recent data.')


def default_goc_params() -> dict:
    """
    Fit Nelson-Siegel to live GoC yields from Bank of Canada.
    Falls back to hardcoded curve if the API is unavailable.
    Returns params dict with an extra 'as_of' and 'live' key.
    """
    try:
        mats, yields, as_of = fetch_goc_yields()
        params = fit_nelson_siegel(mats, yields)
        params['as_of'] = as_of
        params['live']  = True
        params['_maturities'] = mats
        params['_yields']     = yields
        return params
    except Exception:
        params = fit_nelson_siegel(_FALLBACK_MATURITIES, _FALLBACK_YIELDS)
        params['as_of'] = 'Fallback (early 2024)'
        params['live']  = False
        params['_maturities'] = _FALLBACK_MATURITIES
        params['_yields']     = _FALLBACK_YIELDS
        return params


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
