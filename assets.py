import numpy as np
import pandas as pd
import copy
from yield_curve import spot_rate, fit_nelson_siegel, apply_scenario, SCENARIOS


# ── Bond pricing ──────────────────────────────────────────────────────────────

def bond_price(face: float, coupon_rate: float, maturity: float,
               curve_params: dict, frequency: int = 2) -> float:
    """
    Price a fixed-rate bond by discounting each coupon and par using the spot curve.

    Parameters
    ----------
    face        : float — face value
    coupon_rate : float — annual coupon rate (decimal, e.g. 0.04)
    maturity    : float — years to maturity
    curve_params: dict  — Nelson-Siegel parameters
    frequency   : int   — coupon payments per year (2 = semi-annual)

    Returns
    -------
    float — dirty price (no accrued, clean = dirty here since we start at coupon date)
    """
    periods = int(maturity * frequency)
    coupon = face * coupon_rate / frequency
    times = np.arange(1, periods + 1) / frequency

    pvs = coupon * np.exp(-np.array([spot_rate(t, curve_params) for t in times]) * times)
    par_pv = face * np.exp(-spot_rate(maturity, curve_params) * maturity)
    return float(pvs.sum() + par_pv)


def bond_macaulay_duration(face: float, coupon_rate: float, maturity: float,
                            curve_params: dict, frequency: int = 2) -> float:
    """Macaulay duration of a single bond."""
    periods = int(maturity * frequency)
    coupon = face * coupon_rate / frequency
    times = np.arange(1, periods + 1) / frequency

    rates = np.array([spot_rate(t, curve_params) for t in times])
    pvs = coupon * np.exp(-rates * times)

    par_pv = face * np.exp(-spot_rate(maturity, curve_params) * maturity)
    all_times = np.append(times, maturity)
    all_pvs = np.append(pvs, par_pv)

    total_pv = all_pvs.sum()
    if total_pv == 0:
        return 0.0
    return float((all_times * all_pvs).sum() / total_pv)


def bond_convexity(face: float, coupon_rate: float, maturity: float,
                   curve_params: dict, frequency: int = 2) -> float:
    """Convexity of a single bond (continuous compounding)."""
    periods = int(maturity * frequency)
    coupon = face * coupon_rate / frequency
    times = np.arange(1, periods + 1) / frequency

    rates = np.array([spot_rate(t, curve_params) for t in times])
    pvs = coupon * np.exp(-rates * times)

    par_pv = face * np.exp(-spot_rate(maturity, curve_params) * maturity)
    all_times = np.append(times, maturity)
    all_pvs = np.append(pvs, par_pv)

    total_pv = all_pvs.sum()
    if total_pv == 0:
        return 0.0
    return float(((all_times ** 2) * all_pvs).sum() / total_pv)


# ── Bond portfolio ────────────────────────────────────────────────────────────

DEFAULT_PORTFOLIO = [
    {'name': 'GoC 2Y',  'face': 5_000_000, 'coupon': 0.0425, 'maturity': 2.0},
    {'name': 'GoC 5Y',  'face': 8_000_000, 'coupon': 0.0390, 'maturity': 5.0},
    {'name': 'GoC 10Y', 'face': 7_000_000, 'coupon': 0.0375, 'maturity': 10.0},
    {'name': 'GoC 20Y', 'face': 5_000_000, 'coupon': 0.0370, 'maturity': 20.0},
    {'name': 'GoC 30Y', 'face': 5_000_000, 'coupon': 0.0368, 'maturity': 30.0},
]


def portfolio_summary(bonds: list, curve_params: dict) -> pd.DataFrame:
    """
    Compute price, MV, duration, convexity, DV01 for each bond and totals.

    Parameters
    ----------
    bonds : list of dicts with keys: name, face, coupon, maturity
    curve_params : dict — Nelson-Siegel params

    Returns
    -------
    DataFrame with one row per bond + a Totals/Weighted row
    """
    rows = []
    for b in bonds:
        price = bond_price(b['face'], b['coupon'], b['maturity'], curve_params)
        dur   = bond_macaulay_duration(b['face'], b['coupon'], b['maturity'], curve_params)
        conv  = bond_convexity(b['face'], b['coupon'], b['maturity'], curve_params)
        dv01  = price * dur * 0.0001  # ≈ DV01 for continuous compounding
        rows.append({
            'Bond':         b['name'],
            'Face ($M)':    b['face'] / 1e6,
            'Market Value': price,
            'Duration':     dur,
            'Convexity':    conv,
            'DV01':         dv01,
            'Coupon':       b['coupon'],
            'Maturity':     b['maturity'],
        })

    df = pd.DataFrame(rows).set_index('Bond')
    total_mv = df['Market Value'].sum()

    # Weighted duration and convexity
    w = df['Market Value'] / total_mv
    w_dur  = (w * df['Duration']).sum()
    w_conv = (w * df['Convexity']).sum()
    total_dv01 = df['DV01'].sum()

    totals = pd.DataFrame([{
        'Bond':         'TOTAL / WEIGHTED',
        'Face ($M)':    df['Face ($M)'].sum(),
        'Market Value': total_mv,
        'Duration':     w_dur,
        'Convexity':    w_conv,
        'DV01':         total_dv01,
        'Coupon':       np.nan,
        'Maturity':     np.nan,
    }]).set_index('Bond')

    return pd.concat([df, totals])


def portfolio_mv(bonds: list, curve_params: dict) -> float:
    """Total market value of the bond portfolio."""
    return sum(bond_price(b['face'], b['coupon'], b['maturity'], curve_params)
               for b in bonds)


def portfolio_duration(bonds: list, curve_params: dict) -> float:
    """Market-value-weighted Macaulay duration of the portfolio."""
    mvs  = np.array([bond_price(b['face'], b['coupon'], b['maturity'], curve_params) for b in bonds])
    durs = np.array([bond_macaulay_duration(b['face'], b['coupon'], b['maturity'], curve_params) for b in bonds])
    total_mv = mvs.sum()
    return float((mvs * durs).sum() / total_mv) if total_mv > 0 else 0.0


def portfolio_convexity(bonds: list, curve_params: dict) -> float:
    """Market-value-weighted convexity of the portfolio."""
    mvs   = np.array([bond_price(b['face'], b['coupon'], b['maturity'], curve_params) for b in bonds])
    convs = np.array([bond_convexity(b['face'], b['coupon'], b['maturity'], curve_params) for b in bonds])
    total_mv = mvs.sum()
    return float((mvs * convs).sum() / total_mv) if total_mv > 0 else 0.0


def portfolio_scenario_table(bonds: list, base_params: dict) -> pd.DataFrame:
    """Market value of the bond portfolio under each rate scenario."""
    base_mv = portfolio_mv(bonds, base_params)
    rows = []
    for name in SCENARIOS:
        # Shocked rates at bond maturities
        mats = np.array([b['maturity'] for b in bonds])
        shocked_rates = apply_scenario(mats, base_params, name)
        shocked_params = fit_nelson_siegel(mats, shocked_rates)
        mv = portfolio_mv(bonds, shocked_params)
        rows.append({
            'Scenario':    name,
            'MV ($M)':     round(mv / 1e6, 3),
            'Change ($M)': round((mv - base_mv) / 1e6, 3),
            'Change (%)':  round((mv - base_mv) / base_mv * 100, 2),
        })
    return pd.DataFrame(rows).set_index('Scenario')
