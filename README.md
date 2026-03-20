# Pension ALM Suite

A professional Asset-Liability Management (ALM) tool for defined benefit pension funds, built in Python.

**Live App:** https://pension-alm-u4usgpvtagddncv6qoamru.streamlit.app/

---

## What This Is

Pension funds face a fundamental problem: liabilities (future benefit payments to retirees) change in value as interest rates move — and so do the assets held to fund them. If assets and liabilities don't respond to rate changes in the same way, the fund's **surplus** (assets minus PV of liabilities) is exposed to **interest rate risk**.

This tool models that problem end-to-end:

1. **Yield curve modelling** — fit a Nelson-Siegel curve to Government of Canada rates
2. **Liability modelling** — discount 30+ years of pension cash flows using spot rates
3. **Asset modelling** — price a bond portfolio across the curve
4. **ALM dashboard** — track surplus, funding ratio, and duration gap under rate scenarios
5. **LDI matching** — find bond weights that immunise the surplus against rate moves
6. **Monte Carlo** — simulate the funding ratio distribution under stochastic rate paths

---

## Models & Concepts

### Nelson-Siegel Yield Curve
Parameterises the entire spot rate curve with 4 parameters:

```
r(t) = β₀ + β₁·f₁(t,τ) + β₂·f₂(t,τ)
```

| Parameter | Interpretation |
|-----------|----------------|
| **β₀** | Long-run rate (level) |
| **β₁** | Short-end component (slope) |
| **β₂** | Medium-term hump (curvature) |
| **τ** | Decay rate |

Fitted to Government of Canada nominal yields. Supports 8 rate scenarios: parallel shifts (±100, ±200 bp), steepener, flattener, humped.

### Liability Discounting
```
PV = Σ CF_t · exp(-r(t) · t)
```
Continuously-compounded discounting using the full spot curve (not a flat rate). Supports CPI indexation of benefit payments.

### Duration & Convexity
- **Macaulay Duration** — weighted-average time to cash flow receipt
- **Modified Duration** — price sensitivity per unit rate change (= Macaulay for continuous compounding)
- **DV01** — dollar change in PV per 1 basis point rate shift
- **Convexity** — second-order rate sensitivity (curvature correction)

### Surplus & Funding Ratio
```
Surplus        = MV(Assets) − PV(Liabilities)
Funding Ratio  = MV(Assets) / PV(Liabilities)
Duration Gap   = D_A − (PV_L / MV_A) · D_L
```

A funding ratio > 1 means overfunded. Duration gap = 0 means immunised.

### LDI Duration Matching
Optimises bond portfolio weights to minimise the duration gap subject to:
- Weights sum to 1 (fully invested)
- Weights ≥ 0 (long only)

Uses `scipy.optimize.minimize` (SLSQP).

### Monte Carlo
Simulates parallel shifts to β₀ (the long-run rate) using a simplified Vasicek-style model:

```
Δβ₀ ~ N(κ·(θ - β₀)·dt, σ·√dt)
```

Generates a distribution of funding ratios and computes Surplus VaR.

---

## Features

- **Yield curve visualisation** — fitted NS curve vs. GoC observed rates, scenario overlays
- **Cash flow waterfall** — nominal vs. PV by year, cumulative PV (reads off duration visually)
- **Inflation sensitivity** — compare liability PV at different CPI assumptions
- **Scenario table** — surplus and funding ratio under 8 rate shocks
- **LDI optimiser** — optimal bond weights for duration matching
- **Monte Carlo** — funding ratio histogram, surplus VaR, rate vol sensitivity

---

## Tech Stack

- `scipy` — Nelson-Siegel fitting, LDI optimisation
- `numpy` / `pandas` — computation
- `matplotlib` — visualisation
- `streamlit` — web interface

## File Structure

```
├── yield_curve.py   # Nelson-Siegel model, GoC curve, scenario shocks
├── liabilities.py   # Cash flows, PV, duration, convexity, inflation
├── assets.py        # Bond pricing, portfolio duration/convexity/DV01
├── alm.py           # Surplus, funding ratio, duration gap, scenario table
├── ldi.py           # Duration matching optimisation
├── monte_carlo.py   # Yield curve simulation, funding ratio distribution
├── app.py           # Streamlit web app
└── requirements.txt
```

## Installation (Local)

```bash
git clone https://github.com/Spencerpain/pension-alm.git
cd pension-alm
pip install -r requirements.txt
streamlit run app.py
```
