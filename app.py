import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from yield_curve import (
    default_goc_params, nelson_siegel_spot, apply_scenario,
    GOC_MATURITIES, GOC_YIELDS, SCENARIOS, curve_dataframe,
)
from liabilities import (
    generate_pension_cashflows, pv_cashflows, liability_pv,
    macaulay_duration, dv01_liability, convexity_liability,
    inflation_sensitivity, scenario_pv_table,
)
from assets import (
    DEFAULT_PORTFOLIO, portfolio_summary, portfolio_mv,
    portfolio_duration, portfolio_convexity, portfolio_scenario_table,
)
from alm import alm_summary, alm_scenario_table, surplus_dv01
from ldi import ldi_summary_table, ldi_bond_allocations
from monte_carlo import (
    simulate_funding_ratios, funding_ratio_stats, surplus_var,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='Pension ALM Suite',
    page_icon='📊',
    layout='wide',
)

st.title('Pension ALM Suite')
st.caption('Asset-Liability Management for defined benefit pension funds — yield curves, LDI, and Monte Carlo.')

# ── Sidebar inputs ────────────────────────────────────────────────────────────
with st.sidebar:
    st.header('Liability Parameters')
    annual_payment = st.number_input('Annual Benefit Payment ($)', min_value=100_000,
                                     max_value=50_000_000, value=1_000_000, step=100_000,
                                     help='Base annual pension payment in Year 1.')
    horizon = st.slider('Payment Horizon (years)', 10, 50, 35)
    cpi = st.slider('CPI Indexation (%/yr)', 0.0, 5.0, 2.0, 0.25) / 100

    st.divider()
    st.header('Monte Carlo')
    n_sims  = st.select_slider('Simulations', [500, 1000, 2000, 5000], value=1000)
    rate_vol = st.slider('Rate Volatility (annual, %)', 0.5, 3.0, 1.0, 0.25) / 100
    mc_conf  = st.slider('VaR Confidence (%)', 90, 99, 95) / 100

    st.divider()
    st.header('Yield Curve Scenario')
    selected_scenario = st.selectbox('Scenario', list(SCENARIOS), index=0)


# ── Cached computations ───────────────────────────────────────────────────────
@st.cache_data
def get_base_params():
    return default_goc_params()


@st.cache_data
def get_cashflows(annual_payment, horizon, cpi):
    return generate_pension_cashflows(annual_payment, horizon, cpi)


@st.cache_data
def get_mc(annual_payment, horizon, cpi, n_sims, rate_vol, seed=42):
    base = default_goc_params()
    return simulate_funding_ratios(
        DEFAULT_PORTFOLIO, annual_payment, horizon, cpi,
        base, n_sims=n_sims, rate_vol=rate_vol, seed=seed,
    )


base_params = get_base_params()
cashflows   = get_cashflows(annual_payment, horizon, cpi)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    '📈 Yield Curve',
    '📋 Liabilities',
    '🏦 Assets',
    '⚖️ ALM Dashboard',
    '🎯 LDI Matching',
    '🎲 Monte Carlo',
])

# ════════════════════════════════════════════════════════════════════════════════
# Tab 1 — Yield Curve
# ════════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.subheader('Nelson-Siegel Yield Curve — Government of Canada')

    col1, col2 = st.columns([3, 1])
    with col1:
        fine_mats = np.linspace(0.25, 30, 200)
        curve_df = curve_dataframe(fine_mats, base_params,
                                   scenarios=['Base', '+100 bp parallel', '-100 bp parallel',
                                              'Steepener', 'Flattener'])

        fig, ax = plt.subplots(figsize=(9, 4))
        styles = {'Base': ('black', 2.5, '-'), '+100 bp parallel': ('red', 1.5, '--'),
                  '-100 bp parallel': ('green', 1.5, '--'), 'Steepener': ('royalblue', 1.5, ':'),
                  'Flattener': ('darkorange', 1.5, ':')}
        for sc, (color, lw, ls) in styles.items():
            ax.plot(fine_mats, curve_df[sc] * 100, color=color, lw=lw, ls=ls, label=sc)
        ax.scatter(GOC_MATURITIES, GOC_YIELDS * 100, color='black', zorder=5, s=40, label='GoC observed')
        ax.set_xlabel('Maturity (years)')
        ax.set_ylabel('Spot Rate (%)')
        ax.set_title('Nelson-Siegel Fitted Curve + Scenarios')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.metric('β₀ (long run)', f"{base_params['beta0']*100:.2f}%")
        st.metric('β₁ (slope)',    f"{base_params['beta1']*100:.2f}%")
        st.metric('β₂ (curvature)',f"{base_params['beta2']*100:.2f}%")
        st.metric('τ (decay)',     f"{base_params['tau']:.2f} yr")
        st.metric('Fit RMSE',      f"{base_params['rmse']*10000:.1f} bp")

    st.divider()
    st.subheader('Selected Scenario: ' + selected_scenario)
    sc_rates = apply_scenario(fine_mats, base_params, selected_scenario)
    base_rates = nelson_siegel_spot(fine_mats, **{k: base_params[k] for k in ['beta0','beta1','beta2','tau']})
    fig2, ax2 = plt.subplots(figsize=(9, 3))
    ax2.plot(fine_mats, base_rates * 100, 'k-', lw=2, label='Base')
    ax2.plot(fine_mats, sc_rates * 100, 'r--', lw=2, label=selected_scenario)
    ax2.fill_between(fine_mats, base_rates * 100, sc_rates * 100, alpha=0.15, color='red')
    ax2.set_xlabel('Maturity (years)')
    ax2.set_ylabel('Spot Rate (%)')
    ax2.legend()
    ax2.grid(alpha=0.3)
    st.pyplot(fig2)
    plt.close(fig2)


# ════════════════════════════════════════════════════════════════════════════════
# Tab 2 — Liabilities
# ════════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.subheader('Liability Cash Flows & Present Value')

    df_pv = pv_cashflows(cashflows, base_params)
    total_pv = df_pv['pv'].sum()
    d_liab   = macaulay_duration(cashflows, base_params)
    conv_l   = convexity_liability(cashflows, base_params)
    dv01_l   = dv01_liability(cashflows, base_params)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric('Total PV ($M)', f"${total_pv/1e6:.2f}M")
    m2.metric('Macaulay Duration', f"{d_liab:.2f} yr")
    m3.metric('Convexity', f"{conv_l:.2f}")
    m4.metric('DV01 ($)', f"${abs(dv01_l):,.0f}")

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(df_pv['year'], df_pv['cash_flow'] / 1e6, color='steelblue', alpha=0.7, label='Nominal CF')
        ax.bar(df_pv['year'], df_pv['pv'] / 1e6, color='navy', alpha=0.9, label='PV')
        ax.set_xlabel('Year')
        ax.set_ylabel('$ Millions')
        ax.set_title('Cash Flows vs Present Values')
        ax.legend()
        ax.grid(alpha=0.3, axis='y')
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(7, 4))
        cum_pv = df_pv['pv'].cumsum() / total_pv * 100
        ax.plot(df_pv['year'], cum_pv, 'navy', lw=2)
        ax.axhline(50, color='red', ls='--', alpha=0.7, label='50% of PV (≈ Macaulay Duration)')
        ax.axvline(d_liab, color='red', ls=':', alpha=0.7)
        ax.set_xlabel('Year')
        ax.set_ylabel('Cumulative PV (%)')
        ax.set_title('Cumulative PV Distribution')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

    st.divider()
    st.subheader('Rate Scenario Impact on Liabilities')
    try:
        sc_table = scenario_pv_table(cashflows, base_params)
        st.dataframe(sc_table.style.format({
            'PV ($M)': '${:.3f}M',
            'Change ($M)': '${:+.3f}M',
            'Change (%)': '{:+.2f}%',
        }), use_container_width=True)
    except Exception as e:
        st.warning(f"Scenario table error: {e}")

    st.divider()
    st.subheader('Inflation Sensitivity')
    inf = inflation_sensitivity(annual_payment, horizon, base_params, cpi_base=cpi,
                                cpi_shocked=min(cpi + 0.01, 0.10))
    ic1, ic2, ic3 = st.columns(3)
    ic1.metric('PV at current CPI', f"${inf['PV (base CPI)']/1e6:.2f}M")
    ic2.metric('PV at CPI +1%', f"${inf['PV (shocked CPI)']/1e6:.2f}M")
    ic3.metric('Change', f"${inf['Change ($)']/1e6:.2f}M ({inf['Change (%)']:+.1f}%)")


# ════════════════════════════════════════════════════════════════════════════════
# Tab 3 — Assets
# ════════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.subheader('Bond Portfolio')
    st.caption('Default portfolio: Government of Canada bonds across the curve.')

    summ = portfolio_summary(DEFAULT_PORTFOLIO, base_params)

    display_summ = summ[['Face ($M)', 'Market Value', 'Duration', 'Convexity', 'DV01']].copy()
    display_summ['Market Value'] = display_summ['Market Value'].apply(
        lambda x: f"${x/1e6:.3f}M" if pd.notna(x) else '')
    display_summ['DV01'] = display_summ['DV01'].apply(
        lambda x: f"${x:,.0f}" if pd.notna(x) else '')
    st.dataframe(display_summ, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        names = [b['name'] for b in DEFAULT_PORTFOLIO]
        from assets import bond_price
        mvs = [bond_price(b['face'], b['coupon'], b['maturity'], base_params) / 1e6 for b in DEFAULT_PORTFOLIO]
        ax.bar(names, mvs, color='steelblue')
        ax.set_ylabel('Market Value ($M)')
        ax.set_title('Bond Market Values')
        ax.grid(alpha=0.3, axis='y')
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.subheader('Scenario Analysis')
        try:
            asc_table = portfolio_scenario_table(DEFAULT_PORTFOLIO, base_params)
            st.dataframe(asc_table.style.format({
                'MV ($M)': '${:.3f}M',
                'Change ($M)': '${:+.3f}M',
                'Change (%)': '{:+.2f}%',
            }), use_container_width=True)
        except Exception as e:
            st.warning(f"Scenario table error: {e}")


# ════════════════════════════════════════════════════════════════════════════════
# Tab 4 — ALM Dashboard
# ════════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.subheader('ALM Dashboard — Base Case')

    alm = alm_summary(DEFAULT_PORTFOLIO, cashflows, base_params)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Asset MV', f"${alm['Asset MV ($M)']}M")
    c2.metric('Liability PV', f"${alm['Liability PV ($M)']}M")
    surplus_val = alm['Surplus ($M)']
    c3.metric('Surplus', f"${surplus_val}M", delta=f"{'Overfunded' if surplus_val >= 0 else 'Underfunded'}")
    c4.metric('Funding Ratio', f"{alm['Funding Ratio']:.2%}")

    st.divider()
    c5, c6, c7 = st.columns(3)
    c5.metric('Asset Duration', f"{alm['Asset Duration']} yr")
    c6.metric('Liability Duration', f"{alm['Liability Duration']} yr")
    gap = alm['Duration Gap']
    c7.metric('Duration Gap', f"{gap:.2f} yr",
              delta='Immunised' if abs(gap) < 0.5 else ('Assets longer' if gap > 0 else 'Liabilities longer'))

    c8, c9 = st.columns(2)
    c8.metric('Surplus DV01', f"${alm['Surplus DV01 ($)']:+,.0f}")
    c8.caption('Change in surplus per 1 bp rate rise')
    c9.metric('Asset Convexity', f"{alm['Asset Convexity']:.2f}")

    st.divider()
    st.subheader('Scenario Impact on Surplus')
    try:
        alm_sc = alm_scenario_table(DEFAULT_PORTFOLIO, cashflows, base_params)
        def color_surplus(val):
            if isinstance(val, float):
                color = 'green' if val >= 0 else 'red'
                return f'color: {color}'
            return ''
        st.dataframe(alm_sc.style.format({
            'Asset MV ($M)': '${:.3f}M',
            'Liability PV ($M)': '${:.3f}M',
            'Surplus ($M)': '${:+.3f}M',
            'Funding Ratio': '{:.2%}',
        }).applymap(color_surplus, subset=['Surplus ($M)']), use_container_width=True)
    except Exception as e:
        st.warning(f"Scenario table error: {e}")


# ════════════════════════════════════════════════════════════════════════════════
# Tab 5 — LDI Matching
# ════════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.subheader('Liability-Driven Investing — Duration Matching')

    st.markdown("""
    **Goal:** Minimise the *duration gap* = D_Assets − (PV_L / MV_A) × D_Liabilities.

    When the gap is zero, a parallel rate shift affects assets and liabilities equally,
    leaving the surplus unchanged. This is the basic LDI immunisation condition.
    """)

    try:
        ldi_cmp = ldi_summary_table(DEFAULT_PORTFOLIO, cashflows, base_params)
        st.dataframe(ldi_cmp.style.format({
            'Asset Duration': '{:.2f}',
            'Liability Duration': '{:.2f}',
            'Duration Gap': '{:+.4f}',
        }), use_container_width=True)

        st.divider()
        st.subheader('Optimal Bond Allocations (Duration Matched)')
        total_budget = portfolio_mv(DEFAULT_PORTFOLIO, base_params)
        ldi_alloc = ldi_bond_allocations(DEFAULT_PORTFOLIO, cashflows, base_params, total_budget)
        st.dataframe(ldi_alloc.style.format({
            'Weight (%)': '{:.2f}%',
            'Allocation ($M)': '${:.3f}M',
        }), use_container_width=True)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(ldi_alloc.index, ldi_alloc['Weight (%)'], color='steelblue')
        ax.set_ylabel('Weight (%)')
        ax.set_title('LDI Duration-Matched Portfolio Weights')
        ax.grid(alpha=0.3, axis='y')
        st.pyplot(fig)
        plt.close(fig)

    except Exception as e:
        st.error(f"LDI optimisation error: {e}")


# ════════════════════════════════════════════════════════════════════════════════
# Tab 6 — Monte Carlo
# ════════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.subheader('Monte Carlo — Funding Ratio Distribution')

    with st.spinner(f'Running {n_sims:,} simulations...'):
        frs = get_mc(annual_payment, horizon, cpi, n_sims, rate_vol)

    stats = funding_ratio_stats(frs, mc_conf)
    svar  = surplus_var(DEFAULT_PORTFOLIO, annual_payment, horizon, cpi,
                        base_params, n_sims=n_sims, confidence=mc_conf, rate_vol=rate_vol)

    col1, col2 = st.columns([3, 2])
    with col1:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(frs, bins=60, color='steelblue', edgecolor='white', alpha=0.85)
        ax.axvline(1.0, color='red', lw=2, ls='--', label='Fully Funded (1.0)')
        ax.axvline(stats[f"VaR FR ({int(mc_conf*100)}%)"], color='darkorange', lw=2,
                   ls=':', label=f"VaR FR ({int(mc_conf*100)}%)")
        ax.axvline(stats['Mean FR'], color='black', lw=1.5, ls='-', label='Mean')
        ax.set_xlabel('Funding Ratio')
        ax.set_ylabel('Frequency')
        ax.set_title('Simulated Funding Ratio Distribution')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.subheader('FR Statistics')
        for k, v in stats.items():
            if 'P(' in k:
                st.metric(k, f"{v:.1%}")
            else:
                st.metric(k, f"{v:.4f}")

        st.divider()
        st.subheader('Surplus VaR')
        for k, v in svar.items():
            st.metric(k, f"${v}M")

    st.divider()
    st.subheader('Rate Volatility Sensitivity')
    vols = np.arange(0.005, 0.031, 0.005)
    p_unfunded = []
    for v in vols:
        frs_v = simulate_funding_ratios(
            DEFAULT_PORTFOLIO, annual_payment, horizon, cpi,
            base_params, n_sims=500, rate_vol=v, seed=99,
        )
        p_unfunded.append((frs_v < 1.0).mean())

    fig2, ax2 = plt.subplots(figsize=(7, 3))
    ax2.plot(vols * 100, np.array(p_unfunded) * 100, 'o-', color='crimson')
    ax2.set_xlabel('Rate Volatility (%/yr)')
    ax2.set_ylabel('P(Underfunded) (%)')
    ax2.set_title('Probability of Underfunding vs. Rate Volatility')
    ax2.grid(alpha=0.3)
    st.pyplot(fig2)
    plt.close(fig2)
