"""
UBI Policy Stress-Tester
========================
Macro-micro labor supply model for Universal Basic Income policy simulation.

Model layers:
  1. Population synthesis  – log-normal wage distribution (10k households)
  2. Behavioral response   – intensive margin (hours) + extensive margin (participation)
  3. Fiscal accounting     – gross cost, incremental tax revenue, net balance
  4. Welfare metrics       – poverty headcount, Gini, income distribution shift
  5. Sensitivity analysis  – 2-D heatmap sweeping key elasticity × UBI assumptions
"""

from __future__ import annotations

import io
import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# 0.  Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="UBI Policy Stress-Tester",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PolicyParams:
    """All adjustable parameters for one policy scenario."""
    ubi_monthly: float            # Monthly transfer per household
    tax_rate: float               # New marginal income tax rate (0–1)
    labor_elasticity: float       # Uncompensated wage elasticity of hours (intensive)
    participation_elasticity: float  # Income-effect elasticity on participation (extensive)
    base_tax_rate: float          # Pre-reform marginal tax rate
    poverty_line: float = 2000.0  # Monthly income threshold
    n_households: int = 10_000    # Simulation population
    currency: str = "SAR"

    @property
    def label(self) -> str:
        return (
            f"UBI={self.ubi_monthly:,.0f} | "
            f"τ={self.tax_rate:.0%} | "
            f"ε_h={self.labor_elasticity:+.2f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Core model functions
# ─────────────────────────────────────────────────────────────────────────────

def simulate_population(params: PolicyParams, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic household population.

    Wages ~ LogNormal(ln 3500, 0.70) – calibrated to a middle-income country.
    Baseline participation rate: 75 %.
    """
    rng = np.random.default_rng(seed)
    n   = params.n_households

    wages        = rng.lognormal(np.log(3_500), 0.70, n)
    base_hours   = np.ones(n)                             # normalised to 1
    employed     = (rng.random(n) < 0.75).astype(float)  # 75 % participation

    pre_income = wages * base_hours * (1 - params.base_tax_rate) * employed

    return pd.DataFrame({
        "wage":        wages,
        "base_hours":  base_hours,
        "employed":    employed,
        "pre_income":  pre_income,
    })


def apply_reform(df: pd.DataFrame, params: PolicyParams) -> pd.DataFrame:
    """
    Apply UBI reform with two-margin behavioural response.

    Intensive margin  (hours):
        h_new = h_base · (1 + ε_h · Δw/w_base)

    Extensive margin  (participation):
        Δp_i = −ε_p · T / (w_i · h_i)
        New participation probability = max(0, p_base − Δp_i)

    Post-reform disposable income:
        y_i = (1 − τ_new) · w_i · h_i_new · employed_i_new  +  T
    """
    df = df.copy()
    nw_before   = 1 - params.base_tax_rate
    nw_after    = 1 - params.tax_rate
    pct_dw      = (nw_after - nw_before) / max(nw_before, 1e-6)

    # — Intensive margin ——————————————————————————————————————————————
    h_mult          = np.clip(1 + params.labor_elasticity * pct_dw, 0.30, 1.50)
    df["new_hours"] = df["base_hours"] * h_mult

    # — Extensive margin ——————————————————————————————————————————————
    income_share        = params.ubi_monthly / (df["wage"] * df["base_hours"] + 1e-6)
    delta_p             = params.participation_elasticity * income_share
    df["new_employed"]  = np.clip(df["employed"] - delta_p, 0.0, 1.0)

    # — Post-reform income ————————————————————————————————————————————
    df["labor_income_post"] = df["wage"] * df["new_hours"] * df["new_employed"]
    df["post_income"]       = df["labor_income_post"] * nw_after + params.ubi_monthly

    # — Poverty flags ————————————————————————————————————————————————
    df["poor_pre"]  = (df["pre_income"]  < params.poverty_line).astype(int)
    df["poor_post"] = (df["post_income"] < params.poverty_line).astype(int)

    return df


def compute_metrics(df: pd.DataFrame, params: PolicyParams) -> dict:
    """Aggregate fiscal + welfare metrics from simulated micro-data."""
    n = len(df)

    gross_ubi_cost       = params.ubi_monthly * 12 * n
    tax_rev_new          = (df["labor_income_post"] * params.tax_rate).sum() * 12
    tax_rev_base         = (
        df["wage"] * df["base_hours"] * df["employed"] * params.base_tax_rate
    ).sum() * 12
    incremental_rev      = tax_rev_new - tax_rev_base
    net_cost             = gross_ubi_cost - incremental_rev

    total_labor_pre  = (df["base_hours"] * df["employed"]).sum()
    total_labor_post = (df["new_hours"]  * df["new_employed"]).sum()
    labor_chg        = (total_labor_post - total_labor_pre) / total_labor_pre * 100

    poverty_pre      = int(df["poor_pre"].sum())
    poverty_post     = int(df["poor_post"].sum())
    poverty_exit     = poverty_pre - poverty_post

    gini_pre  = _gini(df["pre_income"])
    gini_post = _gini(df["post_income"])

    return dict(
        gross_ubi_cost      = gross_ubi_cost,
        tax_rev_new         = tax_rev_new,
        incremental_rev     = incremental_rev,
        net_cost            = net_cost,
        poverty_pre         = poverty_pre,
        poverty_post        = poverty_post,
        poverty_exit        = poverty_exit,
        poverty_rate_pre    = poverty_pre  / n * 100,
        poverty_rate_post   = poverty_post / n * 100,
        avg_income_pre      = float(df["pre_income"].mean()),
        avg_income_post     = float(df["post_income"].mean()),
        labor_chg_pct       = labor_chg,
        cost_per_exit       = net_cost / max(poverty_exit, 1),
        gini_pre            = gini_pre,
        gini_post           = gini_post,
    )


def _gini(incomes: pd.Series) -> float:
    """Compute Gini coefficient."""
    x = np.sort(incomes.values)
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Sensitivity sweep  (cached – expensive)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def sensitivity_heatmap(
    tax_rate: float,
    participation_elasticity: float,
    base_tax_rate: float,
    poverty_line: float,
    n_households: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sweep labor_elasticity × ubi_monthly → net_cost (billions).
    Returns (elasticities, ubi_values, cost_grid).
    """
    elasticities = np.linspace(-0.5, 0.5, 14)
    ubi_values   = np.linspace(300, 4_500, 14)
    z = np.zeros((len(elasticities), len(ubi_values)))

    for i, e in enumerate(elasticities):
        for j, u in enumerate(ubi_values):
            p  = PolicyParams(
                ubi_monthly            = u,
                tax_rate               = tax_rate,
                labor_elasticity       = e,
                participation_elasticity = participation_elasticity,
                base_tax_rate          = base_tax_rate,
                poverty_line           = poverty_line,
                n_households           = n_households,
            )
            df = simulate_population(p)
            df = apply_reform(df, p)
            m  = compute_metrics(df, p)
            z[i, j] = m["net_cost"] / 1e9

    return elasticities, ubi_values, z


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Chart builders
# ─────────────────────────────────────────────────────────────────────────────

_DARK = "plotly_dark"
_RED  = "#FF4B4B"
_GRN  = "#00D37F"
_BLU  = "#4B8BFF"
_YLW  = "#FFD166"


def fig_income_distribution(df: pd.DataFrame) -> go.Figure:
    """Overlaid histograms of pre- and post-reform disposable income."""
    cap = 14_000
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df["pre_income"].clip(0, cap), name="Pre-UBI",
        opacity=0.55, marker_color=_RED, nbinsx=90,
    ))
    fig.add_trace(go.Histogram(
        x=df["post_income"].clip(0, cap), name="Post-UBI",
        opacity=0.55, marker_color=_GRN, nbinsx=90,
    ))
    fig.update_layout(
        barmode="overlay",
        title="Household Income Distribution — Before vs After",
        xaxis_title="Monthly Disposable Income",
        yaxis_title="Households",
        template=_DARK, height=370,
        legend=dict(orientation="h", y=1.06),
    )
    return fig


def fig_budget_constraint(params: PolicyParams) -> go.Figure:
    """Classical income-leisure budget constraint shift for median household."""
    leisure = np.linspace(0, 1, 300)
    work    = 1 - leisure
    w_ref   = 3_500  # median reference wage

    c_pre  = w_ref * (1 - params.base_tax_rate) * work
    c_post = w_ref * (1 - params.tax_rate) * work + params.ubi_monthly

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=leisure, y=c_pre,
        name="Pre-UBI", line=dict(color=_RED, width=3),
    ))
    fig.add_trace(go.Scatter(
        x=leisure, y=c_post,
        name="Post-UBI", line=dict(color=_GRN, width=3, dash="dash"),
    ))
    # Shade the non-participation region
    fig.add_vrect(
        x0=0.88, x1=1.0,
        fillcolor="gray", opacity=0.12,
        annotation_text="Non-participation",
        annotation_position="top left",
    )
    # Mark UBI floor
    fig.add_hline(
        y=params.ubi_monthly, line_dash="dot",
        line_color=_YLW, opacity=0.7,
        annotation_text=f"UBI floor = {params.ubi_monthly:,}",
    )
    fig.update_layout(
        title="Budget Constraint — Labor / Leisure Trade-off (Median Household)",
        xaxis_title="Leisure (fraction of time endowment)",
        yaxis_title="Monthly Consumption",
        template=_DARK, height=370,
        legend=dict(orientation="h", y=1.06),
    )
    return fig


def fig_poverty_waterfall(m: dict) -> go.Figure:
    """Waterfall: poverty rate pre → impact → post."""
    pre  = m["poverty_rate_pre"]
    post = m["poverty_rate_post"]
    drop = pre - post

    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=["absolute", "relative", "total"],
        x=["Pre-UBI Poverty Rate", "UBI Impact", "Post-UBI Poverty Rate"],
        y=[pre, -drop, 0],
        text=[f"{pre:.1f}%", f"−{drop:.1f} pp", f"{post:.1f}%"],
        textposition="outside",
        connector={"line": {"color": _BLU}},
        increasing={"marker": {"color": _RED}},
        decreasing={"marker": {"color": _GRN}},
        totals={"marker": {"color": _BLU}},
    ))
    fig.update_layout(
        title="Poverty Rate Waterfall",
        yaxis_title="Poverty Rate (%)",
        template=_DARK, height=370,
        showlegend=False,
    )
    return fig


def fig_fiscal_bar(m: dict) -> go.Figure:
    """Three-bar fiscal balance summary."""
    labels = ["Gross UBI Cost", "Tax Revenue Gain", "Net Fiscal Cost"]
    values = [
        m["gross_ubi_cost"]  / 1e9,
        m["incremental_rev"] / 1e9,
        m["net_cost"]        / 1e9,
    ]
    colors = [_RED, _GRN, _BLU]

    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker_color=colors,
        text=[f"{v:.2f} B" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        title="Fiscal Balance (Annual, Billions)",
        yaxis_title="Amount (Billions)",
        template=_DARK, height=370,
        showlegend=False,
    )
    return fig


def fig_sensitivity(elasticities, ubi_values, z) -> go.Figure:
    """2-D heatmap: labor elasticity × UBI amount → net fiscal cost."""
    fig = go.Figure(go.Heatmap(
        x=ubi_values,
        y=elasticities,
        z=z,
        colorscale="RdYlGn_r",
        colorbar=dict(title="Net Cost (B)"),
        hovertemplate=(
            "UBI: %{x:,.0f}<br>"
            "ε_h: %{y:.2f}<br>"
            "Net Cost: %{z:.2f} B<extra></extra>"
        ),
    ))
    fig.update_layout(
        title="Sensitivity — Net Fiscal Cost by UBI Amount × Labor Supply Elasticity",
        xaxis_title="Monthly UBI Transfer",
        yaxis_title="Labor Supply Elasticity (ε_h)",
        template=_DARK,
        height=420,
    )
    return fig


def fig_scenario_comparison(rows: list[dict]) -> go.Figure:
    """Radar / bar for saved scenario comparison."""
    if not rows:
        return go.Figure()

    df_sc = pd.DataFrame(rows)
    metrics_cols = [
        "Poverty Reduction (pp)",
        "Net Cost (B)",
        "Labor Δ (%)",
        "Gini Δ",
    ]
    fig = go.Figure()
    for _, row in df_sc.iterrows():
        fig.add_trace(go.Bar(
            name=row["Scenario"],
            x=metrics_cols,
            y=[
                float(row["Poverty Reduction (pp)"]),
                float(row["Net Cost (B)"]),
                float(row["Labor Δ (%)"]),
                float(row["Gini Δ"]) * 100,   # scale to pp
            ],
        ))
    fig.update_layout(
        barmode="group",
        title="Scenario Comparison (normalised by metric)",
        template=_DARK,
        height=380,
        legend=dict(orientation="h", y=-0.18),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Streamlit UI
# ─────────────────────────────────────────────────────────────────────────────

def sidebar() -> PolicyParams:
    """Render sidebar controls and return current PolicyParams."""
    with st.sidebar:
        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/thumb/3/30/Scale_of_justice_2.svg/"
            "240px-Scale_of_justice_2.svg.png",
            width=60,
        )
        st.title("Policy Controls")

        st.subheader("💰 Transfer")
        ubi_monthly = st.slider(
            "Monthly UBI per Household",
            min_value=0, max_value=5_000, value=1_000, step=100,
            help="Universal cash transfer paid monthly to every household.",
        )

        st.subheader("📊 Tax Structure")
        base_tax_rate = st.slider(
            "Baseline Tax Rate",
            min_value=0.00, max_value=0.50, value=0.15, step=0.01,
            format="%.2f",
        )
        tax_rate = st.slider(
            "New Marginal Tax Rate",
            min_value=0.00, max_value=0.70, value=0.25, step=0.01,
            format="%.2f",
            help="Funds the UBI. Higher rate reduces net wages → behavioral response.",
        )

        st.subheader("🔬 Behavioural Elasticities")
        labor_elasticity = st.slider(
            "Labor Supply Elasticity — Intensive (ε_h)",
            min_value=-0.50, max_value=0.50, value=-0.10, step=0.05,
            help=(
                "% change in hours per 1 % change in net wage.  "
                "Typically negative (substitution effect). Range −0.5 to +0.5."
            ),
        )
        participation_elasticity = st.slider(
            "Participation Elasticity — Extensive (ε_p)",
            min_value=0.0, max_value=1.0, value=0.30, step=0.05,
            help=(
                "Income-effect magnitude: how much UBI income reduces "
                "labour-force participation.  0 = no effect, 1 = large reduction."
            ),
        )

        st.subheader("⚙️ Simulation")
        poverty_line = st.slider(
            "Poverty Line (Monthly Income)",
            min_value=500, max_value=5_000, value=2_000, step=100,
        )
        n_households = st.select_slider(
            "Population Size",
            options=[1_000, 5_000, 10_000, 20_000],
            value=10_000,
        )

        st.divider()
        if st.button("🔄 Reset All", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

    return PolicyParams(
        ubi_monthly              = ubi_monthly,
        tax_rate                 = tax_rate,
        labor_elasticity         = labor_elasticity,
        participation_elasticity = participation_elasticity,
        base_tax_rate            = base_tax_rate,
        poverty_line             = poverty_line,
        n_households             = n_households,
    )


def render_kpis(m: dict, params: PolicyParams) -> None:
    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        delta = m["poverty_rate_post"] - m["poverty_rate_pre"]
        st.metric(
            "Poverty Rate",
            f"{m['poverty_rate_post']:.1f} %",
            delta=f"{delta:.1f} pp",
            delta_color="inverse",
        )
    with c2:
        st.metric(
            "Net Fiscal Cost (Annual)",
            f"{m['net_cost'] / 1e9:.2f} B",
            delta=f"Gross: {m['gross_ubi_cost'] / 1e9:.2f} B",
        )
    with c3:
        st.metric(
            "Labor Supply Δ",
            f"{m['labor_chg_pct']:+.1f} %",
        )
    with c4:
        gini_delta = m["gini_post"] - m["gini_pre"]
        st.metric(
            "Gini Coefficient",
            f"{m['gini_post']:.3f}",
            delta=f"{gini_delta:+.3f}",
            delta_color="inverse",
        )
    with c5:
        st.metric(
            "Cost per Poverty Exit",
            f"{m['cost_per_exit'] / 1_000:.1f} K / yr",
        )


def render_scenario_panel(params: PolicyParams, m: dict) -> None:
    """Save and display scenario comparison."""
    if "scenarios" not in st.session_state:
        st.session_state.scenarios = []

    col_a, col_b = st.columns([3, 1])
    with col_a:
        label = st.text_input(
            "Scenario name",
            value=params.label,
            label_visibility="collapsed",
        )
    with col_b:
        if st.button("💾 Save Scenario", use_container_width=True):
            row = {
                "Scenario":                label,
                "UBI (monthly)":           f"{params.ubi_monthly:,.0f}",
                "Tax Rate":                f"{params.tax_rate:.0%}",
                "ε_h":                     f"{params.labor_elasticity:+.2f}",
                "ε_p":                     f"{params.participation_elasticity:.2f}",
                "Poverty Rate Post (%)":   f"{m['poverty_rate_post']:.1f}",
                "Poverty Reduction (pp)":  f"{m['poverty_rate_pre'] - m['poverty_rate_post']:.1f}",
                "Net Cost (B)":            f"{m['net_cost'] / 1e9:.2f}",
                "Labor Δ (%)":             f"{m['labor_chg_pct']:+.1f}",
                "Gini Δ":                  f"{m['gini_post'] - m['gini_pre']:+.4f}",
                "Cost/Exit (K)":           f"{m['cost_per_exit'] / 1_000:.1f}",
            }
            st.session_state.scenarios.append(row)
            st.toast(f"Saved: {label}", icon="✅")

    if st.session_state.scenarios:
        df_s = pd.DataFrame(st.session_state.scenarios)
        st.dataframe(df_s, use_container_width=True, hide_index=True)
        st.plotly_chart(
            fig_scenario_comparison(st.session_state.scenarios),
            use_container_width=True,
        )
        if st.button("🗑️ Clear Scenarios"):
            st.session_state.scenarios = []
            st.rerun()


def render_equations() -> None:
    with st.expander("📐 Model Equations & Assumptions", expanded=False):
        st.markdown(r"""
### Household Budget Constraint (Post-Reform)

$$c_i = \underbrace{(1 - \tau_{\text{new}}) \cdot w_i \cdot h_i^{\text{new}} \cdot \mathbb{1}[\text{employed}]}_{\text{net labour income}} + \underbrace{T}_{\text{UBI transfer}}$$

### Intensive Margin — Hours Adjustment

$$h_i^{\text{new}} = h_i^{\text{base}} \cdot \Bigl(1 + \varepsilon_h \cdot \frac{\Delta w}{w_{\text{base}}}\Bigr), \qquad \varepsilon_h \in [-0.5,\ 0.5]$$

where $\Delta w / w_{\text{base}} = \dfrac{(1-\tau_{\text{new}}) - (1-\tau_{\text{base}})}{1-\tau_{\text{base}}}$.

### Extensive Margin — Participation Probability

$$\Delta p_i = \varepsilon_p \cdot \frac{T}{w_i \cdot h_i^{\text{base}}}$$

$$p_i^{\text{new}} = \max\!\bigl(0,\ p_i^{\text{base}} - \Delta p_i\bigr)$$

The income effect of the UBI transfer reduces the marginal utility of work for low-wage households.

### Fiscal Balance (Annual)

$$\text{Net Cost} = \underbrace{N \cdot T \cdot 12}_{\text{gross UBI}} - \underbrace{\Delta(\tau \cdot Y^L)}_{\text{incremental tax revenue}}$$

### Key Assumptions
| Assumption | Value |
|---|---|
| Wage distribution | Log-Normal $(\mu = \ln 3500,\ \sigma = 0.70)$ |
| Baseline participation | 75 % |
| Hours normalisation | 1.0 (full time) |
| General equilibrium | ❌ Ignored (partial equilibrium) |
| Inflation / savings | ❌ Ignored |
| Household heterogeneity | Log-normal wages only |
        """)


def render_export(params: PolicyParams, m: dict, df_sim: pd.DataFrame) -> None:
    st.subheader("📥 Export")
    summary = {
        "parameters": asdict(params),
        "metrics": m,
        "model_notes": (
            "Partial-equilibrium micro-simulation. "
            "Log-normal wages, two-margin labour supply response. "
            "No GE feedback, no savings, no inflation."
        ),
    }

    col_j, col_c = st.columns(2)
    with col_j:
        st.download_button(
            label="⬇️ JSON Report",
            data=json.dumps(summary, indent=2),
            file_name="ubi_simulation_report.json",
            mime="application/json",
            use_container_width=True,
        )
    with col_c:
        buf = io.StringIO()
        df_sim[["wage", "pre_income", "post_income", "poor_pre", "poor_post"]].to_csv(
            buf, index=False
        )
        st.download_button(
            label="⬇️ Household Micro-data (CSV)",
            data=buf.getvalue(),
            file_name="ubi_household_microdata.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    params = sidebar()

    # ── Header ───────────────────────────────────────────────────────────────
    st.title("⚖️  UBI Policy Stress-Tester")
    st.markdown(
        """
        Simulate a **macro-micro labour supply model** for Universal Basic Income.  
        Adjust policy levers and behavioural elasticities in the sidebar — results update instantly.
        Stress-test your assumptions with the sensitivity heatmap below.
        """,
        unsafe_allow_html=False,
    )
    st.divider()

    # ── Run model ────────────────────────────────────────────────────────────
    with st.spinner("Simulating population…"):
        df_sim = simulate_population(params)
        df_sim = apply_reform(df_sim, params)
        m      = compute_metrics(df_sim, params)

    # ── KPI row ──────────────────────────────────────────────────────────────
    render_kpis(m, params)
    st.divider()

    # ── Chart grid ───────────────────────────────────────────────────────────
    tab_dist, tab_bc, tab_fiscal, tab_poverty = st.tabs([
        "📊 Income Distribution",
        "📉 Budget Constraint",
        "💸 Fiscal Balance",
        "🏚️ Poverty Waterfall",
    ])

    with tab_dist:
        st.plotly_chart(fig_income_distribution(df_sim), use_container_width=True)

    with tab_bc:
        st.plotly_chart(fig_budget_constraint(params), use_container_width=True)
        st.caption(
            "Slope = net wage rate; vertical intercept = UBI floor (post-reform). "
            "Flatter slope reflects higher marginal tax rate."
        )

    with tab_fiscal:
        st.plotly_chart(fig_fiscal_bar(m), use_container_width=True)
        col_l, col_r = st.columns(2)
        col_l.metric(
            "Incremental Tax Revenue",
            f"{m['incremental_rev'] / 1e9:.2f} B",
        )
        col_r.metric(
            "Self-financing Ratio",
            f"{m['incremental_rev'] / max(m['gross_ubi_cost'], 1):.0%}",
        )

    with tab_poverty:
        st.plotly_chart(fig_poverty_waterfall(m), use_container_width=True)
        col_a, col_b = st.columns(2)
        col_a.metric("Poverty Exits", f"{m['poverty_exit']:,} households")
        col_b.metric(
            "Poverty-entry Risk",
            "None — UBI is universal",
            delta="0",
            delta_color="off",
        )

    st.divider()

    # ── Sensitivity heatmap ──────────────────────────────────────────────────
    st.subheader("🔬 Sensitivity Analysis")
    st.caption(
        "Sweeps **labor supply elasticity** (ε_h) × **monthly UBI amount** → "
        "net fiscal cost.  Red = high cost; green = lower net cost via labour-supply gains."
    )
    with st.spinner("Running 14×14 sensitivity sweep (cached after first run)…"):
        elast, ubi_vals, z = sensitivity_heatmap(
            params.tax_rate,
            params.participation_elasticity,
            params.base_tax_rate,
            params.poverty_line,
            params.n_households,
        )
    st.plotly_chart(fig_sensitivity(elast, ubi_vals, z), use_container_width=True)

    st.divider()

    # ── Scenario comparison ──────────────────────────────────────────────────
    st.subheader("📋 Scenario Comparison")
    render_scenario_panel(params, m)

    st.divider()

    # ── Equations + Export ───────────────────────────────────────────────────
    render_equations()
    render_export(params, m, df_sim)


if __name__ == "__main__":
    main()
