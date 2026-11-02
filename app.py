import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.optimize import fsolve, minimize_scalar
import io
import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# -----------------------------------------------------------------------------
# 1. STREAMLIT CONFIG & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Hispanic Heritage Historical Data Explorer",
    page_icon="🌎",
    layout="wide"
)

st.title("🌎 Hispanic Heritage Historical Data Explorer & STEM Math Laboratory")
st.markdown("""
*Investigating Latin American socioeconomic indicators through World Bank authentic data, high-degree polynomial modeling, and calculus-based rate of change analysis.*
""")

# -----------------------------------------------------------------------------
# 2. CONSTANTS & WORLD BANK API MAPPINGS
# -----------------------------------------------------------------------------
COUNTRIES = {
    "Mexico": "MX",
    "Brazil (Latin American Comparison)": "BR",
    "Argentina": "AR",
    "Colombia": "CO",
    "Chile": "CL",
    "Peru": "PE",
    "Venezuela": "VE",
    "Ecuador": "EC",
    "Uruguay": "UY"
}

INDICATORS = {
    "Life Expectancy at Birth": {
        "code": "SP.DYN.LE00.IN",
        "unit": "years",
        "def": "Number of years a newborn infant would live if prevailing patterns of mortality at the time of its birth were to stay the same throughout its life."
    },
    "Unemployment Rate (National Estimate)": {
        "code": "SL.UEM.TOTL.NE.ZS",
        "unit": "% of total labor force",
        "def": "Share of the labor force that is without work but available for and seeking employment (national estimate)."
    },
    "GDP per Capita": {
        "code": "NY.GDP.PCAP.CD",
        "unit": "current US$",
        "def": "Gross domestic product divided by midyear population (measures economic output per person, not average household wealth or personal income)."
    },
    "Crude Birth Rate": {
        "code": "SP.DYN.CBRT.IN",
        "unit": "per 1,000 people",
        "def": "Annual number of live births per 1,000 midyear population (this is a rate, not a total count of births)."
    },
    "Net Migration": {
        "code": "SM.POP.NETM",
        "unit": "net headcount",
        "def": "Total number of immigrants minus the number of emigrants over a five-year period (net movement, not total emigration count)."
    },
    "Intentional Homicide Rate": {
        "code": "VC.IHR.PSRC.P5",
        "unit": "per 100,000 people",
        "def": "Unlawful deaths purposefully inflicted on a person by another person per 100,000 population."
    },
    "Consumer Price Inflation": {
        "code": "FP.CPI.TOTL.ZG",
        "unit": "annual %",
        "def": "Annual percentage change in the cost to the average consumer of acquiring a basket of goods and services."
    },
    "Central Government Debt": {
        "code": "GC.DOD.TOTL.GD.ZS",
        "unit": "% of GDP",
        "def": "Gross amount of government liabilities owed to creditors as a share of annual economic output."
    },
    "Total Reserves (Includes Gold)": {
        "code": "FI.RES.TOTL.CD",
        "unit": "current US$",
        "def": "Holdings of monetary gold, SDRs, reserves of IMF members, and foreign exchange under monetary authority control."
    },
    "External Balance on Goods and Services": {
        "code": "NE.RSB.GNFS.ZS",
        "unit": "% of GDP",
        "def": "Exports of goods and services minus imports of goods and services expressed as a percentage of GDP."
    }
}

CURRENT_YEAR = datetime.datetime.now().year
DEFAULT_START_YEAR = CURRENT_YEAR - 70
DEFAULT_END_YEAR = CURRENT_YEAR - 1

# -----------------------------------------------------------------------------
# 3. DATA RETRIEVAL & CACHING
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_world_bank_data(country_code, indicator_code, start_yr, end_yr):
    url = f"http://api.worldbank.org/v2/country/{country_code}/indicator/{indicator_code}?date={start_yr}:{end_yr}&format=json&per_page=1000"
    try:
        res = requests.get(url, timeout=12)
        if res.status_code != 200:
            return None, f"API Error: HTTP Status {res.status_code}"
        data = res.json()
        if len(data) < 2 or data[1] is None:
            return None, "No data returned from World Bank API for this query."
        
        records = []
        for entry in data[1]:
            val = entry.get("value")
            yr = entry.get("date")
            if val is not None and yr is not None:
                try:
                    records.append({"Year": int(yr), "Value": float(val)})
                except ValueError:
                    continue
        
        if not records:
            return None, "API returned entries, but all value observations were empty/null."
        
        df = pd.DataFrame(records).sort_values("Year").reset_index(drop=True)
        retrieval_date = datetime.date.today().strftime("%Y-%m-%d")
        return df, retrieval_date
    except Exception as e:
        return None, f"Network or Parsing Exception: {str(e)}"

# -----------------------------------------------------------------------------
# 4. MATHEMATICAL FIT & CALCULUS ENGINES
# -----------------------------------------------------------------------------
def fit_polynomial_model(df, degree):
    if len(df) < degree + 1:
        return None
    
    unique_years = df["Year"].nunique()
    if unique_years < degree + 1:
        return None
    
    base_year = int(df["Year"].min())
    t = df["Year"].values - base_year
    y = df["Value"].values
    
    # Fit degree-d polynomial: P(t) = c_d * t^d + ... + c_0
    coeffs = np.polyfit(t, y, degree)
    poly_func = np.poly1d(coeffs)
    
    y_pred = poly_func(t)
    r2 = r2_score(y, y_pred) if len(y) > 1 else 1.0
    mae = mean_absolute_error(y, y_pred)
    residuals = y - y_pred
    
    # Derivative polynomial functions
    d1_func = np.polyder(poly_func, 1)
    d2_func = np.polyder(poly_func, 2)
    
    return {
        "coeffs": coeffs,
        "poly": poly_func,
        "d1": d1_func,
        "d2": d2_func,
        "base_year": base_year,
        "degree": degree,
        "r2": r2,
        "mae": mae,
        "residuals": residuals,
        "n_obs": len(df),
        "year_min": int(df["Year"].min()),
        "year_max": int(df["Year"].max()),
        "val_mean": float(df["Value"].mean()),
        "t_min": 0,
        "t_max": int(df["Year"].max() - base_year)
    }

def format_poly_equation(coeffs, base_year):
    deg = len(coeffs) - 1
    terms = []
    for i, c in enumerate(coeffs):
        power = deg - i
        if abs(c) < 1e-12:
            continue
        c_str = f"{c:+.4e}" if (abs(c) < 1e-3 or abs(c) > 1e4) else f"{c:+.4f}"
        if power == 0:
            terms.append(f"{c_str}")
        elif power == 1:
            terms.append(f"{c_str}(t)")
        else:
            terms.append(f"{c_str}(t^{power})")
    eq = " ".join(terms)
    if eq.startswith("+"):
        eq = eq[1:].strip()
    return f"y(t) = {eq}, where t = Year - {base_year}"

# -----------------------------------------------------------------------------
# 5. SIDEBAR & CONTROL PANEL
# -----------------------------------------------------------------------------
st.sidebar.header("⚙️ Configuration & Controls")

math_pathway = st.sidebar.selectbox(
    "Mathematical Pathway Level",
    ["Honors Algebra 2", "Precalculus", "Calculus"],
    index=2
)

country_name = st.sidebar.selectbox("Primary Focus Country", list(COUNTRIES.keys()), index=0)
country_code = COUNTRIES[country_name]

indicator_name = st.sidebar.selectbox("Historical Indicator", list(INDICATORS.keys()), index=0)
indicator_info = INDICATORS[indicator_name]

comp_country_name = st.sidebar.selectbox(
    "Comparison Country (Optional)",
    ["None"] + [c for c in COUNTRIES.keys() if c != country_name]
)
comp_country_code = COUNTRIES[comp_country_name] if comp_country_name != "None" else None

st.sidebar.subheader("Model & Visualization Parameters")
poly_degree = st.sidebar.slider("Polynomial Degree (d)", min_value=3, max_value=8, value=3, step=1)
sample_increment = st.sidebar.slider("Graph Sample Increment (Years)", min_value=1, max_value=10, value=1, step=1)
forecast_horizon = st.sidebar.slider("Forward Forecast Horizon (Years)", min_value=0, max_value=50, value=15, step=5)
printer_mode = st.sidebar.checkbox("🖨️ Printer-Friendly View Mode")

# -----------------------------------------------------------------------------
# 6. DATA INITIALIZATION & SESSION STATE MANAGEMENT
# -----------------------------------------------------------------------------
raw_df, retrieval_meta = fetch_world_bank_data(country_code, indicator_info["code"], DEFAULT_START_YEAR, DEFAULT_END_YEAR)

if "working_df" not in st.session_state or st.session_state.get("current_key") != f"{country_code}_{indicator_info['code']}":
    if raw_df is not None:
        st.session_state.working_df = raw_df.copy()
        st.session_state.is_edited = False
        st.session_state.current_key = f"{country_code}_{indicator_info['code']}"
    else:
        st.session_state.working_df = pd.DataFrame(columns=["Year", "Value"])
        st.session_state.is_edited = False

col_reset1, col_reset2 = st.sidebar.columns(2)
if col_reset1.button("Reset Data"):
    if raw_df is not None:
        st.session_state.working_df = raw_df.copy()
        st.session_state.is_edited = False
        st.rerun()

# -----------------------------------------------------------------------------
# 7. MAIN LAYOUT: DATA VALIDATION & EDIT TABLE
# -----------------------------------------------------------------------------
st.subheader("1. Authentic Data Provenance & Interactive Validation Table")

col_meta1, col_meta2 = st.columns([2, 1])
with col_meta1:
    st.markdown(f"**Indicator Definition:** {indicator_info['def']}")
    st.markdown(f"**Units:** {indicator_info['unit']} | **Official Source:** [World Bank API Endpoint](http://api.worldbank.org/v2/country/{country_code}/indicator/{indicator_info['code']}?date={DEFAULT_START_YEAR}:{DEFAULT_END_YEAR})")
with col_meta2:
    st.info(f"**Retrieval Date:** {retrieval_meta if isinstance(retrieval_meta, str) else 'N/A'}\n\n"
            f"**Observed Range Coverage:** {len(st.session_state.working_df)} valid points out of {DEFAULT_END_YEAR - DEFAULT_START_YEAR + 1} requested years.")

if st.session_state.is_edited:
    st.warning("⚠️ **User-Modified Scenario Active:** The baseline dataset has been edited below. Model parameters and fits reflect customized data.")

with st.expander("📝 View & Edit Validated Data Table (Add, Remove, Modify Observations)", expanded=False):
    edited_df = st.data_editor(
        st.session_state.working_df,
        num_rows="dynamic",
        key="data_editor",
        use_container_width=True
    )
    
    # Check for edits
    if not edited_df.equals(st.session_state.working_df):
        # Validation checks
        has_error = False
        if edited_df["Year"].duplicated().any():
            st.error("Validation Error: Duplicate years detected. Each year must be unique.")
            has_error = True
        if edited_df["Value"].isnull().any():
            st.error("Validation Error: Null or missing values detected in the dataset.")
            has_error = True
        
        if not has_error:
            st.session_state.working_df = edited_df.sort_values("Year").reset_index(drop=True)
            st.session_state.is_edited = True
            st.rerun()

df_model = st.session_state.working_df.copy()

# -----------------------------------------------------------------------------
# 8. MATHEMATICAL MODEL FITTING & METRICS
# -----------------------------------------------------------------------------
st.markdown("---")
st.subheader(f"2. Mathematical Model Fitting ({math_pathway} Pathway)")

model_res = fit_polynomial_model(df_model, poly_degree)

if model_res is None:
    st.error(f"Insufficient distinct data points to fit a degree {poly_degree} polynomial model. At least {poly_degree + 1} observations are required.")
    st.stop()

coeffs = model_res["coeffs"]
poly = model_res["poly"]
base_year = model_res["base_year"]
eq_str = format_poly_equation(coeffs, base_year)

# Display Metrics & Model Details
m_col1, m_col2, m_col3, m_col4 = st.columns(4)
m_col1.metric("In-Sample R² Score", f"{model_res['r2']:.4f}")
m_col2.metric("In-Sample MAE", f"{model_res['mae']:.4f} {indicator_info['unit']}")
m_col3.metric("Fitted Observations (n)", model_res['n_obs'])
m_col4.metric("Historical Base Year (t=0)", base_year)

st.latex(eq_str.replace("y(t) =", "P(t) ="))

# Chronological Holdout Validation (If n >= d + 5)
if len(df_model) >= poly_degree + 5:
    split_idx = int(len(df_model) * 0.8)
    train_df = df_model.iloc[:split_idx]
    test_df = df_model.iloc[split_idx:]
    
    train_res = fit_polynomial_model(train_df, poly_degree)
    if train_res is not None:
        t_test = test_df["Year"].values - train_res["base_year"]
        y_test_pred = train_res["poly"](t_test)
        holdout_mae = mean_absolute_error(test_df["Value"].values, y_test_pred)
        st.caption(f"🧪 **Chronological Holdout Validation (80/20 Split):** Training Range: {train_df['Year'].min()}-{train_df['Year'].max()} | Holdout Test MAE: **{holdout_mae:.4f} {indicator_info['unit']}**")

# -----------------------------------------------------------------------------
# 9. INTEGRATED CALCULUS & PATHWAY ANALYSIS
# -----------------------------------------------------------------------------
if math_pathway == "Honors Algebra 2":
    st.markdown("### 📘 Honors Algebra 2 Analysis")
    col_a1, col_a2 = st.columns(2)
    with col_a1:
        st.markdown(f"**Domain:** $t \in [0, {model_res['year_max'] - base_year}]$ (Years {model_res['year_min']} to {model_res['year_max']})")
        st.markdown(f"**Observed Mean:** {model_res['val_mean']:.2f} {indicator_info['unit']}")
        st.markdown(f"**Degree (d):** {poly_degree} ({'Even' if poly_degree % 2 == 0 else 'Odd'})")
    with col_a2:
        lead_coeff = coeffs[0]
        end_beh = f"As $t \\to \\infty, P(t) \\to {'+\\infty' if lead_coeff > 0 else '-\\infty'}$; As $t \\to -\\infty, P(t) \\to {'+\\infty' if (lead_coeff > 0 and poly_degree%2==0) else ('-\\infty' if lead_coeff > 0 else ('+\\infty' if poly_degree%2!=0 else '-\\infty'))}$"
        st.markdown(f"**Leading Coefficient:** `{lead_coeff:+.4e}`")
        st.markdown(f"**End Behavior:** {end_beh}")

elif math_pathway == "Precalculus":
    st.markdown("### 📙 Precalculus Analysis")
    d1 = model_res["d1"]
    roots = d1.roots
    real_roots = roots[np.isreal(roots)].real
    valid_tp = [r + base_year for r in real_roots if 0 <= r <= (model_res['year_max'] - base_year)]
    
    p_col1, p_col2 = st.columns(2)
    with p_col1:
        st.markdown(f"**Polynomial Degree:** {poly_degree} (Maximum possible turning points: {poly_degree - 1})")
        st.markdown(f"**Identified Turning Points in Domain:** {len(valid_tp)} ({', '.join([f'{tp:.1f}' for tp in valid_tp]) if valid_tp else 'None'})")
    with p_col2:
        st.markdown(f"**Global Growth Trend:** {'Overall Ascending' if coeffs[0] > 0 else 'Overall Descending'}")
        st.markdown(f"**Average Rate of Change Across Full Domain:** {((poly(model_res['t_max']) - poly(0)) / model_res['t_max']):.4f} {indicator_info['unit']}/year")

else:  # Calculus Pathway
    st.markdown("### 📕 Calculus Function Analysis")
    d1 = model_res["d1"]
    d2 = model_res["d2"]
    
    # Mathematical notation rendering
    st.latex(f"P'(t) = {format_poly_equation(d1.coeffs, base_year).replace('y(t) = ', '')}")
    st.latex(f"P''(t) = {format_poly_equation(d2.coeffs, base_year).replace('y(t) = ', '')}")
    
    # Critical points & Inflection points
    cp_t = [r.real for r in d1.roots if np.isreal(r) and 0 <= r.real <= model_res['t_max']]
    ip_t = [r.real for r in d2.roots if np.isreal(r) and 0 <= r.real <= model_res['t_max']]
    
    c_col1, c_col2 = st.columns(2)
    with c_col1:
        st.markdown("**Critical Points & Local Extrema ($P'(t) = 0$):**")
        if not cp_t:
            st.write("No interior critical points within historical domain.")
        for t_val in cp_t:
            conc = d2(t_val)
            val = poly(t_val)
            yr = t_val + base_year
            if conc < -1e-6:
                st.write(f"• **Year {yr:.1f} (t={t_val:.1f}):** Local Maximum = {val:.2f} (Second Derivative Test: $P''={conc:.2e} < 0$)")
            elif conc > 1e-6:
                st.write(f"• **Year {yr:.1f} (t={t_val:.1f}):** Local Minimum = {val:.2f} (Second Derivative Test: $P''={conc:.2e} > 0$)")
            else:
                st.write(f"• **Year {yr:.1f} (t={t_val:.1f}):** Inconclusive test ($P''=0$)")
                
    with c_col2:
        st.markdown("**Inflection Points & Concavity ($P''(t) = 0$):**")
        if not ip_t:
            st.write("No inflection points within historical domain.")
        for t_val in ip_t:
            yr = t_val + base_year
            val = poly(t_val)
            st.write(f"• **Year {yr:.1f} (t={t_val:.1f}):** Point of Inflection at value {val:.2f} (Concavity changes direction)")

# -----------------------------------------------------------------------------
# 10. INTERACTIVE SINGLE & BATCH EVALUATOR
# -----------------------------------------------------------------------------
st.markdown("---")
st.subheader("3. Model Prediction & Evaluation Laboratory")

eval_tab1, eval_tab2 = st.columns(2)

with eval_tab1:
    st.markdown("#### Single Year Evaluator")
    eval_year = st.number_input("Select Evaluation Year (1950 - 2100)", min_value=1950, max_value=2100, value=2025, step=1)
    if st.button("Calculate Value"):
        t_eval = eval_year - base_year
        pred_val = poly(t_eval)
        is_in_range = model_res["year_min"] <= eval_year <= model_res["year_max"]
        
        # Check if actual observation exists
        match_obs = df_model[df_model["Year"] == eval_year]
        obs_str = f"{match_obs['Value'].values[0]:.2f}" if not match_obs.empty else "N/A (No direct observation)"
        
        st.markdown(f"**Target Year:** {eval_year} (t = {t_eval})")
        st.markdown(f"**Model Forecast/Fitted Value:** **{pred_val:.4f} {indicator_info['unit']}**")
        st.markdown(f"**Actual Historical Observation:** {obs_str}")
        st.markdown(f"**Domain Classification:** {'🟢 Within Observed Historical Range' if is_in_range else '🔴 Extrapolated (Outside Historical Range)'}")
        
        if not is_in_range:
            st.warning("⚠️ **Extrapolation Caution:** High-degree polynomials can experience Severe Runge Phenomenon outside the historical boundary. Extrapolations should be evaluated with domain uncertainty bounds.")

with eval_tab2:
    st.markdown("#### Batch Year Evaluator")
    batch_mode = st.radio("Batch Selection Method", ["Range", "Custom List"])
    
    if batch_mode == "Range":
        b_start, b_end = st.slider("Batch Year Range", 1950, 2100, (2020, 2030))
        b_step = st.slider("Step Increment", 1, 10, 1)
        batch_years = list(range(b_start, b_end + 1, b_step))
    else:
        custom_inp = st.text_input("Enter comma-separated years:", "2000, 2010, 2020, 2025, 2030")
        try:
            batch_years = [int(y.strip()) for y in custom_inp.split(",") if y.strip().isdigit()]
        except Exception:
            batch_years = []
            st.error("Invalid custom year format.")

    if batch_years:
        batch_results = []
        for y_in in batch_years:
            t_in = y_in - base_year
            val_in = poly(t_in)
            in_b = model_res["year_min"] <= y_in <= model_res["year_max"]
            batch_results.append({
                "Year": y_in,
                "t": t_in,
                "Fitted/Forecast Value": round(val_in, 4),
                "Classification": "In-Domain" if in_b else "Extrapolated"
            })
        batch_df = pd.DataFrame(batch_results)
        st.dataframe(batch_df, use_container_width=True)

# -----------------------------------------------------------------------------
# 11. VISUALIZATION ENGINE & COMPARISON
# -----------------------------------------------------------------------------
st.markdown("---")
st.subheader("4. Dynamic Visualization & Comparative Analytics")

fig, ax = plt.subplots(figsize=(10, 5) if not printer_mode else (8, 4), dpi=150 if not printer_mode else 300)

if printer_mode:
    plt.style.use('grayscale')
else:
    plt.style.use('default')

# Sample points based on user increment
sampled_df = df_model.iloc[::sample_increment]

# Plot original observed points
ax.scatter(df_model["Year"], df_model["Value"], color="gray" if printer_mode else "lightblue", alpha=0.5, label="All Observations")
ax.scatter(sampled_df["Year"], sampled_df["Value"], color="black" if printer_mode else "blue", s=40, zorder=3, label=f"Sampled Points (Step={sample_increment}y)")

# Continuous smooth fitted curve
t_hist_smooth = np.linspace(0, model_res["t_max"], 200)
y_hist_smooth = poly(t_hist_smooth)
ax.plot(t_hist_smooth + base_year, y_hist_smooth, color="black" if printer_mode else "darkblue", linewidth=2, label=f"Fitted Poly Degree {poly_degree}")

# Extrapolation curve
if forecast_horizon > 0:
    t_ext_smooth = np.linspace(model_res["t_max"], model_res["t_max"] + forecast_horizon, 100)
    y_ext_smooth = poly(t_ext_smooth)
    ax.plot(t_ext_smooth + base_year, y_ext_smooth, color="black" if printer_mode else "red", linestyle="--", linewidth=2, label=f"Extrapolation (+{forecast_horizon}y)")

# Optional Comparison Country Plot
if comp_country_code:
    comp_df, _ = fetch_world_bank_data(comp_country_code, indicator_info["code"], DEFAULT_START_YEAR, DEFAULT_END_YEAR)
    if comp_df is not None and not comp_df.empty:
        ax.plot(comp_df["Year"], comp_df["Value"], color="gray" if printer_mode else "orange", linestyle=":", linewidth=2, label=f"Comparison: {comp_country_name}")

# Shading historical domain boundary
ax.axvspan(model_res["year_min"], model_res["year_max"], color="gray", alpha=0.1, label="Historical Boundary Domain")

ax.set_title(f"{indicator_name} - {country_name}", fontsize=12, fontweight="bold")
ax.set_xlabel("Year", fontsize=10)
ax.set_ylabel(indicator_info["unit"], fontsize=10)
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend(loc="best", fontsize=8)

st.pyplot(fig)

# -----------------------------------------------------------------------------
# 12. GENERAL MULTI-CONTEXT CALCULUS SIMULATION ENGINE
# -----------------------------------------------------------------------------
st.markdown("---")
st.subheader("5. Multi-Context Calculus Simulation Laboratory")

CONTEXTS = {
    "Economic Forecasting (World Bank Data Model)": {
        "ind_var": "Time", "ind_unit": "years",
        "dep_var": indicator_name, "dep_unit": indicator_info["unit"],
        "desc": f"Analyzing rate of structural change in {indicator_name} over time."
    },
    "Bioengineering (Drug Concentration in Bloodstream)": {
        "ind_var": "Time after injection", "ind_unit": "hours",
        "dep_var": "Plasma Concentration", "dep_unit": "mg/L",
        "desc": "Evaluating therapeutic onset, peak dosage absorption rate, and renal clearance rates."
    },
    "Sports Science & Biomechanics (Prosthetic Leg Motion)": {
        "ind_var": "Gait Cycle Time", "ind_unit": "seconds",
        "dep_var": "Knee Joint Angular Position", "dep_unit": "degrees",
        "desc": "Studying angular velocity (swing phase) and angular acceleration of a prosthetic knee joint during walking."
    },
    "Business Analytics (Product Adoption Rate)": {
        "ind_var": "Marketing Duration", "ind_unit": "months",
        "dep_var": "Active User Base", "dep_unit": "thousands of users",
        "desc": "Examining customer acquisition velocity and saturation inflection points."
    },
    "Biology (Cellular Colony Growth)": {
        "ind_var": "Incubation Time", "ind_unit": "hours",
        "dep_var": "Biomass Density", "dep_unit": "g/L",
        "desc": "Analyzing exponential growth phases, nutrient depletion, and maximum biomass decay rates."
    }
}

selected_context_key = st.selectbox("Select STEM Context Domain", list(CONTEXTS.keys()))
ctx = CONTEXTS[selected_context_key]

c_col1, c_col2 = st.columns(2)
with c_col1:
    x_val = st.slider(f"Instantaneous Point Input ({ctx['ind_var']})", float(model_res['year_min']), float(model_res['year_max']), float(model_res['year_min'] + 5), step=0.5)
with c_col2:
    sec_range = st.slider(f"Secant Interval Range [{ctx['ind_var']}]", float(model_res['year_min']), float(model_res['year_max']), (float(model_res['year_min']), float(model_res['year_min'] + 10)))

t_point = x_val - base_year
y_point = poly(t_point)
d1_point = model_res["d1"](t_point)
d2_point = model_res["d2"](t_point)

t_sec1, t_sec2 = sec_range[0] - base_year, sec_range[1] - base_year
y_sec1, y_sec2 = poly(t_sec1), poly(t_sec2)
avg_rate = (y_sec2 - y_sec1) / (t_sec2 - t_sec1) if t_sec2 != t_sec1 else 0.0

st.markdown(f"#### Contextual Calculus Interpretations ({selected_context_key})")

interp_col1, interp_col2, interp_col3 = st.columns(3)
with interp_col1:
    st.metric(f"State Function P(t)", f"{y_point:.2f} {ctx['dep_unit']}")
    st.caption(f"**Interpretation:** At year {x_val:.1f}, the absolute value of {ctx['dep_var']} is **{y_point:.2f} {ctx['dep_unit']}**.")

with interp_col2:
    st.metric(f"1st Derivative P'(t) [Instantaneous Rate]", f"{d1_point:.4f} {ctx['dep_unit']}/{ctx['ind_unit']}")
    st.caption(f"**Interpretation:** At year {x_val:.1f}, {ctx['dep_var']} is changing at a rate of **{d1_point:.4f} {ctx['dep_unit']} per {ctx['ind_unit']}**.")

with interp_col3:
    st.metric(f"2nd Derivative P''(t) [Concavity / Acceleration]", f"{d2_point:.4f} {ctx['dep_unit']}/{ctx['ind_unit']}²")
    st.caption(f"**Interpretation:** At year {x_val:.1f}, the rate of change itself is {'accelerating' if d2_point > 0 else 'decelerating'} at **{d2_point:.4f} {ctx['dep_unit']}/{ctx['ind_unit']}²**.")

st.info(f"📊 **Average Rate of Change (Secant Line over [{sec_range[0]:.0f}, {sec_range[1]:.0f}]):** **{avg_rate:.4f} {ctx['dep_unit']}/{ctx['ind_unit']}**")

# -----------------------------------------------------------------------------
# 13. EXPORT MODULE (CSV, EXCEL, PDF)
# -----------------------------------------------------------------------------
st.markdown("---")
st.subheader("6. Export & Reporting Laboratory")

# 1. CSV Download
csv_data = df_model.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Download Dataset (CSV)",
    data=csv_data,
    file_name=f"{country_code}_{indicator_info['code']}_data.csv",
    mime="text/csv"
)

# 2. Excel Download (XlsxWriter)
excel_buffer = io.BytesIO()
with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
    # Working data sheet
    df_model.to_excel(writer, sheet_name="Working Data", index=False)
    
    # Model Summary Sheet
    summary_data = {
        "Parameter": ["Country", "Indicator", "Polynomial Degree", "Base Year", "R-Squared", "MAE", "User Edited"],
        "Value": [country_name, indicator_name, poly_degree, base_year, round(model_res['r2'], 4), round(model_res['mae'], 4), str(st.session_state.is_edited)]
    }
    pd.DataFrame(summary_data).to_excel(writer, sheet_name="Model Summary", index=False)
    
    # Forecast Sheet
    if 'batch_df' in locals():
        batch_df.to_excel(writer, sheet_name="Batch Predictions", index=False)

excel_data = excel_buffer.getvalue()
st.download_button(
    label="📊 Download Complete Report (Excel .xlsx)",
    data=excel_data,
    file_name=f"{country_code}_{indicator_info['code']}_report.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# 3. PDF Report Generation (ReportLab)
def generate_pdf_report():
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    story = []

    # Title
    story.append(Paragraph(f"STEM Data Science Report: {country_name}", styles['Title']))
    story.append(Paragraph(f"<b>Indicator:</b> {indicator_name} ({indicator_info['unit']})", styles['Subtitle']))
    story.append(Spacer(1, 12))

    # Metadata Table
    meta_table_data = [
        ["Polynomial Degree", str(poly_degree), "Base Year (t=0)", str(base_year)],
        ["In-Sample R²", f"{model_res['r2']:.4f}", "In-Sample MAE", f"{model_res['mae']:.4f}"],
        ["Observation Count", str(model_res['n_obs']), "User-Modified Scenario", str(st.session_state.is_edited)]
    ]
    t_meta = Table(meta_table_data, colWidths=[120, 120, 120, 120])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.whitesmoke),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 9)
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 12))

    # Save plot image to buffer and insert into PDF
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format='png', bbox_inches='tight', dpi=200)
    img_buf.seek(0)
    story.append(RLImage(img_buf, width=500, height=250))
    story.append(Spacer(1, 12))

    # Equation
    story.append(Paragraph(f"<b>Fitted Model Equation:</b> {eq_str}", styles['Normal']))
    story.append(Spacer(1, 12))

    # Data Preview Table
    story.append(Paragraph("<b>Data Observation Sample:</b>", styles['Heading3']))
    preview_data = [["Year", f"Value ({indicator_info['unit']})"]] + [[str(r["Year"]), f"{r['Value']:.2f}"] for _, r in df_model.head(8).iterrows()]
    t_data = Table(preview_data, colWidths=[150, 250])
    t_data.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey)
    ]))
    story.append(t_data)

    doc.build(story)
    return pdf_buffer.getvalue()

pdf_bytes = generate_pdf_report()
st.download_button(
    label="📄 Download Executive STEM Summary (PDF)",
    data=pdf_bytes,
    file_name=f"{country_code}_{indicator_info['code']}_summary.pdf",
    mime="application/pdf"
)

# -----------------------------------------------------------------------------
# 14. DEPLOYMENT & VERIFICATION MANUAL
# -----------------------------------------------------------------------------
with st.expander("📚 STEM Student Manual: Code Architecture, Verification & Deployment"):
    st.markdown("""
    ### 1. Local Setup & Execution
    Run the following commands in your local terminal:
    ```bash
    pip install -r requirements.txt
    python -m streamlit run app.py
    ```

    ### 2. Streamlit Community Cloud Deployment
    1. Push `app.py` and `requirements.txt` to a GitHub repository.
    2. Log into [Streamlit Community Cloud](https://share.streamlit.io/).
    3. Select **New App**, choose your repository, set main file to `app.py`, and click **Deploy**.

    ### 3. Render Web Service Deployment
    - **Build Command:** `pip install -r requirements.txt`
    - **Start Command:**
      ```bash
      python -m streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT
      ```

    ### 4. Mathematical Assumptions & Diagnostic Safeguards
    - **Centered Time Transformation:** Data years are centered at $t = \text{Year} - \text{Base Year}$ to prevent matrix ill-conditioning in polynomial fitting.
    - **In-Sample vs. Out-of-Sample Metrics:** $R^2$ and MAE computed over historical data reflect fitting fidelity, not future prediction accuracy.
    - **Runge Phenomenon Caution:** High-degree polynomials ($d \ge 5$) can oscillate wildly near endpoints and outside historical boundaries.
    """)
