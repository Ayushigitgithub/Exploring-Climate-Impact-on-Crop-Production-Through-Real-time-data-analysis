# project_samarth.py
# Streamlit prototype for "Project Samarth — Agri-Climate Insights Q&A"
# - Live rainfall from data.gov.in (CSV API)
# - Crop production from local CSV (wide OR long supported)
# - Simple NL Q&A with rule-based parsing for required sample questions

# project_samarth.py
# project_samarth.py
# project_samarth.py
# project_samarth.py
# Streamlit app: live rainfall (data.gov.in) + crop CSV (All-India + state queries)

# project_samarth.py
# project_samarth.py

# project_samarth.py
# project_samarth.py

# project_samarth.py

import re
import urllib.parse
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Project Samarth – Agri-Climate Q&A Prototype", layout="wide")

# ───────────────────────── CONFIG ─────────────────────────
# Rainfall: one string already contains dataset + key; we append filters at runtime.
RAINFALL_API_WITH_KEY = (
    "https://api.data.gov.in/resource/6c05cd1b-ed59-40c2-bc31-e314f39c6971"
    "?api-key=579b464db66ec23bdd000001372549262ed143327f9f31b45699bbc5"
    "&format=csv&limit=10000"
)

# Crops: fixed local CSV (no input box)
DEFAULT_CROP_CSV = "rice_production.csv"

# ───────────────────────── HELPERS ─────────────────────────
def _to_float(x):
    try:
        return float(str(x).replace(",", "").strip())
    except Exception:
        return None

def _year_from_label(label):
    m = re.search(r"(20\d{2})", str(label))
    return int(m.group(1)) if m else None

# ───────────────────────── LIVE RAINFALL ─────────────────────────
@st.cache_data(show_spinner=False)
def load_rainfall_one_year(year: int) -> pd.DataFrame:
    url = RAINFALL_API_WITH_KEY + "&" + urllib.parse.urlencode({"filters[Year]": str(year)})
    df = pd.read_csv(url)
    cols_low = {c.lower().strip(): c for c in df.columns}

    if "state" in cols_low:
        df.rename(columns={cols_low["state"]: "State"}, inplace=True)
    if "year" in cols_low:
        df.rename(columns={cols_low["year"]: "Year"}, inplace=True)

    rf_col = None
    for k in ["avg_rainfall", "rainfall", "rf", "average_rainfall"]:
        if k in cols_low:
            rf_col = cols_low[k]
            break
    if rf_col is None:
        for c in df.columns:
            if "rain" in c.lower():
                rf_col = c
                break
    if rf_col is None:
        return pd.DataFrame(columns=["State", "Year", "Rainfall"])

    out = df[["State", "Year", rf_col]].copy()
    out.rename(columns={rf_col: "Rainfall"}, inplace=True)
    out["Rainfall"] = out["Rainfall"].apply(_to_float)
    return out.dropna(subset=["State"])

@st.cache_data(show_spinner=False)
def rainfall_last_n_years(latest_year: int, n_years: int) -> pd.DataFrame:
    frames = [load_rainfall_one_year(y) for y in range(latest_year - n_years + 1, latest_year + 1)]
    if not frames:
        return pd.DataFrame(columns=["State", "Year", "Rainfall"])
    rf = pd.concat(frames, ignore_index=True)
    rf = rf[rf["State"].str.strip().str.lower() != "all india"]
    return rf

def avg_annual_rainfall(rf_df: pd.DataFrame, state: str, start_year: int, end_year: int):
    sub = rf_df[
        (rf_df["State"].str.lower() == state.lower()) &
        (rf_df["Year"].between(start_year, end_year))
    ]
    if sub.empty:
        return None
    per_year = sub.groupby("Year")["Rainfall"].mean()
    return float(per_year.mean())

# ───────────────────────── CROPS CSV ─────────────────────────
@st.cache_data(show_spinner=False)
def load_crop_csv(path: str) -> pd.DataFrame:
    """
    Normalize to: State, Crop, Year, Production (float).
    Handles long and wide tables (e.g., 'Production-2020-21').
    """
    raw = pd.read_csv(path)
    df = raw.copy()
    cols_low = {c.lower().strip(): c for c in df.columns}

    state_col = cols_low.get("state") or cols_low.get("state name") or cols_low.get("state_name")
    crop_col  = cols_low.get("crop") or cols_low.get("commodity") or cols_low.get("crop_name")
    year_col  = cols_low.get("year")

    prod_col = None
    for k in ["production", "prod_qty", "quantity", "production_mt", "total_production"]:
        if k in cols_low:
            prod_col = cols_low[k]
            break

    if state_col:
        df.rename(columns={state_col: "State"}, inplace=True)
    else:
        for c in df.columns:
            if "state" in c.lower():
                df.rename(columns={c: "State"}, inplace=True)
                break

    if crop_col:
        df.rename(columns={crop_col: "Crop"}, inplace=True)
    else:
        df["Crop"] = "Rice"

    if year_col:
        df.rename(columns={year_col: "Year"}, inplace=True)

    if prod_col:  # already long
        df.rename(columns={prod_col: "Production"}, inplace=True)
        df["Production"] = df["Production"].apply(_to_float)
        keep = [c for c in ["State", "Crop", "Year", "Production"] if c in df.columns]
        long_df = df[keep].copy()
        if "Year" not in long_df.columns:
            long_df["Year"] = None
        return long_df.dropna(subset=["State", "Production"])

    # wide -> melt year columns
    wide_cols = [c for c in df.columns if any(k in c.lower() for k in ["prod", "yield", "area"])]
    prod_like = [c for c in wide_cols if "prod" in c.lower()]
    target_cols = prod_like if prod_like else wide_cols

    long_rows = []
    for c in target_cols:
        y = _year_from_label(c)
        if y is None:
            continue
        tmp = df[["State", "Crop"]].copy()
        tmp["Year"] = y
        tmp["Production"] = df[c].apply(_to_float)
        long_rows.append(tmp)

    if not long_rows:
        return pd.DataFrame(columns=["State", "Crop", "Year", "Production"])

    long_df = pd.concat(long_rows, ignore_index=True)
    return long_df.dropna(subset=["State"])

def top_crops(df: pd.DataFrame, state: str, y0: int, y1: int, m: int) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Crop", "Total Production"])
    sub = df[
        (df["State"].str.lower() == state.lower()) &
        (df["Year"].notna()) &
        (df["Year"].astype(float).between(y0, y1))
    ]
    if sub.empty:
        return pd.DataFrame(columns=["Crop", "Total Production"])
    out = (
        sub.groupby("Crop")["Production"]
        .sum()
        .sort_values(ascending=False)
        .head(m)
        .reset_index()
    )
    out.rename(columns={"Production": "Total Production"}, inplace=True)
    return out

def all_india_by_year_range(df: pd.DataFrame, y0: int, y1: int) -> pd.DataFrame:
    """All-India totals filtered to [y0, y1]."""
    if df.empty:
        return pd.DataFrame(columns=["Year", "Crop", "Production"])
    sub = df[df["Year"].notna()].copy()
    sub["Year"] = sub["Year"].astype(int)
    sub = sub[sub["Year"].between(y0, y1)]
    out = (
        sub.groupby(["Year", "Crop"])["Production"]
        .sum()
        .reset_index()
        .sort_values(["Year", "Crop"])
    )
    return out

# ───────────────────────── UI ─────────────────────────
st.title("Project Samarth – Agri-Climate Q&A Prototype")

left, right = st.columns([1, 1])

# Rainfall controls
with left:
    st.subheader("Rainfall (live, data.gov.in)")
    rf_latest_year = st.number_input("Latest year to consider (for rainfall averaging)", value=2025, step=1)
    rf_n_years     = st.slider("Last N years (rainfall average window)", 1, 10, 3)

# Crops controls (mirrors rainfall)
with right:
    st.subheader("Crops")
    st.caption(f"Using local file: **{DEFAULT_CROP_CSV}**")
    _crop_summary_placeholder = st.empty()
    crops_latest_year = st.number_input("Latest year to consider (for crop analysis)", value=2025, step=1)
    crops_n_years     = st.slider("Last N years (crop window)", 1, 10, 3)

# Load data
rain_df = rainfall_last_n_years(int(rf_latest_year), int(rf_n_years))
crop_df = load_crop_csv(DEFAULT_CROP_CSV)

# Crop summary (records/states/years) under the Crops header
try:
    _yrs = pd.to_numeric(crop_df.get("Year", pd.Series(dtype=float)), errors="coerce").dropna()
    _ymin = int(_yrs.min()) if not _yrs.empty else None
    _ymax = int(_yrs.max()) if not _yrs.empty else None
    _states = int(crop_df["State"].nunique()) if "State" in crop_df.columns and not crop_df.empty else 0
    _crop_summary_placeholder.caption(
        f"Records: {len(crop_df):,} • States: {_states} • Years: "
        f"{_ymin if _ymin is not None else '—'}–{_ymax if _ymax is not None else '—'}"
    )
    if _ymax is not None and (crops_latest_year > _ymax or (_ymin is not None and crops_latest_year < _ymin)):
        st.warning(f"Crops years available: {_ymin}–{_ymax}. Adjust the crop year controls if needed.")
except Exception:
    _crop_summary_placeholder.caption("Records: — • States: — • Years: —")

# Preview section
with st.expander("Preview first 5 rows"):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Rainfall (sample of current window)**")
        st.dataframe(rain_df.head())
    with c2:
        st.markdown("**Crops (normalized)**")
        st.dataframe(crop_df.head())

st.markdown("---")
st.header("Ask a question")

cL, cR = st.columns(2)
with cL:
    state_x = st.text_input("State X", value="Kerala")
with cR:
    state_y = st.text_input("State Y", value="Maharashtra")

m_top = st.slider("Top M crops to list", 1, 10, 3)

if st.button("Answer"):
    # Rainfall window
    rf_y1 = int(rf_latest_year)
    rf_y0 = rf_y1 - int(rf_n_years) + 1

    # Crops window (independent)
    cr_y1 = int(crops_latest_year)
    cr_y0 = cr_y1 - int(crops_n_years) + 1

    # 1) Rainfall averages
    rx = avg_annual_rainfall(rain_df, state_x, rf_y0, rf_y1)
    ry = avg_annual_rainfall(rain_df, state_y, rf_y0, rf_y1)

    st.subheader("Average annual rainfall")
    st.markdown(f"- **{state_x}**: {rx:.4f}" if rx is not None else f"- **{state_x}**: not available")
    st.markdown(f"- **{state_y}**: {ry:.4f}" if ry is not None else f"- **{state_y}**: not available")

    # 2) Top crops (state-wise), using crop window
    st.subheader("Top 3 crops by production (from CSV)")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Top crops in {state_x} ({cr_y0}–{cr_y1})**")
        tx = top_crops(crop_df, state_x, cr_y0, cr_y1, m_top)
        st.dataframe(tx if not tx.empty else pd.DataFrame(columns=["Crop", "Total Production"]))
        if tx.empty:
            st.info("No crop data found for this state/period in the CSV.")
    with c2:
        st.markdown(f"**Top crops in {state_y} ({cr_y0}–{cr_y1})**")
        ty = top_crops(crop_df, state_y, cr_y0, cr_y1, m_top)
        st.dataframe(ty if not ty.empty else pd.DataFrame(columns=["Crop", "Total Production"]))
        if ty.empty:
            st.info("No crop data found for this state/period in the CSV.")

    # 3) All-India totals (crop window)
    st.subheader("All-India totals by year")
    ai = all_india_by_year_range(crop_df, cr_y0, cr_y1)
    if ai.empty:
        st.info("Could not compute All-India aggregation from the CSV for the selected crop year window.")
    else:
        st.dataframe(ai)

    # Keep ONLY rainfall source (per your request)
    st.markdown("**Rainfall source (live): data.gov.in**")
    st.code(RAINFALL_API_WITH_KEY, language="text")
    st.caption(f"Refreshed at: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
