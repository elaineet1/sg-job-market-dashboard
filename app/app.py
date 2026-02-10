"""
Singapore Job Market Insights Dashboard

Business objective:
- Employers: benchmark salary, understand demand, sense competition
- Jobseekers: shortlist higher pay roles, prioritise less competitive areas

Data files:
- data/cleaned/SGJobData_cleaned_stage1.csv   (base df, demand/trend)
- data/cleaned/SGJobData_salary_eda_tidy.csv  (salary df_salary, salary/competition)
"""

from pathlib import Path
import json
import ast

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(page_title="Singapore Job Market Insights", layout="wide")
st.title("Singapore Job Market Insights")
st.caption("Hiring demand, salary benchmarks, and application competition, built from cleaned job listing data.")


# ----------------------------
# Helpers
# ----------------------------
def extract_category_names(value) -> list[str]:
    """Convert category JSON string into list of category names."""
    if pd.isna(value):
        return []

    s = str(value).strip()
    if not s:
        return []

    try:
        parsed = json.loads(s)
    except Exception:
        try:
            parsed = ast.literal_eval(s)
        except Exception:
            return []

    names: list[str] = []
    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict) and "category" in item:
                names.append(str(item["category"]).strip())
    elif isinstance(parsed, dict) and "category" in parsed:
        names.append(str(parsed["category"]).strip())

    return [n for n in names if n]


def safe_to_datetime(df: pd.DataFrame, col: str) -> None:
    """Convert a column to datetime safely (in place)."""
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")


# ----------------------------
# Load data (cached)
# ----------------------------
@st.cache_data(show_spinner=False)
def load_csvs(dev_sample: bool = True, n_sample: int = 200_000) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load CSVs and preprocess once (cached)."""
    project_root = Path(__file__).resolve().parents[1]

    base_path = project_root / "data" / "cleaned" / "SGJobData_cleaned_stage1_sample.csv"
    sal_path  = project_root / "data" / "cleaned" / "SGJobData_salary_eda_tidy.csv"

    # If full dataset is missing (like on Streamlit Cloud), fall back to sample files
    if not base_path.exists():
        base_path = project_root / "data" / "sample" / "SGJobData_cleaned_stage1_sample.csv"

    if not sal_path.exists():
        sal_path = project_root / "data" / "sample" / "SGJobData_salary_eda_tidy_sample.csv"

    df_base = pd.read_csv(base_path)
    df_sal  = pd.read_csv(sal_path)

    # TEMP: reduce size to keep WSL stable while developing
    if dev_sample:
        df_base = df_base.sample(n=min(n_sample, len(df_base)), random_state=42)
        df_sal  = df_sal.sample(n=min(n_sample, len(df_sal)), random_state=42)

    # keep the rest of your existing preprocessing below (category cleaning etc.)
    return df_base, df_sal



    # TEMP: reduce size to keep WSL stable while developing
    if dev_sample:
        df_base = df_base.sample(n=min(n_sample, len(df_base)), random_state=42)
        df_sal = df_sal.sample(n=min(n_sample, len(df_sal)), random_state=42)

    # Clean category once and cache result
    if "category" in df_sal.columns:
        df_sal["category_name"] = df_sal["category"].apply(extract_category_names)
        df_sal = df_sal.explode("category_name")
        df_sal["category_name"] = df_sal["category_name"].astype(str).str.strip()
        df_sal = df_sal[df_sal["category_name"] != ""]

    # Date parsing (salary file has these columns based on your screenshot)
    safe_to_datetime(df_sal, "new_posting_date")
    safe_to_datetime(df_sal, "original_posting_date")
    safe_to_datetime(df_sal, "expiry_date")

    # Base df date columns might differ, we try a few common ones
    for c in ["new_posting_date", "original_posting_date", "metadata_newPostingDate", "metadata_originalPostingDate"]:
        safe_to_datetime(df_base, c)

    return df_base, df_sal


# Toggle: keep dev_sample True until you deploy
DEV_SAMPLE = True  # set to False when deploying
df, df_salary = load_csvs(dev_sample=DEV_SAMPLE, n_sample=200_000)

# ---- Ensure chosen date columns are datetime for charts ----
date_col_df = None
for c in ["new_posting_date", "original_posting_date", "metadata_newPostingDate", "metadata_originalPostingDate"]:
    if c in df.columns:
        date_col_df = c
        break

if date_col_df:
    df[date_col_df] = pd.to_datetime(df[date_col_df], errors="coerce")

date_col_sal = None
for c in ["new_posting_date", "original_posting_date", "expiry_dat"]:
    if c in df_salary.columns:
        date_col_sal = c
        break

if date_col_sal:
    df_salary[date_col_sal] = pd.to_datetime(df_salary[date_col_sal], errors="coerce")

# ---- Ensure category_name exists in df_salary ----
if "category_name" not in df_salary.columns:
    if "category" in df_salary.columns:
        df_salary["category_name"] = df_salary["category"].apply(extract_category_names)
        df_salary = df_salary.explode("category_name")
        df_salary["category_name"] = df_salary["category_name"].astype(str).str.strip()
        df_salary = df_salary[df_salary["category_name"] != ""]
    else:
        # last-resort fallback (prevents app from crashing)
        df_salary["category_name"] = "Unknown"


# ----------------------------
# How to use (business tone)
# ----------------------------
with st.expander("How to use this dashboard", expanded=True):
    st.markdown(
        """
- **Employers:** Use **salary by category and position level** to benchmark offers and support hiring budget planning.
  Use **views vs applications** to sense market competitiveness and adjust role requirements, compensation, or sourcing strategy.
- **Jobseekers:** Use **salary by category and position level** to shortlist roles with stronger pay potential.
  Use **views vs applications** to prioritise applications, categories with lower applications per view can be less competitive.
        """.strip()
    )


# ----------------------------
# Sidebar filters
# ----------------------------
st.sidebar.header("Filters")

# Keyword (salary dataset has job_title)
keyword = st.sidebar.text_input("Job title keyword", value="")

# Category (from cleaned category_name)
cat_options = (
    df_salary["category_name"]
    .dropna()
    .value_counts()
    .head(200)
    .index
    .tolist()
)
selected_categories = st.sidebar.multiselect("Category", options=cat_options, default=[])

# Position level
pos_options = sorted(df_salary["position_level"].dropna().astype(str).unique().tolist()) if "position_level" in df_salary.columns else []
selected_pos = st.sidebar.multiselect("Position level", options=pos_options, default=[])

# Employment type
emp_options = sorted(df_salary["employment_type"].dropna().astype(str).unique().tolist()) if "employment_type" in df_salary.columns else []
selected_emp = st.sidebar.multiselect("Employment type", options=emp_options, default=[])

# Salary range slider
salary_range = None
if "avg_salary" in df_salary.columns and pd.api.types.is_numeric_dtype(df_salary["avg_salary"]):
    s = df_salary["avg_salary"].dropna()
    if not s.empty:
        p01 = float(s.quantile(0.01))
        p99 = float(s.quantile(0.99))
        salary_range = st.sidebar.slider(
            "Avg salary range",
            min_value=int(max(0, np.floor(p01))),
            max_value=int(np.ceil(p99)),
            value=(int(max(0, np.floor(p01))), int(np.ceil(p99))),
            step=100,
        )

# Reliability sliders
min_group_count = st.sidebar.slider(
    "Minimum records per group (reliability)",
    min_value=5,
    max_value=200,
    value=30,
    step=5,
)

st.sidebar.divider()
st.sidebar.caption("Tip: During development, the app uses a sample to keep WSL stable. Switch DEV_SAMPLE=False when deploying.")


# ----------------------------
# Apply filters (salary dataset)
# ----------------------------
df_salary_f = df_salary.copy()

if keyword and "job_title" in df_salary_f.columns:
    df_salary_f = df_salary_f[df_salary_f["job_title"].astype(str).str.contains(keyword, case=False, na=False)]

if selected_categories:
    df_salary_f = df_salary_f[df_salary_f["category_name"].isin(selected_categories)]

if selected_pos and "position_level" in df_salary_f.columns:
    df_salary_f = df_salary_f[df_salary_f["position_level"].astype(str).isin([str(x) for x in selected_pos])]

if selected_emp and "employment_type" in df_salary_f.columns:
    df_salary_f = df_salary_f[df_salary_f["employment_type"].astype(str).isin([str(x) for x in selected_emp])]

if salary_range is not None:
    lo, hi = salary_range
    df_salary_f = df_salary_f[df_salary_f["avg_salary"].between(lo, hi, inclusive="both")]


# ----------------------------
# KPIs
# ----------------------------
k1, k2, k3, k4 = st.columns(4)

k1.metric("Salary records (filtered)", f"{len(df_salary_f):,}")
k2.metric("Categories (filtered)", f"{df_salary_f['category_name'].nunique():,}")

if "avg_salary" in df_salary_f.columns and not df_salary_f.empty:
    k3.metric("Median avg salary", f"{df_salary_f['avg_salary'].median():,.0f}")
else:
    k3.metric("Median avg salary", "N/A")

# Competition proxy: applications per 100 views
if {"total_views", "total_job_applications"}.issubset(df_salary_f.columns) and not df_salary_f.empty:
    tmp = df_salary_f.dropna(subset=["total_views", "total_job_applications"]).copy()
    tmp = tmp[tmp["total_views"] > 0]
    if not tmp.empty and tmp["total_views"].sum() > 0:
        rate = (tmp["total_job_applications"].sum() / tmp["total_views"].sum()) * 100
        k4.metric("Applications per 100 views", f"{rate:.2f}")
    else:
        k4.metric("Applications per 100 views", "N/A")
else:
    k4.metric("Applications per 100 views", "N/A")

st.divider()


# ----------------------------
# Tabs
# ----------------------------
tab1, tab2, tab3 = st.tabs(["Hiring Demand", "Salary Benchmarks", "Competition Signals"])


# ===== Tab 1: Hiring Demand (using df_salary dates for simplicity) =====
with tab1:
    st.subheader("Hiring demand and trend")

    st.markdown("**Job postings trend (monthly, based on posting date in salary dataset)**")

    # Use new_posting_date if available, else original_posting_date
    date_col = "new_posting_date" if "new_posting_date" in df_salary_f.columns else "original_posting_date"
    if date_col in df_salary_f.columns and df_salary_f[date_col].notna().any():
        tmp = df_salary_f.dropna(subset=[date_col]).copy()
        tmp["month"] = tmp[date_col].dt.to_period("M").dt.to_timestamp()
        trend = tmp.groupby("month", as_index=False).size().rename(columns={"size": "job_postings"})

        fig = px.line(trend, x="month", y="job_postings", markers=True)
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No usable posting date found for trend chart.")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Top companies by postings (filtered salary dataset)**")
        if "company_name" in df_salary_f.columns and not df_salary_f.empty:
            top_n = st.slider("Top N companies", 5, 30, 10, 1, key="top_companies")
            top_companies = df_salary_f["company_name"].dropna().value_counts().head(top_n).reset_index()
            top_companies.columns = ["company_name", "job_postings"]

            fig = px.bar(top_companies, x="job_postings", y="company_name", orientation="h")
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("company_name not available or no data after filtering.")

    with c2:
        st.markdown("**Top categories by postings (filtered)**")
        top_n = st.slider("Top N categories", 5, 30, 10, 1, key="top_categories")
        top_cats = df_salary_f["category_name"].dropna().value_counts().head(top_n).reset_index()
        top_cats.columns = ["category_name", "job_postings"]

        fig = px.bar(top_cats, x="job_postings", y="category_name", orientation="h")
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Preview filtered data (first 200 rows)"):
        preview = df_salary_f.copy()

    # Drop the messy raw category column if we have cleaned names
    if "category_name" in preview.columns:
        preview = preview.drop(columns=["category"], errors="ignore")

        # Move category_name to the FIRST column
        cols = ["category_name"] + [c for c in preview.columns if c != "category_name"]
        preview = preview[cols]

    st.dataframe(preview.head(200), use_container_width=True)




# ===== Tab 2: Salary Benchmarks =====
with tab2:
    st.subheader("Salary benchmarks")

    if df_salary_f.empty:
        st.warning("No salary data after filtering. Try loosening filters.")
    else:
        a, b = st.columns([1, 1])

        with a:
            st.markdown("**Avg salary distribution**")
            fig = px.histogram(df_salary_f.dropna(subset=["avg_salary"]), x="avg_salary", nbins=40)
            fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)

        with b:
            st.markdown("**Top categories by mean salary (reliability filtered)**")
            summary = (
                df_salary_f.dropna(subset=["category_name", "avg_salary"])
                .groupby("category_name")
                .agg(
                    record_count=("avg_salary", "size"),
                    mean_salary=("avg_salary", "mean"),
                    median_salary=("avg_salary", "median"),
                )
                .reset_index()
            )

            summary = summary[summary["record_count"] >= min_group_count]
            summary = summary.sort_values(["mean_salary", "record_count"], ascending=[False, False])

            top_n = st.slider("Top N categories", 5, 30, 10, 1, key="top_salary_cats")
            top_summary = summary.head(top_n)

            fig = px.bar(top_summary, x="mean_salary", y="category_name", orientation="h")
            fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Category salary table (mean, median, count)**")
        st.dataframe(
            summary.head(50).style.format({"mean_salary": "{:,.0f}", "median_salary": "{:,.0f}"}),
            use_container_width=True,
        )

        if "position_level" in df_salary_f.columns:
            st.markdown("**Salary by position level (box plot)**")
            fig = px.box(
                df_salary_f.dropna(subset=["avg_salary", "position_level"]),
                x="position_level",
                y="avg_salary",
                points=False,
            )
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)


# ===== Tab 3: Competition Signals =====
with tab3:
    st.subheader("Competition signals (views and applications)")

    required = {"total_views", "total_job_applications", "category_name"}
    if not required.issubset(df_salary_f.columns):
        st.info("Competition charts require total_views and total_job_applications columns.")
    else:
        tmp = df_salary_f.dropna(subset=["total_views", "total_job_applications", "category_name"]).copy()
        tmp = tmp[tmp["total_views"] > 0]

        if tmp.empty:
            st.warning("No usable views/applications data after filtering.")
        else:
            comp = (
                tmp.groupby("category_name")
                .agg(total_views=("total_views", "sum"), total_applications=("total_job_applications", "sum"))
                .reset_index()
            )
            comp["applications_per_100_views"] = (comp["total_applications"] / comp["total_views"]) * 100

            min_total_views = st.slider(
                "Minimum total views per category (stability filter)",
                min_value=0,
                max_value=int(comp["total_views"].quantile(0.95)) if comp["total_views"].max() > 0 else 1000,
                value=0,
                step=100,
            )
            comp = comp[comp["total_views"] >= min_total_views]

            left, right = st.columns(2)

            with left:
                st.markdown("**Applications per 100 views by category**")
                top_n = st.slider("Top N categories (by competition)", 5, 30, 10, 1, key="top_comp")
                comp_sorted = comp.sort_values("applications_per_100_views", ascending=False).head(top_n)

                fig = px.bar(comp_sorted, x="applications_per_100_views", y="category_name", orientation="h")
                fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig, use_container_width=True)

            with right:
                st.markdown("**Views vs applications (category level)**")
                fig = px.scatter(comp, x="total_views", y="total_applications", hover_name="category_name")
                fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("**Category competition table**")
            st.dataframe(
                comp.sort_values("applications_per_100_views", ascending=False)
                .head(50)
                .style.format({"applications_per_100_views": "{:.2f}"}),
                use_container_width=True,
            )

