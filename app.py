import os
import numpy as np
import pandas as pd
import plotly.express as px

from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc

# ========== Load & Prepare Data ==========
MAIN_CSV = "data/clean_loan_small.csv"

def safe_read_csv(path, **kwargs):
    return pd.read_csv(path, **kwargs) if os.path.exists(path) else None

if not os.path.exists(MAIN_CSV):
    raise FileNotFoundError(
        f"File {MAIN_CSV} not found. Create it with create_small_data.py or check the path."
    )

df = pd.read_csv(MAIN_CSV)
df.columns = df.columns.str.strip().str.lower()

feature_importance = safe_read_csv("data/feature_importance.csv")
feature_defs_csv   = safe_read_csv("data/feature_definitions_filtered.csv")

# loan_status -> 0/1
if "loan_status" not in df.columns:
    raise ValueError("Column 'loan_status' not found.")
if df["loan_status"].dtype == object:
    df["loan_status"] = (
        df["loan_status"].astype(str).str.strip().str.lower()
        .map({"fully paid": 0, "paid": 0, "charged off": 1, "default": 1})
        .fillna(0).astype(int)
    )

# income_group
if "income_group" not in df.columns:
    if "annual_inc" in df.columns:
        df["income_group"] = pd.qcut(
            df["annual_inc"], 4, labels=["Low", "Lower-Mid", "Upper-Mid", "High"]
        )
    else:
        df["income_group"] = "Unknown"

# prediction (fallback)
if "prediction" not in df.columns:
    df["prediction"] = df["loan_status"]

# ---------- Data for Fairness Tab ----------
# policy: prediction==0 = approved, prediction==1 = rejected
approval_by_income = (
    (df["prediction"] == 0)
    .groupby(df["income_group"], observed=False)
    .mean()
    .reset_index()
    .rename(columns={"prediction": "Approval Rate", "income_group": "Income Group"})
)

default_by_income = (
    df.groupby("income_group", observed=False)["loan_status"]
      .mean()
      .reset_index()
      .rename(columns={"loan_status": "Default Rate", "income_group": "Income Group"})
)

approval_default_df = approval_by_income.merge(default_by_income, on="Income Group", how="inner")

# KPIs
overall_default_rate = df["loan_status"].mean()             # 0..1
overall_approval_rate = (df["prediction"] == 0).mean()
overall_reject_rate   = 1 - overall_approval_rate
_dir_series = approval_default_df.set_index("Income Group")["Approval Rate"]
dir_val = (_dir_series.min() / _dir_series.max()) if _dir_series.max() > 0 else np.nan

# Charts (Fairness tab)
fig_default_by_income = px.bar(
    default_by_income,
    x="Income Group", y="Default Rate",
    color="Default Rate", color_continuous_scale="Reds",
    title="Default Rate by Income Group"
)
fig_default_by_income.update_yaxes(title="Default Rate", range=[0, 1])
fig_default_by_income.update_xaxes(title="Income Group")
fig_default_by_income.update_layout(margin=dict(l=10, r=10, t=40, b=10))

fig_approve_default = px.bar(
    approval_default_df.melt(id_vars="Income Group", var_name="Metric", value_name="Rate"),
    x="Income Group", y="Rate", color="Metric", barmode="group",
    title="Approval Rate vs Default Rate by Income Group",
    color_discrete_map={"Approval Rate": "#5469d4", "Default Rate": "#e45756"}
)
fig_approve_default.update_yaxes(title="Rate", range=[0, 1])
fig_approve_default.update_xaxes(title="Income Group")
fig_approve_default.update_layout(margin=dict(l=10, r=10, t=40, b=10))

# ---------- PDP mock (kept) ----------
pdp_data = pd.DataFrame({
    "Value": list(range(10)),
    "Prediction": [0.2, 0.25, 0.3, 0.33, 0.31, 0.35, 0.36, 0.32, 0.30, 0.28]
})

# Feature Definitions (English)
if feature_defs_csv is not None and {"Feature", "Description"}.issubset(feature_defs_csv.columns):
    df_def = feature_defs_csv[["Feature", "Description"]].copy()
else:
    df_def = pd.DataFrame([
        {"Feature": "loan_amnt",   "Description": "Loan amount requested by the borrower"},
        {"Feature": "term",        "Description": "Loan term (in months)"},
        {"Feature": "int_rate",    "Description": "Interest rate (%)"},
        {"Feature": "installment", "Description": "Monthly installment amount"},
        {"Feature": "grade",       "Description": "Credit grade assigned by LendingClub"},
        {"Feature": "annual_inc",  "Description": "Borrower's annual income"},
        {"Feature": "loan_status", "Description": "Loan repayment status (0=paid, 1=default)"},
        {"Feature": "dti",         "Description": "Debt-to-Income ratio"},
        {"Feature": "revol_bal",   "Description": "Revolving credit balance"},
        {"Feature": "revol_util",  "Description": "Revolving credit utilization (%)"},
    ])

# ========== Dash App ==========
app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
server = app.server

app.layout = dbc.Container([
    html.H2("Lending Club Loan Dashboard", style={'textAlign': 'center', 'marginTop': '20px'}),

    dbc.Tabs([
        # ----- Tab 1: Model Explainability -----
        dbc.Tab(label='Model Explainability', children=[
            html.Br(),
            html.H4("Feature Importance", id="fi-label"),
            dbc.Tooltip("Relative importance of each feature", target="fi-label"),
            dcc.Graph(
                figure=(
                    px.bar(
                        feature_importance.head(10), x='Feature', y='Importance',
                        color='Importance', color_continuous_scale='Blues',
                        title="Top 10 Features by Importance"
                    )
                    if (feature_importance is not None and
                        {"Feature", "Importance"}.issubset(set(feature_importance.columns)))
                    else px.bar(pd.DataFrame({"Feature": [], "Importance": []}), x="Feature", y="Importance")
                )
            ),
            html.H4("Partial Dependence Plot", id="pdp-label"),
            dbc.Tooltip("Relationship between a feature value and predicted outcome", target="pdp-label"),
            dcc.Graph(figure=px.line(pdp_data, x='Value', y='Prediction', title="Partial Dependence Plot")),
        ]),

        # ----- Tab 2: Fairness & Transparency -----
        dbc.Tab(label='Fairness & Transparency', children=[
            html.Br(),

            # KPI cards + tooltips
            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H6("Default Rate", id="tt-default-rate"),
                    html.H3(f"{overall_default_rate*100:.1f}%")
                ]), className="mb-3"), md=4),

                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H6("DIR (Approval)", id="tt-dir"),
                    html.H3("NA" if np.isnan(dir_val) else f"{dir_val:.2f}"),
                    html.Small("≥ 0.80 passes the 80% rule", className="text-muted")
                ]), className="mb-3"), md=4),

                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H6("Reject Rate", id="tt-reject-rate"),
                    html.H3(f"{overall_reject_rate*100:.0f}%")
                ]), className="mb-3"), md=4),
            ]),

            # Tooltips for KPI cards
            dbc.Tooltip(
                "Overall proportion of loans that defaulted (1 = default).",
                target="tt-default-rate", placement="bottom"
            ),
            dbc.Tooltip(
                "Disparate Impact Ratio on approval rate = min(group approval) / max(group approval). "
                "Values ≥ 0.80 typically pass the 80% rule.",
                target="tt-dir", placement="bottom"
            ),
            dbc.Tooltip(
                "Percentage of applications that are rejected by the current policy (prediction==1).",
                target="tt-reject-rate", placement="bottom"
            ),

            html.Hr(),

            # Chart 1 + tooltip
            html.H5("Default Rate by Income Group", id="tt-dr-chart"),
            dbc.Tooltip(
                "Compares default risk across income groups. Higher bars mean higher default rates.",
                target="tt-dr-chart", placement="right"
            ),
            dcc.Graph(figure=fig_default_by_income),

            html.Hr(),

            # Chart 2 + tooltip
            html.H5("Approval vs Default Rate by Income Group", id="tt-adv-chart"),
            dbc.Tooltip(
                "Blue = approval rate (share approved). Red = default rate (share that defaulted). "
                "Use together to assess policy vs. risk.",
                target="tt-adv-chart", placement="right"
            ),
            dcc.Graph(figure=fig_approve_default),
        ]),

        # ----- Tab 3: Feature Definitions -----
        dbc.Tab(label='Feature Definitions', children=[
            html.Br(),
            dbc.Input(id='search-input', placeholder="🔍 Search for a feature...", debounce=True),
            html.Div(id='feature-table', className="mt-3")
        ]),
    ])
], fluid=True)

# ========== Callback: Feature search ==========
@app.callback(
    Output('feature-table', 'children'),
    Input('search-input', 'value')
)
def update_table(search_value):
    if df_def.empty:
        return html.P("No feature definitions available.", style={"color": "orange"})
    filtered = df_def.copy()
    if search_value:
        filtered = filtered[
            filtered['Feature'].astype(str).str.contains(str(search_value), case=False, na=False)
        ]
    if filtered.empty:
        return html.P("No matching features found", style={"color": "red"})
    return dbc.Table.from_dataframe(filtered, striped=True, bordered=True, hover=True)

# ========== Run ==========
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8050)))
