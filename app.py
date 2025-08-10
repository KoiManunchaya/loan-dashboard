import os
import pandas as pd
import numpy as np
import plotly.express as px

from dash import Dash, dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc

# ========== Load & Prepare Data ==========

# --- Path หลัก (ใช้ไฟล์เล็กตามที่ตกลง)
MAIN_CSV = "data/clean_loan_small.csv"

def safe_read_csv(path, **kwargs):
    if os.path.exists(path):
        return pd.read_csv(path, **kwargs)
    return None

# โหลดข้อมูลหลัก
if not os.path.exists(MAIN_CSV):
    raise FileNotFoundError(
        f"ไม่พบไฟล์ {MAIN_CSV} กรุณาสร้างด้วย create_small_data.py หรือเช็ก path ให้ถูกต้อง"
    )

df = pd.read_csv(MAIN_CSV)
df.columns = df.columns.str.strip().str.lower()

# พยายามโหลดไฟล์อื่นตามของเดิม (ถ้ามี)
feature_importance = safe_read_csv("data/feature_importance.csv")
feature_defs_csv = safe_read_csv("data/feature_definitions_filtered.csv")

# ----- Clean & map loan_status เป็น 0/1 -----
if "loan_status" not in df.columns:
    raise ValueError("❌ Column 'loan_status' not found. กรุณาเช็กคอลัมน์ในไฟล์ข้อมูล")

if df["loan_status"].dtype == object:
    df["loan_status"] = (
        df["loan_status"]
        .astype(str).str.strip().str.lower()
        .map({"fully paid": 0, "paid": 0, "charged off": 1, "default": 1})
        .fillna(0)
        .astype(int)
    )

# income_group (ถ้าไม่มีจะคำนวณจาก annual_inc)
if "income_group" not in df.columns:
    if "annual_inc" in df.columns:
        df["income_group"] = pd.qcut(
            df["annual_inc"], 4, labels=["Low", "Lower-Mid", "Upper-Mid", "High"]
        )
    else:
        df["income_group"] = "Unknown"

# prediction column (ถ้าไม่มีให้ใช้ loan_status เป็นค่าเริ่ม)
if "prediction" not in df.columns:
    df["prediction"] = df["loan_status"]

# Default rate by income group
default_by_income = (
    df.groupby("income_group", observed=False)["loan_status"]
      .mean()
      .reset_index()
      .rename(columns={"loan_status": "Default Rate"})
)

# TPR & FPR by income group
metrics_by_group = []
for group in df["income_group"].unique():
    subset = df[df["income_group"] == group]
    TP = ((subset["loan_status"] == 1) & (subset["prediction"] == 1)).sum()
    FN = ((subset["loan_status"] == 1) & (subset["prediction"] == 0)).sum()
    FP = ((subset["loan_status"] == 0) & (subset["prediction"] == 1)).sum()
    TN = ((subset["loan_status"] == 0) & (subset["prediction"] == 0)).sum()
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
    metrics_by_group.append({"Income Group": str(group), "TPR": TPR, "FPR": FPR})
df_fairness = pd.DataFrame(metrics_by_group)

# Summary metrics (ใส่ค่าเบื้องต้น ถ้ามีของจริงค่อยแทน)
group_metrics = {
    "default_rate": float(df["loan_status"].mean()),
    "reject_rate": 0.30,
    "dir": 1.25
}

# PDP Mock Data (ถ้าไม่มีของจริง)
pdp_data = pd.DataFrame({
    "Value": list(range(10)),
    "Prediction": [0.2, 0.25, 0.3, 0.33, 0.31, 0.35, 0.36, 0.32, 0.3, 0.28]
})

# Feature definitions (ถ้าไม่มีไฟล์ ให้ใช้ชุดสั้น ๆ)
if feature_defs_csv is not None and {"Feature","Description"}.issubset(set(feature_defs_csv.columns)):
    df_def = feature_defs_csv[["Feature","Description"]].copy()
else:
    feature_descriptions = [
        {"Feature": "loan_amnt", "Description": "จำนวนเงินที่ผู้กู้ขอ"},
        {"Feature": "term", "Description": "ระยะเวลาเงินกู้"},
        {"Feature": "int_rate", "Description": "อัตราดอกเบี้ย"},
        {"Feature": "installment", "Description": "ยอดผ่อนรายเดือน"},
        {"Feature": "grade", "Description": "เกรดเครดิต"},
        {"Feature": "annual_inc", "Description": "รายได้ต่อปี"},
        {"Feature": "loan_status", "Description": "สถานะสินเชื่อ"},
        {"Feature": "dti", "Description": "อัตราส่วนหนี้สิน"},
        {"Feature": "revol_bal", "Description": "หนี้หมุนเวียน"},
        {"Feature": "revol_util", "Description": "วงเงินที่ใช้ (%)"},
    ]
    df_def = pd.DataFrame(feature_descriptions)

# ========== (Option) Train and Evaluate Models ==========
# รันแบบเบา ๆ จากคอลัมน์ตัวเลขเท่านั้น (ถ้ามี)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

df_num = df.select_dtypes(include=[np.number]).copy()
df_num = df_num.drop(columns=[c for c in ["prediction"] if c in df_num.columns], errors="ignore")

results = []
try:
    if "loan_status" in df_num.columns and df_num.shape[1] > 1:
        X = df_num.drop(columns=["loan_status"])
        y = df_num["loan_status"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if y.nunique()==2 else None
        )

        models = {
            "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
            "Neural Network": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
        }
        if HAS_XGB:
            models["XGBoost"] = XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=42
            )

        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds, zero_division=0)
            rec = recall_score(y_test, preds, zero_division=0)
            auc = roc_auc_score(y_test, preds) if y.nunique()==2 else np.nan
            results.append({
                "Model": name,
                "Accuracy": round(acc, 3),
                "Precision": round(prec, 3),
                "Recall": round(rec, 3),
                "AUC": round(auc, 3) if not np.isnan(auc) else "NA"
            })
except Exception as e:
    results = [{"Model": "Error", "Accuracy": "-", "Precision": "-", "Recall": "-", "AUC": str(e)}]

df_results = pd.DataFrame(results)

# ========== Dash App (layout แบบเดิม) ==========
app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
server = app.server  # สำคัญสำหรับ Render

app.layout = dbc.Container([
    html.H2("Lending Club Loan Dashboard", style={'textAlign': 'center', 'marginTop': '20px'}),

    dbc.Tabs([
        # ----- Tab 1: Model Explainability -----
        dbc.Tab(label='Model Explainability', children=[
            html.Br(),
            html.H4("Feature Importance", id="fi-label"),
            dbc.Tooltip("แสดงความสำคัญของแต่ละฟีเจอร์", target="fi-label"),
            dcc.Graph(
                figure=(
                    px.bar(
                        feature_importance.head(10), x='Feature', y='Importance',
                        color='Importance', color_continuous_scale='Blues'
                    )
                    if (feature_importance is not None and
                        {"Feature","Importance"}.issubset(set(feature_importance.columns)))
                    else px.bar(pd.DataFrame({"Feature":[],"Importance":[]}), x="Feature", y="Importance")
                )
            ),

            html.H4("Partial Dependence Plot", id="pdp-label"),
            dbc.Tooltip("ความสัมพันธ์ของฟีเจอร์กับผลลัพธ์", target="pdp-label"),
            dcc.Graph(figure=px.line(pdp_data, x='Value', y='Prediction')),

            html.H5("TPR & FPR by Income Group", id="roc-label"),
            dbc.Tooltip("TPR/FPR ของแต่ละกลุ่ม", target="roc-label"),
            dcc.Graph(
                figure=px.bar(
                    df_fairness.melt(id_vars='Income Group', var_name='Metric', value_name='Rate'),
                    x='Income Group', y='Rate', color='Metric', barmode='group'
                )
            ),
        ]),

        # ----- Tab 2: Fairness & Transparency -----
        dbc.Tab(label='Fairness & Transparency', children=[
            html.Br(),
            dbc.Row([
                dbc.Col(html.Div([
                    html.H6("Default Rate"),
                    html.H3(f"{group_metrics['default_rate']*100:.1f}%")
                ]), width=4),
                dbc.Col(html.Div([
                    html.H6("DIR"),
                    html.H3(f"{group_metrics['dir']:.2f}")
                ]), width=4),
                dbc.Col(html.Div([
                    html.H6("Reject Rate"),
                    html.H3(f"{group_metrics['reject_rate']*100:.0f}%")
                ]), width=4),
            ]),
            html.Hr(),
            html.H5("Default Rate by Income Group"),
            dcc.Graph(
                figure=px.bar(
                    default_by_income, x='income_group', y='Default Rate',
                    color='Default Rate', color_continuous_scale='Reds'
                )
            ),
        ]),

        # ----- Tab 3: Model Performance -----
        dbc.Tab(label='Model Performance', children=[
            html.Br(),
            html.H4("เปรียบเทียบผลลัพธ์ของโมเดล", style={'marginTop': '5px'}),
            dash_table.DataTable(
                columns=[{"name": i, "id": i} for i in df_results.columns],
                data=df_results.to_dict('records'),
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'center'},
                style_header={'backgroundColor': 'rgb(30, 30, 30)', 'color': 'white'},
                style_data={'backgroundColor': 'rgb(50, 50, 50)', 'color': 'white'}
            )
        ]),

        # ----- Tab 4: Feature Definitions -----
        dbc.Tab(label='Feature Definitions', children=[
            html.Br(),
            dbc.Input(id='search-input', placeholder="🔍 ค้นหาฟีเจอร์...", debounce=True),
            html.Div(id='feature-table', className="mt-3")
        ]),
    ])
], fluid=True)

# ========== Callback: ค้นหาฟีเจอร์ ==========
@app.callback(
    Output('feature-table', 'children'),
    Input('search-input', 'value')
)
def update_table(search_value):
    filtered = df_def.copy()
    if search_value:
        filtered = filtered[filtered['Feature'].astype(str).str.contains(str(search_value), case=False, na=False)]
    if filtered.empty:
        return html.P("ไม่พบฟีเจอร์ที่ค้นหา", style={"color": "red"})
    return dbc.Table.from_dataframe(filtered, striped=True, bordered=True, hover=True)

# ========== Run ==========
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8050)))
