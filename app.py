"""Customer Churn Intelligence Platform — app Streamlit del modelo de churn (Telco).

Dos modos:
  1. Simulador individual: el diagnóstico se recalcula en vivo al cambiar cualquier dato.
  2. Carga masiva: se sube un CSV o Excel y el modelo se ejecuta sobre todas las filas.

La app NO reentrena: solo carga model.joblib, generado por train.py.
Ejecutar:  streamlit run app.py
"""

from io import BytesIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import f1_score, recall_score, roc_auc_score

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model.joblib"
DATA_PATH = BASE_DIR / "data" / "telco_churn.csv"
EXAMPLE_PATH = BASE_DIR / "data" / "ejemplo_carga.csv"

# ----------------------------------------------------------------- Paleta
BG = "#051923"
NAVY = "#0B2545"
SKY = "#0593C6"      # acento primario: métricas, cliente retenido, factores que bajan el riesgo
PINK = "#D83995"     # acento neón: CTA, alto riesgo, factores que suben el riesgo
WHITE = "#FCFFFF"
SLATE = "#94A3B8"
# Fuentes para Plotly, con respaldo sans-serif si Google Fonts no carga
PLOT_DISPLAY = "Sora, Segoe UI, Arial, sans-serif"
PLOT_BODY = "Manrope, Segoe UI, Arial, sans-serif"

st.set_page_config(
    page_title="Customer Churn Intelligence Platform",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ----------------------------------------------------------------- Textos de la interfaz
LABELS = {
    "tenure": "Antigüedad (meses)",
    "MonthlyCharges": "Cargo mensual (USD)",
    "SeniorCitizen": "Adulto mayor",
    "Partner": "Tiene pareja",
    "Dependents": "Tiene dependientes",
    "PhoneService": "Servicio telefónico",
    "MultipleLines": "Múltiples líneas",
    "InternetService": "Servicio de internet",
    "OnlineSecurity": "Seguridad online",
    "OnlineBackup": "Respaldo online",
    "DeviceProtection": "Protección de dispositivo",
    "TechSupport": "Soporte técnico premium",
    "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming películas",
    "Contract": "Tipo de contrato",
    "PaperlessBilling": "Factura electrónica",
    "PaymentMethod": "Método de pago",
}
# Traducción solo para mostrar: al modelo siempre llega la categoría original del dataset.
VALUE_ES = {
    "Yes": "Sí",
    "No": "No",
    "No internet service": "Sin internet",
    "No phone service": "Sin teléfono",
    "DSL": "DSL",
    "Fiber optic": "Fibra óptica",
    "Month-to-month": "Mes a mes",
    "One year": "1 año",
    "Two year": "2 años",
    "Electronic check": "Cheque electrónico",
    "Mailed check": "Cheque por correo",
    "Bank transfer (automatic)": "Transferencia automática",
    "Credit card (automatic)": "Tarjeta de crédito automática",
}
INTERNET_DEPENDENT = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                      "TechSupport", "StreamingTV", "StreamingMovies"]


def es(value) -> str:
    return VALUE_ES.get(str(value), str(value))


def fmt_num(x: float, decimals: int = 0) -> str:
    """Formato colombiano: punto de miles y coma decimal."""
    s = f"{x:,.{decimals}f}"
    return s.replace(",", "_").replace(".", ",").replace("_", ".")


def fmt_pct(x: float, decimals: int = 1) -> str:
    return fmt_num(100 * x, decimals) + "%"


# ----------------------------------------------------------------- Estilos (CSS inyectado)
CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@500;600;700&family=Manrope:wght@400;500;600;700&display=swap');

:root {{
  --bg: {BG}; --navy: {NAVY}; --sky: {SKY}; --pink: {PINK}; --white: {WHITE}; --slate: {SLATE};
  --glass: rgba(255, 255, 255, 0.05);
  --glass-strong: rgba(255, 255, 255, 0.08);
  --stroke: rgba(255, 255, 255, 0.10);
  --display: 'Sora', 'Segoe UI', system-ui, sans-serif;
  --body: 'Manrope', 'Segoe UI', system-ui, sans-serif;
}}

/* Fondo con dos halos de color muy suaves */
.stApp {{
  background:
    radial-gradient(900px 500px at 8% -10%, rgba(5, 147, 198, 0.18), transparent 60%),
    radial-gradient(800px 480px at 100% 0%, rgba(216, 57, 149, 0.12), transparent 60%),
    var(--bg);
}}
html, body, .stApp, p, label, input, textarea, button, li, td, th, .stMarkdown {{
  font-family: var(--body);
}}
h1, h2, h3, h4 {{ font-family: var(--display) !important; letter-spacing: -0.01em; }}

/* Ocultar elementos por defecto de Streamlit y reducir el padding */
header[data-testid="stHeader"], #MainMenu, footer, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"] {{ display: none !important; }}
.block-container, [data-testid="stMainBlockContainer"] {{
  padding-top: 2.2rem !important; padding-bottom: 3rem !important; max-width: 1280px;
}}

/* ---------- Hero ---------- */
.hero {{ display: grid; gap: 14px; margin-bottom: 26px; max-width: 820px; }}
.badge-top {{
  justify-self: start; display: inline-flex; align-items: center; gap: 8px;
  padding: 6px 14px; border-radius: 999px; font-size: 13px; font-weight: 600; color: var(--white);
  background: rgba(216, 57, 149, 0.12); border: 1px solid rgba(216, 57, 149, 0.45);
}}
.badge-top .dot {{ width: 7px; height: 7px; border-radius: 50%; background: var(--pink);
  box-shadow: 0 0 10px var(--pink); }}
.hero h1 {{
  font-size: clamp(2.1rem, 4vw, 3.3rem) !important; line-height: 1.08 !important; font-weight: 700 !important;
  color: var(--white); margin: 0 !important; padding: 0 !important;
}}
.hero h1 span {{
  background: linear-gradient(90deg, var(--sky), var(--pink));
  -webkit-background-clip: text; background-clip: text; color: transparent;
}}
.hero p {{ color: var(--slate); font-size: 1.05rem; line-height: 1.6; margin: 0; max-width: 68ch; }}

/* ---------- Tarjetas KPI ---------- */
.kpis {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 16px; margin-bottom: 28px; }}
.kpi {{
  background: var(--glass); border: 1px solid var(--stroke); border-radius: 16px;
  padding: 18px 20px; backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);
  display: grid; gap: 6px;
}}
.kpi .label {{ color: var(--slate); font-size: 12px; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; }}
.kpi .value {{ font-family: var(--display); font-size: 1.9rem; font-weight: 600; color: var(--white);
  font-variant-numeric: tabular-nums; line-height: 1.15; }}
.kpi .value.sky {{ color: var(--sky); }}
.kpi .value.pink {{ color: var(--pink); }}
.kpi .sub {{ color: var(--slate); font-size: 13px; }}
@media (max-width: 900px) {{ .kpis {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}

/* ---------- Paneles glass (contenedores con key) ---------- */
.st-key-panel_inputs, .st-key-panel_result, .st-key-panel_factors, .st-key-panel_batch {{
  background: var(--glass); border: 1px solid var(--stroke); border-radius: 16px;
  padding: 22px 22px 18px; backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);
}}
.panel-title {{ font-family: var(--display); font-size: 1.15rem; font-weight: 600; color: var(--white); margin: 0 0 2px; }}
.panel-sub {{ color: var(--slate); font-size: 14px; margin: 0 0 6px; }}
.group-label {{ color: var(--sky); font-size: 11.5px; font-weight: 700; letter-spacing: .1em;
  text-transform: uppercase; margin: 10px 0 2px; }}

/* Widgets */
label, [data-testid="stWidgetLabel"] p {{ color: #CBD5E1 !important; font-weight: 500; }}
div[data-baseweb="select"] > div, [data-testid="stNumberInput"] input {{
  border-radius: 10px !important; border-color: var(--stroke) !important;
}}
[data-testid="stExpander"] details {{
  background: rgba(255,255,255,0.03); border: 1px solid var(--stroke) !important; border-radius: 12px;
}}
[data-testid="stExpander"] summary p {{ color: var(--white); font-weight: 600; }}

/* Pestañas */
.stTabs [data-baseweb="tab-list"] {{ gap: 6px; border-bottom: 1px solid var(--stroke); }}
.stTabs [data-baseweb="tab"] {{ padding: 10px 16px; color: var(--slate); font-weight: 600; }}
.stTabs [aria-selected="true"] {{ color: var(--white) !important; }}

/* Botones: primario = CTA rosa, secundario = contorno sky */
button[data-testid="stBaseButton-primary"], button[data-testid="stBaseButton-primaryFormSubmit"] {{
  background: var(--pink) !important; border: 1px solid var(--pink) !important; color: var(--white) !important;
  border-radius: 12px !important; font-weight: 700 !important; box-shadow: 0 8px 24px -10px var(--pink);
}}
button[data-testid="stBaseButton-secondary"], [data-testid="stDownloadButton"] button {{
  background: transparent !important; border: 1px solid rgba(5, 147, 198, .6) !important;
  color: var(--white) !important; border-radius: 12px !important; font-weight: 600 !important;
}}
button[data-testid="stBaseButton-secondary"]:hover, [data-testid="stDownloadButton"] button:hover {{
  border-color: var(--sky) !important; background: rgba(5, 147, 198, .12) !important;
}}

/* ---------- Diagnóstico ---------- */
.diag {{ display: grid; gap: 12px; align-content: center; height: 100%; }}
.diag-badge {{
  justify-self: start; display: inline-flex; align-items: center; gap: 8px; padding: 8px 16px;
  border-radius: 999px; font-weight: 800; font-size: 13px; letter-spacing: .08em;
}}
.diag-badge.risk {{ color: var(--white); background: var(--pink); box-shadow: 0 0 28px -6px var(--pink); }}
.diag-badge.safe {{ color: var(--white); background: var(--sky); box-shadow: 0 0 28px -6px var(--sky); }}
.diag p {{ color: var(--slate); margin: 0; line-height: 1.55; font-size: 14.5px; }}
.diag b {{ color: var(--white); }}
.diag .ref {{ display: flex; flex-wrap: wrap; gap: 8px; }}
.chip {{ font-size: 12.5px; color: #CBD5E1; padding: 4px 10px; border-radius: 999px;
  border: 1px solid var(--stroke); background: rgba(255,255,255,0.04); font-variant-numeric: tabular-nums; }}

/* ---------- Recomendaciones ---------- */
.section-title {{ font-family: var(--display); font-size: 1.35rem; font-weight: 600; color: var(--white); margin: 30px 0 4px; }}
.section-sub {{ color: var(--slate); font-size: 14.5px; margin: 0 0 14px; max-width: 80ch; }}
.recs {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px; }}
.rec {{
  background: var(--glass); border: 1px solid var(--stroke); border-radius: 16px; padding: 18px 20px;
  display: grid; gap: 10px; align-content: start; backdrop-filter: blur(14px);
}}
.rec .tag {{ justify-self: start; font-size: 11px; font-weight: 700; letter-spacing: .08em; text-transform: uppercase;
  color: var(--sky); border: 1px solid rgba(5,147,198,.5); padding: 3px 9px; border-radius: 999px; }}
.rec h4 {{ font-size: 1.02rem !important; color: var(--white); margin: 0 !important; padding: 0 !important; line-height: 1.35 !important; }}
.rec p {{ color: var(--slate); font-size: 13.5px; margin: 0; line-height: 1.5; }}
.rec .impact {{ display: flex; flex-wrap: wrap; align-items: baseline; gap: 8px; font-variant-numeric: tabular-nums;
  border-top: 1px solid var(--stroke); padding-top: 10px; color: var(--slate); font-size: 13px; }}
.rec .impact b {{ font-family: var(--display); font-size: 1.05rem; color: var(--white); }}
.rec .impact .down {{ color: var(--sky); font-weight: 700; white-space: nowrap; margin-left: auto; }}
.rec.total {{ border-color: rgba(216,57,149,.45); background: rgba(216,57,149,.07); }}
.rec.total .tag {{ color: var(--pink); border-color: rgba(216,57,149,.5); }}

.note {{ color: var(--slate); font-size: 12.5px; margin-top: 10px; }}
.foot {{ color: var(--slate); font-size: 12.5px; border-top: 1px solid var(--stroke); margin-top: 36px; padding-top: 16px; }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------- Modelo y datos
@st.cache_resource
def load_artifact():
    return joblib.load(MODEL_PATH)


@st.cache_data
def portfolio_kpis(_model, features: tuple):
    """KPIs globales calculados con el dataset y el modelo (no son valores fijos)."""
    if not DATA_PATH.exists():
        return None
    df = pd.read_csv(DATA_PATH)
    proba = _model.predict_proba(df[list(features)])[:, 1]
    return {
        "n": len(df),
        "churn_rate": float((df["Churn"] == "Yes").mean()),
        "revenue_risk": float((proba * df["MonthlyCharges"]).sum()),
        "revenue_total": float(df["MonthlyCharges"].sum()),
        "revenue_lost_real": float(df.loc[df["Churn"] == "Yes", "MonthlyCharges"].sum()),
    }


artifact = load_artifact()
model = artifact["model"]
features = artifact["features"]
numeric_features = artifact["numeric_features"]
categorical_features = artifact["categorical_features"]
category_options = artifact["category_options"]
numeric_ranges = artifact["numeric_ranges"]
threshold = artifact["threshold"]
test_metrics = artifact["metrics"]["test_umbral_elegido"]
base_rate = artifact["metrics"]["tasa_churn"]


def predict(row: dict) -> float:
    return float(model.predict_proba(pd.DataFrame([row], columns=features))[0, 1])


def contributions(row: dict) -> pd.Series:
    """Aporte real de cada variable al log-odds (coeficiente × valor transformado).

    Para una regresión logística esta descomposición es exacta, así que no hace falta simular SHAP.
    """
    clf = model.named_steps["classifier"]
    pre = model.named_steps["preprocessor"]
    X_new = pd.DataFrame([row], columns=features)
    values = pre.transform(X_new)[0]
    contrib = pd.Series(values * clf.coef_[0], index=pre.get_feature_names_out())
    contrib = contrib[np.abs(values) > 0]  # categorías activas y numéricas

    def prettify(name: str) -> str:
        raw = name.split("__", 1)[1]
        if raw in numeric_features:
            v = row[raw]
            if raw == "SeniorCitizen":
                return f"{LABELS[raw]}: {'Sí' if v == 1 else 'No'}"
            return f"{LABELS[raw]}: {fmt_num(v, 0 if raw == 'tenure' else 2)}"
        var, _, cat = raw.partition("_")
        return f"{LABELS.get(var, var)}: {es(cat)}"

    return contrib.rename(prettify)


# ----------------------------------------------------------------- Gráficos Plotly
def gauge_figure(proba: float) -> go.Figure:
    color = PINK if proba >= threshold else SKY
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=proba * 100,
        number={"suffix": "%", "valueformat": ".1f", "font": {"size": 46, "color": WHITE, "family": PLOT_DISPLAY}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": SLATE,
                     "tickvals": [0, 25, 50, 75, 100], "tickfont": {"color": SLATE, "size": 11}},
            "bar": {"color": color, "thickness": 0.32},
            "bgcolor": "rgba(255,255,255,0.04)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, threshold * 100], "color": "rgba(5,147,198,0.10)"},
                {"range": [threshold * 100, 100], "color": "rgba(216,57,149,0.10)"},
            ],
            "threshold": {"line": {"color": WHITE, "width": 3}, "thickness": 0.9, "value": threshold * 100},
        },
    ))
    fig.update_layout(
        height=250, margin=dict(l=34, r=40, t=18, b=6), separators=",.",
        paper_bgcolor="rgba(0,0,0,0)", font={"color": WHITE, "family": PLOT_BODY},
    )
    return fig


def factors_figure(contrib: pd.Series, top: int = 8) -> go.Figure:
    top_c = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(top)[::-1]
    colors = [PINK if v > 0 else SKY for v in top_c.values]
    fig = go.Figure(go.Bar(
        x=top_c.values, y=top_c.index, orientation="h",
        marker={"color": colors, "line": {"width": 0}},
        text=[f"{v:+.2f}".replace(".", ",") for v in top_c.values], textposition="outside",
        textfont={"color": WHITE, "size": 12},
        hovertemplate="%{y}<br>Aporte al log-odds: %{x:+.2f}<extra></extra>",
        cliponaxis=False,
    ))
    span = max(0.5, float(np.abs(top_c.values).max()) * 1.5)
    fig.update_layout(
        height=330, margin=dict(l=10, r=40, t=6, b=30), separators=",.",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#CBD5E1", "family": PLOT_BODY, "size": 12.5},
        xaxis={"range": [-span, span], "zeroline": True, "zerolinecolor": "rgba(255,255,255,0.35)",
               "gridcolor": "rgba(255,255,255,0.06)", "tickfont": {"color": SLATE},
               "title": {"text": "← reduce el riesgo · aumenta el riesgo →", "font": {"color": SLATE, "size": 12}}},
        yaxis={"tickfont": {"color": WHITE}, "automargin": True},
        bargap=0.35, showlegend=False,
    )
    return fig


# ----------------------------------------------------------------- Recomendaciones
# Cada acción es una regla de negocio. El impacto se estima con el propio modelo:
# se cambia la variable correspondiente y se vuelve a calcular la probabilidad.
ACTIONS = [
    {
        "when": lambda r: r["Contract"] == "Month-to-month",
        "change": {"Contract": "One year"},
        "tag": "Contrato",
        "title": "Ofrecer 15% de descuento por migrar a contrato anual",
        "text": "El contrato mes a mes es el factor de abandono más fuerte. Un compromiso anual con descuento cambia ese perfil.",
    },
    {
        "when": lambda r: r["InternetService"] != "No" and r["TechSupport"] == "No",
        "change": {"TechSupport": "Yes"},
        "tag": "Soporte",
        "title": "Asignar soporte técnico premium sin costo por 3 meses",
        "text": "Los clientes con internet y sin soporte cancelan casi el triple que quienes sí lo tienen.",
    },
    {
        "when": lambda r: r["InternetService"] != "No" and r["OnlineSecurity"] == "No",
        "change": {"OnlineSecurity": "Yes"},
        "tag": "Servicios",
        "title": "Activar seguridad online incluida en el plan",
        "text": "Sumar servicios de valor aumenta el costo de cambiarse a la competencia.",
    },
    {
        "when": lambda r: r["PaymentMethod"] == "Electronic check",
        "change": {"PaymentMethod": "Credit card (automatic)"},
        "tag": "Facturación",
        "title": "Incentivar el pago automático con tarjeta",
        "text": "Quienes pagan con cheque electrónico tienen la tasa de churn más alta entre los métodos de pago.",
    },
    {
        "when": lambda r: r["Contract"] == "One year",
        "change": {"Contract": "Two year"},
        "tag": "Contrato",
        "title": "Proponer renovación a 2 años con tarifa congelada",
        "text": "Los contratos de dos años tienen la menor tasa de abandono del portafolio.",
    },
]


def recommendations(row: dict, proba: float):
    recs = []
    for a in ACTIONS:
        if a["when"](row):
            new_p = predict({**row, **a["change"]})
            if new_p < proba - 0.005:  # solo acciones que el modelo asocia con menor riesgo
                recs.append({**a, "new_p": new_p})
    recs.sort(key=lambda r: r["new_p"])
    combined = None
    if len(recs) > 1:
        all_changes = {k: v for r in recs for k, v in r["change"].items()}
        # Si hay dos cambios de contrato, se queda el de mayor efecto (el primero tras ordenar).
        all_changes["Contract"] = next((r["change"]["Contract"] for r in recs if "Contract" in r["change"]),
                                       row["Contract"])
        combined = predict({**row, **all_changes})
    return recs, combined


def rec_card(tag, title, text, before, after, extra_class=""):
    delta = (after - before) * 100
    return (
        f'<div class="rec {extra_class}"><span class="tag">{tag}</span><h4>{title}</h4><p>{text}</p>'
        f'<div class="impact">Riesgo <b>{fmt_pct(before)}</b> → <b>{fmt_pct(after)}</b>'
        f'<span class="down">{fmt_num(delta, 1)} pts</span></div></div>'
    )


# ----------------------------------------------------------------- Carga masiva: utilidades
def read_uploaded(file) -> pd.DataFrame:
    name = file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file)
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(file)
    raise ValueError("Formato no soportado. Use CSV o Excel (.xlsx).")


def prepare_batch(df: pd.DataFrame):
    """Valida columnas y tipos del archivo subido. Devuelve (X, avisos)."""
    df = df.rename(columns=lambda c: str(c).strip())
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError("Faltan columnas requeridas: " + ", ".join(missing))

    X = df[features].copy()
    warnings = []

    # SeniorCitizen puede venir como 0/1 o como Yes/No / Sí/No.
    if not pd.api.types.is_numeric_dtype(X["SeniorCitizen"]):
        X["SeniorCitizen"] = (
            X["SeniorCitizen"].astype(str).str.strip().str.lower()
            .map({"1": 1, "0": 0, "yes": 1, "no": 0, "sí": 1, "si": 1})
        )
    for col in numeric_features:
        X[col] = pd.to_numeric(X[col], errors="coerce")
        n_nan = int(X[col].isna().sum())
        if n_nan:
            warnings.append(f"`{col}`: {n_nan} valor(es) vacío(s) o no numérico(s); se imputan con la mediana.")

    for col in categorical_features:
        X[col] = X[col].astype(str).str.strip()
        unknown = sorted(set(X[col]) - set(category_options[col]))
        if unknown:
            warnings.append(f"`{col}`: categorías no vistas en entrenamiento {unknown}; el modelo las ignora.")
    return X, warnings


# =================================================================== 1. Hero
st.markdown(
    """
    <div class="hero">
      <span class="badge-top"><span class="dot"></span>ML Predictive Engine v2.0</span>
      <h1>Customer Churn <span>Intelligence</span> Platform</h1>
      <p>Un modelo de Machine Learning entrenado con 7.043 clientes de telecomunicaciones identifica
      los patrones que anteceden a una cancelación. Simule un cliente y vea en tiempo real su riesgo,
      qué lo explica y qué acciones lo reducen.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# =================================================================== 2. KPIs globales
kpi = portfolio_kpis(model, tuple(features))
if kpi:
    st.markdown(
        f"""
        <div class="kpis">
          <div class="kpi"><span class="label">Clientes analizados</span>
            <span class="value">{fmt_num(kpi['n'])}</span>
            <span class="sub">IBM Telco Customer Churn</span></div>
          <div class="kpi"><span class="label">Tasa de churn</span>
            <span class="value pink">{fmt_pct(kpi['churn_rate'])}</span>
            <span class="sub">Clientes que cancelaron el servicio</span></div>
          <div class="kpi"><span class="label">AUC-ROC del modelo</span>
            <span class="value sky">{fmt_num(test_metrics['roc_auc'], 3)}</span>
            <span class="sub">En test · detecta {fmt_pct(test_metrics['recall_churn'], 0)} de los churn</span></div>
          <div class="kpi"><span class="label">Ingresos en riesgo / mes</span>
            <span class="value">${fmt_num(kpi['revenue_risk'])}</span>
            <span class="sub">{fmt_pct(kpi['revenue_risk'] / kpi['revenue_total'], 0)} de la facturación ·
            real perdido: ${fmt_num(kpi['revenue_lost_real'])}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

tab_sim, tab_batch = st.tabs(["Simulador individual", "Carga masiva (CSV / Excel)"])

# =================================================================== 3. Simulador individual
with tab_sim:
    col_in, col_out = st.columns([1, 1.55], gap="large")

    # ---------------- Entradas del cliente
    with col_in:
        with st.container(key="panel_inputs"):
            st.markdown('<div class="panel-title">Perfil del cliente</div>'
                        '<div class="panel-sub">El diagnóstico se actualiza al cambiar cualquier dato.</div>',
                        unsafe_allow_html=True)
            row = {}
            r_ten = numeric_ranges["tenure"]
            r_mc = numeric_ranges["MonthlyCharges"]

            st.markdown('<div class="group-label">Cuenta</div>', unsafe_allow_html=True)
            row["tenure"] = st.slider(LABELS["tenure"], int(r_ten["min"]), int(r_ten["max"]),
                                      int(r_ten["median"]), key="tenure")
            row["MonthlyCharges"] = st.slider(LABELS["MonthlyCharges"], float(r_mc["min"]), float(r_mc["max"]),
                                              float(r_mc["median"]), step=0.05, format="$%.2f",
                                              key="MonthlyCharges")
            row["Contract"] = st.selectbox(LABELS["Contract"], category_options["Contract"],
                                           format_func=es, key="Contract")
            row["PaymentMethod"] = st.selectbox(LABELS["PaymentMethod"], category_options["PaymentMethod"],
                                                format_func=es, key="PaymentMethod")

            st.markdown('<div class="group-label">Servicios</div>', unsafe_allow_html=True)
            row["InternetService"] = st.selectbox(LABELS["InternetService"], category_options["InternetService"],
                                                  format_func=es, key="InternetService")
            no_internet = row["InternetService"] == "No"
            c1, c2 = st.columns(2)
            for i, f in enumerate(["TechSupport", "OnlineSecurity", "StreamingTV", "StreamingMovies"]):
                with (c1 if i % 2 == 0 else c2):
                    if no_internet:
                        # Sin internet, estos servicios no aplican: se fija la categoría coherente.
                        st.selectbox(LABELS[f], ["No internet service"], format_func=es,
                                     disabled=True, key=f"{f}_off")
                        row[f] = "No internet service"
                    else:
                        opts = [o for o in category_options[f] if o != "No internet service"]
                        row[f] = st.selectbox(LABELS[f], opts, format_func=es, key=f)

            with st.expander("Más datos del cliente"):
                row["SeniorCitizen"] = int(st.toggle(LABELS["SeniorCitizen"], key="SeniorCitizen"))
                d1, d2 = st.columns(2)
                for i, f in enumerate(["Partner", "Dependents", "PhoneService", "MultipleLines",
                                       "OnlineBackup", "DeviceProtection", "PaperlessBilling"]):
                    with (d1 if i % 2 == 0 else d2):
                        if f in INTERNET_DEPENDENT and no_internet:
                            st.selectbox(LABELS[f], ["No internet service"], format_func=es,
                                         disabled=True, key=f"{f}_off")
                            row[f] = "No internet service"
                        else:
                            opts = category_options[f]
                            if f in INTERNET_DEPENDENT:
                                opts = [o for o in opts if o != "No internet service"]
                            row[f] = st.selectbox(LABELS[f], opts, format_func=es, key=f)

    # ---------------- Resultado del modelo
    proba = predict(row)
    is_churn = proba >= threshold
    contrib = contributions(row)

    with col_out:
        with st.container(key="panel_result"):
            st.markdown('<div class="panel-title">Diagnóstico del modelo</div>', unsafe_allow_html=True)
            g1, g2 = st.columns([1.1, 1], gap="medium")
            with g1:
                st.plotly_chart(gauge_figure(proba), width="stretch", config={"displayModeBar": False},
                                key="gauge")
            with g2:
                badge = ('<span class="diag-badge risk">● ALTO RIESGO DE CHURN</span>' if is_churn
                         else '<span class="diag-badge safe">● CLIENTE RETENIDO / NO CHURN</span>')
                msg = ("La probabilidad supera el umbral de decisión. Conviene priorizar a este "
                       "cliente en la campaña de retención." if is_churn else
                       "La probabilidad está por debajo del umbral de decisión. El cliente "
                       "no requiere una acción inmediata.")
                st.markdown(
                    f'<div class="diag">{badge}<p><b>Probabilidad de churn: {fmt_pct(proba)}</b>. {msg}</p>'
                    f'<div class="ref"><span class="chip">Umbral {fmt_pct(threshold, 0)}</span>'
                    f'<span class="chip">Promedio del portafolio {fmt_pct(base_rate, 0)}</span>'
                    f'<span class="chip">{fmt_num(proba / base_rate, 1 if proba >= base_rate else 2)}× el promedio</span></div></div>',
                    unsafe_allow_html=True,
                )

        with st.container(key="panel_factors"):
            st.markdown('<div class="panel-title">¿Qué explica este diagnóstico?</div>'
                        '<div class="panel-sub">Aporte de cada variable a la predicción de este cliente. '
                        'Rosa: aumenta el riesgo · Azul: lo reduce.</div>', unsafe_allow_html=True)
            st.plotly_chart(factors_figure(contrib), width="stretch", config={"displayModeBar": False},
                            key="factors")
            st.markdown('<div class="note">Aportes reales al log-odds del modelo logístico '
                        '(coeficiente × valor estandarizado). Indican asociación, no causalidad.</div>',
                        unsafe_allow_html=True)

    # ---------------- 4. Recomendaciones de retención
    recs, combined = recommendations(row, proba)
    st.markdown('<div class="section-title">Recomendaciones de retención</div>'
                '<div class="section-sub">Acciones sugeridas según el perfil del cliente. El impacto se '
                'estima con el mismo modelo: se cambia la variable y se recalcula la probabilidad.</div>',
                unsafe_allow_html=True)
    if recs:
        cards = [rec_card(r["tag"], r["title"], r["text"], proba, r["new_p"]) for r in recs[:3]]
        if combined is not None:
            cards.append(rec_card("Plan combinado", "Aplicar todas las acciones sugeridas",
                                  f"{len(recs)} acciones aplicadas a la vez sobre este cliente.",
                                  proba, combined, "total"))
        st.markdown(f'<div class="recs">{"".join(cards)}</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="recs"><div class="rec"><span class="tag">Fidelización</span>'
            '<h4>Mantener en el programa de beneficios</h4>'
            '<p>El perfil ya tiene las condiciones que el modelo asocia con baja cancelación '
            '(contrato largo, servicios de soporte, pago automático). Basta con seguimiento periódico.</p>'
            '</div></div>', unsafe_allow_html=True)

# =================================================================== Carga masiva
with tab_batch:
    with st.container(key="panel_batch"):
        st.markdown(
            '<div class="panel-title">Evaluar un archivo de clientes</div>'
            f'<div class="panel-sub">Suba un CSV o Excel con una fila por cliente y las {len(features)} columnas '
            'del modelo (mismos nombres del dataset). Las columnas adicionales, como <code>customerID</code>, '
            'se conservan en el resultado.</div>', unsafe_allow_html=True)
        b1, b2 = st.columns([3, 1])
        with b1:
            with st.expander("Ver columnas requeridas y valores válidos"):
                st.write("**Numéricas:** " + ", ".join(f"`{c}`" for c in numeric_features)
                         + " (`SeniorCitizen` = 0/1)")
                for col in categorical_features:
                    st.write(f"- `{col}`: " + " · ".join(category_options[col]))
        with b2:
            if EXAMPLE_PATH.exists():
                st.download_button("Descargar archivo de ejemplo", EXAMPLE_PATH.read_bytes(),
                                   file_name="ejemplo_carga.csv", mime="text/csv", width="stretch")
        uploaded = st.file_uploader("Archivo de clientes", type=["csv", "xlsx", "xls"])

    if uploaded is not None:
        try:
            raw = read_uploaded(uploaded)
            X_batch, warnings = prepare_batch(raw)
        except Exception as exc:  # mensaje claro al usuario en lugar de un traceback
            st.error(f"No se pudo procesar el archivo: {exc}")
            st.stop()

        for w in warnings:
            st.warning(w)

        proba_b = model.predict_proba(X_batch)[:, 1]
        result = raw.copy()
        result["prob_churn"] = proba_b.round(4)
        result["prediccion"] = np.where(proba_b >= threshold, "Churn", "No churn")
        result = result.sort_values("prob_churn", ascending=False)

        n = len(result)
        n_churn = int((result["prediccion"] == "Churn").sum())
        risk_rev = (float((proba_b * pd.to_numeric(X_batch["MonthlyCharges"], errors="coerce").fillna(0)).sum()))
        st.markdown(
            f"""
            <div class="kpis" style="margin-top:20px">
              <div class="kpi"><span class="label">Clientes evaluados</span>
                <span class="value">{fmt_num(n)}</span>
                <span class="sub">Filas del archivo</span></div>
              <div class="kpi"><span class="label">Predichos como churn</span>
                <span class="value pink">{fmt_num(n_churn)}</span>
                <span class="sub">{fmt_pct(n_churn / n, 0)} del archivo</span></div>
              <div class="kpi"><span class="label">Probabilidad promedio</span>
                <span class="value sky">{fmt_pct(float(proba_b.mean()))}</span>
                <span class="sub">Promedio del portafolio: {fmt_pct(base_rate)}</span></div>
              <div class="kpi"><span class="label">Ingresos en riesgo / mes</span>
                <span class="value">${fmt_num(risk_rev)}</span>
                <span class="sub">Probabilidad × cargo mensual</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Si el archivo trae la respuesta real, se muestra el desempeño sobre ese archivo.
        if "Churn" in raw.columns:
            y_true = raw["Churn"].astype(str).str.strip().str.lower().isin(["yes", "1", "sí", "si"])
            y_pred = proba_b >= threshold
            if y_true.nunique() == 2:
                st.info(
                    "El archivo incluye la columna real `Churn`: "
                    f"ROC-AUC = {fmt_num(roc_auc_score(y_true, proba_b), 3)} · "
                    f"Recall churn = {fmt_num(recall_score(y_true, y_pred), 2)} · "
                    f"F1 churn = {fmt_num(f1_score(y_true, y_pred), 2)}. "
                    "Si estos datos se usaron para entrenar, esta métrica es optimista."
                )

        st.markdown('<div class="section-title">Resultados ordenados por riesgo</div>', unsafe_allow_html=True)
        first_cols = [c for c in ["customerID", "prob_churn", "prediccion"] if c in result.columns]
        st.dataframe(
            result[first_cols + [c for c in result.columns if c not in first_cols]],
            width="stretch",
            hide_index=True,
            column_config={
                "prob_churn": st.column_config.ProgressColumn(
                    "prob_churn", format="%.2f", min_value=0.0, max_value=1.0
                )
            },
        )

        csv_bytes = result.to_csv(index=False).encode("utf-8")
        excel_buffer = BytesIO()
        result.to_excel(excel_buffer, index=False)
        d1, d2 = st.columns(2)
        d1.download_button("Descargar resultados (CSV)", csv_bytes,
                           file_name="predicciones_churn.csv", mime="text/csv", width="stretch")
        d2.download_button("Descargar resultados (Excel)", excel_buffer.getvalue(),
                           file_name="predicciones_churn.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           width="stretch")

st.markdown(
    f'<div class="foot">Modelo: {artifact["model_name"].replace("_", " ").title()} · '
    f'umbral {fmt_num(threshold, 2)} · Dataset IBM Telco Customer Churn · '
    'Machine Learning II · Universidad Externado de Colombia · Nicolás Cárdenas Díaz y Miguel Camargo</div>',
    unsafe_allow_html=True,
)
