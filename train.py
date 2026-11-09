"""Entrenamiento reproducible del modelo de churn (Telco Customer Churn).

Ejecutar desde la raíz del proyecto:
    python train.py

Genera:
    model.joblib            -> artefacto (pipeline + metadatos) que usa app.py
    reports/cv_results.csv  -> comparación de modelos con validación cruzada
    reports/metrics.json    -> métricas finales en el conjunto de prueba
    reports/importancias.csv -> importancia por permutación (test)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
DATA_PATH = Path("data/telco_churn.csv")
DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
MODEL_PATH = Path("model.joblib")
REPORTS_DIR = Path("reports")
RANDOM_STATE = 42

TARGET = "Churn"
# customerID: identificador sin poder predictivo.
# TotalCharges: casi idéntica a tenure * MonthlyCharges (corr ~0.9996); redundante.
# gender: sin diferencia en la tasa de churn; se excluye por ser atributo sensible.
EXCLUDED = ["customerID", "TotalCharges", "gender"]

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "SeniorCitizen"]
CATEGORICAL_FEATURES = [
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Si la regresión logística queda a menos de esta diferencia de ROC-AUC del mejor
# modelo, se prefiere por interpretabilidad y simplicidad.
AUC_TOLERANCE = 0.01


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    """Lee el CSV local; si no existe, lo descarga y guarda una copia."""
    if not DATA_PATH.exists():
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        pd.read_csv(DATA_URL).to_csv(DATA_PATH, index=False)
    return pd.read_csv(DATA_PATH)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # TotalCharges viene como texto con 11 vacíos (clientes con tenure = 0).
    # Se convierte para dejarlo documentado aunque no entra al modelo.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].str.strip(), errors="coerce")
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype(str).str.strip()
    return df


# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------
def build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, NUMERIC_FEATURES),
        ("cat", categorical_pipe, CATEGORICAL_FEATURES),
    ])


def build_candidates() -> dict[str, Pipeline]:
    # Sin class_weight: así predict_proba queda calibrado (probabilidades
    # interpretables). El desbalance se maneja con el umbral de decisión.
    classifiers = {
        "logistic_regression": LogisticRegression(
            max_iter=2000, random_state=RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=300,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=RANDOM_STATE,
        ),
    }
    return {
        name: Pipeline([("preprocessor", build_preprocessor()), ("classifier", clf)])
        for name, clf in classifiers.items()
    }


def compare_models(candidates, X_train, y_train, cv) -> pd.DataFrame:
    scoring = ["roc_auc", "recall", "precision", "f1", "accuracy"]
    rows = []
    for name, pipe in candidates.items():
        res = cross_validate(
            pipe, X_train, y_train, cv=cv, scoring=scoring, return_train_score=True
        )
        row = {"modelo": name}
        for m in scoring:
            row[f"cv_{m}_mean"] = res[f"test_{m}"].mean()
            row[f"cv_{m}_std"] = res[f"test_{m}"].std()
            row[f"train_{m}_mean"] = res[f"train_{m}"].mean()
        rows.append(row)
        print(
            f"  {name:<24} ROC-AUC CV={row['cv_roc_auc_mean']:.4f} "
            f"(±{row['cv_roc_auc_std']:.4f}) | train={row['train_roc_auc_mean']:.4f} "
            f"| recall CV={row['cv_recall_mean']:.3f}"
        )
    return pd.DataFrame(rows).sort_values("cv_roc_auc_mean", ascending=False)


def select_model(cv_results: pd.DataFrame) -> str:
    best = cv_results.iloc[0]
    logit = cv_results.set_index("modelo").loc["logistic_regression"]
    if best["cv_roc_auc_mean"] - logit["cv_roc_auc_mean"] <= AUC_TOLERANCE:
        return "logistic_regression"
    return best["modelo"]


def choose_threshold(pipe, X_train, y_train, cv) -> float:
    """Umbral que maximiza F1 de la clase churn con predicciones out-of-fold.

    Se calcula solo con entrenamiento (sin tocar test) para evitar fuga.
    """
    oof_proba = cross_val_predict(pipe, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_train, oof_proba)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    best_idx = int(np.nanargmax(f1[:-1]))
    return float(thresholds[best_idx])


def evaluate(y_true, proba, threshold) -> dict:
    pred = (proba >= threshold).astype(int)
    return {
        "threshold": round(threshold, 4),
        "roc_auc": round(roc_auc_score(y_true, proba), 4),
        "recall_churn": round(recall_score(y_true, pred), 4),
        "precision_churn": round(precision_score(y_true, pred), 4),
        "f1_churn": round(f1_score(y_true, pred), 4),
        "f1_weighted": round(f1_score(y_true, pred, average="weighted"), 4),
        "brier": round(brier_score_loss(y_true, proba), 4),
        "prob_media_predicha": round(float(np.mean(proba)), 4),
        "tasa_real": round(float(np.mean(y_true)), 4),
        "accuracy": round(accuracy_score(y_true, pred), 4),
        "confusion_matrix": confusion_matrix(y_true, pred).tolist(),
    }


# ---------------------------------------------------------------------------
# Flujo principal
# ---------------------------------------------------------------------------
def train_model(df: pd.DataFrame) -> dict:
    df = clean_data(df)
    X = df[FEATURES].copy()
    y = (df[TARGET] == "Yes").astype(int)

    churn_by_gender = df.groupby("gender")[TARGET].apply(lambda s: (s == "Yes").mean())
    print("Tasa de churn por género (justifica excluirlo):", churn_by_gender.round(3).to_dict())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train: {X_train.shape} | Test: {X_test.shape} | churn train={y_train.mean():.3f}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    print("\n1) Comparación de modelos (CV estratificada 5 folds, solo train):")
    candidates = build_candidates()
    cv_results = compare_models(candidates, X_train, y_train, cv)

    selected = select_model(cv_results)
    print(f"\n2) Modelo seleccionado: {selected}")

    model = candidates[selected]
    threshold = choose_threshold(model, X_train, y_train, cv)
    print(f"3) Umbral elegido (max F1 out-of-fold): {threshold:.3f}")

    model.fit(X_train, y_train)

    # El conjunto de prueba se usa una sola vez, con el modelo ya elegido.
    proba_test = model.predict_proba(X_test)[:, 1]
    test_metrics = evaluate(y_test, proba_test, threshold)
    test_metrics_05 = evaluate(y_test, proba_test, 0.5)
    print("\n4) Evaluación final en test:")
    print(json.dumps(test_metrics, indent=2))
    print(classification_report(
        y_test, (proba_test >= threshold).astype(int), target_names=["No churn", "Churn"]
    ))

    imp = permutation_importance(
        model, X_test, y_test, scoring="roc_auc", n_repeats=10, random_state=RANDOM_STATE
    )
    importances = (
        pd.DataFrame({
            "variable": FEATURES,
            "importancia_auc": imp.importances_mean,
            "std": imp.importances_std,
        })
        .sort_values("importancia_auc", ascending=False)
        .reset_index(drop=True)
    )
    print("5) Variables más importantes (permutación, test):")
    print(importances.head(6).round(4).to_string(index=False))

    # Reentrenar con todos los datos para el artefacto final de producción.
    model.fit(X, y)

    category_options = {
        col: sorted(df[col].dropna().astype(str).unique().tolist())
        for col in CATEGORICAL_FEATURES
    }
    numeric_ranges = {
        col: {
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "median": float(df[col].median()),
        }
        for col in NUMERIC_FEATURES
    }

    REPORTS_DIR.mkdir(exist_ok=True)
    cv_results.round(4).to_csv(REPORTS_DIR / "cv_results.csv", index=False)
    importances.round(4).to_csv(REPORTS_DIR / "importancias.csv", index=False)
    metrics = {
        "modelo_seleccionado": selected,
        "regla_seleccion": f"Mayor ROC-AUC CV; logística si queda a <= {AUC_TOLERANCE} del mejor",
        "test_umbral_elegido": test_metrics,
        "test_umbral_0_5": test_metrics_05,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "tasa_churn": round(float(y.mean()), 4),
    }
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))

    return {
        "model": model,
        "features": FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "category_options": category_options,
        "numeric_ranges": numeric_ranges,
        "task": "classification",
        "target": TARGET,
        "positive_label": "Churn = Yes",
        "model_name": selected,
        "threshold": threshold,
        "metrics": metrics,
        "top_features": importances.head(5)["variable"].tolist(),
        "sklearn_version": sklearn.__version__,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def main() -> None:
    df = load_data()
    artifact = train_model(df)
    joblib.dump(artifact, MODEL_PATH)
    print(f"\nModelo guardado en {MODEL_PATH}")


if __name__ == "__main__":
    main()
