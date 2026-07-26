# engine/ml_templates.py

def get_data_prep_code(target_col: str) -> str:
    return f"""import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Define Features (X) and Target (y)
X = df.drop(columns=['{target_col}'])
y = df['{target_col}']

# 2. Auto-encode categorical data so ML models don't crash
X = pd.get_dummies(X, drop_first=True)
if y.dtype == 'object' or str(y.dtype) == 'category':
    y = y.astype('category').cat.codes

# 3. Split Data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
"""


# ==========================================================
# PARAMETER SPECS
# ==========================================================
# One entry per model. Each param spec describes how to render a control
# for it (slider / select / checkbox) plus its default, so the frontend
# can build widgets without hardcoding anything, and
# get_model_training_code() consumes whatever the user picks.
#
# widget types:
#   "slider"   -> {"min", "max", "default", "step"}
#   "select"   -> {"options": [...], "default"}
#   "checkbox" -> {"default": bool}

MODEL_PARAM_SPECS = {
    "Linear Regression": {
        "fit_intercept": {"widget": "checkbox", "label": "Fit Intercept", "default": True},
    },
    "Logistic Regression": {
        "C": {"widget": "slider", "label": "Regularization Strength (C)", "min": 0.01, "max": 10.0, "default": 1.0, "step": 0.01},
        "max_iter": {"widget": "slider", "label": "Max Iterations", "min": 100, "max": 5000, "default": 1000, "step": 100},
    },
    "Random Forest": {
        "n_estimators": {"widget": "slider", "label": "Number of Trees", "min": 10, "max": 500, "default": 100, "step": 10},
        "max_depth": {"widget": "slider", "label": "Max Depth (0 = unlimited)", "min": 0, "max": 50, "default": 0, "step": 1},
    },
    "XGBoost": {
        "n_estimators": {"widget": "slider", "label": "Number of Trees", "min": 10, "max": 500, "default": 100, "step": 10},
        "max_depth": {"widget": "slider", "label": "Max Depth", "min": 1, "max": 20, "default": 6, "step": 1},
        "learning_rate": {"widget": "slider", "label": "Learning Rate", "min": 0.01, "max": 0.5, "default": 0.1, "step": 0.01},
    },
    "LightGBM": {
        "n_estimators": {"widget": "slider", "label": "Number of Trees", "min": 10, "max": 500, "default": 100, "step": 10},
        "num_leaves": {"widget": "slider", "label": "Num Leaves", "min": 2, "max": 256, "default": 31, "step": 1},
        "learning_rate": {"widget": "slider", "label": "Learning Rate", "min": 0.01, "max": 0.5, "default": 0.1, "step": 0.01},
    },
    "K-Nearest Neighbors": {
        "n_neighbors": {"widget": "slider", "label": "Number of Neighbors (K)", "min": 1, "max": 50, "default": 5, "step": 1},
        "weights": {"widget": "select", "label": "Weighting", "options": ["uniform", "distance"], "default": "uniform"},
    },
    "Naive Bayes": {
        "var_smoothing": {"widget": "select", "label": "Var Smoothing", "options": [1e-9, 1e-8, 1e-7, 1e-6], "default": 1e-9},
    },
}


def get_available_models(is_classification: bool) -> list:
    """Models offered for the given task. Naive Bayes is classification-only -
    GaussianNB doesn't have a standard, equally-simple regression analogue,
    so it's left out rather than faked."""
    models = ["Linear Regression", "Logistic Regression", "Random Forest",
              "XGBoost", "LightGBM", "K-Nearest Neighbors"]
    if is_classification:
        models.remove("Linear Regression")
        models.append("Naive Bayes")
    else:
        models.remove("Logistic Regression")
    return models


def get_model_param_spec(model_type: str) -> dict:
    """Widget spec for a model, so the frontend can render sliders/selects/
    checkboxes without needing to know defaults or ranges itself."""
    return MODEL_PARAM_SPECS.get(model_type, {})


def _resolve_params(model_type: str, params: dict = None) -> dict:
    """Fills in defaults for any params the caller didn't supply."""
    spec = MODEL_PARAM_SPECS.get(model_type, {})
    resolved = {name: cfg["default"] for name, cfg in spec.items()}
    if params:
        resolved.update({k: v for k, v in params.items() if k in resolved})
    return resolved


# ==========================================================
# MODEL TRAINING CODE
# ==========================================================

def get_model_training_code(model_type: str, is_classification: bool, params: dict = None) -> str:
    p = _resolve_params(model_type, params)

    if model_type == "Linear Regression":
        return (
            "from sklearn.linear_model import LinearRegression\n\n"
            f"model = LinearRegression(fit_intercept={p['fit_intercept']})\n"
            "model.fit(X_train, y_train)"
        )

    elif model_type == "Logistic Regression":
        return (
            "from sklearn.linear_model import LogisticRegression\n\n"
            f"model = LogisticRegression(C={p['C']}, max_iter={p['max_iter']})\n"
            "model.fit(X_train, y_train)"
        )

    elif model_type == "Random Forest":
        max_depth_arg = "None" if p["max_depth"] == 0 else p["max_depth"]
        cls = "RandomForestClassifier" if is_classification else "RandomForestRegressor"
        return (
            f"from sklearn.ensemble import {cls}\n\n"
            f"model = {cls}(n_estimators={p['n_estimators']}, max_depth={max_depth_arg}, random_state=42)\n"
            "model.fit(X_train, y_train)"
        )

    elif model_type == "XGBoost":
        cls = "XGBClassifier" if is_classification else "XGBRegressor"
        extra = ", eval_metric='logloss'" if is_classification else ""
        return (
            f"from xgboost import {cls}\n\n"
            f"model = {cls}(n_estimators={p['n_estimators']}, max_depth={p['max_depth']}, "
            f"learning_rate={p['learning_rate']}{extra})\n"
            "model.fit(X_train, y_train)"
        )

    elif model_type == "LightGBM":
        cls = "LGBMClassifier" if is_classification else "LGBMRegressor"
        return (
            f"from lightgbm import {cls}\n\n"
            f"model = {cls}(n_estimators={p['n_estimators']}, num_leaves={p['num_leaves']}, "
            f"learning_rate={p['learning_rate']})\n"
            "model.fit(X_train, y_train)"
        )

    elif model_type == "K-Nearest Neighbors":
        cls = "KNeighborsClassifier" if is_classification else "KNeighborsRegressor"
        return (
            f"from sklearn.neighbors import {cls}\n\n"
            f"model = {cls}(n_neighbors={p['n_neighbors']}, weights='{p['weights']}')\n"
            "model.fit(X_train, y_train)"
        )

    elif model_type == "Naive Bayes":
        return (
            "from sklearn.naive_bayes import GaussianNB\n\n"
            f"model = GaussianNB(var_smoothing={p['var_smoothing']})\n"
            "model.fit(X_train, y_train)"
        )

    raise ValueError(f"Unknown model_type: {model_type}")


def get_evaluation_code(is_classification: bool) -> str:
    if is_classification:
        return """from sklearn.metrics import accuracy_score, classification_report

preds = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, preds))
print("\\nClassification Report:\\n", classification_report(y_test, preds))"""
    else:
        return """from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

preds = model.predict(X_test)
print("R² Score:", r2_score(y_test, preds))
print("MAE:", mean_absolute_error(y_test, preds))
print("RMSE:", np.sqrt(mean_squared_error(y_test, preds)))"""