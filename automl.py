import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, RandomForestRegressor, ExtraTreesRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor


def detect_task(target: pd.Series) -> str:
    """Heuristic only: users should verify the suggested task."""
    if pd.api.types.is_bool_dtype(target) or pd.api.types.is_object_dtype(target) or pd.api.types.is_categorical_dtype(target):
        return "classification"
    if pd.api.types.is_integer_dtype(target) and target.nunique() <= min(20, max(2, int(len(target) * 0.05))):
        return "classification"
    return "regression"


def validate_target(y: pd.Series, task: str) -> None:
    if y.nunique(dropna=True) < 2:
        raise ValueError("The target column must contain at least two distinct values.")
    if task == "regression" and not pd.api.types.is_numeric_dtype(y):
        raise ValueError("Regression requires a numeric target column. Choose classification or select a numeric target.")


def prepare_features(X: pd.DataFrame) -> pd.DataFrame:
    """Reserved helper for future feature checks."""
    return X.copy()


def make_preprocessor(X: pd.DataFrame, estimator) -> Pipeline:
    numeric_columns = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_columns = [c for c in X.columns if c not in numeric_columns]

    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False)),
    ])
    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    transformers = []
    if numeric_columns:
        transformers.append(("numeric", numeric_pipe, numeric_columns))
    if categorical_columns:
        transformers.append(("categorical", categorical_pipe, categorical_columns))
    if not transformers:
        raise ValueError("No usable feature columns were found.")

    preprocessing = ColumnTransformer(transformers=transformers, remainder="drop")
    return Pipeline(steps=[("preprocess", preprocessing), ("model", estimator)])


def get_model_spaces(task: str, seed: int):
    """Return compact, practical model/search-space options for tabular data."""
    if task == "classification":
        return [
            ("Logistic Regression", LogisticRegression(max_iter=2000), {
                "C": np.logspace(-2, 2, 20),
                "class_weight": [None, "balanced"],
            }),
            ("Decision Tree", DecisionTreeClassifier(random_state=seed), {
                "max_depth": [None, 3, 5, 10, 20],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "criterion": ["gini", "entropy", "log_loss"],
            }),
            ("Random Forest", RandomForestClassifier(random_state=seed, n_jobs=1), {
                "n_estimators": [100, 200, 300],
                "max_depth": [None, 5, 10, 20],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2", None],
                "class_weight": [None, "balanced"],
            }),
            ("Extra Trees", ExtraTreesClassifier(random_state=seed, n_jobs=1), {
                "n_estimators": [100, 200, 300],
                "max_depth": [None, 5, 10, 20],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2", None],
                "class_weight": [None, "balanced"],
            }),
            ("Support Vector Machine", SVC(), {
                "C": np.logspace(-2, 2, 20),
                "kernel": ["linear", "rbf"],
                "gamma": ["scale", "auto"],
                "class_weight": [None, "balanced"],
            }),
            ("K-Nearest Neighbors", KNeighborsClassifier(), {
                "n_neighbors": [3, 5, 7, 9, 11],
                "weights": ["uniform", "distance"],
                "p": [1, 2],
            }),
        ]
    if task == "regression":
        return [
            ("Ridge Regression", Ridge(), {
                "alpha": np.logspace(-3, 3, 30),
            }),
            ("Decision Tree", DecisionTreeRegressor(random_state=seed), {
                "max_depth": [None, 3, 5, 10, 20],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            }),
            ("Random Forest", RandomForestRegressor(random_state=seed, n_jobs=1), {
                "n_estimators": [100, 200, 300],
                "max_depth": [None, 5, 10, 20],
                "min_samples_leaf": [1, 2, 4],
                "max_features": [1.0, "sqrt", 0.8],
            }),
            ("Extra Trees", ExtraTreesRegressor(random_state=seed, n_jobs=1), {
                "n_estimators": [100, 200, 300],
                "max_depth": [None, 5, 10, 20],
                "min_samples_leaf": [1, 2, 4],
                "max_features": [1.0, "sqrt", 0.8],
            }),
            ("Support Vector Regression", SVR(), {
                "C": np.logspace(-2, 2, 20),
                "epsilon": [0.01, 0.1, 0.2, 0.5],
                "kernel": ["linear", "rbf"],
                "gamma": ["scale", "auto"],
            }),
            ("K-Nearest Neighbors", KNeighborsRegressor(), {
                "n_neighbors": [3, 5, 7, 9, 11],
                "weights": ["uniform", "distance"],
                "p": [1, 2],
            }),
        ]
    raise ValueError("Task must be 'classification' or 'regression'.")


def evaluate_model(task: str, y_true, y_pred, unused=None) -> dict:
    from sklearn.metrics import (
        accuracy_score, f1_score, mean_absolute_error,
        mean_squared_error, r2_score
    )
    if task == "classification":
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        return {
            "primary_score": f1,
            "details": {
                "Accuracy": round(float(accuracy), 4),
                "Weighted F1": round(float(f1), 4),
            },
        }
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    return {
        "primary_score": -mae,
        "details": {
            "MAE (lower is better)": round(float(mae), 4),
            "RMSE (lower is better)": round(rmse, 4),
            "R²": round(float(r2), 4),
        },
    }
