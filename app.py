import io
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from automl import (
    detect_task, get_model_spaces, make_preprocessor, evaluate_model,
    validate_target, prepare_features
)

st.set_page_config(page_title="AutoML Studio", page_icon="🧠", layout="wide")
st.title("🧠 AutoML Studio")
st.write("Upload a dataset, compare and tune Scikit-learn models, save the winner, and predict on new rows.")

with st.sidebar:
    st.header("Training settings")
    task_choice = st.selectbox("ML task", ["Auto-detect", "Classification", "Regression"])
    test_size = st.slider("Test set size (%)", min_value=10, max_value=40, value=20, step=5) / 100
    cv_folds = st.selectbox("Cross-validation folds", [3, 5, 10], index=1)
    n_iter = st.slider("Hyperparameter combinations per model", 3, 20, 8)
    random_state = st.number_input("Random seed", min_value=0, max_value=99999, value=42)

uploaded = st.file_uploader("Upload a training CSV file", type=["csv"])

if uploaded is None:
    st.info("Start with a CSV where each row is one example and one column is the value you want to predict.")
    st.subheader("Example datasets to try")
    st.markdown("- **Classification:** a customer churn dataset with a category column as the target.\n- **Regression:** a house-price dataset with a numeric price column as the target.")
    st.stop()

try:
    data = pd.read_csv(uploaded)
except Exception as exc:
    st.error(f"Could not read this CSV: {exc}")
    st.stop()

if data.empty or len(data.columns) < 2:
    st.error("Your CSV needs at least two columns and one data row.")
    st.stop()

st.subheader("Preview")
st.dataframe(data.head(10), use_container_width=True)
st.caption(f"{data.shape[0]:,} rows · {data.shape[1]:,} columns")

target = st.selectbox("Choose the target column (what should the model predict?)", data.columns)
if task_choice == "Auto-detect":
    task = detect_task(data[target])
    st.caption(f"Detected task: **{task.title()}**. Check this carefully; use the selector in the sidebar to override it if needed.")
else:
    task = task_choice.lower()

if task == "classification":
    metric_label = st.selectbox("Choose the model-selection metric", ["F1 weighted", "Accuracy"])
    scoring = {"F1 weighted": "f1_weighted", "Accuracy": "accuracy"}[metric_label]
else:
    metric_label = st.selectbox("Choose the model-selection metric", ["MAE (lower is better)", "RMSE (lower is better)", "R² (higher is better)"])
    scoring = {
        "MAE (lower is better)": "neg_mean_absolute_error",
        "RMSE (lower is better)": "neg_root_mean_squared_error",
        "R² (higher is better)": "r2",
    }[metric_label]

if st.button("🚀 Train, compare and tune models", type="primary", use_container_width=True):
    try:
        cleaned = data.dropna(subset=[target]).copy()
        if len(cleaned) < 20:
            st.error("Please provide at least 20 rows for a more meaningful train/test split and cross-validation.")
            st.stop()
        y = cleaned[target]
        X = cleaned.drop(columns=[target])
        validate_target(y, task)

        from sklearn.model_selection import train_test_split
        stratify = None
        if task == "classification":
            counts = y.value_counts()
            if len(counts) < 2:
                st.error("Classification needs at least two target classes.")
                st.stop()
            if counts.min() >= 2:
                stratify = y

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=int(random_state), stratify=stratify
        )

        model_spaces = get_model_spaces(task, int(random_state))
        results = []
        fitted = {}
        progress = st.progress(0, text="Preparing models...")
        status = st.empty()

        from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, KFold
        if task == "classification":
            min_class_count = int(y_train.value_counts().min())
            actual_cv = min(int(cv_folds), min_class_count)
            if actual_cv < 2:
                st.error("There are too few examples in at least one class. Add more examples or use a different dataset.")
                st.stop()
            cv = StratifiedKFold(n_splits=actual_cv, shuffle=True, random_state=int(random_state))
        else:
            actual_cv = min(int(cv_folds), len(X_train))
            if actual_cv < 2:
                st.error("Not enough training rows for cross-validation.")
                st.stop()
            cv = KFold(n_splits=actual_cv, shuffle=True, random_state=int(random_state))

        for index, (name, estimator, params) in enumerate(model_spaces):
            status.write(f"Training **{name}** ({index + 1}/{len(model_spaces)})…")
            search = RandomizedSearchCV(
                estimator=estimator,
                param_distributions=params,
                n_iter=int(n_iter),
                scoring=scoring,
                cv=cv,
                n_jobs=-1,
                random_state=int(random_state),
                refit=True,
                error_score="raise",
            )
            pipeline = make_preprocessor(X_train, estimator)
            search.estimator = pipeline
            # Tune the estimator inside the full preprocessing pipeline.
            prefixed_params = {f"model__{key}": value for key, value in params.items()}
            search.param_distributions = prefixed_params
            search.fit(X_train, y_train)

            cv_score = float(search.best_score_)
            test_predictions = search.best_estimator_.predict(X_test)
            test_metrics = evaluate_model(task, y_test, test_predictions, y_test)
            if scoring == "neg_mean_absolute_error":
                test_primary_score = test_metrics["details"]["MAE (lower is better)"]
            elif scoring == "neg_root_mean_squared_error":
                test_primary_score = test_metrics["details"]["RMSE (lower is better)"]
            elif scoring == "r2":
                test_primary_score = test_metrics["details"]["R²"]
            elif scoring == "accuracy":
                test_primary_score = test_metrics["details"]["Accuracy"]
            else:
                test_primary_score = test_metrics["details"]["Weighted F1"]
            results.append({
                "Model": name,
                "CV score": cv_score,
                "Test primary score": test_primary_score,
                "Test details": test_metrics["details"],
                "Best parameters": search.best_params_,
            })
            fitted[name] = search
            progress.progress((index + 1) / len(model_spaces), text=f"Completed {name}")

        # Scores from cross-validation are comparable because all candidates use the same split strategy and metric.
        best_name = max(results, key=lambda row: row["CV score"])["Model"]
        best_search = fitted[best_name]
        best_pipeline = best_search.best_estimator_
        best_test = evaluate_model(task, y_test, best_pipeline.predict(X_test), y_test)

        st.session_state["automl_artifact"] = {
            "pipeline": best_pipeline,
            "task": task,
            "target": target,
            "feature_columns": list(X.columns),
            "best_model": best_name,
            "metric": scoring,
            "best_params": best_search.best_params_,
            "cv_score": float(best_search.best_score_),
            "test_metrics": best_test,
        }
        st.session_state["automl_results"] = results
        st.session_state["automl_test_rows"] = len(X_test)
        st.success(f"Training finished. Best model by cross-validation: {best_name}")

    except Exception as exc:
        st.error(f"Training failed: {exc}")
        st.info("Check that the target column and dataset are suitable for the selected task. For debugging, inspect the error details above.")
        st.stop()

if "automl_results" in st.session_state:
    results = st.session_state["automl_results"]
    artifact = st.session_state["automl_artifact"]
    st.divider()
    st.header("Model comparison")
    def human_score(score):
        # Scikit-learn represents loss metrics as negative scores so that higher is always better.
        return -score if artifact["metric"].startswith("neg_") else score

    table = pd.DataFrame([{
        "Model": r["Model"],
        "Cross-validation score": human_score(r["CV score"]),
        "Test primary score": r["Test primary score"],
    } for r in results]).sort_values("Cross-validation score", ascending=(not artifact["metric"].startswith("neg_")))
    # The best model was selected using the original Scikit-learn score; this table is sorted for readability.
    table = table.sort_values("Cross-validation score", ascending=artifact["metric"].startswith("neg_"))
    st.dataframe(table, use_container_width=True, hide_index=True)

    fig, ax = plt.subplots()
    ax.bar(table["Model"], table["Cross-validation score"])
    display_metric = artifact["metric"].replace("neg_mean_absolute_error", "MAE (lower is better)").replace("neg_root_mean_squared_error", "RMSE (lower is better)").replace("f1_weighted", "weighted F1")
    ax.set_ylabel(f"CV score ({display_metric})")
    ax.set_title("Tuned model comparison")
    ax.tick_params(axis="x", labelrotation=25)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    left, right = st.columns(2)
    with left:
        st.metric("Recommended model", artifact["best_model"])
        best_cv_display = human_score(artifact["cv_score"])
        st.metric("Best cross-validation score", f"{best_cv_display:.4f}")
    with right:
        st.write("**Held-out test results**")
        for key, value in artifact["test_metrics"]["details"].items():
            st.write(f"- {key}: **{value}**")
    st.write("**Best hyperparameters**")
    st.json(artifact["best_params"])

    # Package enough metadata to use the trained pipeline later.
    bundle = {
        "pipeline": artifact["pipeline"],
        "task": artifact["task"],
        "target": artifact["target"],
        "feature_columns": artifact["feature_columns"],
        "best_model": artifact["best_model"],
        "metric": artifact["metric"],
        "best_params": artifact["best_params"],
        "cv_score": artifact["cv_score"],
        "test_metrics": artifact["test_metrics"],
    }
    buffer = io.BytesIO()
    joblib.dump(bundle, buffer)
    st.download_button(
        "⬇️ Download best model (.joblib)",
        data=buffer.getvalue(),
        file_name="automl_best_model.joblib",
        mime="application/octet-stream",
        use_container_width=True,
    )

    st.divider()
    st.header("Predict on new data")
    st.write("Upload a CSV containing the same input feature columns used during training. The target column is not required.")
    prediction_file = st.file_uploader("Upload prediction CSV", type=["csv"], key="prediction_csv")
    if prediction_file is not None:
        try:
            new_data = pd.read_csv(prediction_file)
            missing = [col for col in artifact["feature_columns"] if col not in new_data.columns]
            if missing:
                st.error(f"Missing required feature columns: {missing}")
            else:
                new_X = new_data[artifact["feature_columns"]]
                predictions = artifact["pipeline"].predict(new_X)
                output = new_data.copy()
                output["prediction"] = predictions
                st.dataframe(output.head(20), use_container_width=True)
                st.download_button(
                    "Download predictions CSV",
                    data=output.to_csv(index=False).encode("utf-8"),
                    file_name="automl_predictions.csv",
                    mime="text/csv",
                )
        except Exception as exc:
            st.error(f"Could not generate predictions: {exc}")

st.caption("Safety note: only load model files you trust. Joblib files can execute code when loaded.")
