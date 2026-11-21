import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # or 1

import time
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import optuna
# pip install optuna-integration[mlflow]
from optuna.integration.mlflow import MLflowCallback

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, make_scorer, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Put bank.csv in the data/ folder, or set BANK_DATA_PATH to point elsewhere.
DATA_PATH = Path(os.environ.get(
    "BANK_DATA_PATH",
    r"C:\Users\chara\OneDrive\Desktop\Machine Learning\PROJECTS\Bank Marketing Campaign Prediction\bank.csv"
))
TARGET = "deposit"
POSITIVE_LABEL = "yes"
RANDOM_STATE = 42

# Optuna trials and CV folds per model (SVM / RF / GB are the slowest).
N_TRIALS = {
    "KNN": 30,
    "LogisticRegression": 30,
    "DecisionTree": 40,
    "SVM": 15,
    "GaussianNB": 20,
    "RandomForest": 20,
    "GradientBoosting": 15,
}
CV_FOLDS = {"SVM": 3}  # everything else uses 5
DEFAULT_CV_FOLDS = 5

# Load Dataset
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}. Put bank.csv in the data/ folder "
        "or set the BANK_DATA_PATH environment variable."
    )
data = pd.read_csv(DATA_PATH)

# Data Cleaning
data = data.drop_duplicates()
# contact: not important | poutcome: ~75% 'unknown'
data = data.drop(columns=["contact", "poutcome"])

# Segregate features and Target
X = data.drop(columns=[TARGET])
y = data[TARGET]

# Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# Replace 'unknown' with the most frequent value of the TRAINING data only
X_train, X_test = X_train.copy(), X_test.copy()
for col in ["job", "education"]:
    mode_value = X_train.loc[X_train[col] != "unknown", col].mode()[0]
    X_train[col] = X_train[col].replace("unknown", mode_value)
    X_test[col] = X_test[col].replace("unknown", mode_value)

numeric_cols = X_train.select_dtypes(include=np.number).columns.tolist()
categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

# F1 score of the positive class (deposit = 'yes')
f1_scorer = make_scorer(f1_score, pos_label=POSITIVE_LABEL)


# --------------------------------------------------------------------------
# Pipeline: Preprocessor (impute + Scaler + one-hot) -> Model
# --------------------------------------------------------------------------
def make_scaler(scaler_type):
    return StandardScaler() if scaler_type == "standard" else MinMaxScaler()


def build_pipeline(model, scaler_type):
    preprocessor = ColumnTransformer(
        [
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("Scaler", make_scaler(scaler_type)),
            ]), numeric_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), categorical_cols),
        ]
    )
    return Pipeline([("Preprocessor", preprocessor), ("Model", model)])


def suggest_params(name, trial):
    """Hyperparameters suggested by Optuna for each algorithm."""
    if name == "KNN":
        return {
            "n_neighbors": trial.suggest_int("n_neighbors", 3, 31, step=2),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            "p": trial.suggest_int("p", 1, 3),
        }
    if name == "LogisticRegression":
        return {
            "C": trial.suggest_float("C", 1e-3, 1e2, log=True),
            "solver": trial.suggest_categorical("solver", ["liblinear", "lbfgs"]),
        }
    if name == "DecisionTree":
        return {
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
            "max_depth": trial.suggest_int("max_depth", 2, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical("max_features", [None, "sqrt", "log2"]),
        }
    if name == "SVM":
        kernel = trial.suggest_categorical("kernel", ["linear", "rbf"])
        params = {"kernel": kernel, "C": trial.suggest_float("C", 1e-2, 1e1, log=True)}
        if kernel == "rbf":
            params["gamma"] = trial.suggest_float("gamma", 1e-4, 1e-1, log=True)
        return params
    if name == "GaussianNB":
        return {"var_smoothing": trial.suggest_float("var_smoothing", 1e-11, 1e-7, log=True)}
    if name == "RandomForest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
            "max_depth": trial.suggest_int("max_depth", 5, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        }
    if name == "GradientBoosting":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        }
    raise ValueError(f"Unknown model: {name}")


def build_model(name, params):
    if name == "KNN":
        return KNeighborsClassifier(**params)
    if name == "LogisticRegression":
        return LogisticRegression(**params, class_weight="balanced", max_iter=2000,
                                  random_state=RANDOM_STATE)
    if name == "DecisionTree":
        return DecisionTreeClassifier(**params, class_weight="balanced",
                                      random_state=RANDOM_STATE)
    if name == "SVM":
        return SVC(**params, class_weight="balanced", random_state=RANDOM_STATE)
    if name == "GaussianNB":
        return GaussianNB(**params)
    if name == "RandomForest":
        return RandomForestClassifier(**params, class_weight="balanced",
                                      random_state=RANDOM_STATE, n_jobs=-1)
    if name == "GradientBoosting":
        return GradientBoostingClassifier(**params, random_state=RANDOM_STATE)
    raise ValueError(f"Unknown model: {name}")


def make_objective(name):
    def objective(trial):
        scaler_type = trial.suggest_categorical("scaler_type", ["standard", "minmax"])
        params = suggest_params(name, trial)
        pipeline = build_pipeline(build_model(name, params), scaler_type)

        skf = StratifiedKFold(
            n_splits=CV_FOLDS.get(name, DEFAULT_CV_FOLDS),
            shuffle=True, random_state=RANDOM_STATE,
        )
        return cross_val_score(pipeline, X_train, y_train, scoring=f1_scorer, cv=skf).mean()

    return objective


def positive_scores(pipeline, X_data):
    if hasattr(pipeline, "predict_proba"):
        idx = list(pipeline.classes_).index(POSITIVE_LABEL)
        return pipeline.predict_proba(X_data)[:, idx]
    return pipeline.decision_function(X_data)  # SVC without probabilities


# Numeric ids so models / scalers can be plotted as metrics in MLflow
model_dict = {name: i for i, name in enumerate(N_TRIALS)}
scaler_dict = {"standard": 0, "minmax": 1}

# Set experiment
mlflow.set_experiment("Bank_Marketing_PL_RUNS")

results = {}
fitted_models = {}

# --------------------------------------------------------------------------
# Loop through each algorithm
# --------------------------------------------------------------------------
for model_name, n_trials in N_TRIALS.items():
    print(f"\n--- Optimizing {model_name} ---")

    # One parent run per algorithm; every Optuna trial is a child run of it
    with mlflow.start_run(run_name=model_name):

        mlflow_cb = MLflowCallback(
            tracking_uri=None,              # Auto-detect (str/None)
            metric_name="cv_f1",            # Primary metric (str)
            mlflow_kwargs={                 # **MLflow start_run() kwargs**
                "nested": True              # Child runs under parent
            },
        )

        # Create Optuna study
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        )

        start_search = time.time()
        study.optimize(make_objective(model_name), n_trials=n_trials, callbacks=[mlflow_cb])
        search_time = time.time() - start_search

        print(f"Best CV F1 for {model_name}: {study.best_value:.4f}")
        best_params = dict(study.best_params)
        best_scaler = best_params.pop("scaler_type")
        results[model_name] = {"best_params": study.best_params, "best_cv_f1": study.best_value}

        # Fit the pipeline with the best parameters (this is the timed fit)
        pipeline = build_pipeline(build_model(model_name, best_params), best_scaler)
        start_fit = time.time()
        pipeline.fit(X_train, y_train)
        train_time = time.time() - start_fit
        fitted_models[model_name] = pipeline

        # Evaluate on test data
        start_test = time.time()
        y_pred = pipeline.predict(X_test)
        test_time = time.time() - start_test
        y_pred_train = pipeline.predict(X_train)

        train_acc = accuracy_score(y_train, y_pred_train)
        train_f1 = f1_score(y_train, y_pred_train, pos_label=POSITIVE_LABEL)
        test_acc = accuracy_score(y_test, y_pred)
        test_precision = precision_score(y_test, y_pred, pos_label=POSITIVE_LABEL)
        test_recall = recall_score(y_test, y_pred, pos_label=POSITIVE_LABEL)
        test_f1 = f1_score(y_test, y_pred, pos_label=POSITIVE_LABEL)
        test_auc = roc_auc_score(
            (y_test == POSITIVE_LABEL).astype(int), positive_scores(pipeline, X_test)
        )

        print(f"{model_name} Training Accuracy: {train_acc:.4f}, Testing Accuracy: {test_acc:.4f}")
        print(f"{model_name} Training F1: {train_f1:.4f}, Testing F1: {test_f1:.4f}")
        print(f"{model_name} Fit Time: {train_time:.2f}s, Test Time: {test_time:.2f}s")

        # Save model manually to track model size
        model_path = f"{model_name}_final_model.pkl"
        joblib.dump(pipeline, model_path)
        model_size = os.path.getsize(model_path)

        mlflow.log_param("model_name", model_name)
        mlflow.log_param("scaler_type", best_scaler)
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})

        mlflow.log_metric("model_id", model_dict[model_name])
        mlflow.log_metric("Scaler_id", scaler_dict[best_scaler])
        mlflow.log_metric("best_cv_f1", study.best_value)
        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("train_f1", train_f1)
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("test_precision", test_precision)
        mlflow.log_metric("test_recall", test_recall)
        mlflow.log_metric("test_f1", test_f1)
        mlflow.log_metric("test_roc_auc", test_auc)
        mlflow.log_metric("train_time", train_time)
        mlflow.log_metric("test_time", test_time)
        mlflow.log_metric("search_time", search_time)
        mlflow.log_metric("model_size", model_size)
        mlflow.sklearn.log_model(
            pipeline,
            name=f"{model_name}_bank_model",
            serialization_format="cloudpickle",
        )  # , registered_model_name=f"Bank_{model_name}_Best")
        os.remove(model_path)

    results[model_name].update({
        "train_accuracy": train_acc,
        "train_f1": train_f1,
        "test_accuracy": test_acc,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "test_f1": test_f1,
        "test_roc_auc": test_auc,
        "train_time": train_time,
        "test_time": test_time,
        "search_time": search_time,
        "model_size_bytes": model_size,
    })

# --------------------------------------------------------------------------
# Summary (sorted by CV F1; the test set is not used to pick a model)
# --------------------------------------------------------------------------
print("\n--- Summary ---")
ranked = sorted(results.items(), key=lambda kv: kv[1]["best_cv_f1"], reverse=True)
for model_name, res in ranked:
    print(f"{model_name}: CV F1={res['best_cv_f1']:.4f}, Train F1={res['train_f1']:.4f}, "
          f"Test F1={res['test_f1']:.4f}, Test Acc={res['test_accuracy']:.4f}, "
          f"Fit Time={res['train_time']:.2f}s, Model Size={res['model_size_bytes']} bytes")

summary_df = pd.DataFrame(
    [{"Model": n, **{k: v for k, v in r.items() if k != "best_params"}} for n, r in ranked]
)
summary_df.to_csv("model_comparison.csv", index=False)
best_name = ranked[0][0]
print(f"\nBest model by CV F1: {best_name}")
print("Saved: model_comparison.csv")

# Save the best pipeline so the Streamlit app (app.py) can load it
Path("models").mkdir(exist_ok=True)
joblib.dump(fitted_models[best_name], "models/model.pkl")
print(f"Saved best model ({best_name}) to: models/model.pkl")
