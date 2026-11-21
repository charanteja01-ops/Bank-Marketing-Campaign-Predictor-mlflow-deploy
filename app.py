"""Streamlit app: Bank Marketing Campaign Predictor.

Run from the project folder (the one that contains models/ and reports/):

    streamlit run app.py

It loads the best pipeline saved by bank_mlflow_all_models.py (models/model.pkl).
If that file is missing you can upload a trained pipeline (.pkl) in the sidebar.
"""
import io
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, f1_score

MODEL_PATH = Path("models/model.pkl")
DETAILS_PATH = Path("reports/evaluation_details.json")
METRICS_PATH = Path("reports/metrics.json")
FILL_VALUES_PATH = Path("data/processed/cleaning_values.json")
COMPARISON_PATH = Path("model_comparison.csv")  # written by bank_mlflow_all_models.py

TARGET = "deposit"
POSITIVE_LABEL = "yes"

# Columns the champion model was trained on (before any 'exclude_features').
DEFAULT_FEATURES = [
    "age", "job", "marital", "education", "default", "balance", "housing",
    "loan", "day", "month", "duration", "campaign", "pdays", "previous",
]

OPTIONS = {
    "job": [
        "admin.", "blue-collar", "entrepreneur", "housemaid", "management",
        "retired", "self-employed", "services", "student", "technician",
        "unemployed",
    ],
    "marital": ["married", "single", "divorced"],
    "education": ["primary", "secondary", "tertiary"],
    "default": ["no", "yes"],
    "housing": ["yes", "no"],
    "loan": ["no", "yes"],
    "month": ["jan", "feb", "mar", "apr", "may", "jun",
              "jul", "aug", "sep", "oct", "nov", "dec"],
}

DEFAULT_VALUES = {
    "age": 41, "job": "management", "marital": "married", "education": "secondary",
    "default": "no", "balance": 1500, "housing": "yes", "loan": "no",
    "day": 15, "month": "may", "duration": 300, "campaign": 2,
    "pdays": -1, "previous": 0,
}

PRESETS = {
    "Pick an example profile...": None,
    "Example A: young professional": {
        "age": 34, "job": "management", "marital": "single", "education": "tertiary",
        "default": "no", "balance": 3500, "housing": "no", "loan": "no",
        "day": 15, "month": "mar", "duration": 620, "campaign": 1,
        "pdays": 90, "previous": 2,
    },
    "Example B: retired saver": {
        "age": 67, "job": "retired", "marital": "married", "education": "secondary",
        "default": "no", "balance": 5200, "housing": "no", "loan": "no",
        "day": 10, "month": "oct", "duration": 480, "campaign": 1,
        "pdays": -1, "previous": 0,
    },
    "Example C: blue-collar with loans": {
        "age": 41, "job": "blue-collar", "marital": "married", "education": "primary",
        "default": "no", "balance": 250, "housing": "yes", "loan": "yes",
        "day": 20, "month": "may", "duration": 110, "campaign": 3,
        "pdays": -1, "previous": 0,
    },
}

CSS = """
<style>
.block-container {padding-top: 1.6rem; max-width: 1150px;}
.hero {
    background: linear-gradient(120deg, #1e3a8a 0%, #2563eb 55%, #06b6d4 100%);
    color: #ffffff; padding: 1.6rem 2rem; border-radius: 20px; margin-bottom: 1.2rem;
    box-shadow: 0 10px 30px rgba(37, 99, 235, 0.25);
}
.hero h1 {margin: 0; font-size: 2rem; color: #ffffff;}
.hero p {margin: .35rem 0 0 0; opacity: .92; font-size: 1.02rem; color: #ffffff;}
.card {
    border: 1px solid rgba(128,128,128,.28); border-radius: 18px;
    padding: 1.2rem 1.4rem; background: rgba(128,128,128,.07);
}
.section-title {font-weight: 700; font-size: 1.05rem; margin: .2rem 0 .6rem 0;}
div[data-testid="stMetric"] {
    border: 1px solid rgba(128,128,128,.28); border-radius: 16px;
    padding: .8rem 1rem; background: rgba(128,128,128,.07);
}
.ring-wrap {position: relative; width: 190px; height: 190px; margin: 0 auto;}
.ring {
    width: 100%; height: 100%; border-radius: 50%;
    background: conic-gradient(var(--c) calc(var(--p) * 1%), rgba(128,128,128,.25) 0);
    -webkit-mask: radial-gradient(farthest-side, transparent calc(100% - 18px), #000 calc(100% - 17px));
            mask: radial-gradient(farthest-side, transparent calc(100% - 18px), #000 calc(100% - 17px));
}
.ring-text {
    position: absolute; inset: 0; display: flex; flex-direction: column;
    align-items: center; justify-content: center;
}
.ring-text .pct {font-size: 2.3rem; font-weight: 800; line-height: 1;}
.ring-text .lbl {font-size: .8rem; opacity: .7; margin-top: .25rem;}
.badge {
    display: inline-block; padding: .35rem .9rem; border-radius: 999px;
    font-weight: 700; color: #ffffff; font-size: 1rem;
}
.small-note {font-size: .85rem; opacity: .75;}
</style>
"""


# --------------------------------------------------------------------------
# Helpers (no Streamlit calls, so they are easy to test)
# --------------------------------------------------------------------------
def expected_columns(model):
    """Columns the pipeline was fitted on (respects excluded features)."""
    cols = getattr(model, "feature_names_in_", None)
    return list(cols) if cols is not None else list(DEFAULT_FEATURES)


def score_positive(model, X):
    """Likelihood score of deposit = 'yes' for each row.

    Returns (scores, is_probability). Models without predict_proba (SVM) use
    a sigmoid of the decision function instead.
    """
    classes = list(model.classes_)
    pos = classes.index(POSITIVE_LABEL)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, pos], True
    decision = model.decision_function(X)
    if pos == 0:
        decision = -decision
    return 1.0 / (1.0 + np.exp(-decision)), False


def verdict_for(score, threshold):
    if score >= threshold:
        return "Likely to subscribe", "#16a34a"
    if score >= max(threshold - 0.15, 0.0):
        return "Borderline", "#f59e0b"
    return "Unlikely to subscribe", "#ef4444"


def ring_html(score, color):
    pct = int(round(score * 100))
    return (
        f'<div class="ring-wrap"><div class="ring" style="--p:{pct}; --c:{color}"></div>'
        f'<div class="ring-text"><span class="pct">{pct}%</span>'
        f'<span class="lbl">likelihood score</span></div></div>'
    )


def apply_fill_values(df):
    """Replace 'unknown' in job/education with the training-set mode."""
    if not FILL_VALUES_PATH.exists():
        return df
    with open(FILL_VALUES_PATH, "r", encoding="utf-8") as file:
        fill_values = json.load(file)
    df = df.copy()
    for col, value in fill_values.items():
        if col in df.columns:
            df[col] = df[col].replace("unknown", value)
    return df


def load_json(path):
    if path.exists():
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    return None


def load_comparison():
    """Model comparison table written by bank_mlflow_all_models.py (if present)."""
    if COMPARISON_PATH.exists():
        df = pd.read_csv(COMPARISON_PATH)
        if {"Model", "test_f1"}.issubset(df.columns):
            return df
    return None


def advice_for(label):
    if label == "Likely to subscribe":
        return "Prioritise this customer: schedule a call soon and offer the term deposit."
    if label == "Borderline":
        return "Worth a follow-up if capacity allows; a tailored offer could tip the decision."
    return "Low priority for this campaign; consider other products or a later contact."


# --------------------------------------------------------------------------
# Cached loaders
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model_from_path(path_str, mtime):
    return joblib.load(path_str)


@st.cache_resource(show_spinner=False)
def load_model_from_bytes(data):
    return joblib.load(io.BytesIO(data))


# --------------------------------------------------------------------------
# UI pieces
# --------------------------------------------------------------------------
def init_state():
    for col, value in DEFAULT_VALUES.items():
        st.session_state.setdefault(f"in_{col}", value)


def apply_preset():
    preset = PRESETS.get(st.session_state.get("preset"))
    if preset:
        for col, value in preset.items():
            st.session_state[f"in_{col}"] = value


def render_predict_tab(model, columns, threshold):
    st.selectbox(
        "Quick start", list(PRESETS.keys()), key="preset", on_change=apply_preset
    )

    with st.form("customer_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown('<div class="section-title">👤 Customer profile</div>', unsafe_allow_html=True)
            if "age" in columns:
                st.number_input("Age", min_value=18, max_value=100, step=1, key="in_age")
            if "job" in columns:
                st.selectbox("Job", OPTIONS["job"], key="in_job")
            if "marital" in columns:
                st.selectbox("Marital status", OPTIONS["marital"], key="in_marital")
            if "education" in columns:
                st.selectbox("Education", OPTIONS["education"], key="in_education")

        with c2:
            st.markdown('<div class="section-title">💰 Finances</div>', unsafe_allow_html=True)
            if "balance" in columns:
                st.number_input("Average yearly balance", min_value=-10000,
                                max_value=200000, step=100, key="in_balance")
            if "default" in columns:
                st.selectbox("Credit in default?", OPTIONS["default"], key="in_default")
            if "housing" in columns:
                st.selectbox("Housing loan?", OPTIONS["housing"], key="in_housing")
            if "loan" in columns:
                st.selectbox("Personal loan?", OPTIONS["loan"], key="in_loan")

        with c3:
            st.markdown('<div class="section-title">📞 Campaign contact</div>', unsafe_allow_html=True)
            if "month" in columns:
                st.selectbox("Last contact month", OPTIONS["month"], key="in_month")
            if "day" in columns:
                st.number_input("Last contact day of month", min_value=1, max_value=31,
                                step=1, key="in_day")
            if "duration" in columns:
                st.number_input("Last call duration (seconds)", min_value=0, max_value=6000,
                                step=10, key="in_duration")
            if "campaign" in columns:
                st.number_input("Contacts in this campaign", min_value=1, max_value=60,
                                step=1, key="in_campaign")
            if "pdays" in columns:
                st.number_input("Days since previous campaign (-1 = never)", min_value=-1,
                                max_value=900, step=1, key="in_pdays")
            if "previous" in columns:
                st.number_input("Contacts before this campaign", min_value=0, max_value=60,
                                step=1, key="in_previous")

        submitted = st.form_submit_button("🔮 Predict")

    if "duration" in columns:
        st.caption(
            "ℹ️ Call duration is only known after a call ends, so it helps the model "
            "but cannot be used to decide whom to call in advance."
        )

    if not submitted:
        st.info("Fill in the customer details (or pick an example) and press **Predict**.")
        return

    row = {col: st.session_state[f"in_{col}"] for col in columns}
    X = pd.DataFrame([row], columns=columns)
    scores, is_prob = score_positive(model, X)
    score = float(scores[0])
    label, color = verdict_for(score, threshold)

    st.markdown("---")
    left, right = st.columns([1, 1.4])
    with left:
        st.markdown(ring_html(score, color), unsafe_allow_html=True)
    with right:
        st.markdown(
            f'<span class="badge" style="background:{color}">{label}</span>',
            unsafe_allow_html=True,
        )
        st.markdown(f"#### {advice_for(label)}")
        st.markdown(
            f'<p class="small-note">Decision threshold: {threshold:.0%}. '
            f'{"" if is_prob else "This model has no probability output, so the score is a scaled decision value. "}'
            "Scores are model likelihoods, not calibrated probabilities.</p>",
            unsafe_allow_html=True,
        )
    with st.expander("See the submitted profile"):
        st.dataframe(X.T.rename(columns={0: "value"}).astype(str))


def render_batch_tab(model, columns, threshold):
    st.markdown("Score many customers at once. Upload a CSV with these columns:")
    st.code(", ".join(columns), language="text")

    template = pd.DataFrame([{c: DEFAULT_VALUES[c] for c in columns}])
    st.download_button(
        "⬇️ Download CSV template",
        template.to_csv(index=False).encode("utf-8"),
        file_name="bank_template.csv",
        mime="text/csv",
    )

    upload = st.file_uploader("Upload customers (CSV)", type=["csv"], key="batch_upload")
    if upload is None:
        return

    df = pd.read_csv(upload)
    missing = [c for c in columns if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        return

    X = apply_fill_values(df[columns])
    scores, _ = score_positive(model, X)
    out = df.copy()
    out["likelihood_score"] = np.round(scores, 4)
    out["prediction"] = np.where(scores >= threshold, POSITIVE_LABEL, "no")

    n_yes = int((out["prediction"] == POSITIVE_LABEL).sum())
    m1, m2, m3 = st.columns(3)
    m1.metric("Customers scored", f"{len(out):,}")
    m2.metric("Predicted subscribers", f"{n_yes:,}")
    m3.metric("Share of customers", f"{n_yes / len(out):.1%}")

    if TARGET in out.columns:
        actual = out[TARGET].astype(str)
        m4, m5 = st.columns(2)
        m4.metric("Accuracy vs actual", f"{accuracy_score(actual, out['prediction']):.3f}")
        m5.metric("F1 (yes) vs actual",
                  f"{f1_score(actual, out['prediction'], pos_label=POSITIVE_LABEL):.3f}")

    out = out.sort_values("likelihood_score", ascending=False).reset_index(drop=True)
    st.dataframe(
        out,
        column_config={
            "likelihood_score": st.column_config.ProgressColumn(
                "Likelihood score", min_value=0.0, max_value=1.0, format="%.2f"
            )
        },
    )
    st.download_button(
        "⬇️ Download predictions",
        out.to_csv(index=False).encode("utf-8"),
        file_name="bank_predictions.csv",
        mime="text/csv",
    )


def render_comparison_from_csv(comparison):
    best = comparison.iloc[0]  # file is sorted by cross-validated F1
    st.markdown(f"### 🏆 Best model (by cross-validated F1): **{best['Model']}**")
    st.caption("Test-set results of every model trained by bank_mlflow_all_models.py.")

    cards = [("F1 (yes)", "test_f1"), ("ROC-AUC", "test_roc_auc"),
             ("Accuracy", "test_accuracy"), ("Precision", "test_precision"),
             ("Recall", "test_recall")]
    cards = [(label, col) for label, col in cards if col in comparison.columns]
    for column, (label, col) in zip(st.columns(len(cards)), cards):
        column.metric(label, f"{best[col]:.3f}")

    st.markdown("#### Test F1 by model")
    st.bar_chart(comparison.set_index("Model")[["test_f1"]])

    st.markdown("#### All models")
    st.dataframe(comparison.drop(columns=["best_params"], errors="ignore").round(4))


def render_performance_tab():
    metrics = load_json(METRICS_PATH)
    details = load_json(DETAILS_PATH)

    if not metrics or not details:
        comparison = load_comparison()
        if comparison is None:
            st.info(
                "No results found yet. Run `python bank_mlflow_all_models.py` to train the "
                "models; it creates `model_comparison.csv` and `models/model.pkl`."
            )
            return
        render_comparison_from_csv(comparison)
        return

    st.markdown(f"### 🏆 Champion model: **{details['champion_model']}**")
    st.caption(details.get("selection_explanation", ""))

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("F1 (yes)", f"{metrics['f1']:.3f}")
    c2.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
    c3.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    c4.metric("Precision", f"{metrics['precision']:.3f}")
    c5.metric("Recall", f"{metrics['recall']:.3f}")

    left, right = st.columns(2)
    with left:
        st.markdown("#### Test F1 by model")
        all_models = pd.DataFrame(details["all_models_test_metrics"]).T
        st.bar_chart(all_models[["f1"]])
    with right:
        st.markdown("#### Confusion matrix (test set)")
        cm = details["champion_details"]["confusion_matrix_no_yes"]
        cm_df = pd.DataFrame(
            cm, index=["Actual: no", "Actual: yes"],
            columns=["Predicted: no", "Predicted: yes"],
        )
        st.dataframe(cm_df)
        st.caption("Correct predictions are on the diagonal.")

    st.markdown("#### All models: test metrics")
    st.dataframe(all_models.round(4))

    st.markdown("#### Selection criteria (training data only)")
    st.dataframe(pd.DataFrame(details["selection_comparison"]).T)


def render_about_tab():
    st.markdown(
        """
### About this app
This tool estimates how likely a customer is to subscribe to a **term deposit**
after a bank marketing call, using the champion model from the training pipeline.

**How to use it**
1. **Predict**: enter one customer and get a likelihood score and recommendation.
2. **Batch**: upload a CSV to score a whole call list and download the results.
3. **Model performance**: see how the champion did on held-out test data.

**Good to know**
- Scores are model likelihoods, not calibrated probabilities. Use the threshold in
  the sidebar to decide how selective the campaign should be.
- `duration` (call length) is only known after the call. For choosing whom to call,
  retrain with `exclude_features: ["duration"]` in `params.yaml`.
- Predictions support human decisions; they do not replace them.
        """
    )


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Bank Marketing Predictor", page_icon="🏦", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    init_state()

    st.markdown(
        '<div class="hero"><h1>🏦 Bank Marketing Campaign Predictor</h1>'
        "<p>Find the customers most likely to subscribe to a term deposit.</p></div>",
        unsafe_allow_html=True,
    )

    # ---- Sidebar: model + settings ----
    st.sidebar.header("⚙️ Settings")
    uploaded_model = st.sidebar.file_uploader(
        "Use a different model (.pkl)", type=["pkl", "joblib"],
        help="Upload a fitted scikit-learn pipeline trained on the bank dataset.",
    )

    model = None
    model_source = ""
    if uploaded_model is not None:
        model = load_model_from_bytes(uploaded_model.getvalue())
        model_source = f"uploaded file ({uploaded_model.name})"
    elif MODEL_PATH.exists():
        model = load_model_from_path(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
        model_source = str(MODEL_PATH)

    if model is None:
        st.warning(
            f"No trained model found at `{MODEL_PATH}`. Run `python bank_mlflow_all_models.py` "
            "to create it (and commit `models/model.pkl` if the app is deployed), "
            "or upload a model in the sidebar."
        )
        st.stop()

    columns = expected_columns(model)
    threshold = st.sidebar.slider(
        "Decision threshold", 0.10, 0.90, 0.50, 0.05,
        help="Customers scoring at or above this are flagged as likely subscribers.",
    )

    details = load_json(DETAILS_PATH)
    comparison = load_comparison()
    if details:
        st.sidebar.success(f"Champion: {details['champion_model']}")
    elif comparison is not None:
        st.sidebar.success(f"Best model: {comparison.iloc[0]['Model']}")
    st.sidebar.caption(f"Model source: {model_source}")
    st.sidebar.caption(f"{len(columns)} input features")

    tab_predict, tab_batch, tab_perf, tab_about = st.tabs(
        ["🔮 Predict", "📂 Batch", "📊 Model performance", "ℹ️ About"]
    )
    with tab_predict:
        render_predict_tab(model, columns, threshold)
    with tab_batch:
        render_batch_tab(model, columns, threshold)
    with tab_perf:
        render_performance_tab()
    with tab_about:
        render_about_tab()


if __name__ == "__main__":
    main()
