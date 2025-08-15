# streamlit_tfidf_emotions.py
# Streamlit multi-page app for Amazon Fine Food Reviews
# Embedding: TF-IDF (with optional stop-word removal)
# Classifiers: Decision Tree & SVM (linear)
# Metrics: Precision, Recall, F1 (macro), ROC-AUC (macro OvR)  ← numeric table only
# ROC page: **Per-class** ROC curves + per-class AUCs (no micro/macro curves)
# Pages: Upload → Preprocess → Train → Evaluate → Word Clouds → Predict
#
# How to run:
#   pip install streamlit pandas scikit-learn numpy joblib wordcloud pillow matplotlib
#   streamlit run streamlit_tfidf_emotions.py

import os
import io
import json
import re
import time
from contextlib import redirect_stdout, redirect_stderr
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    auc,  # for per-class AUC directly from each ROC curve
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from joblib import dump, load
from wordcloud import WordCloud
import matplotlib.pyplot as plt

st.set_page_config(page_title="TF-IDF Emotion (Sentiment) Classifier", layout="wide")

ART_DIR = "artifacts"
os.makedirs(ART_DIR, exist_ok=True)

# ---------- Utilities ----------

def clean_text(s: str) -> str:
    s = s or ""
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = re.sub(r"[^A-Za-z0-9' ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

def map_score_to_sentiment(score: int) -> str:
    try:
        s = int(score)
    except Exception:
        return "neutral"
    if s <= 2:
        return "negative"
    if s == 3:
        return "neutral"
    return "positive"

def ensure_df_has_columns(df: pd.DataFrame) -> Tuple[bool, str]:
    required = {"Score"}
    textish = set(df.columns)
    if not required.issubset(df.columns):
        return False, f"CSV must include at least a 'Score' column. Found: {list(df.columns)}"
    if not ({"Text"}.issubset(textish) or {"Summary"}.issubset(textish)):
        return False, "Need a text field: provide 'Text' and/or 'Summary' columns."
    return True, ""

def make_target_and_text(df: pd.DataFrame, use_summary: bool, use_text: bool) -> pd.DataFrame:
    parts = []
    if use_summary and "Summary" in df.columns:
        parts.append(df["Summary"].fillna(""))
    if use_text and "Text" in df.columns:
        parts.append(df["Text"].fillna(""))
    if not parts:
        raise ValueError("Select at least one of Summary/Text as input.")
    combined = (parts[0] if len(parts) == 1 else (parts[0] + " " + parts[1]))
    out = pd.DataFrame({
        "score": df["Score"].astype(int),
        "text": combined.astype(str).map(clean_text)
    })
    out["label"] = out["score"].map(map_score_to_sentiment)
    return out

def split_and_vectorize(
    df_clean: pd.DataFrame,
    test_size=0.2,
    random_state=42,
    max_features=50000,
    min_df=3,
    ngram_range=(1, 2),
    stop_words=None,
):
    X_train, X_test, y_train, y_test = train_test_split(
        df_clean["text"], df_clean["label"], test_size=test_size,
        random_state=random_state, stratify=df_clean["label"]
    )
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        ngram_range=ngram_range,
        stop_words=stop_words,
    )
    Xtr = vectorizer.fit_transform(X_train)
    Xte = vectorizer.transform(X_test)

    le = LabelEncoder()
    ytr = le.fit_transform(y_train)
    yte = le.transform(y_test)
    return Xtr, Xte, ytr, yte, vectorizer, le

def _scores_for_roc(clf, X):
    """Return class scores for ROC curves (probabilities if available, else decision_function)."""
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)
    scores = clf.decision_function(X)
    if scores.ndim == 1:  # binary case shape (n_samples,)
        scores = np.vstack([-scores, scores]).T
    return scores  # raw scores are fine for ROC

def evaluate_model(clf, Xte, yte, class_names):
    y_pred = clf.predict(Xte)
    report = classification_report(yte, y_pred, target_names=class_names, output_dict=True, zero_division=0)

    y_scores = _scores_for_roc(clf, Xte)
    Y_true = label_binarize(yte, classes=list(range(len(class_names))))
    auc_macro_ovr = roc_auc_score(Y_true, y_scores, average="macro", multi_class="ovr")
    cm = confusion_matrix(yte, y_pred, labels=list(range(len(class_names))))
    metrics = {
        "precision_macro": report["macro avg"]["precision"],
        "recall_macro": report["macro avg"]["recall"],
        "f1_macro": report["macro avg"]["f1-score"],
        "roc_auc_macro_ovr": float(auc_macro_ovr),
        "per_class": {cls: {
            "precision": report[cls]["precision"],
            "recall": report[cls]["recall"],
            "f1": report[cls]["f1-score"],
            "support": int(report[cls]["support"]),
        } for cls in class_names},
        "confusion_matrix": cm.tolist()
    }
    return metrics

# ----- Per-class ROC helpers (no micro/macro curves) -----

def _compute_roc_one_vs_rest(y_true_int, y_scores, class_idx, n_classes):
    """Binary ROC for class_idx vs rest; returns fpr, tpr, auc_value."""
    Y = label_binarize(y_true_int, classes=list(range(n_classes)))
    fpr, tpr, _ = roc_curve(Y[:, class_idx], y_scores[:, class_idx])
    return fpr, tpr, auc(fpr, tpr)

def _plot_roc_per_class(model_name, clf, Xte, yte, class_names, selected="All classes"):
    """Plot per-class ROC curves (or a single class) with AUC labels."""
    scores = _scores_for_roc(clf, Xte)
    n_classes = len(class_names)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)

    # Which classes to plot?
    indices = range(n_classes) if selected == "All classes" else [class_names.index(selected)]

    for i in indices:
        fpr, tpr, auc_i = _compute_roc_one_vs_rest(yte, scores, i, n_classes)
        ax.plot(fpr, tpr, label=f"{class_names[i]} (AUC={auc_i:.3f})")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves — {model_name}")
    ax.legend(loc="lower right", fontsize="small")
    st.pyplot(fig, use_container_width=True)

def save_artifacts(vectorizer, le, dt, svm, metrics_dt, metrics_svm, class_names):
    dump(vectorizer, os.path.join(ART_DIR, "tfidf_vectorizer.joblib"))
    dump(le, os.path.join(ART_DIR, "label_encoder.joblib"))
    dump(dt, os.path.join(ART_DIR, "decision_tree.joblib"))
    dump(svm, os.path.join(ART_DIR, "svm_linear.joblib"))
    with open(os.path.join(ART_DIR, "metrics_dt.json"), "w") as f:
        json.dump(metrics_dt, f, indent=2)
    with open(os.path.join(ART_DIR, "metrics_svm.json"), "w") as f:
        json.dump(metrics_svm, f, indent=2)
    with open(os.path.join(ART_DIR, "class_names.json"), "w") as f:
        json.dump(class_names, f, indent=2)

def load_artifacts():
    vec = load(os.path.join(ART_DIR, "tfidf_vectorizer.joblib"))
    le = load(os.path.join(ART_DIR, "label_encoder.joblib"))
    dt = load(os.path.join(ART_DIR, "decision_tree.joblib"))
    svm = load(os.path.join(ART_DIR, "svm_linear.joblib"))
    with open(os.path.join(ART_DIR, "class_names.json")) as f:
        class_names = json.load(f)
    metrics_dt = json.load(open(os.path.join(ART_DIR, "metrics_dt.json"))) if os.path.exists(os.path.join(ART_DIR, "metrics_dt.json")) else None
    metrics_svm = json.load(open(os.path.join(ART_DIR, "metrics_svm.json"))) if os.path.exists(os.path.join(ART_DIR, "metrics_svm.json")) else None
    return vec, le, dt, svm, class_names, metrics_dt, metrics_svm

# ---------- State helpers ----------

def reset_training_state():
    for key in ["df_raw", "df_clean", "vectorizer", "label_encoder", "Xtr", "Xte", "ytr", "yte",
                "dt_model", "svm_model", "class_names", "metrics_dt", "metrics_svm",
                "stop_words_choice"]:
        if key in st.session_state:
            del st.session_state[key]

# ---------- Pages ----------

def page_upload():
    st.header("📥 Data Upload")
    st.write("Upload *Amazon Fine Food Reviews* CSV (from Kaggle). At minimum, it should include Score and Text and/or Summary.")

    file = st.file_uploader("Upload Reviews.csv", type=["csv"])
    if file is not None:
        try:
            df = pd.read_csv(file)
        except Exception:
            file.seek(0)
            df = pd.read_csv(io.BytesIO(file.read()), encoding_errors="ignore")
        ok, msg = ensure_df_has_columns(df)
        if not ok:
            st.error(msg)
            return

        st.session_state.df_raw = df
        st.success(f"Loaded {len(df):,} rows.")
        st.dataframe(df.head(20))
        st.info("Next: go to *Preprocessing*.")

def page_preprocess():
    st.header("🧼 Preprocessing")
    if "df_raw" not in st.session_state:
        st.warning("Upload a dataset first in *Data Upload*.")
        return

    df = st.session_state.df_raw.copy()
    use_sum = st.checkbox("Use Summary", value=("Summary" in df.columns))
    use_txt = st.checkbox("Use Text", value=("Text" in df.columns))

    st.markdown("**TF-IDF settings**")
    test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
    max_features = st.select_slider("TF-IDF max features", options=[10000, 20000, 30000, 40000, 50000, 75000, 100000], value=50000)
    ngram = st.selectbox("n-gram range", options=["(1,1)", "(1,2)", "(1,3)"], index=1)
    min_df = st.number_input("min_df (ignore terms with document frequency < min_df)", min_value=1, value=3, step=1)

    st.markdown("**Stop words**")
    remove_sw = st.checkbox("Remove English stop words", value=True)
    extra_sw = st.text_input("Extra stop words (comma-separated, optional)", "")
    stop_words = None
    if remove_sw and extra_sw.strip():
        sw = set(ENGLISH_STOP_WORDS)
        sw.update(w.strip().lower() for w in extra_sw.split(",") if w.strip())
        stop_words = sw
    elif remove_sw:
        stop_words = "english"
    elif extra_sw.strip():
        stop_words = {w.strip().lower() for w in extra_sw.split(",") if w.strip()}

    if st.button("Run Preprocessing"):
        if not (use_sum or use_txt):
            st.error("Select at least one of Summary/Text.")
            return
        df_clean = make_target_and_text(df, use_sum, use_txt)
        st.session_state.df_clean = df_clean

        ngram_map = {"(1,1)": (1, 1), "(1,2)": (1, 2), "(1,3)": (1, 3)}
        Xtr, Xte, ytr, yte, vectorizer, le = split_and_vectorize(
            df_clean,
            test_size=test_size,
            max_features=max_features,
            min_df=int(min_df),
            ngram_range=ngram_map[ngram],
            stop_words=stop_words,
        )

        st.session_state.vectorizer = vectorizer
        st.session_state.label_encoder = le
        st.session_state.Xtr, st.session_state.Xte = Xtr, Xte
        st.session_state.ytr, st.session_state.yte = ytr, yte
        st.session_state.class_names = list(le.classes_)
        st.session_state.stop_words_choice = (
            "english" if stop_words == "english"
            else ("custom" if isinstance(stop_words, (set, frozenset)) else "none")
        )

        st.success("Preprocessing complete.")
        c1, c2 = st.columns(2)
        with c1:
            st.write("Class distribution:")
            st.dataframe(df_clean["label"].value_counts().rename_axis("label").to_frame("count"))
            st.caption(f"Stop words: **{st.session_state.stop_words_choice}**")
        with c2:
            st.write("Sample cleaned text:")
            st.write(df_clean["text"].head(10).tolist())

def page_train():
    st.header("🏋️ Model Training")
    missing = [k for k in ["Xtr", "Xte", "ytr", "yte", "label_encoder", "vectorizer", "class_names"] if k not in st.session_state]
    if missing:
        st.warning("Please finish *Preprocessing* first.")
        return

    cw = st.checkbox("Use class_weight='balanced'", value=True)
    max_depth = st.number_input("Decision Tree: max_depth (0 for None)", min_value=0, value=0, step=1)
    svm_c = st.number_input("SVM (linear): C", min_value=0.01, value=1.0, step=0.1)

    if st.button("Train Models"):
        Xtr, Xte = st.session_state.Xtr, st.session_state.Xte
        ytr, yte = st.session_state.ytr, st.session_state.yte
        class_names = st.session_state.class_names

        prog = st.progress(0, text="Initializing…")
        status_area = st.status("Training models…", expanded=True)
        step, total = 0, 6

        def bump(msg, inc=1, force_pct=None):
            nonlocal step
            step += inc
            pct = int(100 * step / total) if force_pct is None else force_pct
            prog.progress(min(pct, 100), text=msg)
            with status_area:
                st.write(msg)

        try:
            bump("Building Decision Tree (step 1/6)…", force_pct=5)
            dt = DecisionTreeClassifier(
                random_state=42,
                class_weight=("balanced" if cw else None),
                max_depth=(None if max_depth == 0 else int(max_depth))
            )

            t0 = time.perf_counter()
            bump("Fitting Decision Tree (step 2/6)…", force_pct=15)
            dt.fit(Xtr, ytr)
            bump(f"Decision Tree fit complete in {time.perf_counter() - t0:.2f}s (step 2/6 done).", force_pct=35)
            st.session_state.dt_model = dt

            bump("Evaluating Decision Tree (step 3/6)…", force_pct=50)
            m_dt = evaluate_model(dt, Xte, yte, class_names)
            st.session_state.metrics_dt = m_dt

            bump("Building SVM (linear) (step 4/6)…", force_pct=55)
            svm = SVC(
                kernel="linear",
                probability=True,
                class_weight=("balanced" if cw else None),
                random_state=42,
                C=float(svm_c),
                verbose=True
            )

            bump("Fitting SVM (this can take a while) (step 5/6)…", force_pct=70)
            log_buf = io.StringIO()
            t1 = time.perf_counter()
            with redirect_stdout(log_buf), redirect_stderr(log_buf):
                svm.fit(Xtr, ytr)
            fit_secs = time.perf_counter() - t1
            bump(f"SVM fit complete in {fit_secs:.2f}s (step 5/6 done).", force_pct=85)
            st.session_state.svm_model = svm
            logs = log_buf.getvalue().strip()
            if logs:
                with status_area:
                    with st.expander("SVM training logs", expanded=False):
                        st.code(logs or "No logs", language="text")

            bump("Evaluating SVM (step 6/6)…", force_pct=95)
            m_svm = evaluate_model(svm, Xte, yte, class_names)
            st.session_state.metrics_svm = m_svm

            prog.progress(100, text="✅ Training finished")
            status_area.update(label="Training complete", state="complete")
            st.success("Training complete. See *Models Evaluation* to compare metrics.")

            if st.button("Save Artifacts to ./artifacts"):
                save_artifacts(st.session_state.vectorizer, st.session_state.label_encoder, dt, svm,
                               st.session_state.metrics_dt, st.session_state.metrics_svm, class_names)
                st.success("Artifacts saved.")

        except Exception as e:
            status_area.update(label="Training failed", state="error")
            st.error(f"Training error: {e}")

def metrics_table(metrics: Dict, title: str):
    st.subheader(title)
    macro = pd.DataFrame([{
        "precision_macro": metrics["precision_macro"],
        "recall_macro": metrics["recall_macro"],
        "f1_macro": metrics["f1_macro"],
        "roc_auc_macro_ovr": metrics["roc_auc_macro_ovr"],  # numeric summary remains
    }])
    st.dataframe(macro.style.format("{:.3f}"))
    st.write("Per-class metrics:")
    rows = []
    for cls, d in metrics["per_class"].items():
        rows.append({"class": cls, **d})
    st.dataframe(pd.DataFrame(rows).style.format({"precision": "{:.3f}", "recall": "{:.3f}", "f1": "{:.3f}"}))
    st.write("Confusion matrix [rows=true, cols=pred]:")
    st.dataframe(pd.DataFrame(metrics["confusion_matrix"], index=list(metrics["per_class"].keys()),
                              columns=list(metrics["per_class"].keys())))

def page_evaluate():
    st.header("📊 Models Evaluation")
    if ("metrics_dt" not in st.session_state) or ("metrics_svm" not in st.session_state):
        st.info("You can also load saved artifacts from disk.")
        if st.button("Load artifacts from ./artifacts"):
            try:
                vec, le, dt, svm, class_names, mdt, msvm = load_artifacts()
                st.session_state.vectorizer = vec
                st.session_state.label_encoder = le
                st.session_state.dt_model = dt
                st.session_state.svm_model = svm
                st.session_state.class_names = class_names
                if mdt and msvm:
                    st.session_state.metrics_dt = mdt
                    st.session_state.metrics_svm = msvm
                st.success("Artifacts loaded.")
            except Exception as e:
                st.error(f"Failed to load artifacts: {e}")
                return

    if ("metrics_dt" not in st.session_state) or ("metrics_svm" not in st.session_state):
        st.warning("Train models first on the *Model Training* page.")
        return

    c1, c2 = st.columns(2)
    with c1:
        metrics_table(st.session_state.metrics_dt, "Decision Tree")
    with c2:
        metrics_table(st.session_state.metrics_svm, "SVM (linear)")

    st.markdown("---")
    st.subheader("Side-by-side (macro metrics)")
    comp = pd.DataFrame([
        {"model": "Decision Tree", **{k: v for k, v in st.session_state.metrics_dt.items() if "macro" in k}},
        {"model": "SVM (linear)", **{k: v for k, v in st.session_state.metrics_svm.items() if "macro" in k}},
    ])
    st.dataframe(comp.style.format({
        "precision_macro": "{:.3f}",
        "recall_macro": "{:.3f}",
        "f1_macro": "{:.3f}",
        "roc_auc_macro_ovr": "{:.3f}"
    }))

    # ---------- ROC curves (per-class only) ----------
    st.markdown("---")
    st.subheader("ROC Curves")
    model_choice = st.radio("Select model", ["Decision Tree", "SVM (linear)"], horizontal=True)
    class_choice = st.selectbox("Show ROC for", ["All classes"] + st.session_state.class_names, index=0)

    if model_choice == "Decision Tree":
        _plot_roc_per_class("Decision Tree",
                            st.session_state.dt_model,
                            st.session_state.Xte,
                            st.session_state.yte,
                            st.session_state.class_names,
                            selected=class_choice)
    else:
        _plot_roc_per_class("SVM (linear)",
                            st.session_state.svm_model,
                            st.session_state.Xte,
                            st.session_state.yte,
                            st.session_state.class_names,
                            selected=class_choice)

# ---------- Word Clouds Page ----------

def page_wordclouds():
    st.header("☁️ Word Clouds (by Sentiment)")
    if ("Xtr" not in st.session_state) or ("ytr" not in st.session_state) or ("vectorizer" not in st.session_state) or ("class_names" not in st.session_state):
        st.warning("Please finish *Preprocessing* first (we need TF-IDF and the training split).")
        return

    Xtr = st.session_state.Xtr
    ytr = st.session_state.ytr
    class_names = st.session_state.class_names
    vec: TfidfVectorizer = st.session_state.vectorizer

    st.write("Word clouds are computed from **TF-IDF mean weights per class** on the *training* split. This is more informative than raw frequency.")
    top_k = st.slider("Top words per class", 50, 400, 200, 25)
    width = st.number_input("Image width", min_value=400, value=1200, step=100)
    height = st.number_input("Image height", min_value=300, value=600, step=50)
    show_bigrams = st.checkbox("Include bigrams (replace spaces with underscores in display)", value=True)

    feature_names = np.array(vec.get_feature_names_out())

    cols = st.columns(min(3, len(class_names)))  # up to 3 per row
    for i, cls in enumerate(class_names):
        mask = (ytr == i)
        if mask.sum() == 0:
            st.info(f"No training samples for class '{cls}'.")
            continue
        mean_vec = Xtr[mask].mean(axis=0).A1  # (n_features,)
        top_idx = np.argsort(mean_vec)[-top_k:]
        freqs = {}
        for j in top_idx:
            token = feature_names[j]
            if show_bigrams:
                token = token.replace(" ", "_")
            w = float(mean_vec[j])
            if w > 0:
                freqs[token] = w

        if not freqs:
            st.info(f"Not enough signal to build a cloud for '{cls}'.")
            continue

        wc = WordCloud(width=int(width), height=int(height), background_color="white")
        img = wc.generate_from_frequencies(freqs).to_image()
        with cols[i % len(cols)]:
            st.image(img, caption=f"{cls} — top {len(freqs)} TF-IDF terms", use_container_width=True)

# ---------- Prediction Page ----------

def page_predict():
    st.header("🔮 Prediction")

    if ("dt_model" not in st.session_state) or ("svm_model" not in st.session_state) or ("vectorizer" not in st.session_state) or ("label_encoder" not in st.session_state):
        st.info("Load trained artifacts from disk to enable prediction.")
        if st.button("Load artifacts from ./artifacts"):
            try:
                vec, le, dt, svm, class_names, _, _ = load_artifacts()
                st.session_state.vectorizer = vec
                st.session_state.label_encoder = le
                st.session_state.dt_model = dt
                st.session_state.svm_model = svm
                st.session_state.class_names = class_names
                st.success("Artifacts loaded. You can now predict.")
            except Exception as e:
                st.error(f"Failed to load artifacts: {e}")
                return

    if ("dt_model" not in st.session_state) or ("svm_model" not in st.session_state) or ("vectorizer" not in st.session_state) or ("label_encoder" not in st.session_state):
        return

    class_names = st.session_state.class_names
    vec = st.session_state.vectorizer
    dt = st.session_state.dt_model
    svm = st.session_state.svm_model

    st.subheader("📝 Single Text")
    txt = st.text_area(
        "Enter a product review (Summary + Text):",
        "I absolutely loved this product! Tastes amazing and delivery was quick."
    )
    if st.button("Predict"):
        X = vec.transform([clean_text(txt)])
        proba_dt = dt.predict_proba(X)[0]
        pred_dt = class_names[int(np.argmax(proba_dt))]
        proba_svm = svm.predict_proba(X)[0]
        pred_svm = class_names[int(np.argmax(proba_svm))]

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Decision Tree", pred_dt)
            st.dataframe(
                pd.DataFrame({"emotion": class_names, "probability": proba_dt})
                  .sort_values("probability", ascending=False)
                  .reset_index(drop=True)
            )
        with c2:
            st.metric("SVM (linear)", pred_svm)
            st.dataframe(
                pd.DataFrame({"emotion": class_names, "probability": proba_svm})
                  .sort_values("probability", ascending=False)
                  .reset_index(drop=True)
            )

    st.markdown("---")

    st.subheader("📦 Batch Prediction (CSV)")
    st.caption("Upload a CSV with *Text* and/or *Summary* columns. We'll clean, vectorize, and predict for each row.")
    up = st.file_uploader("Upload CSV for batch prediction", type=["csv"])

    colA, colB = st.columns([1, 1])
    with colA:
        use_summary = st.checkbox("Use Summary", value=True)
        use_text = st.checkbox("Use Text", value=True)
    with colB:
        models_to_use = st.multiselect(
            "Models",
            ["Decision Tree", "SVM (linear)"],
            default=["Decision Tree", "SVM (linear)"]
        )
        include_probs = st.checkbox("Include per-class probabilities", value=True)

    run_batch = st.button("🚀 Batch Predict", use_container_width=True)

    if run_batch:
        if up is None:
            st.error("Please upload a CSV first.")
            return
        if not (use_summary or use_text):
            st.error("Select at least one text source (Summary and/or Text).")
            return
        if not models_to_use:
            st.error("Select at least one model to run.")
            return

        try:
            df_in = pd.read_csv(up)
        except Exception:
            up.seek(0)
            df_in = pd.read_csv(io.BytesIO(up.read()), encoding_errors="ignore")

        if ("Summary" not in df_in.columns) and ("Text" not in df_in.columns):
            st.error("CSV must include at least one of: 'Summary', 'Text'.")
            return

        n = len(df_in)
        s_sum = df_in["Summary"].fillna("") if (use_summary and "Summary" in df_in.columns) else pd.Series([""] * n)
        s_txt = df_in["Text"].fillna("") if (use_text and "Text" in df_in.columns) else pd.Series([""] * n)
        combined = (s_sum.astype(str) + " " + s_txt.astype(str)).str.strip()

        if combined.empty:
            st.error("No rows to predict after combining selected columns.")
            return

        with st.spinner("Vectorizing…"):
            texts = combined.astype(str).map(clean_text).tolist()
            Xbatch = vec.transform(texts)

        results = pd.DataFrame(index=df_in.index)
        keep_cols = [c for c in ["Summary", "Text"] if c in df_in.columns]
        results[keep_cols] = df_in[keep_cols]

        if "Decision Tree" in models_to_use:
            with st.spinner("Predicting with Decision Tree…"):
                proba_dt = dt.predict_proba(Xbatch)
                pred_dt_idx = np.argmax(proba_dt, axis=1)
                results["pred_dt"] = [class_names[i] for i in pred_dt_idx]
                if include_probs:
                    for j, cls in enumerate(class_names):
                        results[f"dt_prob_{cls}"] = proba_dt[:, j]

        if "SVM (linear)" in models_to_use:
            with st.spinner("Predicting with SVM (linear)…"):
                proba_svm = svm.predict_proba(Xbatch)
                pred_svm_idx = np.argmax(proba_svm, axis=1)
                results["pred_svm"] = [class_names[i] for i in pred_svm_idx]
                if include_probs:
                    for j, cls in enumerate(class_names):
                        results[f"svm_prob_{cls}"] = proba_svm[:, j]

        st.success(f"Done! Predicted {len(results):,} rows.")
        st.dataframe(results.head(50), use_container_width=True)

        csv_bytes = results.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download predictions (CSV)",
            data=csv_bytes,
            file_name="batch_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )

# ---------- Sidebar Navigation ----------

st.title("Emotion (Sentiment) Detection • TF-IDF + DT/SVM")
st.caption("Amazon Fine Food Reviews • Pages: Upload → Preprocess → Train → Evaluate → Word Clouds → Predict")

with st.sidebar:
    st.header("Navigation")
    page = st.radio("Go to", ["Data Upload", "Preprocessing", "Model Training", "Models Evaluation", "Word Clouds", "Prediction"])
    if st.button("Reset Session"):
        reset_training_state()
        st.experimental_rerun()

if page == "Data Upload":
    page_upload()
elif page == "Preprocessing":
    page_preprocess()
elif page == "Model Training":
    page_train()
elif page == "Models Evaluation":
    page_evaluate()
elif page == "Word Clouds":
    page_wordclouds()
else:
    page_predict()
