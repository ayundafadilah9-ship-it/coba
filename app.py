import os
import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


# ======================================================
# CONFIG
# ======================================================
DEFAULT_DATA_PATH = "heart_failure_clinical_records_dataset (1).csv"  # kalau ada sefolder
DEFAULT_TARGET = "DEATH_EVENT"


# ======================================================
# PAGE CONFIG + STYLE
# ======================================================
st.set_page_config(page_title="Dashboard Analisis Kesehatan", page_icon="📊", layout="wide")

st.markdown(
    """
<style>
.stApp { background: #F5F7FB; }
section[data-testid="stSidebar"]{
    background: #EEF2F7;
    border-right: 1px solid #E5E7EB;
}
section[data-testid="stSidebar"] *{ color: #111827 !important; }

.big-header{
    background: #E9EEF6;
    border: 1px solid #D7DFEA;
    border-radius: 16px;
    padding: 18px 22px;
    box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
    margin-top: 6px;
}
.big-title{
    font-size: 40px;
    font-weight: 800;
    color: #111827;
    line-height: 1.1;
    text-align: center;
}
.big-sub{
    color: #374151;
    text-align: center;
    margin-top: 8px;
}

.section-title{
    font-size: 32px;
    font-weight: 800;
    color: #111827;
    margin-top: 18px;
}

.success-wrap{
    background: #E9FBEF;
    border: 1px solid #B7F0C4;
    border-radius: 12px;
    padding: 10px 14px;
    margin: 12px 0 8px 0;
    color: #065F46;
    font-weight: 700;
}

.card{
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 16px;
    padding: 14px 16px;
    box-shadow: 0 8px 20px rgba(15,23,42,0.05);
}
.small-note{ color:#6B7280; font-size: 13px; }
</style>
""",
    unsafe_allow_html=True
)


# ======================================================
# HELPERS
# ======================================================
def header():
    st.markdown(
        """
        <div class="big-header">
            <div class="big-title">🧾 Dashboard Analisis Kesehatan</div>
            <div class="big-sub">about → dataset → preprocessing → visualisasi → machine learning (5 metode) → metode terbaik → analysis terbaik → prediksi → contact</div>
        </div>
        """,
        unsafe_allow_html=True
    )

@st.cache_data
def read_csv_cached(path: str):
    return pd.read_csv(path)

def try_load_default():
    if os.path.exists(DEFAULT_DATA_PATH):
        return read_csv_cached(DEFAULT_DATA_PATH)
    return None

def init_state():
    if "df" not in st.session_state:
        st.session_state.df = try_load_default()
    if "target" not in st.session_state:
        st.session_state.target = DEFAULT_TARGET

    # preprocessing flags (optional)
    if "preprocessed" not in st.session_state:
        st.session_state.preprocessed = False

    # training multi-model
    if "trained_all" not in st.session_state:
        st.session_state.trained_all = False
    if "results" not in st.session_state:
        st.session_state.results = None

    # best model
    if "best_model_name" not in st.session_state:
        st.session_state.best_model_name = None
    if "best_pipeline" not in st.session_state:
        st.session_state.best_pipeline = None
    if "best_metrics" not in st.session_state:
        st.session_state.best_metrics = None

    # test split saved
    if "X_test" not in st.session_state:
        st.session_state.X_test = None
    if "y_test" not in st.session_state:
        st.session_state.y_test = None
    if "y_pred_best" not in st.session_state:
        st.session_state.y_pred_best = None
    if "y_proba_best" not in st.session_state:
        st.session_state.y_proba_best = None

def reset_training():
    st.session_state.trained_all = False
    st.session_state.results = None
    st.session_state.best_model_name = None
    st.session_state.best_pipeline = None
    st.session_state.best_metrics = None
    st.session_state.X_test = None
    st.session_state.y_test = None
    st.session_state.y_pred_best = None
    st.session_state.y_proba_best = None

def get_models():
    # 5 metode
    return {
        "Logistic Regression": LogisticRegression(max_iter=4000),
        "Random Forest": RandomForestClassifier(n_estimators=500, random_state=42),
        "SVM (RBF)": SVC(C=1.0, kernel="rbf", probability=True),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }

def make_pipeline(model):
    # dataset numeric → scaler aman dipakai
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])

def safe_auc(y_true, y_proba):
    try:
        return roc_auc_score(y_true, y_proba)
    except Exception:
        return None

def eval_binary(y_true, y_pred, y_proba=None):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1v = f1_score(y_true, y_pred, zero_division=0)
    auc = safe_auc(y_true, y_proba) if y_proba is not None else None
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1v, "auc": auc}

def pick_best(results_df: pd.DataFrame):
    # Best by F1, tie-breaker AUC, lalu accuracy
    df = results_df.copy()
    df["auc_fill"] = df["auc"].fillna(-1)
    df = df.sort_values(["f1", "auc_fill", "accuracy"], ascending=False)
    return df.iloc[0]["model"]

def kpi_df(df: pd.DataFrame):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", int(df.shape[0]))
    c2.metric("Jumlah Kolom", int(df.shape[1]))
    c3.metric("Missing (Total)", int(df.isna().sum().sum()))
    c4.metric("Target", st.session_state.target)

def clean_Xy(df: pd.DataFrame, target: str):
    if df is None:
        return None, None, "dataset kosong."
    if target not in df.columns:
        return None, None, f"kolom target '{target}' tidak ditemukan."

    X = df.drop(columns=[target]).copy()
    y = df[target].copy()

    # hanya numerik (biar aman)
    X = X.select_dtypes(include=[np.number])
    X = X.dropna(axis=1, how="all")

    # target biner
    uniq = sorted(pd.Series(y).dropna().unique().tolist())
    if len(uniq) != 2:
        return None, None, f"target harus biner (2 kelas). nilai target sekarang: {uniq}"

    return X, y, None

init_state()


# ======================================================
# SIDEBAR NAV
# ======================================================
st.sidebar.markdown("### 📊 Navigasi Dashboard")
page = st.sidebar.radio(
    "Pilih Halaman:",
    [
        "About",
        "Dataset",
        "Preprocessing",
        "Visualisasi",
        "Machine Learning (5 Metode)",
        "Hasil Akhir (Terbaik)",
        "Analysis Terbaik",
        "Prediksi",
        "Contact",
    ],
    index=1
)

header()


# ======================================================
# ABOUT
# ======================================================
if page == "About":
    st.markdown('<div class="section-title">📘 About</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.write("dashboard ini dibuat untuk analisis data kesehatan dan klasifikasi risiko.")
    st.write("alur kerja: upload data → preprocessing → visualisasi → bandingkan 5 model → pilih model terbaik → interpretasi → prediksi data baru.")
    st.markdown('<div class="small-note">tips presentasi: sebut “prediksi risiko” (bukan meramal). target 0/1 dipakai sebagai label hasil.</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ======================================================
# DATASET (UPLOAD)
# ======================================================
elif page == "Dataset":
    st.markdown('<div class="section-title">📊 Dataset</div>', unsafe_allow_html=True)

    left, right = st.columns([1, 2])

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📤 Upload Data Baru")

        up = st.file_uploader("Upload file CSV", type=["csv"])
        target_guess = st.text_input("Nama kolom target", value=st.session_state.target)

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Muat Dataset", use_container_width=True):
                if up is not None:
                    st.session_state.df = pd.read_csv(up)
                else:
                    st.session_state.df = try_load_default()

                st.session_state.target = target_guess.strip()
                st.session_state.preprocessed = False
                reset_training()

        with c2:
            if st.button("Reset Dataset", use_container_width=True):
                st.session_state.df = None
                st.session_state.preprocessed = False
                reset_training()

        st.markdown('<div class="small-note">kalau file csv ada sefolder app.py, klik “Muat Dataset” tanpa upload juga bisa.</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        df = st.session_state.df
        if df is None:
            st.warning("dataset belum kebaca. upload CSV dulu.")
        else:
            st.markdown(f'<div class="success-wrap">✅ Data berhasil dimuat: {len(df):,} records</div>', unsafe_allow_html=True)
            kpi_df(df)
            st.markdown('<div class="section-title">📑 Dataset Preview</div>', unsafe_allow_html=True)
            st.dataframe(df.head(50), use_container_width=True)


# ======================================================
# PREPROCESSING
# ======================================================
elif page == "Preprocessing":
    st.markdown('<div class="section-title">⚙️ Preprocessing</div>', unsafe_allow_html=True)

    df = st.session_state.df
    target = st.session_state.target

    if df is None:
        st.warning("muat dataset dulu di halaman dataset.")
    else:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Setting Preprocessing")

        drop_dupe = st.checkbox("hapus duplikasi", value=True)
        dropna = st.checkbox("hapus baris yang ada missing (dropna)", value=False)

        if st.button("Terapkan Preprocessing", use_container_width=True):
            work = df.copy()
            if drop_dupe:
                work = work.drop_duplicates()
            if dropna:
                work = work.dropna()

            st.session_state.df = work
            st.session_state.preprocessed = True
            reset_training()
            st.success("preprocessing selesai diterapkan ✅")

        st.divider()

        if target not in df.columns:
            st.error(f"kolom target '{target}' tidak ditemukan. cek lagi di halaman dataset.")
        else:
            st.write("fitur (X) = semua kolom selain target")
            st.write([c for c in df.columns if c != target])

        st.markdown("</div>", unsafe_allow_html=True)


# ======================================================
# VISUALISASI
# ======================================================
elif page == "Visualisasi":
    st.markdown('<div class="section-title">📊 Visualisasi</div>', unsafe_allow_html=True)

    df = st.session_state.df
    target = st.session_state.target

    if df is None:
        st.warning("muat dataset dulu di halaman dataset.")
    else:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        kpi_df(df)
        st.markdown("</div>", unsafe_allow_html=True)

        if target not in df.columns:
            st.error(f"kolom target '{target}' tidak ditemukan. cek lagi di halaman dataset.")
        else:
            # 1) distribusi target
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Distribusi Target (0 vs 1)")
            vc = df[target].value_counts().reset_index()
            vc.columns = [target, "count"]
            fig = px.bar(vc, x=target, y="count")
            fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # 2) heatmap korelasi numerik
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Heatmap Korelasi (Numerik)")
            num_df = df.select_dtypes(include=[np.number]).copy()
            if num_df.shape[1] < 2:
                st.info("kolom numerik kurang untuk korelasi.")
            else:
                corr = num_df.corr(numeric_only=True)
                fig2 = px.imshow(corr, aspect="auto")
                fig2.update_layout(height=520)
                st.plotly_chart(fig2, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # 3) perbandingan fitur per kelas
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Perbandingan Fitur per Kelas (Boxplot)")

            feature_cols = [c for c in df.columns if c != target]
            feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]

            if len(feature_cols) == 0:
                st.info("tidak ada fitur numerik untuk dibandingkan.")
            else:
                pick = st.selectbox("pilih fitur:", feature_cols, index=0)
                fig3 = px.box(df, x=target, y=pick, points="all")
                fig3.update_layout(height=420)
                st.plotly_chart(fig3, use_container_width=True)

            st.markdown("</div>", unsafe_allow_html=True)


# ======================================================
# MACHINE LEARNING (5 METODE)
# ======================================================
elif page == "Machine Learning (5 Metode)":
    st.markdown('<div class="section-title">🤖 Machine Learning (5 Metode)</div>', unsafe_allow_html=True)

    df = st.session_state.df
    target = st.session_state.target

    if df is None:
        st.warning("muat dataset dulu di halaman dataset.")
    else:
        X, y, err = clean_Xy(df, target)
        if err:
            st.error(err)
        else:
            st.markdown('<div class="card">', unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)
            with c1:
                test_size = st.slider("test size", 0.1, 0.4, 0.2, 0.05)
            with c2:
                random_state = st.number_input("random state", value=42, step=1)
            with c3:
                metric_pick = st.selectbox("patokan terbaik", ["F1-score", "AUC", "Accuracy"], index=0)

            st.divider()

            if st.button("🚀 Train 5 Metode", use_container_width=True):
                models = get_models()

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y,
                    test_size=float(test_size),
                    random_state=int(random_state),
                    stratify=y
                )

                rows = []
                best_score = -1
                best_name = None
                best_pipe = None
                best_pack = None

                for name, model in models.items():
                    pipe = make_pipeline(model)
                    pipe.fit(X_train, y_train)

                    y_pred = pipe.predict(X_test)
                    y_proba = None
                    if hasattr(pipe, "predict_proba"):
                        try:
                            y_proba = pipe.predict_proba(X_test)[:, 1]
                        except Exception:
                            y_proba = None

                    m = eval_binary(y_test, y_pred, y_proba)

                    rows.append({
                        "model": name,
                        "accuracy": m["accuracy"],
                        "precision": m["precision"],
                        "recall": m["recall"],
                        "f1": m["f1"],
                        "auc": m["auc"]
                    })

                    # scoring selection
                    if metric_pick == "F1-score":
                        score = m["f1"]
                    elif metric_pick == "AUC":
                        score = m["auc"] if m["auc"] is not None else -1
                    else:
                        score = m["accuracy"]

                    if score is not None and score > best_score:
                        best_score = score
                        best_name = name
                        best_pipe = pipe
                        best_pack = {"y_pred": y_pred, "y_proba": y_proba, "metrics": m}

                results_df = pd.DataFrame(rows).sort_values("f1", ascending=False)

                st.session_state.trained_all = True
                st.session_state.results = results_df
                st.session_state.best_model_name = best_name
                st.session_state.best_pipeline = best_pipe
                st.session_state.best_metrics = best_pack["metrics"]
                st.session_state.X_test = X_test
                st.session_state.y_test = y_test
                st.session_state.y_pred_best = best_pack["y_pred"]
                st.session_state.y_proba_best = best_pack["y_proba"]

                st.success(f"selesai training ✅ metode terbaik sementara: **{best_name}**")

            st.markdown("</div>", unsafe_allow_html=True)

            if st.session_state.results is not None:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("Tabel Hasil 5 Metode")
                st.dataframe(st.session_state.results, use_container_width=True)

                # chart ringkas (F1)
                figm = px.bar(st.session_state.results, x="model", y="f1")
                figm.update_layout(height=360, title="Perbandingan F1-score (5 Metode)")
                st.plotly_chart(figm, use_container_width=True)

                st.markdown("</div>", unsafe_allow_html=True)


# ======================================================
# HASIL AKHIR (TERBAIK)
# ======================================================
elif page == "Hasil Akhir (Terbaik)":
    st.markdown('<div class="section-title">🏆 Hasil Akhir (Metode Terbaik)</div>', unsafe_allow_html=True)

    if not st.session_state.trained_all:
        st.warning("jalankan training 5 metode dulu di halaman machine learning.")
    else:
        best_name = st.session_state.best_model_name
        best_metrics = st.session_state.best_metrics

        st.markdown(f'<div class="success-wrap">✅ Metode terbaik: {best_name}</div>', unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{best_metrics['accuracy']:.4f}")
        c2.metric("Precision", f"{best_metrics['precision']:.4f}")
        c3.metric("Recall", f"{best_metrics['recall']:.4f}")
        c4.metric("F1-score", f"{best_metrics['f1']:.4f}")

        if best_metrics.get("auc") is not None:
            st.info(f"AUC: {best_metrics['auc']:.4f}")

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Ringkasan")
        st.write("metode terbaik dipilih berdasarkan metrik yang kamu tentukan saat training (default: F1-score).")
        st.write("selanjutnya kamu bisa lihat interpretasi di halaman analysis terbaik dan pakai modelnya untuk prediksi data baru.")
        st.markdown("</div>", unsafe_allow_html=True)


# ======================================================
# ANALYSIS TERBAIK (interpretasi / faktor)
# ======================================================
elif page == "Analysis Terbaik":
    st.markdown('<div class="section-title">🧠 Analysis Terbaik</div>', unsafe_allow_html=True)

    df = st.session_state.df
    target = st.session_state.target

    if df is None:
        st.warning("muat dataset dulu.")
    elif not st.session_state.trained_all:
        st.warning("train 5 metode dulu.")
    else:
        best_name = st.session_state.best_model_name
        pipe = st.session_state.best_pipeline

        feature_cols = [c for c in df.columns if c != target]
        feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]

        st.markdown(f'<div class="success-wrap">✅ Model terbaik aktif: {best_name}</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Perbandingan Rata-rata Fitur (kelas 0 vs 1)")
        grp = df.groupby(target)[feature_cols].mean().T
        if grp.shape[1] == 2:
            grp.columns = [f"{target}=0", f"{target}=1"]
        st.dataframe(grp, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Fitur Paling Berpengaruh (dari model)")

        model = pipe.named_steps.get("model", None)

        # RandomForest / Tree based
        if model is not None and hasattr(model, "feature_importances_"):
            imp = pd.DataFrame({
                "Fitur": feature_cols,
                "Importance": model.feature_importances_
            }).sort_values("Importance", ascending=False)

            topn = st.slider("Top-N (importance)", 5, min(20, len(feature_cols)), 10)
            imp_top = imp.head(topn)

            fig = px.bar(imp_top[::-1], x="Importance", y="Fitur", orientation="h")
            fig.update_layout(height=420, title="Feature Importance")
            st.plotly_chart(fig, use_container_width=True)

            st.caption("semakin besar importance → semakin besar kontribusi fitur ke keputusan model.")

        # Logistic Regression
        elif model is not None and hasattr(model, "coef_"):
            coef = model.coef_[0]
            coef_df = pd.DataFrame({"Fitur": feature_cols, "Koefisien": coef}).sort_values("Koefisien", ascending=False)

            topn = st.slider("Top-N (koefisien)", 5, min(20, len(feature_cols)), 10)
            st.write("koefisien positif → cenderung meningkatkan risiko (kelas 1), negatif → menurunkan.")

            fig = px.bar(coef_df.head(topn)[::-1], x="Koefisien", y="Fitur", orientation="h")
            fig.update_layout(height=420, title="Koefisien Model (Logistic Regression)")
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(coef_df.head(topn), use_container_width=True)

        else:
            st.info("model ini tidak punya importance/coef yang gampang ditampilkan. (coba Random Forest / Logistic Regression)")

        st.divider()
        st.subheader("Visual Cepat (Boxplot fitur vs kelas)")
        if len(feature_cols) > 0:
            pick = st.selectbox("pilih fitur:", feature_cols, index=0, key="boxpick_analysis")
            fig2 = px.box(df, x=target, y=pick, points="all")
            fig2.update_layout(height=420)
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)


# ======================================================
# PREDIKSI (UPLOAD CSV + BUTTON)
# ======================================================
elif page == "Prediksi":
    st.markdown('<div class="section-title">🔮 Prediksi Data Baru</div>', unsafe_allow_html=True)

    df_train = st.session_state.df
    target = st.session_state.target

    if df_train is None:
        st.warning("muat dataset dulu.")
    elif not st.session_state.trained_all:
        st.warning("train 5 metode dulu biar ada model terbaik.")
    else:
        pipe = st.session_state.best_pipeline
        best_name = st.session_state.best_model_name

        feature_cols = [c for c in df_train.columns if c != target]
        feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df_train[c])]

        # layout mirip screenshot: upload kiri, pilih algo kanan
        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("📤 Upload Data Baru")
            new_file = st.file_uploader("Upload file CSV", type=["csv"], key="new_data_file")
            st.markdown('<div class="small-note">file baru harus punya kolom fitur yang sama seperti data training.</div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_right:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("⚙️ Pilih Algoritma")
            st.selectbox("Algoritma:", [f"Metode Terbaik: {best_name}"], index=0)
            run = st.button("🚀 Jalankan Prediksi", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        if run:
            if new_file is None:
                st.warning("upload file CSV dulu.")
            else:
                new_df = pd.read_csv(new_file)

                missing = set(feature_cols) - set(new_df.columns)
                if missing:
                    st.error(f"kolom ini tidak ada di file baru: {sorted(list(missing))}")
                else:
                    X_new = new_df[feature_cols].copy()
                    X_new = X_new.select_dtypes(include=[np.number])

                    preds = pipe.predict(X_new)

                    # probabilitas kalau ada
                    proba = None
                    if hasattr(pipe, "predict_proba"):
                        try:
                            proba = pipe.predict_proba(X_new)[:, 1]
                        except Exception:
                            proba = None

                    out = new_df.copy()
                    out["Prediksi_Risiko"] = preds
                    if proba is not None:
                        out["Prob_Risiko_Tinggi"] = proba

                    st.success("prediksi selesai ✅")

                    a, b = st.columns(2)
                    a.metric("Risiko Rendah (0)", int((preds == 0).sum()))
                    b.metric("Risiko Tinggi (1)", int((preds == 1).sum()))

                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.subheader("📄 Hasil Prediksi")
                    st.dataframe(out, use_container_width=True)

                    # pie persentase
                    pie_df = pd.DataFrame({
                        "kelas": ["0 (rendah)", "1 (tinggi)"],
                        "jumlah": [int((preds == 0).sum()), int((preds == 1).sum())]
                    })
                    figp = px.pie(pie_df, names="kelas", values="jumlah")
                    figp.update_layout(height=360, title="Proporsi Hasil Prediksi")
                    st.plotly_chart(figp, use_container_width=True)

                    csv_bytes = out.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇️ Download hasil prediksi (CSV)",
                        csv_bytes,
                        file_name="hasil_prediksi.csv",
                        mime="text/csv"
                    )
                    st.markdown("</div>", unsafe_allow_html=True)


# ======================================================
# CONTACT
# ======================================================
elif page == "Contact":
    st.markdown('<div class="section-title">📞 Contact</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown("""
**Nama Mahasiswa:** Fadilah Andriana Putri Ayunda  
**Program Studi:** S1 Sains Data  
**Universitas:** Universitas Muhammadiyah Semarang  

📧 **Email:** Ayundafadilah9@gmail.com
""")

    st.markdown("</div>", unsafe_allow_html=True)
