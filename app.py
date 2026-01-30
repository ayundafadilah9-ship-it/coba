import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


# ======================================================
# CONFIG
# ======================================================
DEFAULT_DATA_PATH = "heart_failure_clinical_records_dataset (1).csv"  # taruh sefolder app.py di repo
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
            <div class="big-sub">about → dataset → preprocessing → visualisasi → machine learning (5 metode + terbaik + langkah) → analysis terbaik → prediksi → contact</div>
        </div>
        """,
        unsafe_allow_html=True
    )

@st.cache_data
def read_csv_cached(path: str):
    return pd.read_csv(path)

def try_load_default():
    # AUTO LOAD: Streamlit Cloud bisa baca file yang ada di repo
    try:
        return read_csv_cached(DEFAULT_DATA_PATH)
    except Exception:
        return None

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d.columns = [c.strip() for c in d.columns]
    return d

def find_target_col(df: pd.DataFrame, target_name: str):
    t = (target_name or "").strip().lower()
    for c in df.columns:
        if c.strip().lower() == t:
            return c
    return None

def init_state():
    if "df" not in st.session_state:
        st.session_state.df = try_load_default()
    if "target" not in st.session_state:
        st.session_state.target = DEFAULT_TARGET.strip()

    if "preprocessed" not in st.session_state:
        st.session_state.preprocessed = False

    if "trained_all" not in st.session_state:
        st.session_state.trained_all = False
    if "results" not in st.session_state:
        st.session_state.results = None

    if "best_model_name" not in st.session_state:
        st.session_state.best_model_name = None
    if "best_pipeline" not in st.session_state:
        st.session_state.best_pipeline = None
    if "best_metrics" not in st.session_state:
        st.session_state.best_metrics = None

def reset_training():
    st.session_state.trained_all = False
    st.session_state.results = None
    st.session_state.best_model_name = None
    st.session_state.best_pipeline = None
    st.session_state.best_metrics = None

def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=4000),
        "Random Forest": RandomForestClassifier(n_estimators=500, random_state=42),
        "SVM (RBF)": SVC(C=1.0, kernel="rbf", probability=True),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }

def make_pipeline(model):
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

def kpi_df(df: pd.DataFrame, target_col: str | None):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", int(df.shape[0]))
    c2.metric("Jumlah Kolom", int(df.shape[1]))
    c3.metric("Missing (Total)", int(df.isna().sum().sum()))
    c4.metric("Target", target_col if target_col else st.session_state.target)

def clean_Xy(df: pd.DataFrame, target_name: str):
    if df is None:
        return None, None, None, "dataset kosong."

    df2 = normalize_cols(df)
    target_col = find_target_col(df2, target_name)
    if target_col is None:
        return None, None, None, f"kolom target '{target_name}' tidak ditemukan."

    X = df2.drop(columns=[target_col]).copy()
    y = df2[target_col].copy()

    # hanya numerik
    X = X.select_dtypes(include=[np.number])
    X = X.dropna(axis=1, how="all")

    uniq = sorted(pd.Series(y).dropna().unique().tolist())
    if len(uniq) != 2:
        return None, None, target_col, f"target harus biner (2 kelas). nilai target sekarang: {uniq}"

    return X, y, target_col, None


def render_steps_for_best_model(best_name: str):
    st.subheader("🧾 Langkah Metode Terbaik (sesuai pemenang)")

    st.markdown("### langkah umum (selalu sama)")
    st.write("1) data dipilih: fitur (X) = semua kolom numerik selain target, target (y) = label 0/1.")
    st.write("2) split data train-test (misal 80:20) pakai stratify supaya proporsi 0/1 tetap seimbang.")
    st.write("3) scaling fitur pakai StandardScaler (di Pipeline).")
    st.write("4) model dilatih di data train, lalu diuji di data test.")
    st.write("5) evaluasi pakai accuracy, precision, recall, f1-score (AUC jika tersedia).")

    st.divider()
    st.markdown("### langkah khusus model pemenang")

    if "Random Forest" in best_name:
        st.write("Random Forest = kumpulan banyak decision tree yang voting untuk menentukan hasil akhir.")
        st.write("1) membuat banyak pohon (n_estimators).")
        st.write("2) setiap pohon dilatih dari sampel acak (bootstrap).")
        st.write("3) setiap split hanya mempertimbangkan sebagian fitur (random feature selection).")
        st.write("4) tiap pohon memprediksi 0/1, hasil akhir = voting mayoritas.")
        st.write("5) probabilitas risiko = proporsi vote kelas 1.")
        st.write("6) faktor penting ditampilkan lewat feature importance.")
        return

    if "Gradient Boosting" in best_name:
        st.write("Gradient Boosting = membangun model bertahap, model baru fokus memperbaiki error model sebelumnya.")
        st.write("1) mulai dari prediksi awal (baseline).")
        st.write("2) hitung error/loss dari prediksi.")
        st.write("3) buat tree kecil untuk mempelajari residual (kesalahan).")
        st.write("4) tambahkan tree baru ke model, ulangi sampai jumlah estimator tercapai.")
        st.write("5) hasil akhir adalah gabungan semua tree berurutan (boosting).")
        st.write("6) faktor penting ditampilkan lewat feature importance.")
        return

    if "Logistic Regression" in best_name:
        st.write("Logistic Regression = menghitung probabilitas kelas 1 pakai fungsi sigmoid.")
        st.write("1) hitung skor linear (w·x + b).")
        st.write("2) ubah ke probabilitas dengan sigmoid (0–1).")
        st.write("3) proba >= 0.5 → kelas 1, selain itu → kelas 0.")
        st.write("4) koefisien terbesar menunjukkan fitur yang paling menaikkan/menurunkan risiko.")
        return

    if "SVM" in best_name:
        st.write("SVM (RBF) = mencari batas pemisah terbaik, dan kernel RBF bantu menangani pola non-linear.")
        st.write("1) data dipetakan ke ruang fitur via kernel RBF.")
        st.write("2) cari hyperplane dengan margin terbesar.")
        st.write("3) support vectors adalah titik paling berpengaruh.")
        st.write("4) hasil kelas ditentukan dari sisi hyperplane; probabilitas (jika ada) dari kalibrasi internal.")
        return

    if "KNN" in best_name:
        st.write("KNN = prediksi berdasarkan tetangga terdekat.")
        st.write("1) tentukan K (misal 7).")
        st.write("2) hitung jarak data uji ke semua data train.")
        st.write("3) ambil K tetangga terdekat.")
        st.write("4) voting mayoritas → kelas 0/1.")
        st.write("5) probabilitas kelas 1 = proporsi tetangga bernilai 1.")
        return

    st.info("langkah khusus belum tersedia untuk model ini (tapi langkah umum tetap berlaku).")


init_state()
header()


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
        "Analysis Terbaik",
        "Prediksi",
        "Contact",
    ],
    index=1
)


# ======================================================
# ABOUT
# ======================================================
if page == "About":
    st.markdown('<div class="section-title">📘 About</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.write("dashboard ini dibuat untuk analisis data kesehatan dan klasifikasi risiko.")
    st.write("alur kerja: dataset → preprocessing → visualisasi → train 5 model → pilih terbaik → jelaskan langkah model terbaik → analisis faktor → prediksi.")
    st.markdown('<div class="small-note">tips presentasi: sebut “prediksi risiko” (bukan meramal). target 0/1 adalah label hasil.</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ======================================================
# DATASET
# ======================================================
elif page == "Dataset":
    st.markdown('<div class="section-title">📊 Dataset</div>', unsafe_allow_html=True)

    # auto-load kalau belum ada
    if st.session_state.df is None:
        st.session_state.df = try_load_default()

    left, right = st.columns([1, 2])

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📤 Ganti Dataset (Opsional)")
        st.caption("default dataset auto-load dari CSV di repo. upload hanya kalau mau ganti dataset.")

        up = st.file_uploader("Upload file CSV (opsional)", type=["csv"])
        target_guess = st.text_input("Nama kolom target", value=st.session_state.target)

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Pakai Dataset Ini", use_container_width=True):
                if up is not None:
                    st.session_state.df = pd.read_csv(up)
                else:
                    st.session_state.df = try_load_default()

                st.session_state.target = target_guess.strip()
                st.session_state.preprocessed = False
                reset_training()

        with c2:
            if st.button("Reset ke Default", use_container_width=True):
                st.session_state.df = try_load_default()
                st.session_state.preprocessed = False
                reset_training()

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        df = st.session_state.df
        if df is None:
            st.warning(
                "dataset default tidak ditemukan.\n\n"
                f"pastikan file **{DEFAULT_DATA_PATH}** ada sefolder app.py di GitHub repo, atau upload CSV manual di kiri."
            )
        else:
            df2 = normalize_cols(df)
            target_col = find_target_col(df2, st.session_state.target)

            st.markdown(f'<div class="success-wrap">✅ Data berhasil dimuat: {len(df2):,} records</div>', unsafe_allow_html=True)
            kpi_df(df2, target_col)
            st.markdown('<div class="section-title">📑 Dataset Preview</div>', unsafe_allow_html=True)
            st.dataframe(df2.head(50), use_container_width=True)


# ======================================================
# PREPROCESSING
# ======================================================
elif page == "Preprocessing":
    st.markdown('<div class="section-title">⚙️ Preprocessing</div>', unsafe_allow_html=True)

    df = st.session_state.df
    if df is None:
        st.warning("dataset belum kebaca. cek halaman Dataset dulu.")
    else:
        df2 = normalize_cols(df)
        target_col = find_target_col(df2, st.session_state.target)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Setting Preprocessing")

        drop_dupe = st.checkbox("hapus duplikasi", value=True)
        dropna = st.checkbox("hapus baris yang ada missing (dropna)", value=False)

        if st.button("Terapkan Preprocessing", use_container_width=True):
            work = df2.copy()
            if drop_dupe:
                work = work.drop_duplicates()
            if dropna:
                work = work.dropna()

            st.session_state.df = work
            st.session_state.preprocessed = True
            reset_training()
            st.success("preprocessing selesai diterapkan ✅")

        st.divider()

        if target_col is None:
            st.error(f"kolom target '{st.session_state.target}' tidak ditemukan. cek lagi di halaman Dataset.")
        else:
            st.write("fitur (X) = semua kolom selain target")
            feature_list = [c for c in df2.columns if c != target_col]
            st.write(feature_list)

        st.markdown("</div>", unsafe_allow_html=True)


# ======================================================
# VISUALISASI
# ======================================================
elif page == "Visualisasi":
    st.markdown('<div class="section-title">📊 Visualisasi</div>', unsafe_allow_html=True)

    df = st.session_state.df
    if df is None:
        st.warning("dataset belum kebaca. cek halaman Dataset dulu.")
    else:
        df2 = normalize_cols(df)
        target_col = find_target_col(df2, st.session_state.target)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        kpi_df(df2, target_col)
        st.markdown("</div>", unsafe_allow_html=True)

        if target_col is None:
            st.error(f"kolom target '{st.session_state.target}' tidak ditemukan.")
        else:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Distribusi Target (0 vs 1)")
            vc = df2[target_col].value_counts().reset_index()
            vc.columns = [target_col, "count"]
            fig = px.bar(vc, x=target_col, y="count")
            fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Heatmap Korelasi (Numerik)")
            num_df = df2.select_dtypes(include=[np.number]).copy()
            if num_df.shape[1] < 2:
                st.info("kolom numerik kurang untuk korelasi.")
            else:
                corr = num_df.corr(numeric_only=True)
                fig2 = px.imshow(corr, aspect="auto")
                fig2.update_layout(height=520)
                st.plotly_chart(fig2, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Perbandingan Fitur per Kelas (Boxplot)")

            feature_cols = [c for c in df2.columns if c != target_col]
            feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df2[c])]

            if len(feature_cols) == 0:
                st.info("tidak ada fitur numerik untuk dibandingkan.")
            else:
                pick = st.selectbox("pilih fitur:", feature_cols, index=0)
                fig3 = px.box(df2, x=target_col, y=pick, points="all")
                fig3.update_layout(height=420)
                st.plotly_chart(fig3, use_container_width=True)

            st.markdown("</div>", unsafe_allow_html=True)


# ======================================================
# MACHINE LEARNING (5 METODE) + TERBAIK + LANGKAH
# ======================================================
elif page == "Machine Learning (5 Metode)":
    st.markdown('<div class="section-title">🤖 Machine Learning (5 Metode)</div>', unsafe_allow_html=True)

    df = st.session_state.df
    if df is None:
        st.warning("dataset belum kebaca. cek halaman Dataset dulu.")
    else:
        X, y, target_col, err = clean_Xy(df, st.session_state.target)
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
                        best_pack = {"metrics": m}

                results_df = pd.DataFrame(rows).sort_values("f1", ascending=False)

                st.session_state.trained_all = True
                st.session_state.results = results_df
                st.session_state.best_model_name = best_name
                st.session_state.best_pipeline = best_pipe
                st.session_state.best_metrics = best_pack["metrics"]

                st.success(f"selesai training ✅ metode terbaik: **{best_name}**")

            st.markdown("</div>", unsafe_allow_html=True)

            # ===== tampil hasil jika sudah train =====
            if st.session_state.results is not None:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("Tabel Hasil 5 Metode")
                st.dataframe(st.session_state.results, use_container_width=True)

                figm = px.bar(st.session_state.results, x="model", y="f1")
                figm.update_layout(height=360, title="Perbandingan F1-score (5 Metode)")
                st.plotly_chart(figm, use_container_width=True)

                # ===== METODE TERBAIK + METRIK =====
                best_name = st.session_state.best_model_name
                best_metrics = st.session_state.best_metrics

                st.markdown(f'<div class="success-wrap">🏆 Metode Terbaik: {best_name}</div>', unsafe_allow_html=True)

                a1, a2, a3, a4 = st.columns(4)
                a1.metric("Accuracy", f"{best_metrics['accuracy']:.4f}")
                a2.metric("Precision", f"{best_metrics['precision']:.4f}")
                a3.metric("Recall", f"{best_metrics['recall']:.4f}")
                a4.metric("F1-score", f"{best_metrics['f1']:.4f}")

                if best_metrics.get("auc") is not None:
                    st.info(f"AUC: {best_metrics['auc']:.4f}")

                st.divider()

                # ===== LANGKAH SESUAI METODE TERBAIK =====
                render_steps_for_best_model(best_name)

                st.markdown("</div>", unsafe_allow_html=True)


# ======================================================
# ANALYSIS TERBAIK
# ======================================================
elif page == "Analysis Terbaik":
    st.markdown('<div class="section-title">🧠 Analysis Terbaik</div>', unsafe_allow_html=True)

    df = st.session_state.df
    if df is None:
        st.warning("dataset belum kebaca. cek halaman Dataset dulu.")
    elif not st.session_state.trained_all:
        st.warning("train 5 metode dulu di halaman Machine Learning.")
    else:
        df2 = normalize_cols(df)
        target_col = find_target_col(df2, st.session_state.target)
        if target_col is None:
            st.error(f"kolom target '{st.session_state.target}' tidak ditemukan.")
        else:
            best_name = st.session_state.best_model_name
            pipe = st.session_state.best_pipeline

            feature_cols = [c for c in df2.columns if c != target_col]
            feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df2[c])]

            st.markdown(f'<div class="success-wrap">✅ Model terbaik aktif: {best_name}</div>', unsafe_allow_html=True)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Perbandingan Rata-rata Fitur (kelas 0 vs 1)")
            grp = df2.groupby(target_col)[feature_cols].mean().T
            if grp.shape[1] == 2:
                grp.columns = [f"{target_col}=0", f"{target_col}=1"]
            st.dataframe(grp, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Fitur Paling Berpengaruh (dari model)")
            model = pipe.named_steps.get("model", None)

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

            elif model is not None and hasattr(model, "coef_"):
                coef = model.coef_[0]
                coef_df = pd.DataFrame({"Fitur": feature_cols, "Koefisien": coef}).sort_values("Koefisien", ascending=False)

                topn = st.slider("Top-N (koefisien)", 5, min(20, len(feature_cols)), 10)
                fig = px.bar(coef_df.head(topn)[::-1], x="Koefisien", y="Fitur", orientation="h")
                fig.update_layout(height=420, title="Koefisien Model (Logistic Regression)")
                st.plotly_chart(fig, use_container_width=True)
                st.caption("koefisien positif → cenderung menaikkan risiko kelas 1, negatif → menurunkan.")

            else:
                st.info("model ini tidak punya importance/coef yang gampang ditampilkan.")

            st.divider()
            st.subheader("Visual Cepat (Boxplot fitur vs kelas)")
            if len(feature_cols) > 0:
                pick = st.selectbox("pilih fitur:", feature_cols, index=0, key="boxpick_analysis")
                fig2 = px.box(df2, x=target_col, y=pick, points="all")
                fig2.update_layout(height=420)
                st.plotly_chart(fig2, use_container_width=True)

            st.markdown("</div>", unsafe_allow_html=True)


# ======================================================
# PREDIKSI (masih upload CSV data baru)
# ======================================================
elif page == "Prediksi":
    st.markdown('<div class="section-title">🔮 Prediksi Data Baru</div>', unsafe_allow_html=True)

    df_train = st.session_state.df
    if df_train is None:
        st.warning("dataset belum kebaca. cek halaman Dataset dulu.")
    elif not st.session_state.trained_all:
        st.warning("train 5 metode dulu biar ada model terbaik.")
    else:
        df_train2 = normalize_cols(df_train)
        target_col = find_target_col(df_train2, st.session_state.target)
        if target_col is None:
            st.error(f"kolom target '{st.session_state.target}' tidak ditemukan.")
        else:
            pipe = st.session_state.best_pipeline
            best_name = st.session_state.best_model_name

            feature_cols = [c for c in df_train2.columns if c != target_col]
            feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df_train2[c])]

            st.markdown(f'<div class="success-wrap">✅ Model aktif: {best_name}</div>', unsafe_allow_html=True)

            col_left, col_right = st.columns([1, 1])
            with col_left:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("📤 Upload Data Baru")
                new_file = st.file_uploader("Upload file CSV", type=["csv"], key="new_data_file")
                st.caption("file baru harus punya kolom fitur yang sama seperti data training.")
                st.markdown("</div>", unsafe_allow_html=True)

            with col_right:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("⚙️ Algoritma")
                st.selectbox("Algoritma:", [f"Metode Terbaik: {best_name}"], index=0)
                run = st.button("🚀 Jalankan Prediksi", use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            if run:
                if new_file is None:
                    st.warning("upload file CSV dulu.")
                else:
                    new_df = pd.read_csv(new_file)
                    new_df = normalize_cols(new_df)

                    missing = set(feature_cols) - set(new_df.columns)
                    if missing:
                        st.error(f"kolom ini tidak ada di file baru: {sorted(list(missing))}")
                    else:
                        X_new = new_df[feature_cols].copy()
                        X_new = X_new.select_dtypes(include=[np.number])

                        preds = pipe.predict(X_new)

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
