import os
import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


# ======================================================
# CONFIG
# ======================================================
DEFAULT_DATA_PATH = "heart_failure_clinical_records_dataset (1).csv"      # taruh sefolder app.py (di repo)
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
            <div class="big-sub">about → dataset → preprocessing → visualisasi → machine learning (5 metode) → metode terbaik → prediksi → contact</div>
        </div>
        """,
        unsafe_allow_html=True
    )

@st.cache_data
def read_csv_cached(path: str):
    return pd.read_csv(path)

def try_load_default():
    # cari file default di folder kerja (repo)
    if os.path.exists(DEFAULT_DATA_PATH):
        return read_csv_cached(DEFAULT_DATA_PATH)
    return None

def reset_training_state():
    st.session_state.trained_all = False
    st.session_state.results = None
    st.session_state.best_model_name = None
    st.session_state.best_pipeline = None
    st.session_state.best_metrics = None
    st.session_state.feature_cols = None

def init_state():
    if "df" not in st.session_state:
        st.session_state.df = try_load_default()
    if "target" not in st.session_state:
        st.session_state.target = DEFAULT_TARGET

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

    if "feature_cols" not in st.session_state:
        st.session_state.feature_cols = None

def kpi_df(df: pd.DataFrame):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", int(df.shape[0]))
    c2.metric("Jumlah Kolom", int(df.shape[1]))
    c3.metric("Missing (Total)", int(df.isna().sum().sum()))
    c4.metric("Target", st.session_state.target)

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

def clean_Xy(df: pd.DataFrame, target: str):
    if df is None:
        return None, None, "dataset kosong / belum kebaca."
    if target not in df.columns:
        return None, None, f"kolom target '{target}' tidak ditemukan."

    X = df.drop(columns=[target]).copy()
    y = df[target].copy()

    # ambil numerik saja (stabil)
    X = X.select_dtypes(include=[np.number]).copy()
    X = X.dropna(axis=1, how="all")

    uniq = sorted(pd.Series(y).dropna().unique().tolist())
    if len(uniq) != 2:
        return None, None, f"target harus biner (2 kelas). nilai target sekarang: {uniq}"

    return X, y, None


# ======================================================
# INIT
# ======================================================
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
        "Metode Terbaik",
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
    st.write("alur kerja: dataset → preprocessing → visualisasi → bandingkan 5 model → ambil metode terbaik → prediksi input pasien.")
    st.markdown(
        '<div class="small-note">catatan: target 0/1 adalah label hasil (bukan “meramal”), dan digunakan untuk klasifikasi.</div>',
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)


# ======================================================
# DATASET
# ======================================================
elif page == "Dataset":
    st.markdown('<div class="section-title">📊 Dataset</div>', unsafe_allow_html=True)
    df = st.session_state.df
    target = st.session_state.target

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📌 Auto Load Dataset (tanpa upload)")
    st.write(f"File default yang dicari: `{DEFAULT_DATA_PATH}` (harus satu folder dengan `app.py`).")
    st.write(f"Target default: `{DEFAULT_TARGET}`")

    colA, colB = st.columns([1, 1])
    with colA:
        new_target = st.text_input("Nama kolom target", value=target)
    with colB:
        if st.button("🔄 Reload Dataset", use_container_width=True):
            st.session_state.df = try_load_default()
            st.session_state.target = new_target.strip()
            reset_training_state()
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    df = st.session_state.df
    if df is None:
        st.error("dataset belum kebaca. pastikan file CSV sudah ada di repo dan namanya benar.")
        st.info("saran: rename file jadi `heart_failure.csv` lalu push ke GitHub dan reboot app di Streamlit Cloud.")
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
        st.warning("dataset belum kebaca. cek halaman Dataset.")
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
            reset_training_state()
            st.success("preprocessing selesai diterapkan ✅")

        st.divider()

        if target not in df.columns:
            st.error(f"kolom target '{target}' tidak ditemukan.")
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
        st.warning("dataset belum kebaca. cek halaman Dataset.")
    elif target not in df.columns:
        st.error(f"kolom target '{target}' tidak ditemukan.")
    else:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        kpi_df(df)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Distribusi Target (0 vs 1)")
        vc = df[target].value_counts().reset_index()
        vc.columns = [target, "count"]
        fig = px.bar(vc, x=target, y="count")
        fig.update_layout(height=360)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Perbandingan Fitur per Kelas (Boxplot)")
        num_cols = [c for c in df.columns if c != target and pd.api.types.is_numeric_dtype(df[c])]
        if len(num_cols) == 0:
            st.info("tidak ada fitur numerik untuk divisualisasikan.")
        else:
            pick = st.selectbox("Pilih fitur:", num_cols, index=0)
            fig2 = px.box(df, x=target, y=pick, points="all")
            fig2.update_layout(height=420)
            st.plotly_chart(fig2, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ======================================================
# MACHINE LEARNING (5 METODE)
# ======================================================
elif page == "Machine Learning (5 Metode)":
    st.markdown('<div class="section-title">🤖 Machine Learning (5 Metode)</div>', unsafe_allow_html=True)

    df = st.session_state.df
    target = st.session_state.target

    if df is None:
        st.warning("dataset belum kebaca. cek halaman Dataset.")
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
                best_metrics = None

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
                    rows.append({"model": name, **m})

                    # pilih skor terbaik
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
                        best_metrics = m

                results_df = pd.DataFrame(rows).sort_values("f1", ascending=False)

                st.session_state.trained_all = True
                st.session_state.results = results_df
                st.session_state.best_model_name = best_name
                st.session_state.best_pipeline = best_pipe
                st.session_state.best_metrics = best_metrics
                st.session_state.feature_cols = list(X.columns)

                st.success(f"selesai training ✅ metode terbaik: **{best_name}**")

            st.markdown("</div>", unsafe_allow_html=True)

            if st.session_state.results is not None:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("Tabel Hasil 5 Metode")
                show = st.session_state.results.copy()
                # rapiin angka
                for c in ["accuracy", "precision", "recall", "f1", "auc"]:
                    if c in show.columns:
                        show[c] = show[c].astype(float).round(4)
                st.dataframe(show, use_container_width=True)

                figm = px.bar(st.session_state.results, x="model", y="f1", title="Perbandingan F1-score")
                figm.update_layout(height=360)
                st.plotly_chart(figm, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)


# ======================================================
# METODE TERBAIK (STEP BY STEP)
# ======================================================
elif page == "Metode Terbaik":
    st.markdown('<div class="section-title">🏆 Metode Terbaik</div>', unsafe_allow_html=True)

    if not st.session_state.trained_all:
        st.warning("jalankan training 5 metode dulu di halaman Machine Learning.")
    else:
        best_name = st.session_state.best_model_name
        best_metrics = st.session_state.best_metrics

        st.markdown(f'<div class="success-wrap">✅ Metode terbaik terpilih: <b>{best_name}</b></div>', unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{best_metrics['accuracy']:.4f}")
        c2.metric("Precision", f"{best_metrics['precision']:.4f}")
        c3.metric("Recall", f"{best_metrics['recall']:.4f}")
        c4.metric("F1-score", f"{best_metrics['f1']:.4f}")
        if best_metrics.get("auc") is not None:
            st.info(f"AUC: {best_metrics['auc']:.4f}")

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📌 Langkah Kerja Metode Terbaik (sesuai model yang menang)")

        if best_name == "Random Forest":
            st.write("""
1) membagi data jadi train & test  
2) membuat banyak decision tree dari sampel data random (bootstrap)  
3) setiap split memilih subset fitur secara acak  
4) prediksi = voting mayoritas dari semua tree  
5) evaluasi hasil dengan accuracy / precision / recall / f1 / auc  
            """)
        elif best_name == "Gradient Boosting":
            st.write("""
1) mulai dari model sederhana (tree kecil)  
2) hitung error prediksi (residual)  
3) buat tree baru untuk memperbaiki error sebelumnya  
4) ulangi bertahap sampai jumlah tree terpenuhi  
5) prediksi akhir = gabungan semua tree  
6) evaluasi dengan accuracy / precision / recall / f1 / auc  
            """)
        elif best_name == "Logistic Regression":
            st.write("""
1) normalisasi fitur (scaling)  
2) hitung probabilitas kelas 1 pakai fungsi sigmoid  
3) training mencari koefisien terbaik (optimisasi)  
4) hasil probabilitas diubah ke 0/1 pakai threshold  
5) evaluasi dengan accuracy / precision / recall / f1 / auc  
            """)
        elif best_name == "SVM (RBF)":
            st.write("""
1) normalisasi fitur (scaling)  
2) cari hyperplane pemisah terbaik dengan margin terbesar  
3) kernel RBF membantu pemisahan kalau pola tidak linear  
4) hasil kelas ditentukan dari sisi hyperplane  
5) evaluasi dengan accuracy / precision / recall / f1 / auc  
            """)
        elif best_name == "KNN":
            st.write("""
1) normalisasi fitur (scaling)  
2) tentukan nilai k (jumlah tetangga)  
3) hitung jarak data baru ke data train  
4) ambil k tetangga terdekat  
5) voting mayoritas = kelas prediksi  
6) evaluasi dengan accuracy / precision / recall / f1 / auc  
            """)
        else:
            st.write("step model belum disediakan.")

        st.caption("buat dosen: metode terbaik dipilih berdasarkan metrik yang kamu set saat training (default F1-score).")
        st.markdown("</div>", unsafe_allow_html=True)


# ======================================================
# PREDIKSI (FORM INPUT PASIEN)
# ======================================================
elif page == "Prediksi":
    st.markdown('<div class="section-title">🧾 Prediksi Risiko (Input Pasien)</div>', unsafe_allow_html=True)
    st.caption("input numerik + interpretasi kategori klinis")

    df_train = st.session_state.df
    target = st.session_state.target

    if df_train is None:
        st.warning("dataset belum kebaca. cek halaman Dataset.")
    elif not st.session_state.trained_all:
        st.warning("train 5 metode dulu biar ada model terbaik.")
    else:
        pipe = st.session_state.best_pipeline
        best_name = st.session_state.best_model_name
        feature_cols = st.session_state.feature_cols

        st.markdown(f'<div class="success-wrap">✅ Model aktif: <b>{best_name}</b></div>', unsafe_allow_html=True)

        # helper default median
        def default_val(col):
            try:
                return float(df_train[col].median())
            except Exception:
                return 0.0

        # label lebih enak
        label_map = {
            "age": "Usia (Tahun)",
            "sex": "Jenis Kelamin",
            "diabetes": "Diabetes",
            "high_blood_pressure": "Tekanan Darah Tinggi",
            "smoking": "Merokok",
            "creatinine_phosphokinase": "Creatinine Phosphokinase (CPK)",
            "ejection_fraction": "Ejection Fraction (%)",
            "serum_creatinine": "Serum Creatinine",
            "serum_sodium": "Serum Sodium",
            "platelets": "Platelets",
            "time": "Follow-up Time (days)",
            "anaemia": "Anaemia",
        }

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📋 Input Data Pasien")

        # layout 3 kolom seperti contoh
        c1, c2, c3 = st.columns(3)
        input_dict = {}

        # ========= kolom 1 =========
        with c1:
            if "age" in feature_cols:
                input_dict["age"] = st.number_input(label_map["age"], 0.0, 120.0, default_val("age"), 1.0)

            if "creatinine_phosphokinase" in feature_cols:
                input_dict["creatinine_phosphokinase"] = st.number_input(
                    label_map["creatinine_phosphokinase"], 0.0, value=default_val("creatinine_phosphokinase"), step=1.0
                )

            if "ejection_fraction" in feature_cols:
                input_dict["ejection_fraction"] = st.number_input(
                    label_map["ejection_fraction"], 0.0, 100.0, default_val("ejection_fraction"), 1.0
                )

        # ========= kolom 2 =========
        with c2:
            if "sex" in feature_cols:
                sex_opt = st.selectbox(label_map["sex"], ["Female", "Male"], index=1)
                input_dict["sex"] = 1.0 if sex_opt == "Male" else 0.0

            if "serum_creatinine" in feature_cols:
                input_dict["serum_creatinine"] = st.number_input(
                    label_map["serum_creatinine"], 0.0, value=default_val("serum_creatinine"), step=0.1
                )

            if "serum_sodium" in feature_cols:
                input_dict["serum_sodium"] = st.number_input(
                    label_map["serum_sodium"], 0.0, value=default_val("serum_sodium"), step=1.0
                )

        # ========= kolom 3 =========
        with c3:
            # indikator komorbid biner
            for bin_col in ["anaemia", "diabetes", "high_blood_pressure", "smoking"]:
                if bin_col in feature_cols:
                    v = st.selectbox(label_map.get(bin_col, bin_col), ["Tidak", "Ya"], index=int(round(default_val(bin_col))))
                    input_dict[bin_col] = 1.0 if v == "Ya" else 0.0

            if "platelets" in feature_cols:
                input_dict["platelets"] = st.number_input(
                    label_map["platelets"], 0.0, value=default_val("platelets"), step=1000.0
                )

        # full width bawah (misal time)
        if "time" in feature_cols:
            input_dict["time"] = st.number_input(label_map["time"], 0.0, value=default_val("time"), step=1.0)

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        run = st.button("🔍 Prediksi Risiko", use_container_width=True)

        if run:
            # pastikan semua fitur ada
            for col in feature_cols:
                if col not in input_dict:
                    input_dict[col] = default_val(col)

            X_new = pd.DataFrame([input_dict], columns=feature_cols)

            pred = int(pipe.predict(X_new)[0])

            proba = None
            if hasattr(pipe, "predict_proba"):
                try:
                    proba = float(pipe.predict_proba(X_new)[:, 1][0])
                except Exception:
                    proba = None

            if pred == 1:
                st.error("⚠️ Prediksi: Risiko Tinggi (kelas 1)")
            else:
                st.success("✅ Prediksi: Risiko Rendah (kelas 0)")

            if proba is not None:
                st.info(f"Probabilitas kelas 1 (risiko tinggi): {proba:.3f}")

            st.caption("catatan: ini prediksi model dari pola data training, bukan diagnosis medis.")
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
