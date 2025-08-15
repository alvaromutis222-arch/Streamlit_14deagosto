# app.py — EDA con IA generativa (heurística) en Streamlit
# Ejecuta con:  streamlit run app.py

import io
import re
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# -----------------------------------------
# Configuración general de la página
# -----------------------------------------
st.set_page_config(
    page_title="EDA + IA desde GitHub",
    page_icon="🤖",
    layout="wide",
)
alt.data_transformers.disable_max_rows()

st.title("🤖 Exploración de Datos (EDA) + IA generativa — desde GitHub o archivo")
st.write(
    "Carga un dataset y haz preguntas en lenguaje natural (español). También obtén sugerencias sobre variables más representativas."
)

# -----------------------------------------
# Helpers de carga
# -----------------------------------------
RAW_PATTERNS = [
    (
        r"https?://github\.com/([^/]+)/([^/]+)/blob/([^/]+)/(.+)",
        r"https://raw.githubusercontent.com/\1/\2/\3/\4",
    ),
]


def github_to_raw(url: str) -> str:
    if not url:
        return url
    if "raw.githubusercontent.com" in url:
        return url
    for pat, repl in RAW_PATTERNS:
        if re.match(pat, url):
            return re.sub(pat, repl, url)
    return url


@st.cache_data(show_spinner=False)
def load_data_from_text(content: str, sep: str, header: bool) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(content), sep=sep, header=0 if header else None)


@st.cache_data(show_spinner=True)
def load_data_from_url(url: str, sep: str, header: bool) -> pd.DataFrame:
    raw_url = github_to_raw(url)
    df = pd.read_csv(raw_url, sep=sep, header=0 if header else None)
    return df


def split_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    cnum = df.select_dtypes(include=[np.number]).columns.tolist()
    ccat = df.select_dtypes(exclude=[np.number]).columns.tolist()
    return cnum, ccat

# --------------------------
# Sidebar: entrada de datos
# --------------------------
st.sidebar.header("1) Fuente de datos")
EXAMPLE_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"
source = st.sidebar.radio("Elige cómo cargar los datos", ("Enlace de GitHub", "Subir archivo"), index=0)
if st.sidebar.button("Usar dataset de ejemplo (penguins)"):
    st.session_state["example_url"] = EXAMPLE_URL
if "example_url" in st.session_state:
    st.sidebar.info(f"URL ejemplo: {st.session_state['example_url']}")
sep = st.sidebar.selectbox("Separador", options=[",", ";", "\t"], format_func=lambda x:{',':'Coma (,)', ';':'Punto y coma (;)', '\t':'Tabulador (TAB)'}[x])
use_header = st.sidebar.checkbox("Primera fila es encabezado", value=True)

uploaded_df = None
load_error = None

if source == "Enlace de GitHub":
    gh_url_default = st.session_state.get("example_url", "")
    gh_url = st.sidebar.text_input(
        "Pega la URL del archivo (GitHub: blob o RAW)",
        placeholder="https://github.com/usuario/repo/blob/main/data.csv",
        value=gh_url_default,
    )
    if gh_url:
        try:
            with st.spinner("Cargando datos desde GitHub..."):
                uploaded_df = load_data_from_url(gh_url.strip(), sep, use_header)
        except Exception as e:
            load_error = str(e)
else:
    uploaded_file = st.sidebar.file_uploader("Sube un archivo CSV/TSV", type=["csv", "tsv"])
    if uploaded_file is not None:
        try:
            content = uploaded_file.read().decode("utf-8", errors="ignore")
            with st.spinner("Leyendo archivo..."):
                uploaded_df = load_data_from_text(content, sep, use_header)
        except Exception as e:
            load_error = str(e)

if load_error:
    st.error(f"No se pudo cargar el dataset: {load_error}")

if uploaded_df is None:
    st.info("Carga un dataset desde la barra lateral para comenzar (o usa el ejemplo).")
    st.stop()

# --------------------------
# Vista general
# --------------------------
df = uploaded_df.copy()
cnum, ccat = split_columns(df)

st.markdown("---")
st.header("2) Vista general del dataset")
colA, colB, colC, colD = st.columns([1.2,1.2,1,1])
with colA:
    st.metric("Filas", df.shape[0])
with colB:
    st.metric("Columnas", df.shape[1])
with colC:
    st.metric("Numéricas", len(cnum))
with colD:
    st.metric("Categóricas", len(ccat))

st.subheader("Muestra (primeras filas)")
st.dataframe(df.head(12), use_container_width=True)

with st.expander("Tipos de datos y nulos"):
    dtypes = pd.DataFrame({"columna": df.columns, "dtype": df.dtypes.astype(str), "nulos": df.isna().sum()})
    st.dataframe(dtypes, use_container_width=True)

# --------------------------
# Visualizaciones rápidas
# --------------------------
st.markdown("---")
st.header("3) Visualizaciones rápidas")

c1, c2 = st.columns(2)
with c1:
    if ccat:
        cat_col = st.selectbox("Categórica para conteos", options=ccat)
        top_n = st.slider("Top categorías", 5, 50, 15)
        freq = (
            df[cat_col].astype("string").value_counts(dropna=False).reset_index()
            .rename(columns={"index": cat_col, cat_col: "conteo"})
            .head(top_n)
        )
        st.altair_chart(
            alt.Chart(freq).mark_bar().encode(
                x=alt.X("conteo:Q", title="Conteo"),
                y=alt.Y(f"{cat_col}:N", sort='-x', title=cat_col),
                tooltip=[cat_col, "conteo"],
            ).properties(height=380),
            use_container_width=True,
        )
with c2:
    if cnum:
        num_col = st.selectbox("Numérica para histograma", options=cnum)
        bins = st.slider("Bins", 5, 100, 30)
        st.altair_chart(
            alt.Chart(df).mark_bar().encode(
                x=alt.X(f"{num_col}:Q", bin=alt.Bin(maxbins=bins), title=num_col),
                y=alt.Y("count():Q", title="Conteo"),
                tooltip=[alt.Tooltip(f"{num_col}:Q", title=num_col), alt.Tooltip("count():Q", title="conteo")],
            ).properties(height=380),
            use_container_width=True,
        )

# --------------------------
# ⚡ IA generativa (heurística)
# --------------------------
# En esta demo no usamos un LLM externo: entendemos tu pregunta con reglas sencillas y
# generamos una respuesta apoyada en estadísticas, correlaciones y (si eliges target)
# importancia de variables.

st.markdown("---")
st.header("4) Asistente IA: pregunta en español sobre tus datos")

# — Target opcional para relevancia de variables
left, right = st.columns([1.2, 1])
with left:
    target = st.selectbox("(Opcional) Selecciona una variable objetivo para evaluar 'las más representativas'", options=["(sin objetivo)"] + list(df.columns), index=0)
with right:
    top_k = st.slider("Top variables", 3, min(20, max(3, len(df.columns)-1)), 8)

# Funciones de soporte para "variables más representativas"

def _encode_categoricals(X: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    encoders = {}
    Xc = X.copy()
    for col in Xc.columns:
        if Xc[col].dtype == object or str(Xc[col].dtype).startswith("string"):
            le = LabelEncoder()
            Xc[col] = le.fit_transform(Xc[col].astype(str).fillna("__NA__"))
            encoders[col] = le
    return Xc, encoders


def feature_relevance(df: pd.DataFrame, target: str, k: int) -> pd.DataFrame:
    cols = [c for c in df.columns if c != target]
    if len(cols) == 0:
        return pd.DataFrame()
    y = df[target]
    X = df[cols]
    # codificar categóricas
    X_enc, _ = _encode_categoricals(X)
    # decidir tipo de objetivo
    if y.dtype.kind in 'ifu' and y.nunique() > 15:
        # regresión
        y_clean = y.astype(float)
        y_clean = y_clean.fillna(y_clean.median())
        X_enc = X_enc.fillna(X_enc.median(numeric_only=True))
        try:
            mi = mutual_info_regression(X_enc, y_clean, random_state=0)
            df_mi = pd.DataFrame({"variable": X_enc.columns, "score": mi, "metrica": "MI (reg)"})
        except Exception:
            df_mi = pd.DataFrame(columns=["variable", "score", "metrica"])        
        try:
            rf = RandomForestRegressor(n_estimators=300, random_state=0)
            rf.fit(X_enc, y_clean)
            imp = rf.feature_importances_
            df_rf = pd.DataFrame({"variable": X_enc.columns, "score": imp, "metrica": "RF (imp)"})
        except Exception:
            df_rf = pd.DataFrame(columns=["variable", "score", "metrica"])        
    else:
        # clasificación
        y_enc = y.astype(str).fillna("__NA__")
        le_y = LabelEncoder()
        y_enc = le_y.fit_transform(y_enc)
        X_enc = X_enc.fillna(X_enc.median(numeric_only=True))
        try:
            mi = mutual_info_classif(X_enc, y_enc, random_state=0)
            df_mi = pd.DataFrame({"variable": X_enc.columns, "score": mi, "metrica": "MI (clf)"})
        except Exception:
            df_mi = pd.DataFrame(columns=["variable", "score", "metrica"])        
        try:
            rf = RandomForestClassifier(n_estimators=300, random_state=0)
            rf.fit(X_enc, y_enc)
            imp = rf.feature_importances_
            df_rf = pd.DataFrame({"variable": X_enc.columns, "score": imp, "metrica": "RF (imp)"})
        except Exception:
            df_rf = pd.DataFrame(columns=["variable", "score", "metrica"])        
    out = pd.concat([df_mi, df_rf], axis=0, ignore_index=True)
    if out.empty:
        return out
    out = out.groupby("variable", as_index=False)["score"].mean().sort_values("score", ascending=False)
    return out.head(k)

# Parser sencillo de preguntas (ES)

def answer_question(question: str, df: pd.DataFrame, cnum: List[str], ccat: List[str]) -> str:
    q = (question or "").strip().lower()
    if not q:
        return "Escribe una pregunta como: '¿Cuáles son las variables más representativas de species?' o '¿Cuánta cantidad hay de la variable island?'."

    # ¿Cantidad/conteos?
    if any(k in q for k in ["cuánto", "cuanta", "cuánta", "cantidad", "conteo", "cuantos", "cuántos", "frecuencia"]):
        # detectar columna
        # patrones tipo "de <col>" o solo nombre de columna
        target_col = None
        for col in df.columns:
            if f" {col.lower()}" in f" {q} ":
                target_col = col
                break
        if target_col is None:
            return "¿De qué columna quieres el conteo? Ejemplo: 'cantidad de species'"
        vc = df[target_col].astype("string").value_counts(dropna=False)
        top_items = vc.head(10)
        total = int(vc.sum())
        parts = [f"Total de registros: {total}"] + [f"• {idx}: {int(cnt)}" for idx, cnt in top_items.items()]
        extra = " (mostrando top 10)" if len(vc) > 10 else ""
        return f"Conteo por '{target_col}'{extra}:\n" + "\n".join(parts)

    # ¿Variables más representativas?
    if any(k in q for k in ["representativas", "importantes", "relevantes", "influyentes", "explicativas"]):
        # intentar detectar objetivo mencionada en la pregunta
        tgt = None
        for col in df.columns:
            if f" {col.lower()}" in f" {q} ":
                tgt = col
                break
        if tgt is None:
            return "¿Para qué variable objetivo? Ej.: 'variables más representativas de species'"
        topk = feature_relevance(df, tgt, k=min(10, max(3, len(df.columns)-1)))
        if topk.empty:
            return "No pude estimar relevancias (quizá hay muy pocos datos o todos son nulos)."
        lines = [f"Top variables que mejor explican '{tgt}' (promedio de MI y RF):"]
        for i, row in topk.iterrows():
            lines.append(f"{i+1}. {row['variable']} — score≈{row['score']:.3f}")
        return "\n".join(lines)

    # ¿Resumen descriptivo?
    if any(k in q for k in ["resumen", "describe", "estadística", "estadisticas", "estadístico"]):
        parts = [f"Filas: {df.shape[0]}", f"Columnas: {df.shape[1]}", f"Numéricas: {len(cnum)}", f"Categóricas: {len(ccat)}"]
        if cnum:
            desc = df[cnum].describe().T
            top = desc[["mean", "std", "min", "max"]].round(3).head(5)
            parts.append("Top numéricas (media±std, min–max):")
            for idx, r in top.iterrows():
                parts.append(f"• {idx}: {r['mean']}±{r['std']} ({r['min']}–{r['max']})")
        return "\n".join(parts)

    # Si no entendemos, ofrecer ayuda
    return (
        "No entendí la intención. Prueba con ejemplos: \n"
        "- '¿Cantidad de species?'\n"
        "- 'Variables más representativas de species'\n"
        "- 'Dame un resumen estadístico'"
    )

# UI del asistente
qcol, bcol = st.columns([2, 1])
user_q = qcol.text_input("Tu pregunta en lenguaje natural")
if bcol.button("Responder con IA"):
    st.write("🧠 **Respuesta**:")
    st.code(answer_question(user_q, df, cnum, ccat), language="markdown")

st.divider()

# Bloque: descubrir variables representativas con selector explícito
if target != "(sin objetivo)":
    st.subheader("Variables más representativas (según objetivo seleccionado)")
    rel = feature_relevance(df, target, k=top_k)
    if rel.empty:
        st.info("No fue posible calcular relevancias. Revisa nulos o elige otra columna objetivo.")
    else:
        st.dataframe(rel, use_container_width=True)
        st.altair_chart(
            alt.Chart(rel).mark_bar().encode(
                x=alt.X("score:Q", title="Score medio (MI + RF)"),
                y=alt.Y("variable:N", sort='-x', title="Variable"),
                tooltip=["variable", alt.Tooltip("score:Q", format=".3f")],
            ).properties(height=420),
            use_container_width=True,
        )

st.caption("Nota: La 'IA generativa' de esta demo es determinista y se basa en reglas/estadísticas locales. Si quieres conectar un LLM real (OpenAI, etc.), puedo dejar hooks listos para tu API.")
