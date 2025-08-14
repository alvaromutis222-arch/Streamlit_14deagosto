# app.py — Explorador EDA en Streamlit desde GitHub o archivo local
# Para ejecutar:  streamlit run app.py

import io
import re
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# -----------------------------------------
# Configuración general de la página
# -----------------------------------------
st.set_page_config(
    page_title="EDA rápido desde GitHub",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Exploración de Datos (EDA) desde GitHub o archivo")
st.write(
    "Carga un dataset (CSV/TSV) desde un enlace de GitHub o sube un archivo local y genera visualizaciones interactivas."
)

# -----------------------------------------
# Helpers
# -----------------------------------------
RAW_PATTERNS = [
    # https://github.com/user/repo/blob/branch/path/file.csv -> https://raw.githubusercontent.com/user/repo/branch/path/file.csv
    (
        r"https?://github\.com/([^/]+)/([^/]+)/blob/([^/]+)/(.+)",
        r"https://raw.githubusercontent.com/\1/\2/\3/\4",
    ),
    # https://gitlab.com/user/repo/-/raw/branch/path/file.csv -> dejar igual
]


def github_to_raw(url: str) -> str:
    """Convierte URL estándar de GitHub a raw si aplica; si ya es raw, la devuelve igual."""
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


def numeric_columns(df: pd.DataFrame):
    return df.select_dtypes(include=[np.number]).columns.tolist()


def categorical_columns(df: pd.DataFrame):
    return df.select_dtypes(exclude=[np.number]).columns.tolist()


# -----------------------------------------
# Sidebar: Entrada de datos
# -----------------------------------------
st.sidebar.header("1) Fuente de datos")
source = st.sidebar.radio(
    "Elige cómo cargar los datos",
    ("Enlace de GitHub", "Subir archivo"),
    index=0,
)

sep = st.sidebar.selectbox("Separador", options=[",", ";", "\t"], format_func=lambda x: {',': 'Coma (,)', ';': 'Punto y coma (;)', '\t': 'Tabulador (TAB)'}[x])
use_header = st.sidebar.checkbox("Primera fila es encabezado", value=True)

uploaded_df = None
load_error = None

if source == "Enlace de GitHub":
    gh_url = st.sidebar.text_input(
        "Pega la URL del archivo (GitHub). Se acepta URL normal o RAW.",
        placeholder="https://github.com/usuario/repo/blob/main/data.csv",
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

# -----------------------------------------
# Si hay datos, mostramos EDA
# -----------------------------------------
if uploaded_df is not None:
    df = uploaded_df.copy()

    st.markdown("---")
    st.header("2) Vista general del dataset")

    c1, c2, c3, c4 = st.columns([1.2, 1.2, 1, 1])
    with c1:
        st.metric("Filas", df.shape[0])
    with c2:
        st.metric("Columnas", df.shape[1])
    with c3:
        st.metric("Numéricas", len(numeric_columns(df)))
    with c4:
        st.metric("Categóricas", len(categorical_columns(df)))

    st.subheader("Muestra (primeras filas)")
    n_head = st.slider("Número de filas a mostrar", 5, 50, 10)
    st.dataframe(df.head(n_head), use_container_width=True)

    with st.expander("Tipos de datos"):        
        dtypes = pd.DataFrame({"columna": df.columns, "dtype": df.dtypes.astype(str)})
        st.dataframe(dtypes, use_container_width=True)

    with st.expander("Valores perdidos (por columna)"):
        na_tbl = (
            df.isna()
            .sum()
            .reset_index()
            .rename(columns={"index": "columna", 0: "faltantes"})
        )
        na_tbl["%"] = (na_tbl["faltantes"] / len(df) * 100).round(2)
        st.dataframe(na_tbl.sort_values("%", ascending=False), use_container_width=True)

    cnum = numeric_columns(df)
    ccat = categorical_columns(df)

    if cnum:
        with st.expander("Resumen estadístico (numéricas)"):
            st.dataframe(df[cnum].describe().T, use_container_width=True)

    if ccat:
        with st.expander("Frecuencias (categóricas)"):
            col_sel = st.selectbox("Columna categórica", options=ccat)
            top_n = st.slider("Top categorías", 5, 50, 20)
            freq = (
                df[col_sel]
                .astype("string")
                .value_counts(dropna=False)
                .reset_index()
                .rename(columns={"index": col_sel, col_sel: "conteo"})
                .head(top_n)
            )
            chart = (
                alt.Chart(freq)
                .mark_bar()
                .encode(
                    x=alt.X("conteo:Q", title="Conteo"),
                    y=alt.Y(f"{col_sel}:N", sort='-x', title=col_sel),
                    tooltip=[col_sel, "conteo"],
                )
                .properties(height=400)
            )
            st.altair_chart(chart, use_container_width=True)

    st.markdown("---")
    st.header("3) Visualizaciones")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Barras (agrupación)",
        "Histograma",
        "Dispersión",
        "Boxplot",
        "Correlación",
    ])

    # Barras (agrupación)
    with tab1:
        if ccat and cnum:
            c_cat = st.selectbox("Categoría", options=ccat, key="bar_cat")
            c_val = st.selectbox("Valor numérico", options=cnum, key="bar_val")
            agg = st.selectbox("Agregación", ["suma", "media", "conteo"], index=0)
            top_n_bars = st.slider("Top N (ordenado por valor)", 5, 50, 20)

            df_tmp = df[[c_cat, c_val]].copy()

            if agg == "suma":
                g = df_tmp.groupby(c_cat, dropna=False)[c_val].sum().reset_index(name="valor")
            elif agg == "media":
                g = df_tmp.groupby(c_cat, dropna=False)[c_val].mean().reset_index(name="valor")
            else:
                # conteo por categoría (ignora c_val)
                g = df.groupby(c_cat, dropna=False).size().reset_index(name="valor")

            g = g.sort_values("valor", ascending=False).head(top_n_bars)
            bar = (
                alt.Chart(g)
                .mark_bar()
                .encode(
                    x=alt.X("valor:Q", title=f"{agg.title()}"),
                    y=alt.Y(f"{c_cat}:N", sort='-x', title=c_cat),
                    tooltip=[c_cat, alt.Tooltip("valor:Q", title=agg)],
                )
                .properties(height=450)
            )
            st.altair_chart(bar, use_container_width=True)
        else:
            st.info("Se requieren columnas categóricas y numéricas para este gráfico.")

    # Histograma
    with tab2:
        if cnum:
            c_hist = st.selectbox("Columna numérica", options=cnum, key="hist_num")
            bins = st.slider("Bins", 5, 100, 30)
            hist = (
                alt.Chart(df)
                .mark_bar()
                .encode(
                    x=alt.X(f"{c_hist}:Q", bin=alt.Bin(maxbins=bins), title=c_hist),
                    y=alt.Y("count():Q", title="Conteo"),
                    tooltip=[alt.Tooltip(f"{c_hist}:Q", title=c_hist), alt.Tooltip("count():Q", title="conteo")],
                )
                .properties(height=450)
            )
            st.altair_chart(hist, use_container_width=True)
        else:
            st.info("No hay columnas numéricas para el histograma.")

    # Dispersión
    with tab3:
        if len(cnum) >= 2:
            x = st.selectbox("Eje X", options=cnum, key="scat_x")
            y = st.selectbox("Eje Y", options=cnum, key="scat_y")
            color = st.selectbox("Color (opcional)", options=[None] + ccat, index=0, key="scat_color")
            chart = (
                alt.Chart(df)
                .mark_circle(opacity=0.7)
                .encode(
                    x=alt.X(f"{x}:Q", title=x),
                    y=alt.Y(f"{y}:Q", title=y),
                    color=(alt.Color(f"{color}:N", title=color) if color else alt.value("steelblue")),
                    tooltip=[x, y] + ([color] if color else []),
                )
                .interactive()
                .properties(height=450)
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("Se requieren al menos dos columnas numéricas para la dispersión.")

    # Boxplot
    with tab4:
        if cnum and ccat:
            n = st.selectbox("Numérica", options=cnum, key="box_num")
            c = st.selectbox("Categoría", options=ccat, key="box_cat")
            box = (
                alt.Chart(df)
                .mark_boxplot(outliers=True)
                .encode(
                    x=alt.X(f"{c}:N", title=c),
                    y=alt.Y(f"{n}:Q", title=n),
                    tooltip=[c, n],
                )
                .properties(height=450)
            )
            st.altair_chart(box, use_container_width=True)
        else:
            st.info("Se requieren columnas numéricas y categóricas para el boxplot.")

    # Correlación
    with tab5:
        if len(cnum) >= 2:
            corr = df[cnum].corr(numeric_only=True).round(3)
            corr_melt = (
                corr.reset_index()
                .melt(id_vars='index')
                .rename(columns={'index': 'var1', 'variable': 'var2', 'value': 'corr'})
            )
            heat = (
                alt.Chart(corr_melt)
                .mark_rect()
                .encode(
                    x=alt.X('var1:N', title=''),
                    y=alt.Y('var2:N', title=''),
                    color=alt.Color('corr:Q', scale=alt.Scale(scheme='blueorange', domain=(-1, 1))),
                    tooltip=['var1', 'var2', alt.Tooltip('corr:Q', title='correlación')],
                )
                .properties(height=500)
            )
            st.altair_chart(heat, use_container_width=True)
        else:
            st.info("Se requieren al menos dos columnas numéricas para la correlación.")

    st.markdown("---")
    st.caption("💡 Consejo: guarda el enlace de GitHub en la barra lateral para reutilizar esta app con diferentes datasets.")

else:
    st.info("Carga un dataset desde la barra lateral para comenzar.")
