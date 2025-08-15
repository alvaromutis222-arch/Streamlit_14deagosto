# app.py — EDA "super pro" en Streamlit con interacciones avanzadas
# Ejecuta con:  streamlit run app.py

import io
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# ==========================
# Configuración
# ==========================
st.set_page_config(page_title="EDA Pro desde GitHub", page_icon="🚀", layout="wide")
alt.data_transformers.disable_max_rows()  # permitir datasets medianos

# ==========================
# Utilidades
# ==========================
RAW_PATTERNS = [
    (r"https?://github\.com/([^/]+)/([^/]+)/blob/([^/]+)/(.+)", r"https://raw.githubusercontent.com/\1/\2/\3/\4"),
]

EXAMPLES: Dict[str, str] = {
    "Penguins": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv",
    "Iris": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv",
    "Tips": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv",
    "Titanic": "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
    "Wine Quality (red)": "https://raw.githubusercontent.com/avinashkranjan/Amazing-Python-Scripts/master/wine-quality/winequality-red.csv",
    "Gapminder": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/gapminder.csv",
}


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


# ==========================
# Encabezado
# ==========================
st.title("🚀 EDA Interactivo 'Super Pro'")
st.caption("Carga desde GitHub o archivo, aplica filtros dinámicos, crea gráficos interactivos y guarda tu galería.")

# ==========================
# Sidebar: Carga de datos
# ==========================
st.sidebar.header("1) Fuente de datos")
source = st.sidebar.radio("¿Cómo cargar los datos?", ("Enlace de GitHub", "Subir archivo", "Dataset de ejemplo"), index=2)
sep = st.sidebar.selectbox("Separador", options=[",", ";", "\t"], format_func=lambda x: {',': 'Coma (,)', ';': 'Punto y coma (;)', '\t': 'Tabulador (TAB)'}[x])
use_header = st.sidebar.checkbox("Primera fila es encabezado", value=True)

uploaded_df = None
load_error = None

if source == "Enlace de GitHub":
    gh_url = st.sidebar.text_input("URL de GitHub (normal o RAW)", placeholder="https://github.com/usuario/repo/blob/main/data.csv")
    if gh_url:
        try:
            with st.spinner("Cargando datos desde GitHub..."):
                uploaded_df = load_data_from_url(gh_url.strip(), sep, use_header)
        except Exception as e:
            load_error = str(e)
elif source == "Subir archivo":
    up = st.sidebar.file_uploader("Sube un CSV/TSV", type=["csv", "tsv"])
    if up is not None:
        try:
            content = up.read().decode("utf-8", errors="ignore")
            with st.spinner("Leyendo archivo..."):
                uploaded_df = load_data_from_text(content, sep, use_header)
        except Exception as e:
            load_error = str(e)
else:
    st.sidebar.markdown("**Datasets de ejemplo**")
    chosen = st.sidebar.selectbox("Selecciona uno", options=list(EXAMPLES.keys()), index=0)
    if st.sidebar.button("Cargar ejemplo"):
        st.session_state["_example_url"] = EXAMPLES[chosen]
    if st.session_state.get("_example_url"):
        st.sidebar.code(st.session_state["_example_url"], language="text")
        try:
            with st.spinner("Cargando ejemplo..."):
                uploaded_df = load_data_from_url(st.session_state["_example_url"], sep, use_header)
        except Exception as e:
            load_error = str(e)

if load_error:
    st.sidebar.error(f"No se pudo cargar el dataset: {load_error}")

# ==========================
# Si hay datos -> Paneles
# ==========================
if uploaded_df is None:
    st.info("Carga un dataset desde la barra lateral para comenzar. También puedes elegir un ejemplo.")
    st.stop()

df = uploaded_df.copy()
cnum, ccat = split_columns(df)

# KPIs
st.markdown("---")
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("Filas", f"{df.shape[0]:,}")
with c2:
    st.metric("Columnas", f"{df.shape[1]:,}")
with c3:
    st.metric("Numéricas", len(cnum))
with c4:
    st.metric("Categóricas", len(ccat))
with c5:
    st.metric("Nulos (%)", f"{(df.isna().sum().sum()/(df.size) * 100):.1f}")

# ==========================
# Filtros dinámicos
# ==========================
st.markdown("### 🔎 Filtros dinámicos")
with st.expander("Abrir/ocultar filtros", expanded=False):
    # Elegir columnas a filtrar
    cols_to_filter = st.multiselect("Selecciona columnas para filtrar", options=list(df.columns), default=[])

    # Aplicar filtros
    filtered_df = df.copy()
    for col in cols_to_filter:
        if col in cnum:
            mn, mx = float(np.nanmin(df[col])), float(np.nanmax(df[col]))
            r = st.slider(f"Rango • {col}", min_value=mn, max_value=mx, value=(mn, mx))
            filtered_df = filtered_df[(filtered_df[col] >= r[0]) & (filtered_df[col] <= r[1])]
        else:
            # limitar opciones si hay demasiadas categorías
            vals = pd.Series(filtered_df[col].astype("string").unique()).dropna().tolist()
            if len(vals) > 200:
                vals = vals[:200]
            sel = st.multiselect(f"Valores • {col}", options=vals, default=vals[: min(10, len(vals))])
            if sel:
                filtered_df = filtered_df[filtered_df[col].astype("string").isin(sel)]

st.success(f"Filtrado activo: {filtered_df.shape[0]} filas (de {df.shape[0]})")

# Botones útiles
b1, b2, _ = st.columns([1,1,2])
with b1:
    st.download_button("⬇️ Descargar CSV filtrado", data=filtered_df.to_csv(index=False).encode("utf-8"), file_name="datos_filtrados.csv", mime="text/csv")
with b2:
    if st.button("♻️ Limpiar filtros"):
        st.experimental_rerun()

# ==========================
# Constructor de gráficos (Chart Builder)
# ==========================
st.markdown("---")
st.header("🎨 Constructor de gráficos interactivos")

if "gallery" not in st.session_state:
    st.session_state["gallery"] = []  # almacenará tuplas (titulo, chart_json)

chart_col, preview_col = st.columns([1.2, 1])

with chart_col:
    chart_type = st.selectbox("Tipo de gráfico", [
        "Barras",
        "Histograma",
        "Dispersión",
        "Boxplot",
        "Línea",
        "Mapa de calor (correlación)",
    ])

    title = st.text_input("Título del gráfico", value=f"{chart_type}")

    chart = None

    if chart_type == "Barras":
        if ccat and cnum:
            c_cat = st.selectbox("Categoría", options=ccat)
            c_val = st.selectbox("Valor numérico", options=cnum)
            agg = st.selectbox("Agregación", ["suma", "media", "conteo"], index=0)
            topn = st.slider("Top N", 5, 100, 20)
            df_tmp = filtered_df[[c_cat] + ([c_val] if agg != "conteo" else [])]
            if agg == "suma":
                g = df_tmp.groupby(c_cat, dropna=False)[c_val].sum().reset_index(name="valor")
            elif agg == "media":
                g = df_tmp.groupby(c_cat, dropna=False)[c_val].mean().reset_index(name="valor")
            else:
                g = df_tmp.groupby(c_cat, dropna=False).size().reset_index(name="valor")
            g = g.sort_values("valor", ascending=False).head(topn)
            chart = alt.Chart(g, title=title).mark_bar().encode(
                x=alt.X("valor:Q", title=agg.title()),
                y=alt.Y(f"{c_cat}:N", sort='-x', title=c_cat),
                tooltip=[c_cat, alt.Tooltip("valor:Q", title=agg)],
            ).interactive()
        else:
            st.info("Se requieren columnas categóricas y numéricas.")

    elif chart_type == "Histograma":
        if cnum:
            c_hist = st.selectbox("Columna numérica", options=cnum)
            bins = st.slider("Bins", 5, 100, 30)
            chart = alt.Chart(filtered_df, title=title).mark_bar().encode(
                x=alt.X(f"{c_hist}:Q", bin=alt.Bin(maxbins=bins), title=c_hist),
                y=alt.Y("count():Q", title="Conteo"),
                tooltip=[alt.Tooltip(f"{c_hist}:Q", title=c_hist), alt.Tooltip("count():Q", title="conteo")],
            ).interactive()
        else:
            st.info("No hay columnas numéricas.")

    elif chart_type == "Dispersión":
        if len(cnum) >= 2:
            x = st.selectbox("Eje X", options=cnum)
            y = st.selectbox("Eje Y", options=cnum, index=min(1, len(cnum)-1))
            color = st.selectbox("Color (opcional)", options=["(sin color)"] + ccat, index=0)
            size_opt = st.selectbox("Tamaño (opcional)", options=["(sin tamaño)"] + cnum, index=0)
            base = alt.Chart(filtered_df, title=title).mark_circle(opacity=0.75, color="steelblue" if color == "(sin color)" else None)
            enc = {
                "x": alt.X(f"{x}:Q", title=x),
                "y": alt.Y(f"{y}:Q", title=y),
                "tooltip": [x, y],
            }
            if color != "(sin color)":
                enc["color"] = alt.Color(f"{color}:N", title=color)
            if size_opt != "(sin tamaño)":
                enc["size"] = alt.Size(f"{size_opt}:Q", title=size_opt)
            chart = base.encode(**enc).interactive()
        else:
            st.info("Se requieren al menos dos numéricas.")

    elif chart_type == "Boxplot":
        if cnum and ccat:
            n = st.selectbox("Numérica", options=cnum)
            c = st.selectbox("Categoría", options=ccat)
            chart = alt.Chart(filtered_df, title=title).mark_boxplot().encode(
                x=alt.X(f"{c}:N", title=c),
                y=alt.Y(f"{n}:Q", title=n),
                tooltip=[c, n],
            )
        else:
            st.info("Requiere numéricas y categóricas.")

    elif chart_type == "Línea":
        # Si hay una fecha intenta parsear; si no, permite elegir una numérica como eje X
        dt_candidates = [c for c in filtered_df.columns if np.issubdtype(filtered_df[c].dtype, np.datetime64)]
        if not dt_candidates:
            # Intentar parsear columnas que parecen fecha
            for c in filtered_df.columns:
                if filtered_df[c].dtype == object:
                    try:
                        filtered_df[c] = pd.to_datetime(filtered_df[c])
                    except Exception:
                        pass
            dt_candidates = [c for c in filtered_df.columns if np.issubdtype(filtered_df[c].dtype, np.datetime64)]
        x_col = st.selectbox("Eje X (fecha o numérico)", options=dt_candidates + cnum)
        y_col = st.selectbox("Eje Y (numérico)", options=cnum)
        group = st.selectbox("Agrupar por (opcional)", options=["(sin grupo)"] + ccat, index=0)
        base = alt.Chart(filtered_df, title=title).mark_line(point=True)
        enc = {
            "x": alt.X(f"{x_col}:{'T' if x_col in dt_candidates else 'Q'}", title=x_col),
            "y": alt.Y(f"{y_col}:Q", title=y_col),
            "tooltip": [x_col, y_col],
        }
        if group != "(sin grupo)":
            enc["color"] = alt.Color(f"{group}:N", title=group)
        chart = base.encode(**enc).interactive()

    else:  # Mapa de calor (correlación)
        if len(cnum) >= 2:
            corr = filtered_df[cnum].corr().round(3)
            corr_melt = corr.reset_index().melt(id_vars='index').rename(columns={'index': 'var1', 'variable': 'var2', 'value': 'corr'})
            chart = alt.Chart(corr_melt, title=title).mark_rect().encode(
                x=alt.X('var1:N', title=''),
                y=alt.Y('var2:N', title=''),
                color=alt.Color('corr:Q', scale=alt.Scale(scheme='blueorange', domain=(-1, 1))),
                tooltip=['var1', 'var2', alt.Tooltip('corr:Q', title='corr')],
            ).properties(height=500)
        else:
            st.info("Se requieren al menos dos numéricas.")

    # Mostrar preview y acciones
    if chart is not None:
        with preview_col:
            st.altair_chart(chart, use_container_width=True)
            # Botones de acción
            add, spec = st.columns(2)
            with add:
                if st.button("➕ Agregar a galería"):
                    st.session_state["gallery"].append((title, chart.to_json()))
                    st.success("Gráfico agregado.")
            with spec:
                st.download_button("⬇️ Descargar spec (JSON)", data=chart.to_json().encode("utf-8"), file_name="chart_spec.json", mime="application/json")

# ==========================
# Galería de gráficos
# ==========================
st.markdown("---")
st.header("🖼️ Galería de gráficos")
cols = st.columns(3)
if len(st.session_state["gallery"]) == 0:
    st.info("Aún no has agregado gráficos. Usa el constructor de arriba y pulsa 'Agregar a galería'.")
else:
    for i, (ttl, spec_json) in enumerate(st.session_state["gallery"]):
        with cols[i % 3]:
            st.subheader(ttl)
            st.altair_chart(alt.Chart.from_json(spec_json), use_container_width=True)

cA, cB = st.columns([1,1])
with cA:
    if st.button("🗑️ Limpiar galería"):
        st.session_state["gallery"] = []
        st.experimental_rerun()
with cB:
    if st.button("🔁 Reprocesar (recargar página)"):
        st.experimental_rerun()

# ==========================
# Panel de datos
# ==========================
st.markdown("---")
st.header("📄 Datos (vista filtrada)")
show_rows = st.slider("Filas a mostrar", 5, 200, 20)
st.dataframe(filtered_df.head(show_rows), use_container_width=True)

with st.expander("Tipos de datos y nulos por columna"):
    dtypes = pd.DataFrame({"columna": df.columns, "dtype": df.dtypes.astype(str), "nulos": df.isna().sum()})
    st.dataframe(dtypes, use_container_width=True)

st.caption("Consejo: usa la galería para guardar múltiples vistas y compártalas exportando sus especificaciones JSON.")
