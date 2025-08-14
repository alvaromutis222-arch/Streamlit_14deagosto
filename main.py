import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Título de la aplicación
st.title("📊 Análisis Exploratorio de Datos con Streamlit")

# URL del dataset en GitHub (puedes cambiarlo por el tuyo)
dataset_url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"

# Cargar el dataset
@st.cache_data
def load_data(url):
    return pd.read_csv(url)

df = load_data(dataset_url)

# Mostrar el dataframe
st.subheader("📄 Vista previa del dataset")
st.dataframe(df.head())

# Mostrar estadísticas descriptivas
st.subheader("📈 Estadísticas Descriptivas")
st.write(df.describe())

# Selección de columna para gráfico de barras
st.subheader("📊 Gráfico de Barras")
column_bar = st.selectbox("Selecciona una columna categórica para visualizar en barras:", df.select_dtypes(include='object').columns)

if column_bar:
    fig_bar, ax_bar = plt.subplots()
    df[column_bar].value_counts().plot(kind='bar', ax=ax_bar, color='skyblue')
    ax_bar.set_title(f"Distribución de {column_bar}")
    st.pyplot(fig_bar)

# Histograma
st.subheader("📉 Histograma")
column_hist = st.selectbox("Selecciona una columna numérica para visualizar histograma:", df.select_dtypes(include='number').columns)

if column_hist:
    fig_hist, ax_hist = plt.subplots()
    sns.histplot(df[column_hist], kde=True, ax=ax_hist, color='orange')
    ax_hist.set_title(f"Histograma de {column_hist}")
    st.pyplot(fig_hist)

# Gráfico de dispersión
st.subheader("📌 Gráfico de Dispersión")
x_axis = st.selectbox("Eje X:", df.select_dtypes(include='number').columns)
y_axis = st.selectbox("Eje Y:", df.select_dtypes(include='number').columns)

if x_axis and y_axis:
    fig_scatter, ax_scatter = plt.subplots()
    sns.scatterplot(data=df, x=x_axis, y=y_axis, ax=ax_scatter)
    ax_scatter.set_title(f"Dispersión entre {x_axis} y {y_axis}")
    st.pyplot(fig_scatter)

# Pie de página
st.markdown("---")
st.markdown("📁 Dataset usado: [tips.csv](https://github.com/mwaskom/seaborn-data/blob/master/ttream
