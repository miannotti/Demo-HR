import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# Título de la aplicación
st.title("Evaluación de Candidatos y Probabilidad de Contratación")

# Carga y cacheo del modelo entrenado
@st.cache(allow_output_mutation=True)
def load_model():
    return joblib.load('modelo_entrenado.pkl')

model = load_model()

# Subida o carga del nuevo dataset
uploaded_file = st.file_uploader("Sube tu nuevo dataset (CSV)", type=['csv'])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    st.warning("Por favor, sube un archivo CSV para continuar.")
    st.stop()

# Definición de las columnas utilizadas por el modelo
features = [
    'educacion',
    'experiencia_anios',
    'experiencia_sector',
    'certificaciones',
    'puntaje_test',
    'puntaje_entrevista',
    'nivel_ingles',
    'referencia_interna'
]

# Verificación de columnas necesarias
def validate_features(df, features):
    missing = [col for col in features if col not in df.columns]
    if missing:
        st.error(f"El dataset debe contener las columnas: {', '.join(missing)}")
        st.stop()

validate_features(df, features)

# Generación de predicciones y probabilidades
X = df[features]
df['prediccion_contratacion'] = model.predict(X)
df['probabilidad_contratacion'] = model.predict_proba(X)[:, 1]

# Mostrar tabla con todas las predicciones
st.subheader("Resultados de Predicciones")
st.dataframe(df)

# Selección de vacante para el análisis
vacantes = df['vacante_id'].unique().tolist()
vacante_seleccionada = st.selectbox("Selecciona Vacante", vacantes)

# Filtrado y ordenación de candidatos por vacante seleccionada
df_filtrado = df[df['vacante_id'] == vacante_seleccionada].sort_values(
    'probabilidad_contratacion', ascending=False
)

# Gráfico interactivo de probabilidad de contratación por candidato
# Este gráfico muestra, para la vacante seleccionada, la probabilidad estimada de contratación de cada candidato
fig = go.Figure()
fig.add_trace(
    go.Bar(
        x=df_filtrado.index.astype(str),
        y=df_filtrado['probabilidad_contratacion'],
        hovertemplate=
            "<b>Candidato</b>: %{x}<br>" +
            "<b>Probabilidad</b>: %{y:.2%}<br>" +
            "<b>Educación</b>: %{customdata[0]}<br>" +
            "<b>Años de Experiencia</b>: %{customdata[1]}<br>" +
            "<b>Sector</b>: %{customdata[2]}<br>" +
            "<extra></extra>",
        customdata=df_filtrado[['educacion','experiencia_anios','experiencia_sector']].values
    )
)
fig.update_layout(
    title=f"Probabilidades de Contratación - Vacante {vacante_seleccionada}",
    xaxis_title="Índice de Candidato",
    yaxis_title="Probabilidad de Contratación",
    margin=dict(t=50)
)

st.plotly_chart(fig)

# Comentario explicativo del gráfico
st.markdown(
    "**Gráfico de Barras**: Muestra la probabilidad de contratación de cada candidato para la vacante seleccionada. "
    "Los candidatos están ordenados de mayor a menor probabilidad, facilitando la identificación de los mejores perfiles."
)

# Visualización de los Top N candidatos
# Esta tabla presenta los mejores candidatos según su probabilidad de contratación
top_n = st.slider(
    "Número de candidatos a mostrar", min_value=1, max_value=len(df_filtrado), value=min(5, len(df_filtrado))
)
st.subheader(f"Top {top_n} candidatos por probabilidad")
st.table(
    df_filtrado.head(top_n)[[
        'probabilidad_contratacion',
        'prediccion_contratacion',
        'educacion',
        'experiencia_anios',
        'experiencia_sector'
    ]]
)

# Comentario explicativo de la tabla
st.markdown(
    "**Tabla de los Top Candidatos**: Presenta los candidatos mejor posicionados según la probabilidad de contratación, "
    "junto con sus características clave."
)
