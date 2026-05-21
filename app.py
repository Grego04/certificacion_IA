
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Custom CSS for modern look (Dark theme, cards, gradients, fonts) ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

html, body, [class*="st-"] {
    font-family: 'Poppins', sans-serif;
}

body {
    background-color: #0E1117;
    color: #FAFAFA;
}

.reportview-container {
    background: #0E1117;
}

.sidebar .sidebar-content {
    background: #1A1A1A;
}

.stApp {
    background-color: #0E1117;
    color: #FAFAFA;
}

/* Main container styling */
.main .block-container {
    padding-top: 2rem;
    padding-right: 2rem;
    padding-left: 2rem;
    padding-bottom: 2rem;
}

/* Hero Section */
.hero-section {
    background: linear-gradient(135deg, #2a0a5e 0%, #4a0d6d 100%);
    padding: 3rem 2rem;
    border-radius: 15px;
    margin-bottom: 2rem;
    color: white;
    text-align: center;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
}

.hero-section h1 {
    font-size: 3.5rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
    background: -webkit-linear-gradient(45deg, #00C9FF, #92FE9D);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-section p {
    font-size: 1.3rem;
    font-weight: 300;
    opacity: 0.9;
    max-width: 800px;
    margin: 0.5rem auto 1.5rem auto;
}

/* Card Styling */
.st-emotion-cache-nahz7x, .st-emotion-cache-1fttfdf, .st-emotion-cache-1v0mbdj, .st-emotion-cache-1w0pm1c, .st-emotion-cache-1jmps3v {
    background-color: #1A1A1A; /* Darker background for containers/cards */
    border-radius: 10px;
    padding: 1.5rem;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    border: 1px solid rgba(255, 255, 255, 0.1);
}

.st-emotion-cache-nahz7x:hover, .st-emotion-cache-1fttfdf:hover, .st-emotion-cache-1v0mbdj:hover, .st-emotion-cache-1w0pm1c:hover, .st-emotion-cache-1jmps3v:hover {
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
    transform: translateY(-2px);
    transition: all 0.3s ease-in-out;
}

/* Input widgets */
.stSlider > div > div > div[data-baseweb="slider"] {
    background: #333333;
}
.stSlider > div > div > div[data-baseweb="slider"] > div {
    background: #00C9FF; /* Slider fill color */
}

.st-emotion-cache-1n1p7o2 .st-emotion-cache-1e5xgrd {
    background-color: #282828; /* Selectbox background */
    color: #FAFAFA;
    border: 1px solid #444444;
}
.st-emotion-cache-1n1p7o2 .st-emotion-cache-1e5xgrd:hover {
    border-color: #00C9FF;
}

/* Button styling */
.stButton>button {
    background-color: #6A1B9A; /* Purple-ish button */
    color: white;
    border-radius: 8px;
    border: none;
    padding: 0.75rem 1.5rem;
    font-size: 1.1rem;
    font-weight: 600;
    transition: background-color 0.3s ease, transform 0.2s ease;
}

.stButton>button:hover {
    background-color: #7B1FA2; /* Darker purple on hover */
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(106, 27, 154, 0.4);
}

/* Metrics */
[data-testid="stMetricValue"] {
    font-size: 2.5rem;
    color: #92FE9D;
}
[data-testid="stMetricLabel"] {
    font-size: 1rem;
    color: #CCCCCC;
}

/* Progress bar */
.st-emotion-cache-1v0mbdj .st-emotion-cache-1c1kky .st-emotion-cache-1678f8v {
    background-color: #333333; /* Progress bar background */
}
.st-emotion-cache-1v0mbdj .st-emotion-cache-1c1kky .st-emotion-cache-1678f8v > div {
    background: linear-gradient(90deg, #FF4B4B 0%, #FFA500 50%, #00C9FF 100%); /* Gradient for progress */
}

/* Footer */
.footer {
    text-align: center;
    padding: 1.5rem;
    margin-top: 3rem;
    border-top: 1px solid rgba(255, 255, 255, 0.1);
    color: #888888;
    font-size: 0.9rem;
}

.section-header {
    font-size: 1.8rem;
    font-weight: 600;
    color: #E0E0E0;
    margin-top: 2rem;
    margin-bottom: 1.5rem;
    border-bottom: 2px solid #333333;
    padding-bottom: 0.5rem;
}

</style>
""", unsafe_allow_html=True)

# =============================
# CONFIGURACIÓN STREAMLIT
# =============================

st.set_page_config(
    page_title="AI Student Burnout Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Hero Section ---
st.markdown("""
<div class="hero-section">
    <h1><span style='font-size: 1.2em;'>🤖</span> AI Student Burnout Predictor <span style='font-size: 1.2em;'>🧠</span></h1>
    <p>Modelo predictivo basado en Machine Learning para detección temprana de burnout académico</p>
</div>
""", unsafe_allow_html=True)

# =============================
# CARGAR MODELO Y SCALER
# =============================

try:
    # Assuming the models are in the same directory as the app.py
    scaler = joblib.load('scaler.pkl')
    model = joblib.load('modelo_burnout.pkl')

except Exception as e:
    st.error(f"Error cargando modelo o scaler: {e}")
    st.stop()

# =============================
# INPUTS FORM
# =============================

st.markdown('<h2 class="section-header">📊 Ingrese los datos del estudiante</h2>', unsafe_allow_html=True)

# Image for visual appeal - Replaced broken link with a placeholder that should work.
st.image("https://fotos.perfil.com/2023/06/15/trim/1140/641/hay-tecnicas-para-evitar-el-burnout-estudiantil-1589936.jpg", use_column_width=True, caption="AI-powered student insights")


col_inputs1, col_inputs2, col_inputs3 = st.columns(3)

with col_inputs1:
    st.markdown('### 🧑‍🎓 Información Académica')
    with st.container(border=True):
        age = st.slider(
            "Edad del estudiante",
            18, 35, 21, step=1, help="Edad del estudiante en años."
        )

        academic_year = st.slider(
            "Año académico actual",
            1, 5, 3, step=1, help="Año en curso de la carrera o programa académico."
        )

        study_hours_per_day = st.slider(
            "Horas de estudio por día",
            0.0, 12.0, 5.0, step=0.5, help="Promedio de horas dedicadas al estudio diariamente."
        )

with col_inputs2:
    st.markdown('### 🏋️ Hábitos de Vida')
    with st.container(border=True):
        sleep_hours = st.slider(
            "Horas de sueño por noche",
            0.0, 12.0, 7.0, step=0.5, help="Promedio de horas de sueño por noche."
        )

        physical_activity = st.slider(
            "Nivel de actividad física (0-10)",
            0.0, 10.0, 3.0, step=1.0, help="Nivel percibido de actividad física semanal (0: nada, 10: muy activo)."
        )

        screen_time = st.slider(
            "Horas frente a pantallas (diario)",
            0.0, 14.0, 6.0, step=0.5, help="Horas diarias usando dispositivos con pantalla (ordenador, móvil, TV)."
        )

with col_inputs3:
    st.markdown('### 😟 Factores de Estrés')
    with st.container(border=True):
        internet_usage = st.slider(
            "Uso de internet (horas diarias)",
            0.0, 14.0, 5.0, step=0.5, help="Horas diarias dedicadas al uso de internet (no estudio)."
        )

        exam_pressure = st.slider(
            "Presión académica percibida (0-10)",
            0.0, 10.0, 5.0, step=1.0, help="Nivel de estrés o presión sentido por los exámenes o estudios (0: nulo, 10: extremo)."
        )

        family_expectation = st.slider(
            "Expectativa familiar (0-10)",
            0.0, 10.0, 5.0, step=1.0, help="Nivel de presión o expectativa que siente por parte de su familia (0: nulo, 10: extremo)."
        )

        financial_stress = st.slider(
            "Estrés financiero (0-10)",
            0.0, 10.0, 5.0, step=1.0, help="Nivel de estrés relacionado con la situación económica (0: nulo, 10: extremo)."
        )

        gender = st.selectbox(
            "Género",
            ["Female", "Male", "Other"], help="Género del estudiante."
        )

# =============================
# ONE HOT ENCODING
# =============================

gender_Male = 1 if gender == "Male" else 0
gender_Other = 1 if gender == "Other" else 0

# =============================
# DATAFRAME
# =============================

input_data = pd.DataFrame({
    'age': [age],
    'academic_year': [academic_year],
    'study_hours_per_day': [study_hours_per_day],
    'sleep_hours': [sleep_hours],
    'physical_activity': [physical_activity],
    'screen_time': [screen_time],
    'internet_usage': [internet_usage],
    'exam_pressure': [exam_pressure],
    'family_expectation': [family_expectation],
    'financial_stress': [financial_stress],
    'gender_Male': [gender_Male],
    'gender_Other': [gender_Other]
})

# =============================
# PREDICCIÓN Y RESULTADO
# =============================

st.markdown('<h2 class="section-header">🎯 Resultado de la Predicción</h2>', unsafe_allow_html=True)

if st.button("🚀 Predecir Nivel de Burnout"):
    with st.spinner('Analizando datos y prediciendo burnout...'):
        import time
        time.sleep(2) # Simulate calculation time

        try:
            # Escalar datos
            scaled_data = scaler.transform(input_data)

            # Predicción
            prediction = model.predict(scaled_data)[0]

            # Limitar entre 0 y 1
            prediction = np.clip(prediction, 0, 1)

            st.balloons()

            st.subheader("Nivel de Burnout Predicho")
            
            # Dynamic color for the metric based on prediction
            metric_color = ""
            if prediction <= 0.35:
                metric_color = "green"
            elif prediction < 0.7:
                metric_color = "orange"
            else:
                metric_color = "red"

            st.markdown(f"<h3 style='color:{metric_color}'>Score: {prediction:.2f}</h3>", unsafe_allow_html=True)

            col_result_status, col_result_bar = st.columns([1, 3])

            with col_result_status:
                burnout_level_text = ""
                burnout_emoji = ""
                if prediction <= 0.35:
                    burnout_level_text = "Bajo"
                    burnout_emoji = "🟢"
                    st.success(f"{burnout_emoji} Nivel de burnout {burnout_level_text}")
                elif prediction < 0.7:
                    burnout_level_text = "Moderado"
                    burnout_emoji = "🟡"
                    st.warning(f"{burnout_emoji} Nivel de burnout {burnout_level_text}")
                else:
                    burnout_level_text = "Alto"
                    burnout_emoji = "🔴"
                    st.error(f"{burnout_emoji} Nivel de burnout {burnout_level_text}")

            with col_result_bar:
                st.markdown("### Indicador de Riesgo")
                st.progress(prediction)

            st.markdown("### Interpretación Inteligente")
            if prediction <= 0.35:
                st.info("El estudiante muestra un riesgo **bajo** de burnout. Es importante mantener hábitos saludables y un buen balance académico para preservar su bienestar.")
            elif prediction < 0.7:
                st.warning("El estudiante presenta un riesgo **moderado** de burnout. Se recomienda revisar los hábitos de estudio y descanso, y considerar estrategias de manejo del estrés para evitar un aumento en el riesgo.")
            else:
                st.error("¡Atención! El estudiante tiene un riesgo **alto** de burnout. Es crucial intervenir con apoyo académico y psicológico de inmediato, y ajustar las rutinas para reducir la presión y el agotamiento.")

        except Exception as e:
            st.error(f"Error durante la predicción: {e}")

# =============================
# FACTORES DE INFLUENCIA (Placeholder)
# =============================

st.markdown('<h2 class="section-header">💡 Factores clave de influencia</h2>', unsafe_allow_html=True)

st.info("**Nota:** Esta sección proporcionaría insights sobre los factores que más contribuyen al burnout para este estudiante en particular. Para una implementación completa, se requeriría un análisis de interpretabilidad del modelo (ej. SHAP o LIME).")
st.write("En general, el tiempo de estudio excesivo, la falta de sueño, el alto uso de pantallas y la presión académica suelen ser predictores importantes del burnout estudiantil.")

# =============================
# FOOTER
# =============================

st.markdown("""
<div class="footer">
    ---<br>
    Aplicación desarrollada como proyecto final de Analítica de Datos con IA.
    <br>Desarrollado con ❤️ para la comunidad de estudiantes.
</div>
""", unsafe_allow_html=True)
