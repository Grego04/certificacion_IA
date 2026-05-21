
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
st.image("data:image/jpeg;base64,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", use_column_width=True, caption="AI-powered student insights")


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
