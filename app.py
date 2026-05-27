import time

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# 1. PAGE CONFIG — debe ser la primera llamada a Streamlit
# ============================================================
st.set_page_config(
    page_title="AI Student Burnout Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# 2. COLUMNAS Y UMBRALES
# ============================================================
EXPECTED_COLS = [
    "age",
    "academic_year",
    "study_hours_per_day",
    "sleep_hours",
    "physical_activity",
    "screen_time",
    "internet_usage",
    "exam_pressure",
    "family_expectation",
    "financial_stress",
    "gender_Male",
    "gender_Other",
]

# Ajusta estos valores según la distribución real de tu modelo.
# Para calcularlos: preds = model.predict(scaler.transform(X_test))
#                   LOW  = np.percentile(preds, 33)
#                   HIGH = np.percentile(preds, 66)
LOW_THRESH  = 0.40
HIGH_THRESH = 0.60

# ============================================================
# 3. CSS / DISEÑO
# ============================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

html, body, [class*="st-"] { font-family: 'Poppins', sans-serif; }
.stApp { background-color: #0E1117; color: #FAFAFA; }
.main .block-container { padding: 2rem; }

/* Hero */
.hero-section {
    background: linear-gradient(135deg, #2a0a5e 0%, #4a0d6d 100%);
    padding: 3rem 2rem;
    border-radius: 15px;
    margin-bottom: 2rem;
    text-align: center;
    box-shadow: 0 10px 30px rgba(0,0,0,.3);
}
.hero-section h1 {
    font-size: 3rem;
    font-weight: 700;
    background: -webkit-linear-gradient(45deg,#00C9FF,#92FE9D);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: .5rem;
}
.hero-section p { font-size: 1.1rem; font-weight: 300; opacity: .9; margin: 0; }

/* Sección headers */
.section-header {
    font-size: 1.5rem;
    font-weight: 600;
    color: #E0E0E0;
    margin: 2rem 0 1rem;
    border-bottom: 2px solid #333;
    padding-bottom: .5rem;
}

/* Score badge */
.score-badge {
    display: inline-block;
    font-size: 3rem;
    font-weight: 700;
    padding: .4rem 1.5rem;
    border-radius: 12px;
    margin-bottom: 1rem;
}
.badge-low  { background: rgba(0,200,100,.15); color: #00C864; border: 1px solid #00C864; }
.badge-mid  { background: rgba(255,165,0,.15);  color: #FFA500; border: 1px solid #FFA500; }
.badge-high { background: rgba(255,75,75,.15);  color: #FF4B4B; border: 1px solid #FF4B4B; }

/* Risk chips */
.risk-chip {
    display: inline-block;
    background: rgba(255,165,0,.15);
    color: #FFA500;
    border: 1px solid rgba(255,165,0,.4);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: .8rem;
    margin: 4px 4px 4px 0;
}
.risk-chip-ok {
    display: inline-block;
    background: rgba(0,200,100,.1);
    color: #00C864;
    border: 1px solid rgba(0,200,100,.3);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: .8rem;
}

/* Buttons */
.stButton > button {
    background-color: #6A1B9A;
    color: white;
    border-radius: 8px;
    border: none;
    padding: .75rem 2rem;
    font-size: 1.05rem;
    font-weight: 600;
    transition: background-color .3s, transform .2s;
}
.stButton > button:hover {
    background-color: #7B1FA2;
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(106,27,154,.4);
}

/* Debug expander */
.stExpander { border: 1px solid #333 !important; border-radius: 8px !important; }

/* Footer */
.footer {
    text-align: center;
    padding: 1.5rem;
    margin-top: 3rem;
    border-top: 1px solid rgba(255,255,255,.1);
    color: #888;
    font-size: .85rem;
}
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# 4. HERO
# ============================================================
st.markdown(
    """
<div class="hero-section">
    <h1>🤖 AI Student Burnout Predictor 🧠</h1>
    <p>Modelo predictivo basado en Machine Learning para detección temprana de burnout académico</p>
</div>
""",
    unsafe_allow_html=True,
)

# ============================================================
# 5. CARGAR MODELO Y SCALER
# ============================================================
@st.cache_resource
def load_model():
    try:
        scaler = joblib.load("scaler.pkl")
        model  = joblib.load("modelo_burnout.pkl")
        return model, scaler
    except FileNotFoundError as e:
        st.error(
            f"❌ No se encontró el archivo del modelo: {e}\n\n"
            "Asegúrate de que **modelo_burnout.pkl** y **scaler.pkl** "
            "estén en el mismo directorio que este script."
        )
        st.stop()


model, scaler = load_model()

# ============================================================
# 6. FORMULARIO DE INPUTS
# ============================================================
st.markdown(
    '<h2 class="section-header">📊 Datos del estudiante</h2>',
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🧑‍🎓 Información académica")
    with st.container(border=True):
        age = st.slider(
            "Edad del estudiante", 18, 35, 21, step=1,
            help="Edad en años.",
        )
        academic_year = st.slider(
            "Año académico actual", 1, 5, 3, step=1,
            help="Año en curso del programa académico.",
        )
        study_hours_per_day = st.slider(
            "Horas de estudio por día", 0.0, 12.0, 5.0, step=0.5,
            help="Promedio de horas dedicadas al estudio diariamente.",
        )

with col2:
    st.markdown("### 🏋️ Hábitos de vida")
    with st.container(border=True):
        sleep_hours = st.slider(
            "Horas de sueño por noche", 0.0, 12.0, 7.0, step=0.5,
            help="Promedio de horas de sueño.",
        )
        physical_activity = st.slider(
            "Actividad física (0–10)", 0.0, 10.0, 3.0, step=1.0,
            help="0 = nada activo · 10 = muy activo.",
        )
        screen_time = st.slider(
            "Horas frente a pantallas", 0.0, 14.0, 6.0, step=0.5,
            help="Horas diarias con cualquier dispositivo.",
        )

with col3:
    st.markdown("### 😟 Factores de estrés")
    with st.container(border=True):
        internet_usage = st.slider(
            "Uso de internet (h/día)", 0.0, 14.0, 5.0, step=0.5,
            help="Horas de internet fuera del estudio.",
        )
        exam_pressure = st.slider(
            "Presión de exámenes (0–10)", 0.0, 10.0, 5.0, step=1.0,
            help="0 = sin presión · 10 = presión extrema.",
        )
        family_expectation = st.slider(
            "Expectativa familiar (0–10)", 0.0, 10.0, 5.0, step=1.0,
            help="Nivel de presión familiar percibida.",
        )
        financial_stress = st.slider(
            "Estrés financiero (0–10)", 0.0, 10.0, 5.0, step=1.0,
            help="0 = sin estrés · 10 = estrés extremo.",
        )
        gender = st.selectbox(
            "Género", ["Female", "Male", "Other"],
            help="Género del estudiante.",
        )

# ============================================================
# 7. ONE-HOT ENCODING  (Female = categoría de referencia)
# ============================================================
gender_Male  = 1 if gender == "Male"  else 0
gender_Other = 1 if gender == "Other" else 0

# ============================================================
# 8. DATAFRAME — orden idéntico al del entrenamiento
# ============================================================
input_data = pd.DataFrame(
    {
        "age":                 [age],
        "academic_year":       [academic_year],
        "study_hours_per_day": [study_hours_per_day],
        "sleep_hours":         [sleep_hours],
        "physical_activity":   [physical_activity],
        "screen_time":         [screen_time],
        "internet_usage":      [internet_usage],
        "exam_pressure":       [exam_pressure],
        "family_expectation":  [family_expectation],
        "financial_stress":    [financial_stress],
        "gender_Male":         [gender_Male],
        "gender_Other":        [gender_Other],
    }
)[EXPECTED_COLS]  # ← garantiza el orden correcto

# ============================================================
# 9. PREDICCIÓN
# ============================================================
st.markdown(
    '<h2 class="section-header">🎯 Resultado de la predicción</h2>',
    unsafe_allow_html=True,
)

if st.button("🚀 Predecir nivel de burnout"):
    with st.spinner("Analizando datos…"):
        time.sleep(1.2)

    try:
        # --- Verificar compatibilidad de columnas con el scaler ---
        if hasattr(scaler, "feature_names_in_"):
            expected = list(scaler.feature_names_in_)
            actual   = list(input_data.columns)
            if expected != actual:
                st.error(
                    "Las columnas no coinciden con las del scaler.\n\n"
                    f"Esperadas: {expected}\n\nRecibidas: {actual}"
                )
                st.stop()

        # --- Escalar y predecir ---
        scaled_data = scaler.transform(input_data)
        prediction  = float(np.clip(model.predict(scaled_data)[0], 0.0, 1.0))

        # --- Determinar nivel ---
        if prediction <= LOW_THRESH:
            level      = "Bajo"
            emoji      = "🟢"
            badge_cls  = "badge-low"
            alert_fn   = st.success
            msg = (
                "El estudiante muestra un riesgo **bajo** de burnout. "
                "Es importante mantener hábitos saludables y el balance académico actual."
            )
        elif prediction <= HIGH_THRESH:
            level      = "Moderado"
            emoji      = "🟡"
            badge_cls  = "badge-mid"
            alert_fn   = st.warning
            msg = (
                "El estudiante presenta un riesgo **moderado** de burnout. "
                "Se recomienda revisar hábitos de estudio y descanso, "
                "y aplicar estrategias de manejo del estrés."
            )
        else:
            level      = "Alto"
            emoji      = "🔴"
            badge_cls  = "badge-high"
            alert_fn   = st.error
            msg = (
                "¡Atención! El estudiante tiene un riesgo **alto** de burnout. "
                "Es crucial brindar apoyo académico y psicológico de inmediato, "
                "y ajustar la rutina para reducir la presión y el agotamiento."
            )

        st.balloons()

        # --- Score badge + barra ---
        res_col1, res_col2 = st.columns([1, 2])
        with res_col1:
            st.markdown(
                f"<div class='score-badge {badge_cls}'>{emoji} {prediction:.2f}</div>"
                f"<p style='color:#CCCCCC;font-size:.95rem;'>Nivel: <strong>{level}</strong></p>",
                unsafe_allow_html=True,
            )
            alert_fn(f"{emoji} Burnout {level}")

        with res_col2:
            st.markdown("#### Indicador de riesgo")
            st.progress(prediction)
            st.markdown(f"<small style='color:#888'>Umbral bajo ≤ {LOW_THRESH} · alto > {HIGH_THRESH}</small>", unsafe_allow_html=True)

        # --- Interpretación ---
        st.markdown("#### 💬 Interpretación")
        alert_fn(msg)

        # --- Factores de riesgo detectados ---
        st.markdown("#### ⚠️ Factores de riesgo detectados")

        risks = []
        if study_hours_per_day >= 9:
            risks.append("Estudio excesivo (≥ 9 h/día)")
        if sleep_hours <= 5:
            risks.append("Pocas horas de sueño (≤ 5 h)")
        if exam_pressure >= 8:
            risks.append("Alta presión de exámenes (≥ 8/10)")
        if financial_stress >= 8:
            risks.append("Estrés financiero alto (≥ 8/10)")
        if family_expectation >= 8:
            risks.append("Expectativa familiar alta (≥ 8/10)")
        if physical_activity <= 2:
            risks.append("Actividad física muy baja (≤ 2/10)")
        if screen_time >= 10:
            risks.append("Tiempo de pantalla excesivo (≥ 10 h)")
        if internet_usage >= 8:
            risks.append("Uso de internet excesivo (≥ 8 h/día)")

        if risks:
            chips = "".join(
                f"<span class='risk-chip'>⚠️ {r}</span>" for r in risks
            )
            st.markdown(chips, unsafe_allow_html=True)
        else:
            st.markdown(
                "<span class='risk-chip-ok'>✅ Sin factores de riesgo extremos detectados</span>",
                unsafe_allow_html=True,
            )

        # --- Debug expander ---
        with st.expander("🔍 Detalle técnico"):
            st.write(f"**Score crudo:** `{prediction:.6f}`")
            st.write(f"**Umbrales:** bajo ≤ {LOW_THRESH} · alto > {HIGH_THRESH}")
            st.markdown("**Features enviadas al modelo:**")
            st.dataframe(input_data, use_container_width=True)
            st.markdown("**Valores escalados:**")
            st.dataframe(
                pd.DataFrame(scaled_data, columns=EXPECTED_COLS),
                use_container_width=True,
            )

    except Exception as e:
        st.error(f"Error durante la predicción: {e}")
        st.info(
            "Revisa que el modelo y el scaler fueron entrenados "
            "con exactamente las mismas columnas que se están enviando."
        )

# ============================================================
# 10. SECCIÓN INFORMATIVA — Factores clave
# ============================================================
st.markdown(
    '<h2 class="section-header">💡 Factores clave de influencia</h2>',
    unsafe_allow_html=True,
)

info_col1, info_col2 = st.columns(2)

with info_col1:
    st.info(
        "**¿Qué factores suelen predecir el burnout?**\n\n"
        "- 📚 Horas de estudio excesivas (> 9 h/día)\n"
        "- 😴 Falta de sueño (< 6 h/noche)\n"
        "- 📱 Alto tiempo de pantalla y uso de internet\n"
        "- 📝 Presión académica y familiar elevada\n"
        "- 💸 Estrés económico sostenido\n"
        "- 🏃 Sedentarismo (poca actividad física)"
    )

with info_col2:
    st.success(
        "**¿Cómo reducir el riesgo?**\n\n"
        "- 🕐 Distribuir el tiempo de estudio con pausas regulares\n"
        "- 🛌 Priorizar 7–9 horas de sueño por noche\n"
        "- 🏋️ Incorporar al menos 30 min de actividad física al día\n"
        "- 📵 Establecer límites de uso de pantallas\n"
        "- 🗣️ Buscar apoyo psicológico ante signos tempranos\n"
        "- 🎯 Gestionar expectativas propias y familiares"
    )

st.caption(
    "**Nota:** Para obtener explicaciones por estudiante (qué features "
    "empujan el score hacia arriba o hacia abajo), considera integrar "
    "**SHAP** o **LIME** en una versión futura del modelo."
)

# ============================================================
# 11. FOOTER
# ============================================================
st.markdown(
    f"""
<div class="footer">
    Aplicación desarrollada como proyecto final de Analítica de Datos con IA.<br>
    Desarrollado con ❤️ para la comunidad de estudiantes.<br>
    <small>Umbrales: bajo ≤ {LOW_THRESH} · moderado ≤ {HIGH_THRESH} · alto > {HIGH_THRESH}</small>
</div>
""",
    unsafe_allow_html=True,
)
