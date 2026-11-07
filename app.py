"""
================================================================================
UNIVERSIDAD DE PAMPLONA - DEPARTAMENTO DE MATEMÁTICAS
TALLER DE PROBABILIDAD Y ESTADÍSTICA (SEGUNDO CORTE)
EJERCICIO 5: LABORATORIO VIRTUAL DE CONFIABILIDAD Y ANÁLISIS DE RIESGOS
================================================================================
"""

import streamlit as st
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd
import base64
import os

# ------------------------------------------------------------------------------
# CONFIGURACIÓN DE PÁGINA Y ESTILO VISUAL PERSONALIZADO (CSS)
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Laboratorio de Confiabilidad y Riesgos",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inicialización de estado para navegación interna a demostraciones PDF
if "ver_demostracion" not in st.session_state:
    st.session_state["ver_demostracion"] = None

# Función auxiliar para renderizar archivos PDF locales en Streamlit con botón de descarga
def mostrar_pdf(ruta_pdf):
    if not os.path.exists(ruta_pdf):
        if ruta_pdf.endswith("_pdf.pdf"):
            alt_ruta = ruta_pdf.replace("_pdf.pdf", ".pdf")
            if os.path.exists(alt_ruta):
                ruta_pdf = alt_ruta

    if os.path.exists(ruta_pdf):
        with open(ruta_pdf, "rb") as f:
            pdf_bytes = f.read()
            base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
        
        # Botón de respaldo para abrir/descargar directamente si el navegador bloquea el iframe
        st.download_button(
            label="📥 Descargar / Abrir documento PDF directamente",
            data=pdf_bytes,
            file_name=os.path.basename(ruta_pdf),
            mime="application/pdf",
            key=f"btn_dl_{ruta_pdf}"
        )
        st.caption("ℹ️ Si tu navegador (Microsoft Edge/Chrome) bloquea la vista previa abajo, usa el botón azul de arriba para abrir el archivo.")

        # Intentar renderizar el visor interactivo
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="850" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)
    else:
        st.error(f"⚠️ No se encontró el archivo `{ruta_pdf}` en el servidor. Verifica que esté subido a GitHub.")

# Estilo CSS inyectado: Borde verde al enfocar/escribir en las entradas
st.markdown("""
<style>
    div[data-baseweb="input"]:focus-within {
        border-color: #10B981 !important;
        box-shadow: 0 0 0 1px #10B981 !important;
    }
</style>
""", unsafe_allow_html=True)

# Estilo gráfico limpio
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# Encabezado principal
st.title("⚙️ Laboratorio Virtual de Confiabilidad y Análisis de Riesgos")
st.caption("Universidad de Pamplona • Taller de Probabilidad y Estadística • Ejercicio 5")
st.markdown("---")

# ------------------------------------------------------------------------------
# MENÚ LATERAL DE NAVEGACIÓN
# ------------------------------------------------------------------------------
st.sidebar.title("Módulos del Sistema")

modulo = st.sidebar.radio(
    "Seleccione el entorno de trabajo:",
    [
        "1. Explorador de Distribuciones (PDF / CDF)",
        "2. Confiabilidad y Supervivencia (R(t) y Tasa de Falla)",
        "3. Análisis de Riesgo y Variabilidad (Beta y Chi²)",
        "4. Comparador de Modelos de Vida Útil",
        "5. Casos Prácticos y Toma de Decisiones",
        "6. Ejercicios Prácticos y Cálculo de Probabilidades",
        "7. Fundamentación Teórica y Fórmulas"
    ]
)

st.sidebar.markdown("---")

# WIDGET EXCLUSIVO DE GUÍA DE USUARIO EN LA BARRA LATERAL
nombre_pdf_guia = "Guía de Usuario - Laboratorio Virtual de Confiabilidad.pdf"

if os.path.exists(nombre_pdf_guia):
    with open(nombre_pdf_guia, "rb") as f:
        pdf_guia_bytes = f.read()
    st.sidebar.download_button(
        label="📖 Guía de Usuario",
        data=pdf_guia_bytes,
        file_name=nombre_pdf_guia,
        mime="application/pdf",
        help="Haz clic para descargar la guía de usuario oficial en PDF."
    )
else:
    st.sidebar.warning("⚠️ Guía PDF no encontrada en el servidor.")

st.sidebar.markdown("---")
st.sidebar.caption("""
**Modelos implementados:**
1. Gamma
2. Exponencial
3. Chi-cuadrada (χ²)
4. Log-Normal
5. Beta
6. Weibull
""")

# ==============================================================================
# MÓDULO 1: EXPLORADOR DE DISTRIBUCIONES (PDF Y CDF)
# ==============================================================================
if modulo == "1. Explorador de Distribuciones (PDF / CDF)":
    st.session_state["ver_demostracion"] = None
    st.header("📈 Módulo 1: Visualizador y Explorador de Distribuciones")
    st.write("Analice interactivamente la **Función de Densidad de Probabilidad (PDF)** y la **Distribución Acumulada (CDF)**.")

    col_ctrl, col_plot = st.columns([16, 17])

    with col_ctrl:
        st.subheader("Configuración")
        dist_choice = st.selectbox(
            "Distribución a examinar:",
            ["Gamma", "Exponencial", "Chi-cuadrada", "Lognormal", "Beta", "Weibull"],
            key="m1_dist_choice"
        )

        if dist_choice == "Gamma":
            st.latex(r"f(x) = \frac{1}{\beta^\alpha \Gamma(\alpha)} x^{\alpha-1} e^{-x/\beta}")
            alpha = st.number_input("Forma α (parámetro a):", min_value=1e-9, max_value=None, value=2.5, step=0.5, format="%g", help="📌 Significado: Número de etapas o fases independientes requeridas para el evento.\n\n🔢 Dominio válido: Debe ser estrictamente mayor que 0.", key="m1_gam_a")
            beta = st.number_input("Escala β (parámetro b):", min_value=1e-9, max_value=None, value=2.0, step=0.2, format="%g", help="📌 Significado: Tiempo medio de duración asignado a cada etapa individual.\n\n🔢 Dominio válido: Debe ser estrictamente mayor que 0.", key="m1_gam_b")
            
            x_max = float(stats.gamma.ppf(0.999, a=alpha, scale=beta))
            x = np.linspace(0.001, x_max, 500)
            pdf = stats.gamma.pdf(x, a=alpha, scale=beta)
            cdf = stats.gamma.cdf(x, a=alpha, scale=beta)
            media = alpha * beta
            varianza = alpha * (beta**2)

        elif dist_choice == "Exponencial":
            st.latex(r"f(x) = \lambda e^{-\lambda x} = \frac{1}{\beta}e^{-x/\beta}")
            param_tipo = st.radio("Entrada:", ["Tasa de falla λ", "Escala β (Media)"], key="m1_exp_tipo")
            if param_tipo == "Tasa de falla λ":
                lam = st.number_input("Tasa λ:", min_value=1e-9, max_value=None, value=0.5, step=0.01, format="%g", help="📌 Significado: Frecuencia promedio con la que ocurren las fallas por unidad de tiempo.\n\n🔢 Dominio válido: Debe ser estrictamente mayor que 0.", key="m1_lam")
                scale = 1.0 / lam
            else:
                scale = st.number_input("Media β:", min_value=1e-9, max_value=None, value=2.0, step=0.1, format="%g", help="📌 Significado: Tiempo medio esperado de ocurrencia de un evento (MTTF).\n\n🔢 Dominio válido: Debe ser estrictamente mayor que 0.", key="m1_scale")
                lam = 1.0 / scale
            
            x_max = float(stats.expon.ppf(0.999, scale=scale))
            x = np.linspace(0.001, x_max, 500)
            pdf = stats.expon.pdf(x, scale=scale)
            cdf = stats.expon.cdf(x, scale=scale)
            media = scale
            varianza = scale**2

        elif dist_choice == "Chi-cuadrada":
            st.latex(r"\chi^2(\nu), \quad E[X] = \nu, \quad V(X) = 2\nu")
            df = st.number_input("Grados de libertad ν = n-1:", min_value=1, max_value=None, value=12, step=1, help="📌 Significado: Cantidad de variables normales independientes sumadas al cuadrado.\n\n🔢 Dominio válido: Debe ser un número entero mayor o igual a 1.", key="m1_chi_df")
            
            x_max = float(stats.chi2.ppf(0.999, df=df))
            x = np.linspace(0.001, x_max, 500)
            pdf = stats.chi2.pdf(x, df=df)
            cdf = stats.chi2.cdf(x, df=df)
            media = float(df)
            varianza = float(2 * df)

        elif dist_choice == "Lognormal":
            st.latex(r"\ln(X) \sim \mathcal{N}(\mu, \sigma^2)")
            mu_ln = st.number_input("Media logarítmica μ:", min_value=None, max_value=None, value=1.2, step=0.1, format="%g", help="📌 Significado: Promedio de los datos transformados mediante el logaritmo natural ln(X).\n\n🔢 Dominio válido: Puede tomar cualquier valor real (-∞, ∞).", key="m1_ln_mu")
            sigma_ln = st.number_input("Desv. estándar logarítmica σ:", min_value=1e-9, max_value=None, value=0.5, step=0.05, format="%g", help="📌 Significado: Dispersión o variabilidad de los datos en la escala logarítmica.\n\n🔢 Dominio válido: Debe ser estrictamente mayor que 0.", key="m1_ln_sig")
            
            x_max = float(stats.lognorm.ppf(0.995, s=sigma_ln, scale=np.exp(mu_ln)))
            x = np.linspace(0.001, x_max, 500)
            pdf = stats.lognorm.pdf(x, s=sigma_ln, scale=np.exp(mu_ln))
            cdf = stats.lognorm.cdf(x, s=sigma_ln, scale=np.exp(mu_ln))
            media = np.exp(mu_ln + (sigma_ln**2)/2.0)
            varianza = np.exp(2*mu_ln + sigma_ln**2) * (np.exp(sigma_ln**2) - 1.0)

        elif dist_choice == "Beta":
            st.latex(r"f(x) = \frac{1}{B(\alpha, \beta)} x^{\alpha-1} (1-x)^{\beta-1}")
            alpha_b = st.number_input("Parámetro α (a):", min_value=1e-9, max_value=None, value=2.0, step=0.5, format="%g", help="📌 Significado: Primer parámetro de forma que controla la densidad cerca del extremo 0.\n\n🔢 Dominio válido: Debe ser estrictamente mayor que 0.", key="m1_beta_a")
            beta_b = st.number_input("Parámetro β (b):", min_value=1e-9, max_value=None, value=5.0, step=0.5, format="%g", help="📌 Significado: Segundo parámetro de forma que controla la densidad cerca del extremo 1.\n\n🔢 Dominio válido: Debe ser estrictamente mayor que 0.", key="m1_beta_b")
            
            x = np.linspace(0.001, 0.999, 500)
            pdf = stats.beta.pdf(x, a=alpha_b, b=beta_b)
            cdf = stats.beta.cdf(x, a=alpha_b, b=beta_b)
            media = alpha_b / (alpha_b + beta_b)
            varianza = (alpha_b * beta_b) / (((alpha_b + beta_b)**2) * (alpha_b + beta_b + 1))

        elif dist_choice == "Weibull":
            st.latex(r"f(x) = \frac{\alpha}{\beta} \left(\frac{x}{\beta}\right)^{\alpha-1} e^{-(x/\beta)^\alpha}")
            forma_w = st.number_input("Parámetro de forma α:", min_value=1e-9, max_value=None, value=2.5, step=0.1, format="%g", help="📌 Significado: Gobierna la tendencia del riesgo (α<1 falla temprana, α=1 aleatoria, α>1 desgaste).\n\n🔢 Dominio válido: Debe ser estrictamente mayor que 0.", key="m1_w_forma")
            escala_w = st.number_input("Escala β:", min_value=1e-9, max_value=None, value=6.0, step=0.5, format="%g", help="📌 Significado: Vida característica en la que falla aproximadamente el 63.2% de la población.\n\n🔢 Dominio válido: Debe ser estrictamente mayor que 0.", key="m1_w_escala")
            
            x_max = float(stats.weibull_min.ppf(0.999, c=forma_w, scale=escala_w))
            x = np.linspace(0.001, x_max, 500)
            pdf = stats.weibull_min.pdf(x, c=forma_w, scale=escala_w)
            cdf = stats.weibull_min.cdf(x, c=forma_w, scale=escala_w)
            media = stats.weibull_min.mean(c=forma_w, scale=escala_w)
            varianza = stats.weibull_min.var(c=forma_w, scale=escala_w)

        st.markdown("---")
        with st.container(border=True):
            st.markdown("**📊 Métricas Calculadas:**")
            st.markdown(f"- **Media E[X]:** `{media:g}`")
            st.markdown(f"- **Varianza V(X):** `{varianza:g}`")
            st.markdown(f"- **Desviación Estándar σ:** `{np.sqrt(varianza):g}`")

    with col_plot:
        with st.container():
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7.5))
            plt.subplots_adjust(hspace=0.35)

            ax1.plot(x, pdf, color="#1E3A8A", lw=2.5, label="PDF: f(x)")
            ax1.fill_between(x, pdf, color="#3B82F6", alpha=0.25)
            ax1.axvline(media, color="#DC2626", linestyle="--", lw=1.8, label=f"E[X] = {media:g}")
            ax1.set_title(f"Densidad de Probabilidad (PDF) - {dist_choice}", fontweight="bold")
            ax1.set_xlabel("x")
            ax1.set_ylabel("f(x)")
            ax1.grid(True, linestyle=":", alpha=0.6)
            ax1.legend(loc="upper right")

            ax2.plot(x, cdf, color="#059669", lw=2.5, label="CDF: F(x) = P(X ≤ x)")
            ax2.set_title(f"Distribución Acumulada (CDF) - {dist_choice}", fontweight="bold")
            ax2.set_xlabel("x")
            ax2.set_ylabel("F(x)")
            ax2.set_ylim(-0.02, 1.02)
            ax2.grid(True, linestyle=":", alpha=0.6)
            ax2.legend(loc="lower right")

            st.pyplot(fig, clear_figure=True)
            plt.close(fig)

# ==============================================================================
# MÓDULO 2: CONFIABILIDAD Y SUPERVIVENCIA
# ==============================================================================
elif modulo == "2. Confiabilidad y Supervivencia (R(t) y Tasa de Falla)":
    st.session_state["ver_demostracion"] = None
    st.header("🛡️ Módulo 2: Confiabilidad y Tasa de Falla")
    st.write("""
    Evalúe el desempeño temporal y la vida útil de componentes mecánicos o electrónicos:
    - **Función de Supervivencia / Confiabilidad:** R(t) = P(T > t) = 1 - F(t)
    - **Función de Tasa de Falla (Hazard Rate):** h(t) = λ(t) = f(t) / R(t)
    """)

    c1, c2 = st.columns([16, 17])

    with c1:
        dist_rel = st.selectbox(
            "Distribución para confiabilidad:",
            ["Gamma (Fallas por etapas)", "Exponencial (Tasa de falla constante)", "Lognormal (Fatiga cíclica)", "Weibull (Desgaste / Fatiga)"],
            key="m2_dist_rel"
        )

        t_mision = st.number_input("Tiempo de misión t (horas):", min_value=1e-9, max_value=None, value=1200.0, step=100.0, format="%g", help="📌 Significado: Duración de trabajo exigida sobre la cual se evalúa la probabilidad de supervivencia.\n\n🔢 Dominio válido: Debe ser estrictamente mayor que 0.", key="m2_t_mision")

        if "Gamma" in dist_rel:
            alpha_g = st.number_input("Etapas de degradación α:", min_value=1e-9, max_value=None, value=3.0, step=0.5, format="%g", help="📌 Significado: Cantidad de etapas parciales de deterioro requeridas para el fallo total.\n\n🔢 Dominio válido: Debe ser estrictamente mayor que 0.", key="m2_ag")
            beta_g = st.number_input("Escala de etapa β (horas):", min_value=1e-9, max_value=None, value=1000.0, step=50.0, format="%g", help="📌 Significado: Tiempo medio característico transcurrido en cada etapa de degradación.\n\n🔢 Dominio válido: Debe ser estrictamente mayor que 0.", key="m2_bg")
            dist_obj = stats.gamma(a=alpha_g, scale=beta_g)

        elif "Exponencial" in dist_rel:
            mttf = st.number_input("Tiempo Medio de Falla MTTF (horas):", min_value=1e-9, max_value=None, value=3000.0, step=200.0, format="%g", help="📌 Significado: Promedio esperado de horas de funcionamiento continuo antes de una avería.\n\n🔢 Dominio válido: Debe ser estrictamente mayor que 0.", key="m2_mttf")
            dist_obj = stats.expon(scale=mttf)

        elif "Lognormal" in dist_rel:
            mu_l = st.number_input("Media logarítmica μ:", min_value=None, max_value=None, value=7.8, step=0.1, format="%g", help="📌 Significado: Media del tiempo de vida expresada en escala logarítmica ln(T).\n\n🔢 Dominio válido: Puede tomar cualquier valor real (-∞, ∞).", key="m2_mul")
            sigma_l = st.number_input("Dispersión logarítmica σ:", min_value=1e-9, max_value=None, value=0.5, step=0.05, format="%g", help="📌 Significado: Nivel de dispersión de fallas debidas a fatiga o estrés operativo.\n\n🔢 Dominio válido: Debe ser estrictamente mayor que 0.", key="m2_sigl")
            dist_obj = stats.lognorm(s=sigma_l, scale=np.exp(mu_l))

        elif "Weibull" in dist_rel:
            beta_w = st.number_input("Parámetro de forma α:", min_value=1e-9, max_value=None, value=2.5, step=0.1, format="%g", help="📌 Significado: Comportamiento y evolución del riesgo de falla acumulado en el tiempo.\n\n🔢 Dominio válido: Debe ser estrictamente mayor que 0.", key="m2_bw")
            eta_w = st.number_input("Vida característica β (horas):", min_value=1e-9, max_value=None, value=3500.0, step=200.0, format="%g", help="📌 Significado: Escala temporal representativa de la vida esperada del equipo o componente.\n\n🔢 Dominio válido: Debe ser estrictamente mayor que 0.", key="m2_ew")
            dist_obj = stats.weibull_min(c=beta_w, scale=eta_w)
            
            if beta_w < 1.0:
                st.info("⚠️ Forma α < 1: Tasa de falla decreciente (Mortalidad infantil / defectos de fábrica).")
            elif np.isclose(beta_w, 1.0, atol=0.08):
                st.info("ℹ️ Forma α = 1: Tasa de falla constante (Fallas aleatorias / sin memoria).")
            else:
                st.info("⚙️ Forma α > 1: Tasa de falla creciente (Desgaste físico, fatiga y envejecimiento).")

        r_mision = float(1.0 - dist_obj.cdf(t_mision))
        f_mision = float(dist_obj.cdf(t_mision))
        mttf_calc = float(dist_obj.mean())

        st.markdown("---")
        st.metric(label=f"Confiabilidad R({t_mision:g}h)", value=f"{r_mision*100:g}%")
        st.metric(label=f"Probabilidad de Falla F({t_mision:g}h)", value=f"{f_mision*100:g}%")
        st.metric(label="Vida Media MTTF (E[T])", value=f"{mttf_calc:g} h")

        if r_mision >= 0.90:
            st.success("✅ Nivel de Confiabilidad Alto (R ≥ 90%): Apto para servicio continuo.")
        elif r_mision >= 0.70:
            st.warning("⚠️ Confiabilidad Moderada (70% ≤ R < 90%): Programar inspección preventiva.")
        else:
            st.error("🚨 Alto Riesgo de Falla (R < 70%): Reemplazo mandatario.")

    with c2:
        with st.container():
            t_max_plot = max(t_mision * 2.2, mttf_calc * 1.8)
            t_vals = np.linspace(0.01, t_max_plot, 400)
            
            r_vals = 1.0 - dist_obj.cdf(t_vals)
            f_vals = dist_obj.pdf(t_vals)
            h_vals = np.divide(f_vals, np.where(r_vals > 1e-7, r_vals, np.nan))

            fig, (ax_r, ax_h) = plt.subplots(2, 1, figsize=(8, 7.5))
            plt.subplots_adjust(hspace=0.35)

            ax_r.plot(t_vals, r_vals, color="#1E3A8A", lw=2.5, label="Supervivencia R(t)")
            ax_r.axvline(t_mision, color="#DC2626", linestyle=":", lw=2, label=f"Misión {t_mision:g}h (R={r_mision*100:g}%)")
            ax_r.scatter([t_mision], [r_mision], color="#DC2626", s=60, zorder=5)
            ax_r.set_title("Función de Confiabilidad R(t) = P(T > t)", fontweight="bold")
            ax_r.set_xlabel("Tiempo t (horas)")
            ax_r.set_ylabel("R(t)")
            ax_r.set_ylim(-0.02, 1.02)
            ax_r.grid(True, linestyle=":", alpha=0.6)
            ax_r.legend()

            ax_h.plot(t_vals, h_vals, color="#D97706", lw=2.5, label="Tasa de Falla h(t) = f(t)/R(t)")
            ax_h.set_title("Función de Riesgo / Tasa de Falla Instantánea h(t)", fontweight="bold")
            ax_h.set_xlabel("Tiempo t (horas)")
            ax_h.set_ylabel("Tasa de Falla h(t) [fallas/hora]")
            ax_h.grid(True, linestyle=":", alpha=0.6)
            ax_h.legend()

            st.pyplot(fig, clear_figure=True)
            plt.close(fig)

# ==============================================================================
# MÓDULO 3: ANÁLISIS DE RIESGO Y VARIABILIDAD (BETA Y CHI²)
# ==============================================================================
elif modulo == "3. Análisis de Riesgo y Variabilidad (Beta y Chi²)":
    st.session_state["ver_demostracion"] = None
    st.header("🎲 Módulo 3: Análisis de Riesgo, Proporciones y Variabilidad")
    st.write("Herramientas para modelado de proporciones de falla y control de variabilidad muestral:")

    opcion_riesgo = st.radio("Seleccione el análisis:", ["Distribución Chi-Cuadrada (Control de Variabilidad Muestral)", "Distribución Beta (Proporciones y Tasas Acotadas)"], horizontal=True, key="m3_opt")

    if "Chi-Cuadrada" in opcion_riesgo:
        st.subheader("Control de Variabilidad Muestral con Distribución Chi-Cuadrada (χ²)")
        st.write(r"Evaluación formal de la variabilidad del proceso mediante la relación (n-1)S² / σ² ~ χ² con ν = n-1 grados de libertad.")

        cc1, cc2 = st.columns([1, 1.5])
        with cc1:
            n_muestras = st.number_input("Tamaño de muestra n:", min_value=2, max_value=None, value=31, step=1, help="📌 Significado: Número total de observaciones tomadas de la línea de producción.\n\n🔢 Dominio válido: Debe ser un entero mayor o igual a 2.", key="m3_n")
            sigma_objetivo = st.number_input("Desviación estándar nominal σ:", min_value=1e-9, max_value=None, value=7.0, step=0.5, format="%g", help="📌 Significado: Dispersión estándar exigida por las especificaciones de calidad.\n\n🔢 Dominio válido: Debe ser estrictamente mayor que 0.", key="m3_sigobj")
            s_muestral = st.number_input("Desviación estándar observada S:", min_value=1e-9, max_value=None, value=9.0, step=0.5, format="%g", help="📌 Significado: Dispersión empírica calculada sobre los datos muestreados.\n\n🔢 Dominio válido: Debe ser estrictamente mayor que 0.", key="m3_smuest")

            nu_df = n_muestras - 1
            chi_calculado = (nu_df * (s_muestral**2)) / (sigma_objetivo**2)
            p_valor_cola = 1.0 - stats.chi2.cdf(chi_calculado, df=nu_df)

            with st.container(border=True):
                st.markdown("**📋 Resultados de Inferencia:**")
                st.markdown(f"- **Grados de libertad ν = n-1:** `{nu_df}`")
                st.markdown(f"- **Estadístico χ² calc:** `{chi_calculado:g}`")
                st.markdown(f"- **Probabilidad P(χ² > χ² calc):** `{p_valor_cola:g}` ({p_valor_cola*100:g}%)")

            if p_valor_cola < 0.05:
                st.error("🚨 Variabilidad Crítica: Se rechaza que la dispersión cumpla con el estándar nominal (p < 0.05). Ajustar maquinaria.")
            else:
                st.success("✅ Proceso Estable: La variabilidad muestral es consistente con la tolerancia (p ≥ 0.05).")

        with cc2:
            with st.container():
                x_chi_max = max(stats.chi2.ppf(0.999, df=nu_df), chi_calculado * 1.3)
                x_chi = np.linspace(0.01, x_chi_max, 400)
                y_chi = stats.chi2.pdf(x_chi, df=nu_df)

                fig_chi, ax_chi = plt.subplots(figsize=(7, 4.5))
                ax_chi.plot(x_chi, y_chi, color="#059669", lw=2.5, label=f"PDF χ²(ν={nu_df})")
                
                x_chi_sh = np.linspace(chi_calculado, x_chi_max, 200)
                if len(x_chi_sh) > 0 and chi_calculado < x_chi_max:
                    y_chi_sh = stats.chi2.pdf(x_chi_sh, df=nu_df)
                    ax_chi.fill_between(x_chi_sh, y_chi_sh, color="#DC2626", alpha=0.4, label=f"Región Crítica (p={p_valor_cola*100:g}%)")

                ax_chi.axvline(chi_calculado, color="#DC2626", linestyle="--", lw=2, label=f"χ² calc = {chi_calculado:g}")
                ax_chi.set_title(f"Distribución Chi-Cuadrada (ν={nu_df})", fontweight="bold")
                ax_chi.set_xlabel("χ²")
                ax_chi.set_ylabel("f(χ²)")
                ax_chi.grid(True, linestyle=":", alpha=0.6)
                ax_chi.legend()
                st.pyplot(fig_chi, clear_figure=True)
                plt.close(fig_chi)

    else:
        st.subheader("Análisis de Proporciones y Defectos con Distribución Beta")
        st.write("Modela variables aleatorias continuas acotadas en el intervalo unitario (0, 1), como tasas de impureza o proporciones de producto no conforme.")

        cb1, cb2 = st.columns([1, 1.5])
        with cb1:
            a_b = st.number_input("Parámetro α (a):", min_value=1e-9, max_value=None, value=4.0, step=0.5, format="%g", help="📌 Significado: Recuento de defectos u observaciones no conformes previas + 1.\n\n🔢 Dominio válido: Debe ser estrictamente mayor que 0.", key="m3_ba")
            b_b = st.number_input("Parámetro β (b):", min_value=1e-9, max_value=None, value=2.0, step=0.5, format="%g", help="📌 Significado: Recuento de unidades totalmente aceptadas o conformes previas + 1.\n\n🔢 Dominio válido: Debe ser estrictamente mayor que 0.", key="m3_bb")
            umbral_riesgo = st.number_input("Umbral de riesgo (xlim):", min_value=1e-7, max_value=1.0 - 1e-7, value=0.90, step=0.01, format="%g", help="📌 Significado: Proporción máxima aceptable de piezas no conformes en el lote.\n\n🔢 Dominio válido: Debe estar estrictamente entre 0 y 1.", key="m3_umb")

            prob_exceso = 1.0 - stats.beta.cdf(umbral_riesgo, a=a_b, b=b_b)
            e_beta = a_b / (a_b + b_b)
            v_beta = (a_b * b_b) / (((a_b + b_b)**2) * (a_b + b_b + 1))

            with st.container(border=True):
                st.markdown("**📋 Resultados del Análisis Beta:**")
                st.markdown(f"- **Esperanza E[X]:** `{e_beta:g}` ({e_beta*100:g}%)")
                st.markdown(f"- **Varianza V(X):** `{v_beta:g}`")
                st.markdown(f"- **Probabilidad Crítica P(X > {umbral_riesgo:g}):** `{prob_exceso:g}` ({prob_exceso*100:g}%)")

            if prob_exceso > 0.10:
                st.error(f"🚨 Alerta de Riesgo: P(X > {umbral_riesgo:g}) = {prob_exceso*100:g}% supera el umbral del 10%.")
            else:
                st.success(f"✅ Rango Seguro: P(X > {umbral_riesgo:g}) = {prob_exceso*100:g}% dentro de tolerancia.")

        with cb2:
            with st.container():
                x_dom = np.linspace(0.001, 0.999, 400)
                y_dom = stats.beta.pdf(x_dom, a=a_b, b=b_b)

                fig_beta, ax_beta = plt.subplots(figsize=(7, 4.5))
                ax_beta.plot(x_dom, y_dom, color="#7C3AED", lw=2.5, label="PDF Beta")
                
                x_sh = np.linspace(umbral_riesgo, 0.999, 200)
                y_sh = stats.beta.pdf(x_sh, a=a_b, b=b_b)
                ax_beta.fill_between(x_sh, y_sh, color="#EF4444", alpha=0.4, label=f"Zona de Riesgo ({prob_exceso*100:g}%)")
                ax_beta.axvline(umbral_riesgo, color="#DC2626", linestyle="--", lw=1.8)
                ax_beta.set_title(f"Distribución Beta(α={a_b:g}, β={b_b:g})", fontweight="bold")
                ax_beta.set_xlabel("Proporción x")
                ax_beta.set_ylabel("Densidad f(x)")
                ax_beta.grid(True, linestyle=":", alpha=0.6)
                ax_beta.legend()
                st.pyplot(fig_beta, clear_figure=True)
                plt.close(fig_beta)

# ==============================================================================
# MÓDULO 4: COMPARACIÓN DE MODELOS
# ==============================================================================
elif modulo == "4. Comparador de Modelos de Vida Útil":
    st.session_state["ver_demostracion"] = None
    st.header("⚖️ Módulo 4: Comparación Simultánea de Distribuciones")
    st.write("Contraste directamente dos distribuciones sobre el mismo componente para determinar el modelo más adecuado.")

    opciones_m4 = ["Gamma", "Exponencial", "Lognormal", "Weibull"]

    cm1, cm2 = st.columns(2)

    with cm1:
        st.subheader("Modelo 1")
        m1_tipo = st.selectbox("Seleccione modelo 1:", opciones_m4, key="m4_m1_tipo")
        if m1_tipo == "Gamma":
            g_a1 = st.number_input("Forma α1:", min_value=1e-9, max_value=None, value=3.0, step=0.5, format="%g", help="📌 Significado: Número de etapas del Modelo 1.\n\n🔢 Dominio válido: Debe ser estrictamente mayor que 0.", key="m4_ga1")
            g_b1 = st.number_input("Escala β1 (horas):", min_value=1e-9, max_value=None, value=1000.0, step=100.0, format="%g", help="📌 Significado: Escala temporal por etapa para el Modelo 1.\n\n🔢 Dominio válido: Debe ser estrictamente mayor que 0.", key="m4_gb1")
            dist1 = stats.gamma(a=g_a1, scale=g_b1)
        elif m1_tipo == "Exponencial":
            exp_sc1 = st.number_input("MTTF (horas):", min_value=1e-9, max_value=None, value=3000.0, step=100.0, format="%g", help="📌 Significado: Tiempo medio antes de fallar para el Modelo 1.\n\n🔢 Dominio válido: Debe ser estrictamente mayor que 0.", key="m4_esc1")
            dist1 = stats.expon(scale=exp_sc1)
        elif m1_tipo == "Lognormal":
            ln_u1 = st.number_input("μ1 (log):", min_value=None, max_value=None, value=7.8, step=0.1, format="%g", help="📌 Significado: Media logarítmica de la Lognormal del Modelo 1.\n\n🔢 Dominio válido: Puede tomar cualquier valor real (-∞, ∞).", key="m4_lnu1")
            ln_s1 = st.number_input("σ1 (log):", min_value=1e-9, max_value=None, value=0.5, step=0.05, format="%g", help="📌 Significado: Desviación estándar logarítmica del Modelo 1.\n\n🔢 Dominio válido: Debe ser estrictamente mayor que 0.", key="m4_lns1")
            dist1 = stats.lognorm(s=ln_s1, scale=np.exp(ln_u1))
        else:
            w_c1 = st.number_input("Forma α1:", min_value=1e-9, max_value=None, value=2.5, step=0.1, format="%g", help="📌 Significado: Parámetro de forma del Modelo 1.\n\n🔢 Dominio válido: Debe ser estrictamente mayor que 0.", key="m4_wc1")
            w_sc1 = st.number_input("Escala β1 (horas):", min_value=1e-9, max_value=None, value=3000.0, step=100.0, format="%g", help="📌 Significado: Escala de vida característica del Modelo 1.\n\n🔢 Dominio válido: Debe ser estrictamente mayor que 0.", key="m4_wsc1")
            dist1 = stats.weibull_min(c=w_c1, scale=w_sc1)

    with cm2:
        st.subheader("Modelo 2")
        m2_tipo = st.selectbox("Seleccione modelo 2:", opciones_m4, index=1, key="m4_m2_tipo")
        if m2_tipo == "Gamma":
            g_a2 = st.number_input("Forma α2:", min_value=1e-9, max_value=None, value=2.0, step=0.5, format="%g", help="📌 Significado: Número de etapas del Modelo 2.\n\n🔢 Dominio válido: Debe ser estrictamente mayor que 0.", key="m4_ga2")
            g_b2 = st.number_input("Escala β2 (horas):", min_value=1e-9, max_value=None, value=1500.0, step=100.0, format="%g", help="📌 Significado: Escala temporal por etapa para el Modelo 2.\n\n🔢 Dominio válido: Debe ser estrictamente mayor que 0.", key="m4_gb2")
            dist2 = stats.gamma(a=g_a2, scale=g_b2)
        elif m2_tipo == "Exponencial":
            exp_sc2 = st.number_input("MTTF (horas):", min_value=1e-9, max_value=None, value=3000.0, step=100.0, format="%g", help="📌 Significado: Tiempo medio antes de fallar para el Modelo 2.\n\n🔢 Dominio válido: Debe ser estrictamente mayor que 0.", key="m4_esc2")
            dist2 = stats.expon(scale=exp_sc2)
        elif m2_tipo == "Lognormal":
            ln_u2 = st.number_input("μ2 (log):", min_value=None, max_value=None, value=7.8, step=0.1, format="%g", help="📌 Significado: Media logarítmica de la Lognormal del Modelo 2.\n\n🔢 Dominio válido: Puede tomar cualquier valor real (-∞, ∞).", key="m4_lnu2")
            ln_s2 = st.number_input("σ2 (log):", min_value=1e-9, max_value=None, value=0.8, step=0.05, format="%g", help="📌 Significado: Desviación estándar logarítmica del Modelo 2.\n\n🔢 Dominio válido: Debe ser estrictamente mayor que 0.", key="m4_lns2")
            dist2 = stats.lognorm(s=ln_s2, scale=np.exp(ln_u2))
        else:
            w_c2 = st.number_input("Forma α2:", min_value=1e-9, max_value=None, value=1.2, step=0.1, format="%g", help="📌 Significado: Parámetro de forma del Modelo 2.\n\n🔢 Dominio válido: Debe ser estrictamente mayor que 0.", key="m4_wc2")
            w_sc2 = st.number_input("Escala β2 (horas):", min_value=1e-9, max_value=None, value=3000.0, step=100.0, format="%g", help="📌 Significado: Escala de vida característica del Modelo 2.\n\n🔢 Dominio válido: Debe ser estrictamente mayor que 0.", key="m4_wsc2")
            dist2 = stats.weibull_min(c=w_c2, scale=w_sc2)

    t_obj = st.number_input("Tiempo objetivo t (horas):", min_value=1e-9, max_value=None, value=2000.0, step=50.0, format="%g", help="📌 Significado: Punto de corte en horas para contrastar la confiabilidad R(t).\n\n🔢 Dominio válido: Debe ser estrictamente mayor que 0.", key="m4_t_obj")

    with st.container():
        t_fin = max(dist1.ppf(0.99), dist2.ppf(0.99))
        t_vector = np.linspace(0.1, t_fin, 400)

        r1_vec = 1.0 - dist1.cdf(t_vector)
        r2_vec = 1.0 - dist2.cdf(t_vector)

        fig_cmp, (ax_c1, ax_c2) = plt.subplots(1, 2, figsize=(14, 5))

        ax_c1.plot(t_vector, dist1.pdf(t_vector), color="#1E3A8A", lw=2.2, label=f"Modelo 1: {m1_tipo}")
        ax_c1.plot(t_vector, dist2.pdf(t_vector), color="#DC2626", lw=2.2, linestyle="--", label=f"Modelo 2: {m2_tipo}")
        ax_c1.set_title("Superposición de Densidades f(t)", fontweight="bold")
        ax_c1.set_xlabel("Tiempo (horas)")
        ax_c1.set_ylabel("f(t)")
        ax_c1.grid(True, linestyle=":", alpha=0.6)
        ax_c1.legend()

        ax_c2.plot(t_vector, r1_vec, color="#1E3A8A", lw=2.5, label=f"R1(t) ({m1_tipo})")
        ax_c2.plot(t_vector, r2_vec, color="#DC2626", lw=2.5, linestyle="--", label=f"R2(t) ({m2_tipo})")
        ax_c2.axvline(t_obj, color="#059669", linestyle=":", lw=2, label=f"t = {t_obj:g} h")
        ax_c2.set_title("Superposición de Confiabilidad R(t)", fontweight="bold")
        ax_c2.set_xlabel("Tiempo (horas)")
        ax_c2.set_ylabel("R(t)")
        ax_c2.set_ylim(-0.02, 1.02)
        ax_c2.grid(True, linestyle=":", alpha=0.6)
        ax_c2.legend()

        st.pyplot(fig_cmp, clear_figure=True)
        plt.close(fig_cmp)

    tabla_datos = {
        "Métrica": [
            f"Confiabilidad a {t_obj:g} h (R(t))",
            "Probabilidad acumulada de falla (F(t))",
            "Vida Media (MTTF)",
            "Vida B10 (10% fallan)",
            "Vida Mediana (B50)"
        ],
        f"Modelo 1 ({m1_tipo})": [
            f"{(1.0 - dist1.cdf(t_obj))*100:g}%",
            f"{dist1.cdf(t_obj)*100:g}%",
            f"{dist1.mean():g} h",
            f"{dist1.ppf(0.10):g} h",
            f"{dist1.ppf(0.50):g} h"
        ],
        f"Modelo 2 ({m2_tipo})": [
            f"{(1.0 - dist2.cdf(t_obj))*100:g}%",
            f"{dist2.cdf(t_obj)*100:g}%",
            f"{dist2.mean():g} h",
            f"{dist2.ppf(0.10):g} h",
            f"{dist2.ppf(0.50):g} h"
        ]
    }
    st.table(pd.DataFrame(tabla_datos))

# ==============================================================================
# MÓDULO 5: CASOS PRÁCTICOS Y TOMA DE DECISIONES
# ==============================================================================
elif modulo == "5. Casos Prácticos y Toma de Decisiones":
    st.session_state["ver_demostracion"] = None
    st.header("🎯 Módulo 5: Casos Prácticos y Criterios de Decisión")
    st.write("Resolución de 3 problemas de ingeniería contextualizados con recomendaciones automáticas.")

    caso_sel = st.selectbox(
        "Seleccione el caso de estudio:",
        [
            "Caso 1: Mantenimiento Preventivo de Turbinas Térmicas (Weibull)",
            "Caso 2: Aceptación de Lotes de Microchips Semiconductores (Beta)",
            "Caso 3: Monitoreo de Emisiones Químicas en Chimenea (Lognormal)"
        ],
        key="m5_caso_sel"
    )

    if "Caso 1" in caso_sel:
        st.subheader("Caso 1: Reemplazo Preventivo de Turbinas Térmicas")
        st.write("""
        **Contexto:** Los componentes sufren fatiga por ciclos térmicos. El tiempo de falla sigue una distribución **Weibull** con β = 8,000 h y α = 2.8 (desgaste acelerado).
        
        **Criterio de Decisión:** La norma técnica exige ejecutar el reemplazo antes de que el riesgo de falla acumulada supere la tolerancia establecida (F(t) ≤ Tolerancia).
        """)

        eta_t = st.number_input("Vida característica β (horas):", min_value=1e-9, max_value=None, value=8000.0, step=500.0, format="%g", help="📌 Significado: Vida nominal proyectada para los álabes de la turbina.\n\n🔢 Dominio válido: Debe ser estrictamente mayor que 0.", key="c1_eta")
        beta_t = st.number_input("Parámetro de forma α:", min_value=1e-9, max_value=None, value=2.8, step=0.1, format="%g", help="📌 Significado: Grado de desgaste o fatiga térmica acumulada.\n\n🔢 Dominio válido: Debe ser estrictamente mayor que 0.", key="c1_beta")
        tol_val = st.number_input("Tolerancia máxima de falla (%):", min_value=1e-7, max_value=100.0 - 1e-7, value=5.0, step=0.5, format="%g", help="📌 Significado: Porcentaje máximo aceptable de componentes fallados antes de parar.\n\n🔢 Dominio válido: Debe estar estrictamente entre 0 y 100%.", key="c1_tol")
        tol_max = tol_val / 100.0

        w_dist = stats.weibull_min(c=beta_t, scale=eta_t)
        t_optimo = float(w_dist.ppf(tol_max))
        mttf_t = float(w_dist.mean())

        with st.container(border=True):
            st.markdown("**📋 Recomendación Automática del Sistema:**")
            st.markdown(f"- **Tiempo Límite Seguro de Operación:** `{t_optimo:g} horas`")
            st.markdown(f"- **Vida Media MTTF (sin mantenimiento preventivo):** `{mttf_t:g} horas`")
            st.markdown(f"- **Decisión:** Programar la parada de mantenimiento cada `{t_optimo:g} horas` para garantizar que no más del `{tol_max*100:g}%` de las unidades fallen en servicio imprevistamente.")

    elif "Caso 2" in caso_sel:
        st.subheader("Caso 2: Control de Calidad y Rechazo de Lotes de Microchips")
        st.write("""
        **Contexto:** La proporción de unidades defectuosas de un lote sigue una distribución **Beta** (α, β) calibrada con inspecciones previas.
        
        **Criterio de Decisión:** Se rechaza el lote si la probabilidad de que la proporción de defectos supere el valor crítico es mayor al umbral de tolerancia.
        """)

        a_lot = st.number_input("Parámetro α:", min_value=1e-9, max_value=None, value=3.0, step=0.5, format="%g", help="📌 Significado: Cantidad de microchips defectuosos observados en muestreo previo + 1.\n\n🔢 Dominio válido: Debe ser estrictamente mayor que 0.", key="c2_alot")
        b_lot = st.number_input("Parámetro β:", min_value=1e-9, max_value=None, value=50.0, step=5.0, format="%g", help="📌 Significado: Cantidad de microchips operativos observados en muestreo previo + 1.\n\n🔢 Dominio válido: Debe ser estrictamente mayor que 0.", key="c2_blot")
        prop_critica_val = st.number_input("Proporción crítica (%):", min_value=1e-7, max_value=100.0 - 1e-7, value=8.0, step=0.5, format="%g", help="📌 Significado: Porcentaje máximo de defectos tolerado en el lote.\n\n🔢 Dominio válido: Debe estar estrictamente entre 0 y 100%.", key="c2_propcrit")
        tol_riesgo_val = st.number_input("Umbral máximo de riesgo (%):", min_value=1e-7, max_value=100.0 - 1e-7, value=10.0, step=0.5, format="%g", help="📌 Significado: Riesgo máximo aceptable de liberar un lote defectuoso al cliente.\n\n🔢 Dominio válido: Debe estar estrictamente entre 0 y 100%.", key="c2_tolriesgo")

        prop_critica = prop_critica_val / 100.0
        tol_riesgo = tol_riesgo_val / 100.0

        p_excede = 1.0 - stats.beta.cdf(prop_critica, a=a_lot, b=b_lot)
        tasa_media = a_lot / (a_lot + b_lot)

        st.markdown(f"- **Tasa media esperada de defectos:** `{tasa_media*100:g}%`")
        st.markdown(f"- **Probabilidad calculada P(X > {prop_critica:g}):** `{p_excede*100:g}%`")

        if p_excede > tol_riesgo:
            st.error(f"🚨 DECISIÓN: RECHAZAR EL LOTE. La probabilidad de exceder el {prop_critica_val:g}% de defectos es {p_excede*100:g}%, lo que supera el umbral de tolerancia ({tol_riesgo_val:g}%).")
        else:
            st.success(f"✅ DECISIÓN: ACEPTAR EL LOTE. La probabilidad de exceder el {prop_critica_val:g}% de defectos es {p_excede*100:g}%, cumpliendo las especificaciones de calidad.")

    elif "Caso 3" in caso_sel:
        st.subheader("Caso 3: Monitoreo Ambiental de Emisiones de Solventes")
        st.write("""
        **Contexto:** La concentración de contaminantes químicos (ppm) sigue una distribución **Lognormal** con parámetros μ y σ.
        
        **Criterio de Decisión:** La autoridad ambiental exige activar filtros de emergencia si la probabilidad de exceder la concentración límite supera el umbral de tiempo permitido.
        """)

        mu_amb = st.number_input("Media logarítmica μ:", min_value=None, max_value=None, value=3.2, step=0.1, format="%g", help="📌 Significado: Promedio continuo de concentración en escala logarítmica.\n\n🔢 Dominio válido: Puede tomar cualquier valor real (-∞, ∞).", key="c3_muamb")
        sig_amb = st.number_input("Desviación logarítmica σ:", min_value=1e-9, max_value=None, value=1.0, step=0.05, format="%g", help="📌 Significado: Variabilidad o dispersión de picos de emisión.\n\n🔢 Dominio válido: Debe ser estrictamente mayor que 0.", key="c3_sigamb")
        lim_ppm = st.number_input("Límite normativo (ppm):", min_value=1e-9, max_value=None, value=8.0, step=1.0, format="%g", help="📌 Significado: Concentración máxima de químico tolerada en partes por millón.\n\n🔢 Dominio válido: Debe ser estrictamente mayor que 0.", key="c3_limppm")
        tol_tiempo_pct = st.number_input("Porcentaje de tiempo límite (%):", min_value=1e-7, max_value=100.0 - 1e-7, value=20.0, step=1.0, format="%g", help="📌 Significado: Porcentaje de tiempo máximo tolerable en exceso de contaminantes.\n\n🔢 Dominio válido: Debe estar estrictamente entre 0 y 100%.", key="c3_toltiempo")

        tol_tiempo_ratio = tol_tiempo_pct / 100.0
        z_score = (np.log(lim_ppm) - mu_amb) / sig_amb
        p_exceso_amb = float(1.0 - stats.norm.cdf(z_score))

        st.markdown(f"- **Puntaje estandarizado Z:** `{z_score:g}`")
        st.markdown(f"- **Probabilidad calculada P(X > {lim_ppm:g} ppm):** `{p_exceso_amb*100:g}%`")

        if p_exceso_amb > tol_tiempo_ratio:
            st.error(f"🚨 ACCIÓN OBLIGATORIA: ACTIVAR FILTROS DE EMERGENCIA. La probabilidad de emisión excesiva ({p_exceso_amb*100:g}%) supera el {tol_tiempo_pct:g}% normativo.")
        else:
            st.success(f"✅ OPERACIÓN NORMAL: La concentración se mantiene dentro de los límites ambientales admisibles.")

# ==============================================================================
# MÓDULO 6: EJERCICIOS PRÁCTICOS Y CÁLCULO DE PROBABILIDADES
# ==============================================================================
elif modulo == "6. Ejercicios Prácticos y Cálculo de Probabilidades":
    st.session_state["ver_demostracion"] = None
    st.header("✏️ Módulo 6: Resolvedor Dinámico de Ejercicios de Probabilidad")
    st.write("Seleccione la **distribución teórica**, configure los **parámetros del modelo** e ingrese el **evento o desigualdad** a evaluar para obtener las métricas y la solución gráfica integrada.")

    col_e1, col_e2 = st.columns([1, 1.2])

    with col_e1:
        st.subheader("1. Selección y Configuración del Modelo")
        dist_ejercicio = st.selectbox(
            "Seleccione la función de densidad / distribución:",
            ["Gamma", "Exponencial", "Chi-cuadrada", "Lognormal", "Beta", "Weibull"],
            key="m6_dist_sel"
        )

        if dist_ejercicio == "Gamma":
            alpha_ej = st.number_input("Forma α:", min_value=1e-9, max_value=None, value=3.0, step=0.5, format="%g", help="📌 Significado: Número de etapas acumuladas en el proceso.\n\n🔢 Dominio válido: Debe ser estrictamente mayor que 0.", key="m6_gam_a")
            beta_ej = st.number_input("Escala β:", min_value=1e-9, max_value=None, value=2.0, step=0.2, format="%g", help="📌 Significado: Escala temporal asignada a cada etapa.\n\n🔢 Dominio válido: Debe ser estrictamente mayor que 0.", key="m6_gam_b")
            dist_inst = stats.gamma(a=alpha_ej, scale=beta_ej)
            media_ej = alpha_ej * beta_ej
            var_ej = alpha_ej * (beta_ej**2)
            domain_min, domain_max = 0.0, float(dist_inst.ppf(0.999))

        elif dist_ejercicio == "Exponencial":
            lam_ej = st.number_input("Tasa λ:", min_value=1e-9, max_value=None, value=0.2, step=0.01, format="%g", help="📌 Significado: Tasa media de eventos por unidad de tiempo.\n\n🔢 Dominio válido: Debe ser estrictamente mayor que 0.", key="m6_exp_lam")
            dist_inst = stats.expon(scale=1.0/lam_ej)
            media_ej = 1.0 / lam_ej
            var_ej = 1.0 / (lam_ej**2)
            domain_min, domain_max = 0.0, float(dist_inst.ppf(0.999))

        elif dist_ejercicio == "Chi-cuadrada":
            df_ej = st.number_input("Grados de libertad ν:", min_value=1, max_value=None, value=8, step=1, help="📌 Significado: Grados de libertad de la variable Chi-cuadrada.\n\n🔢 Dominio válido: Debe ser un entero mayor o igual a 1.", key="m6_chi_df")
            dist_inst = stats.chi2(df=df_ej)
            media_ej = float(df_ej)
            var_ej = float(2 * df_ej)
            domain_min, domain_max = 0.0, float(dist_inst.ppf(0.999))

        elif dist_ejercicio == "Lognormal":
            mu_ej = st.number_input("Media logarítmica μ:", min_value=None, max_value=None, value=2.0, step=0.1, format="%g", help="📌 Significado: Media de la transformación logarítmica ln(X).\n\n🔢 Dominio válido: Puede tomar cualquier valor real (-∞, ∞).", key="m6_log_mu")
            sig_ej = st.number_input("Desviación logarítmica σ:", min_value=1e-9, max_value=None, value=0.5, step=0.05, format="%g", help="📌 Significado: Desviación estándar de ln(X).\n\n🔢 Dominio válido: Debe ser estrictamente mayor que 0.", key="m6_log_sig")
            dist_inst = stats.lognorm(s=sig_ej, scale=np.exp(mu_ej))
            media_ej = np.exp(mu_ej + (sig_ej**2)/2.0)
            var_ej = np.exp(2*mu_ej + sig_ej**2) * (np.exp(sig_ej**2) - 1.0)
            domain_min, domain_max = 0.0, float(dist_inst.ppf(0.995))

        elif dist_ejercicio == "Beta":
            a_ej = st.number_input("Parámetro α:", min_value=1e-9, max_value=None, value=2.0, step=0.5, format="%g", help="📌 Significado: Primer parámetro de forma sobre el intervalo (0, 1).\n\n🔢 Dominio válido: Debe ser estrictamente mayor que 0.", key="m6_bet_a")
            b_ej = st.number_input("Parámetro β:", min_value=1e-9, max_value=None, value=5.0, step=0.5, format="%g", help="📌 Significado: Segundo parámetro de forma sobre el intervalo (0, 1).\n\n🔢 Dominio válido: Debe ser estrictamente mayor que 0.", key="m6_bet_b")
            dist_inst = stats.beta(a=a_ej, b=b_ej)
            media_ej = a_ej / (a_ej + b_ej)
            var_ej = (a_ej * b_ej) / (((a_ej + b_ej)**2) * (a_ej + b_ej + 1))
            domain_min, domain_max = 0.0, 1.0

        elif dist_ejercicio == "Weibull":
            c_ej = st.number_input("Forma α:", min_value=1e-9, max_value=None, value=2.0, step=0.1, format="%g", help="📌 Significado: Parámetro de forma que define el riesgo acumulado.\n\n🔢 Dominio válido: Debe ser estrictamente mayor que 0.", key="m6_wei_c")
            sc_ej = st.number_input("Escala β:", min_value=1e-9, max_value=None, value=10.0, step=0.5, format="%g", help="📌 Significado: Vida característica del modelo Weibull.\n\n🔢 Dominio válido: Debe ser estrictamente mayor que 0.", key="m6_wei_sc")
            dist_inst = stats.weibull_min(c=c_ej, scale=sc_ej)
            media_ej = float(dist_inst.mean())
            var_ej = float(dist_inst.var())
            domain_min, domain_max = 0.0, float(dist_inst.ppf(0.999))

        st.subheader("2. Planteamiento de la Desigualdad")
        tipo_ineq = st.selectbox(
            "Seleccione la forma de la probabilidad a calcular:",
            ["P(X ≤ x)", "P(X < x)", "P(X ≥ x)", "P(X > x)", "P(a ≤ X ≤ b)"],
            key="m6_ineq_type"
        )

        if tipo_ineq in ["P(X ≤ x)", "P(X < x)", "P(X ≥ x)", "P(X > x)"]:
            if dist_ejercicio == "Beta":
                x_val = st.number_input("Valor límite x:", min_value=1e-7, max_value=1.0 - 1e-7, value=0.4, step=0.05, format="%g", help="📌 Significado: Punto numérico límite x a evaluar en la desigualdad.\n\n🔢 Dominio válido: Debe estar estrictamente entre 0 y 1.", key="m6_xval")
            else:
                x_val = st.number_input("Valor límite x:", min_value=0.0, max_value=None, value=float(np.round(media_ej, 2)), step=0.5, format="%g", help="📌 Significado: Punto numérico límite x a evaluar en la desigualdad.\n\n🔢 Dominio válido: Debe ser mayor o igual a 0.", key="m6_xval")

            if tipo_ineq in ["P(X ≤ x)", "P(X < x)"]:
                prob_res = float(dist_inst.cdf(x_val))
                texto_res = f"P(X ≤ {x_val:g}) = {prob_res:g}"
            else:
                prob_res = float(1.0 - dist_inst.cdf(x_val))
                texto_res = f"P(X ≥ {x_val:g}) = {prob_res:g}"

        else:
            if dist_ejercicio == "Beta":
                a_val = st.number_input("Límite inferior a:", min_value=1e-7, max_value=1.0 - 1e-7, value=0.2, step=0.05, format="%g", help="📌 Significado: Extremo inferior del intervalo a evaluar.\n\n🔢 Dominio válido: Debe cumplir que a < b y estar entre 0 y 1.", key="m6_aval")
                b_val = st.number_input("Límite superior b:", min_value=1e-7, max_value=1.0 - 1e-7, value=0.7, step=0.05, format="%g", help="📌 Significado: Extremo superior del intervalo a evaluar.\n\n🔢 Dominio válido: Debe cumplir que a < b y estar entre 0 y 1.", key="m6_bval")
            else:
                a_val = st.number_input("Límite inferior a:", min_value=0.0, max_value=None, value=float(np.round(media_ej * 0.5, 2)), step=0.5, format="%g", help="📌 Significado: Extremo inferior del intervalo a evaluar.\n\n🔢 Dominio válido: Debe cumplir que a < b.", key="m6_aval")
                b_val = st.number_input("Límite superior b:", min_value=0.0, max_value=None, value=float(np.round(media_ej * 1.5, 2)), step=0.5, format="%g", help="📌 Significado: Extremo superior del intervalo a evaluar.\n\n🔢 Dominio válido: Debe cumplir que a < b.", key="m6_bval")

            if a_val >= b_val:
                st.warning("⚠️ El límite inferior a debe ser estrictamente menor que el límite superior b.")
                prob_res = 0.0
                texto_res = "P(a ≤ X ≤ b) = 0"
            else:
                prob_res = float(dist_inst.cdf(b_val) - dist_inst.cdf(a_val))
                texto_res = f"P({a_val:g} ≤ X ≤ {b_val:g}) = {prob_res:g}"

        with st.container(border=True):
            st.markdown("**📋 Resultados Analíticos del Ejercicio:**")
            st.markdown(f"- **Resultado Evaluado:** `{texto_res}` ({prob_res*100:g}%)")
            st.markdown(f"- **Media E[X]:** `{media_ej:g}`")
            st.markdown(f"- **Varianza V(X):** `{var_ej:g}`")
            st.markdown(f"- **Desviación Estándar σ:** `{np.sqrt(var_ej):g}`")

    with col_e2:
        st.subheader("3. Representación Gráfica del Evento")
        x_plot = np.linspace(domain_min + 0.0001 if domain_min == 0 else domain_min, domain_max, 500)
        y_plot = dist_inst.pdf(x_plot)

        fig_ej, ax_ej = plt.subplots(figsize=(7, 5.2))
        ax_ej.plot(x_plot, y_plot, color="#1E3A8A", lw=2.5, label=f"PDF {dist_ejercicio}")

        if tipo_ineq in ["P(X ≤ x)", "P(X < x)"]:
            x_shade = np.linspace(domain_min + 0.0001, x_val, 300)
            y_shade = dist_inst.pdf(x_shade)
            ax_ej.fill_between(x_shade, y_shade, color="#10B981", alpha=0.4, label=f"Área = {prob_res:g}")
            ax_ej.axvline(x_val, color="#059669", linestyle="--", lw=2)

        elif tipo_ineq in ["P(X ≥ x)", "P(X > x)"]:
            x_shade = np.linspace(x_val, domain_max, 300)
            y_shade = dist_inst.pdf(x_shade)
            ax_ej.fill_between(x_shade, y_shade, color="#10B981", alpha=0.4, label=f"Área = {prob_res:g}")
            ax_ej.axvline(x_val, color="#059669", linestyle="--", lw=2)

        else:
            if a_val < b_val:
                x_shade = np.linspace(a_val, b_val, 300)
                y_shade = dist_inst.pdf(x_shade)
                ax_ej.fill_between(x_shade, y_shade, color="#10B981", alpha=0.4, label=f"Área = {prob_res:g}")
                ax_ej.axvline(a_val, color="#059669", linestyle="--", lw=1.8)
                ax_ej.axvline(b_val, color="#059669", linestyle="--", lw=1.8)

        ax_ej.axvline(media_ej, color="#DC2626", linestyle=":", lw=1.8, label=f"E[X] = {media_ej:g}")
        ax_ej.set_title(f"Visualización del Evento de Probabilidad ({dist_ejercicio})", fontweight="bold")
        ax_ej.set_xlabel("Variable Aleatoria X")
        ax_ej.set_ylabel("Densidad f(x)")
        ax_ej.grid(True, linestyle=":", alpha=0.6)
        ax_ej.legend(loc="upper right")

        st.pyplot(fig_ej, clear_figure=True)
        plt.close(fig_ej)

# ==============================================================================
# MÓDULO 7: FUNDAMENTACIÓN TEÓRICA Y VISUALIZADOR DE DEMOSTRACIONES (PDF)
# ==============================================================================
elif modulo == "7. Fundamentación Teórica y Fórmulas":
    
    # SI HAY UNA DEMOSTRACIÓN SELECCIONADA, MUESTRA EL VISOR DE PDF DEDICADO
    if st.session_state["ver_demostracion"] is not None:
        dist_dem = st.session_state["ver_demostracion"]
        
        if st.button("⬅️ Regresar al Módulo 7 (Fórmulas Resumidas)"):
            st.session_state["ver_demostracion"] = None
            st.rerun()

        st.markdown(f"## 📄 Documento de Demostración: Distribución {dist_dem}")
        st.write("Consulta el desarrollo matemático original del cuaderno cargado en PDF:")
        st.markdown("---")

        # Mapeo exacto a los nombres de tus archivos PDF
        archivos_pdf = {
            "Gamma": "demostracion_gamma_pdf.pdf",
            "Exponencial": "demostracion_exponencial_pdf.pdf",
            "Chi-cuadrada": "demostracion_chicuadrada_pdf.pdf",
            "Lognormal": "demostracion_lognormal_pdf.pdf",
            "Beta": "demostracion_beta_pdf.pdf",
            "Weibull": "demostracion_weibull_pdf.pdf"
        }

        nombre_archivo = archivos_pdf.get(dist_dem)
        if nombre_archivo:
            mostrar_pdf(nombre_archivo)

    # VISTA PRINCIPAL ORDENADA: 1. Gamma, 2. Exponencial, 3. Chi-Cuadrada, 4. Log-Normal, 5. Beta, 6. Weibull
    else:
        st.header("📚 Módulo 7: Fundamentación Matemática y Procedimientos Analíticos")
        st.write("Fórmulas, momentos matemáticos y acceso directo a las demostraciones escaneadas en PDF:")

        # 1. DISTRIBUCIÓN GAMMA
        with st.expander("1. Distribución Gamma", expanded=True):
            st.latex(r"\Gamma(\alpha) = \int_{0}^{\infty} t^{\alpha-1} e^{-t} dt")
            st.latex(r"f(x) = \frac{1}{\beta^\alpha \Gamma(\alpha)} x^{\alpha-1} e^{-x/\beta}, \quad x > 0")
            st.latex(r"E[X] = \alpha \beta, \quad V(X) = \alpha \beta^2")
            st.markdown("---")
            if st.button("🔗 Ver Demostración en PDF (Gamma)", key="btn_pdf_gam"):
                st.session_state["ver_demostracion"] = "Gamma"
                st.rerun()

        # 2. DISTRIBUCIÓN EXPONENCIAL
        with st.expander("2. Distribución Exponencial"):
            st.latex(r"f(x) = \lambda e^{-\lambda x} = \frac{1}{\beta} e^{-x/\beta}, \quad x > 0")
            st.latex(r"E[X] = \frac{1}{\lambda} = \beta, \quad V(X) = \frac{1}{\lambda^2} = \beta^2")
            st.markdown("**Propiedad sin memoria:** P(X > s + t | X > s) = P(X > t)")
            st.markdown("---")
            if st.button("🔗 Ver Demostración en PDF (Exponencial)", key="btn_pdf_exp"):
                st.session_state["ver_demostracion"] = "Exponencial"
                st.rerun()

        # 3. DISTRIBUCIÓN CHI-CUADRADA
        with st.expander("3. Distribución Chi-Cuadrada (χ²)"):
            st.latex(r"E[X] = \nu, \quad V(X) = 2\nu")
            st.latex(r"\frac{(n-1)S^2}{\sigma^2} \sim \chi^2_{n-1}")
            st.markdown("---")
            if st.button("🔗 Ver Demostración en PDF (Chi-cuadrada)", key="btn_pdf_chi"):
                st.session_state["ver_demostracion"] = "Chi-cuadrada"
                st.rerun()

        # 4. DISTRIBUCIÓN LOG-NORMAL
        with st.expander("4. Distribución Log-Normal"):
            st.latex(r"Y = \ln(X) \sim \mathcal{N}(\mu, \sigma^2)")
            st.latex(r"f(x) = \frac{1}{x \sigma \sqrt{2\pi}} e^{-\frac{(\ln x - \mu)^2}{2\sigma^2}}, \quad x > 0")
            st.latex(r"E[X] = e^{\mu + \frac{\sigma^2}{2}}, \quad V(X) = e^{2\mu + \sigma^2}(e^{\sigma^2} - 1)")
            st.markdown("---")
            if st.button("🔗 Ver Demostración en PDF (Log-Normal)", key="btn_pdf_log"):
                st.session_state["ver_demostracion"] = "Lognormal"
                st.rerun()

        # 5. DISTRIBUCIÓN BETA
        with st.expander("5. Distribución Beta"):
            st.latex(r"B(\alpha, \beta) = \frac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha+\beta)}")
            st.latex(r"f(x) = \frac{1}{B(\alpha, \beta)} x^{\alpha-1}(1-x)^{\beta-1}, \quad 0 < x < 1")
            st.latex(r"E[X] = \frac{\alpha}{\alpha+\beta}, \quad V(X) = \frac{\alpha \beta}{(\alpha+\beta)^2(\alpha+\beta+1)}")
            st.markdown("---")
            if st.button("🔗 Ver Demostración en PDF (Beta)", key="btn_pdf_bet"):
                st.session_state["ver_demostracion"] = "Beta"
                st.rerun()

        # 6. DISTRIBUCIÓN WEIBULL
        with st.expander("6. Distribución Weibull y Confiabilidad"):
            st.latex(r"h(t) = \frac{\alpha}{\beta}\left(\frac{t}{\beta}\right)^{\alpha-1}, \quad R(t) = e^{-(t/\beta)^\alpha}")
            st.latex(r"f(t) = \frac{\alpha}{\beta}\left(\frac{t}{\beta}\right)^{\alpha-1} e^{-(t/\beta)^\alpha}")
            st.markdown("---")
            if st.button("🔗 Ver Demostración en PDF (Weibull)", key="btn_pdf_wei"):
                st.session_state["ver_demostracion"] = "Weibull"
                st.rerun()
