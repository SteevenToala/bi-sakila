# =============================================================================
# INFORME DEL PROYECTO BI3
# Archivo: generar_informe.py
#
# Responsabilidad:
#   Generar el informe HTML del proyecto a partir de las imágenes
#   producidas por el análisis exploratorio (Paso 3). El informe
#   se guarda en esta misma carpeta como "informe_proyecto_bi3.html"
#   y puede abrirse en cualquier navegador web.
#
# IMPORTANTE: Este script solo genera el informe visual.
#   No contiene lógica de análisis ni de entrenamiento.
#   Para generar las imágenes primero ejecuta:
#     python paso_3_eda/analisis_exploratorio.py
# =============================================================================

import os

DIRECTORIO_INFORME  = os.path.dirname(os.path.abspath(__file__))
DIRECTORIO_IMAGENES = os.path.join(DIRECTORIO_INFORME, "imagenes")
RUTA_SALIDA_HTML    = os.path.join(DIRECTORIO_INFORME, "informe_proyecto_bi3.html")

# Subcarpetas de imágenes (generadas por el Paso 3)
DIR_MATRICES       = os.path.join(DIRECTORIO_IMAGENES, "01_matrices_correlacion")
DIR_DISTRIBUCIONES = os.path.join(DIRECTORIO_IMAGENES, "02_distribuciones")
DIR_QQPLOTS        = os.path.join(DIRECTORIO_IMAGENES, "03_qqplots_normalidad")
DIR_BOXPLOTS       = os.path.join(DIRECTORIO_IMAGENES, "04_boxplots_por_etiqueta")
DIR_DISPERSIONES   = os.path.join(DIRECTORIO_IMAGENES, "05_dispersiones_bivariadas")

VARIABLES_PREDICTORAS = [
    "ingreso", "veces_alquilada_pelicula", "alquileres_por_cliente",
    "duracion_alquiler", "pelicula_costo_reposicion", "pelicula_duracion",
    "pelicula_precio_renta", "variedad_generos_cliente",
    "pelicula_cantidad_actores", "pelicula_popularidad_actores",
]

ETIQUETAS_OBJETIVO = [
    "es_ingreso_alto", "renta_fin_de_semana", "cliente_prefiere_genero", "renta_larga",
    "rango_precio_renta", "grupo_edad_pelicula", "nivel_fidelidad_cliente", "popularidad_pelicula",
]

CRUCES_POR_ETIQUETA = {
    "es_ingreso_alto":        [("pelicula_precio_renta", "ingreso"), ("pelicula_duracion", "ingreso"), ("pelicula_costo_reposicion", "ingreso")],
    "renta_fin_de_semana":    [("veces_alquilada_pelicula", "ingreso"), ("alquileres_por_cliente", "ingreso"), ("pelicula_precio_renta", "ingreso")],
    "cliente_prefiere_genero":[("alquileres_por_cliente", "variedad_generos_cliente"), ("pelicula_precio_renta", "variedad_generos_cliente"), ("ingreso", "variedad_generos_cliente")],
    "renta_larga":            [("pelicula_duracion", "duracion_alquiler"), ("pelicula_precio_renta", "duracion_alquiler"), ("pelicula_costo_reposicion", "duracion_alquiler")],
    "rango_precio_renta":     [("pelicula_precio_renta", "ingreso"), ("pelicula_duracion", "ingreso"), ("pelicula_costo_reposicion", "ingreso")],
    "grupo_edad_pelicula":    [("pelicula_duracion", "pelicula_costo_reposicion"), ("pelicula_precio_renta", "pelicula_costo_reposicion"), ("ingreso", "pelicula_costo_reposicion")],
    "nivel_fidelidad_cliente":[("alquileres_por_cliente", "variedad_generos_cliente"), ("alquileres_por_cliente", "ingreso"), ("alquileres_por_cliente", "pelicula_costo_reposicion")],
    "popularidad_pelicula":   [("veces_alquilada_pelicula", "pelicula_precio_renta"), ("veces_alquilada_pelicula", "ingreso"), ("veces_alquilada_pelicula", "pelicula_duracion")],
}


def generar_informe_html():
    """Genera el informe HTML completo a partir de las imágenes del EDA."""

    css = """
        body { font-family: 'Segoe UI', sans-serif; background: #f4f6f9; color: #222; margin: 0; padding: 0; }
        header { background: #002060; color: white; padding: 28px 40px; text-align: center; border-bottom: 4px solid #001240; }
        header h1 { margin: 0 0 6px 0; font-size: 26px; }
        header p  { margin: 0; font-size: 14px; opacity: .85; }
        .container { max-width: 1280px; margin: 30px auto; padding: 0 24px 60px; }
        .section { background: white; border-radius: 10px; padding: 28px; margin-bottom: 36px;
                   box-shadow: 0 2px 8px rgba(0,0,0,.07); }
        .section-title { font-size: 20px; color: #002060; border-bottom: 2px solid #002060;
                         padding-bottom: 10px; margin-bottom: 20px; }
        .section-subtitle { font-size: 16px; color: #003090; margin: 20px 0 12px; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 22px; }
        .grid-3 { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 18px; }
        .card { background: #fafafa; border: 1px solid #e0e4ed; border-radius: 8px; padding: 16px; text-align: center; }
        .card img { max-width: 100%; border: 1px solid #ddd; border-radius: 6px;
                    transition: transform .2s; cursor: zoom-in; }
        .card img:hover { transform: scale(1.04); }
        .card h4 { margin: 10px 0 0; font-size: 13px; color: #334; }
        .etiqueta-header { background: #e8edf8; border-left: 5px solid #002060;
                           padding: 12px 16px; border-radius: 0 8px 8px 0; margin-bottom: 16px; }
        .etiqueta-header h3 { margin: 0; color: #002060; font-size: 15px; }
        footer { text-align: center; color: #888; font-size: 13px; padding: 20px; }
    """

    lineas_html = [
        "<!DOCTYPE html>",
        '<html lang="es">',
        "<head>",
        '  <meta charset="UTF-8">',
        '  <meta name="viewport" content="width=device-width, initial-scale=1.0">',
        "  <title>Informe EDA — Proyecto BI3</title>",
        f"  <style>{css}</style>",
        "</head>",
        "<body>",
        "  <header>",
        "    <h1>Universidad Técnica de Ambato</h1>",
        "    <p>Facultad de Ingeniería en Sistemas, Electrónica e Industrial — Carrera de Software</p>",
        "    <p style='margin-top:8px; font-size:18px; font-weight:bold;'>Informe de Análisis Exploratorio de Datos (EDA) — Proyecto BI3</p>",
        "    <p>Red Neuronal Artificial para la Toma de Decisiones Comerciales</p>",
        "  </header>",
        '  <div class="container">',
    ]

    # ─── Sección 1: Matrices ───────────────────────────────────────────────
    lineas_html += [
        '    <div class="section">',
        '      <div class="section-title">1. Matrices de Correlación y Covarianza</div>',
        '      <p>Análisis de dependencias lineales entre las variables predictoras y las etiquetas objetivo.</p>',
        '      <div class="grid">',
    ]
    imagenes_matrices = [
        ("imagenes/01_matrices_correlacion/covarianza_predictores.png",    "Covarianza de Predictores"),
        ("imagenes/01_matrices_correlacion/correlacion_predictores.png",   "Correlación de Pearson (Predictores)"),
        ("imagenes/01_matrices_correlacion/covarianza_etiquetas.png",      "Covarianza de Etiquetas Objetivo"),
        ("imagenes/01_matrices_correlacion/correlacion_etiquetas.png",     "Correlación de Pearson (Etiquetas)"),
    ]
    for ruta_img, titulo in imagenes_matrices:
        lineas_html += [
            '        <div class="card">',
            f'          <img src="{ruta_img}" alt="{titulo}">',
            f'          <h4>{titulo}</h4>',
            "        </div>",
        ]
    lineas_html += ["      </div>", "    </div>"]

    # ─── Sección 2: Distribuciones ─────────────────────────────────────────
    lineas_html += [
        '    <div class="section">',
        '      <div class="section-title">2. Distribución de Variables Predictoras (Histogramas + Curva Normal)</div>',
        '      <p>Cada gráfico muestra la densidad empírica de la variable con la campana de Gauss teórica ajustada y el p-valor del test de Kolmogorov-Smirnov.</p>',
        '      <div class="grid">',
    ]
    for variable in VARIABLES_PREDICTORAS:
        ruta = f"imagenes/02_distribuciones/dist_{variable}.png"
        lineas_html += [
            '        <div class="card">',
            f'          <img src="{ruta}" alt="Distribución {variable}">',
            f'          <h4>{variable}</h4>',
            "        </div>",
        ]
    lineas_html += ["      </div>", "    </div>"]

    # ─── Sección 3: Q-Q Plots ──────────────────────────────────────────────
    lineas_html += [
        '    <div class="section">',
        '      <div class="section-title">3. Gráficos Q-Q de Normalidad por Variable</div>',
        '      <p>Compara los cuantiles observados vs. los teóricos de una distribución normal. Cuanto más cerca de la diagonal, más normal es la variable.</p>',
        '      <div class="grid">',
    ]
    for variable in VARIABLES_PREDICTORAS:
        ruta = f"imagenes/03_qqplots_normalidad/qq_{variable}.png"
        lineas_html += [
            '        <div class="card">',
            f'          <img src="{ruta}" alt="Q-Q {variable}">',
            f'          <h4>Q-Q: {variable}</h4>',
            "        </div>",
        ]
    lineas_html += ["      </div>", "    </div>"]

    # ─── Sección 4: Dispersiones por Etiqueta ──────────────────────────────
    lineas_html += [
        '    <div class="section">',
        '      <div class="section-title">4. Gráficos de Dispersión por Etiqueta Objetivo</div>',
        '      <p>Cada scatter plot muestra cómo se distribuyen los datos de entrada segmentados por la clase de la etiqueta objetivo (n=5,000 puntos de muestra).</p>',
    ]
    for nombre_etiqueta, pares in CRUCES_POR_ETIQUETA.items():
        lineas_html += [
            '      <div class="etiqueta-header">',
            f'        <h3>Etiqueta: {nombre_etiqueta}</h3>',
            "      </div>",
            '      <div class="grid-3">',
        ]
        for var_x, var_y in pares:
            nombre_archivo = f"scatter__{nombre_etiqueta}__{var_x}__vs__{var_y}.png"
            ruta = f"imagenes/05_dispersiones_bivariadas/{nombre_archivo}"
            titulo = f"{var_x} vs {var_y}"
            lineas_html += [
                '        <div class="card">',
                f'          <img src="{ruta}" alt="{titulo}">',
                f'          <h4>{titulo}</h4>',
                "        </div>",
            ]
        lineas_html += ["      </div>"]
    lineas_html += ["    </div>"]

    # ─── Sección 5: Arquitectura de la Red ─────────────────────────────────
    lineas_html += [
        '    <div class="section">',
        '      <div class="section-title">5. Arquitectura de la Red Neuronal Multitarea</div>',
        "      <p>El modelo de aprendizaje multitarea (Multi-Task Learning) predice simultáneamente 8 variables objetivo usando un cuerpo compartido de capas densas.</p>",
        "      <ul>",
        "        <li><strong>Capa de Entrada:</strong> 31 neuronas (10 continuas + 16 géneros OHE + 5 clasificaciones OHE)</li>",
        "        <li><strong>Capas Ocultas Compartidas:</strong> Densas con ReLU + BatchNormalization + Dropout</li>",
        "        <li><strong>4 Salidas Binarias:</strong> Sigmoid — es_ingreso_alto, renta_fin_de_semana, cliente_prefiere_genero, renta_larga</li>",
        "        <li><strong>4 Salidas Multiclase:</strong> Softmax (3 clases) — rango_precio_renta, grupo_edad_pelicula, nivel_fidelidad_cliente, popularidad_pelicula</li>",
        "        <li><strong>5 Arquitecturas evaluadas</strong> con distintas capas, dropout, épocas y batch size. El mejor modelo se guarda como <code>modelo_principal.keras</code></li>",
        "      </ul>",
        "    </div>",
    ]

    lineas_html += [
        "  </div>",
        "  <footer><p>Universidad Técnica de Ambato — Inteligencia de Negocios — Proyecto BI3</p></footer>",
        "</body>",
        "</html>",
    ]

    with open(RUTA_SALIDA_HTML, "w", encoding="utf-8") as archivo_html:
        archivo_html.write("\n".join(lineas_html))

    print(f"  ✓ Informe generado en: {RUTA_SALIDA_HTML}")


if __name__ == "__main__":
    print("\n=== GENERANDO INFORME HTML DEL PROYECTO BI3 ===")
    generar_informe_html()
    print("  Abre el archivo 'informe_proyecto_bi3.html' en tu navegador.")
