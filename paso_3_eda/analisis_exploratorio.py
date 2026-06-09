# =============================================================================
# PASO 3 - ANÁLISIS EXPLORATORIO DE DATOS (EDA)
# Archivo: analisis_exploratorio.py
#
# Responsabilidad:
#   Leer el dataset base construido en el Paso 2 y generar todas las
#   visualizaciones estadísticas que permiten entender la calidad de los
#   datos antes de entrenar la red neuronal. Las imágenes se guardan en
#   la carpeta "imagenes/" organizada por subcarpetas temáticas.
#
# Gráficos generados:
#   - Matrices de covarianza y correlación de Pearson (features y etiquetas)
#   - Histogramas con ajuste de curva normal (Kolmogorov-Smirnov)
#   - Gráficos Q-Q de normalidad individual por variable
#   - Boxplots de cada feature segmentados por cada etiqueta objetivo
#   - Gráficos de dispersión bivariados (scatter plots) por etiqueta
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

# --------------------------------------------------------------------------
# Rutas de entrada y salida
# --------------------------------------------------------------------------
DIRECTORIO_ACTUAL     = os.path.dirname(os.path.abspath(__file__))
RUTA_DATASET_BASE     = os.path.join(DIRECTORIO_ACTUAL, "..", "paso_2_dataset", "dataset_base.csv")
DIRECTORIO_IMAGENES   = os.path.join(DIRECTORIO_ACTUAL, "imagenes")

# Subcarpetas por tipo de análisis
DIR_MATRICES          = os.path.join(DIRECTORIO_IMAGENES, "01_matrices_correlacion")
DIR_DISTRIBUCIONES    = os.path.join(DIRECTORIO_IMAGENES, "02_distribuciones")
DIR_QQPLOTS           = os.path.join(DIRECTORIO_IMAGENES, "03_qqplots_normalidad")
DIR_BOXPLOTS          = os.path.join(DIRECTORIO_IMAGENES, "04_boxplots_por_etiqueta")
DIR_DISPERSIONES      = os.path.join(DIRECTORIO_IMAGENES, "05_dispersiones_bivariadas")

# Crear todas las carpetas si no existen
for directorio in [DIR_MATRICES, DIR_DISTRIBUCIONES, DIR_QQPLOTS, DIR_BOXPLOTS, DIR_DISPERSIONES]:
    os.makedirs(directorio, exist_ok=True)

# --------------------------------------------------------------------------
# Constantes: columnas del dataset
# --------------------------------------------------------------------------
VARIABLES_PREDICTORAS = [
    "ingreso", "veces_alquilada_pelicula", "alquileres_por_cliente",
    "duracion_alquiler", "pelicula_costo_reposicion", "pelicula_duracion",
    "pelicula_precio_renta", "variedad_generos_cliente",
    "pelicula_cantidad_actores", "pelicula_popularidad_actores",
]

ETIQUETAS_BINARIAS = [
    "es_ingreso_alto", "renta_fin_de_semana",
    "cliente_prefiere_genero", "renta_larga",
]

ETIQUETAS_MULTICLASE = [
    "rango_precio_renta", "grupo_edad_pelicula",
    "nivel_fidelidad_cliente", "popularidad_pelicula",
]

# Cruces clave de pares (X, Y) para los scatter plots por etiqueta
CRUCES_DE_DISPERSION_POR_ETIQUETA = {
    "es_ingreso_alto":       [("pelicula_precio_renta", "ingreso"),        ("pelicula_duracion", "ingreso"),       ("pelicula_costo_reposicion", "ingreso")],
    "renta_fin_de_semana":   [("veces_alquilada_pelicula", "ingreso"),     ("alquileres_por_cliente", "ingreso"),  ("pelicula_precio_renta", "ingreso")],
    "cliente_prefiere_genero":[("alquileres_por_cliente", "variedad_generos_cliente"), ("pelicula_precio_renta", "variedad_generos_cliente"), ("ingreso", "variedad_generos_cliente")],
    "renta_larga":           [("pelicula_duracion", "duracion_alquiler"),  ("pelicula_precio_renta", "duracion_alquiler"), ("pelicula_costo_reposicion", "duracion_alquiler")],
    "rango_precio_renta":    [("pelicula_precio_renta", "ingreso"),        ("pelicula_duracion", "ingreso"),       ("pelicula_costo_reposicion", "ingreso")],
    "grupo_edad_pelicula":   [("pelicula_duracion", "pelicula_costo_reposicion"), ("pelicula_precio_renta", "pelicula_costo_reposicion"), ("ingreso", "pelicula_costo_reposicion")],
    "nivel_fidelidad_cliente":[("alquileres_por_cliente", "variedad_generos_cliente"), ("alquileres_por_cliente", "ingreso"), ("alquileres_por_cliente", "pelicula_costo_reposicion")],
    "popularidad_pelicula":  [("veces_alquilada_pelicula", "pelicula_precio_renta"), ("veces_alquilada_pelicula", "ingreso"), ("veces_alquilada_pelicula", "pelicula_duracion")],
}

# Colores y nombres para los puntos de los scatter plots por clase
COLORES_POR_ETIQUETA = {
    "es_ingreso_alto":        {0: "steelblue",  1: "orange"},
    "renta_fin_de_semana":    {0: "purple",     1: "green"},
    "cliente_prefiere_genero":{0: "darkred",    1: "cyan"},
    "renta_larga":            {0: "olive",      1: "magenta"},
    "rango_precio_renta":     {0: "coral",      1: "teal",       2: "indigo"},
    "grupo_edad_pelicula":    {0: "gold",       1: "darkorchid", 2: "crimson"},
    "nivel_fidelidad_cliente":{0: "saddlebrown",1: "darkgray",   2: "goldenrod"},
    "popularidad_pelicula":   {0: "salmon",     1: "limegreen",  2: "blueviolet"},
}

NOMBRES_DE_CLASES_POR_ETIQUETA = {
    "es_ingreso_alto":        {0: "Normal",       1: "Alto"},
    "renta_fin_de_semana":    {0: "Laborable",    1: "Fin de Semana"},
    "cliente_prefiere_genero":{0: "Baja Afinidad",1: "Alta Afinidad"},
    "renta_larga":            {0: "Corto",        1: "Largo"},
    "rango_precio_renta":     {0: "Económico",    1: "Estándar",   2: "Premium"},
    "grupo_edad_pelicula":    {0: "Infantil",     1: "Adolescentes", 2: "Adultos"},
    "nivel_fidelidad_cliente":{0: "Bronce",       1: "Plata",      2: "Oro"},
    "popularidad_pelicula":   {0: "Baja",         1: "Media",      2: "Alta"},
}


# --------------------------------------------------------------------------
# Función auxiliar
# --------------------------------------------------------------------------

def cargar_dataset_para_eda() -> pd.DataFrame:
    """Carga el dataset base desde el Paso 2 y valida que exista."""
    if not os.path.exists(RUTA_DATASET_BASE):
        raise FileNotFoundError(
            f"No se encontró el dataset en: {RUTA_DATASET_BASE}\n"
            "Ejecuta primero: python paso_2_dataset/construir_dataset.py"
        )
    df = pd.read_csv(RUTA_DATASET_BASE)
    print(f"  Dataset cargado: {df.shape[0]:,} filas × {df.shape[1]} columnas")
    return df


# --------------------------------------------------------------------------
# Análisis 1: Matrices de correlación y covarianza
# --------------------------------------------------------------------------

def generar_matriz_de_covarianza_de_predictores(df: pd.DataFrame):
    """Genera y guarda la heatmap de covarianza entre las variables predictoras."""
    matriz_covarianza = df[VARIABLES_PREDICTORAS].cov()
    fig, eje = plt.subplots(figsize=(11, 9))
    imagen = eje.imshow(matriz_covarianza.values, cmap="Blues")
    plt.colorbar(imagen, ax=eje)
    eje.set_xticks(range(len(VARIABLES_PREDICTORAS)))
    eje.set_xticklabels(VARIABLES_PREDICTORAS, rotation=45, ha="right", fontsize=8)
    eje.set_yticks(range(len(VARIABLES_PREDICTORAS)))
    eje.set_yticklabels(VARIABLES_PREDICTORAS, fontsize=8)
    for fila in range(len(VARIABLES_PREDICTORAS)):
        for col in range(len(VARIABLES_PREDICTORAS)):
            eje.text(col, fila, f"{matriz_covarianza.values[fila, col]:.2f}",
                     ha="center", va="center", color="black", fontsize=7)
    eje.set_title("Matriz de Covarianza — Variables Predictoras (Features)", fontsize=13, pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_MATRICES, "covarianza_predictores.png"), dpi=150)
    plt.close()
    print("    ✓ covarianza_predictores.png")


def generar_matriz_de_correlacion_de_predictores(df: pd.DataFrame):
    """Genera y guarda la heatmap de correlación de Pearson entre las variables predictoras."""
    correlacion_pearson = df[VARIABLES_PREDICTORAS].corr(method="pearson")
    fig, eje = plt.subplots(figsize=(11, 9))
    imagen = eje.imshow(correlacion_pearson.values, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(imagen, ax=eje)
    eje.set_xticks(range(len(VARIABLES_PREDICTORAS)))
    eje.set_xticklabels(VARIABLES_PREDICTORAS, rotation=45, ha="right", fontsize=8)
    eje.set_yticks(range(len(VARIABLES_PREDICTORAS)))
    eje.set_yticklabels(VARIABLES_PREDICTORAS, fontsize=8)
    for fila in range(len(VARIABLES_PREDICTORAS)):
        for col in range(len(VARIABLES_PREDICTORAS)):
            eje.text(col, fila, f"{correlacion_pearson.values[fila, col]:.2f}",
                     ha="center", va="center", color="black", fontsize=8)
    eje.set_title("Correlación de Pearson — Variables Predictoras (Features)", fontsize=13, pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_MATRICES, "correlacion_predictores.png"), dpi=150)
    plt.close()
    print("    ✓ correlacion_predictores.png")


def generar_matrices_de_etiquetas(df: pd.DataFrame):
    """Genera covarianza y correlación de las 8 etiquetas objetivo."""
    todas_las_etiquetas = ETIQUETAS_BINARIAS + ETIQUETAS_MULTICLASE

    for nombre_analisis, metodo_corr, mapa_color in [
        ("covarianza", None,       "Blues"),
        ("correlacion", "pearson", "coolwarm"),
    ]:
        if metodo_corr:
            matriz = df[todas_las_etiquetas].corr(method=metodo_corr)
        else:
            matriz = df[todas_las_etiquetas].cov()

        fig, eje = plt.subplots(figsize=(9, 7))
        params_imagen = {"cmap": mapa_color}
        if metodo_corr:
            params_imagen.update({"vmin": -1, "vmax": 1})
        imagen = eje.imshow(matriz.values, **params_imagen)
        plt.colorbar(imagen, ax=eje)
        eje.set_xticks(range(len(todas_las_etiquetas)))
        eje.set_xticklabels(todas_las_etiquetas, rotation=45, ha="right", fontsize=9)
        eje.set_yticks(range(len(todas_las_etiquetas)))
        eje.set_yticklabels(todas_las_etiquetas, fontsize=9)
        for fila in range(len(todas_las_etiquetas)):
            for col in range(len(todas_las_etiquetas)):
                eje.text(col, fila, f"{matriz.values[fila, col]:.2f}",
                         ha="center", va="center", color="black", fontsize=8)
        titulo = "Correlación" if metodo_corr else "Covarianza"
        eje.set_title(f"{titulo} — Todas las Etiquetas Objetivo (8 targets)", fontsize=12, pad=15)
        plt.tight_layout()
        plt.savefig(os.path.join(DIR_MATRICES, f"{nombre_analisis}_etiquetas.png"), dpi=150)
        plt.close()
        print(f"    ✓ {nombre_analisis}_etiquetas.png")


# --------------------------------------------------------------------------
# Análisis 2: Histogramas de distribución con ajuste normal (KS test)
# --------------------------------------------------------------------------

def generar_histogramas_de_distribucion(df: pd.DataFrame):
    """
    Genera un histograma por variable predictora con la curva de densidad
    gaussiana ajustada y el p-valor del test de Kolmogorov-Smirnov.
    """
    for nombre_columna in VARIABLES_PREDICTORAS:
        valores = df[nombre_columna].dropna()
        media, desviacion = stats.norm.fit(valores)
        estadistico_ks, p_valor_ks = stats.kstest(valores, "norm", args=(media, desviacion))

        fig, eje = plt.subplots(figsize=(6, 4.5))
        eje.hist(valores, bins=30, density=True, alpha=0.65, color="#4c72b0", edgecolor="white")

        rango_x = np.linspace(valores.min(), valores.max(), 200)
        eje.plot(rango_x, stats.norm.pdf(rango_x, media, desviacion), "k-", linewidth=2, label="Curva Normal")

        eje.set_title(f"Distribución de '{nombre_columna}'\nKS p-valor: {p_valor_ks:.2e} | μ={media:.2f}, σ={desviacion:.2f}", fontsize=10)
        eje.set_xlabel("Valor escalado")
        eje.set_ylabel("Densidad de probabilidad")
        eje.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(DIR_DISTRIBUCIONES, f"dist_{nombre_columna}.png"), dpi=130)
        plt.close()

    print(f"    ✓ {len(VARIABLES_PREDICTORAS)} histogramas generados en '{DIR_DISTRIBUCIONES}'")


# --------------------------------------------------------------------------
# Análisis 3: Gráficos Q-Q de normalidad
# --------------------------------------------------------------------------

def generar_graficos_qq_de_normalidad(df: pd.DataFrame):
    """
    Genera un gráfico Q-Q (Quantile-Quantile) por variable predictora para
    evaluar visualmente si los datos siguen una distribución normal.
    """
    for nombre_columna in VARIABLES_PREDICTORAS:
        fig, eje = plt.subplots(figsize=(6, 4.5))
        stats.probplot(df[nombre_columna].dropna().values, dist="norm", plot=eje)
        eje.set_title(f"Gráfico Q-Q — '{nombre_columna}'", fontsize=11)
        eje.set_xlabel("Cuantiles teóricos (Distribución Normal)")
        eje.set_ylabel("Cuantiles observados (Datos)")
        plt.tight_layout()
        plt.savefig(os.path.join(DIR_QQPLOTS, f"qq_{nombre_columna}.png"), dpi=130)
        plt.close()

    print(f"    ✓ {len(VARIABLES_PREDICTORAS)} gráficos Q-Q generados en '{DIR_QQPLOTS}'")


# --------------------------------------------------------------------------
# Análisis 4: Boxplots de features segmentados por etiqueta
# --------------------------------------------------------------------------

def generar_boxplots_por_etiqueta(df: pd.DataFrame):
    """
    Para cada combinación (etiqueta × variable predictora), genera un boxplot
    que muestra cómo se distribuye la variable dentro de cada clase de la etiqueta.
    """
    todas_las_etiquetas = ETIQUETAS_BINARIAS + ETIQUETAS_MULTICLASE
    total_graficos = 0

    for nombre_etiqueta in todas_las_etiquetas:
        for nombre_variable in VARIABLES_PREDICTORAS:
            clases_unicas = sorted(df[nombre_etiqueta].unique())
            grupos_de_datos = [df[df[nombre_etiqueta] == clase][nombre_variable].dropna().values for clase in clases_unicas]

            fig, eje = plt.subplots(figsize=(5.5, 4.5))
            eje.boxplot(grupos_de_datos, patch_artist=True)
            eje.set_xticklabels(clases_unicas, fontsize=9)
            eje.set_xlabel(f"Clases de '{nombre_etiqueta}'")
            eje.set_ylabel(nombre_variable)
            eje.set_title(f"'{nombre_variable}'\nagrupado por: {nombre_etiqueta}", fontsize=10)
            plt.tight_layout()

            nombre_archivo = f"boxplot__{nombre_etiqueta}__{nombre_variable}.png"
            plt.savefig(os.path.join(DIR_BOXPLOTS, nombre_archivo), dpi=120)
            plt.close()
            total_graficos += 1

    print(f"    ✓ {total_graficos} boxplots generados en '{DIR_BOXPLOTS}'")


# --------------------------------------------------------------------------
# Análisis 5: Scatter plots bivariados por etiqueta
# --------------------------------------------------------------------------

def generar_scatter_plots_bivariados(df: pd.DataFrame):
    """
    Para cada etiqueta, genera gráficos de dispersión entre los pares de variables
    más relevantes, coloreando cada punto según la clase a la que pertenece.
    Usa una muestra de 5,000 registros para evitar sobrecarga visual.
    """
    TAMANO_MUESTRA = 5_000
    df_muestra = df.sample(n=min(len(df), TAMANO_MUESTRA), random_state=42)
    total_graficos = 0

    for nombre_etiqueta, pares_xy in CRUCES_DE_DISPERSION_POR_ETIQUETA.items():
        mapa_colores = COLORES_POR_ETIQUETA[nombre_etiqueta]
        mapa_nombres = NOMBRES_DE_CLASES_POR_ETIQUETA[nombre_etiqueta]

        for variable_x, variable_y in pares_xy:
            colores_de_puntos = df_muestra[nombre_etiqueta].map(mapa_colores)

            fig, eje = plt.subplots(figsize=(6, 5))
            eje.scatter(df_muestra[variable_x], df_muestra[variable_y],
                        c=colores_de_puntos, alpha=0.4, s=10, linewidths=0)
            eje.set_xlabel(variable_x, fontsize=9)
            eje.set_ylabel(variable_y, fontsize=9)
            eje.set_title(f"{variable_x} vs {variable_y}\nSegmentado por: {nombre_etiqueta} (n={TAMANO_MUESTRA:,})", fontsize=9)

            leyenda = [mpatches.Patch(color=color, label=mapa_nombres[clase])
                       for clase, color in mapa_colores.items()]
            eje.legend(handles=leyenda, loc="best", fontsize=8)
            plt.tight_layout()

            nombre_archivo = f"scatter__{nombre_etiqueta}__{variable_x}__vs__{variable_y}.png"
            plt.savefig(os.path.join(DIR_DISPERSIONES, nombre_archivo), dpi=120)
            plt.close()
            total_graficos += 1

    print(f"    ✓ {total_graficos} scatter plots generados en '{DIR_DISPERSIONES}'")


# --------------------------------------------------------------------------
# Pipeline principal del EDA
# --------------------------------------------------------------------------

def ejecutar_analisis_exploratorio_completo():
    """Orquesta la ejecución de todos los análisis estadísticos del EDA."""
    print("\n=== PASO 3: ANÁLISIS EXPLORATORIO DE DATOS (EDA) ===")

    print("\nCargando dataset base...")
    df = cargar_dataset_para_eda()

    print("\n[1/5] Generando matrices de covarianza y correlación...")
    generar_matriz_de_covarianza_de_predictores(df)
    generar_matriz_de_correlacion_de_predictores(df)
    generar_matrices_de_etiquetas(df)

    print("\n[2/5] Generando histogramas de distribución...")
    generar_histogramas_de_distribucion(df)

    print("\n[3/5] Generando gráficos Q-Q de normalidad...")
    generar_graficos_qq_de_normalidad(df)

    print("\n[4/5] Generando boxplots por etiqueta...")
    generar_boxplots_por_etiqueta(df)

    print("\n[5/5] Generando scatter plots bivariados...")
    generar_scatter_plots_bivariados(df)

    print("\n✓ EDA completado. Todas las imágenes se encuentran en:")
    print(f"  {DIRECTORIO_IMAGENES}")


# --------------------------------------------------------------------------
# Ejecución directa: python analisis_exploratorio.py
# --------------------------------------------------------------------------
if __name__ == "__main__":
    ejecutar_analisis_exploratorio_completo()
