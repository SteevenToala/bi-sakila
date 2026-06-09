# =============================================================================
# PASO 2 - CONSTRUCCIÓN DEL DATASET BASE
# Archivo: construir_dataset.py
#
# Responsabilidad:
#   Tomar el DataFrame crudo de alquileres extraído desde MongoDB y:
#     1. Calcular variables de apoyo (popularidad, frecuencia, diversidad)
#     2. Calcular popularidad del elenco de actores
#     3. Definir 4 etiquetas binarias de clasificación
#     4. Definir 4 etiquetas multiclase de clasificación
#     5. Aplicar escalado Z-score y Min-Max a las variables numéricas
#     6. Aplicar One-Hot Encoding a las variables categóricas nominales
#     7. Guardar el dataset limpio y listo para la red neuronal en CSV
# =============================================================================

import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

# Permite importar el módulo del paso anterior desde la ruta del proyecto
DIRECTORIO_PROYECTO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(DIRECTORIO_PROYECTO, "paso_1_extraccion"))

DIRECTORIO_ACTUAL   = os.path.dirname(os.path.abspath(__file__))
RUTA_SALIDA_DATASET = os.path.join(DIRECTORIO_ACTUAL, "dataset_base.csv")
RUTA_SALIDA_NEURONAL = os.path.join(DIRECTORIO_ACTUAL, "dataset_neuronal.csv")

# --------------------------------------------------------------------------
# Columnas que se estandarizan con Z-score (variables con distribución libre
# o con valores extremos que distorsionarían el entrenamiento si no se escalan)
# --------------------------------------------------------------------------
COLUMNAS_A_ESTANDARIZAR_ZSCORE = [
    "ingreso",
    "veces_alquilada_pelicula",
    "alquileres_por_cliente",
    "duracion_alquiler",
    "pelicula_costo_reposicion",
]

# --------------------------------------------------------------------------
# Columnas que se normalizan con Min-Max [0, 1] (variables acotadas
# con rango conocido y sin valores extremos significativos)
# --------------------------------------------------------------------------
COLUMNAS_A_NORMALIZAR_MINMAX = [
    "pelicula_duracion",
    "pelicula_precio_renta",
    "variedad_generos_cliente",
    "pelicula_cantidad_actores",
    "pelicula_popularidad_actores",
]


# --------------------------------------------------------------------------
# PASO 2A: Ingeniería de características (Feature Engineering)
# --------------------------------------------------------------------------

def calcular_popularidad_de_cada_pelicula(df_alquileres: pd.DataFrame) -> pd.Series:
    """Cuenta cuántas veces fue rentada cada película en toda la historia."""
    return df_alquileres["pelicula_ref"].value_counts().rename("veces_alquilada_pelicula")


def calcular_frecuencia_de_alquiler_por_cliente(df_alquileres: pd.DataFrame) -> pd.Series:
    """Cuenta el total histórico de alquileres de cada cliente."""
    return df_alquileres["cliente_ref"].value_counts().rename("alquileres_por_cliente")


def calcular_diversidad_de_generos_por_cliente(df_alquileres: pd.DataFrame) -> pd.Series:
    """
    Calcula cuántos géneros distintos ha consumido cada cliente.
    Resultado: un valor por cada fila del DataFrame original (aligned por índice).
    """
    return (
        df_alquileres
        .groupby("cliente_ref")["categoria_nombre"]
        .transform("nunique")
    )


def calcular_popularidad_promedio_del_elenco(df_alquileres: pd.DataFrame) -> pd.Series:
    """
    Calcula la popularidad promedio del elenco para cada transacción.
    La popularidad de un actor = cantidad de veces que aparece en rentas del dataset.
    Se promedian los valores de todos los actores de la película.
    """
    # Desanidar la lista de actores y contar cuántas veces aparece cada uno
    todos_los_actores = df_alquileres["pelicula_actores"].explode()
    frecuencia_por_actor = todos_los_actores.value_counts()

    def _promedio_popularidad_del_reparto(lista_actores: list) -> float:
        if not isinstance(lista_actores, list) or len(lista_actores) == 0:
            return 0.0
        popularidades = [frecuencia_por_actor.get(actor, 0.0) for actor in lista_actores]
        return float(np.mean(popularidades))

    return df_alquileres["pelicula_actores"].apply(_promedio_popularidad_del_reparto)


# --------------------------------------------------------------------------
# PASO 2B: Definición de etiquetas objetivo (Target Engineering)
# --------------------------------------------------------------------------

def crear_etiqueta_es_ingreso_alto(df: pd.DataFrame) -> pd.Series:
    """
    Etiqueta binaria: 1 si el ingreso es >= al percentil 75 dentro de su género.
    Compara cada transacción contra la medida de su propio género para ser justo.
    """
    umbral_percentil_75_por_genero = df.groupby("categoria_nombre")["ingreso"].transform(
        lambda ingresos_del_genero: ingresos_del_genero.quantile(0.75)
    )
    return (df["ingreso"] >= umbral_percentil_75_por_genero).astype(int)


def crear_etiqueta_renta_en_fin_de_semana(df: pd.DataFrame) -> pd.Series:
    """
    Etiqueta binaria: 1 si la transacción ocurrió un viernes (4), sábado (5) o domingo (6).
    """
    return df["tiempo_fecha"].dt.dayofweek.isin([4, 5, 6]).astype(int)


def crear_etiqueta_cliente_prefiere_este_genero(df: pd.DataFrame) -> pd.Series:
    """
    Etiqueta binaria: 1 si el cliente alquiló este género más que su promedio personal.
    Detecta si el cliente tiene una afinidad real por el género de la transacción.
    """
    # Contar cuántas veces el cliente alquiló cada género específico
    conteo_por_cliente_y_genero = (
        df.groupby(["cliente_ref", "categoria_nombre"])
        .size()
        .reset_index(name="cantidad_veces_este_genero")
    )
    # Calcular el promedio de consumo por género de ese cliente
    promedio_por_cliente = (
        conteo_por_cliente_y_genero
        .groupby("cliente_ref")["cantidad_veces_este_genero"]
        .mean()
        .reset_index(name="promedio_de_generos_del_cliente")
    )

    df_con_conteo   = df.merge(conteo_por_cliente_y_genero, on=["cliente_ref", "categoria_nombre"], how="left")
    df_con_promedio = df_con_conteo.merge(promedio_por_cliente, on="cliente_ref", how="left")

    etiqueta = (df_con_promedio["cantidad_veces_este_genero"] > df_con_promedio["promedio_de_generos_del_cliente"]).astype(int)
    etiqueta.index = df.index  # Restaurar índice original
    return etiqueta


def crear_etiqueta_renta_de_larga_duracion(df: pd.DataFrame) -> pd.Series:
    """
    Etiqueta binaria: 1 si los días de posesión superan la mediana general del dataset.
    """
    mediana_global_de_dias = df["duracion_alquiler"].median()
    return (df["duracion_alquiler"] > mediana_global_de_dias).astype(int)


def crear_etiqueta_rango_de_precio_de_renta(df: pd.DataFrame) -> pd.Series:
    """
    Etiqueta multiclase (3 categorías):
      0 = Económico (precio <= $1.00)
      1 = Estándar  ($1.00 < precio <= $3.00)
      2 = Premium   (precio > $3.00)
    """
    return pd.cut(
        df["pelicula_precio_renta"],
        bins=[-np.inf, 1.00, 3.00, np.inf],
        labels=[0, 1, 2]
    ).astype(int)


def crear_etiqueta_grupo_de_audiencia_por_edad(df: pd.DataFrame) -> pd.Series:
    """
    Etiqueta multiclase basada en la clasificación de edad de la película:
      0 = Infantil/Familiar (G, PG)
      1 = Adolescentes      (PG-13)
      2 = Adultos           (R, NC-17)
    """
    mapa_clasificacion_a_grupo_edad = {"G": 0, "PG": 0, "PG-13": 1, "R": 2, "NC-17": 2}
    return df["pelicula_clasificacion"].map(mapa_clasificacion_a_grupo_edad).fillna(0).astype(int)


def crear_etiqueta_nivel_de_fidelidad_del_cliente(df: pd.DataFrame) -> pd.Series:
    """
    Etiqueta multiclase basada en el historial total de alquileres del cliente:
      0 = Bronce (1 - 20 alquileres)
      1 = Plata  (21 - 30 alquileres)
      2 = Oro    (más de 30 alquileres)
    """
    return pd.cut(
        df["alquileres_por_cliente"],
        bins=[-np.inf, 20, 30, np.inf],
        labels=[0, 1, 2]
    ).astype(int)


def crear_etiqueta_popularidad_de_la_pelicula(df: pd.DataFrame) -> pd.Series:
    """
    Etiqueta multiclase basada en cuántas veces fue rentada la película:
      0 = Baja popularidad  (1 - 12 rentas)
      1 = Media popularidad (13 - 22 rentas)
      2 = Alta popularidad  (más de 22 rentas)
    """
    return pd.cut(
        df["veces_alquilada_pelicula"],
        bins=[-np.inf, 12, 22, np.inf],
        labels=[0, 1, 2]
    ).astype(int)


# --------------------------------------------------------------------------
# PASO 2C: Función principal de construcción del dataset
# --------------------------------------------------------------------------

def construir_dataset_completo(df_alquileres: pd.DataFrame) -> pd.DataFrame:
    """
    Orquesta toda la ingeniería de variables sobre el DataFrame crudo de alquileres.
    Aplica feature engineering y target engineering y retorna el dataset base
    (sin escalado aún, para que el escalado se pueda hacer por separado).
    """
    df = df_alquileres.copy()

    # ---- Variables de apoyo ----
    codificador_genero         = LabelEncoder()
    df["id_genero"]            = codificador_genero.fit_transform(df["categoria_nombre"])
    df["veces_alquilada_pelicula"] = df.join(calcular_popularidad_de_cada_pelicula(df), on="pelicula_ref")["veces_alquilada_pelicula"]
    df["alquileres_por_cliente"]   = df.join(calcular_frecuencia_de_alquiler_por_cliente(df), on="cliente_ref")["alquileres_por_cliente"]
    df["variedad_generos_cliente"] = calcular_diversidad_de_generos_por_cliente(df)
    df["pelicula_popularidad_actores"] = calcular_popularidad_promedio_del_elenco(df)

    # ---- Etiquetas binarias ----
    df["es_ingreso_alto"]         = crear_etiqueta_es_ingreso_alto(df)
    df["renta_fin_de_semana"]     = crear_etiqueta_renta_en_fin_de_semana(df)
    df["cliente_prefiere_genero"] = crear_etiqueta_cliente_prefiere_este_genero(df)
    df["renta_larga"]             = crear_etiqueta_renta_de_larga_duracion(df)

    # ---- Etiquetas multiclase ----
    df["rango_precio_renta"]       = crear_etiqueta_rango_de_precio_de_renta(df)
    df["grupo_edad_pelicula"]      = crear_etiqueta_grupo_de_audiencia_por_edad(df)
    df["nivel_fidelidad_cliente"]  = crear_etiqueta_nivel_de_fidelidad_del_cliente(df)
    df["popularidad_pelicula"]     = crear_etiqueta_popularidad_de_la_pelicula(df)

    # ---- Codificaciones de apoyo ----
    codificador_cliente = LabelEncoder()
    codificador_pelicula = LabelEncoder()
    df["id_cliente"]  = codificador_cliente.fit_transform(df["cliente_ref"].astype(str))
    df["id_pelicula"] = codificador_pelicula.fit_transform(df["pelicula_ref"].astype(str))

    # ---- Selección de columnas finales ----
    columnas_finales = [
        "cliente_ref", "pelicula_ref", "pelicula_titulo", "categoria_nombre",
        "ingreso", "tiempo_fecha", "pelicula_duracion", "pelicula_precio_renta",
        "pelicula_clasificacion", "veces_alquilada_pelicula", "alquileres_por_cliente",
        "variedad_generos_cliente", "id_genero", "duracion_alquiler",
        "pelicula_costo_reposicion", "pelicula_actores", "pelicula_cantidad_actores",
        "pelicula_popularidad_actores",
        # Etiquetas
        "es_ingreso_alto", "renta_fin_de_semana", "cliente_prefiere_genero", "renta_larga",
        "rango_precio_renta", "grupo_edad_pelicula", "nivel_fidelidad_cliente", "popularidad_pelicula",
    ]
    return df[columnas_finales].copy()


def escalar_y_codificar_para_red_neuronal(df_base: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler, MinMaxScaler]:
    """
    Aplica el escalado y codificación al dataset base para prepararlo
    como vector de entrada de la red neuronal.

    Pasos:
      1. Estandarización Z-score para variables con outliers
      2. Normalización Min-Max para variables acotadas
      3. One-Hot Encoding para género y clasificación de edad

    Retorna el DataFrame transformado junto con los dos escaladores
    entrenados (necesarios para escalar nuevas predicciones en producción).
    """
    df = df_base.copy()

    # Convertir actores a texto antes de guardar
    df["pelicula_actores"] = df["pelicula_actores"].apply(
        lambda lista: ", ".join(lista) if isinstance(lista, list) else ""
    )

    print("  Aplicando estandarización Z-score...")
    escalador_zscore = StandardScaler()
    df[COLUMNAS_A_ESTANDARIZAR_ZSCORE] = escalador_zscore.fit_transform(df[COLUMNAS_A_ESTANDARIZAR_ZSCORE])

    print("  Aplicando normalización Min-Max [0, 1]...")
    escalador_minmax = MinMaxScaler()
    df[COLUMNAS_A_NORMALIZAR_MINMAX] = escalador_minmax.fit_transform(df[COLUMNAS_A_NORMALIZAR_MINMAX])

    print("  Aplicando One-Hot Encoding a género y clasificación de edad...")
    # Respaldar textos antes del OHE para conservarlos en el CSV
    df["categoria_nombre_texto"]      = df["categoria_nombre"]
    df["pelicula_clasificacion_texto"] = df["pelicula_clasificacion"]

    df = pd.get_dummies(df, columns=["categoria_nombre", "pelicula_clasificacion"], prefix=["cat", "clas"], dtype=int)

    # Renombrar columnas de respaldo a sus nombres originales
    df = df.rename(columns={
        "categoria_nombre_texto":       "categoria_nombre",
        "pelicula_clasificacion_texto": "pelicula_clasificacion",
    })

    return df, escalador_zscore, escalador_minmax


# --------------------------------------------------------------------------
# PASO 2D: Función de pipeline completo — extrae, construye y guarda
# --------------------------------------------------------------------------

def ejecutar_pipeline_de_construccion():
    """
    Ejecuta el pipeline completo:
      1. Extrae datos desde MongoDB (Paso 1)
      2. Construye el dataset base con todas las variables y etiquetas
      3. Aplica escalado y codificación
      4. Guarda los resultados en CSV
    """
    from conexion_mongodb import cargar_todos_los_dataframes

    print("\n=== PASO 2: CONSTRUCCIÓN DEL DATASET ===")

    print("Cargando datos desde MongoDB (Paso 1)...")
    datos_mongo = cargar_todos_los_dataframes()
    df_alquileres_crudo = datos_mongo["alquileres"]
    print(f"  Registros cargados: {len(df_alquileres_crudo):,}")

    print("\nConstruyendo variables e ingeniería de etiquetas...")
    df_base = construir_dataset_completo(df_alquileres_crudo)
    print(f"  Dataset base construido: {df_base.shape}")

    print("\nAplicando escalado y codificación para la red neuronal...")
    df_neuronal, escalador_z, escalador_mm = escalar_y_codificar_para_red_neuronal(df_base)

    print(f"\nGuardando datasets en disco...")
    df_base.to_csv(RUTA_SALIDA_DATASET, index=False, encoding="utf-8-sig")
    print(f"  ✓ Dataset base guardado:     {RUTA_SALIDA_DATASET}")

    df_neuronal.to_csv(RUTA_SALIDA_NEURONAL, index=False, encoding="utf-8")
    print(f"  ✓ Dataset neuronal guardado: {RUTA_SALIDA_NEURONAL}")
    print(f"  ✓ Shape final del dataset neuronal: {df_neuronal.shape}")

    print("\nColumnas listas para la red neuronal:")
    columnas_entrada = [c for c in df_neuronal.columns if c not in [
        "es_ingreso_alto", "renta_fin_de_semana", "cliente_prefiere_genero", "renta_larga",
        "rango_precio_renta", "grupo_edad_pelicula", "nivel_fidelidad_cliente", "popularidad_pelicula",
    ]]
    print(f"  Total de variables de entrada (X): {len(columnas_entrada)}")
    print(f"  Total de variables objetivo  (Y): 8 etiquetas (4 binarias + 4 multiclase)")


# --------------------------------------------------------------------------
# Ejecución directa: python construir_dataset.py
# --------------------------------------------------------------------------
if __name__ == "__main__":
    ejecutar_pipeline_de_construccion()
