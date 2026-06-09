# =============================================================================
# PASO 4 - ENTRENAMIENTO DE LA RED NEURONAL
# Archivo: entrenar_red_neuronal.py
#
# Responsabilidad:
#   Leer el dataset neuronal del Paso 2 y entrenar 5 arquitecturas diferentes
#   de redes neuronales multitarea usando TensorFlow/Keras. Cada modelo tiene
#   hiperparámetros distintos (capas, neuronas, dropout, épocas, batch size).
#
#   Al finalizar:
#     - Guarda cada uno de los 5 modelos entrenados como archivo .keras
#     - Guarda el mejor modelo (mayor accuracy promedio) como modelo principal
#     - Guarda los dos escaladores (Z-score y Min-Max) en formato .joblib
#       para usarlos luego en predicciones nuevas sin reentrenar
#     - Imprime una tabla comparativa con la precisión de cada arquitectura
# =============================================================================

import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Permite importar desde el paso anterior
DIRECTORIO_PROYECTO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(DIRECTORIO_PROYECTO, "paso_2_dataset"))

DIRECTORIO_ACTUAL        = os.path.dirname(os.path.abspath(__file__))
RUTA_DATASET_NEURONAL    = os.path.join(DIRECTORIO_PROYECTO, "paso_2_dataset", "dataset_neuronal.csv")
RUTA_MODELO_PRINCIPAL    = os.path.join(DIRECTORIO_ACTUAL, "modelo_principal.keras")
RUTA_ESCALADOR_ZSCORE    = os.path.join(DIRECTORIO_ACTUAL, "escalador_zscore.joblib")
RUTA_ESCALADOR_MINMAX    = os.path.join(DIRECTORIO_ACTUAL, "escalador_minmax.joblib")
RUTA_METRICAS_JSON       = os.path.join(DIRECTORIO_ACTUAL, "metricas_entrenamiento.json")

# --------------------------------------------------------------------------
# Nombres de las 8 etiquetas objetivo (Y)
# --------------------------------------------------------------------------
NOMBRES_ETIQUETAS_OBJETIVO = [
    "es_ingreso_alto",         # Binaria
    "renta_fin_de_semana",     # Binaria
    "cliente_prefiere_genero", # Binaria
    "renta_larga",             # Binaria
    "rango_precio_renta",      # Multiclase (3 categorías)
    "grupo_edad_pelicula",     # Multiclase (3 categorías)
    "nivel_fidelidad_cliente", # Multiclase (3 categorías)
    "popularidad_pelicula",    # Multiclase (3 categorías)
]

# --------------------------------------------------------------------------
# Definición de las 5 arquitecturas a comparar
# --------------------------------------------------------------------------
CONFIGURACIONES_DE_ARQUITECTURAS = [
    {
        "nombre":       "Base",
        "descripcion":  "2 capas medianas, configuración de referencia equilibrada.",
        "capas_densas": [128, 64],
        "dropout":      0.20,
        "epocas_max":   40,
        "batch_size":   64,
    },
    {
        "nombre":       "Profunda",
        "descripcion":  "4 capas decrecientes, captura patrones de alta complejidad.",
        "capas_densas": [256, 128, 64, 32],
        "dropout":      0.30,
        "epocas_max":   50,
        "batch_size":   64,
    },
    {
        "nombre":       "Ancha",
        "descripcion":  "2 capas muy grandes, amplía la capacidad de representación.",
        "capas_densas": [512, 256],
        "dropout":      0.40,
        "epocas_max":   40,
        "batch_size":   128,
    },
    {
        "nombre":       "Rapida",
        "descripcion":  "2 capas pequeñas, entrenamiento más rápido con menos parámetros.",
        "capas_densas": [64, 32],
        "dropout":      0.10,
        "epocas_max":   30,
        "batch_size":   32,
    },
    {
        "nombre":       "Conservadora",
        "descripcion":  "3 capas del mismo ancho, aprendizaje estable y progresivo.",
        "capas_densas": [128, 128, 64],
        "dropout":      0.25,
        "epocas_max":   60,
        "batch_size":   32,
    },
]


# --------------------------------------------------------------------------
# Construcción dinámica de cada arquitectura de red neuronal
# --------------------------------------------------------------------------

def construir_red_neuronal_multitarea(numero_variables_entrada: int, config: dict):
    """
    Construye un modelo Keras de aprendizaje multitarea con la configuración dada.
    Las cinco arquitecturas varían: número de capas, neuronas por capa, dropout,
    épocas máximas y batch size. Todas usan Adam, ReLU e inicialización He.
    """
    import tensorflow as tf
    from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
    from tensorflow.keras.models import Model

    capa_entrada = Input(shape=(numero_variables_entrada,), name="entrada_variables_predictoras")

    capa_actual = capa_entrada
    for indice, numero_neuronas in enumerate(config["capas_densas"], start=1):
        capa_actual = Dense(numero_neuronas, activation="relu",
                            kernel_initializer="he_uniform",
                            name=f"capa_densa_{indice}")(capa_actual)
        capa_actual = BatchNormalization(name=f"normalizacion_{indice}")(capa_actual)
        capa_actual = Dropout(config["dropout"], name=f"dropout_{indice}")(capa_actual)

    salida_es_ingreso_alto         = Dense(1, activation="sigmoid", name="es_ingreso_alto")(capa_actual)
    salida_renta_fin_de_semana     = Dense(1, activation="sigmoid", name="renta_fin_de_semana")(capa_actual)
    salida_cliente_prefiere_genero = Dense(1, activation="sigmoid", name="cliente_prefiere_genero")(capa_actual)
    salida_renta_larga             = Dense(1, activation="sigmoid", name="renta_larga")(capa_actual)

    salida_rango_precio_renta      = Dense(3, activation="softmax", name="rango_precio_renta")(capa_actual)
    salida_grupo_edad_pelicula     = Dense(3, activation="softmax", name="grupo_edad_pelicula")(capa_actual)
    salida_nivel_fidelidad_cliente = Dense(3, activation="softmax", name="nivel_fidelidad_cliente")(capa_actual)
    salida_popularidad_pelicula    = Dense(3, activation="softmax", name="popularidad_pelicula")(capa_actual)

    modelo = Model(
        inputs=capa_entrada,
        outputs=[
            salida_es_ingreso_alto, salida_renta_fin_de_semana,
            salida_cliente_prefiere_genero, salida_renta_larga,
            salida_rango_precio_renta, salida_grupo_edad_pelicula,
            salida_nivel_fidelidad_cliente, salida_popularidad_pelicula,
        ],
        name=f"red_neuronal_{config['nombre'].lower()}"
    )

    modelo.compile(
        optimizer="adam",
        loss={
            "es_ingreso_alto":         "binary_crossentropy",
            "renta_fin_de_semana":     "binary_crossentropy",
            "cliente_prefiere_genero": "binary_crossentropy",
            "renta_larga":             "binary_crossentropy",
            "rango_precio_renta":      "sparse_categorical_crossentropy",
            "grupo_edad_pelicula":     "sparse_categorical_crossentropy",
            "nivel_fidelidad_cliente": "sparse_categorical_crossentropy",
            "popularidad_pelicula":    "sparse_categorical_crossentropy",
        },
        metrics={etiqueta: "accuracy" for etiqueta in NOMBRES_ETIQUETAS_OBJETIVO},
    )

    return modelo


# --------------------------------------------------------------------------
# Guardado de escaladores para producción
# --------------------------------------------------------------------------

def guardar_escaladores_de_produccion(df_alquileres_crudos: pd.DataFrame = None):
    """
    Carga (o genera) los escaladores entrenados con los datos de MongoDB
    y los guarda como archivos .joblib para usarlos en la app Flask.

    Si no se pueden cargar datos de MongoDB, crea escaladores con valores
    representativos del dominio del negocio como alternativa de respaldo.
    """
    COLUMNAS_ZSCORE = ["ingreso", "veces_alquilada_pelicula", "alquileres_por_cliente", "duracion_alquiler", "pelicula_costo_reposicion"]
    COLUMNAS_MINMAX = ["pelicula_duracion", "pelicula_precio_renta", "variedad_generos_cliente", "pelicula_cantidad_actores", "pelicula_popularidad_actores"]

    escaladores_cargados_correctamente = False

    if df_alquileres_crudos is not None:
        try:
            sys.path.insert(0, os.path.join(DIRECTORIO_PROYECTO, "paso_2_dataset"))
            from construir_dataset import construir_dataset_completo

            print("  Ajustando escaladores con datos reales de MongoDB...")
            df_variables_crudas = construir_dataset_completo(df_alquileres_crudos)

            escalador_zscore = StandardScaler()
            escalador_zscore.fit(df_variables_crudas[COLUMNAS_ZSCORE])
            joblib.dump(escalador_zscore, RUTA_ESCALADOR_ZSCORE)

            escalador_minmax = MinMaxScaler()
            escalador_minmax.fit(df_variables_crudas[COLUMNAS_MINMAX])
            joblib.dump(escalador_minmax, RUTA_ESCALADOR_MINMAX)

            escaladores_cargados_correctamente = True
            print("  ✓ Escaladores guardados con datos reales.")
        except Exception as error_mongo:
            print(f"  [!] No se pudieron usar los datos de MongoDB: {error_mongo}")

    if not escaladores_cargados_correctamente:
        print("  Creando escaladores de respaldo con valores del dominio del negocio...")
        datos_limite_zscore = pd.DataFrame(
            [[1.0, 4.0, 12.0, 1.0, 9.99], [15.0, 34.0, 42.0, 10.0, 29.99]],
            columns=COLUMNAS_ZSCORE
        )
        escalador_zscore = StandardScaler()
        escalador_zscore.fit(datos_limite_zscore)
        joblib.dump(escalador_zscore, RUTA_ESCALADOR_ZSCORE)

        datos_limite_minmax = pd.DataFrame(
            [[46.0, 0.99, 1.0, 1.0, 4.0], [185.0, 4.99, 16.0, 15.0, 342.0]],
            columns=COLUMNAS_MINMAX
        )
        escalador_minmax = MinMaxScaler()
        escalador_minmax.fit(datos_limite_minmax)
        joblib.dump(escalador_minmax, RUTA_ESCALADOR_MINMAX)
        print("  ✓ Escaladores de respaldo guardados.")


# --------------------------------------------------------------------------
# Pipeline principal de entrenamiento
# --------------------------------------------------------------------------

def ejecutar_entrenamiento_de_los_5_modelos():
    """
    Orquesta el entrenamiento experimental de las 5 arquitecturas.

    Flujo:
      1. Carga el dataset neuronal y lo divide en X (variables) e Y (etiquetas)
      2. Separa 80% entrenamiento / 20% testeo con semilla fija (reproducible)
      3. Por cada arquitectura: construye → entrena → evalúa → guarda .keras
      4. Selecciona el modelo con mayor accuracy promedio como modelo principal
      5. Imprime tabla resumen comparativa
    """
    import tensorflow as tf

    print("\n=== PASO 4: ENTRENAMIENTO DE LA RED NEURONAL ===")

    # ---- 1. Carga del dataset ----
    print(f"\nCargando dataset desde: {RUTA_DATASET_NEURONAL}")
    df_neuronal = pd.read_csv(RUTA_DATASET_NEURONAL, encoding="utf-8")
    print(f"  Shape del dataset: {df_neuronal.shape[0]:,} filas × {df_neuronal.shape[1]} columnas")

    columnas_numericas = df_neuronal.select_dtypes(include=[np.number]).columns.tolist()
    nombres_variables_entrada = [c for c in columnas_numericas if c not in NOMBRES_ETIQUETAS_OBJETIVO]
    print(f"  Variables de entrada (X): {len(nombres_variables_entrada)}")
    print(f"  Variables objetivo   (Y): {len(NOMBRES_ETIQUETAS_OBJETIVO)}")

    X_todas = df_neuronal[nombres_variables_entrada].values.astype(np.float32)
    Y_por_etiqueta = {etiqueta: df_neuronal[etiqueta].values for etiqueta in NOMBRES_ETIQUETAS_OBJETIVO}

    # ---- 2. División entrenamiento / testeo ----
    indices_dataset = np.arange(len(df_neuronal))
    indices_entrenamiento, indices_testeo = train_test_split(
        indices_dataset, test_size=0.20, random_state=42
    )

    X_entrenamiento = X_todas[indices_entrenamiento]
    X_testeo        = X_todas[indices_testeo]
    Y_entrenamiento = {e: Y_por_etiqueta[e][indices_entrenamiento] for e in NOMBRES_ETIQUETAS_OBJETIVO}
    Y_testeo        = {e: Y_por_etiqueta[e][indices_testeo]        for e in NOMBRES_ETIQUETAS_OBJETIVO}

    print(f"\n  Registros de entrenamiento: {len(X_entrenamiento):,} (80%)")
    print(f"  Registros de testeo:        {len(X_testeo):,} (20%)")

    # ---- 3. Entrenamiento de los 5 modelos ----
    import json

    total_registros = len(df_neuronal)
    n_entrenamiento = len(X_entrenamiento)
    n_testeo        = len(X_testeo)

    # Cálculo de suficiencia de datos (regla: ≥20 muestras × variables × clases)
    min_recomendado = len(nombres_variables_entrada) * 8 * 20
    factor_cobertura = round(n_entrenamiento / min_recomendado, 2)

    tabla_de_resultados = []
    mejor_accuracy_promedio = 0.0
    mejor_modelo_entrenado  = None
    mejor_nombre            = ""

    metricas_json = {
        "dataset": {
            "total_registros":    total_registros,
            "n_entrenamiento":    n_entrenamiento,
            "n_testeo":           n_testeo,
            "pct_entrenamiento":  80,
            "pct_testeo":         20,
            "semilla_division":   42,
            "variables_entrada":  len(nombres_variables_entrada),
            "etiquetas_salida":   len(NOMBRES_ETIQUETAS_OBJETIVO),
            "min_recomendado":    min_recomendado,
            "factor_cobertura":   factor_cobertura,
        },
        "tecnicas_anti_overfitting": [
            "BatchNormalization en cada capa oculta",
            "Dropout (desactivación aleatoria de neuronas por época)",
            "EarlyStopping con paciencia=5 sobre val_loss",
            "Inicialización He uniforme (estabiliza gradientes en capas ReLU)",
            "Validación interna 10% separada del entrenamiento",
            "División train/test estricta (test nunca visto durante ajuste)",
        ],
        "modelos": [],
    }

    callback_parada_temprana = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    print("\n=== INICIANDO EXPERIMENTACIÓN CON 5 ARQUITECTURAS ===")

    for numero_modelo, configuracion in enumerate(CONFIGURACIONES_DE_ARQUITECTURAS, start=1):
        print(f"\n--- Modelo {numero_modelo}/5: '{configuracion['nombre']}' ---")
        print(f"  Descripción: {configuracion['descripcion']}")
        print(f"  Capas:       {configuracion['capas_densas']}  |  Dropout: {configuracion['dropout']}")
        print(f"  Épocas máx:  {configuracion['epocas_max']}   |  Batch:   {configuracion['batch_size']}")

        modelo = construir_red_neuronal_multitarea(len(nombres_variables_entrada), configuracion)

        print(f"  Entrenando... (salida silenciada, espere)")
        historial = modelo.fit(
            X_entrenamiento,
            [Y_entrenamiento[e] for e in NOMBRES_ETIQUETAS_OBJETIVO],
            validation_split=0.10,
            epochs=configuracion["epocas_max"],
            batch_size=configuracion["batch_size"],
            callbacks=[callback_parada_temprana],
            verbose=0,
        )

        epocas_ejecutadas = len(historial.history["loss"])
        train_loss_final  = float(historial.history["loss"][-1])
        val_loss_final    = float(historial.history["val_loss"][-1])

        # Accuracy por salida en entrenamiento (última época)
        train_acc_vals = [v[-1] for k, v in historial.history.items()
                          if "accuracy" in k and not k.startswith("val_")]
        train_accuracy = float(np.mean(train_acc_vals)) if train_acc_vals else 0.0

        # Evaluación en testeo
        resultados_evaluacion = modelo.evaluate(
            X_testeo,
            [Y_testeo[e] for e in NOMBRES_ETIQUETAS_OBJETIVO],
            return_dict=True,
            verbose=0,
        )
        test_loss     = float(resultados_evaluacion.get("loss", 0.0))
        valores_acc   = [v for k, v in resultados_evaluacion.items() if "accuracy" in k]
        test_accuracy = float(np.mean(valores_acc))

        diferencia    = round((train_accuracy - test_accuracy) * 100, 2)
        estado_ovfit  = "OK" if diferencia < 5 else ("LEVE" if diferencia < 10 else "ALTO")

        print(f"  → Train accuracy: {train_accuracy*100:.2f}%  |  Test accuracy: {test_accuracy*100:.2f}%")
        print(f"  → Train loss:     {train_loss_final:.4f}     |  Test loss:      {test_loss:.4f}")
        print(f"  → Épocas reales:  {epocas_ejecutadas}  |  Overfitting gap: {diferencia:+.2f}pp [{estado_ovfit}]")

        # Guardar modelo individual
        nombre_archivo_modelo = f"modelo_v{numero_modelo}_{configuracion['nombre'].lower()}.keras"
        ruta_archivo_modelo   = os.path.join(DIRECTORIO_ACTUAL, nombre_archivo_modelo)
        modelo.save(ruta_archivo_modelo)
        print(f"  ✓ Guardado como: {nombre_archivo_modelo}")

        fila = {
            "Nº":              numero_modelo,
            "Nombre":          configuracion["nombre"],
            "Capas":           str(configuracion["capas_densas"]),
            "Dropout":         configuracion["dropout"],
            "Épocas máx":      configuracion["epocas_max"],
            "Épocas reales":   epocas_ejecutadas,
            "Batch":           configuracion["batch_size"],
            "Optimizador":     configuracion.get("optimizador", "adam"),
            "Activación":      configuracion.get("activacion", "relu"),
            "Regularización":  str(configuracion.get("regularizacion", "ninguna")),
            "Train Acc %":     round(train_accuracy * 100, 2),
            "Test Acc %":      round(test_accuracy  * 100, 2),
            "Train Loss":      round(train_loss_final, 4),
            "Test Loss":       round(test_loss, 4),
            "Gap pp":          diferencia,
            "Overfitting":     estado_ovfit,
            "Archivo":         nombre_archivo_modelo,
        }
        tabla_de_resultados.append(fila)
        metricas_json["modelos"].append(fila)

        if test_accuracy > mejor_accuracy_promedio:
            mejor_accuracy_promedio = test_accuracy
            mejor_modelo_entrenado  = modelo
            mejor_nombre            = configuracion["nombre"]

    # ---- 4. Guardar el mejor como modelo principal ----
    metricas_json["mejor_modelo"] = mejor_nombre
    metricas_json["mejor_test_accuracy"] = round(mejor_accuracy_promedio * 100, 2)

    print(f"\nGuardando el mejor modelo '{mejor_nombre}' ({mejor_accuracy_promedio*100:.2f}%) como principal...")
    mejor_modelo_entrenado.save(RUTA_MODELO_PRINCIPAL)
    print(f"  ✓ Guardado en: {RUTA_MODELO_PRINCIPAL}")

    with open(RUTA_METRICAS_JSON, "w", encoding="utf-8") as f:
        json.dump(metricas_json, f, ensure_ascii=False, indent=2)
    print(f"  ✓ Métricas guardadas en: {RUTA_METRICAS_JSON}")

    # ---- 5. Tabla resumen ----
    print("\n" + "=" * 110)
    print(f"{'Nº':<4} {'Nombre':<14} {'Train Acc%':<12} {'Test Acc%':<11} {'Train Loss':<12} {'Test Loss':<11} {'Gap pp':<9} {'Ovf':<6} {'Ep.'}")
    print("-" * 110)
    for fila in tabla_de_resultados:
        marca = " ← MEJOR" if fila["Test Acc %"] == max(f["Test Acc %"] for f in tabla_de_resultados) else ""
        print(f"{fila['Nº']:<4} {fila['Nombre']:<14} {fila['Train Acc %']:<12} {fila['Test Acc %']:<11} "
              f"{fila['Train Loss']:<12} {fila['Test Loss']:<11} {fila['Gap pp']:+.2f}{'':6} {fila['Overfitting']:<6} "
              f"{fila['Épocas reales']}{marca}")
    print("=" * 90)

    print("\n✓ Entrenamiento completado. El sistema Flask usará 'modelo_principal.keras'.")


# --------------------------------------------------------------------------
# Ejecución directa: python entrenar_red_neuronal.py
# --------------------------------------------------------------------------
if __name__ == "__main__":
    # Intentar guardar escaladores con datos reales de MongoDB primero
    try:
        sys.path.insert(0, os.path.join(DIRECTORIO_PROYECTO, "paso_1_extraccion"))
        from conexion_mongodb import cargar_todos_los_dataframes
        datos_mongo = cargar_todos_los_dataframes()
        guardar_escaladores_de_produccion(datos_mongo["alquileres"])
    except Exception as error_conexion:
        print(f"[!] MongoDB no disponible: {error_conexion}")
        guardar_escaladores_de_produccion()

    ejecutar_entrenamiento_de_los_5_modelos()
