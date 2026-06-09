# =============================================================================
# PASO 5 - APLICACIÓN WEB FLASK (INTERFAZ DE PREDICCIÓN)
# Archivo: app.py
#
# Responsabilidad:
#   Servir la interfaz web de predicción multitarea usando Flask.
#   Al recibir datos del formulario HTML, carga el modelo principal
#   entrenado en el Paso 4, escala los datos de entrada con los mismos
#   escaladores usados durante el entrenamiento y retorna las predicciones
#   en formato JSON para que el frontend las muestre al usuario.
#
# Cómo ejecutar:
#   cd paso_5_aplicacion_web
#   python app.py
#   Abrir: http://127.0.0.1:5000/
# =============================================================================

import os
import sys
import pandas as pd
import numpy as np
import joblib
from flask import Flask, render_template, request, jsonify

# --------------------------------------------------------------------------
# Rutas de archivos del proyecto
# --------------------------------------------------------------------------
DIRECTORIO_ACTUAL       = os.path.dirname(os.path.abspath(__file__))
DIRECTORIO_PROYECTO     = os.path.dirname(DIRECTORIO_ACTUAL)

RUTA_MODELO_PRINCIPAL   = os.path.join(DIRECTORIO_PROYECTO, "paso_4_entrenamiento", "modelo_principal.keras")
RUTA_ESCALADOR_ZSCORE   = os.path.join(DIRECTORIO_PROYECTO, "paso_4_entrenamiento", "escalador_zscore.joblib")
RUTA_ESCALADOR_MINMAX   = os.path.join(DIRECTORIO_PROYECTO, "paso_4_entrenamiento", "escalador_minmax.joblib")
RUTA_DATASET_NEURONAL   = os.path.join(DIRECTORIO_PROYECTO, "paso_2_dataset", "dataset_neuronal.csv")
RUTA_METRICAS_JSON      = os.path.join(DIRECTORIO_PROYECTO, "paso_4_entrenamiento", "metricas_entrenamiento.json")

# --------------------------------------------------------------------------
# Configuración de Flask
# --------------------------------------------------------------------------
app = Flask(__name__)

# --------------------------------------------------------------------------
# Metadatos del dominio: categorías de entrada para la red neuronal
# --------------------------------------------------------------------------
GENEROS_DE_PELICULAS = [
    "Acción", "Animación", "Ciencia ficción", "Clásicos", "Comedia", "Deportes",
    "Documental", "Drama", "Extranjero", "Familia", "Horror", "Juegos",
    "Música", "Niños", "Nuevo", "Viajar",
]

# LabelEncoder ordena alfabéticamente por código Unicode, igual que sorted().
# Este mapa reproduce exactamente el id_genero que se usó en el entrenamiento.
ID_GENERO_MAP = {genero: idx for idx, genero in enumerate(sorted(GENEROS_DE_PELICULAS))}

CLASIFICACIONES_DE_EDAD = ["G", "NC-17", "PG", "PG-13", "R"]

NOMBRES_ETIQUETAS_OBJETIVO = [
    "es_ingreso_alto", "renta_fin_de_semana", "cliente_prefiere_genero", "renta_larga",
    "rango_precio_renta", "grupo_edad_pelicula", "nivel_fidelidad_cliente", "popularidad_pelicula",
]

# Columnas que se estandarizan antes de predecir (deben coincidir con el entrenamiento)
COLUMNAS_ESTANDARIZADAS_ZSCORE = [
    "ingreso", "veces_alquilada_pelicula", "alquileres_por_cliente",
    "duracion_alquiler", "pelicula_costo_reposicion",
]

# Columnas que se normalizan antes de predecir (deben coincidir con el entrenamiento)
COLUMNAS_NORMALIZADAS_MINMAX = [
    "pelicula_duracion", "pelicula_precio_renta", "variedad_generos_cliente",
    "pelicula_cantidad_actores", "pelicula_popularidad_actores",
]

# --------------------------------------------------------------------------
# Lookups de MongoDB: películas y clientes precalculados al iniciar
# --------------------------------------------------------------------------
peliculas_lookup     = {}   # id_pelicula (int) -> dict con todos los campos
clientes_lookup      = {}   # id_cliente  (int) -> dict con nombre y métricas
lookups_mongo_listos = False


def _cargar_lookups_desde_mongo() -> bool:
    """
    Conecta a MongoDB BI_Final.alquileres y precalcula en memoria:
      - Para cada película: veces_alquilada, popularidad_actores y atributos estáticos.
      - Para cada cliente:  nombre_completo, alquileres totales, variedad de géneros.
    Se llama una sola vez al inicio. Si MongoDB no está disponible, retorna False
    y el formulario manual sigue funcionando como siempre.
    """
    global peliculas_lookup, clientes_lookup, lookups_mongo_listos
    try:
        from pymongo import MongoClient
        from collections import Counter, defaultdict

        mg = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=3000)
        mg.admin.command("ping")          # falla rápido si no hay conexión
        db = mg["BI_Final"]

        docs = list(db.alquileres.find({}, {
            "_id": 0, "pelicula": 1, "categoria": 1, "cliente": 1
        }))

        if not docs:
            return False

        # --- Popularidad global de cada actor ---
        conteo_actores = Counter()
        for doc in docs:
            for info in doc.get("pelicula", {}).get("actores", {}).values():
                nombre = info.get("nombre_actor", "")
                if nombre:
                    conteo_actores[nombre] += 1

        # --- Películas ---
        conteo_peli = Counter()
        datos_peli_tmp = {}
        for doc in docs:
            peli = doc.get("pelicula", {})
            cat  = doc.get("categoria", {})
            id_p = peli.get("id_pelicula")
            if id_p is None:
                continue
            conteo_peli[id_p] += 1
            if id_p not in datos_peli_tmp:
                nombres_actores = [
                    info.get("nombre_actor", "")
                    for info in peli.get("actores", {}).values()
                    if info.get("nombre_actor")
                ]
                pop = (
                    round(sum(conteo_actores[n] for n in nombres_actores) / len(nombres_actores), 2)
                    if nombres_actores else 0.0
                )
                datos_peli_tmp[id_p] = {
                    "id_pelicula":          id_p,
                    "titulo":               peli.get("titulo", ""),
                    "duracion":             peli.get("duracion", 120),
                    "clasificacion":        peli.get("clasificacion", "PG"),
                    "precio_renta":         peli.get("precio_renta", 2.99),
                    "costo_reposicion":     peli.get("costo_reposicion", 19.99),
                    "cantidad_actores":     peli.get("cantidad_actores", 5),
                    "categoria":            cat.get("nombre", "Comedia"),
                    "popularidad_actores":  pop,
                }

        for id_p, data in datos_peli_tmp.items():
            data["veces_alquilada"] = conteo_peli[id_p]
        peliculas_lookup = datos_peli_tmp

        # --- Clientes ---
        conteo_cli  = Counter()
        generos_cli = defaultdict(set)
        nombres_cli = {}
        for doc in docs:
            cli = doc.get("cliente", {})
            cat = doc.get("categoria", {})
            id_c = cli.get("id_cliente")
            if id_c is None:
                continue
            conteo_cli[id_c] += 1
            generos_cli[id_c].add(cat.get("nombre", ""))
            if id_c not in nombres_cli:
                nombres_cli[id_c] = cli.get("nombre_completo") or f"Cliente {id_c}"

        clientes_lookup = {
            id_c: {
                "id_cliente":       id_c,
                "nombre_completo":  nombre,
                "alquileres":       conteo_cli[id_c],
                "variedad_generos": len(generos_cli[id_c]),
            }
            for id_c, nombre in nombres_cli.items()
        }

        mg.close()
        lookups_mongo_listos = True
        print(f"  ✓ Lookups MongoDB: {len(peliculas_lookup)} películas, {len(clientes_lookup)} clientes.")
        return True

    except Exception as err:
        print(f"  [!] Lookups MongoDB no disponibles: {err}")
        return False


# --------------------------------------------------------------------------
# Carga lazy del modelo y escaladores (se cargan una sola vez en memoria)
# --------------------------------------------------------------------------
modelo_cargado    = None
escalador_zscore  = None
escalador_minmax  = None


def cargar_modelo_y_escaladores_si_no_estan_en_memoria():
    """
    Carga el modelo Keras y los escaladores joblib en variables globales.
    Solo se ejecuta la primera vez que se recibe una predicción.
    Retorna True si todo se cargó correctamente, False si falta algún archivo.
    """
    global modelo_cargado, escalador_zscore, escalador_minmax

    archivos_requeridos = [RUTA_MODELO_PRINCIPAL, RUTA_ESCALADOR_ZSCORE, RUTA_ESCALADOR_MINMAX]
    todos_los_archivos_existen = all(os.path.exists(ruta) for ruta in archivos_requeridos)

    if not todos_los_archivos_existen:
        print("[!] Faltan archivos del modelo. Ejecuta primero: python paso_4_entrenamiento/entrenar_red_neuronal.py")
        return False

    if modelo_cargado is None:
        import tensorflow as tf
        modelo_cargado   = tf.keras.models.load_model(RUTA_MODELO_PRINCIPAL)
        escalador_zscore = joblib.load(RUTA_ESCALADOR_ZSCORE)
        escalador_minmax = joblib.load(RUTA_ESCALADOR_MINMAX)
        print("  ✓ Modelo y escaladores cargados en memoria.")

    return True


# --------------------------------------------------------------------------
# Preparación del vector de entrada para la predicción
# --------------------------------------------------------------------------

def preparar_vector_de_entrada(datos_del_formulario: dict) -> np.ndarray:
    """
    Toma el JSON recibido del formulario web y construye el vector de entrada
    que la red neuronal espera recibir, en el mismo orden y escalado que
    durante el entrenamiento.

    Pasos:
      1. Extraer y convertir variables numéricas continuas
      2. Aplicar escalado Z-score y Min-Max con los escaladores guardados
      3. Agregar One-Hot Encoding de género y clasificación de edad
      4. Ordenar las columnas exactamente igual que el dataset de entrenamiento

    Retorna un array NumPy de forma (1, N_columnas).
    """
    # ---- Variables numéricas ----
    variables_numericas = {
        "ingreso":                  float(datos_del_formulario.get("ingreso", 4.99)),
        "veces_alquilada_pelicula": float(datos_del_formulario.get("veces_alquilada_pelicula", 15)),
        "alquileres_por_cliente":   float(datos_del_formulario.get("alquileres_por_cliente", 25)),
        "duracion_alquiler":        float(datos_del_formulario.get("duracion_alquiler", 5)),
        "pelicula_costo_reposicion":float(datos_del_formulario.get("pelicula_costo_reposicion", 19.99)),
        "pelicula_duracion":        float(datos_del_formulario.get("pelicula_duracion", 120)),
        "pelicula_precio_renta":    float(datos_del_formulario.get("pelicula_precio_renta", 2.99)),
        "variedad_generos_cliente": float(datos_del_formulario.get("variedad_generos_cliente", 8)),
        "pelicula_cantidad_actores":float(datos_del_formulario.get("pelicula_cantidad_actores", 5)),
        "pelicula_popularidad_actores": float(datos_del_formulario.get("pelicula_popularidad_actores", 120)),
    }

    genero_seleccionado        = datos_del_formulario.get("genero", "Comedia")
    clasificacion_seleccionada = datos_del_formulario.get("clasificacion", "PG")

    # ---- Construir fila del DataFrame ----
    fila_de_prediccion = dict(variables_numericas)

    # id_genero: codificación ordinal del género (igual que LabelEncoder del entrenamiento)
    fila_de_prediccion["id_genero"] = float(ID_GENERO_MAP.get(genero_seleccionado, 0))

    # One-Hot Encoding de género
    for genero in GENEROS_DE_PELICULAS:
        fila_de_prediccion[f"cat_{genero}"] = 1.0 if genero == genero_seleccionado else 0.0

    # One-Hot Encoding de clasificación de edad
    for clasificacion in CLASIFICACIONES_DE_EDAD:
        fila_de_prediccion[f"clas_{clasificacion}"] = 1.0 if clasificacion == clasificacion_seleccionada else 0.0

    df_prediccion = pd.DataFrame([fila_de_prediccion])

    # ---- Escalado con los mismos escaladores del entrenamiento ----
    df_prediccion[COLUMNAS_ESTANDARIZADAS_ZSCORE] = escalador_zscore.transform(df_prediccion[COLUMNAS_ESTANDARIZADAS_ZSCORE])
    df_prediccion[COLUMNAS_NORMALIZADAS_MINMAX]   = escalador_minmax.transform(df_prediccion[COLUMNAS_NORMALIZADAS_MINMAX])

    # ---- Ordenar columnas exactamente igual que el dataset de entrenamiento ----
    # nrows=1 para que pandas infiera los tipos correctamente y select_dtypes
    # devuelva solo las columnas numéricas (igual que la lógica del entrenamiento).
    df_muestra = pd.read_csv(RUTA_DATASET_NEURONAL, nrows=1, encoding="utf-8")
    columnas_numericas_csv = df_muestra.select_dtypes(include=[np.number]).columns.tolist()
    orden_columnas_de_entrada = [c for c in columnas_numericas_csv if c not in NOMBRES_ETIQUETAS_OBJETIVO]

    return df_prediccion[orden_columnas_de_entrada].values


# --------------------------------------------------------------------------
# Interpretación de las predicciones brutas de la red neuronal
# --------------------------------------------------------------------------

def interpretar_predicciones_de_la_red(predicciones_brutas: list) -> dict:
    """
    Convierte los tensores de salida de la red neuronal en un diccionario
    legible con etiquetas, probabilidades y el valor ganador por etiqueta.

    Predicciones brutas:
      - Binarias: float entre 0 y 1 (probabilidad de la clase positiva)
      - Multiclase: array de 3 floats que suman 1.0 (softmax)
    """
    valores_por_etiqueta = {NOMBRES_ETIQUETAS_OBJETIVO[i]: predicciones_brutas[i][0]
                            for i in range(len(NOMBRES_ETIQUETAS_OBJETIVO))}

    resultado_interpretado = {
        # --- Etiquetas binarias ---
        "es_ingreso_alto": {
            "probabilidad": float(valores_por_etiqueta["es_ingreso_alto"][0] * 100),
            "etiqueta":     "SÍ (Ingreso Alto)" if valores_por_etiqueta["es_ingreso_alto"][0] >= 0.5 else "NO (Ingreso Normal)",
        },
        "renta_fin_de_semana": {
            "probabilidad": float(valores_por_etiqueta["renta_fin_de_semana"][0] * 100),
            "etiqueta":     "SÍ (Fin de Semana)" if valores_por_etiqueta["renta_fin_de_semana"][0] >= 0.5 else "NO (Entre Semana)",
        },
        "cliente_prefiere_genero": {
            "probabilidad": float(valores_por_etiqueta["cliente_prefiere_genero"][0] * 100),
            "etiqueta":     "SÍ (Alta Afinidad)" if valores_por_etiqueta["cliente_prefiere_genero"][0] >= 0.5 else "NO (Baja Afinidad)",
        },
        "renta_larga": {
            "probabilidad": float(valores_por_etiqueta["renta_larga"][0] * 100),
            "etiqueta":     "SÍ (Larga duración)" if valores_por_etiqueta["renta_larga"][0] >= 0.5 else "NO (Normal/Corta)",
        },
        # --- Etiquetas multiclase ---
        "rango_precio_renta": {
            "probabilidades": [float(p) for p in valores_por_etiqueta["rango_precio_renta"]],
            "ganador":        int(np.argmax(valores_por_etiqueta["rango_precio_renta"])),
            "etiqueta":       ["Económico (<= $1.00)", "Estándar ($1.00 - $3.00)", "Premium (> $3.00)"][int(np.argmax(valores_por_etiqueta["rango_precio_renta"]))],
        },
        "grupo_edad_pelicula": {
            "probabilidades": [float(p) for p in valores_por_etiqueta["grupo_edad_pelicula"]],
            "ganador":        int(np.argmax(valores_por_etiqueta["grupo_edad_pelicula"])),
            "etiqueta":       ["Infantil/Familiar (G, PG)", "Adolescentes (PG-13)", "Adultos (R, NC-17)"][int(np.argmax(valores_por_etiqueta["grupo_edad_pelicula"]))],
        },
        "nivel_fidelidad_cliente": {
            "probabilidades": [float(p) for p in valores_por_etiqueta["nivel_fidelidad_cliente"]],
            "ganador":        int(np.argmax(valores_por_etiqueta["nivel_fidelidad_cliente"])),
            "etiqueta":       ["Bronce (<= 20 rentas)", "Plata (21 - 30 rentas)", "Oro (> 30 rentas)"][int(np.argmax(valores_por_etiqueta["nivel_fidelidad_cliente"]))],
        },
        "popularidad_pelicula": {
            "probabilidades": [float(p) for p in valores_por_etiqueta["popularidad_pelicula"]],
            "ganador":        int(np.argmax(valores_por_etiqueta["popularidad_pelicula"])),
            "etiqueta":       ["Baja popularidad (<= 12 rentas)", "Media popularidad (13-22 rentas)", "Alta popularidad (> 22 rentas)"][int(np.argmax(valores_por_etiqueta["popularidad_pelicula"]))],
        },
    }
    return resultado_interpretado


# --------------------------------------------------------------------------
# Rutas de Flask
# --------------------------------------------------------------------------

@app.route("/")
def pagina_inicio():
    """
    Renderiza la página principal. Carga los lookups de MongoDB la primera vez.
    """
    if not lookups_mongo_listos:
        _cargar_lookups_desde_mongo()
    return render_template(
        "index.html",
        model_ready=os.path.exists(RUTA_MODELO_PRINCIPAL),
        lookups_disponibles=lookups_mongo_listos,
        generos=GENEROS_DE_PELICULAS,
        clasificaciones=CLASIFICACIONES_DE_EDAD,
    )


@app.route("/api/peliculas")
def endpoint_peliculas():
    """Lista de películas únicas con sus atributos y métricas precalculadas."""
    peliculas_lista = sorted(peliculas_lookup.values(), key=lambda p: p.get("titulo", ""))
    return jsonify(peliculas_lista)


@app.route("/api/clientes")
def endpoint_clientes():
    """Lista de clientes únicos con nombre y métricas precalculadas."""
    clientes_lista = sorted(clientes_lookup.values(), key=lambda c: c.get("nombre_completo", ""))
    return jsonify(clientes_lista)


@app.route("/metricas")
def endpoint_metricas():
    """Devuelve las métricas de entrenamiento guardadas como JSON."""
    import json
    if not os.path.exists(RUTA_METRICAS_JSON):
        return jsonify({"error": "No hay métricas. Ejecuta primero el entrenamiento."}), 404
    with open(RUTA_METRICAS_JSON, encoding="utf-8") as f:
        return jsonify(json.load(f))


@app.route("/predict", methods=["POST"])
def endpoint_de_prediccion():
    """
    Endpoint POST que recibe un JSON con los datos de una transacción,
    aplica la red neuronal y retorna las 8 predicciones en formato JSON.

    Espera un JSON con las claves:
      ingreso, veces_alquilada_pelicula, alquileres_por_cliente,
      duracion_alquiler, pelicula_costo_reposicion, pelicula_duracion,
      pelicula_precio_renta, variedad_generos_cliente,
      pelicula_cantidad_actores, pelicula_popularidad_actores,
      genero, clasificacion
    """
    global modelo_cargado

    if not cargar_modelo_y_escaladores_si_no_estan_en_memoria():
        return jsonify({"error": "El modelo no está disponible. Ejecuta primero el entrenamiento."}), 500

    try:
        datos_recibidos = request.json
        vector_de_entrada = preparar_vector_de_entrada(datos_recibidos)
        predicciones_brutas = modelo_cargado.predict(vector_de_entrada, verbose=0)
        respuesta_interpretada = interpretar_predicciones_de_la_red(predicciones_brutas)
        return jsonify(respuesta_interpretada)

    except Exception as error_prediccion:
        return jsonify({"error": f"Error durante la predicción: {str(error_prediccion)}"}), 400


# --------------------------------------------------------------------------
# Punto de entrada principal
# --------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  Sistema Predictivo Multitarea - Red Neuronal BI3")
    print("=" * 60)
    print(f"  Modelo:     {'LISTO' if os.path.exists(RUTA_MODELO_PRINCIPAL) else 'NO ENCONTRADO — ejecuta entrenar_red_neuronal.py'}")
    print(f"  Escaladores: {'LISTOS' if os.path.exists(RUTA_ESCALADOR_ZSCORE) else 'NO ENCONTRADOS'}")
    print(f"\n  Servidor iniciado en: http://127.0.0.1:5000/")
    print("=" * 60)
    app.run(host="127.0.0.1", port=5000, debug=True)
