# =============================================================================
# PASO 1 - EXTRACCIÓN DE DATOS DESDE MONGODB
# Archivo: conexion_mongodb.py
#
# Responsabilidad:
#   Conectarse a la base de datos MongoDB local (BI_Final),
#   extraer la colección de alquileres con todas las dimensiones
#   y retornar DataFrames limpios y tipados para su uso posterior.
# =============================================================================

import pandas as pd
from pymongo import MongoClient

# --------------------------------------------------------------------------
# Configuración de conexión
# --------------------------------------------------------------------------
MONGO_URI   = "mongodb://localhost:27017/"
NOMBRE_BD   = "BI_Final"
NOMBRE_COLECCION = "alquileres"


def _abrir_conexion_mongo():
    """Abre la conexión a MongoDB y retorna la colección de alquileres."""
    cliente_mongo = MongoClient(MONGO_URI)
    base_de_datos = cliente_mongo[NOMBRE_BD]
    return base_de_datos[NOMBRE_COLECCION]


def _convertir_valor_monetario_a_flotante(valor_crudo) -> float:
    """
    Convierte un valor monetario de cualquier formato a float.
    Maneja formatos como: "$4.99", "4,99", "4.99", 4.99, etc.
    Retorna 0.0 si el valor no es convertible.
    """
    if pd.isna(valor_crudo):
        return 0.0
    if isinstance(valor_crudo, (int, float)):
        return float(valor_crudo)

    valor_texto = str(valor_crudo).strip()
    valor_texto = valor_texto.replace(" ", "").replace("$", "").replace("Euro", "").replace("€", "")

    if "," in valor_texto and "." in valor_texto:
        # Formato: 1,234.56  →  el punto es decimal
        valor_texto = valor_texto.replace(",", "")
    elif "," in valor_texto and "." not in valor_texto:
        partes = valor_texto.split(",")
        if len(partes) == 2 and len(partes[1]) <= 2:
            # Formato europeo: 4,99  →  4.99
            valor_texto = valor_texto.replace(",", ".")
        else:
            # Separador de miles: 1,234  →  1234
            valor_texto = valor_texto.replace(",", "")
    elif valor_texto.count(".") > 1:
        # Múltiples puntos: 1.234.56  →  1234.56
        partes = valor_texto.rsplit(".", 1)
        valor_texto = partes[0].replace(".", "") + "." + partes[1]

    try:
        return float(valor_texto)
    except ValueError:
        return 0.0


def extraer_alquileres_desde_mongo() -> pd.DataFrame:
    """
    Extrae todos los documentos de alquileres desde MongoDB usando un pipeline
    de agregación que unifica múltiples esquemas de documento.

    Retorna un DataFrame con las siguientes columnas ya tipadas:
      - cliente_ref, pelicula_ref, pelicula_titulo, categoria_nombre (str)
      - ingreso (float), tiempo_fecha (datetime)
      - pelicula_duracion, pelicula_precio_renta (float)
      - pelicula_clasificacion (str), duracion_alquiler (int)
      - pelicula_costo_reposicion (float)
      - pelicula_actores (list[str]), pelicula_cantidad_actores (float)
    """
    coleccion = _abrir_conexion_mongo()

    pipeline_de_extraccion = [
        {
            "$project": {
                "_id": 0,
                "cliente_ref": {
                    "$ifNull": [
                        "$cliente_nombre_completo",
                        "$cliente.nombre_completo",
                        "$dimensiones.cliente.nombre_completo",
                        "$cliente.nombre",
                        "$dimensiones.cliente.nombre",
                        {"$toString": {"$ifNull": ["$id_cliente", "$cliente.id_cliente", "$dimensiones.cliente.id_cliente"]}}
                    ]
                },
                "pelicula_ref": {
                    "$ifNull": [
                        "$pelicula_titulo",
                        "$pelicula.titulo",
                        "$dimensiones.pelicula.titulo",
                        {"$toString": {"$ifNull": ["$id_pelicula", "$pelicula.id_pelicula", "$dimensiones.pelicula.id_pelicula"]}}
                    ]
                },
                "pelicula_titulo": {
                    "$ifNull": ["$pelicula_titulo", "$pelicula.titulo", "$dimensiones.pelicula.titulo"]
                },
                "categoria_nombre": {
                    "$ifNull": ["$categoria_nombre", "$categoria.nombre", "$dimensiones.categoria.nombre", "Sin categoria"]
                },
                "ingreso":              {"$ifNull": ["$ingreso", 0]},
                "tiempo_fecha":         {"$ifNull": ["$tiempo_fecha", "$tiempo.fecha", "2005-01-01"]},
                "pelicula_duracion":    {"$ifNull": ["$pelicula_duracion", "$pelicula.duracion", 0]},
                "pelicula_precio_renta":{"$ifNull": ["$pelicula_precio_renta", "$pelicula.precio_renta", 0]},
                "pelicula_clasificacion":{"$ifNull": ["$pelicula_clasificacion", "$pelicula.clasificacion", "G"]},
                "duracion_alquiler":    {"$ifNull": ["$duracion_alquiler", 0]},
                "pelicula_costo_reposicion": {"$ifNull": ["$pelicula_costo_reposicion", "$pelicula.costo_reposicion", 0.0]},
                "pelicula_actores":     {"$ifNull": ["$pelicula_actores", "$pelicula.actores", []]},
                "pelicula_cantidad_actores": {"$ifNull": ["$pelicula_cantidad_actores", "$pelicula.cantidad_actores", 0]}
            }
        }
    ]

    registros_crudos = list(coleccion.aggregate(pipeline_de_extraccion))
    df_alquileres = pd.DataFrame(registros_crudos)

    # ---- Conversión de tipos ----
    df_alquileres["cliente_ref"]      = df_alquileres["cliente_ref"].astype(str)
    df_alquileres["pelicula_ref"]     = df_alquileres["pelicula_ref"].astype(str)
    df_alquileres["pelicula_titulo"]  = df_alquileres["pelicula_titulo"].astype(str)
    df_alquileres["categoria_nombre"] = df_alquileres["categoria_nombre"].astype(str)
    df_alquileres["ingreso"]          = df_alquileres["ingreso"].apply(_convertir_valor_monetario_a_flotante)
    df_alquileres["tiempo_fecha"]     = pd.to_datetime(df_alquileres["tiempo_fecha"], errors="coerce")
    df_alquileres["pelicula_duracion"]= pd.to_numeric(df_alquileres["pelicula_duracion"], errors="coerce").fillna(0.0)
    df_alquileres["pelicula_precio_renta"] = pd.to_numeric(df_alquileres["pelicula_precio_renta"], errors="coerce").fillna(0.0)
    df_alquileres["pelicula_clasificacion"] = df_alquileres["pelicula_clasificacion"].astype(str)
    df_alquileres["duracion_alquiler"] = pd.to_numeric(df_alquileres["duracion_alquiler"], errors="coerce").fillna(0.0).astype(int)
    df_alquileres["pelicula_costo_reposicion"] = pd.to_numeric(df_alquileres["pelicula_costo_reposicion"], errors="coerce").fillna(0.0)
    df_alquileres["pelicula_cantidad_actores"] = pd.to_numeric(df_alquileres["pelicula_cantidad_actores"], errors="coerce").fillna(0.0)

    def _extraer_nombres_de_actores(campo_actores) -> list:
        """Normaliza el campo actores a una lista de nombres de cadena."""
        nombres = []
        if isinstance(campo_actores, dict):
            for valor in campo_actores.values():
                if isinstance(valor, dict):
                    nombres.append(valor.get("nombre_actor") or valor.get("nombre", ""))
                elif isinstance(valor, str):
                    nombres.append(valor)
        elif isinstance(campo_actores, list):
            for item in campo_actores:
                if isinstance(item, dict):
                    nombres.append(item.get("nombre_actor") or item.get("nombre", ""))
                elif isinstance(item, str):
                    nombres.append(item)
        return sorted(n for n in nombres if n)

    df_alquileres["pelicula_actores"] = df_alquileres["pelicula_actores"].apply(_extraer_nombres_de_actores)

    return df_alquileres


def extraer_catalogo_peliculas_desde_mongo() -> pd.DataFrame:
    """
    Extrae el catálogo único de películas (una fila por película) desde MongoDB.
    Retorna: DataFrame con columnas pelicula_ref, pelicula_titulo, categoria_nombre.
    """
    coleccion = _abrir_conexion_mongo()

    pipeline_catalogo_peliculas = [
        {
            "$group": {
                "_id": {
                    "$ifNull": [
                        "$pelicula_titulo", "$pelicula.titulo",
                        "$dimensiones.pelicula.titulo",
                        {"$toString": {"$ifNull": ["$id_pelicula", "$pelicula.id_pelicula"]}}
                    ]
                },
                "pelicula_titulo": {
                    "$first": {"$ifNull": ["$pelicula_titulo", "$pelicula.titulo"]}
                },
                "categoria_nombre": {
                    "$first": {"$ifNull": ["$categoria_nombre", "$categoria.nombre", "Sin categoria"]}
                }
            }
        },
        {
            "$project": {"_id": 0, "pelicula_ref": "$_id", "pelicula_titulo": 1, "categoria_nombre": 1}
        },
        {"$sort": {"pelicula_titulo": 1}}
    ]

    registros = list(coleccion.aggregate(pipeline_catalogo_peliculas))
    df_peliculas = pd.json_normalize(registros)
    df_peliculas["pelicula_ref"]     = df_peliculas["pelicula_ref"].astype(str)
    df_peliculas["pelicula_titulo"]  = df_peliculas["pelicula_titulo"].astype(str)
    df_peliculas["categoria_nombre"] = df_peliculas["categoria_nombre"].astype(str)
    return df_peliculas


def extraer_directorio_clientes_desde_mongo() -> pd.DataFrame:
    """
    Extrae el directorio único de clientes (una fila por cliente) desde MongoDB.
    Retorna: DataFrame con la columna cliente_ref.
    """
    coleccion = _abrir_conexion_mongo()

    pipeline_directorio_clientes = [
        {
            "$group": {
                "_id": {
                    "$ifNull": [
                        "$cliente_nombre_completo", "$cliente.nombre_completo",
                        "$dimensiones.cliente.nombre_completo",
                        {"$toString": {"$ifNull": ["$id_cliente", "$cliente.id_cliente"]}}
                    ]
                }
            }
        },
        {"$project": {"_id": 0, "cliente_ref": "$_id"}},
        {"$sort": {"cliente_ref": 1}}
    ]

    registros = list(coleccion.aggregate(pipeline_directorio_clientes))
    df_clientes = pd.json_normalize(registros)
    df_clientes["cliente_ref"] = df_clientes["cliente_ref"].astype(str)
    return df_clientes


def cargar_todos_los_dataframes() -> dict:
    """
    Función principal del módulo. Extrae las tres colecciones desde MongoDB
    y las retorna en un diccionario con claves descriptivas.

    Retorna:
        {
          "alquileres":  DataFrame de transacciones de alquiler,
          "peliculas":   DataFrame del catálogo de películas,
          "clientes":    DataFrame del directorio de clientes
        }

    Lanza ValueError si no se encontraron datos en MongoDB.
    """
    print("Conectando a MongoDB y extrayendo colecciones...")
    df_alquileres = extraer_alquileres_desde_mongo()
    df_peliculas  = extraer_catalogo_peliculas_desde_mongo()
    df_clientes   = extraer_directorio_clientes_desde_mongo()

    if df_alquileres.empty:
        raise ValueError("La colección de alquileres está vacía. Verifique la conexión a MongoDB.")

    print(f"  ✓ Alquileres extraídos: {len(df_alquileres):,} registros")
    print(f"  ✓ Películas únicas:     {len(df_peliculas):,} registros")
    print(f"  ✓ Clientes únicos:      {len(df_clientes):,} registros")

    return {
        "alquileres": df_alquileres,
        "peliculas":  df_peliculas,
        "clientes":   df_clientes
    }


# --------------------------------------------------------------------------
# Ejecución directa: python conexion_mongodb.py
# --------------------------------------------------------------------------
if __name__ == "__main__":
    datos = cargar_todos_los_dataframes()
    print("\nResumen de los DataFrames extraídos:")
    print("-" * 50)
    for nombre_df, df in datos.items():
        print(f"  {nombre_df:15} → {len(df):>6} filas, {len(df.columns):>2} columnas")
    print("\nPrimeros 3 registros de alquileres:")
    print(datos["alquileres"].head(3).to_string(index=False))
