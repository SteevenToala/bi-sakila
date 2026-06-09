# =============================================================================
# EJECUTOR MAESTRO DEL PROYECTO BI3
# Archivo: ejecutar_todo.py
#
# Responsabilidad:
#   Orquesta la ejecución secuencial de todos los pasos del pipeline.
#   Puedes ejecutar este archivo para correr el proyecto completo de
#   principio a fin, o ejecutar cada paso individualmente.
#
# USO:
#   python ejecutar_todo.py
# =============================================================================

import os
import sys

DIRECTORIO_PROYECTO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(DIRECTORIO_PROYECTO, "paso_1_extraccion"))
sys.path.insert(0, os.path.join(DIRECTORIO_PROYECTO, "paso_2_dataset"))
sys.path.insert(0, os.path.join(DIRECTORIO_PROYECTO, "paso_3_eda"))
sys.path.insert(0, os.path.join(DIRECTORIO_PROYECTO, "paso_4_entrenamiento"))
sys.path.insert(0, os.path.join(DIRECTORIO_PROYECTO, "informe"))


def ejecutar_paso_1_extraccion():
    """Extrae datos desde MongoDB y verifica la conexión."""
    print("\n" + "=" * 60)
    print("  PASO 1 — Extracción de datos desde MongoDB")
    print("=" * 60)
    from conexion_mongodb import cargar_todos_los_dataframes
    return cargar_todos_los_dataframes()


def ejecutar_paso_2_construccion_dataset(datos_mongo: dict):
    """Construye el dataset base y el dataset neuronal con todos los features."""
    print("\n" + "=" * 60)
    print("  PASO 2 — Construcción del Dataset")
    print("=" * 60)
    from construir_dataset import construir_dataset_completo, escalar_y_codificar_para_red_neuronal
    import pandas as pd

    df_base = construir_dataset_completo(datos_mongo["alquileres"])
    df_neuronal, escalador_z, escalador_mm = escalar_y_codificar_para_red_neuronal(df_base)

    ruta_base     = os.path.join(DIRECTORIO_PROYECTO, "paso_2_dataset", "dataset_base.csv")
    ruta_neuronal = os.path.join(DIRECTORIO_PROYECTO, "paso_2_dataset", "dataset_neuronal.csv")

    df_base.to_csv(ruta_base, index=False, encoding="utf-8-sig")
    df_neuronal.to_csv(ruta_neuronal, index=False, encoding="utf-8")

    print(f"  ✓ dataset_base.csv guardado    ({df_base.shape})")
    print(f"  ✓ dataset_neuronal.csv guardado ({df_neuronal.shape})")


def ejecutar_paso_3_analisis_exploratorio():
    """Genera todas las visualizaciones estadísticas del EDA."""
    print("\n" + "=" * 60)
    print("  PASO 3 — Análisis Exploratorio de Datos (EDA)")
    print("=" * 60)
    from analisis_exploratorio import ejecutar_analisis_exploratorio_completo
    ejecutar_analisis_exploratorio_completo()


def ejecutar_paso_4_entrenamiento():
    """Entrena las 5 arquitecturas y guarda el mejor modelo."""
    print("\n" + "=" * 60)
    print("  PASO 4 — Entrenamiento de la Red Neuronal")
    print("=" * 60)
    from entrenar_red_neuronal import ejecutar_entrenamiento_de_los_5_modelos, guardar_escaladores_de_produccion
    guardar_escaladores_de_produccion()
    ejecutar_entrenamiento_de_los_5_modelos()


def ejecutar_paso_5_informe():
    """Genera el informe HTML del proyecto."""
    print("\n" + "=" * 60)
    print("  PASO 5 — Generación del Informe HTML")
    print("=" * 60)
    from generar_informe import generar_informe_html
    generar_informe_html()


# --------------------------------------------------------------------------
# Ejecución del pipeline completo
# --------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "█" * 60)
    print("  PROYECTO BI3 — Pipeline Completo")
    print("  Red Neuronal Multitarea para Alquileres de Películas")
    print("█" * 60)

    try:
        datos_mongo = ejecutar_paso_1_extraccion()
        ejecutar_paso_2_construccion_dataset(datos_mongo)
    except Exception as error_datos:
        print(f"\n[!] Error en Pasos 1-2 (MongoDB): {error_datos}")
        print("    Si ya tienes los CSV generados, puedes continuar desde el Paso 3.")

    ejecutar_paso_3_analisis_exploratorio()
    ejecutar_paso_4_entrenamiento()
    ejecutar_paso_5_informe()

    print("\n" + "█" * 60)
    print("  ✓ Pipeline completado exitosamente.")
    print("  → Para la app web: cd paso_5_aplicacion_web && python app.py")
    print("  → Para el informe: abre informe/informe_proyecto_bi3.html")
    print("█" * 60)
