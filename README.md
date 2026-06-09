# Proyecto BI3 — Red Neuronal Multitarea para Alquileres de Películas
## Universidad Técnica de Ambato — Inteligencia de Negocios

---

## 📁 Estructura del Proyecto

```
U2_PROYECTO_BI/
│
├── ejecutar_todo.py               ← Pipeline maestro (corre todos los pasos)
│
├── paso_1_extraccion/
│   └── conexion_mongodb.py        ← Extrae datos desde MongoDB (colección alquileres)
│
├── paso_2_dataset/
│   ├── construir_dataset.py       ← Feature engineering + target engineering + escalado
│   ├── dataset_base.csv           ← (generado) Dataset con todas las variables y etiquetas
│   └── dataset_neuronal.csv       ← (generado) Dataset escalado y codificado para la red
│
├── paso_3_eda/
│   ├── analisis_exploratorio.py   ← Genera todas las visualizaciones estadísticas
│   └── imagenes/                  ← (generadas) Todas las imágenes del EDA
│       ├── 01_matrices_correlacion/
│       ├── 02_distribuciones/
│       ├── 03_qqplots_normalidad/
│       ├── 04_boxplots_por_etiqueta/
│       └── 05_dispersiones_bivariadas/
│
├── paso_4_entrenamiento/
│   ├── entrenar_red_neuronal.py   ← Entrena 5 arquitecturas, guarda la mejor
│   ├── modelo_principal.keras     ← (generado) El mejor modelo entrenado
│   ├── modelo_v1_base.keras       ← (generado) Modelo 1: Base
│   ├── modelo_v2_profunda.keras   ← (generado) Modelo 2: Profunda
│   ├── modelo_v3_ancha.keras      ← (generado) Modelo 3: Ancha
│   ├── modelo_v4_rapida.keras     ← (generado) Modelo 4: Rápida
│   ├── modelo_v5_conservadora.keras ← (generado) Modelo 5: Conservadora
│   ├── escalador_zscore.joblib    ← (generado) Escalador Z-score para producción
│   └── escalador_minmax.joblib    ← (generado) Escalador Min-Max para producción
│
├── paso_5_aplicacion_web/
│   ├── app.py                     ← Servidor Flask (interfaz web de predicción)
│   └── templates/
│       └── index.html             ← Interfaz visual con formulario y resultados
│
└── informe/
    ├── generar_informe.py         ← Genera el informe HTML a partir de las imágenes
    ├── informe_proyecto_bi3.html  ← (generado) Informe navegable del proyecto
    └── imagenes/                  ← Copia de las imágenes del EDA para el informe
        ├── 01_matrices_correlacion/
        ├── 02_distribuciones/
        ├── 03_qqplots_normalidad/
        ├── 04_boxplots_por_etiqueta/
        └── 05_dispersiones_bivariadas/
```

---

## 🔄 Flujo Secuencial del Proyecto

```
MongoDB (BI_Final.alquileres)
       ↓
[PASO 1] conexion_mongodb.py     → Extrae y limpia los datos crudos
       ↓
[PASO 2] construir_dataset.py    → Ingeniería de variables + etiquetas + escalado → CSV
       ↓
[PASO 3] analisis_exploratorio.py → Genera ~200 imágenes estadísticas del EDA
       ↓
[PASO 4] entrenar_red_neuronal.py → Entrena 5 redes → guarda la mejor .keras
       ↓
[PASO 5] app.py (Flask)           → Interfaz web para predecir nuevas transacciones
       ↓
[INFORME] generar_informe.py      → Informe HTML navegable del análisis completo
```

---

## 🚀 Cómo Ejecutar

### Opción A: Todo de una vez
```bash
cd U2_PROYECTO_BI
python ejecutar_todo.py
```

### Opción B: Paso a paso
```bash
# Paso 1: Extracción (requiere MongoDB corriendo en localhost:27017)
python paso_1_extraccion/conexion_mongodb.py

# Paso 2: Construcción del dataset
python paso_2_dataset/construir_dataset.py

# Paso 3: Análisis Exploratorio
python paso_3_eda/analisis_exploratorio.py

# Paso 4: Entrenamiento (requiere TensorFlow)
python paso_4_entrenamiento/entrenar_red_neuronal.py

# Paso 5: Aplicación web
cd paso_5_aplicacion_web
python app.py
# Abrir: http://127.0.0.1:5000/

# Informe
python informe/generar_informe.py
# Abrir: informe/informe_proyecto_bi3.html
```

---

## 📊 Variables del Modelo

### Variables de Entrada (X) — 31 en total
| Tipo | Escalado | Columnas |
|------|----------|----------|
| Continuas | Z-score | ingreso, veces_alquilada_pelicula, alquileres_por_cliente, duracion_alquiler, pelicula_costo_reposicion |
| Continuas | Min-Max | pelicula_duracion, pelicula_precio_renta, variedad_generos_cliente, pelicula_cantidad_actores, pelicula_popularidad_actores |
| Categóricas | One-Hot | 16 géneros (cat_*) + 5 clasificaciones de edad (clas_*) |

### Variables Objetivo (Y) — 8 etiquetas
| Tipo | Etiqueta | Descripción |
|------|----------|-------------|
| Binaria | es_ingreso_alto | ¿El ingreso supera el percentil 75 de su género? |
| Binaria | renta_fin_de_semana | ¿Ocurrió en viernes, sábado o domingo? |
| Binaria | cliente_prefiere_genero | ¿El cliente alquila este género más que su promedio? |
| Binaria | renta_larga | ¿Los días de posesión superan la mediana global? |
| Multiclase | rango_precio_renta | 0=Económico, 1=Estándar, 2=Premium |
| Multiclase | grupo_edad_pelicula | 0=Infantil, 1=Adolescentes, 2=Adultos |
| Multiclase | nivel_fidelidad_cliente | 0=Bronce, 1=Plata, 2=Oro |
| Multiclase | popularidad_pelicula | 0=Baja, 1=Media, 2=Alta |

---

## 🏗 Dependencias
```
pip install pymongo pandas numpy scikit-learn matplotlib scipy flask joblib tensorflow
```
