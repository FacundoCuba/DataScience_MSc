# **En Proceso**
---
# 🧬 Viromica TFI: Determinación de Familias Génicas Virales mediante Espacios Vectoriales y Clustering

Repositorio oficial del Trabajo Final Integrador (TFI) de la **Maestría en Ciencia de Datos** de la Universidad Nacional Arturo Jauretche (UNAJ). Este proyecto implementa un pipeline escalable de ingeniería de datos, vectorización de secuencias proteicas (composicionales, tradicionales basadas en alineamiento y Modelos de Lenguaje de Proteínas - pLLMs), y evaluación comparativa de algoritmos de agrupamiento (*clustering*) para la delimitación de familias génicas en genomas virales completos y de referencia.

---

## 📂 Estructura del Repositorio

El proyecto se encuentra modularizado en scripts de Python independientes que cubren de punta a punta el flujo metodológico:

```text
├── scripts/
│   ├── descarga_genomas.py                # Ingesta y descarga de genomas desde NCBI Virus
│   ├── extraccion_genbank.py              # Transformación de archivos GenBank a JSON
│   ├── eda_viromica.py                    # Análisis exploratorio de datos (Base Bruta)
│   ├── curar_viromica.py                  # Pipeline de filtrado y curación estricta
│   ├── eda_viromica_curada.py             # Análisis exploratorio de datos (Base Curada)
│   ├── benchmark_kmers_k_selection.py     # Optimización del tamaño k para K-mers
│   ├── vec_kmers.py                       # Vectorización composicional (K-mers 5 y 6)
│   ├── vec_align_mds.py                   # Vectorización por alineamiento + Landmark MDS
│   ├── vec_mmseqs2.py                     # Vectorización por perfiles MMseqs2
│   ├── vec_cdhit.py                       # Vectorización por gradiente CD-HIT
│   ├── vec_esm2.py                        # Extracción de embeddings latentes (ESM-2)
│   ├── vec_esm3.py                        # Extracción de embeddings latentes (ESM-3)
│   ├── vec_prott5.py                      # Extracción de embeddings latentes (ProtT5)
│   ├── benchmark_kmers_k_selection.py     # Selección de hiperparámetros
│   ├── benchmark_comparaciones_simples.py # Benchmark simple (Vector vs. Algoritmo)
│   ├── benchmark_comparaciones_multiples.py# Análisis de sensibilidad y fusión vectorial
│   ├── build_unified_features.py          # Consolidación del Data Lake NoSQL unificado
│   ├── metricas_clusters.py               # Cálculo masivo de métricas intrínsecas (GPU/CuPy)
│   ├── visualizaciones_HDBSCAN.py         # Generación de visualizaciones interactivas UMAP+HDBSCAN
│   ├── visualizaciones_Leiden.py          # Generación de visualizaciones interactivas k-NN+Leiden
│   └── debug_mongodb.py                   # Utilidad de auditoría y conexión a MongoDB
├── .gitignore
└── README.md
```

---

⚙️ Arquitectura y Metodología
El pipeline metodológico se divide en cuatro etapas principales correspondientes a los objetivos específicos del TFI:
  - ETL y Data Lake NoSQL (Objetivo Específico 1):
    - Ingesta automatizada de ~20.000 genomas virales completos y de referencia desde NCBI Virus.
    - Procesamiento y almacenamiento escalable de más de ~703.000 registros proteicos en MongoDB (genes_virales).
    - Curación y filtrado estadístico basado en longitud de aminoácidos con ~690.000 registros (genes_curados).
  - Estrategias de Vectorización (Objetivo Específico 2):
      - Composicionales: Frecuencia de subcadenas (5-mers y 6-mers) en matrices dispersas de alta dimensión.
      - Alineamiento y Similitud: Landmark Multidimensional Scaling (MDS), perfiles de identidad multi-umbral con MMseqs2 y aglomeración jerárquica con CD-HIT.
      - Modelos de Lenguaje de Proteínas (pLLMs): Extracción de representaciones latentes profundas utilizando ESM-2, ESM-3 y ProtT5.
  - Benchmark y Clustering (Objetivo Específico 3):
      - Comparación rigurosa entre el enfoque de variedades no lineales y densidad (UMAP + HDBSCAN) frente a la arquitectura topológica de redes y detección de comunidades (Grafo k-NN + Leiden).
      - Evaluación mediante validación interna (Silueta, Davies-Bouldin, Calinski-Harabasz) y validación externa supervisada frente a la anotación biológica real (product: ARI, NMI, Homogeneidad, Completitud, V-Measure).
  - Validación y Anotación Funcional (Objetivo Específico 4):
      - Caracterización intrínseca por clúster (pureza, entropía de Shannon, conductancia de grafos).
      - Transferencia de anotación funcional por proximidad espacial para la resolución automatizada de proteínas hipotéticas en el proteoma oscuro viral.

---

## 🚀 Requisitos e Instalación
El proyecto está optimizado para ejecutarse en entornos Linux (Pop!_OS / Ubuntu) con soporte para aceleración por GPU (NVIDIA CUDA / CuPy / cuML).

1.Clonar el repositorio
```Bash
git clone [https://github.com/tu-usuario/viromica-tfi.git](https://github.com/tu-usuario/viromica-tfi.git)
cd viromica-tfi
```

2. Configurar el entorno virtual e instalar dependencias
```Bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

(Nota: La instancia local de MongoDB se encuentra ejecutándose en el puerto predeterminado mongodb://localhost:27017/).

---

## 🛠️ Ejecución del Pipeline
Para reproducir los análisis principales, puedes ejecutar los scripts secuencialmente desde el directorio raíz o mediante la carpeta scripts/:

Ingesta y Curación:
```Bash
python scripts/descarga_genomas.py
python scripts/extraccion_genbank.py
python scripts/curar_viromica.py
```
Vectorización y Modelos (Ejemplo con ProtT5):
```Bash
python scripts/vec_prott5.py
```
Evaluación de Benchmark y Clustering:
```Bash
python scripts/benchmark_comparaciones_simples.py
python scripts/benchmark_comparaciones_multiples.py
```
Inspección de Métricas por Clúster (Acelerado por GPU):
```Bash
python scripts/metricas_clusters.py
```

---

## 👤 Autor
Facundo Cuba
Biólogo / Bioinformático / Maestría en Ciencia de Datos
Unidad Operativa Centro Nacional de Genómica y Bioinformática (UOCNGyB) - ANLIS "Dr. Carlos G. Malbrán".
