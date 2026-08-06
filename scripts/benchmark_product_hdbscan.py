#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Benchmark de Representaciones Vectoriales enfocado en Anotación Funcional Proteica.

Este script evalúa y compara cuantitativamente la capacidad de distintas representaciones
vectoriales (embeddings) para capturar la función biológica de secuencias proteicas 
mediante algoritmos de agrupamiento (clustering).

Estrategias de Representación Evaluadas:
    1. 5-Mers (vec_kmers): Frecuencias de subpalabras de 5 aminoácidos (representadas como
       matrices dispersas CSR de ultra-alta dimensionalidad).
    2. Landmark MDS (vec_align_mds): Proyección métrica global de distancias de alineamiento 
       de secuencia respecto a puntos de referencia (landmarks).
    3. ESM-2 pLLM (vec_esm2): Embeddings densos derivados del modelo de lenguaje proteico de 
       gran escala ESM-2 (promedio de la última capa oculta).

Estrategias de Agrupamiento Comparadas:
    - MiniBatchKMeans (k=100): Particionamiento esférico directo sobre el espacio original.
    - Pipeline UMAP (d=10) + HDBSCAN: Reducción manifold no lineal basada en métrica coseno 
      seguida de agrupamiento por densidad con detección automática de ruido.

Evaluación y Métricas:
    - Ground Truth: Anotación funcional ('product') extraída de MongoDB (viromica_db.genes_curados).
    - Métricas Externas (Supervisadas):
        * Adjusted Rand Index (ARI_Product)
        * Normalized Mutual Information (NMI_Product)
    - Métricas Internas (No Supervisadas):
        * Coeficiente de Silhouette (optimizada para matrices dispersas mediante métrica Coseno)
        * Índice de Calinski-Harabasz
        * Índice de Davies-Bouldin
    - Estadísticas de Estructura: Clusters Detectados, Proporción de Ruido.

Flujo de Ejecución:
    1. Extrae una muestra aleatoria de secuencias con anotación válida de MongoDB.
    2. Carga y alinea las tres matrices de embeddings correspondientes.
    3. Ejecuta de forma secuencial MiniBatchKMeans y UMAP + HDBSCAN para cada representación.
    4. Saca métricas de rendimiento y genera una tabla comparativa consolidada.
    5. Exporta los resultados finales a 'benchmark_functional_annotation_results.csv'.
"""

import time
import numpy as np
import pandas as pd
from pymongo import MongoClient
import umap
from hdbscan import HDBSCAN
from scipy.sparse import issparse
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)


def load_functional_ground_truth(db, sample_size: int = 70000) -> tuple[dict, list[str], list[str]]:
    """Extrae una muestra representativa de secuencias proteicas con anotación funcional.

    Consulta la colección `genes_curados` en MongoDB mediante una tubería de
    agregación (pipeline) para filtrar secuencias que posean anotaciones válidas de 
    `product` y `protein_id`, muestreando un subconjunto aleatorio.
    """
    print(f"[{time.strftime('%H:%M:%S')}] Extrayendo muestra de {sample_size:,} genes con anotación funcional...")
    src_col = db["genes_curados"]
    
    pipeline = [
        {"$match": {"product": {"$ne": None}, "protein_id": {"$ne": None}}},
        {"$sample": {"size": sample_size}},
        {"$project": {"_id": 0, "protein_id": 1, "product": 1}}
    ]
    
    docs = list(src_col.aggregate(pipeline))
    protein_id_map = {d["protein_id"]: idx for idx, d in enumerate(docs)}
    products = [d["product"] for d in docs]
    
    return protein_id_map, products

def fetch_vectors_by_ids(col, vec_field: str, protein_id_map: dict, is_dict_vector: bool = False) -> tuple:
    """Recupera, vectoriza y alinea las representaciones vectoriales desde MongoDB.

    Filtra la colección especificada obteniendo únicamente los vectores correspondientes
    a las proteínas muestreadas en `protein_id_map`. Procesa las representaciones 
    según la estructura subyacente (diccionarios dispersos de k-mers o arreglos densos).
    """
    print(f"[{time.strftime('%H:%M:%S')}] Obteniendo vectores de '{col.name}'...")
    cursor = col.find(
        {"protein_id": {"$in": list(protein_id_map.keys())}},
        {"_id": 0, "protein_id": 1, vec_field: 1}
    )
    
    raw_vectors = {doc["protein_id"]: doc[vec_field] for doc in cursor}
    valid_ids = [p_id for p_id in protein_id_map.keys() if p_id in raw_vectors]
    indices = [protein_id_map[p_id] for p_id in valid_ids]
    
    if is_dict_vector:
        dict_list = [raw_vectors[p_id] for p_id in valid_ids]
        # USAR MATRIZ DISPERSA (sparse=True)
        vec_tool = DictVectorizer(sparse=True)
        X = vec_tool.fit_transform(dict_list)  # Devuelve scipy.sparse.csr_matrix
    else:
        sample_val = raw_vectors[valid_ids[0]]
        if isinstance(sample_val, dict):
            X = np.array([[v for _, v in sorted(raw_vectors[p_id].items(), key=lambda x: int(x[0]))] for p_id in valid_ids])
        else:
            X = np.array([raw_vectors[p_id] for p_id in valid_ids])
            
    return X, np.array(indices)

def run_minibatch_kmeans(X: np.ndarray, n_clusters: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Ejecuta el algoritmo de agrupamiento MiniBatchKMeans sobre el espacio completo.

    Aplica un particionamiento geométrico esférico optimizado mediante mini-lotes, 
    sirviendo como baseline de clustering directo sin reducción de dimensionalidad previo.
    """
    model = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=2048)
    labels = model.fit_predict(X)
    return labels, X

def run_umap_hdbscan(X: np.ndarray, n_components: int = 10, min_cluster_size: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Aplica reducción de dimensionalidad con UMAP seguida de clustering por densidad con HDBSCAN.

    Proyecta el espacio vectorial de alta dimensión a un espacio manifold latente
    preservando la estructura local mediante métrica Coseno. Posteriormente, detecta 
    agrupamientos de forma arbitraria y asigna un id `-1` a secuencias ruidosas o atípicas.
    """
    print(f"[{time.strftime('%H:%M:%S')}] Reduciendo dimensión con UMAP (d={n_components})...")
    
    # Si la dimensión original es menor o igual a n_components, omitimos UMAP
    if X.shape[1] <= n_components:
        X_reduced = X
    else:
        reducer = umap.UMAP(n_components=n_components, n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
        X_reduced = reducer.fit_transform(X)
        
    print(f"[{time.strftime('%H:%M:%S')}] Ajustando HDBSCAN (min_cluster_size={min_cluster_size})...")
    clusterer = HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean", cluster_selection_method="eom")
    labels = clusterer.fit_predict(X_reduced)
    
    return labels, X_reduced

def compute_functional_metrics(X_eval, labels: np.ndarray, y_product: np.ndarray) -> dict:
    """Calcula un conjunto exhaustivo de métricas de evaluación supervisadas e internas.

    Soporta eficientemente matrices dispersas y estructuras con ruido (etiquetas `-1`). 
    Para optimizar el cómputo del Coeficiente de Silhouette en grandes volúmenes, realiza 
    un submuestreo aleatorio de hasta 3,000 puntos válidos.
    """
    valid_mask = labels != -1 if -1 in labels else np.ones(len(labels), dtype=bool)
    n_clusters = len(np.unique(labels[valid_mask]))
    noise_ratio = round(float(np.sum(~valid_mask) / len(labels)), 4) if -1 in labels else 0.0

    metrics = {
        "Clusters_Detectados": n_clusters,
        "Proporcion_Ruido": noise_ratio,
        "ARI_Product": round(adjusted_rand_score(y_product, labels), 4),
        "NMI_Product": round(normalized_mutual_info_score(y_product, labels), 4),
    }

    if np.sum(valid_mask) > 1 and n_clusters > 1:
        eval_indices = np.where(valid_mask)[0]
        # Reducimos la muestra de evaluación a 3,000 para rapidez de Silhouette
        sample_size = min(3000, len(eval_indices))
        sub_sample_idx = np.random.choice(eval_indices, size=sample_size, replace=False)
        
        X_sub = X_eval[sub_sample_idx]
        labels_sub = labels[sub_sample_idx]
        
        try:
            if issparse(X_sub):
                # Silhouette soporta scipy.sparse de forma nativa indicando métrica coseno
                metrics["Silhouette"] = round(silhouette_score(X_sub, labels_sub, metric="cosine"), 4)
                metrics["Calinski_Harabasz"] = np.nan
                metrics["Davies_Bouldin"] = np.nan
            else:
                metrics["Silhouette"] = round(silhouette_score(X_sub, labels_sub), 4)
                X_valid = X_eval[valid_mask]
                metrics["Calinski_Harabasz"] = round(calinski_harabasz_score(X_valid, labels[valid_mask]), 2)
                metrics["Davies_Bouldin"] = round(davies_bouldin_score(X_valid, labels[valid_mask]), 4)
        except Exception:
            metrics["Silhouette"], metrics["Calinski_Harabasz"], metrics["Davies_Bouldin"] = np.nan, np.nan, np.nan
    else:
        metrics["Silhouette"], metrics["Calinski_Harabasz"], metrics["Davies_Bouldin"] = np.nan, np.nan, np.nan

    return metrics

def main():
    """Flujo principal de ejecución del Benchmark.

    Conecta a la base de datos de MongoDB (`viromica_db`), extrae la muestra de anotación 
    funcional, iterativamente carga cada una de las 3 representaciones (5-Mers, Landmark MDS, 
    ESM-2), aplica los dos métodos de clustering, calcula métricas, muestra la tabla 
    comparativa por consola y exporta el archivo `benchmark_functional_annotation_results.csv`.
    """
    client = MongoClient("mongodb://localhost:27017/", maxPoolSize=50)
    db = client["viromica_db"]

    # 1. Ground Truth enfocado únicamente en la función proteica
    sample_size = 70000
    protein_map, products = load_functional_ground_truth(db, sample_size=sample_size)
    
    embeddings_config = [
        {"name": "5-Mers (vec_kmers)", "col": db["vec_kmers"], "field": "kmer_vector", "is_dict": True},
        {"name": "Landmark MDS (vec_align_mds)", "col": db["vec_align_mds"], "field": "align_mds_vector", "is_dict": False},
        {"name": "ESM-2 pLLM (vec_esm2)", "col": db["vec_esm2"], "field": "esm2_vector", "is_dict": False},
    ]

    results = []

    for cfg in embeddings_config:
        print(f"\n==================================================")
        print(f" EVALUANDO FUNCIONALMENTE: {cfg['name']}")
        print(f"==================================================")
        
        X, valid_indices = fetch_vectors_by_ids(cfg["col"], cfg["field"], protein_map, is_dict_vector=cfg["is_dict"])
        y_prod = np.array([products[i] for i in valid_indices])

        # Experimento 1: MiniBatchKMeans
        print(f" Running MiniBatchKMeans (k=100)...")
        labels_km, X_eval_km = run_minibatch_kmeans(X, n_clusters=100)
        res_km = compute_functional_metrics(X_eval_km, labels_km, y_prod)
        res_km["Estrategia"] = cfg["name"]
        res_km["Algoritmo"] = "MiniBatchKMeans (k=100)"
        results.append(res_km)

        # Experimento 2: UMAP (d=10) + HDBSCAN
        print(f" Running UMAP (d=10) + HDBSCAN...")
        labels_hdb, X_eval_hdb = run_umap_hdbscan(X, n_components=10, min_cluster_size=20)
        res_hdb = compute_functional_metrics(X_eval_hdb, labels_hdb, y_prod)
        res_hdb["Estrategia"] = cfg["name"]
        res_hdb["Algoritmo"] = "UMAP (d=10) + HDBSCAN"
        results.append(res_hdb)

    # 2. Presentar tabla comparativa consolidada
    df_results = pd.DataFrame(results)
    cols_order = [
        "Estrategia", "Algoritmo", "Clusters_Detectados", "Proporcion_Ruido", 
        "NMI_Product", "ARI_Product", "Silhouette", "Calinski_Harabasz", "Davies_Bouldin"
    ]
    df_results = df_results[cols_order]

    print("\n\n" + "="*85)
    print(" EVALUACIÓN COMPARATIVA DE ANOTACIÓN FUNCIONAL PROTEICA")
    print("="*85)
    print(df_results.to_string(index=False))

    df_results.to_csv("benchmark_functional_annotation_results.csv", index=False)
    print(f"\n[{time.strftime('%H:%M:%S')}] Resultados guardados en 'benchmark_functional_annotation_results.csv'.")

if __name__ == "__main__":
    main()