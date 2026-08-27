#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Benchmark de Fusión Tardía (Late Fusion / Consensus Clustering).

Este módulo implementa un pipeline de clustering por consenso sobre representaciones 
vectoriales proteicas heterogéneas (ESM-2, K-Mers y Landmark MDS).

El flujo de trabajo se divide en tres etapas principales:
1. Generación de particiones independientes ejecutando UMAP + HDBSCAN sobre cada
   espacio de características individual.
2. Codificación de las asignaciones de cluster en un espacio de co-asociación
   utilizando One-Hot Encoding (preservando el ruido como una categoría independiente).
3. Reducción dimensional con UMAP (métrica Coseno) y clustering secundario con HDBSCAN
   sobre el espacio de consenso para determinar la partición integrada final.

Las particiones finales se evalúan contra la anotación biológica funcional ('product')
mediante métricas de validación externa (NMI, ARI) e interna (Silhouette).
"""

import time
import numpy as np
import pandas as pd
from pymongo import MongoClient
import umap
from hdbscan import HDBSCAN
from scipy.sparse import issparse
from sklearn.preprocessing import OneHotEncoder, normalize
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score


def load_unified_sample(db, sample_size: int = 70000):
    """Extrae una muestra aleatoria desde MongoDB y construye las matrices de características base.

    Consulta la colección consolidada 'genes_unified_features' para obtener el conjunto de
    proteínas de prueba, recuperando las etiquetas funcionales ('product') y construyendo
    las representaciones matriciales para K-Mers, Landmark MDS y ESM-2.
    """
    print(f"[{time.strftime('%H:%M:%S')}] Cargando muestra de {sample_size:,} desde 'genes_unified_features'...")
    col = db["genes_unified_features"]
    
    docs = list(col.aggregate([{"$sample": {"size": sample_size}}]))
    y_product = np.array([d["product"] for d in docs])

    # 1. K-Mers
    X_kmers = DictVectorizer(sparse=True).fit_transform([d["kmer_vector"] for d in docs])
    # 2. Landmark MDS y ESM-2
    X_mds = np.array([d["align_mds_vector"] for d in docs], dtype=np.float32)
    X_esm2 = np.array([d["esm2_vector"] for d in docs], dtype=np.float32)

    return y_product, X_kmers, X_mds, X_esm2

def run_base_clustering(X, n_components: int = 10, min_cluster_size: int = 20) -> np.ndarray:
    """Ejecuta el pipeline base de clustering (UMAP + HDBSCAN) sobre un único espacio vectorial.

    Normaliza la matriz de entrada en la norma L2, reduce su dimensionalidad a través de UMAP
    utilizando distancia Coseno si la matriz es dispersa o de alta dimensión, y aplica HDBSCAN
    para generar la partición inicial de clusters.
    """
    if issparse(X) or X.shape[1] > n_components:
        X_norm = normalize(X, norm="l2", axis=1)
        reducer = umap.UMAP(n_components=n_components, n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
        X = reducer.fit_transform(X_norm)

    clusterer = HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean", cluster_selection_method="eom")
    return clusterer.fit_predict(X)

def run_consensus_clustering(labels_list: list[np.ndarray], min_cluster_size: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Integra múltiples particiones independientes mediante un enfoque de espacio de co-asociación disperso.

    Apila las etiquetas asignadas por cada modelo individual, construye una representación binaria dispersa
    vía One-Hot Encoding (donde cada cluster actúa como una característica y el ruido -1 como categoría propia),
    y ejecuta un ciclo secundario de UMAP + HDBSCAN sobre dicha matriz de coincidencia.
    """
    print(f"[{time.strftime('%H:%M:%S')}] Construyendo matriz dispersa de co-asociación (One-Hot)...")
    
    # Matriz de etiquetas (N, M) donde M es el número de modelos fusionados
    label_matrix = np.column_stack(labels_list)
    
    # One-Hot Encoding de las etiquetas asignadas (trata el ruido -1 como una categoría propia)
    encoder = OneHotEncoder(sparse_output=True, handle_unknown="ignore")
    X_consensus_sparse = encoder.fit_transform(label_matrix)

    print(f"[{time.strftime('%H:%M:%S')}] Proyectando espacio de consenso con UMAP (metric=cosine)...")
    reducer = umap.UMAP(n_components=10, n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
    X_consensus_reduced = reducer.fit_transform(X_consensus_sparse)

    print(f"[{time.strftime('%H:%M:%S')}] Generando clusters finales de consenso con HDBSCAN...")
    clusterer = HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean", cluster_selection_method="eom")
    final_labels = clusterer.fit_predict(X_consensus_reduced)

    return final_labels, X_consensus_reduced

def evaluate_metrics(X_eval, labels, y_product) -> dict:
    """Calcula métricas de evaluación externa (ARI, NMI) e interna (Silhouette) para las etiquetas de consenso.

    Filtra los elementos catalogados como ruido (-1) para determinar la cantidad de clusters válidos
    y la tasa de ruido relativa. Calcula el Coeficiente de Silhouette sobre una submuestra de puntos válidos.
    """
    valid_mask = labels != -1 if -1 in labels else np.ones(len(labels), dtype=bool)
    n_clusters = len(np.unique(labels[valid_mask]))
    noise_ratio = round(float(np.sum(~valid_mask) / len(labels)), 4) if -1 in labels else 0.0

    res = {
        "Clusters": n_clusters,
        "Ruido": noise_ratio,
        "ARI": round(adjusted_rand_score(y_product, labels), 4),
        "NMI": round(normalized_mutual_info_score(y_product, labels), 4),
        "Silhouette": np.nan
    }

    if np.sum(valid_mask) > 1 and n_clusters > 1:
        eval_indices = np.where(valid_mask)[0]
        sub_sample_idx = np.random.choice(eval_indices, size=min(3000, len(eval_indices)), replace=False)
        try:
            res["Silhouette"] = round(silhouette_score(X_eval[sub_sample_idx], labels[sub_sample_idx]), 4)
        except Exception:
            pass

    return res

def main():
    """Ejecuta el experimento completo de Late Fusion (Clustering por Consenso).

    Inicializa la conexión con MongoDB, obtiene la muestra de datos, calcula las particiones
    de clustering base de forma independiente para cada espacio de características, evalúa
    las distintas combinaciones de consenso, muestra los resultados tabulados en consola y
    exporta el reporte final en formato CSV.
    """
    client = MongoClient("mongodb://localhost:27017/", maxPoolSize=50)
    db = client["viromica_db"]

    y_product, X_kmers, X_mds, X_esm2 = load_unified_sample(db, sample_size=70000)

    # 1. Clustering Individual previo
    print(f"\n[{time.strftime('%H:%M:%S')}] Generando particiones base independientes...")
    print(" -> Generando partición 5-Mers...")
    l_kmers = run_base_clustering(X_kmers)
    
    print(" -> Generando partición Landmark MDS...")
    l_mds = run_base_clustering(X_mds)
    
    print(" -> Generando partición ESM-2...")
    l_esm2 = run_base_clustering(X_esm2)

    # 2. Combinaciones de Late Fusion
    late_experiments = [
        {"name": "Consenso: ESM-2 + 5-Mers", "labels": [l_esm2, l_kmers]},
        {"name": "Consenso: ESM-2 + Landmark MDS", "labels": [l_esm2, l_mds]},
        {"name": "Consenso: Landmark MDS + 5-Mers", "labels": [l_mds, l_kmers]},
        {"name": "Consenso: Trío Híbrido", "labels": [l_esm2, l_mds, l_kmers]},
    ]

    results = []

    for exp in late_experiments:
        print(f"\n==================================================")
        print(f" EVALUANDO LATE FUSION: {exp['name']}")
        print(f"==================================================")

        final_labels, X_eval = run_consensus_clustering(exp["labels"], min_cluster_size=20)
        res = evaluate_metrics(X_eval, final_labels, y_product)
        res["Estrategia_Late_Fusion"] = exp["name"]
        results.append(res)

    df_results = pd.DataFrame(results)
    cols = ["Estrategia_Late_Fusion", "Clusters", "Ruido", "NMI", "ARI", "Silhouette"]
    df_results = df_results[cols]

    print("\n\n" + "="*75)
    print(" RESULTADOS DE FUSIÓN TARDÍA (LATE FUSION / CONSENSUS)")
    print("="*75)
    print(df_results.to_string(index=False))
    df_results.to_csv("benchmark_late_fusion_results.csv", index=False)

if __name__ == "__main__":
    main()