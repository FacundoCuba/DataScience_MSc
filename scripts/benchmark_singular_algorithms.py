#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Benchmark de Agrupamiento sobre Representaciones Vectoriales Singulares.

Este script evalúa y compara el desempeño de tres paradigmas de clustering:
  1. Basado en Centroides: MiniBatchKMeans (k=100)
  2. Basado en Densidad: UMAP (d=10) + HDBSCAN
  3. Basado en Grafos: Grafo k-NN + Algoritmo de Leiden

La evaluación se realiza de forma independiente sobre cada espacio vectorial singular:
  - 5-Mers (vec_kmers): Estadísticas de composición de subpalabras (matriz dispersa CSR).
  - Landmark MDS (vec_align_mds): Proyección métrica global de alineamientos.
  - ESM-2 pLLM (vec_esm2): Embeddings latentes de lenguaje proteico.
"""

import time
import igraph as ig
import leidenalg
import numpy as np
import pandas as pd
from pymongo import MongoClient
import umap
from hdbscan import HDBSCAN
from scipy.sparse import issparse
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction import DictVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)


def load_functional_ground_truth(db, sample_size: int = 70000) -> tuple[dict, list[str]]:
    """Extrae una muestra representativa de secuencias proteicas con anotación funcional.
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
        vec_tool = DictVectorizer(sparse=True)
        X = vec_tool.fit_transform(dict_list)
    else:
        sample_val = raw_vectors[valid_ids[0]]
        if isinstance(sample_val, dict):
            X = np.array([[v for _, v in sorted(raw_vectors[p_id].items(), key=lambda x: int(x[0]))] for p_id in valid_ids], dtype=np.float32)
        else:
            X = np.array([raw_vectors[p_id] for p_id in valid_ids], dtype=np.float32)
            
    return X, np.array(indices)


def run_minibatch_kmeans(X, n_clusters: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Ejecuta el particionamiento esférico MiniBatchKMeans sobre el espacio vectorial.
    """
    model = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=2048)
    labels = model.fit_predict(X)
    return labels, X


def run_umap_hdbscan(X, n_components: int = 10, min_cluster_size: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Aplica reducción de dimensionalidad no lineal con UMAP seguida de clustering por densidad HDBSCAN.
    """
    print(f"[{time.strftime('%H:%M:%S')}] Reduciendo dimensión con UMAP (d={n_components})...")
    if X.shape[1] <= n_components:
        X_reduced = X if isinstance(X, np.ndarray) else X.toarray()
    else:
        X_norm = normalize(X, norm="l2", axis=1)
        reducer = umap.UMAP(n_components=n_components, n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
        X_reduced = reducer.fit_transform(X_norm)
        
    print(f"[{time.strftime('%H:%M:%S')}] Ajustando HDBSCAN (min_cluster_size={min_cluster_size})...")
    clusterer = HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean", cluster_selection_method="eom")
    labels = clusterer.fit_predict(X_reduced)
    
    return labels, X_reduced


def run_leiden_clustering(X, n_neighbors: int = 15) -> tuple[np.ndarray, np.ndarray]:
    """Construye un grafo de k-Vecinos Más Cercanos (k-NN) y ejecuta la partición de comunidades de Leiden.

    Calcula la topología de red utilizando distancia Coseno sobre los vectores normalizados L2,
    convierte la matriz de adyacencia a un grafo de `igraph` y optimiza la modularidad mediante el algoritmo de Leiden.
    """
    print(f"[{time.strftime('%H:%M:%S')}] Construyendo grafo k-NN (k={n_neighbors}, metric=cosine)...")
    X_norm = normalize(X, norm="l2", axis=1)
    
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine", n_jobs=-1)
    nn.fit(X_norm)
    knn_matrix = nn.kneighbors_graph(X_norm, mode="connectivity")

    print(f"[{time.strftime('%H:%M:%S')}] Construyendo grafo igraph y ejecutando algoritmo de Leiden...")
    sources, targets = knn_matrix.nonzero()
    edges = list(zip(sources, targets))
    
    g = ig.Graph(n=X.shape[0], edges=edges, directed=False)
    g.simplify(combine_edges=None)

    partition = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition, seed=42)
    labels = np.array(partition.membership)
    
    return labels, X_norm


def compute_functional_metrics(X_eval, labels: np.ndarray, y_product: np.ndarray) -> dict:
    """Calcula las métricas supervisadas (ARI, NMI) y no supervisadas (Silhouette, CH, DB).
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
        sample_size = min(3000, len(eval_indices))
        sub_sample_idx = np.random.choice(eval_indices, size=sample_size, replace=False)
        
        X_sub = X_eval[sub_sample_idx]
        labels_sub = labels[sub_sample_idx]
        
        try:
            if issparse(X_sub):
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
    """Ejecuta la evaluación comparativa completa de algoritmos sobre bases vectoriales singulares."""
    client = MongoClient("mongodb://localhost:27017/", maxPoolSize=50)
    db = client["viromica_db"]

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

        # Experimento 1: MiniBatchKMeans (k=100)
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

        # Experimento 3: Grafo k-NN (k=15) + Leiden
        print(f" Running Grafo k-NN + Leiden...")
        labels_lei, X_eval_lei = run_leiden_clustering(X, n_neighbors=15)
        res_lei = compute_functional_metrics(X_eval_lei, labels_lei, y_prod)
        res_lei["Estrategia"] = cfg["name"]
        res_lei["Algoritmo"] = "Grafo k-NN + Leiden"
        results.append(res_lei)

    df_results = pd.DataFrame(results)
    cols_order = [
        "Estrategia", "Algoritmo", "Clusters_Detectados", "Proporcion_Ruido", 
        "NMI_Product", "ARI_Product", "Silhouette", "Calinski_Harabasz", "Davies_Bouldin"
    ]
    df_results = df_results[cols_order]

    print("\n\n" + "="*85)
    print(" EVALUACIÓN COMPARATIVA MULTI-ALGORITMO EN ESPACIOS SINGULARES")
    print("="*85)
    print(df_results.to_string(index=False))

    df_results.to_csv("benchmark_singular_algorithms_results.csv", index=False)
    print(f"\n[{time.strftime('%H:%M:%S')}] Resultados guardados en 'benchmark_singular_algorithms_results.csv'.")


if __name__ == "__main__":
    main()