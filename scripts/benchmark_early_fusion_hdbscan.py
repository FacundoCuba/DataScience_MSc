#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Benchmark Multi-Algoritmo sobre Estrategias de Fusión Vectorial (Early Fusion).

Este script evalúa y compara el desempeño de tres algoritmos de clustering:
  1. MiniBatchKMeans (k=100) - Particionamiento esférico basado en centroides.
  2. UMAP (d=10) + HDBSCAN - Reducción manifold y agrupamiento por densidad.
  3. Grafo k-NN (k=15) + Leiden - Detección de comunidades en grafos de conectividad.

Sobre cuatro espacios vectoriales combinados mediante Early Fusion:
  - ESM-2 + 5-Mers
  - ESM-2 + Landmark MDS
  - Landmark MDS + 5-Mers
  - Trio Híbrido (ESM-2 + Landmark MDS + 5-Mers)
"""

import time
import igraph as ig
import leidenalg
import numpy as np
import pandas as pd
from pymongo import MongoClient
import umap
from hdbscan import HDBSCAN
from scipy.sparse import issparse, hstack
from sklearn.preprocessing import normalize
from sklearn.feature_extraction import DictVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score


def load_unified_sample(db, sample_size: int = 70000):
    """Extrae una muestra aleatoria desde MongoDB y vectoriza los tres espacios de características."""
    print(f"[{time.strftime('%H:%M:%S')}] Cargando muestra de {sample_size:,} desde 'genes_unified_features'...")
    col = db["genes_unified_features"]
    
    pipeline = [{"$sample": {"size": sample_size}}]
    docs = list(col.aggregate(pipeline))

    y_product = np.array([d["product"] for d in docs])

    # 1. Vectorizar K-Mers (matriz dispersa CSR)
    dict_list = [d["kmer_vector"] for d in docs]
    vec_tool = DictVectorizer(sparse=True)
    X_kmers = vec_tool.fit_transform(dict_list)

    # 2. Arreglos densos NumPy
    X_mds = np.array([d["align_mds_vector"] for d in docs], dtype=np.float32)
    X_esm2 = np.array([d["esm2_vector"] for d in docs], dtype=np.float32)

    return y_product, X_kmers, X_mds, X_esm2


def fuse_matrices(matrices: list):
    """Normaliza L2 individualmente los espacios vectoriales y los concatena horizontalmente."""
    norm_mats = [normalize(m, norm="l2", axis=1) for m in matrices]
    if any(issparse(m) for m in norm_mats):
        return hstack(norm_mats).tocsr()
    return np.hstack(norm_mats)


def run_minibatch_kmeans(X, n_clusters: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Ejecuta MiniBatchKMeans sobre el espacio vectorial fusionado."""
    model = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=2048)
    labels = model.fit_predict(X)
    return labels, X


def run_umap_hdbscan(X, n_components: int = 10, min_cluster_size: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Aplica proyección UMAP basada en coseno seguida de clustering HDBSCAN."""
    print(f"[{time.strftime('%H:%M:%S')}] Reduciendo con UMAP (d={n_components}, metric=cosine)...")
    reducer = umap.UMAP(n_components=n_components, n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
    X_reduced = reducer.fit_transform(X)

    print(f"[{time.strftime('%H:%M:%S')}] Clustering HDBSCAN (min_cluster_size={min_cluster_size})...")
    clusterer = HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean", cluster_selection_method="eom")
    labels = clusterer.fit_predict(X_reduced)
    return labels, X_reduced


def run_leiden_clustering(X, n_neighbors: int = 15) -> tuple[np.ndarray, np.ndarray]:
    """Construye un grafo k-NN con métrica coseno y extrae comunidades con el algoritmo de Leiden."""
    print(f"[{time.strftime('%H:%M:%S')}] Construyendo grafo k-NN (k={n_neighbors}, metric=cosine)...")
    X_norm = normalize(X, norm="l2", axis=1)
    
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine", n_jobs=-1)
    nn.fit(X_norm)
    knn_matrix = nn.kneighbors_graph(X_norm, mode="connectivity")

    print(f"[{time.strftime('%H:%M:%S')}] Extrayendo comunidades con Leiden...")
    sources, targets = knn_matrix.nonzero()
    edges = list(zip(sources, targets))
    
    g = ig.Graph(n=X.shape[0], edges=edges, directed=False)
    g.simplify(combine_edges=None)

    partition = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition, seed=42)
    labels = np.array(partition.membership)
    
    return labels, X_norm


def evaluate_metrics(X_eval, labels: np.ndarray, y_product: np.ndarray) -> dict:
    """Calcula las métricas de evaluación supervisadas e internas."""
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
        X_sub = X_eval[sub_sample_idx]
        try:
            res["Silhouette"] = round(silhouette_score(X_sub, labels[sub_sample_idx], metric="cosine" if issparse(X_sub) else "euclidean"), 4)
        except Exception:
            pass

    return res


def main():
    """Función principal de ejecución del pipeline de Early Fusion Multi-Algoritmo."""
    client = MongoClient("mongodb://localhost:27017/", maxPoolSize=50)
    db = client["viromica_db"]

    y_product, X_kmers, X_mds, X_esm2 = load_unified_sample(db, sample_size=70000)

    fusion_experiments = [
        {"name": "ESM-2 + 5-Mers", "mats": [X_esm2, X_kmers]},
        {"name": "ESM-2 + Landmark MDS", "mats": [X_esm2, X_mds]},
        {"name": "Landmark MDS + 5-Mers", "mats": [X_mds, X_kmers]},
        {"name": "Trio Híbrido (ESM-2 + MDS + K-Mers)", "mats": [X_esm2, X_mds, X_kmers]},
    ]

    results = []

    for exp in fusion_experiments:
        print(f"\n==================================================")
        print(f" EVALUANDO FUSIÓN: {exp['name']}")
        print(f"==================================================")
        
        X_fused = fuse_matrices(exp["mats"])

        # 1. MiniBatchKMeans (k=100)
        print(f" Running MiniBatchKMeans (k=100)...")
        labels_km, X_eval_km = run_minibatch_kmeans(X_fused, n_clusters=100)
        res_km = evaluate_metrics(X_eval_km, labels_km, y_product)
        res_km["Estrategia_Fusion"] = exp["name"]
        res_km["Algoritmo"] = "MiniBatchKMeans (k=100)"
        results.append(res_km)

        # 2. UMAP (d=10) + HDBSCAN
        print(f" Running UMAP (d=10) + HDBSCAN...")
        labels_hdb, X_eval_hdb = run_umap_hdbscan(X_fused, n_components=10, min_cluster_size=20)
        res_hdb = evaluate_metrics(X_eval_hdb, labels_hdb, y_product)
        res_hdb["Estrategia_Fusion"] = exp["name"]
        res_hdb["Algoritmo"] = "UMAP (d=10) + HDBSCAN"
        results.append(res_hdb)

        # 3. Grafo k-NN + Leiden
        print(f" Running Grafo k-NN + Leiden...")
        labels_lei, X_eval_lei = run_leiden_clustering(X_fused, n_neighbors=15)
        res_lei = evaluate_metrics(X_eval_lei, labels_lei, y_product)
        res_lei["Estrategia_Fusion"] = exp["name"]
        res_lei["Algoritmo"] = "Grafo k-NN + Leiden"
        results.append(res_lei)

    df_results = pd.DataFrame(results)
    cols = ["Estrategia_Fusion", "Algoritmo", "Clusters", "Ruido", "NMI", "ARI", "Silhouette"]
    df_results = df_results[cols]

    print("\n\n" + "="*85)
    print(" RESULTADOS DE CLUSTERING MULTI-ALGORITMO EN EARLY FUSION")
    print("="*85)
    print(df_results.to_string(index=False))
    df_results.to_csv("benchmark_early_fusion_algorithms_results.csv", index=False)
    print(f"\n[{time.strftime('%H:%M:%S')}] Resultados exportados a 'benchmark_early_fusion_algorithms_results.csv'.")


if __name__ == "__main__":
    main()