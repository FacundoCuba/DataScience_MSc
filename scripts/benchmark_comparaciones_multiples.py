#!/usr.bin/env python3
# -*- coding: utf-8 -*-

"""Benchmark Multi-Algoritmo sobre Estrategias de Fusión Vectorial (Early Fusion).

Incluye reporte completo de métricas supervisadas (ARI, NMI, Homogeneidad, Completitud, V-Measure)
e internas (Silhouette, Davies-Bouldin, Calinski-Harabasz).
"""

import gc
import time
import igraph as ig
import leidenalg
import numpy as np
import pandas as pd
from pymongo import MongoClient
import umap
from hdbscan import HDBSCAN
from scipy.sparse import issparse, hstack, csr_matrix
from sklearn.preprocessing import normalize
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_score,
    completeness_score,
    v_measure_score,
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from pynndescent import NNDescent


def load_unified_sample(db, sample_size: int = 70000):
    """Extrae una muestra aleatoria desde MongoDB y vectoriza los cuatro espacios de características."""
    print(f"[{time.strftime('%H:%M:%S')}] Cargando muestra de {sample_size:,} desde 'genes_unified_features'...")
    col = db["genes_unified_features"]

    pipeline = [{"$sample": {"size": sample_size}}]
    docs = list(col.aggregate(pipeline))

    y_product = np.array([d["product"] for d in docs])

    # 1. Vectorizar K-Mers 6
    first_kmer = docs[0].get("kmers6_vector")
    if isinstance(first_kmer, dict):
        dict_list = [d["kmers6_vector"] for d in docs]
        vec_tool = DictVectorizer(sparse=True)
        X_kmers6 = vec_tool.fit_transform(dict_list).astype(np.float32)
    else:
        X_kmers6 = np.array([d["kmers6_vector"] for d in docs], dtype=np.float32)

    # 2. Arreglos densos NumPy
    X_mds = np.array([d["align_mds_vector"] for d in docs], dtype=np.float32)
    X_esm3 = np.array([d["esm3_vector"] for d in docs], dtype=np.float32)
    X_prott5 = np.array([d["prott5_vector"] for d in docs], dtype=np.float32)

    del docs
    gc.collect()

    return y_product, X_kmers6, X_mds, X_esm3, X_prott5


def fuse_matrices(matrices: list):
    """Normaliza L2 individualmente los espacios vectoriales y los concatena horizontalmente."""
    norm_mats = [normalize(m, norm="l2", axis=1) for m in matrices]
    if any(issparse(m) for m in norm_mats):
        return hstack([csr_matrix(m, dtype=np.float32) if not issparse(m) else m for m in norm_mats]).tocsr()
    return np.hstack(norm_mats).astype(np.float32)


def run_umap_hdbscan(X, n_components: int = 10, min_cluster_size: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Aplica proyección UMAP basada en coseno seguida de clustering HDBSCAN."""
    print(f"[{time.strftime('%H:%M:%S')}] Reduciendo con UMAP (d={n_components}, metric=cosine)...")
    reducer = umap.UMAP(
        n_components=n_components, 
        n_neighbors=15, 
        min_dist=0.1, 
        metric="cosine", 
        random_state=42, 
        low_memory=True
    )
    X_reduced = reducer.fit_transform(X)

    print(f"[{time.strftime('%H:%M:%S')}] Clustering HDBSCAN (min_cluster_size={min_cluster_size})...")
    clusterer = HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean", cluster_selection_method="eom", core_dist_n_jobs=-1)
    labels = clusterer.fit_predict(X_reduced)
    return labels, X_reduced


def run_leiden_clustering(X, n_neighbors: int = 15) -> tuple[np.ndarray, np.ndarray]:
    """Construye un grafo k-NN con PyNNDescent y extrae comunidades con Leiden."""
    print(f"[{time.strftime('%H:%M:%S')}] Construyendo grafo k-NN con PyNNDescent (k={n_neighbors}, metric=cosine)...")
    index = NNDescent(X, metric="cosine", n_neighbors=n_neighbors + 1, n_jobs=-1, random_state=42)
    indices, _ = index.neighbor_graph

    n_samples = X.shape[0]
    sources = np.repeat(np.arange(n_samples), n_neighbors)
    targets = indices[:, 1:].reshape(-1)

    print(f"[{time.strftime('%H:%M:%S')}] Generando estructura igraph...")
    edges = np.column_stack((sources, targets))
    g = ig.Graph(n=n_samples, edges=edges, directed=False)
    g.simplify(combine_edges=None)

    del index, indices, sources, targets, edges
    gc.collect()

    print(f"[{time.strftime('%H:%M:%S')}] Extrayendo comunidades con Leiden...")
    partition = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition, seed=42)
    labels = np.array(partition.membership)

    del g, partition
    gc.collect()

    return labels, X


def evaluate_metrics(X_eval, labels: np.ndarray, y_product: np.ndarray, orig_dim: int) -> dict:
    """Calcula todas las métricas de evaluación supervisadas e internas especificadas sin sobrecargar la RAM."""
    valid_mask = labels != -1 if -1 in labels else np.ones(len(labels), dtype=bool)
    n_clusters = len(np.unique(labels[valid_mask]))
    noise_ratio = round(float(np.sum(~valid_mask) / len(labels)), 4) if -1 in labels else 0.0

    res = {
        "Dimensiones_Originales": orig_dim,
        "Clusters": n_clusters,
        "Ruido_Ratio": noise_ratio,
        "ARI": round(float(adjusted_rand_score(y_product, labels)), 4),
        "NMI": round(float(normalized_mutual_info_score(y_product, labels)), 4),
        "Homogeneidad": round(float(homogeneity_score(y_product, labels)), 4),
        "Completitud": round(float(completeness_score(y_product, labels)), 4),
        "V_Measure": round(float(v_measure_score(y_product, labels)), 4),
        "Silueta": np.nan,
        "Davies_Bouldin": np.nan,
        "Calinski_Harabasz": np.nan,
    }

    # Cálculo de métricas internas sobre submuestra optimizada
    if np.sum(valid_mask) > 1 and n_clusters > 1:
        eval_indices = np.where(valid_mask)[0]
        sub_sample_size = min(3000, len(eval_indices))
        sub_sample_idx = np.random.choice(eval_indices, size=sub_sample_size, replace=False)
        
        X_sub = X_eval[sub_sample_idx]
        labels_sub = labels[sub_sample_idx]

        # 1. Silueta (mantiene matriz dispersa si es csr_matrix)
        try:
            metric = "cosine" if issparse(X_sub) else "euclidean"
            res["Silueta"] = round(float(silhouette_score(X_sub, labels_sub, metric=metric)), 4)
        except Exception:
            pass

        # 2. Métricas que requieren espacio denso (solo si la dimensión es manejable < 10,000)
        if not issparse(X_sub) or X_sub.shape[1] < 10000:
            try:
                X_sub_dense = X_sub.toarray() if issparse(X_sub) else X_sub
                res["Davies_Bouldin"] = round(float(davies_bouldin_score(X_sub_dense, labels_sub)), 4)
                res["Calinski_Harabasz"] = round(float(calinski_harabasz_score(X_sub_dense, labels_sub)), 4)
            except Exception:
                pass

    return res


def main():
    """Función principal de ejecución del pipeline de Early Fusion Multi-Algoritmo."""
    client = MongoClient("mongodb://localhost:27017/", maxPoolSize=50)
    db = client["viromica_db"]

    y_product, X_kmers6, X_mds, X_esm3, X_prott5 = load_unified_sample(db, sample_size=70000)

    fusion_experiments = [
        # Parejas
        {"name": "ESM-3 + 6-Mers", "mats": [X_esm3, X_kmers6]},
        {"name": "ProtT5 + 6-Mers", "mats": [X_prott5, X_kmers6]},
        {"name": "ESM-3 + Landmark MDS", "mats": [X_esm3, X_mds]},
        {"name": "ProtT5 + Landmark MDS", "mats": [X_prott5, X_mds]},
        {"name": "Landmark MDS + 6-Mers", "mats": [X_mds, X_kmers6]},
        {"name": "pLM: ESM-3 + ProtT5", "mats": [X_esm3, X_prott5]},
        
        # Tríos
        {"name": "Trío: ESM-3 + 6-Mers + Landmark MDS", "mats": [X_esm3, X_kmers6, X_mds]},
        {"name": "Trío: ProtT5 + 6-Mers + Landmark MDS", "mats": [X_prott5, X_kmers6, X_mds]},
        {"name": "Trío: ESM-3 + ProtT5 + Landmark MDS", "mats": [X_esm3, X_prott5, X_mds]},
        {"name": "Trío: ESM-3 + ProtT5 + 6-Mers", "mats": [X_esm3, X_prott5, X_kmers6]},

        # Cuarteto
        {"name": "Cuarteto Completo (ESM-3 + ProtT5 + MDS + 6-Mers)", "mats": [X_esm3, X_prott5, X_mds, X_kmers6]},
    ]

    results = []

    for exp in fusion_experiments:
        print("\n" + "=" * 50)
        print(f" EVALUANDO FUSIÓN: {exp['name']}")
        print("=" * 50)

        X_fused = fuse_matrices(exp["mats"])
        orig_dim = X_fused.shape[1]

        # 1. UMAP (d=10) + HDBSCAN
        print(" Running UMAP (d=10) + HDBSCAN...")
        labels_hdb, X_eval_hdb = run_umap_hdbscan(X_fused, n_components=10, min_cluster_size=20)
        res_hdb = evaluate_metrics(X_eval_hdb, labels_hdb, y_product, orig_dim)
        res_hdb["Estrategia"] = exp["name"]
        res_hdb["Algoritmo"] = "UMAP (d=10) + HDBSCAN"
        results.append(res_hdb)

        del labels_hdb, X_eval_hdb
        gc.collect()

        # 2. Grafo k-NN + Leiden
        print(" Running Grafo k-NN + Leiden...")
        labels_lei, X_eval_lei = run_leiden_clustering(X_fused, n_neighbors=15)
        res_lei = evaluate_metrics(X_eval_lei, labels_lei, y_product, orig_dim)
        res_lei["Estrategia"] = exp["name"]
        res_lei["Algoritmo"] = "Grafo k-NN + Leiden"
        results.append(res_lei)

        del labels_lei, X_eval_lei, X_fused
        gc.collect()

    df_results = pd.DataFrame(results)

    cols = [
        "Estrategia",
        "Algoritmo",
        "Dimensiones_Originales",
        "Clusters",
        "Ruido_Ratio",
        "ARI",
        "NMI",
        "Homogeneidad",
        "Completitud",
        "V_Measure",
        "Silueta",
        "Davies_Bouldin",
        "Calinski_Harabasz",
    ]
    df_results = df_results[cols]

    print("\n\n" + "=" * 100)
    print(" RESULTADOS COMPLETO DE EARLY FUSION CON TODAS LAS MÉTRICAS")
    print("=" * 100)
    print(df_results.to_string(index=False))
    df_results.to_csv("benchmark_early_fusion_complete_metrics.csv", index=False)
    print(f"\n[{time.strftime('%H:%M:%S')}] Resultados exportados a 'benchmark_early_fusion_complete_metrics.csv'.")


if __name__ == "__main__":
    main()