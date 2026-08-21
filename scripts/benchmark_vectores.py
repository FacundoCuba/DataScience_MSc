#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Benchmark Comparativo Completo de Estrategias de Vectorización.

Evalúa y compara las 8 colecciones de vectores (k-mers, alineamientos y pLLMs)
extraídas desde MongoDB sobre una muestra de 70.000 registros usando dos pipelines:
1. UMAP (d=10) + HDBSCAN
2. Grafo k-NN + Algoritmo de Leiden (CPMVertexPartition)

Genera pantalla completa de resultados y exporta un reporte tabular detallado en TXT.
"""

import os
import time
import igraph as ig
import leidenalg
import numpy as np
import pandas as pd
from pymongo import MongoClient
from scipy.sparse import issparse
from sklearn.feature_extraction import DictVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    homogeneity_completeness_v_measure,
)
import umap
from hdbscan import HDBSCAN


def extract_vector_or_dict(doc_field) -> np.ndarray | dict:
    """Extrae el campo vectorial devolviendo un arreglo 1D o el diccionario disperso original."""
    if isinstance(doc_field, list):
        return np.array(doc_field, dtype=np.float32)
    elif isinstance(doc_field, dict):
        return doc_field
    return np.array([], dtype=np.float32)


def get_field_data(doc: dict, field_candidates: list[str]):
    """Extrae la información del vector desde los campos candidatos."""
    for field in field_candidates:
        if field in doc and doc[field] is not None:
            return extract_vector_or_dict(doc[field])
    return None


def get_sample_base(db, sample_size: int = 70000) -> tuple[list[str], dict[str, str]]:
    """Extrae una muestra aleatoria de 'genes_curados' con campos válidos."""
    print(f"[{time.strftime('%H:%M:%S')}] Extrayendo muestra base de {sample_size:,} registros desde 'genes_curados'...")
    src_col = db["genes_curados"]

    pipeline = [
        {
            "$match": {
                "product": {"$exists": True, "$ne": None, "$ne": ""},
                "protein_id": {"$exists": True, "$ne": None, "$ne": ""},
            }
        },
        {"$sample": {"size": sample_size}},
        {"$project": {"_id": 0, "protein_id": 1, "product": 1}},
    ]

    docs = list(src_col.aggregate(pipeline))
    if not docs:
        raise ValueError("No se encontraron registros válidos en 'genes_curados'.")

    protein_id_to_product = {d["protein_id"]: d["product"] for d in docs}
    sample_ids = list(protein_id_to_product.keys())

    return sample_ids, protein_id_to_product


def load_vectors_for_collection(
    db, col_name: str, candidates: list[str], sample_ids: list[str], protein_id_to_product: dict[str, str]
):
    """Carga los vectores manejando matrices densas y diccionarios dispersos."""
    query_docs = db[col_name].find(
        {"protein_id": {"$in": sample_ids}},
        {field: 1 for field in ["_id", "protein_id"] + candidates},
    )

    vec_map = {}
    for d in query_docs:
        p_id = d.get("protein_id")
        data = get_field_data(d, candidates)
        if p_id and data is not None and len(data) > 0:
            vec_map[p_id] = data

    valid_ids = [p_id for p_id in sample_ids if p_id in vec_map]

    if not valid_ids:
        raise ValueError(f"No se encontraron vectores válidos para la colección '{col_name}'.")

    sample_val = vec_map[valid_ids[0]]

    if isinstance(sample_val, dict):
        dict_list = [vec_map[p_id] for p_id in valid_ids]
        vectorizer = DictVectorizer(sparse=True, dtype=np.float32)
        X = vectorizer.fit_transform(dict_list)
    else:
        X = np.array([vec_map[p_id] for p_id in valid_ids], dtype=np.float32)

    y = np.array([protein_id_to_product[p_id] for p_id in valid_ids])

    return X, y


def reduce_umap(X, umap_metric: str = "cosine") -> np.ndarray:
    """Aplica reducción UMAP a d=10 para estandarizar las métricas intrínsecas."""
    print(f"[{time.strftime('%H:%M:%S')}] Reduciendo dimensionalidad con UMAP (métrica='{umap_metric}', d=10)...")
    reducer = umap.UMAP(
        n_components=10,
        n_neighbors=15,
        min_dist=0.1,
        metric=umap_metric,
        random_state=42,
        low_memory=True,
    )
    return reducer.fit_transform(X)


def run_hdbscan(X_reduced: np.ndarray, min_cluster_size: int = 20) -> np.ndarray:
    """Clustering HDBSCAN sobre el espacio reducido UMAP."""
    print(f"[{time.strftime('%H:%M:%S')}] Ejecutando HDBSCAN (min_cluster_size={min_cluster_size})...")
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    return clusterer.fit_predict(X_reduced)


def run_knn_leiden(X_reduced: np.ndarray, k_neighbors: int = 15, resolution: float = 0.05) -> np.ndarray:
    """Construye grafo k-NN y detecta comunidades con el algoritmo de Leiden."""
    print(f"[{time.strftime('%H:%M:%S')}] Construyendo grafo k-NN (k={k_neighbors})...")
    nn = NearestNeighbors(n_neighbors=k_neighbors, metric="euclidean", n_jobs=-1)
    nn.fit(X_reduced)
    knn_graph = nn.kneighbors_graph(mode="distance")

    print(f"[{time.strftime('%H:%M:%S')}] Convirtiendo a igraph y optimizando modularidad con Leiden...")
    sources, targets = knn_graph.nonzero()
    weights = np.asarray(knn_graph[sources, targets]).ravel()

    # Invertir distancia a similitud para los pesos del grafo
    weights = 1.0 / (1.0 + weights)

    g = ig.Graph(n=X_reduced.shape[0], edges=list(zip(sources, targets)), directed=False, edge_attrs={"weight": weights})
    g.simplify(combine_edges=max)

    partition = leidenalg.find_partition(
        g,
        leidenalg.CPMVertexPartition,
        weights="weight",
        resolution_parameter=resolution,
        seed=42,
    )

    return np.array(partition.membership)


def compute_clustering_metrics(
    X_eval: np.ndarray, labels: np.ndarray, y_true: np.ndarray
) -> dict[str, float]:
    """Calcula métricas intrínsecas (sobre espacio UMAP) y extrínsecas."""
    valid_mask = labels != -1
    n_clusters = len(np.unique(labels[valid_mask]))
    noise_ratio = round(float(np.sum(~valid_mask) / len(labels)), 4)

    nmi = round(float(normalized_mutual_info_score(y_true, labels)), 4)
    ari = round(float(adjusted_rand_score(y_true, labels)), 4)
    homo, comp, v_meas = homogeneity_completeness_v_measure(y_true, labels)

    metrics = {
        "Clusters": n_clusters,
        "Ruido_Ratio": noise_ratio,
        "ARI": ari,
        "NMI": nmi,
        "Homogeneidad": round(float(homo), 4),
        "Completitud": round(float(comp), 4),
        "V_Measure": round(float(v_meas), 4),
        "Silueta": np.nan,
        "Davies_Bouldin": np.nan,
        "Calinski_Harabasz": np.nan,
    }

    if n_clusters > 1 and np.sum(valid_mask) > n_clusters:
        X_valid = X_eval[valid_mask]
        labels_valid = labels[valid_mask]

        sil = silhouette_score(
            X_valid, labels_valid, metric="euclidean", sample_size=10000, random_state=42
        )
        db_idx = davies_bouldin_score(X_valid, labels_valid)
        ch_idx = calinski_harabasz_score(X_valid, labels_valid)

        metrics["Silueta"] = round(float(sil), 4)
        metrics["Davies_Bouldin"] = round(float(db_idx), 4)
        metrics["Calinski_Harabasz"] = round(float(ch_idx), 2)

    return metrics


def main():
    client = MongoClient("mongodb://localhost:27017/", maxPoolSize=50)
    db = client["viromica_db"]

    sample_ids, protein_id_to_product = get_sample_base(db, sample_size=70000)

    collections_config = [
        ("vec_kmers5", "vec_kmers5", ["kmer_vector", "kmer5_vector", "kmers5_vector", "vector"], "cosine"),
        ("vec_kmers6", "vec_kmers6", ["kmer_vector", "kmer6_vector", "kmers6_vector", "vector"], "cosine"),
        ("vec_align_mds", "vec_align_mds", ["align_mds_vector", "vector"], "cosine"),
        ("vec_mmseqs2", "vec_mmseqs2", ["mmseq2_vector", "mmseqs2_vector", "vector"], "manhattan"),
        ("vec_cdhit", "vec_cdhit", ["cdhit_vector", "vector"], "manhattan"),
        ("vec_esm2", "vec_esm2", ["esm2_vector", "vector"], "cosine"),
        ("vec_esm3", "vec_esm3", ["esm3_vector", "vector"], "cosine"),
        ("vec_prott5", "vec_prott5", ["prott5_vector", "vector"], "cosine"),
    ]

    results = []

    for strategy_name, col_name, candidates, umap_metric in collections_config:
        print(f"\n==================================================")
        print(f" EVALUANDO ESTRATEGIA: {strategy_name}")
        print(f"==================================================")

        try:
            X, y_product = load_vectors_for_collection(
                db, col_name, candidates, sample_ids, protein_id_to_product
            )
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] ERROR al cargar {col_name}: {e}")
            continue

        print(f"[{time.strftime('%H:%M:%S')}] Registros evaluados: {X.shape[0]:,}")
        print(f"[{time.strftime('%H:%M:%S')}] Dimensiones de entrada: {X.shape[1]:,}")

        # 1. Reducción UMAP base
        X_reduced = reduce_umap(X, umap_metric=umap_metric)

        # 2. Algoritmo 1: UMAP + HDBSCAN
        labels_hdbscan = run_hdbscan(X_reduced, min_cluster_size=20)
        metrics_hdbscan = compute_clustering_metrics(X_reduced, labels_hdbscan, y_product)
        results.append({
            "Estrategia": strategy_name,
            "Algoritmo": "UMAP+HDBSCAN",
            "Dimensiones_Originales": X.shape[1],
            **metrics_hdbscan,
        })

        # 3. Algoritmo 2: k-NN + Leiden
        labels_leiden = run_knn_leiden(X_reduced, k_neighbors=15, resolution=0.005)
        metrics_leiden = compute_clustering_metrics(X_reduced, labels_leiden, y_product)
        results.append({
            "Estrategia": strategy_name,
            "Algoritmo": "kNN+Leiden",
            "Dimensiones_Originales": X.shape[1],
            **metrics_leiden,
        })

    df_res = pd.DataFrame(results)

    expected_cols = [
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

    df_res = df_res[expected_cols]

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    formatted_table = df_res.to_string(index=False)

    print("\n\n" + "=" * 120)
    print(" BENCHMARK UNIFICADO DE ESTRATEGIAS Y ALGORITMOS DE CLUSTERING")
    print("=" * 120)
    print(formatted_table)

    # Exportación del reporte a archivo TXT
    output_filename = "reporte_benchmark_vectores.txt"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write("========================================================================================================\n")
        f.write(" REPORTE DE BENCHMARK DE VECTORIZACIÓN Y CLUSTERING DE PROTEÍNAS VIRALES\n")
        f.write(f" Fecha de generación: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(" Muestra evaluada: 70.000 secuencias proteicas curadas\n")
        f.write(" Algoritmos evaluados: UMAP + HDBSCAN vs. Grafo k-NN + Leiden\n")
        f.write("========================================================================================================\n\n")
        f.write(formatted_table)
        f.write("\n\n========================================================================================================\n")

    print(f"\n[{time.strftime('%H:%M:%S')}] Reporte consolidado guardado exitosamente en '{os.path.abspath(output_filename)}'.")


if __name__ == "__main__":
    main()