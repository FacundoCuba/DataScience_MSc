#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Benchmark de Combinaciones Múltiples (Feature Fusion) de Embeddings.

Evalúa combinaciones de vectores (ej. kmers6 + prott5, esm3 + prott5)
manejando matrices densas y diccionarios dispersos.
Aplica normalización L2 previa a la concatenación horizontal.
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
from sklearn.preprocessing import normalize
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


def load_and_normalize_vector(db, col_name: str, candidates: list[str], sample_ids: list[str]):
    """Carga una colección de vectores (densa o dispersa) y aplica normalización L2."""
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
        raise ValueError(f"No se encontraron vectores para {col_name}")

    sample_val = vec_map[valid_ids[0]]

    # Si es diccionario disperso (ej. k-mers)
    if isinstance(sample_val, dict):
        dict_list = [vec_map[p_id] for p_id in valid_ids]
        vectorizer = DictVectorizer(sparse=True, dtype=np.float32)
        X_sparse = vectorizer.fit_transform(dict_list)
        X_norm = normalize(X_sparse, norm="l2", axis=1)
        # Se convierte a denso para permitir concatenación horizontal limpia
        return X_norm.toarray(), valid_ids
    else:
        X = np.array([vec_map[p_id] for p_id in valid_ids], dtype=np.float32)
        X_norm = normalize(X, norm="l2", axis=1)
        return X_norm, valid_ids


def build_fused_matrix(db, combo_configs: list, sample_ids: list[str], protein_id_to_product: dict):
    """Combina múltiples colecciones de vectores mediante concatenación horizontal."""
    # 1. Filtrar los IDs que existen en TODAS las colecciones seleccionadas de la combinación
    current_valid_ids = set(sample_ids)

    for col_name, candidates in combo_configs:
        query_ids = db[col_name].distinct("protein_id", {"protein_id": {"$in": list(current_valid_ids)}})
        current_valid_ids.intersection_update(query_ids)

    valid_ids = list(current_valid_ids)
    if not valid_ids:
        raise ValueError("La intersección de IDs entre las colecciones seleccionadas quedó vacía.")

    # 2. Cargar y concatenar horizontalmente
    matrices = []
    for col_name, candidates in combo_configs:
        X_norm, _ = load_and_normalize_vector(db, col_name, candidates, valid_ids)
        matrices.append(X_norm)

    X_fused = np.hstack(matrices)
    y_fused = np.array([protein_id_to_product[p_id] for p_id in valid_ids])

    return X_fused, y_fused, valid_ids


def run_umap_hdbscan(X, min_cluster_size=20):
    reducer = umap.UMAP(n_components=10, n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
    X_reduced = reducer.fit_transform(X)
    
    clusterer = HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean", cluster_selection_method="eom")
    labels = clusterer.fit_predict(X_reduced)
    return labels, X_reduced


def run_knn_leiden(X_reduced, k_neighbors=15, resolution=0.005):
    nn = NearestNeighbors(n_neighbors=k_neighbors, metric="euclidean", n_jobs=-1)
    nn.fit(X_reduced)
    knn_graph = nn.kneighbors_graph(mode="distance")

    sources, targets = knn_graph.nonzero()
    weights = np.asarray(knn_graph[sources, targets]).ravel()
    weights = 1.0 / (1.0 + weights)

    g = ig.Graph(n=X_reduced.shape[0], edges=list(zip(sources, targets)), directed=False, edge_attrs={"weight": weights})
    g.simplify(combine_edges=max)

    partition = leidenalg.find_partition(
        g, leidenalg.CPMVertexPartition, weights="weight", resolution_parameter=resolution, seed=42
    )
    return np.array(partition.membership)


def compute_metrics(X_eval, labels, y_true):
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
    }

    if n_clusters > 1 and np.sum(valid_mask) > n_clusters:
        X_valid = X_eval[valid_mask]
        labels_valid = labels[valid_mask]
        metrics["Silueta"] = round(float(silhouette_score(X_valid, labels_valid, metric="euclidean", sample_size=10000, random_state=42)), 4)
        metrics["Davies_Bouldin"] = round(float(davies_bouldin_score(X_valid, labels_valid)), 4)
        metrics["Calinski_Harabasz"] = round(float(calinski_harabasz_score(X_valid, labels_valid)), 2)
    else:
        metrics["Silueta"], metrics["Davies_Bouldin"], metrics["Calinski_Harabasz"] = np.nan, np.nan, np.nan

    return metrics


def main():
    client = MongoClient("mongodb://localhost:27017/", maxPoolSize=50)
    db = client["viromica_db"]

    # Extraer muestra base de genes curados
    src_col = db["genes_curados"]
    pipeline = [
        {"$match": {"product": {"$exists": True, "$ne": None, "$ne": ""}, "protein_id": {"$exists": True, "$ne": None, "$ne": ""}}},
        {"$sample": {"size": 70000}},
        {"$project": {"_id": 0, "protein_id": 1, "product": 1}},
    ]
    docs = list(src_col.aggregate(pipeline))
    protein_id_to_product = {d["protein_id"]: d["product"] for d in docs}
    sample_ids = list(protein_id_to_product.keys())

    # Mapeo de vectores disponibles
    vector_configs = {
        "kmers6": ("vec_kmers6", ["kmer_vector", "kmer6_vector", "kmers6_vector", "vector"]),
        "align_mds": ("vec_align_mds", ["align_mds_vector", "vector"]),
        "esm3": ("vec_esm3", ["esm3_vector", "vector"]),
        "prott5": ("vec_prott5", ["prott5_vector", "vector"]),
    }

    # Combinaciones a evaluar
    combinations = [
        ("kmers6 + prott5", [vector_configs["kmers6"], vector_configs["prott5"]]),
        ("esm3 + prott5", [vector_configs["esm3"], vector_configs["prott5"]]),
        ("kmers6 + esm3 + prott5", [vector_configs["kmers6"], vector_configs["esm3"], vector_configs["prott5"]]),
        ("align_mds + prott5", [vector_configs["align_mds"], vector_configs["prott5"]]),
    ]

    results = []

    for combo_name, combo_configs in combinations:
        print(f"\n==================================================")
        print(f" EVALUANDO COMBINACIÓN: {combo_name}")
        print(f"==================================================")

        try:
            X_fused, y_fused, valid_ids = build_fused_matrix(db, combo_configs, sample_ids, protein_id_to_product)
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] ERROR en la combinación {combo_name}: {e}")
            continue

        print(f"[{time.strftime('%H:%M:%S')}] Muestra efectiva: {X_fused.shape[0]:,} registros")
        print(f"[{time.strftime('%H:%M:%S')}] Dimensiones fusionadas: {X_fused.shape[1]:,}")

        # Pipeline 1: UMAP + HDBSCAN
        labels_hdb, X_red = run_umap_hdbscan(X_fused, min_cluster_size=20)
        m_hdb = compute_metrics(X_red, labels_hdb, y_fused)
        results.append({"Combinacion": combo_name, "Algoritmo": "UMAP+HDBSCAN", "Dims": X_fused.shape[1], **m_hdb})

        # Pipeline 2: kNN + Leiden
        labels_lei = run_knn_leiden(X_red, k_neighbors=15, resolution=0.005)
        m_lei = compute_metrics(X_red, labels_lei, y_fused)
        results.append({"Combinacion": combo_name, "Algoritmo": "kNN+Leiden", "Dims": X_fused.shape[1], **m_lei})

    df_res = pd.DataFrame(results)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    print("\n" + "=" * 120)
    print(" REPORTE DE COMBINACIONES MÚLTIPLES (FEATURE FUSION)")
    print("=" * 120)
    print(df_res.to_string(index=False))


if __name__ == "__main__":
    main()